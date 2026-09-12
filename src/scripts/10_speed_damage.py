"""Contact force versus driving speed — how fast can a robot work the row?

A rigid-body simulator can tell a planner that the manipulator touched a
plant. It cannot answer the question an operator actually has: *how fast can I
drive before I start breaking stems?* That needs a plant that pushes back,
which is what this project provides.

Method
------
A single plant stands at the origin. The vehicle drives past it along +x at a
fixed speed while the arm holds a reach pose into the row. The experiment is
repeated across speeds.

The reported quantity is **peak bending stress in the stem**, not contact
force. That choice is forced by the physics and was arrived at by measurement,
not preference:

  * Stem failure is a bending failure. The quantity that decides whether the
    manipulator breaks the plant is peak fibre stress, sigma = E * kappa * r,
    which follows from the curvature of the deflected centreline and the
    material -- both known, neither read out of the solver.

  * Peak *contact force* turned out not to converge. The arm is kinematically
    driven, so contact is a near-impulsive event whose peak depends on
    resolving a sharp transient. Measured spread across a 16x timestep
    refinement was 50-80% at every contact compliance tried, including a
    correctly formulated XPBD contact constraint. A number that moves by 80%
    when the timestep changes is not a result.

  * Peak bending stress does converge, but only if the *contact event* is
    resolved -- and a fixed timestep does not do that. A 3 m/s pass is over
    in a tenth of the time a 0.3 m/s pass takes, so a rate that converges the
    slow case under-resolves the fast one (measured: +16% still moving at
    960 Hz and 3 m/s).

    The timestep is therefore set from a fixed *spatial* step,
    dt = advance / speed, so every speed advances the arm the same distance
    per frame and every pass costs the same number of frames. Each speed is
    then re-run at half the advance and the change reported, so convergence
    is established per data point rather than assumed from one spot check.

Produces:
    results/speed_damage.png
    results/speed_damage.csv
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from _bootstrap import RESULTS
from plantsim import SimConfig
from plantsim.isaac_robot import ProceduralArm
from plantsim.solver import PlantSolver
from plantsim.validation import bending_stress
from plantsim.viz import plot_speed_damage

SPEEDS = [0.15, 0.25, 0.4, 0.6, 0.9, 1.3, 1.8, 2.4, 3.0]

# Flexural strength band for a green herbaceous stem. ASSUMED, not measured
# for any particular species -- override with --failure-mpa once a real
# three-point bending test on the plant being modelled is available. The
# figure labels it as assumed, and the crossing speed it implies inherits
# that uncertainty.
FAILURE_LOW_MPA = 15.0
FAILURE_HIGH_MPA = 45.0


def run_pass(speed, advance, cfg_template, standoff, height, x0=-0.45, x1=0.45,
             settle_seconds=0.4, device=None):
    """Drive the arm past one plant; return peak stress, deflection, frames.

    `advance` is the distance the vehicle moves per frame, which fixes the
    timestep as advance/speed. Holding it constant across speeds is what makes
    the comparison fair: every pass resolves the contact geometry equally.
    """
    cfg = cfg_template
    dt = advance / speed
    cfg.solver.dt = dt
    settle_frames = max(20, int(settle_seconds / dt))

    solver = PlantSolver(cfg, device=device)
    solver.run(settle_frames)                  # settle under gravity

    arm = ProceduralArm(base_position=(x0, 0.0, -standoff), tool_radius=0.06)
    template = solver.scene.template
    stem_slice = solver.scene.plant_slice(0)
    stem_radii = template.radius[template.stem_path]
    rest = solver.positions()[stem_slice][template.stem_path].copy()

    frames = int(np.ceil((x1 - x0) / advance))
    peak_stress = 0.0
    peak_deflection = 0.0

    for frame in range(frames):
        x = x0 + speed * frame * dt
        arm.set_base_position((x, 0.0, -standoff))
        arm.set_tool_target((x + 0.10, height, 0.0))
        centres, radii = arm.collision_spheres()
        solver.set_robot_spheres(centres, radii)
        solver.step()

        stem = solver.positions()[stem_slice][template.stem_path]
        stress = float(bending_stress(stem, stem_radii,
                                      cfg.stem.youngs_modulus).max())
        peak_stress = max(peak_stress, stress)
        peak_deflection = max(peak_deflection,
                              float(np.abs(stem[:, 0] - rest[:, 0]).max()))

    solver.sync()
    return peak_stress / 1e6, peak_deflection, frames


def base_config(plants=1):
    cfg = SimConfig()
    cfg.scene.n_plants = plants
    cfg.scene.height_jitter = 0.0
    # Still air: the measurement is of contact, not of wind.
    cfg.wind.speed = 0.0
    cfg.wind.gust_amplitude = 0.0
    cfg.wind.turbulence_amplitude = 0.0
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speeds", type=float, nargs="*", default=SPEEDS)
    ap.add_argument("--standoff", type=float, default=0.55)
    ap.add_argument("--height", type=float, default=0.55)
    ap.add_argument("--advance", type=float, default=0.001,
                    help="vehicle travel per frame (m). Sets the timestep as "
                         "advance/speed, so every speed resolves the contact "
                         "equally. Each point is re-run at half this value to "
                         "check convergence.")
    ap.add_argument("--failure-mpa", type=float, nargs=2,
                    default=[FAILURE_LOW_MPA, FAILURE_HIGH_MPA],
                    metavar=("LOW", "HIGH"),
                    help="assumed flexural strength band (MPa)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    probe = ProceduralArm(tool_radius=0.06)
    print(f"spatial step {args.advance * 1000:.2f} mm/frame "
          f"({100 * args.advance / probe.link_radius:.1f}% of a sphere radius); "
          f"timestep = advance/speed")
    print(f"arm: reach {probe.reach:.2f} m, "
          f"{len(probe.collision_spheres()[0])} spheres, "
          f"coverage_ok={probe.coverage_ok()}")
    print()

    low_mpa, high_mpa = args.failure_mpa
    header = (f"{'speed m/s':>10} {'dt ms':>8} {'stress MPa':>11} "
              f"{'refined':>9} {'change':>8} {'deflect m':>10} {'verdict':>14}")
    print(header)
    print("-" * len(header))

    rows = []
    unconverged = []
    for speed in args.speeds:
        stress, deflection, frames = run_pass(
            speed, args.advance, base_config(), args.standoff, args.height,
            device=args.device,
        )
        # Every point gets its own refinement check. One spot check at a
        # single speed is not enough: the contact event shortens as speed
        # rises, so convergence is speed-dependent.
        refined, _, _ = run_pass(
            speed, args.advance / 2, base_config(), args.standoff, args.height,
            device=args.device,
        )
        change = abs(refined - stress) / max(stress, 1e-9)
        if change > 0.10:
            unconverged.append(speed)

        if refined >= high_mpa:
            verdict = "likely breaks"
        elif refined >= low_mpa:
            verdict = "may damage"
        else:
            verdict = "safe"

        rows.append({"speed": speed, "force": refined, "coarse": stress,
                     "change": change, "deflection": deflection,
                     "frames": frames})
        print(f"{speed:10.2f} {1000 * args.advance / speed:8.2f} {stress:11.2f} "
              f"{refined:9.2f} {100 * change:+7.1f}% {deflection:10.3f} "
              f"{verdict:>14}")

    speeds = np.array([r["speed"] for r in rows])
    forces = np.array([r["force"] for r in rows])

    crossing = None
    if forces.max() >= low_mpa > forces.min():
        crossing = float(np.interp(low_mpa, forces, speeds))

    mask = forces > 1e-6
    exponent = float(np.polyfit(np.log(speeds[mask]), np.log(forces[mask]), 1)[0])

    print()
    print(f"power-law fit: peak stress proportional to v^{exponent:.2f}")
    if crossing:
        print(f"crosses the assumed {low_mpa:.0f} MPa damage threshold at "
              f"{crossing:.2f} m/s")
    else:
        print(f"never reaches the assumed {low_mpa:.0f} MPa threshold over "
              f"the speeds tested")
    print("NOTE: the strength band is assumed, not measured. The crossing "
          "speed inherits that uncertainty.")

    print()
    if unconverged:
        print(f"WARNING: {len(unconverged)} point(s) moved more than 10% under "
              f"refinement: {unconverged}")
        print("         Re-run with a smaller --advance before quoting them.")
    else:
        print("All points converged to within 10% under halved spatial step.")

    csv = ["speed_ms,peak_bending_stress_MPa,coarse_MPa,refinement_change,"
           "peak_deflection_m,frames"]
    csv += [f"{r['speed']},{r['force']:.4f},{r['coarse']:.4f},"
            f"{r['change']:.4f},{r['deflection']:.4f},{r['frames']}"
            for r in rows]
    (RESULTS / "speed_damage.csv").write_text("\n".join(csv))
    print(f"\nwrote {RESULTS / 'speed_damage.csv'}")

    figure = plot_speed_damage(
        rows, RESULTS / "speed_damage.png",
        failure_low=low_mpa, failure_high=high_mpa,
        crossing=crossing, exponent=exponent,
    )
    print(f"wrote {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
