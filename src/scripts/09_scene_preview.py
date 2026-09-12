"""Render the Isaac Sim scene offline, without needing a GPU.

Same solver, same arm, same trajectory as `05_isaac_sim.py` -- but drawn with
matplotlib instead of Isaac's renderer. That matters while the GPU driver is
down: `--gui` cannot render, and a `.usd` file is not something you can look
at without a working machine.

Produces:
    results/scene_panels.png
    results/scene_preview.gif
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from _bootstrap import RESULTS
from plantsim import SimConfig, build_scene
from plantsim.isaac_robot import ProceduralArm
from plantsim.solver import PlantSolver
from plantsim.viz import animate_scene, plot_scene_panels

PANEL_TIMES = [0.5, 2.5, 4.0, 5.5, 7.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plants", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--speed", type=float, default=0.30, help="driving speed (m/s)")
    ap.add_argument("--standoff", type=float, default=0.55)
    ap.add_argument("--height", type=float, default=0.55)
    ap.add_argument("--start", type=float, default=-0.7)
    ap.add_argument("--failure-n", type=float, default=5.0)
    args = ap.parse_args()

    cfg = SimConfig()
    cfg.scene.n_plants = args.plants
    cfg.scene.plants_per_row = args.plants      # one straight row
    cfg.scene.spacing = 0.35

    scene = build_scene(cfg, n_plants=args.plants)
    solver = PlantSolver(cfg, scene)
    arm = ProceduralArm(base_position=(args.start, 0.0, -args.standoff),
                        tool_radius=0.06)
    print(scene.summary())
    print(f"arm reach {arm.reach:.2f} m, {len(arm.collision_spheres()[0])} "
          f"collision spheres, coverage_ok={arm.coverage_ok()}")

    solver.run(60)                              # let the canopy settle

    total = int(args.seconds / cfg.solver.dt)
    settle = int(0.12 * total)
    panel_frames = {int(round(t / cfg.solver.dt)) for t in PANEL_TIMES}

    snapshots, gif_frames = [], []
    peak = 0.0
    for frame in range(total):
        elapsed = max(0.0, (frame - settle) * cfg.solver.dt)
        x = args.start + args.speed * elapsed

        arm.set_base_position((x, 0.0, -args.standoff))
        arm.set_tool_target((x + 0.10, args.height, 0.0))
        centres, radii = arm.collision_spheres()
        solver.set_robot_spheres(centres, radii)
        solver.step()

        forces = solver.reaction_forces()
        force = float(np.linalg.norm(forces.sum(axis=0))) if len(forces) else 0.0
        peak = max(peak, force)

        if frame in panel_frames or frame % 3 == 0:
            state = (frame * cfg.solver.dt, solver.positions().copy(),
                     arm.joint_positions(), (centres.copy(), radii.copy()), force)
            if frame in panel_frames:
                snapshots.append(state)
            if frame % 3 == 0:
                gif_frames.append(state)

    solver.sync()
    print(f"peak whole-arm contact force {peak:.3f} N at {args.speed:.2f} m/s")

    panels = plot_scene_panels(
        snapshots, scene, RESULTS / "scene_panels.png",
        f"Manipulator driving through the canopy at {args.speed:.2f} m/s "
        f"— {args.plants} plants, {scene.n_nodes} particles",
    )
    print(f"wrote {panels}")

    gif = animate_scene(gif_frames, scene, RESULTS / "scene_preview.gif",
                        fps=20, force_limit=args.failure_n)
    print(f"wrote {gif}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
