"""Week 1–2 deliverable — a single deformable plant bending in wind.

Produces:
    results/stem_bending.png      time panels of the plant deflecting
    results/stiffness_profile.png rigidity vs height, and why it falls off
    results/stem_bending.gif      animation for the demo video
    results/validation.txt        cantilever check + CPU/GPU agreement
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from _bootstrap import RESULTS
from plantsim import SimConfig
from plantsim.solver import PlantSolver
from plantsim.validation import (
    cross_implementation,
    inextensibility,
    refinement_study,
    convergence_study,
)
from plantsim.viz import animate_plant, plot_bending_panels, plot_stiffness_profile

N_FRAMES = 300          # 5 s at 60 Hz
PANEL_TIMES = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5]


def capture(solver, cfg, n_frames, sample_frames):
    """Run the sim, keeping a host copy only at the frames we plot."""
    frames = []
    for f in range(n_frames + 1):
        if f in sample_frames:
            frames.append((f * cfg.solver.dt, solver.positions().copy()))
        solver.step()
    return frames


def main() -> int:
    cfg = SimConfig()
    solver = PlantSolver(cfg)
    tpl = solver.scene.template
    print(solver.scene.summary())

    sample_frames = {int(round(t / cfg.solver.dt)) for t in PANEL_TIMES}
    frames = capture(solver, cfg, N_FRAMES, sample_frames)
    solver.sync()

    panels = plot_bending_panels(
        frames, tpl, RESULTS / "stem_bending.png",
        "Deformable plant in turbulent wind — stiff base, compliant tip (GPU XPBD)",
        stem_length=cfg.stem.length,
    )
    print(f"wrote {panels}")

    profile = plot_stiffness_profile(
        tpl, RESULTS / "stiffness_profile.png", cfg.stem.n_nodes
    )
    print(f"wrote {profile}")

    # Re-run so the animation can sample every third frame.
    solver.reset()
    gif_frames = capture(solver, cfg, N_FRAMES, set(range(0, N_FRAMES, 3)))
    gif = animate_plant(gif_frames, tpl, RESULTS / "stem_bending.gif",
                        stem_length=cfg.stem.length, fps=20)
    print(f"wrote {gif}")

    # ---- validation ----------------------------------------------------
    print("\nrunning validation (this takes a few minutes on CPU)...")
    solver.reset()
    solver.run(N_FRAMES)
    solver.sync()
    measured, rest, stretch_err = inextensibility(solver)
    tip = solver.tip_deflection()[0]
    stem = solver.positions()[solver.scene.plant_slice(0)][tpl.stem_path]

    agreement = cross_implementation(cfg, frames=120)

    report = f"""Validation — GPU-accelerated plant biomechanics
{'=' * 62}

Scene     : {solver.scene.summary()}
Device    : {solver.device}
Solver    : mode={cfg.solver.mode}, substeps={cfg.solver.substeps}, \
iterations={cfg.solver.iterations}

1. Closed-form beam theory
{refinement_study()}

2. Solver convergence
{convergence_study()}

3. Independent implementation agreement
{agreement}

4. Inextensibility under wind load
  stem arc length           : {measured:.5f} m
  rest length               : {rest:.5f} m
  relative stretch          : {100 * stretch_err:.4f}%

5. Wind response of the default plant
  tip deflection            : {tip:.3f} m ({100 * tip / cfg.stem.length:.0f}% of stem length)
  stem tip height           : {stem[-1, 1]:.3f} m
  base bending stiffness    : {1 / tpl.bend_compliance[0]:.3e} N/m
  tip bending stiffness     : {1 / tpl.bend_compliance[cfg.stem.n_nodes - 3]:.3e} N/m
  base/tip rigidity ratio   : {tpl.bend_compliance[cfg.stem.n_nodes - 3] / tpl.bend_compliance[0]:.0f}x
"""
    (RESULTS / "validation.txt").write_text(report)
    print()
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
