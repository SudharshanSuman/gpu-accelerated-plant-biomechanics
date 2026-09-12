"""Week 2–3 deliverable — a whole field of plants under one wind field.

This is the scene an agricultural robot would actually be trained in: rows of
plants of varying height, each bending independently according to where it
sits in the gust pattern.

Produces:
    results/canopy.png         side elevation and plan view of the field
    results/canopy_stats.txt   deflection statistics across the population
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from _bootstrap import RESULTS
from plantsim import SimConfig, build_scene
from plantsim.solver import PlantSolver
from plantsim.viz import plot_field


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plants", type=int, default=60)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = SimConfig()
    cfg.scene.n_plants = args.plants
    cfg.scene.plants_per_row = 12

    scene = build_scene(cfg, n_plants=args.plants)
    solver = PlantSolver(cfg, scene, device=args.device)
    print(scene.summary())
    print(f"device: {solver.device}")

    history = []
    for f in range(args.frames):
        solver.step()
        if f >= args.frames // 4:
            history.append(solver.tip_deflection().copy())
    solver.sync()

    figure = plot_field(
        solver.positions(), scene, RESULTS / "canopy.png",
        f"Simulated crop row — {args.plants} plants, {scene.n_nodes} particles, "
        f"one shared turbulent wind field",
    )
    print(f"wrote {figure}")

    history = np.array(history)                 # (frames, plants)
    per_plant_mean = history.mean(axis=0)
    per_frame_mean = history.mean(axis=1)

    stats = f"""Canopy statistics — {args.plants} plants, {args.frames} frames

Scene            : {scene.summary()}
Device           : {solver.device}

Tip deflection across the population (m)
  population mean          : {per_plant_mean.mean():.4f}
  population std           : {per_plant_mean.std():.4f}
  quietest plant           : {per_plant_mean.min():.4f}
  most exposed plant       : {per_plant_mean.max():.4f}
  spread (max/min)         : {per_plant_mean.max() / max(per_plant_mean.min(), 1e-9):.2f}x

Whole-canopy sway over time (m)
  time-mean                : {per_frame_mean.mean():.4f}
  time-min                 : {per_frame_mean.min():.4f}
  time-max                 : {per_frame_mean.max():.4f}

Plants do not move in lockstep: the wind field varies with height and with
position along the row, and stems differ in size by +/-{100 * cfg.scene.height_jitter:.0f}%.
That variation is the point. A perception model trained on a canopy where
every plant bends identically learns the wrong prior about what a plant
looks like when a robot approaches it.
"""
    (RESULTS / "canopy_stats.txt").write_text(stats)
    print()
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
