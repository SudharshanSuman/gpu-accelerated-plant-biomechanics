"""Week 1, Day 1 — verify the machine can run the project.

Run this first. It reports what Warp can see and whether the GPU path is
actually available, so that a later benchmark cannot silently measure the CPU.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import warp as wp

from _bootstrap import RESULTS  # noqa: F401
from plantsim import SimConfig, device_report, resolve_device, describe_device
from plantsim.solver import PlantSolver


def main() -> int:
    print("=" * 68)
    print("Environment check — GPU-accelerated plant biomechanics")
    print("=" * 68)
    print(f"Python  {sys.version.split()[0]}")
    print(f"NumPy   {np.__version__}")
    print()
    print(device_report())
    print()

    device = resolve_device()
    print(f"Selected device: {describe_device(device)}")

    if not wp.is_cuda_available():
        print()
        print("  NOTE: no CUDA driver is visible, so the solver will run on the CPU.")
        print("  Everything still works and every figure will still be produced,")
        print("  but the numbers in 03_benchmark.py are only meaningful once the")
        print("  GPU is up. Check `nvidia-smi`; on this machine the driver was not")
        print("  responding at the time the project was written.")

    print()
    print("Running 60 frames of a single plant as a smoke test...")
    cfg = SimConfig()
    solver = PlantSolver(cfg)
    print(f"  {solver.scene.summary()}")
    solver.run(60)
    solver.sync()

    pos = solver.plant_positions(0)
    tip = pos[solver.scene.template.stem_path[-1]]
    deflection = solver.tip_deflection()[0]

    print(f"  stem tip at ({tip[0]:+.3f}, {tip[1]:+.3f}, {tip[2]:+.3f}) m")
    print(f"  horizontal tip deflection: {deflection * 100:.1f} cm")

    ok = np.isfinite(pos).all() and tip[1] > 0.5 * cfg.stem.length
    print()
    print("PASS — the solver is stable and the plant is standing." if ok
          else "FAIL — the plant collapsed or the state went non-finite.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
