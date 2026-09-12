"""Week 4 deliverable — CPU vs GPU throughput.

Three backends are timed on identical scenes and identical solver settings:

    numpy      the vectorised NumPy reference implementation
    warp-cpu   the same Warp kernels, compiled for the CPU
    warp-cuda  the same Warp kernels, on the GPU

The middle row is what makes the comparison honest. Benchmarking hand-written
CUDA against interpreted Python measures the language as much as the hardware.
Warp compiles one source to both targets, so warp-cuda vs warp-cpu isolates
the device, while numpy vs warp-cpu shows what the implementation alone buys.

Produces:
    results/benchmark.png   scaling curves and speed-up
    results/benchmark.md    the table for the report
    results/benchmark.json  raw numbers
"""

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import warp as wp

from _bootstrap import RESULTS
from plantsim import CPUPlantSolver, SimConfig, build_scene
from plantsim.solver import PlantSolver
from plantsim.viz import plot_benchmark

DEFAULT_SIZES = [1, 4, 16, 64, 256, 1024, 4096]


def _summarise(samples):
    """Mean and sample standard deviation in ms.

    A single timing run is not a measurement. Thermal state, turbo clocks and
    whatever else the OS is doing move these numbers by several percent, so
    every point is repeated and reported as mean +/- sd. `ddof=1` because
    these are samples, not the population.
    """
    a = np.asarray(samples, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def time_warp(cfg, build, device, frames, warmup, repeats):
    samples = []
    for _ in range(repeats):
        solver = PlantSolver(cfg, build(), device=device)
        solver.run(warmup)      # compile kernels, capture the CUDA graph
        solver.sync()
        start = time.perf_counter()
        solver.run(frames)
        solver.sync()           # the GPU is asynchronous: never time without this
        samples.append(1000.0 * (time.perf_counter() - start) / frames)
    return _summarise(samples)


def time_numpy(cfg, build, frames, warmup, repeats):
    samples = []
    for _ in range(repeats):
        solver = CPUPlantSolver(cfg, build())
        solver.run(warmup)
        start = time.perf_counter()
        solver.run(frames)
        samples.append(1000.0 * (time.perf_counter() - start) / frames)
    return _summarise(samples)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*", default=DEFAULT_SIZES)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=5,
                    help="independent timing runs per point; reported as mean +/- sd")
    ap.add_argument("--numpy-particle-cap", type=int, default=30000,
                    help="skip the NumPy backend above this particle count")
    ap.add_argument("--skip-numpy", action="store_true")
    args = ap.parse_args()

    cuda = wp.is_cuda_available()
    cfg = SimConfig()

    print(f"frames per measurement: {args.frames}  (after {args.warmup} warm-up)")
    print(f"repeats per point: {args.repeats}  (reported as mean +/- sd)")
    print(f"solver: mode={cfg.solver.mode}, substeps={cfg.solver.substeps}, "
          f"iterations={cfg.solver.iterations}")
    print(f"CUDA available: {cuda}")
    if not cuda:
        print("  -> warp-cuda column will be empty. Re-run once `nvidia-smi` works.")
    print()

    header = (f"{'plants':>7} {'particles':>10} {'numpy ms':>14} "
              f"{'warp-cpu ms':>16} {'warp-cuda ms':>17} {'speed-up':>9}")
    print(header)
    print("-" * len(header))

    results = []
    for n in args.sizes:
        scene = build_scene(cfg, n_plants=n)
        particles = scene.n_nodes

        build = lambda: build_scene(cfg, n_plants=n)

        numpy_ms, numpy_sd = float("nan"), float("nan")
        if not args.skip_numpy and particles <= args.numpy_particle_cap:
            numpy_ms, numpy_sd = time_numpy(cfg, build, args.frames,
                                            args.warmup, args.repeats)

        warp_cpu_ms, warp_cpu_sd = time_warp(cfg, build, "cpu", args.frames,
                                             args.warmup, args.repeats)

        warp_cuda_ms, warp_cuda_sd = float("nan"), float("nan")
        if cuda:
            warp_cuda_ms, warp_cuda_sd = time_warp(cfg, build, "cuda:0",
                                                   args.frames, args.warmup,
                                                   args.repeats)

        speedup = warp_cpu_ms / warp_cuda_ms if cuda else float("nan")

        row = {
            "plants": n,
            "particles": particles,
            "constraints": scene.n_constraints,
            "numpy_ms": numpy_ms,
            "numpy_sd": numpy_sd,
            "warp_cpu_ms": warp_cpu_ms,
            "warp_cpu_sd": warp_cpu_sd,
            "warp_cuda_ms": warp_cuda_ms,
            "warp_cuda_sd": warp_cuda_sd,
            "repeats": args.repeats,
            "speedup_vs_warp_cpu": speedup,
            # The chart compares the best CPU option available against the GPU.
            "cpu_ms": warp_cpu_ms,
            "gpu_ms": warp_cuda_ms if cuda else warp_cpu_ms,
        }
        results.append(row)

        def fmt(mean, sd):
            return "         -" if np.isnan(mean) else f"{mean:7.2f}±{sd:.2f}"

        print(f"{n:>7} {particles:>10} {fmt(numpy_ms, numpy_sd):>14} "
              f"{fmt(warp_cpu_ms, warp_cpu_sd):>16} {fmt(warp_cuda_ms, warp_cuda_sd):>17} "
              f"{'   -' if np.isnan(speedup) else f'{speedup:8.1f}x'}")

    (RESULTS / "benchmark.json").write_text(json.dumps(results, indent=2))

    lines = [
        "# Benchmark — GPU-accelerated plant biomechanics",
        "",
        f"- solver: mode=`{cfg.solver.mode}`, substeps={cfg.solver.substeps}, "
        f"iterations={cfg.solver.iterations}",
        f"- {args.frames} timed frames per point, {args.warmup} warm-up frames",
        f"- particles per plant: {results[0]['particles']}",
        f"- CUDA available during this run: {cuda}",
        "",
        "| plants | particles | constraints | numpy (ms) | warp-cpu (ms) | warp-cuda (ms) | speed-up |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        def pm(mean, sd):
            if np.isnan(mean):
                return "—"
            return f"{mean:.2f} ± {sd:.2f}"
        su = "—" if np.isnan(r["speedup_vs_warp_cpu"]) else f"{r['speedup_vs_warp_cpu']:.1f}x"
        lines.append(
            f"| {r['plants']} | {r['particles']} | {r['constraints']} | "
            f"{pm(r['numpy_ms'], r['numpy_sd'])} | "
            f"{pm(r['warp_cpu_ms'], r['warp_cpu_sd'])} | "
            f"{pm(r['warp_cuda_ms'], r['warp_cuda_sd'])} | {su} |"
        )

    if not cuda:
        lines += [
            "",
            "> The CUDA column is empty because no NVIDIA driver was reachable when",
            "> this table was generated (`nvidia-smi` could not talk to the driver).",
            "> The plot therefore shows the CPU curve only. Re-run this script on the",
            "> RTX 4070 to fill in the GPU column; nothing else needs to change.",
        ]

    (RESULTS / "benchmark.md").write_text("\n".join(lines))
    print(f"\nwrote {RESULTS / 'benchmark.md'}")

    figure = plot_benchmark(results, RESULTS / "benchmark.png")
    print(f"wrote {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
