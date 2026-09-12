# One-month plan, mapped onto what exists

The original plan was four weeks. This maps each week's goal to the code that
delivers it, so the report can point at a file for every claim.

| Week | Goal | Delivered by | Evidence |
|---|---|---|---|
| 1 | Environment up, single deformable stem bending on the GPU | `00_check_env.py`, `plantsim/kernels.py`, `plantsim/topology.py` | `results/stem_bending.png` |
| 2 | Leaves, multiple plants, a real wind field | `01_single_stem.py`, `02_canopy.py` | `results/canopy.png`, `canopy_stats.txt` |
| 3 | Isaac Sim integration + robot interaction | `05_isaac_sim.py`, `04_robot_interaction.py` | `results/robot_interaction.png`, `robot_forces.csv` |
| 4 | Benchmarking, validation, write-up | `03_benchmark.py`, `plantsim/validation.py` | `results/benchmark.md`, `results/validation.txt` |

## Objectives from the proposal

**Objective 1 — model plants as deformable structures responding to wind and
gravity.** Done. Tapered stems with beam-theory bending stiffness, compliant
leaves, turbulent wind with a boundary-layer shear profile, gravity, and soil
clamping. Validated against `qL⁴/(8EI)`.

**Objective 2 — GPU acceleration.** Implemented and architecturally complete:
the entire solve is Warp kernels, device-resident, with CUDA graph capture.
**Not yet measured on GPU hardware** — see the status note in `README.md`.
`03_benchmark.py` fills in the CUDA column automatically once a driver is
available.

**Objective 3 — robot interaction inside Isaac Sim.** Contact model, reaction
force reporting, and the Isaac Sim scene layer are all implemented.
`04_robot_interaction.py` demonstrates and validates the mechanics without
Isaac Sim; `05_isaac_sim.py` is the integration and needs Isaac Sim installed.

## What remains

1. **Get the NVIDIA driver working**, then re-run `00_check_env.py` and
   `03_benchmark.py`. This is the single highest-value remaining step — it
   turns Objective 2 from "implemented" into "measured".
2. **Install Isaac Sim** and run `05_isaac_sim.py --robot` to capture the demo
   video. Budget a day: the download is ~15 GB.
3. **Record the demo video** from `stem_bending.gif` plus an Isaac Sim capture.

If Isaac Sim installation runs long, do not block on it. Everything else in
the project runs without it, and the figures that carry the technical argument
are already generated.

## Scope deliberately left out

Kept out to protect the core, and worth listing in the report as future work
rather than pretending they were never considered:

- soil mechanics (root-soil interaction beyond a rigid clamp)
- rain loading and water film mass
- plant growth over time
- plastic deformation and stem breakage
- learned surrogate models of the physics
