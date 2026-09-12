# GPU-Accelerated Plant Biomechanics Simulation for Agricultural Robots

A GPU physics solver that models crop plants as **deformable structures** —
stems that bend, leaves that flutter, and a restoring stiffness that varies
along the plant — so that agricultural robots can be trained and tested
against plants that behave like plants.

Built with [NVIDIA Warp](https://github.com/NVIDIA/warp), with an integration
layer for NVIDIA Isaac Sim.

---

## Why this exists

Mainstream robotics simulators — Gazebo, CoppeliaSim, and Isaac Sim's own
PhysX rigid-body pipeline — treat a plant as a rigid obstacle. That is a
reasonable model for a wall and a poor one for a canola stem. It breaks two
things an agricultural robot depends on:

1. **Perception.** A rigid plant always looks the same. A real one is moving
   in the wind, and its silhouette from the robot's camera changes as it
   moves. A detector trained only on rigid plants learns a prior that does not
   survive contact with a field.
2. **Manipulation.** A rigid simulator can tell a planner that the gripper
   *collided* with a plant. It cannot tell the planner whether the approach
   would have **snapped the stem**, because a rigid plant has no notion of how
   hard it pushes back.

This project supplies the missing piece: plants with real bending mechanics,
fast enough to simulate a whole field in real time.

---

## What it does

| Layer | What it models |
|---|---|
| **Plant** | Tapered stem as a particle chain with axial and bending constraints, leaves as compliant blades, clamped at the soil |
| **Environment** | Turbulent wind with a height-shear profile, gusts and buffeting; gravity; ground contact with friction |
| **Robot** | Manipulator collision spheres that push the plant aside and report the reaction force back to the controller |

Everything runs as XPBD (extended position-based dynamics) in Warp kernels
that stay resident on the GPU.

---

## Quick start

```bash
cd "GPU acclerated Plant biomechanics"
pip install -r src/requirements.txt
```

```bash
python3 src/scripts/00_check_env.py
```

That reports what Warp can see and runs a smoke test. Then, in order:

```bash
python3 src/scripts/01_single_stem.py
```

```bash
python3 src/scripts/02_canopy.py --plants 60
```

```bash
python3 src/scripts/03_benchmark.py
```

```bash
python3 src/scripts/04_robot_interaction.py
```

Everything above runs in plain Python and does **not** need Isaac Sim.

### Isaac Sim

Verified against **Isaac Sim 6.0.1**. One command — the launcher finds Isaac,
fixes the environment, and runs:

```bash
./run_isaac.sh
```

```bash
./run_isaac.sh --plants 64 --seconds 10 --sweep-speed 0.5
```

With an articulated arm on a driving base (the realistic agri scenario):

```bash
./run_isaac.sh --arm --plants 32 --seconds 10
```

To drop the canopy into your own Isaac scene, three lines:

```python
bridge = PlantCanopyBridge(SimConfig(), n_plants=64)
bridge.build()
bridge.attach(world)
```

See `Documentation/ISAAC_SIM.md`. The demo records by default, baking motion
into `results/canopy_isaac.usd` as time samples — so it produces a viewable
animation even on a machine whose GPU driver is down.

Tests:

```bash
python3 -m pytest src/tests -v
```

---

## Results

All figures land in `results/`.

| File | What it shows |
|---|---|
| `stem_bending.png` | One plant bending through a gust cycle — the headline figure |
| `stem_bending.gif` | The same, animated, for the demo video |
| `stiffness_profile.png` | Rigidity vs height, and the `EI ∝ r⁴` taper behind it |
| `canopy.png` | A 60-plant crop row in one shared wind field |
| `robot_interaction.png` | An end effector sweeping through a plant, with the reaction force trace |
| `benchmark.png` / `.md` | CPU vs GPU scaling |
| `validation.txt` | Beam-theory agreement, convergence study, solver cross-check |

Headline numbers for the default plant (0.85 m row crop, 4 m/s wind):

- tip deflection **0.26 m**, about **30 % of stem length**, swaying with the gusts
- base of the stem is **23× stiffer** than the tip — it bends as an arc, not a lean
- stem stretches by **0.0024 %** under load, i.e. effectively inextensible
- peak contact force from a 0.46 m/s end-effector sweep: **≈ 1 N**, against a
  stem failure load of roughly 5–15 N

---

## Status of the GPU numbers

**The GPU path is implemented but was not measured on this machine.** At the
time of writing, `nvidia-smi` could not reach the NVIDIA driver here, so Warp
reported no CUDA device and every result above was produced on the CPU
backend.

Nothing about this is a code problem, and nothing needs to change to fix it:

- `PlantSolver` selects `cuda:0` automatically whenever a driver is present.
- The same Warp kernels compile for both targets.
- `03_benchmark.py` fills in the CUDA column on its own and says so plainly in
  `benchmark.md` when it cannot.

Once the driver is working on the RTX 4070, re-run `00_check_env.py` (it
should print `cuda:0`) and then `03_benchmark.py`. The CUDA column and the
speed-up curve populate with no edits.

---

## Repository layout

```
Documentation/
  README.md          this file
  ARCHITECTURE.md    the model, the maths, and the GPU design decisions
  VALIDATION.md      how the solver is verified, and against what
  PARAMETERS.md      authoritative parameter table, generated from code
  ISAAC_SIM.md       Isaac Sim integration and version compatibility
  WEEK_PLAN.md       the one-month plan mapped onto what exists
Research/
  NOTES.md           literature, novelty argument, parameter sources
src/
  plantsim/
    config.py        all physical and solver parameters, SI units
    topology.py      builds the particle/constraint graph; beam-theory compliance
    kernels.py       Warp GPU kernels — the whole solve
    solver.py        frame orchestration, CUDA graph capture, robot contacts
    cpu_reference.py NumPy implementation of the same algorithm
    validation.py    beam theory, convergence, and cross-solver checks
    field.py         comparison against real measured plants
    isaac_bridge.py  drop-in Isaac Sim integration
    isaac_robot.py   procedural manipulator (no asset download needed)
    viz.py           every figure
    device.py        device selection
  scripts/           00–08, run in order
  tests/             pytest suite
paper/               IEEE manuscript; numbers generated, never typed
results/             generated figures, tables and logs
run_isaac.sh         Isaac Sim launcher
```

---

## The three things worth knowing about the implementation

**1. Rigidity comes from geometry, not from a tuning knob.**
A stem's flexural rigidity is `EI`, and `I = πr⁴/4`. The default stem tapers
from 5 mm at the soil to 2 mm at the tip, so the base is `(5/2)⁴ ≈ 39×` stiffer
by geometry alone — 23× after the mass distribution is accounted for. "Stiffness
inversely proportional to distance from the ground" is a *consequence* of
modelling the plant correctly, not a rule imposed on top.

**2. Wind is a drag force, not an acceleration.**
Drag depends on the velocity of the air *relative to the moving plant*:
`F = ½ρC_dA|u−v|(u−v)`. As the plant yields, the relative speed drops and so
does the force, so the bend settles at equilibrium. A constant wind
acceleration has no such feedback and will drive a stem flat no matter how the
damping is tuned.

**3. The solver parallelises over plants, not over constraints.**
A stem is a chain, and a bending correction reaches the root only by
travelling hinge to hinge. Colour-parallel constraint solving moves
information one hinge per pass, which left a 17-node stem measurably **7×
too compliant** at low iteration counts. Running one thread per plant and
sweeping its chain sequentially solves the chain almost directly, and the
parallelism comes from the hundreds of plants in the scene — which is the
right axis for an agricultural canopy anyway. Both modes are implemented;
`solver.mode` selects between them and the tests check they agree.

See `ARCHITECTURE.md` for the full reasoning and `VALIDATION.md` for the
measurements behind these claims.
