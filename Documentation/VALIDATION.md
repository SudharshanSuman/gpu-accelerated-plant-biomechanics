# Validation

How the solver is checked, and against what. The raw output of the current run
is in `validation_output.txt` and `../results/validation.txt`.

"The picture looks like a plant" is not validation — a plausible-looking bend
can be reached by tuning almost any model. These are the checks that put
numbers on it.

---

## 1. Closed-form beam theory

A uniformly loaded cantilever has an exact small-deflection solution:

```
δ_tip = qL⁴ / (8EI)
```

To reproduce that case the plant is stripped to a uniform cylinder — no taper,
no leaves, no wind — and gravity is pointed along `−x` so it loads the vertical
stem transversely rather than axially. The solver then runs until it stops
moving.

**Mesh refinement**, each row independently converged to 1 % by doubling
substeps until the answer stops changing:

| nodes | segments | substeps to converge | tip (mm) | vs analytic |
|---:|---:|---:|---:|---:|
| 9 | 8 | 16 | 65.24 | +25.1 % |
| 17 | 16 | 64 | 58.49 | +12.2 % |
| 33 | 32 | 256 | 56.52 | +8.4 % |

Analytic value: **52.15 mm**.

Two things are visible here, and both matter more than the raw agreement.

**The error shrinks monotonically with resolution.** That is the spatial
discretisation converging, and it is what separates a converging scheme from
a coincidence. A single number agreeing at a single setting would prove
nothing.

**The residual ≈ 8 % offset is the discrete model's own bias, and it has a
known cause.** The soil clamp is an elastic hinge carrying the base segment's
compliance rather than a perfectly rigid encastre, and mass is lumped at nodes
rather than distributed. Both make the chain slightly more compliant than the
continuum beam the closed form describes. This is stated rather than tuned
away.

**The substeps needed grow roughly as n².** 16 → 64 → 256 for 8 → 16 → 32
segments. A beam's constraint system is stiffer the finer it is meshed, so
resolution costs solver work super-linearly. This is the practical reason the
default stem uses 16 nodes rather than 64: past a point, resolution buys
accuracy a robot cannot use.

---

## 2. Solver convergence

See the table in `ARCHITECTURE.md` §3. The finding: **substepping converges a
chain roughly 4× more cheaply than iterating**, and the defaults (32 substeps,
4 iterations) sit within 0.5 % of the converged value.

This study is the reason the defaults are what they are. It also diagnosed the
project's main failure mode: at 8 substeps × 2 iterations the stem is several
times too compliant, which presents as a plant flopping like a rope. That
looks exactly like a modelling error and is in fact a solver convergence
error — the material was right all along.

---

## 3. Independent implementation agreement

The solver exists twice:

- `kernels.py` — Warp kernels, GPU-resident
- `cpu_reference.py` — vectorised NumPy

Same constraints, same compliance, same substep schedule, written separately.
After 300 frames of turbulent wind:

```
max particle difference : ~5e-05 m
mean particle difference: ~4e-06 m
```

Agreement to a few tens of micrometres on a 0.85 m plant. Two independent
implementations converging on the same trajectory is real evidence the kernels
are right; a GPU-only result can only be checked against intuition.

The NumPy version is not a strawman kept around for benchmarking — it is the
correctness oracle.

---

## 4. Inextensibility

Real stems do not measurably stretch under wind or under the loads a robot
applies. Measured stem arc length against its rest length after 300 frames of
loading:

```
relative stretch: 0.0024 %
```

The axial constraints hold.

---

## 5. Contact force, against statics

Pushing the default stem sideways at 0.55 m height:

| | |
|---|---|
| peak stem deflection during contact | 0.218 m |
| simulated peak contact force | 0.97 N |
| static estimate `3EI·d/h³` | 1.93 N |

Same order, with the simulated force the lower of the two — the expected
direction, since at 26 % of stem length the small-deflection formula still
assumes a full-length moment arm and over-predicts the force required.

For context, a green herbaceous stem of this size fails in bending at roughly
5–15 N applied at mid-height, so this approach is comfortably safe — which is
exactly the kind of judgement the simulation exists to support.

---

## 6. Behavioural tests

`src/tests/test_plantsim.py`, 15 tests. Beyond the numerical checks above:

| Test | What it protects |
|---|---|
| `test_colouring_is_conflict_free` | no two constraints in a colour share a particle — a silent bug here would show up only as a subtly too-soft plant |
| `test_root_is_pinned_and_clamped` | the buried anchor and its hinge exist; without them the plant topples as a rigid pole |
| `test_rigidity_decreases_with_height` | compliance rises monotonically toward the tip |
| `test_base_is_stiffer_than_tip` | the tip must displace >3× more per segment than the base — i.e. it bends as an **arc**, not a straight lean |
| `test_plant_stays_standing_in_wind` | responds to wind without collapsing |
| `test_scene_tiling_preserves_topology` | no constraint spans two plants |
| `test_both_solver_modes_agree` | chain and coloured modes solve the same system |
| `test_robot_contact_pushes_plant_and_reports_force` | contact both moves the plant and reports a force |

`test_base_is_stiffer_than_tip` is the one that encodes the actual project
requirement. A plant that leans uniformly would pass a naive "did it move?"
check and fail this one.

---

## What is *not* validated

Stated plainly, because a validation section that claims too much is worse
than one that claims little:

- **No comparison against real plant measurements.** Every check here is
  against analytic beam theory or a second implementation. Material
  properties (`E ≈ 1 GPa`, `ρ ≈ 700 kg/m³`, `C_d`) are taken from published
  ranges for green herbaceous tissue, not measured from a specific crop.
  Matching a real species would need bending tests on real stems.
- **No GPU timings.** The CUDA driver was unreachable on this machine; see the
  status note in `README.md`.
- **Aerodynamics are a drag model, not a flow solve.** There is no wake, no
  plant-to-plant sheltering. Every plant sees the prescribed field. For a
  dense canopy, downwind plants would really be sheltered by upwind ones.
- **Leaves are 1-D chains, not surfaces.** They carry the right order of drag
  area but cannot twist, furl, or flutter about their own axis.
- **No plastic deformation or breakage.** Stems are purely elastic, so the
  simulation reports the force that *would* break a stem without modelling
  what happens after.
