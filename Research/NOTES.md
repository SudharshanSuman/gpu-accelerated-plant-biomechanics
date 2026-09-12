# Research notes

## Novelty argument

The defensible claim is narrow and specific. Stated carefully:

> A GPU-resident XPBD solver for tapered, height-varying-stiffness plant stems,
> coupled to a robot manipulator through measured contact reaction forces, at
> canopy scale inside a robotics simulator.

What is genuinely established elsewhere, and should be cited rather than
claimed:

- **Position-based dynamics** — Müller et al., 2007. The base algorithm.
- **XPBD** — Macklin, Müller & Chentanez, 2016. Adds compliance so stiffness
  is independent of iteration count.
- **Small Steps in Physics Simulation** — Macklin et al., 2019. The result
  that substepping converges far more cheaply than iterating. Reproduced
  independently in `VALIDATION.md` §2.
- **Cosserat rod models for plants/hair** — well established for stems.
- **GPU cloth and hair solvers** — the parallelisation patterns are standard.

What is not well covered in the robotics-simulation literature, and is where
this project sits:

1. Mainstream robotics simulators (Gazebo, CoppeliaSim, Isaac Sim's PhysX
   path) treat plants as **rigid** obstacles.
2. Plant biomechanics work that *does* model deformation is generally offline,
   agronomy-facing, and not built for closed-loop robot interaction.
3. The combination — **deformable plants + real-time + robot contact forces +
   canopy scale** — is where the gap is.

Be honest about the scale of the claim: this is a well-engineered integration
of established methods aimed at an application gap, not a new algorithm. That
is a perfectly good mini-project result, and it is a much stronger claim than
an overreaching one that a reviewer can puncture.

## Parameter sources

All in `src/plantsim/config.py`, SI units, with the reasoning inline.

| Parameter | Value | Basis |
|---|---|---|
| Young's modulus, stem | 1.0 GPa | green herbaceous stem tissue, ~0.3–2 GPa range |
| Young's modulus, leaf | 0.1 GPa | lamina is roughly an order of magnitude more compliant |
| Tissue density | 700 kg/m³ | fresh plant tissue, mostly water and fibre |
| Drag coefficient, stem | 1.1 | bluff cylinder in cross-flow |
| Drag coefficient, leaf | 1.4 | flat plate normal to flow |
| Wind shear exponent | 0.20 | power-law atmospheric boundary layer over cropland |
| Air density | 1.225 kg/m³ | ISA sea level |
| Damping rate | 1.2 /s | ζ ≈ 0.1 at ~1.5 Hz ⇒ 2ζω ≈ 1.9 /s; drag supplies part of it |

The default geometry (0.85 m, 5 mm → 2 mm taper) describes a **row crop** —
canola or a staked tomato at working height. A maize stalk is far stouter
(~9 mm base) and, correctly, barely moves in a breeze: since `EI ∝ r⁴`,
doubling the base radius makes the plant 16× harder to bend.

## Reading list, in priority order

1. **Müller et al., "Position Based Dynamics" (2007)** — read §2 (the
   algorithm) and §3 (constraints). This is the core.
2. **Macklin et al., "XPBD" (2016)** — why compliance matters and how `α̃ = α/h²`
   arises.
3. **Macklin et al., "Small Steps in Physics Simulation" (2019)** — the
   substeps-vs-iterations result, which directly sets this project's defaults.
4. **Any Cosserat rod / elastic rod formulation** — the continuum model the
   discrete chain approximates.
5. **Euler–Bernoulli cantilever solutions** — needed for the validation in
   `VALIDATION.md`, and worth being fluent in.

## Things that cost time, recorded so they are not rediscovered

**A pinned root is not a clamped root.** Fixing the root particle's position
leaves the stem free to rotate rigidly about it, and the plant falls over like
a pole regardless of stiffness. A clamp needs a second fixed reference — hence
the buried anchor particle. This presents as "my plant collapses no matter how
stiff I make it", and no amount of parameter tuning will fix it.

**Wind as acceleration never reaches equilibrium.** Applying wind as a
constant acceleration means the force never drops as the plant yields, so the
stem is driven flat. The usual reaction is to add heavy damping, which then
kills the motion entirely. The actual fix is to model drag against the
*relative* velocity, which is self-limiting.

**Per-frame velocity multipliers are a trap.** `v *= 0.85` every frame means
`0.85¹²⁰ ≈ 0` after two seconds — the motion dies and the wind appears to do
nothing. Damping must be expressed as a rate, `exp(−c·h)`, or the material
changes whenever the timestep does.

**A floppy stem is usually a solver problem, not a material problem.** At low
substep counts the measured stiffness was 7× below the specified material.
Check convergence before touching parameters: `validation.convergence_study()`
exists for exactly this.

**Reaction force must accumulate over the frame.** Sampling only the last
substep reports zero contact force while the plant is visibly being pushed,
because by then it has already been resolved out of the collision.
