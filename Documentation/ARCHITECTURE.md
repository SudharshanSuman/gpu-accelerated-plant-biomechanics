# Architecture

How a plant becomes particles, how the particles are solved, and why the
solver is arranged the way it is.

---

## 1. The plant model

### Discretisation

A plant is a set of **particles** joined by **constraints**.

```
                    tip (compliant)
                     ●
                     │  ← bending hinge, compliance α = L³/(4EI)
                     ●──────●  leaf blade
                     │
                     ●──────●
                     │
                     ●         ← stem particle: lumped mass, drag area
                     │  ← axial constraint, compliance α = L/(EA)
                     ●
                     │
         ════════════●════════════  soil surface
                     │
                     ○  ← buried anchor, pinned
                   base (rigid)
```

The stem is a chain of `n_nodes` particles. Each adjacent pair carries an
**axial constraint** holding the segment length; each interior particle
carries a **bending constraint** resisting curvature. Leaves are short chains
branching off stem particles, with their own blade bending and a petiole
constraint holding the attachment angle.

### Where the numbers come from

Nothing in the model is a free tuning parameter. Everything derives from beam
theory and from the geometry in `config.py`:

| Quantity | Formula | Meaning |
|---|---|---|
| Cross-sectional area | `A = πr²` | |
| Second moment of area | `I = πr⁴/4` | resistance to bending |
| Axial compliance | `α = L/(EA)` | inverse of stiffness `EA/L` |
| Bending compliance | `α = L³/(4EI)` | derived below |
| Lumped node mass | `ρ·A·L`, split between segment ends | |
| Drag reference area | `2r·L` for stems, `width·L` for leaves | projected area facing the flow |

**Deriving the bending compliance.** The bending constraint measures the
offset of a hinge particle from the midpoint of its neighbours:

```
C = |x_b − (x_a + x_c)/2| − rest
```

For segments of length `L` meeting at angle `θ`, that offset is `L·sin(θ/2) ≈ Lθ/2`,
so `θ = 2C/L`. The elastic energy of a hinge in a beam of flexural rigidity
`EI` is `U = ½(EI/L)θ²`. Substituting:

```
U = ½ · (4EI/L³) · C²
```

and since XPBD writes energy as `U = ½(1/α)C²`, the compliance is
**`α = L³/(4EI)`**.

**Why this constraint and not `|x_c − x_a|`.** The more commonly seen bending
constraint measures the distance between the two *outer* particles. That
quantity is `2L·cos(θ/2)`, which is **quadratic** in `θ` near a straight
configuration — its gradient vanishes exactly where a standing plant sits, so
the constraint goes numerically dead and the stem behaves like a rope until it
is already bent. The midpoint-offset form is linear in `θ` and stays well
conditioned.

### The soil clamp

This is the single most important structural detail, and the easiest to get
wrong.

Pinning the root particle fixes the plant's **position** but not its
**direction**. Every bending constraint remains satisfied while the whole stem
rotates rigidly about that one point, so the plant topples like a felled pole
no matter how stiff the material is.

A real stem is *clamped* by the soil: position and tangent are both fixed.
Fixing a tangent needs a second reference point, so the model buries a pinned
**anchor particle** one segment below the root and hinges a bending constraint
at the root between the anchor and the first free particle. That reproduces
the cantilever's built-in end condition, and it is what makes the stem stand
up and bend as an arc.

### Rigidity vs height

Because `I ∝ r⁴`, a modest taper produces a large stiffness gradient. The
default stem goes from 5 mm radius at the soil to 2 mm at the tip, so the
base is `(5/2)⁴ ≈ 39×` stiffer purely from geometry. The observed base/tip
ratio is 23× once the mass distribution is included.

This is the requirement "rigidity should be inversely proportional to the
distance from the ground" — satisfied as a *result* of modelling the taper,
rather than as a rule bolted on afterwards. See `results/stiffness_profile.png`.

---

## 2. Forces

### Aerodynamic drag

```
F = ½ · ρ · C_d · A · |u − v| · (u − v)
```

where `u` is the local air velocity and `v` the particle velocity. The
dependence on the **relative** velocity is essential. As the plant gives way,
`|u − v|` falls, the force falls with it, and the deflection settles where
drag balances the elastic restoring force.

Treating wind as a constant *acceleration* instead removes that feedback
entirely: the force never decreases as the plant yields, so the stem
accelerates until it lies flat. No amount of damping tuning fixes it, because
the problem is the force model, not the dissipation.

### The wind field

```
u(x, t) = d · U · (y/y_ref)^a · [1 + g·sin(2πf_g t − 0.3φ) + b·sin(2πf_t t − φ)]
          + lateral · (cross-flow buffeting term)
```

with `φ = k·(x + 0.7z)` giving spatial decorrelation, so plants at different
positions along a row feel different instantaneous wind.

The `(y/y_ref)^a` term is the atmospheric boundary layer: wind is stronger
higher up. Combined with the `r⁴` stiffness taper, it is the second reason a
real stem bends as an arc — the top of the plant is both weaker *and* more
heavily loaded.

### Damping

Velocity decays as `exp(−c·h)` per substep rather than by a per-frame
multiplier. This matters: a constant factor per step means the effective
damping changes whenever the substep count changes, so the plant silently
becomes a different material when solver settings are tuned. Exponential decay
in `h` is timestep-independent.

---

## 3. The solver

### XPBD, and why not plain PBD

Position-based dynamics projects constraints directly onto positions. Its
well-known flaw is that effective stiffness depends on **iteration count** —
change the iterations and the material changes. XPBD adds a compliance term:

```
Δλ = (−C − α̃λ) / (Σ wᵢ|∇ᵢC|² + α̃)      where  α̃ = α/h²
Δxᵢ = wᵢ ∇ᵢC Δλ
```

Now stiffness is a property of the plant (`α`, from beam theory) and the
iteration count only affects how well the solve converges. That separation is
what makes it possible to state "this stem has `E = 1 GPa`" and have the claim
mean something.

### Frame structure

```
for each substep (default 32):
    reset Lagrange multipliers
    integrate            gravity + drag + damping, predict positions
    for each iteration (default 4):
        solve constraints
    ground contact       non-penetration + friction
    robot contact        push out of collision spheres, log reaction
    update velocities    v = (x − x_prev)/h
```

### Chain mode vs coloured mode

Two constraint-solving strategies are implemented.

**Coloured mode** partitions constraints into conflict-free groups (computed
once at build time by greedy graph colouring, and asserted correct). Within a
colour, no two constraints touch the same particle, so all of them project in
parallel. One thread per constraint.

This is the textbook GPU approach, and for this problem it converges badly.
A stem is a chain: a bending correction at the tip reaches the root only by
travelling hinge to hinge, roughly one hinge per pass. Measured on a 17-node
cantilever, the default work budget left the stem **7× too compliant** — the
floppy-rope failure mode, arrived at through a solver deficiency rather than a
modelling one.

**Chain mode** (the default) runs **one thread per plant** and sweeps that
plant's constraints sequentially — forward along the chain, then backward.
A symmetric sweep over a tridiagonal system propagates corrections end to end,
so it solves the chain almost directly.

The parallelism then comes from the number of plants, which is the right axis
for this workload: an agricultural scene has hundreds or thousands of plants
and only a few dozen particles each.

Both modes solve the same system and the test suite checks they agree when
each is given enough work.

### Convergence: substeps beat iterations

Measured on a uniform 17-node cantilever against `qL⁴/(8EI)`:

| substeps | iterations | work | tip deflection |
|---:|---:|---:|---:|
| 8 | 2 | 16 | 301.8 mm |
| 8 | 8 | 64 | 106.9 mm |
| 8 | 32 | 256 | 60.6 mm |
| 8 | 128 | 1024 | 57.7 mm |
| 16 | 4 | 64 | 76.4 mm |
| **32** | **4** | **128** | **59.1 mm** |
| 64 | 4 | 256 | 58.5 mm |

Both knobs converge eventually, but not at the same price: 64 substeps × 4
iterations reaches what 8 substeps × 128 iterations reaches, for a quarter of
the work. Hence the defaults of 32 substeps and 4 iterations, which sit within
0.5 % of the converged answer.

### Staying on the device

Every array lives on the GPU for the whole run, and a frame is a fixed
sequence of kernel launches. Simulation time is kept in a **device array**
rather than passed as a kernel argument, which lets the entire frame be
captured once as a **CUDA graph** and replayed — otherwise the launch overhead
of the per-frame kernels dominates at small scene sizes.

The only host transfers are the ones a caller explicitly asks for
(`positions()`, `tip_deflection()`, `reaction_forces()`).

This is the difference between a GPU solver and a solver that merely contains
GPU kernels. Copying positions to NumPy and back every frame — a natural
first implementation — makes the simulation bounded by PCIe bandwidth rather
than arithmetic, and it will lose to a decent CPU implementation at every
problem size.

---

## 4. Robot contact

The manipulator is represented as a set of collision spheres. Particles
penetrating a sphere are pushed out along the contact normal, and the
correction is accumulated as a reaction:

```
F = Σ_substeps (m · Δx / h²) / n_substeps
```

Each positional correction `Δx` on a particle of mass `m` corresponds to an
impulse `m·Δx/h`, hence a force `m·Δx/h²`; averaging over the frame's substeps
gives the mean force the effector felt.

The accumulation is **per frame, not per substep**. By the final substep the
plant has already been pushed clear, so sampling only that substep reports no
contact at all while the plant is visibly being shoved aside.

Cross-checked against statics: pushing the default stem 0.22 m sideways at
0.55 m height measured ≈ 1.0 N, against `3EI·d/h³ ≈ 1.9 N`. The simulated
value is lower, which is the expected direction — at 26 % of stem length the
small-deflection formula still assumes a full-length moment arm and so
over-predicts.

---

## 5. Isaac Sim integration

Isaac Sim is a **renderer and robot host** here, not a physics authority.
PhysX has no model for a tapered anisotropic stem, which is why this solver
exists.

`05_isaac_sim.py` registers a physics-step callback that:

1. reads the manipulator's end-effector pose and hands it to the solver as a
   collision sphere,
2. steps the Warp solver,
3. writes the new particle positions into USD `BasisCurves` points.

Plants are written as curves rather than one rigid prim per particle because a
500-plant canopy is 12 500 particles, and 12 500 USD prims would make the
*stage update* the bottleneck rather than the physics.
