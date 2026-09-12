# Related work

Every entry below was verified to exist on 2026-09-12. Nothing here is
reconstructed from memory — if a paper is not listed, it has not been checked,
and it must not go into the bibliography until it has been.

---

## The two papers that most constrain this project's novelty claim

### Gazebo Plants (2024)

> Junchen Deng, Samhita Marri, Jonathan Klein, Wojtek Pałubicki, Sören Pirk,
> Girish Chowdhary, Dominik L. Michels. *Gazebo Plants: Simulating
> Plant–Robot Interaction with Cosserat Rods.* arXiv:2402.02570, 2024.

A Gazebo plugin that models plants as Cosserat rods to simulate plant–robot
interaction, demonstrated on robotic fruit-picking and compared against
real-world harvesting experiments.

**Why it matters here:** it already occupies "deformable plants + robot
interaction in a robotics simulator". The claim "existing simulators treat
plants as rigid" is *not* defensible as stated, and a reviewer will know this
paper. It must be cited and positioned against, not omitted.

### OrchardBench (2026)

> Humphrey Munn. *OrchardBench: A Physically-Grounded, GPU-Parallel
> Apple-Orchard Simulation Benchmark for Agricultural Robotics.*
> arXiv:2607.06337, July 2026.

GPU-parallel orchard simulation on the Newton engine. Branches are compliant
torsional spring-dampers **with stiffness from Euler–Bernoulli beam theory**;
branches break at a wood modulus of rupture; fruit detaches at
literature-grounded pull forces; many parallel environments run at interactive
rates on a laptop GPU, with every physical parameter tied to a published
source.

**Why it matters here:** this is the closest prior art and it is two months
old. It overlaps this project on *GPU-parallel*, *beam-theory stiffness*,
*robot contact*, and *parameters from published sources* simultaneously. Any
claim of novelty that does not explicitly separate this work from OrchardBench
will not survive review.

---

## Other verified context

> Humphrey Munn et al. *CropCraft: A Procedural World Generator for Robotic
> Simulation of Agricultural Tasks.* arXiv:2511.02417.

Procedural agricultural world generation (Blender + Python, YAML-configured
crop fields). Geometry generation, not physics — complementary rather than
competing.

> *Agri-Sim: Agricultural Simulation Platform for Embodied Intelligence
> Evaluation in Greenhouse Robotics.* arXiv:2608.29100.

Unity–ROS2 greenhouse platform with a dual-arm harvesting robot. Unity handles
**rigid-body** dynamics; it names crop deformability as a difficulty rather
than solving it. This is the clearest example supporting the "mainstream
platforms are rigid" framing — cite this for that claim rather than making it
about Gazebo generally.

> *Find the Fruit: Zero-Shot Sim2Real RL for Occlusion-Aware Plant
> Manipulation.* arXiv:2505.16547.

Sim-to-real transfer for plant manipulation — useful for motivating why
plant-contact fidelity matters downstream.

## Method foundations (standard, safe to cite)

- Müller, Heidelberger, Hennix, Ratcliff. *Position Based Dynamics.* 2007.
- Macklin, Müller, Chentanez. *XPBD: Position-Based Simulation of Compliant
  Constrained Dynamics.* 2016.
- Macklin et al. *Small Steps in Physics Simulation.* 2019. — the
  substeps-beat-iterations result, reproduced independently in `VALIDATION.md`.

---

## A novelty claim that actually survives this prior art

The original framing — "GPU-accelerated deformable plants for agri robots" —
is now occupied by OrchardBench. The following is what remains, and it is
narrower but defensible:

> **Wind-driven biomechanics of slender herbaceous row crops**, modelled as
> XPBD particle chains whose bending compliance is derived from a tapered
> `EI ∝ r⁴` profile, with the solver's parallelisation strategy chosen from a
> measured convergence study, and correctness established against a
> closed-form cantilever solution and an independent second implementation.

The separations from the closest prior art, stated precisely:

| | OrchardBench | Gazebo Plants | This work |
|---|---|---|---|
| Plant type | woody apple trees | plants w/ branches | slender herbaceous row crops |
| Element model | articulated bodies, torsional spring-dampers | Cosserat rods | XPBD particle chains, continuous taper |
| Dominant load | manipulator contact, fruit pull | manipulator contact | **aerodynamic wind drag** |
| Stiffness variation | per-branch, beam theory | rod parameters | **continuous `r⁴` taper along one stem** |
| Correctness evidence | parameters from published sources | vs. real harvesting experiments | **closed-form cantilever + second implementation** |
| GPU | yes, measured | no | implemented, **not yet measured** |

**Wind loading is the cleanest separation.** Neither of the closest papers
treats aerodynamic drag as the primary load case, and it is the load case that
matters for perception training — a plant's silhouette changes continuously in
wind, which is precisely what a rigid or contact-only model cannot provide.

**Be honest about the GPU row.** Until the driver works and
`03_benchmark.py` produces CUDA timings, "GPU-accelerated" cannot be a
headline claim in a paper that is up against OrchardBench's *measured*
GPU results. Either get the numbers or lead with the wind-biomechanics and
validation contributions instead.
