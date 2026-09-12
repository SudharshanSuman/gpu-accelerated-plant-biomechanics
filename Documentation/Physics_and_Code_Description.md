# Physics Formulas and Code Description

*GPU-Accelerated Plant Biomechanics Simulation for Agricultural Robotics*

A complete technical reference: every equation traced to the line of code that implements it.

**Sudharshan S · URK23CS1300**
**Karunya Institute of Technology and Sciences**

---

---

> ## ⚠ Reconciliation notice — read before citing anything in this document
>
> **This document describes an earlier prototype, not the simulator in `src/`.**
> It was checked against the code on 2026-09-12 and the two disagree in ways
> that matter for a paper. No function named `pbd_solve()`,
> `compute_pbd_stiffness()`, `build_sunflower_plant()`, `k_apply_forces()`,
> `flexural_rigidity()` or `cycle_wind_direction()` exists anywhere in this
> repository, and that prototype's source is not on this machine.
>
> Where the two disagree, **`src/` is authoritative**: it is the version that
> is tested (15 passing tests), validated against beam theory, and used to
> produce every figure in `results/`.
>
> ### Numerical errors that must not reach the paper
>
> | Quantity | This document | Actual code | Why it matters |
> |---|---|---|---|
> | `E_stem` | 10–20 MPa | **1.0 GPa** | At 15 MPa the simulated plant **collapses flat under its own weight** (tip height 0.000 m). This is reproducible in one run: `plantsim.validation.material_sanity()`. A reviewer can falsify it in a minute. |
> | `E_leaf` | 2–5 MPa | **0.1 GPa** | Same failure mode. |
> | `Cd_leaf` | 2.5–3.0 | **1.4** | A flat plate normal to flow is ≈1.1–2.0. Values above 2 apply to concave shapes (parachutes ≈1.75), and real leaves *reconfigure* under load, which lowers Cd rather than raising it. |
> | `Cd_stem` | 1.2–1.8 | **1.1** | Within range, but the paper must quote one number. |
> | `R_base` / `R_tip` | 8–10 / 2–4 mm | **5.0 / 2.0 mm** | The prototype's geometry is a maize-class stalk; `src/` models a slender row crop. Both are defensible — but `EI ∝ r⁴`, so they are not interchangeable. |
> | `DAMPING` | 0.97 per substep | **1.2 s⁻¹ (rate)** | A per-substep factor makes the material change whenever the substep count changes. Damping must be a rate. |
> | `SUBSTEPS` / `PBD_ITERS` | 6–8 / 3–4 | **32 / 4** | Measured: at 8×2 the stem is several times too compliant. See `VALIDATION.md` §2. |
> | Solver | PBD | **XPBD** | In plain PBD, stiffness depends on iteration count, so "E = X Pa" is not a well-defined claim. |
> | Constraint solve | CPU (§B.6) | **GPU-resident** | The paper's central contribution is GPU acceleration; a CPU constraint solver contradicts it. |
>
> ### Features described here but **not implemented** in `src/`
>
> Do not claim these in the paper without implementing them first:
> §A.4 and §A.17 rain loading; §A.15 robot-arm forward kinematics (`src/` takes
> the end-effector pose as input rather than computing it); §A.3 uses a
> **logarithmic** wind profile, whereas `src/` uses a **power-law** profile.
>
> ### What to do
>
> 1. Treat `Documentation/PARAMETERS.md` as the single source of truth — it is
>    generated from `config.py` by `src/scripts/07_parameter_table.py` and
>    cannot drift.
> 2. Use `ARCHITECTURE.md` for the method section.
> 3. Keep this document only as a record of the earlier design.

---

## How to Read This Document

This document has two parts. **Part A** lists every physics equation used anywhere in the project, in plain language first and then with full mathematical notation. **Part B** walks through every function and class in the codebase, explaining what it does, what it takes in, what it gives back, and any tricky implementation detail worth remembering. Each physics equation in Part A is cross-referenced to the exact function in Part B that implements it.

### Quick Formula-to-Code Index

| Formula | Implemented in | Section |
|---|---|---|
| Gravity | `k_apply_forces()` | A.1 / B.5 |
| Aerodynamic wind drag | `k_apply_forces()` | A.2 / B.5 |
| Logarithmic wind height profile | `k_apply_forces()` | A.3 / B.5 |
| Rain momentum flux | `k_apply_forces()` | A.4 / B.5 |
| Node mass from density | `build_single_plant()` | A.5 / B.3 |
| Cross-sectional area | `area()` | A.6 / B.2 |
| Second moment of area | `second_moment()` | A.7 / B.2 |
| Flexural rigidity (Young's mod.) | `flexural_rigidity()` | A.8 / B.2 |
| EI → PBD stiffness mapping | `compute_pbd_stiffness()` | A.9 / B.2 |
| PBD distance constraint | `pbd_solve()` | A.10 / B.6 |
| PBD bending constraint | `pbd_solve()` | A.11 / B.6 |
| Semi-implicit Euler integration | `k_apply_forces()` | A.12 / B.5 |
| Acceleration clamping | `k_apply_forces()` | A.13 / B.5 |
| Soil damping | `k_apply_forces()` | A.14 / B.5 |
| Robot forward kinematics | `SimpleRobotArm.forward_kinematics()` | A.15 / B.7 |
| Contact / gripper repulsion force | `apply_robot_contact()` | A.16 / B.8 |
| Rain particle ballistic motion | `RainParticles.step()` | A.17 / B.9 |

---

## Part A — Physics Formulas Used

> **In simple words:** Every force acting on the plant comes from a real equation used in physics and engineering — nothing is guessed. This section lists them one at a time, explains what each symbol means, and gives the real-world values we used for a sunflower plant.

> **Technically:** All quantities are expressed in SI units (metres, kilograms, seconds, Newtons). Material constants (Young's modulus, density, drag coefficients) are drawn from published plant-biomechanics literature and adjusted within realistic bounds for visual clarity in the simulation.

### A.1 — Gravity

Every mass in the simulation is pulled straight down by Earth's gravity.

$$F_{gravity} = -m \cdot g \cdot \hat{j} \quad \text{(A.1)}$$

- **m** — mass of the node (kg)
- **g** — gravitational acceleration, 9.81 m/s²
- **ĵ** — unit vector pointing up (+Y); force acts in -Y

### A.2 — Aerodynamic Wind Drag

Wind pushes on every part of the plant proportional to the square of the relative wind speed — this is the same drag equation used for cars, aircraft, and sails.

$$F_{wind} = \tfrac{1}{2}\rho_{air} \, C_d \, A \, |v_{rel}| \, v_{rel} \quad \text{(A.2)}$$

- **ρ_air** — air density, 1.225 kg/m³
- **Cd** — drag coefficient. ~~1.2–1.8 for stem, 2.5–3.0 for leaves~~ — **corrected: 1.1 for stem (bluff cylinder in cross-flow), 1.4 for leaves (flat plate normal to flow).** A rigid flat plate sits at ≈1.1–2.0; values above 2 describe concave shapes such as parachutes (≈1.75), and real leaves reconfigure under load, which *lowers* effective Cd at high wind rather than raising it.
- **A** — frontal (projected) area of the node facing the wind (m²)
- **v_rel** — wind velocity minus the node's own velocity (m/s); using the vector form (not just speed) makes the node accelerate toward the wind direction

*Note the direction: multiplying by v_rel (not just its magnitude) means the force always points along the relative wind direction — this is what lets the simulator support wind from any compass direction, not just +X.*

### A.3 — Logarithmic Wind Height Profile

In real life, wind is slower near the ground (friction with soil and grass) and faster higher up. We approximate this with a simple linear height factor.

$$h_{factor}(y) = 0.2 + 0.9 \cdot y \quad \text{(A.3)}$$

- **y** — height of the node above ground (m)
- **h_factor** — multiplier applied to the wind-drag force; 0.2 at ground level, up to ~1.28 at the top of a 1.2 m sunflower

### A.4 — Rain Momentum Flux

Each raindrop hitting a leaf or stem transfers momentum to it. The total downward push depends on how much water is falling, how fast it falls, and how much surface area catches it.

$$F_{rain} = \rho_{water} \cdot R \cdot v_{rain} \cdot A \quad \text{(A.4)}$$

- **ρ_water** — density of water, 1000 kg/m³
- **R** — rain rate (m of rain per second, per m² of catching area); tuned to 1.5×10⁻⁴
- **v_rain** — terminal velocity of falling raindrops, ≈ 9 m/s
- **A** — frontal area of the node
- **multipliers** — leaves multiply this by ×30–35 (broad flat area catches rain efficiently); stem multiplies by ×5–6

*The resulting force is split: most of it (F_rain) pushes straight down, and a smaller fraction (0.5–0.6× F_rain) pushes sideways in the wind direction, modelling rain being blown at a slight angle.*

### A.5 — Node Mass from Material Density

Instead of assigning arbitrary weights, every node's mass comes from real material density times its actual volume.

$$m = \rho_{stem} \cdot A(r) \cdot L_0 \quad \text{(A.5)}$$

- **ρ_stem** — stem tissue density, 700 kg/m³ (typical for herbaceous plant stems)
- **A(r)** — cross-sectional area at that point (see A.6)
- **L₀** — rest length of the segment this node represents (m)

### A.6 — Cross-Sectional Area

$$A(r) = \pi \cdot r^2 \quad \text{(A.6)}$$

- **r** — stem or leaf radius at that point (m)

### A.7 — Second Moment of Area

This measures how resistant a circular cross-section is to bending — it grows with the fourth power of radius, which is the single most important equation in the whole project.

$$I(r) = \frac{\pi \cdot r^4}{4} \quad \text{(A.7)}$$

- **r** — radius at that point (m)

*Why r⁴ matters: our stem tapers from 8 mm at the base to 2–4 mm at the tip — a 2–4× reduction in radius. Because I depends on r⁴, this alone makes the tip 16–256× less resistant to bending than the base, which is exactly the rigid-base / floppy-tip behaviour real plants show.*

### A.8 — Flexural Rigidity (Young's Modulus)

Flexural rigidity (EI) is the true engineering measure of bending stiffness — it combines the material's inherent stiffness (E) with the shape's resistance to bending (I).

$$EI(r) = E \cdot I(r) = E \cdot \frac{\pi \cdot r^4}{4} \quad \text{(A.8)}$$

- **E** — Young's modulus. ~~10–20 MPa for stem, 2–5 MPa for leaf~~ — **corrected: 1.0 GPa for stem, 0.1 GPa for leaf.** The original figures are roughly the modulus of isolated parenchyma tissue, not the *effective flexural modulus of a whole stem*, which is dominated by the stiff fibrous outer ring and measures ≈0.3–5 GPa for herbaceous crops. The distinction is decisive rather than academic: simulated at 15 MPa, this plant collapses flat under its own weight (settled tip height 0.000 m). Herbaceous stems are indeed softer than wood (~10 GPa), but by roughly one order of magnitude, not three.
- **I(r)** — second moment of area from A.7

### A.9 — Converting EI into a Position-Based-Dynamics Stiffness

Our constraint solver (PBD, explained in A.10–A.11) needs a stiffness value between 0 (completely floppy) and 1 (perfectly rigid), not a raw EI value in Pascals. We normalise EI against the stem's base value and reshape the curve so it feels natural.

$$k = 0.04 + (0.92 - 0.04) \cdot \text{clip}\left(\frac{EI(r)}{EI(R_{base})}, 0, 1\right)^{0.4} \quad \text{(A.9)}$$

- **EI(r)** — flexural rigidity at this segment's radius
- **EI(R_base)** — flexural rigidity at the thickest (root) radius — the normalising reference
- **0.4 exponent** — a shaping power that spreads out the stiffness curve so the middle of the stem isn't either "all rigid" or "all floppy" — it gives a smooth gradient
- **k** — final PBD stiffness, clamped to [0.04, 0.92] so no segment is ever perfectly rigid or perfectly free

### A.10 — PBD Distance Constraint

After forces move each node freely for one substep, this constraint pulls neighbouring nodes back to their correct rest distance, preventing the stem from stretching like rubber.

$$|p_{child} - p_{parent}| \to L_0 \quad \text{(corrected every substep)} \quad \text{(A.10)}$$

- **p_child, p_parent** — current 3D positions of the two connected nodes
- **L₀** — rest length of that segment
- **correction rule** — if the parent is the root (fixed to ground), only the child moves; otherwise both move half the error, in opposite directions

### A.11 — PBD Bending Constraint

This second constraint pass makes each segment want to point in the same direction as it did at rest, weighted by how stiff that segment is — this is what actually produces visible bending behaviour rather than just a floppy chain.

$$p_{child} \leftarrow p_{child} \cdot (1-k) + \text{target} \cdot k \quad \text{(A.11)}$$

- **target** — the position the child WOULD be at if the segment pointed in its original rest direction, at its current length
- **k** — PBD stiffness for this segment, from A.9
- **effect** — k close to 0.92 (near the root) keeps the segment almost frozen in its rest orientation; k close to 0.04 (near the tip) lets the segment swing almost freely

### A.12 — Semi-Implicit Euler Integration with Damping

Once the total force on a node is known, this equation converts force into a new velocity and position for the next time step.

$$v_{t+dt} = (v_t + (F/m) \cdot dt) \cdot d \quad \text{(A.12a)}$$

$$p_{t+dt} = p_t + v_{t+dt} \cdot dt \quad \text{(A.12b)}$$

- **F/m** — acceleration (Newton's second law, a = F/m)
- **dt** — substep time, 1/60 s divided by the substep count
- **d** — damping factor (0.97 normally, or a lower "soil damping" value for nodes below ground level) — removes a little energy every step so oscillations settle instead of growing forever

### A.13 — Acceleration Clamping (Stability Safeguard)

Stiff springs combined with large forces can occasionally produce extremely large accelerations that make the simulation explode into NaN values. This safeguard caps acceleration at a safe ceiling.

$$\text{if } |a| > a_{max}: \quad a \leftarrow a \cdot \frac{a_{max}}{|a|} \quad \text{(A.13)}$$

- **a_max** — 800–1000 m/s², chosen empirically as the largest acceleration that still lets the plant respond visibly without becoming unstable

### A.14 — Soil Damping and Ground Collision

Nodes that dip below the soil surface (from a very strong downward force) are damped much more heavily, simulating the resistance of packed earth, and are never allowed to go below ground level.

$$\text{if } y < \text{SOIL\_DEPTH}: \quad d \leftarrow 0.55 \quad \text{(instead of 0.97)} \quad \text{(A.14a)}$$

$$\text{if } y < 0: \quad y \leftarrow 0, \quad v_y \leftarrow 0 \quad \text{(A.14b)}$$

### A.15 — Robot Arm Forward Kinematics

The robot arm is a chain of rigid links connected by rotating joints. Forward kinematics means: given the joint angles, compute where every joint and the gripper actually is in 3D space.

$$p_{i+1} = p_i + L_i \cdot (\cos\theta_i, \sin\theta_i, 0) \quad \text{(A.15)}$$

- **p_i** — position of joint i
- **L_i** — length of link i (0.30 m, 0.25 m, 0.15 m for the three links)
- **θ_i** — absolute angle of link i, built up by summing each joint's own angle with all the angles before it in the chain
- **driving function** — joint angles are driven by a smooth ease curve `s = 0.5·(1 − cos(2πt))`, which goes 0 → 1 → 0 so the arm reaches toward the plant and retracts smoothly with no sudden velocity changes

### A.16 — Robot–Plant Contact (Repulsion) Force

When the robot's gripper gets close to a plant node, it pushes that node away — but only nodes very close to the gripper, and harder the closer they are. This makes the plant deform only exactly where it's touched.

$$F_{contact} = \left(\frac{\text{penetration}}{r_{contact}}\right)^2 \cdot F_{strength} \cdot \hat{d} \quad \text{(A.16)}$$

- **penetration** — r_contact − distance(node, gripper); how far the node is "inside" the contact zone
- **r_contact** — radius of the contact zone, 0.08–0.10 m
- **F_strength** — maximum push strength, 120–200 (tuned units)
- **d̂** — unit vector pointing from the gripper toward the node
- **quadratic falloff** — squaring the penetration ratio means a node barely inside the contact zone is pushed very gently, while a node right at the gripper's centre is pushed hard — this localises the deformation realistically

### A.17 — Rain Particle Motion (Visual Layer Only)

Separately from the physics forces in A.4, we also draw individual raindrops falling through the 3D scene for visual effect. These follow simple ballistic motion and do not affect the plant physics.

$$p_{t+dt} = p_t + (v_{drift}, -9.0, 0) \cdot dt \quad \text{(A.17)}$$

- **v_drift** — small horizontal velocity (1.5–2.0 m/s) so rain appears to be blown sideways slightly
- **−9.0** — vertical fall speed matching the physics terminal velocity in A.4
- **respawn rule** — when a drop's y position reaches 0, it is teleported back to a random position near the top of the scene

---

## Part B — Code Architecture and Function Description

> **In simple words:** The code is organised into clear building blocks: first we describe one plant's shape and material, then we let the GPU push and pull on every node, then a second pass fixes up distances so the plant doesn't stretch, and finally everything is drawn on screen.

> **Technically:** The pipeline follows a standard real-time physics architecture: (1) scene construction — build flat NumPy arrays describing topology, mass, and material; (2) GPU force integration — a single Warp kernel computes forces and performs semi-implicit Euler integration for every node in parallel; (3) CPU constraint projection — a sequential Gauss-Seidel Position-Based-Dynamics pass corrects distances and bending, exploiting the chain's root-to-tip dependency structure; (4) rendering — Open3D meshes are rebuilt every frame from the live node positions.

### B.1 — High-Level Architecture (in words)

- `build_single_plant()` / `build_sunflower_plant()` — creates the node graph (positions, mass, stiffness) for ONE plant
- `build_field()` — stacks many single plants into one big flat array for GPU-parallel simulation
- `k_apply_forces()` — GPU kernel; every node computed by its own thread
- `pbd_solve()` — CPU function; runs after every GPU substep to keep the stem/leaves from stretching and to apply bending
- `SimpleRobotArm` — computes where the robot's joints and gripper are at any moment in time
- `apply_robot_contact()` — pushes plant nodes away from the gripper when they get close
- `RainParticles` — purely visual raindrop animation, independent of the physics forces
- `make_*_mesh()` functions — convert live node positions into Open3D 3D geometry every frame
- `SimState` — holds all keyboard-toggleable settings (wind on/off, rain on/off, arm on/off, wind direction/speed)
- `visualize_interactive()` — the main loop: physics step → contact → PBD → rebuild geometry → render → repeat

### B.2 — Material and Stiffness Functions

#### `stem_radius(i, n)`
*Figures out how thick the stem is at a given point, tapering smoothly from thick at the bottom to thin at the top.*

Linearly interpolates between R_BASE and R_TIP based on normalised node index `t = i/(n-1)`.

- **Inputs:** i (node index), n (total stem nodes)
- **Returns:** radius in metres (float)

#### `area(r)`
*Computes how much cross-sectional area a stem of a given thickness has.*

Implements equation A.6: A = π·r².

- **Inputs:** r (radius, metres)
- **Returns:** cross-sectional area (m²)

#### `second_moment(r)`
*Measures how hard it is to bend a rod of this thickness — the key quantity that makes thick parts stiff and thin parts floppy.*

Implements equation A.7: I = π·r⁴/4.

- **Inputs:** r (radius, metres)
- **Returns:** second moment of area (m⁴)

#### `flexural_rigidity(E, r)`
*Combines how stiff the plant material itself is with how thick the stem is at this point, giving the true bending stiffness.*

Implements equation A.8: EI = E · I(r).

- **Inputs:** E (Young's modulus, Pa), r (radius, m)
- **Returns:** flexural rigidity EI (N·m²)

#### `compute_pbd_stiffness(E, r, L0)`
*Converts the raw engineering stiffness number into a 0–1 scale that our constraint solver can use directly.*

Implements equation A.9: normalises EI(r) against EI at the base radius, applies a 0.4 power-law reshaping, and rescales into [0.04, 0.92].

- **Inputs:** E (Young's modulus), r (radius at this segment), L0 (rest length, unused in formula but kept for API consistency)
- **Returns:** PBD stiffness k (float, 0.04 to 0.92)
- **Key detail:** This is the single function most responsible for the visually correct "rigid base, floppy tip" behaviour — get this formula wrong and the whole plant either looks like a rigid pole or a wet noodle.

### B.3 — Plant Construction

#### `build_single_plant(x_offset)` / `build_sunflower_plant(x_offset)`
*Builds one complete plant from scratch: a chain of stem nodes, leaves branching off at several heights, and (in the sunflower version) a flower head at the very top.*

Constructs parallel NumPy arrays: `pos`, `rest_pos`, `vel`, `mass`, `is_root`, `is_leaf`, `frontal_area`, `parents`, `children`, `L0`, `bend_k`. Stem nodes are placed at increasing height; leaves are attached at specific stem indices (`LEAF_CONFIG`) at fixed angles, each leaf being its own short chain of 2–4 segments. Applies equations A.5 (mass) and A.9 (stiffness) to every stem and leaf edge as it is created.

- **Inputs:** x_offset (where along the X-axis this plant's root sits — lets multiple plants be placed side by side)
- **Returns:** dict containing all the arrays needed to simulate this plant
- **Key detail:** `parents[]` and `children[]` describe a tree structure (a graph, not just a straight line) — this is what lets leaves branch off the main stem at multiple points.

#### `build_field(n_plants, spacing)`
*Takes the single-plant builder and repeats it many times, placing each plant a fixed distance apart, then glues all the resulting arrays together into one giant array so the GPU can process every plant at once.*

Calls `build_single_plant()` n_plants times with increasing x_offset, then `np.concatenate()`s every field across all plants. Critically, parent/child edge indices from each plant are shifted by a running `node_offset` so they still point to the correct nodes inside the merged flat array.

- **Inputs:** n_plants (how many plants), spacing (metres between plant roots, default 0.6)
- **Returns:** dict with the same fields as `build_single_plant()`, but sized for the whole field, plus `n_nodes_per_plant` for indexing individual plants back out during rendering
- **Key detail:** This function requires ZERO changes to the GPU kernel — the same `k_apply_forces()` kernel runs unmodified whether there are 32 nodes (1 plant) or 1600 nodes (50 plants).

### B.4 — GPU Setup

Before the simulation loop starts, every NumPy array is uploaded once to GPU memory using `wp.array(..., device="cuda")`. Position and velocity arrays are read back to the CPU once per substep (for the PBD pass) and re-uploaded — this hybrid CPU/GPU round-trip is the main scalability bottleneck discussed in the benchmark results.

### B.5 — The GPU Force Kernel

#### `k_apply_forces(...)`
*This is the single function that runs on the graphics card. It looks at ONE node, adds up gravity, wind, and rain forces on it, and moves it forward in time by one small step — and it does this for every node in the plant (or field of plants) all at the exact same instant.*

A `@wp.kernel` decorated function. `wp.tid()` gives each GPU thread its own node index i. Root nodes are frozen (velocity forced to zero) and skipped. For all other nodes: computes gravity (A.1), wind drag (A.2) scaled by the height profile (A.3), rain force (A.4) if `rain_on`, sums them into f, converts to acceleration `a = f/m`, clamps a per equation A.13, applies soil damping (A.14a), then integrates velocity and position per equation A.12. Finally clamps y ≥ 0 (A.14b).

- **Inputs:** positions, velocities (wp.vec3 arrays); mass, area (wp.float32 arrays); is_root, is_leaf (wp.int32 flag arrays); wind_x or wind_vec (float or wp.vec3); rain_on (int flag); physical constants (rho_air, cd_stem, etc.)
- **Returns:** None — modifies positions and velocities arrays IN PLACE on the GPU
- **Key detail:** Launched with `wp.launch(k_apply_forces, dim=n, inputs=[...])` where n is the TOTAL node count across every plant in the field — this one line is what makes multi-plant simulation "free" on the GPU side.

### B.6 — The CPU Constraint Solver

#### `pbd_solve(pos, rest_pos, parents, children, L0, bend_k, is_root, n_iters)`
*After the GPU has moved every node freely under forces, this function goes through the plant from root to tip and gently pulls each segment back to its correct length and, separately, nudges it to point the way it did before any force acted on it — proportional to how stiff that part of the plant is.*

Two nested loops, repeated n_iters times (usually 3–4): Pass 1 walks every edge and corrects the distance between parent and child toward L0 (equation A.10), using Gauss-Seidel ordering (root-fixed edges move only the child; others split the correction). Pass 2 walks every edge again and blends the child's position toward a bending target based on bend_k (equation A.11). Finally clamps every y-coordinate to ≥ 0.

- **Inputs:** pos, rest_pos (NumPy arrays, downloaded from GPU); parents, children (int arrays defining the tree); L0, bend_k (per-edge float arrays); is_root (flags); n_iters (solver iteration count)
- **Returns:** the corrected pos array (NumPy)
- **Key detail:** This function MUST run on CPU, not GPU, because the correction for node c depends on the just-corrected position of its parent p — a sequential dependency chain that GPU threads running in parallel cannot respect without explicit synchronisation. This is why the project uses a hybrid CPU–GPU architecture rather than a pure-GPU one.

### B.7 — Robot Arm

#### `class SimpleRobotArm`
*Represents a 3-jointed robot arm standing on the ground next to the plant, with a gripper (a red ball) at the very tip that reaches toward the plant and pulls back, over and over.*

Stores base position and the three link lengths. The core method `forward_kinematics(t)` computes joint positions p0 (base) through p4 (gripper tip) using equation A.15, driven by a smooth ease-in-ease-out parameter t ∈ [0,1] that represents where in the reach-and-retract cycle the arm currently is.

- **Inputs:** base_pos (where the arm's base sits on the ground)
- **Returns:** `forward_kinematics(t)` returns a list of 5 wp.vec3-compatible positions [p0, p1, p2, p3, p4]; `get_gripper_pos(t)` returns just p4
- **Key detail:** The angle formulas use `np.pi - theta1` rather than theta1 directly for the first joint — this was a deliberate fix so the arm's elbow bends toward the plant (in -X) instead of away from it.

### B.8 — Robot–Plant Contact

#### `apply_robot_contact(pos, vel, gripper_pos, field, force_strength)`
*Checks every single node in the plant: is it close enough to the gripper to be touched? If yes, give it a push away from the gripper — harder if it's very close, gentler if it's near the edge of the contact zone.*

Loops over all nodes; skips root nodes (which are pinned to the ground and must not move). For each remaining node, computes distance to gripper_pos; if inside contact_radius, applies the quadratic repulsion force from equation A.16 directly to the node's velocity.

- **Inputs:** pos, vel (current NumPy position/velocity arrays); gripper_pos (current 3D gripper location); field (plant data, for is_root flags); force_strength (push intensity, default 120–200)
- **Returns:** the modified vel array
- **Key detail:** This function is called once per physics substep, right before `pbd_solve()` — the order matters: contact changes velocity, then PBD re-applies distance constraints so the push doesn't stretch or tear the stem.

### B.9 — Rain Visual Effect

#### `class RainParticles`
*Manages a swarm of individual raindrops that fall through the sky in the 3D scene — purely for visual effect, separate from the rain FORCE that actually pushes the plant.*

Stores an (n_drops, 3) position array and a matching velocity array. `step(dt, active)` advances every drop by its velocity (equation A.17) and respawns any drop that reaches the ground back near the top of the scene at a new random position.

- **Inputs:** n_drops (default 250–300), region (bounding box for where drops can appear)
- **Returns:** `step()` returns the current (n_active, 3) array of drop positions to draw this frame
- **Key detail:** When rain is toggled off (`state.rain_on = False`), `step()` returns an empty array and no rain geometry is drawn — but the particle objects themselves persist in memory so rain can resume instantly when toggled back on.

### B.10 — Open3D Rendering Functions

**`make_stem_mesh(positions)`** — Builds a chain of tapered cylinders connecting consecutive stem node positions — radius shrinks from base to tip to match the real stem taper.

**`make_leaf_mesh_cylinders(positions)`** — Builds a chain of green cylinders for one leaf, plus a small sphere at the tip — chosen over flattened-sphere "blade" geometry because cylinder rotation makes wind-driven motion visually obvious.

**`make_sunflower_head(center, stem_direction, radius)`** — Builds a brown flattened disk plus 16 flattened yellow spheres arranged in a ring, then rotates the whole assembly so it faces along whatever direction the stem tip is currently pointing — this is what makes the flower visibly tilt as the stem bends.

**`make_robot_arm_mesh(joints)`** — Builds the visual robot: a box base platform, cylinders for each link, dark spheres at the intermediate joints, and one large bright red sphere at the gripper tip (deliberately oversized so it never gets visually lost inside the arm geometry).

**`make_ground_with_grid(size)`** — Builds a flat brown box for the ground plus a wireframe grid drawn just above it, giving depth perception that a plain flat colour lacks.

**`make_rain_lines(rain_positions)`** — Converts the current raindrop positions into a LineSet — each drop becomes a short diagonal line segment (a streak) rather than a dot, which reads much more clearly as "falling rain" to the eye.

**`make_wind_arrow(direction, strength, position)`** — Builds a 3D arrow (cylinder shaft + cone head) hovering above the scene, oriented along the current wind direction with its length scaled by wind speed — a compact way to show an invisible force.

### B.11 — Interactive State and Main Loop

#### `class SimState`
*A single object that remembers whether wind, rain, and the robot arm are currently switched on, plus the current wind speed and direction — the keyboard callbacks all just flip flags on this one object.*

Plain Python class with boolean flags (wind_on, rain_on, arm_on), a float (wind_speed), and a direction vector cycled through 4 presets by `cycle_wind_direction()`. No physics happens inside this class — it purely holds settings that the main loop reads every frame.

- **Key detail:** Because it's just a plain object (not tied to Open3D directly), the keyboard callback for each key is a one-line lambda that calls the matching toggle method.

#### `visualize_interactive(plant_data)`
*This is the function that actually opens the 3D window and keeps it running: read keyboard state → simulate physics → apply robot contact → fix up distances → redraw everything → repeat, roughly 60 times a second, forever until the window is closed.*

Uploads plant arrays to GPU once. Creates an `o3d.visualization.VisualizerWithKeyCallback` and registers W/R/A/S/+/- key callbacks that mutate a shared SimState instance. Main while-loop: reads state each frame to build the current wind vector and rain flag, runs SUBSTEPS iterations of (GPU force kernel → contact force → CPU PBD solve), then removes all previous frame's Open3D geometry and rebuilds it from the fresh node positions (stem, leaves, sunflower head, robot arm, wind arrow, rain), then calls `vis.poll_events()` and `vis.update_renderer()`.

- **Inputs:** plant_data (the dict returned by `build_single_plant` / `build_sunflower_plant`)
- **Returns:** None — runs until the window is closed
- **Key detail:** Geometry is destroyed and rebuilt from scratch every single frame rather than updated in place. This is simpler to write correctly and fast enough for one plant, but is the reason frame rate would need optimisation (in-place vertex updates) before scaling this exact visualizer to dozens of plants.

### B.12 — Benchmarking

#### `benchmark()`
*Runs the whole simulation at several different field sizes (1, 5, 10, 25, 50 plants) and times how many frames per second the GPU can manage at each size — this produces the table used to prove the system scales.*

For each n_plants value: builds a field with `build_field()`, runs it for 60 frames while timing with `time.perf_counter()`, computes FPS = frames / elapsed_seconds, and prints a formatted row.

- **Inputs:** none (iterates a fixed list of plant counts internally)
- **Returns:** prints a results table to the console; can be modified to return the raw results list for programmatic use

---

## Summary — Constants Used Throughout the Project

> **SUPERSEDED.** The table that stood here listed the prototype's constants,
> several of which are physically impossible (see the reconciliation notice at
> the top of this document — a 10–20 MPa stem cannot hold itself up).
>
> The authoritative table is **`Documentation/PARAMETERS.md`**, generated
> directly from `src/plantsim/config.py`. Regenerate it with:
>
> ```bash
> python3 src/scripts/07_parameter_table.py
> ```
>
> That script also writes `paper/parameters.tex` for direct inclusion in the
> manuscript, so the paper's table cannot drift from the code either.
| contact_radius | 0.08–0.10 m | Robot gripper contact zone size |

---

*End of document — Sudharshan S, Karunya University, 2026*
