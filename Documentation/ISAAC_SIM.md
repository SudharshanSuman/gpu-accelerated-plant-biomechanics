# Isaac Sim integration

Verified working against **Isaac Sim 6.0.1** on this machine.

## Quick start

```bash
./run_isaac.sh
```

That is the whole thing. The launcher finds Isaac, fixes the environment, runs
a canopy, and writes an animated USD to `results/canopy_isaac.usd`.

```bash
./run_isaac.sh --plants 64 --seconds 10 --sweep-speed 0.5
```

```bash
./run_isaac.sh --gui
```

With an articulated manipulator on a driving base — the realistic agricultural
scenario:

```bash
./run_isaac.sh --arm --plants 32 --seconds 10 --sweep-speed 0.4
```

## Integrating into your own scene

Three lines. The bridge is a normal Python object with no global state:

```python
from plantsim import SimConfig
from plantsim.isaac_bridge import PlantCanopyBridge

bridge = PlantCanopyBridge(SimConfig(), n_plants=64)
bridge.build()            # authors USD under /World/Canopy
bridge.attach(world)      # registers the physics callback
```

Per frame, if you have a manipulator:

```python
bridge.set_effector(franka.end_effector.get_world_pose()[0], radius=0.06)
force = bridge.contact_force_magnitude()      # newtons
```

`PlantCanopyBridge` imports Isaac lazily inside the methods that need it, so
the module can be imported by the test suite and by plain-Python scripts
without Isaac Sim present.

### API

| Method | Purpose |
|---|---|
| `build(stage=None)` | author the canopy as USD `BasisCurves` |
| `attach(world, name=...)` | register the solver as a physics callback |
| `set_effector(pos, radius)` | manipulator position for this frame |
| `set_effectors(pos, radii)` | several collision spheres (a multi-link gripper) |
| `contact_force()` | reaction per sphere, N, as a `(k, 3)` array |
| `contact_force_magnitude()` | scalar convenience for a single sphere |
| `publish(time_code=None)` | push solver state into USD |
| `start_recording(fps)` / `record_frame()` | bake motion as time samples |
| `export_usd(path)` | write the stage, animation included |

## Two problems the launcher solves for you

**1. Spaces in the install path.** Isaac's own `python.sh` contains
`source ${SCRIPT_DIR}/setup_python_env.sh` — unquoted. This machine's install
is at `~/Robotics Tools/isaac`, and that space makes the launcher fail with
`/home/sudharshan-s/Robotics: No such file or directory`. `run_isaac.sh`
routes through a space-free symlink in `~/.cache/plantsim/`; bash resolves
`pwd` logically, so Isaac only ever sees a clean path. NVIDIA's files are not
modified.

**2. Import order.** Kit requires `SimulationApp` to be constructed *before*
any `omni` or `isaacsim` module is imported. Violating it produces a confusing
extension error rather than an import error. Both the demo script and the
bridge respect this.

## Version compatibility

Isaac Sim 5 removed the `omni.isaac.*` Python namespace. Most tutorials and
older code — including this project's first draft — use it and will not run on
Isaac Sim 6.

| Old (≤ 4.5) | Current (5.x / 6.x) |
|---|---|
| `omni.isaac.core.World` | `isaacsim.core.api.World` |
| `omni.isaac.core.objects.GroundPlane` | `isaacsim.core.api.objects.GroundPlane` |
| `omni.isaac.franka.Franka` | removed; load a USD via `add_reference_to_stage` |

Confirmed on this install: `isaacsim.core.api` present, `omni.isaac.core`
absent.

## Running without a GPU

The NVIDIA driver on this machine is not loading, so Isaac cannot create a GPU
device and `--gui` will not render. **Headless physics and USD export work
regardless**, which is why the demo records by default:

- the solver runs on the CPU backend,
- motion is baked into the USD as per-frame time samples,
- the resulting file opens on any machine with a working driver.

`results/canopy_isaac.usd` from a 24-plant, 6-second run contains 360 time
samples at 60 fps across 120 `BasisCurves` prims. Open it with
`~/isaac-sim/isaac-sim.sh` or `usdview` and scrub the timeline.

## Why the plants are curves, not prims

A 500-plant canopy is 12,500 particles. One prim per particle would make the
*stage update* the bottleneck rather than the physics. Each plant is instead
one `BasisCurves` prim for the stem plus one per leaf, with per-vertex widths
so the rendered stem tapers exactly as the physics does. Positions are pushed
via `Vt.Vec3fArray.FromNumpy` where available, avoiding a per-point Python
conversion.

PhysX never sees the plants. Isaac Sim is the renderer and robot host; the
biomechanics are this project's Warp solver, because PhysX has no model for a
tapered anisotropic stem.

## The manipulator

Isaac Sim ships **no robot meshes in the install** — Franka, UR10 and the rest
live on NVIDIA's content server. A demo that needs a download fails on an
offline machine and in CI, so `--arm` builds a two-link arm from USD
primitives at runtime and drives it with closed-form kinematics.

```python
from plantsim.isaac_robot import ProceduralArm

arm = ProceduralArm(base_position=(0.0, 0.0, -0.55))
arm.build(stage, "/World/Arm")

arm.set_base_position((x, 0.0, -0.55))          # vehicle drives down the row
arm.set_tool_target((x + 0.1, 0.55, 0.0))       # arm reaches into the canopy
bridge.set_effectors(*arm.collision_spheres())
```

Three things about it matter.

**The vehicle drives; the arm holds a pose.** A real agricultural manipulator
does not sweep a long arc through a canopy — the vehicle moves and the arm
reaches. That makes *driving speed* the parameter governing contact force,
which is the one an operator actually sets.

**Contact is whole-arm, not a point.** A manipulator pushes the canopy with
its forearm, not just its gripper. Each link is sampled into overlapping
collision spheres (25 by default) and the load reported is the vector sum over
all of them.

**Sphere count is derived, not hardcoded.** Proxies must overlap or a stem can
pass *between* two spheres and register no contact at all. The count is
computed from link length and radius, `coverage_ok()` asserts it, and a test
fails if a geometry change ever breaks it. The first version of this arm had
0.125 m spacing on 0.090 m spheres — a gap a stem fits straight through.

IK is exact for reachable targets (round-trips to under 1 µm in the tests) and
clamps to the workspace boundary for unreachable ones rather than raising.

`--robot-usd` can attach a real manipulator instead, but it is **untested**
here for the asset reason above.

## A result worth reporting

Approach speed dominates contact force. From two runs at 24 plants:

| Sweep speed | Peak contact force | Verdict |
|---|---:|---|
| 0.30 m/s | 1.50 N | below the 5 N stem-damage threshold |
| ~2.7 m/s | 10.50 N | **exceeds it** — the manipulator would damage the crop |

`--stem-failure-n` sets the threshold and the run reports how many frames
exceeded it. This is the kind of judgement a rigid-body simulator cannot make
at all, and it is a good candidate figure for the paper: sweep speed on the
x-axis, peak contact force on the y, with the damage threshold as a horizontal
line.
