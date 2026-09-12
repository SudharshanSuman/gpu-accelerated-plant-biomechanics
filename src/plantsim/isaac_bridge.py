"""Drop-in bridge between the plant solver and an NVIDIA Isaac Sim stage.

Designed to be added to somebody else's scene in three lines::

    bridge = PlantCanopyBridge(SimConfig(), n_plants=64)
    bridge.build()                 # authors USD under /World/Canopy
    bridge.attach(world)           # registers the physics callback

After that the canopy simulates itself. Feed it a manipulator pose each frame
with :meth:`set_effector` and read the contact reaction with
:meth:`contact_force`.

Isaac Sim is imported lazily inside the methods that need it, never at module
import time. That is deliberate: it keeps this file importable by the test
suite and by plain-Python scripts, and it respects Kit's hard requirement that
``SimulationApp`` be constructed before any ``omni``/``isaacsim`` module is
imported.

Compatibility
-------------
Targets the Isaac Sim 5.x/6.x module layout (``isaacsim.core.api``). The older
``omni.isaac.core`` namespace was removed in Isaac Sim 5 and is not supported.
Verified against Isaac Sim 6.0.1.
"""

from __future__ import annotations

import numpy as np

from .config import SimConfig
from .solver import PlantSolver
from .topology import build_scene

DEFAULT_ROOT = "/World/Canopy"
STEM_COLOUR = (0.11, 0.62, 0.46)
LEAF_COLOUR = (0.25, 0.75, 0.50)


class PlantCanopyBridge:
    """A field of deformable plants, driven by the Warp solver, drawn in USD.

    Plants are authored as ``BasisCurves`` -- one curve for the stem and one
    per leaf -- rather than as a prim per particle. A 500-plant canopy is
    12,500 particles, and 12,500 USD prims would make the stage update, not
    the physics, the bottleneck.
    """

    def __init__(self, cfg: SimConfig | None = None, n_plants: int = 32,
                 root_path: str = DEFAULT_ROOT, device: str | None = None,
                 solver: PlantSolver | None = None):
        self.cfg = cfg if cfg is not None else SimConfig()
        if solver is not None:
            self.solver = solver
        else:
            scene = build_scene(self.cfg, n_plants=n_plants)
            self.solver = PlantSolver(self.cfg, scene, device=device)
        self.scene = self.solver.scene
        self.root_path = root_path

        self._curves = []          # (plant index, node ids, points attribute)
        self._stage = None
        self._recording = False
        self._frames_recorded = 0
        self._use_numpy_vt = None  # probed once: Vt.Vec3fArray.FromNumpy support

    # ------------------------------------------------------------------
    @property
    def n_plants(self) -> int:
        return self.scene.n_plants

    @property
    def n_particles(self) -> int:
        return self.scene.n_nodes

    def summary(self) -> str:
        return f"{self.scene.summary()} | device {self.solver.device}"

    # ------------------------------------------------------------------
    def build(self, stage=None):
        """Author the canopy into a USD stage.

        Pass a stage explicitly, or leave it to pick up the current Isaac Sim
        stage. Safe to call once per session.
        """
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, Vt

        if stage is None:
            stage = omni.usd.get_context().get_stage()
        self._stage = stage

        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, Sdf.Path(self.root_path))

        template = self.scene.template
        positions = self.solver.positions()
        self._curves = []

        for plant in range(self.n_plants):
            plant_path = f"{self.root_path}/Plant_{plant:04d}"
            UsdGeom.Xform.Define(stage, Sdf.Path(plant_path))
            local = positions[self.scene.plant_slice(plant)]

            strands = [("Stem", template.stem_path, STEM_COLOUR)]
            strands += [(f"Leaf_{i:02d}", path, LEAF_COLOUR)
                        for i, path in enumerate(template.leaf_paths)]

            for name, node_ids, colour in strands:
                node_ids = np.asarray(node_ids)
                curve = UsdGeom.BasisCurves.Define(
                    stage, Sdf.Path(f"{plant_path}/{name}")
                )
                curve.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
                curve.CreateCurveVertexCountsAttr().Set(
                    Vt.IntArray([len(node_ids)])
                )
                curve.CreatePointsAttr().Set(
                    Vt.Vec3fArray([Gf.Vec3f(*map(float, p))
                                   for p in local[node_ids]])
                )
                # Per-vertex widths taper the stem the way the physics does.
                widths = 2.0 * template.radius[node_ids]
                curve.CreateWidthsAttr().Set(
                    Vt.FloatArray([float(w) for w in widths])
                )
                curve.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
                curve.CreateDisplayColorAttr().Set(
                    Vt.Vec3fArray([Gf.Vec3f(*colour)])
                )
                self._curves.append((plant, node_ids, curve.GetPointsAttr()))

        return self

    # ------------------------------------------------------------------
    def attach(self, world, callback_name: str = "plant_biomechanics"):
        """Register the solver as a physics callback on an Isaac Sim World."""
        world.add_physics_callback(callback_name, self.on_physics_step)
        return self

    def on_physics_step(self, step_size: float = None):
        """Advance the plants one frame and publish the result to USD.

        Ordering matters. The manipulator pose must already have been supplied
        via :meth:`set_effector` for *this* step, the solver runs, and only
        then are positions pushed -- so what is drawn is what was solved.
        """
        self.solver.step()
        self.publish()

    # ------------------------------------------------------------------
    def publish(self, time_code=None):
        """Copy solver state into the USD curves.

        One host readback per frame serves every plant, because the solver
        keeps the whole canopy in one contiguous array.
        """
        from pxr import Gf, Vt

        positions = self.solver.positions()

        if self._use_numpy_vt is None:
            self._use_numpy_vt = hasattr(Vt.Vec3fArray, "FromNumpy")

        for plant, node_ids, points_attr in self._curves:
            local = positions[self.scene.plant_slice(plant)][node_ids]
            if self._use_numpy_vt:
                value = Vt.Vec3fArray.FromNumpy(
                    np.ascontiguousarray(local, dtype=np.float32)
                )
            else:
                value = Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in local])

            if time_code is None:
                points_attr.Set(value)
            else:
                points_attr.Set(value, time_code)

    # ------------------------------------------------------------------
    def set_effector(self, position, radius: float = 0.06):
        """Tell the solver where the manipulator is this frame.

        Accepts anything array-like of length 3, so an Isaac articulation's
        ``get_world_pose()[0]`` can be passed straight through.
        """
        position = np.asarray(position, dtype=np.float32).reshape(1, 3)
        self.solver.set_robot_spheres(position, np.array([radius], np.float32))
        return self

    def set_effectors(self, positions, radii):
        """Multiple collision spheres, for a gripper with several links."""
        self.solver.set_robot_spheres(np.asarray(positions), np.asarray(radii))
        return self

    def contact_force(self) -> np.ndarray:
        """Reaction force on each collision sphere, in newtons."""
        return self.solver.reaction_forces()

    def contact_force_magnitude(self) -> float:
        force = self.contact_force()
        return float(np.linalg.norm(force[0])) if len(force) else 0.0

    # ------------------------------------------------------------------
    def start_recording(self, fps: float = 60.0):
        """Begin writing per-frame time samples into the stage.

        This is what makes the integration useful on a machine whose GPU
        driver is down: the physics runs headless on the CPU, the motion is
        baked into the USD as time samples, and the file can be rendered or
        opened in usdview later on a working machine.
        """
        from pxr import Usd

        self._recording = True
        self._frames_recorded = 0
        self._fps = fps
        self._stage.SetStartTimeCode(0.0)
        self._stage.SetTimeCodesPerSecond(fps)
        self._stage.SetFramesPerSecond(fps)
        return self

    def record_frame(self):
        """Bake the current state as one time sample."""
        if not self._recording:
            raise RuntimeError("call start_recording() first")
        self.publish(time_code=float(self._frames_recorded))
        self._frames_recorded += 1
        self._stage.SetEndTimeCode(float(self._frames_recorded))

    def export_usd(self, path) -> str:
        """Write the stage, including any recorded animation, to disk."""
        path = str(path)
        self._stage.GetRootLayer().Export(path)
        return path

    # ------------------------------------------------------------------
    def step_and_record(self, effector_pose=None, effector_radius: float = 0.06):
        """Convenience: optional contact, one solver step, one time sample."""
        if effector_pose is not None:
            self.set_effector(effector_pose, effector_radius)
        self.solver.step()
        self.record_frame()
