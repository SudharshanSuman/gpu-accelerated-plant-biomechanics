"""GPU XPBD plant solver -- the core of the project.

One frame is:

    for each substep:
        reset Lagrange multipliers
        integrate (gravity + aerodynamic drag + viscous damping)
        for each iteration:
            project stretch constraints, one colour at a time
            project bending constraints, one colour at a time
        resolve ground and robot contacts
        recover velocities from corrected positions

Every array stays on the device for the whole run. The only host transfers are
the ones the caller explicitly asks for via `positions()`.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from . import kernels as K
from .config import SimConfig
from .device import ensure_init, resolve_device
from .topology import Scene, build_scene


def _wind_struct(cfg: SimConfig, device: str) -> K.WindField:
    w = cfg.wind
    d = np.array(w.direction, dtype=np.float64)
    d /= max(np.linalg.norm(d), 1e-12)
    lateral = np.cross(d, np.array([0.0, 1.0, 0.0]))
    n = np.linalg.norm(lateral)
    lateral = lateral / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])

    wind = K.WindField()
    wind.direction = wp.vec3(*d.astype(np.float32))
    wind.lateral = wp.vec3(*lateral.astype(np.float32))
    wind.speed = float(w.speed)
    wind.reference_height = float(w.reference_height)
    wind.shear_exponent = float(w.shear_exponent)
    wind.gust_amplitude = float(w.gust_amplitude)
    wind.gust_frequency = float(w.gust_frequency)
    wind.turbulence_amplitude = float(w.turbulence_amplitude)
    wind.turbulence_frequency = float(w.turbulence_frequency)
    wind.turbulence_wavenumber = float(2.0 * np.pi / max(w.turbulence_wavelength, 1e-6))
    wind.air_density = float(w.air_density)
    return wind


class PlantSolver:
    def __init__(
        self,
        cfg: SimConfig,
        scene: Scene | None = None,
        device: str | None = None,
        use_graph: bool = True,
    ):
        ensure_init()
        self.cfg = cfg
        self.device = resolve_device(device if device is not None else cfg.device)
        self.scene = scene if scene is not None else build_scene(cfg)
        s = self.scene

        f32 = np.float32
        dev = self.device

        self.x = wp.array(s.positions.astype(f32), dtype=wp.vec3, device=dev)
        self.x_prev = wp.zeros(s.n_nodes, dtype=wp.vec3, device=dev)
        self.x_rest = wp.array(s.positions.astype(f32), dtype=wp.vec3, device=dev)
        self.v = wp.zeros(s.n_nodes, dtype=wp.vec3, device=dev)
        self.inv_mass = wp.array(s.inv_mass.astype(f32), dtype=float, device=dev)
        self.drag_area = wp.array(s.drag_area.astype(f32), dtype=float, device=dev)
        self.drag_cd = wp.array(s.drag_cd.astype(f32), dtype=float, device=dev)
        self.node_radius = wp.array(s.radius.astype(f32), dtype=float, device=dev)

        self.s_ia = wp.array(s.stretch_nodes[:, 0].copy(), dtype=wp.int32, device=dev)
        self.s_ib = wp.array(s.stretch_nodes[:, 1].copy(), dtype=wp.int32, device=dev)
        self.s_rest = wp.array(s.stretch_rest.astype(f32), dtype=float, device=dev)
        self.s_alpha = wp.array(s.stretch_compliance.astype(f32), dtype=float, device=dev)
        self.s_lambda = wp.zeros(len(s.stretch_rest), dtype=float, device=dev)
        self.s_colours = [wp.array(c, dtype=wp.int32, device=dev) for c in s.stretch_colours]

        self.b_ia = wp.array(s.bend_nodes[:, 0].copy(), dtype=wp.int32, device=dev)
        self.b_ib = wp.array(s.bend_nodes[:, 1].copy(), dtype=wp.int32, device=dev)
        self.b_ic = wp.array(s.bend_nodes[:, 2].copy(), dtype=wp.int32, device=dev)
        self.b_rest = wp.array(s.bend_rest.astype(f32), dtype=float, device=dev)
        self.b_alpha = wp.array(s.bend_compliance.astype(f32), dtype=float, device=dev)
        self.b_lambda = wp.zeros(len(s.bend_rest), dtype=float, device=dev)
        self.b_colours = [wp.array(c, dtype=wp.int32, device=dev) for c in s.bend_colours]

        tpl = s.template
        self.nodes_per_plant = tpl.n_nodes
        self.s_ia_local = wp.array(tpl.stretch_nodes[:, 0].copy(), dtype=wp.int32, device=dev)
        self.s_ib_local = wp.array(tpl.stretch_nodes[:, 1].copy(), dtype=wp.int32, device=dev)
        self.b_ia_local = wp.array(tpl.bend_nodes[:, 0].copy(), dtype=wp.int32, device=dev)
        self.b_ib_local = wp.array(tpl.bend_nodes[:, 1].copy(), dtype=wp.int32, device=dev)
        self.b_ic_local = wp.array(tpl.bend_nodes[:, 2].copy(), dtype=wp.int32, device=dev)

        self.sim_time = wp.zeros(1, dtype=float, device=dev)
        self.wind = _wind_struct(cfg, dev)
        self.gravity = wp.vec3(*np.array(cfg.gravity, dtype=np.float32))

        # Robot collision spheres. Allocated empty; `set_robot_spheres` fills them.
        self.sphere_centres = wp.zeros(0, dtype=wp.vec3, device=dev)
        self.sphere_radii = wp.zeros(0, dtype=float, device=dev)
        self.sphere_reaction = wp.zeros(0, dtype=wp.vec3, device=dev)
        self._n_spheres = 0

        # Per-particle XPBD contact state.
        self.contact_lambda = wp.zeros(s.n_nodes, dtype=float, device=dev)
        self.contact_sphere = wp.zeros(s.n_nodes, dtype=wp.int32, device=dev)
        self.contact_normal = wp.zeros(s.n_nodes, dtype=wp.vec3, device=dev)
        self.contact_compliance = self.cfg.solver.contact_compliance

        tip_idx = np.array(
            [p * s.template.n_nodes + s.template.stem_path[-1] for p in range(s.n_plants)],
            dtype=np.int32,
        )
        self.tip_index = wp.array(tip_idx, dtype=wp.int32, device=dev)
        self.tip_out = wp.zeros(s.n_plants, dtype=float, device=dev)

        self.frame = 0
        self._graph = None
        self._use_graph = use_graph and wp.get_device(dev).is_cuda

        wp.load_module(K, device=dev)

    # ------------------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return self.scene.n_nodes

    def set_robot_spheres(self, centres: np.ndarray, radii: np.ndarray) -> None:
        """Define the manipulator's collision proxies for this frame.

        Changing the sphere *count* invalidates a captured CUDA graph, so the
        graph is dropped and re-captured. Moving existing spheres does not --
        write into `sphere_centres` in place and the graph keeps replaying.
        """
        centres = np.asarray(centres, dtype=np.float32).reshape(-1, 3)
        radii = np.asarray(radii, dtype=np.float32).reshape(-1)
        if len(centres) != self._n_spheres:
            self._graph = None
            self._n_spheres = len(centres)
            self.sphere_centres = wp.array(centres, dtype=wp.vec3, device=self.device)
            self.sphere_radii = wp.array(radii, dtype=float, device=self.device)
            self.sphere_reaction = wp.zeros(len(centres), dtype=wp.vec3, device=self.device)
        else:
            self.sphere_centres.assign(centres)
            self.sphere_radii.assign(radii)

    def reaction_forces(self) -> np.ndarray:
        """Force the plants push back with, per robot sphere, in newtons.

        Recovered from the XPBD contact multipliers as lambda/h^2, summed over
        the particles touching each sphere and averaged over the frame's
        substeps. Unlike a force read from positional corrections, this is
        timestep-independent -- see `solve_contact` for why that matters.
        """
        if self._n_spheres == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return self.sphere_reaction.numpy() / self.cfg.solver.substeps

    # ------------------------------------------------------------------
    def _substep(self, h: float) -> None:
        cfg = self.cfg.solver
        dev = self.device
        inv_h2 = 1.0 / (h * h)

        self.s_lambda.zero_()
        self.b_lambda.zero_()

        wp.launch(
            K.integrate,
            dim=self.n_nodes,
            inputs=[self.x, self.v, self.x_prev, self.inv_mass, self.drag_area,
                    self.drag_cd, self.wind, self.gravity, cfg.velocity_damping,
                    h, self.sim_time, cfg.max_velocity],
            device=dev,
        )

        if cfg.mode == "chain":
            # One launch, one thread per plant, whole chain solved inside it.
            wp.launch(
                K.solve_plant_chain,
                dim=self.scene.n_plants,
                inputs=[self.x, self.inv_mass,
                        self.s_ia_local, self.s_ib_local, self.s_rest, self.s_alpha,
                        self.s_lambda,
                        self.b_ia_local, self.b_ib_local, self.b_ic_local,
                        self.b_rest, self.b_alpha, self.b_lambda,
                        self.nodes_per_plant, cfg.iterations, inv_h2],
                device=dev,
            )
        elif cfg.mode == "coloured":
            for _ in range(cfg.iterations):
                for cidx in self.s_colours:
                    wp.launch(
                        K.solve_stretch,
                        dim=len(cidx),
                        inputs=[cidx, self.s_ia, self.s_ib, self.s_rest, self.s_alpha,
                                self.s_lambda, self.x, self.inv_mass, inv_h2],
                        device=dev,
                    )
                for cidx in self.b_colours:
                    wp.launch(
                        K.solve_bend,
                        dim=len(cidx),
                        inputs=[cidx, self.b_ia, self.b_ib, self.b_ic, self.b_rest,
                                self.b_alpha, self.b_lambda, self.x, self.inv_mass, inv_h2],
                        device=dev,
                    )
        else:
            raise ValueError(f"unknown solver mode {cfg.mode!r}")

        wp.launch(
            K.apply_ground,
            dim=self.n_nodes,
            inputs=[self.x, self.x_prev, self.inv_mass, cfg.ground_height,
                    cfg.ground_friction],
            device=dev,
        )

        if self._n_spheres:
            self.contact_lambda.zero_()
            # Interleave with the elastic solve: contact and bending fight
            # each other, and resolving contact only once per substep leaves
            # the multiplier short of the value that balances the stem.
            for _ in range(cfg.contact_iterations):
                wp.launch(
                    K.solve_contact,
                    dim=self.n_nodes,
                    inputs=[self.x, self.inv_mass, self.node_radius,
                            self.sphere_centres, self.sphere_radii,
                            self.contact_lambda, self.contact_sphere,
                            self.contact_normal, self.contact_compliance, inv_h2],
                    device=dev,
                )
            wp.launch(
                K.accumulate_contact_force,
                dim=self.n_nodes,
                inputs=[self.contact_lambda, self.contact_sphere,
                        self.contact_normal, self.sphere_reaction, inv_h2],
                device=dev,
            )

        wp.launch(
            K.update_velocity,
            dim=self.n_nodes,
            inputs=[self.x, self.x_prev, self.v, 1.0 / h],
            device=dev,
        )
        wp.launch(K.advance_time, dim=1, inputs=[self.sim_time, h], device=dev)

    def _frame_work(self) -> None:
        h = self.cfg.solver.dt / self.cfg.solver.substeps
        # Reaction accumulates across the whole frame, not per substep. By the
        # last substep the plant has already been pushed clear of the sphere,
        # so sampling only that substep reports no contact at all even while
        # the plant is visibly being shoved aside.
        if self._n_spheres:
            self.sphere_reaction.zero_()
        for _ in range(self.cfg.solver.substeps):
            self._substep(h)

    def step(self) -> None:
        """Advance one frame."""
        if self._use_graph:
            if self._graph is None:
                with wp.ScopedCapture(device=self.device) as capture:
                    self._frame_work()
                self._graph = capture.graph
            wp.capture_launch(self._graph)
        else:
            self._frame_work()
        self.frame += 1

    def run(self, n_frames: int) -> None:
        for _ in range(n_frames):
            self.step()

    # ------------------------------------------------------------------
    def positions(self) -> np.ndarray:
        """Host copy of all particle positions, (N, 3)."""
        return self.x.numpy()

    def plant_positions(self, p: int = 0) -> np.ndarray:
        return self.positions()[self.scene.plant_slice(p)]

    def tip_deflection(self) -> np.ndarray:
        """Horizontal displacement of each stem tip from its rest position."""
        wp.launch(
            K.tip_offsets,
            dim=self.scene.n_plants,
            inputs=[self.x, self.x_rest, self.tip_index, self.tip_out],
            device=self.device,
        )
        return self.tip_out.numpy()

    def sync(self) -> None:
        wp.synchronize_device(self.device)

    def reset(self) -> None:
        self.x.assign(self.scene.positions.astype(np.float32))
        self.v.zero_()
        self.sim_time.zero_()
        self.frame = 0
