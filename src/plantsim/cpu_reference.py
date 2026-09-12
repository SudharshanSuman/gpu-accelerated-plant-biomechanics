"""NumPy reference solver -- the CPU baseline for the speed-up claim.

This implements exactly the same XPBD algorithm as `kernels.py`: same
constraints, same compliance, same colour groups, same substep schedule. It
serves two purposes.

1.  Correctness. Two independent implementations agreeing to within float
    tolerance is real evidence the GPU kernels are right. A GPU-only result
    can only be checked against intuition.

2.  A *fair* baseline. The comparison is vectorised NumPy, not a Python
    for-loop over particles. Beating a deliberately slow baseline proves
    nothing, and a reviewer will assume the worst if the baseline is not
    stated. Within a colour no two constraints share a particle, so the
    scatter writes have unique indices and plain fancy indexing is safe --
    which is what lets the CPU path vectorise at all.
"""

from __future__ import annotations

import numpy as np

from .config import SimConfig
from .topology import Scene, build_scene

EPS = 1e-9


def wind_velocity(cfg: SimConfig, p: np.ndarray, t: float) -> np.ndarray:
    w = cfg.wind
    d = np.array(w.direction, dtype=np.float32)
    d /= max(np.linalg.norm(d), 1e-12)
    lateral = np.cross(d, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    n = np.linalg.norm(lateral)
    lateral = lateral / n if n > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float32)

    height = np.maximum(p[:, 1], 0.02)
    shear = (height / w.reference_height) ** w.shear_exponent
    k = 2.0 * np.pi / max(w.turbulence_wavelength, 1e-6)
    phase = k * (p[:, 0] + 0.7 * p[:, 2])

    gust = 1.0 + w.gust_amplitude * np.sin(2 * np.pi * w.gust_frequency * t - 0.3 * phase)
    buffet = w.turbulence_amplitude * np.sin(
        2 * np.pi * w.turbulence_frequency * t - phase
    )
    along = w.speed * shear * (gust + buffet)
    across = w.speed * shear * 0.35 * w.turbulence_amplitude * np.sin(
        2 * np.pi * 0.61 * w.turbulence_frequency * t - 1.7 * phase
    )
    return along[:, None] * d[None, :] + across[:, None] * lateral[None, :]


class CPUPlantSolver:
    """Vectorised NumPy XPBD solver, algorithmically identical to the GPU path."""

    def __init__(self, cfg: SimConfig, scene: Scene | None = None):
        self.cfg = cfg
        self.scene = scene if scene is not None else build_scene(cfg)
        s = self.scene

        self.x = s.positions.astype(np.float32).copy()
        self.x_prev = np.zeros_like(self.x)
        self.x_rest = self.x.copy()
        self.v = np.zeros_like(self.x)
        self.inv_mass = s.inv_mass.astype(np.float32)
        self.drag_area = s.drag_area.astype(np.float32)
        self.drag_cd = s.drag_cd.astype(np.float32)

        self.s_a = s.stretch_nodes[:, 0]
        self.s_b = s.stretch_nodes[:, 1]
        self.s_rest = s.stretch_rest.astype(np.float32)
        self.s_alpha = s.stretch_compliance.astype(np.float32)
        self.s_colours = s.stretch_colours

        self.b_a = s.bend_nodes[:, 0]
        self.b_b = s.bend_nodes[:, 1]
        self.b_c = s.bend_nodes[:, 2]
        self.b_rest = s.bend_rest.astype(np.float32)
        self.b_alpha = s.bend_compliance.astype(np.float32)
        self.b_colours = s.bend_colours

        tpl = s.template
        self.nodes_per_plant = tpl.n_nodes
        self.n_s = len(tpl.stretch_rest)
        self.n_b = len(tpl.bend_rest)
        self.s_a_local = tpl.stretch_nodes[:, 0]
        self.s_b_local = tpl.stretch_nodes[:, 1]
        self.b_a_local = tpl.bend_nodes[:, 0]
        self.b_b_local = tpl.bend_nodes[:, 1]
        self.b_c_local = tpl.bend_nodes[:, 2]
        self.plant_base = np.arange(s.n_plants, dtype=np.int64) * tpl.n_nodes
        self.s_lambda_base = np.arange(s.n_plants, dtype=np.int64) * self.n_s
        self.b_lambda_base = np.arange(s.n_plants, dtype=np.int64) * self.n_b

        self.time = 0.0
        self.frame = 0

    # ------------------------------------------------------------------
    def _integrate(self, h: float) -> None:
        cfg = self.cfg
        self.x_prev[:] = self.x
        free = self.inv_mass > 0.0

        u_rel = wind_velocity(cfg, self.x, self.time) - self.v
        speed = np.linalg.norm(u_rel, axis=1)
        coef = 0.5 * cfg.wind.air_density * self.drag_cd * self.drag_area * speed
        force = u_rel * coef[:, None]

        acc = np.array(cfg.gravity, dtype=np.float32)[None, :] + force * self.inv_mass[:, None]
        vel = self.v + acc * h
        vel *= np.exp(-cfg.solver.velocity_damping * h)

        s = np.linalg.norm(vel, axis=1)
        over = s > cfg.solver.max_velocity
        if np.any(over):
            vel[over] *= (cfg.solver.max_velocity / s[over])[:, None]

        self.v = np.where(free[:, None], vel, 0.0).astype(np.float32)
        self.x = np.where(free[:, None], self.x + self.v * h, self.x).astype(np.float32)

    def _solve_stretch(self, inv_h2: float) -> None:
        for cidx in self.s_colours:
            a, b = self.s_a[cidx], self.s_b[cidx]
            wa, wb = self.inv_mass[a], self.inv_mass[b]
            w_sum = wa + wb

            d = self.x[b] - self.x[a]
            dist = np.linalg.norm(d, axis=1)
            ok = (dist > EPS) & (w_sum > 0.0)
            if not np.any(ok):
                continue
            n = np.zeros_like(d)
            n[ok] = d[ok] / dist[ok, None]

            c_val = dist - self.s_rest[cidx]
            at = self.s_alpha[cidx] * inv_h2
            # lambda starts at zero every substep, so the first-iteration
            # update reduces to -C/(w_sum + at); accumulating it across the
            # loop is what makes this XPBD rather than PBD.
            dlam = np.where(ok, -(c_val + at * self.s_lambda[cidx]) / (w_sum + at), 0.0)
            self.s_lambda[cidx] += dlam

            self.x[a] -= n * (wa * dlam)[:, None]
            self.x[b] += n * (wb * dlam)[:, None]

    def _solve_bend(self, inv_h2: float) -> None:
        for cidx in self.b_colours:
            a, b, c = self.b_a[cidx], self.b_b[cidx], self.b_c[cidx]
            wa, wb, wc = self.inv_mass[a], self.inv_mass[b], self.inv_mass[c]
            w_sum = wb + 0.25 * wa + 0.25 * wc

            d = self.x[b] - 0.5 * (self.x[a] + self.x[c])
            dist = np.linalg.norm(d, axis=1)
            ok = (dist > EPS) & (w_sum > 0.0)
            if not np.any(ok):
                continue
            n = np.zeros_like(d)
            n[ok] = d[ok] / dist[ok, None]

            c_val = dist - self.b_rest[cidx]
            at = self.b_alpha[cidx] * inv_h2
            dlam = np.where(ok, -(c_val + at * self.b_lambda[cidx]) / (w_sum + at), 0.0)
            self.b_lambda[cidx] += dlam

            self.x[b] += n * (wb * dlam)[:, None]
            self.x[a] -= n * (0.5 * wa * dlam)[:, None]
            self.x[c] -= n * (0.5 * wc * dlam)[:, None]

    def _solve_chain(self, inv_h2: float) -> None:
        """Sequential sweep along each chain, vectorised across plants.

        This mirrors `solve_plant_chain` exactly: the constraint order within
        a plant is sequential (so corrections propagate along the stem), while
        every plant advances through that order together. The GPU does the
        same thing with one thread per plant.
        """
        base = self.plant_base
        total = self.n_s + self.n_b

        for _ in range(self.cfg.solver.iterations):
            for pass_id in (0, 1):
                for k in range(total):
                    j = k if pass_id == 0 else total - 1 - k

                    if j < self.n_s:
                        c = j
                        a = base + self.s_a_local[c]
                        b = base + self.s_b_local[c]
                        wa, wb = self.inv_mass[a], self.inv_mass[b]
                        w_sum = wa + wb
                        d = self.x[b] - self.x[a]
                        dist = np.linalg.norm(d, axis=1)
                        ok = (dist > EPS) & (w_sum > 0.0)
                        if not np.any(ok):
                            continue
                        n = np.zeros_like(d)
                        n[ok] = d[ok] / dist[ok, None]
                        li = self.s_lambda_base + c
                        at = self.s_alpha[li] * inv_h2
                        dlam = np.where(
                            ok,
                            -(dist - self.s_rest[li] + at * self.s_lambda[li]) / (w_sum + at),
                            0.0,
                        )
                        self.s_lambda[li] += dlam
                        self.x[a] -= n * (wa * dlam)[:, None]
                        self.x[b] += n * (wb * dlam)[:, None]
                    else:
                        c = j - self.n_s
                        a = base + self.b_a_local[c]
                        b = base + self.b_b_local[c]
                        cc = base + self.b_c_local[c]
                        wa, wb, wc = self.inv_mass[a], self.inv_mass[b], self.inv_mass[cc]
                        w_sum = wb + 0.25 * wa + 0.25 * wc
                        d = self.x[b] - 0.5 * (self.x[a] + self.x[cc])
                        dist = np.linalg.norm(d, axis=1)
                        ok = (dist > EPS) & (w_sum > 0.0)
                        if not np.any(ok):
                            continue
                        n = np.zeros_like(d)
                        n[ok] = d[ok] / dist[ok, None]
                        li = self.b_lambda_base + c
                        at = self.b_alpha[li] * inv_h2
                        dlam = np.where(
                            ok,
                            -(dist - self.b_rest[li] + at * self.b_lambda[li]) / (w_sum + at),
                            0.0,
                        )
                        self.b_lambda[li] += dlam
                        self.x[b] += n * (wb * dlam)[:, None]
                        self.x[a] -= n * (0.5 * wa * dlam)[:, None]
                        self.x[cc] -= n * (0.5 * wc * dlam)[:, None]

    def _ground(self) -> None:
        cfg = self.cfg.solver
        below = (self.x[:, 1] < cfg.ground_height) & (self.inv_mass > 0.0)
        if not np.any(below):
            return
        dx = self.x[below, 0] - self.x_prev[below, 0]
        dz = self.x[below, 2] - self.x_prev[below, 2]
        self.x[below, 0] -= dx * cfg.ground_friction
        self.x[below, 2] -= dz * cfg.ground_friction
        self.x[below, 1] = cfg.ground_height

    def step(self) -> None:
        cfg = self.cfg.solver
        h = cfg.dt / cfg.substeps
        inv_h2 = 1.0 / (h * h)

        for _ in range(cfg.substeps):
            self.s_lambda = np.zeros(len(self.s_rest), dtype=np.float32)
            self.b_lambda = np.zeros(len(self.b_rest), dtype=np.float32)

            self._integrate(h)
            if cfg.mode == "chain":
                self._solve_chain(inv_h2)
            elif cfg.mode == "coloured":
                for _ in range(cfg.iterations):
                    self._solve_stretch(inv_h2)
                    self._solve_bend(inv_h2)
            else:
                raise ValueError(f"unknown solver mode {cfg.mode!r}")
            self._ground()
            self.v = ((self.x - self.x_prev) / h).astype(np.float32)
            self.time += h
        self.frame += 1

    def run(self, n_frames: int) -> None:
        for _ in range(n_frames):
            self.step()

    def positions(self) -> np.ndarray:
        return self.x

    def plant_positions(self, p: int = 0) -> np.ndarray:
        return self.x[self.scene.plant_slice(p)]

    def tip_deflection(self) -> np.ndarray:
        tpl = self.scene.template
        idx = np.array(
            [p * tpl.n_nodes + tpl.stem_path[-1] for p in range(self.scene.n_plants)]
        )
        d = self.x[idx] - self.x_rest[idx]
        return np.sqrt(d[:, 0] ** 2 + d[:, 2] ** 2)
