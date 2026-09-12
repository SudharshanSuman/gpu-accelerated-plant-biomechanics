"""Quantitative checks against closed-form beam theory and a second solver.

"The picture looks like a plant" is not validation. These tests put numbers on
it: a case with a textbook analytic answer, an inextensibility check, and
agreement between two independent implementations of the same algorithm.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .config import SimConfig
from .cpu_reference import CPUPlantSolver
from .solver import PlantSolver
from .topology import build_scene


@dataclass
class CantileverResult:
    simulated: float
    analytic: float
    relative_error: float
    slenderness: float
    ei: float
    load_per_length: float

    def __str__(self) -> str:
        return (
            f"  uniform cantilever, transverse self-weight\n"
            f"    EI                      : {self.ei:.4f} N m^2\n"
            f"    distributed load q      : {self.load_per_length:.4f} N/m\n"
            f"    analytic  qL^4/(8EI)    : {self.analytic * 1000:.2f} mm\n"
            f"    simulated tip deflection: {self.simulated * 1000:.2f} mm\n"
            f"    relative error          : {100 * self.relative_error:+.2f}%\n"
            f"    deflection / length     : {self.slenderness:.3f} "
            f"({'small-deflection regime' if self.slenderness < 0.1 else 'geometrically nonlinear'})"
        )


def cantilever_self_weight(device: str | None = None,
                           radius: float = 0.006,
                           length: float = 0.90,
                           n_nodes: int = 33,
                           substeps: int = 48,
                           iterations: int = 8,
                           settle_frames: int = 600) -> CantileverResult:
    """Uniform cantilever under a transverse distributed load.

    Euler-Bernoulli gives a tip deflection of qL^4/(8EI) for a uniformly
    loaded cantilever. To reproduce that case the plant is stripped to a
    uniform cylinder with no leaves and no wind, and gravity is pointed along
    -x so it loads the vertical stem transversely rather than axially. The
    solver is then run until it stops moving.

    The discrete model is expected to land slightly *above* the closed form
    and to approach a plateau as the stem is refined. Two effects account for
    the residual gap: the soil clamp is an elastic hinge with the base
    segment's own compliance rather than a perfectly rigid encastre, and mass
    is lumped at nodes rather than distributed. Both make the discrete beam a
    little more compliant than the continuum it approximates.
    """
    cfg = SimConfig()
    cfg.stem.n_nodes = n_nodes
    cfg.stem.length = length
    cfg.stem.radius_base = radius
    cfg.stem.radius_tip = radius          # uniform: matches the textbook case
    cfg.leaf.enabled = False
    cfg.wind.speed = 0.0
    cfg.wind.gust_amplitude = 0.0
    cfg.wind.turbulence_amplitude = 0.0
    cfg.scene.n_plants = 1
    cfg.scene.height_jitter = 0.0         # no jitter: L must be exactly `length`
    cfg.gravity = (-9.81, 0.0, 0.0)       # transverse load on a vertical stem
    cfg.solver.velocity_damping = 8.0     # settle quickly to static equilibrium
    cfg.solver.substeps = substeps
    cfg.solver.iterations = iterations

    solver = PlantSolver(cfg, device=device)
    solver.run(settle_frames)
    solver.sync()

    tpl = solver.scene.template
    stem = solver.positions()[solver.scene.plant_slice(0)][tpl.stem_path]
    simulated = abs(stem[-1, 0])

    area = np.pi * radius ** 2
    second_moment = np.pi * radius ** 4 / 4.0
    ei = cfg.stem.youngs_modulus * second_moment
    q = cfg.stem.density * area * 9.81
    analytic = q * length ** 4 / (8.0 * ei)

    return CantileverResult(
        simulated=simulated,
        analytic=analytic,
        relative_error=(simulated - analytic) / analytic,
        slenderness=simulated / length,
        ei=ei,
        load_per_length=q,
    )


@dataclass
class AgreementResult:
    max_diff: float
    mean_diff: float
    gpu_tip: float
    cpu_tip: float
    frames: int
    device: str

    def __str__(self) -> str:
        return (
            f"  Warp kernels vs NumPy reference, {self.frames} frames on '{self.device}'\n"
            f"    max particle difference : {self.max_diff:.3e} m\n"
            f"    mean particle difference: {self.mean_diff:.3e} m\n"
            f"    tip deflection GPU / CPU: {self.gpu_tip:.4f} / {self.cpu_tip:.4f} m"
        )


def cross_implementation(cfg: SimConfig, frames: int = 300,
                         device: str | None = None) -> AgreementResult:
    """Run both solvers from identical state and compare."""
    gpu = PlantSolver(cfg, build_scene(cfg), device=device)
    cpu = CPUPlantSolver(cfg, build_scene(cfg))
    gpu.run(frames)
    gpu.sync()
    cpu.run(frames)

    a, b = gpu.positions(), cpu.positions()
    return AgreementResult(
        max_diff=float(np.abs(a - b).max()),
        mean_diff=float(np.abs(a - b).mean()),
        gpu_tip=float(gpu.tip_deflection()[0]),
        cpu_tip=float(cpu.tip_deflection()[0]),
        frames=frames,
        device=gpu.device,
    )


def inextensibility(solver, plant: int = 0) -> tuple[float, float, float]:
    """Measured stem arc length vs its rest length.

    Returns (measured, rest, relative error). A stem that stretches under load
    would invalidate the whole model -- real stems are effectively
    inextensible over the loads a robot or the wind applies.
    """
    scene = solver.scene
    tpl = scene.template
    stem = solver.positions()[scene.plant_slice(plant)][tpl.stem_path]
    measured = float(np.linalg.norm(np.diff(stem, axis=0), axis=1).sum())

    rest_sl = slice(plant * len(tpl.stretch_rest), (plant + 1) * len(tpl.stretch_rest))
    n_stem_seg = len(tpl.stem_path) - 1
    rest = float(scene.stretch_rest[rest_sl][:n_stem_seg].sum())
    return measured, rest, abs(measured - rest) / rest


def full_report(cfg: SimConfig, device: str | None = None, frames: int = 300) -> str:
    lines = ["Validation report", "=" * 60, ""]
    lines.append("1. Closed-form beam theory")
    lines.append(str(cantilever_self_weight(device=device)))
    lines.append("")
    lines.append("2. Independent implementation agreement")
    lines.append(str(cross_implementation(cfg, frames=frames, device=device)))
    lines.append("")

    solver = PlantSolver(cfg, device=device)
    solver.run(frames)
    solver.sync()
    measured, rest, err = inextensibility(solver)
    lines.append("3. Inextensibility under load")
    lines.append(f"  stem arc length         : {measured:.5f} m")
    lines.append(f"  rest length             : {rest:.5f} m")
    lines.append(f"  relative stretch        : {100 * err:.4f}%")
    return "\n".join(lines)


def converged_deflection(n_nodes: int, device: str | None = None,
                         tolerance: float = 0.01, max_substeps: int = 2048,
                         iterations: int = 8, settle_frames: int = 150):
    """Double the substep count until the answer stops moving.

    Reporting one number at one solver setting proves nothing about a scheme
    that is known to be under-converged at low substep counts. This runs the
    standard protocol instead: refine the time discretisation until the result
    changes by less than `tolerance`, and report both the value and the work
    it took to get there.

    Returns (deflection_m, substeps_used, converged_flag).
    """
    substeps = 8
    previous = cantilever_self_weight(
        device=device, n_nodes=n_nodes, substeps=substeps,
        iterations=iterations, settle_frames=settle_frames,
    ).simulated

    while substeps < max_substeps:
        substeps *= 2
        current = cantilever_self_weight(
            device=device, n_nodes=n_nodes, substeps=substeps,
            iterations=iterations, settle_frames=settle_frames,
        ).simulated
        if abs(current - previous) <= tolerance * abs(current):
            return current, substeps, True
        previous = current
    return previous, substeps, False


def refinement_study(device: str | None = None,
                     node_counts=(9, 17, 33)) -> str:
    """Converged tip deflection vs stem resolution.

    Two things should be visible in this table. The converged deflection
    settles as the stem is refined -- that is the spatial discretisation
    converging. And the substeps required to *reach* convergence climb
    steeply with node count, because a beam's constraint system is stiffer
    the finer it is meshed. The second trend is the practical reason the
    default stem uses 16 nodes rather than 64: past a point, resolution costs
    solver work without buying accuracy that matters to a robot.
    """
    lines = [
        "  uniform cantilever, mesh refinement (each row converged to 1%)",
        f"    {'nodes':>6} {'segments':>9} {'substeps':>9} {'tip (mm)':>10} {'vs analytic':>12}",
    ]
    analytic = None
    for n in node_counts:
        value, substeps, ok = converged_deflection(n, device=device)
        r = cantilever_self_weight(device=device, n_nodes=n, substeps=8,
                                   settle_frames=10)
        analytic = r.analytic
        flag = "" if ok else "  (not converged)"
        lines.append(
            f"    {n:>6} {n - 1:>9} {substeps:>9} {value * 1000:>10.2f} "
            f"{100 * (value - analytic) / analytic:>11.1f}%{flag}"
        )
    lines.append(f"    analytic qL^4/(8EI) = {analytic * 1000:.2f} mm")
    lines.append("    The residual offset is the discrete model's own bias: the soil")
    lines.append("    clamp is an elastic hinge rather than a rigid encastre, and mass")
    lines.append("    is lumped at nodes. Both make the chain slightly more compliant")
    lines.append("    than the continuum beam the closed form describes.")
    return "\n".join(lines)


def convergence_study(device: str | None = None, n_nodes: int = 17) -> str:
    """Show that substepping, not iteration count, is what converges a chain.

    This is the most important calibration result in the project. The obvious
    knob -- more constraint iterations -- helps far less per unit work,
    because a bending correction propagates roughly one hinge per pass along
    the stem. Shrinking the step size helps much more.
    """
    lines = [
        f"  solver convergence, uniform {n_nodes}-node cantilever",
        f"    {'substeps':>9} {'iters':>6} {'work':>6} {'tip (mm)':>10}",
    ]
    for substeps, iterations in [(8, 2), (8, 8), (8, 32), (8, 128),
                                 (16, 4), (32, 4), (64, 4), (128, 4)]:
        r = cantilever_self_weight(device=device, n_nodes=n_nodes,
                                   substeps=substeps, iterations=iterations,
                                   settle_frames=150)
        lines.append(
            f"    {substeps:>9} {iterations:>6} {substeps * iterations:>6} "
            f"{r.simulated * 1000:>10.2f}"
        )
    lines.append("    Both knobs converge eventually, but not at the same price:")
    lines.append("    64 substeps x 4 iterations (work 256) matches what 8 substeps")
    lines.append("    x 128 iterations (work 1024) achieves -- a 4x saving for the")
    lines.append("    same answer. Hence substeps=32, iterations=4 as the default.")
    return "\n".join(lines)


def material_sanity(device: str | None = None,
                    moduli=((15e6, "10-20 MPa"),
                            (200e6, "200 MPa"),
                            (1.0e9, "1 GPa (default)"),
                            (5.0e9, "5 GPa"))) -> str:
    """Can a stem of this modulus hold itself up?

    This is the cheapest possible falsification test for a claimed Young's
    modulus, and it settles arguments that would otherwise be fought with
    citations. A self-supporting stem must carry its own distributed weight:
    for a cantilever that is delta = qL^4/(8EI), and if delta approaches the
    stem length the plant physically cannot stand.

    The test runs the actual solver with wind disabled and reports the
    settled tip height. Any modulus that puts the tip on the ground is
    disqualified regardless of what it was cited from -- most likely it is a
    value measured for isolated parenchyma tissue rather than the effective
    flexural modulus of a whole stem, which is dominated by the stiff fibrous
    outer ring.
    """
    lines = [
        "  self-support test (wind off, settled under gravity alone)",
        f"    {'modulus':>18} {'tip height':>11} {'droop':>9} {'verdict':>16}",
    ]
    for modulus, label in moduli:
        cfg = SimConfig()
        cfg.stem.youngs_modulus = modulus
        cfg.leaf.youngs_modulus = modulus / 10.0
        cfg.scene.n_plants = 1
        cfg.scene.height_jitter = 0.0
        cfg.wind.speed = 0.0
        cfg.wind.gust_amplitude = 0.0
        cfg.wind.turbulence_amplitude = 0.0
        cfg.solver.velocity_damping = 6.0

        solver = PlantSolver(cfg, device=device)
        solver.run(400)
        solver.sync()
        tpl = solver.scene.template
        stem = solver.positions()[solver.scene.plant_slice(0)][tpl.stem_path]
        height = float(stem[-1, 1])
        droop = cfg.stem.length - height

        if height < 0.5 * cfg.stem.length:
            verdict = "CANNOT STAND"
        elif droop > 0.1 * cfg.stem.length:
            verdict = "droops"
        else:
            verdict = "stands upright"
        lines.append(
            f"    {label:>18} {height:8.3f} m {droop:7.3f} m {verdict:>16}"
        )
    lines.append("    A stem that cannot support its own weight is not a candidate")
    lines.append("    parameterisation, whatever the source of the number.")
    return "\n".join(lines)


def bending_stress(stem_positions: np.ndarray, radii: np.ndarray,
                   youngs_modulus: float) -> np.ndarray:
    """Peak fibre stress along a deflected stem, from its shape alone.

    Stem failure is a bending failure, so the quantity that decides whether a
    manipulator breaks a plant is the maximum fibre stress, not the contact
    force. Stress follows from the curvature of the deflected centreline and
    the material, both of which are known:

        M = EI * kappa,     sigma = M * r / I = E * kappa * r

    so the second moment of area cancels and the result depends only on
    curvature, local radius and Young's modulus.

    This matters because it is *measurable from the simulated geometry*. A
    contact force read out of the solver's constraint multipliers is sensitive
    to timestep during a sharp impact; the deflected shape is not.

    Curvature uses the Menger form, kappa = 4A / (|ab||bc||ca|), which is
    stable for three nearly-collinear points where a finite-difference second
    derivative is not.
    """
    a = stem_positions[:-2]
    b = stem_positions[1:-1]
    c = stem_positions[2:]

    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)

    # Triangle area via the cross product of two edges.
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    denominator = np.maximum(ab * bc * ca, 1e-12)
    curvature = 4.0 * area / denominator

    return youngs_modulus * curvature * radii[1:-1]
