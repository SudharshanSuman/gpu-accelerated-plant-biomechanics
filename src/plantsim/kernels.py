"""Warp GPU kernels: the entire XPBD solve, resident on the device.

Nothing here copies back to the host. A frame is a fixed sequence of kernel
launches over device-resident arrays, which is what makes the GPU path worth
measuring -- a solver that round-trips positions to NumPy every step is
bounded by PCIe bandwidth, not by arithmetic, and will lose to a pure CPU
implementation at every problem size.
"""

from __future__ import annotations

import warp as wp

EPS = 1.0e-9


@wp.struct
class WindField:
    """Analytic turbulent wind: shear profile x gust envelope x buffeting."""

    direction: wp.vec3
    lateral: wp.vec3
    speed: float
    reference_height: float
    shear_exponent: float
    gust_amplitude: float
    gust_frequency: float
    turbulence_amplitude: float
    turbulence_frequency: float
    turbulence_wavenumber: float
    air_density: float


@wp.func
def wind_velocity(w: WindField, p: wp.vec3, t: float) -> wp.vec3:
    """Air velocity at a point in space and time.

    The shear term makes wind stronger higher up, so the top of the canopy is
    loaded harder than the base -- on top of the r^4 stiffness taper this is
    the second reason real plants bend as an arc rather than a straight lean.
    """
    height = wp.max(p[1], 0.02)
    shear = wp.pow(height / w.reference_height, w.shear_exponent)

    phase = w.turbulence_wavenumber * (p[0] + 0.7 * p[2])
    gust = 1.0 + w.gust_amplitude * wp.sin(
        6.2831853 * w.gust_frequency * t - 0.3 * phase
    )
    buffet = w.turbulence_amplitude * wp.sin(
        6.2831853 * w.turbulence_frequency * t - phase
    )

    along = w.speed * shear * (gust + buffet)
    across = w.speed * shear * 0.35 * w.turbulence_amplitude * wp.sin(
        6.2831853 * 0.61 * w.turbulence_frequency * t - 1.7 * phase
    )
    return w.direction * along + w.lateral * across


@wp.kernel
def integrate(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    x_prev: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    drag_area: wp.array(dtype=float),
    drag_cd: wp.array(dtype=float),
    wind: WindField,
    gravity: wp.vec3,
    damping: float,
    h: float,
    sim_time: wp.array(dtype=float),
    max_velocity: float,
):
    # Simulation time lives in a device array rather than arriving as a kernel
    # argument so the whole frame can be captured once as a CUDA graph and
    # replayed, instead of paying launch overhead for ~50 tiny kernels a frame.
    t = sim_time[0]
    i = wp.tid()
    x_prev[i] = x[i]

    w = inv_mass[i]
    if w == 0.0:
        v[i] = wp.vec3(0.0, 0.0, 0.0)
        return

    # Aerodynamic drag uses air velocity *relative to the moving particle*.
    # As the plant yields to the wind the relative speed drops and so does the
    # force, so the bend settles at equilibrium. A constant wind acceleration
    # has no such feedback and drives the stem flat no matter how it is tuned.
    u_rel = wind_velocity(wind, x[i], t) - v[i]
    speed = wp.length(u_rel)
    force = wp.vec3(0.0, 0.0, 0.0)
    if speed > EPS:
        force = u_rel * (0.5 * wind.air_density * drag_cd[i] * drag_area[i] * speed)

    vel = v[i] + (gravity + force * w) * h

    # Exponential decay is timestep-independent: halving h halves the per-step
    # loss, so the material does not silently change when substeps change.
    vel = vel * wp.exp(-damping * h)

    s = wp.length(vel)
    if s > max_velocity:
        vel = vel * (max_velocity / s)

    v[i] = vel
    x[i] = x[i] + vel * h


@wp.kernel
def solve_stretch(
    cidx: wp.array(dtype=wp.int32),
    ia: wp.array(dtype=wp.int32),
    ib: wp.array(dtype=wp.int32),
    rest: wp.array(dtype=float),
    alpha: wp.array(dtype=float),
    lam: wp.array(dtype=float),
    x: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    inv_h2: float,
):
    """XPBD axial constraint: C = |x_b - x_a| - L."""
    k = wp.tid()
    c = cidx[k]
    a = ia[c]
    b = ib[c]

    wa = inv_mass[a]
    wb = inv_mass[b]
    w_sum = wa + wb
    if w_sum == 0.0:
        return

    d = x[b] - x[a]
    dist = wp.length(d)
    if dist < EPS:
        return
    n = d / dist

    c_val = dist - rest[c]
    alpha_tilde = alpha[c] * inv_h2
    d_lambda = (-c_val - alpha_tilde * lam[c]) / (w_sum + alpha_tilde)
    lam[c] = lam[c] + d_lambda

    x[a] = x[a] - n * (wa * d_lambda)
    x[b] = x[b] + n * (wb * d_lambda)


@wp.kernel
def solve_bend(
    cidx: wp.array(dtype=wp.int32),
    ia: wp.array(dtype=wp.int32),
    ib: wp.array(dtype=wp.int32),
    ic: wp.array(dtype=wp.int32),
    rest: wp.array(dtype=float),
    alpha: wp.array(dtype=float),
    lam: wp.array(dtype=float),
    x: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    inv_h2: float,
):
    """XPBD bending constraint: C = |x_b - (x_a + x_c)/2| - rest.

    The offset of the hinge particle from the midpoint of its neighbours grows
    linearly with the bend angle, which keeps the constraint well conditioned
    near the straight configuration. The commonly used |x_c - x_a| constraint
    is quadratic in the angle there and goes numerically dead on a straight
    stem -- the plant then behaves like a rope until it is already bent.
    """
    k = wp.tid()
    c = cidx[k]
    a = ia[c]
    b = ib[c]
    cc = ic[c]

    wa = inv_mass[a]
    wb = inv_mass[b]
    wc = inv_mass[cc]
    w_sum = wb + 0.25 * wa + 0.25 * wc
    if w_sum == 0.0:
        return

    d = x[b] - (x[a] + x[cc]) * 0.5
    dist = wp.length(d)
    if dist < EPS:
        return
    n = d / dist

    c_val = dist - rest[c]
    alpha_tilde = alpha[c] * inv_h2
    d_lambda = (-c_val - alpha_tilde * lam[c]) / (w_sum + alpha_tilde)
    lam[c] = lam[c] + d_lambda

    x[b] = x[b] + n * (wb * d_lambda)
    x[a] = x[a] - n * (0.5 * wa * d_lambda)
    x[cc] = x[cc] - n * (0.5 * wc * d_lambda)


@wp.kernel
def apply_ground(
    x: wp.array(dtype=wp.vec3),
    x_prev: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    ground_y: float,
    friction: float,
):
    """Non-penetration plus Coulomb-style tangential drag against the soil."""
    i = wp.tid()
    if inv_mass[i] == 0.0:
        return
    p = x[i]
    if p[1] >= ground_y:
        return
    dx = p[0] - x_prev[i][0]
    dz = p[2] - x_prev[i][2]
    x[i] = wp.vec3(p[0] - dx * friction, ground_y, p[2] - dz * friction)


@wp.kernel
def apply_sphere_contact(
    x: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    node_radius: wp.array(dtype=float),
    centres: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=float),
    reaction: wp.array(dtype=wp.vec3),
    stiffness: float,
):
    """Push particles out of robot collision spheres, logging the reaction.

    The accumulated reaction is the force the plant exerts back on the
    manipulator. That number is what an agricultural robot actually needs: it
    tells the controller whether a grasp would snap the stem.
    """
    i = wp.tid()
    wi = inv_mass[i]
    if wi == 0.0:
        return

    p = x[i]
    for s in range(centres.shape[0]):
        d = p - centres[s]
        dist = wp.length(d)
        min_dist = radii[s] + node_radius[i]
        if dist < min_dist and dist > EPS:
            n = d / dist
            correction = n * ((min_dist - dist) * stiffness)
            p = p + correction
            wp.atomic_add(reaction, s, correction * (-1.0 / wp.max(wi, EPS)))
    x[i] = p


@wp.kernel
def update_velocity(
    x: wp.array(dtype=wp.vec3),
    x_prev: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    inv_h: float,
):
    """Recover velocity from the constraint-corrected positions (PBD rule)."""
    i = wp.tid()
    v[i] = (x[i] - x_prev[i]) * inv_h


@wp.kernel
def tip_offsets(
    x: wp.array(dtype=wp.vec3),
    x_rest: wp.array(dtype=wp.vec3),
    tip_index: wp.array(dtype=wp.int32),
    out: wp.array(dtype=float),
):
    """Horizontal deflection of each plant's stem tip, for diagnostics."""
    i = wp.tid()
    t = tip_index[i]
    d = x[t] - x_rest[t]
    out[i] = wp.sqrt(d[0] * d[0] + d[2] * d[2])


@wp.kernel
def advance_time(sim_time: wp.array(dtype=float), h: float):
    sim_time[0] = sim_time[0] + h


@wp.kernel
def solve_plant_chain(
    x: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    s_ia: wp.array(dtype=wp.int32),
    s_ib: wp.array(dtype=wp.int32),
    s_rest: wp.array(dtype=float),
    s_alpha: wp.array(dtype=float),
    s_lambda: wp.array(dtype=float),
    b_ia: wp.array(dtype=wp.int32),
    b_ib: wp.array(dtype=wp.int32),
    b_ic: wp.array(dtype=wp.int32),
    b_rest: wp.array(dtype=float),
    b_alpha: wp.array(dtype=float),
    b_lambda: wp.array(dtype=float),
    nodes_per_plant: int,
    iterations: int,
    inv_h2: float,
):
    """One thread per plant: sequential symmetric Gauss-Seidel along the chain.

    Why not one thread per constraint? A stem is a chain, and a bending
    correction at the tip only reaches the root by travelling hinge to hinge.
    Colour-parallel Jacobi sweeps move information one hinge per pass, so a
    16-node stem needs on the order of 16 passes before it behaves like the
    material it was given. At 2 passes it behaves like a much softer plant --
    measured at 7x too compliant for a 17-node stem, which is precisely the
    "floppy rope" failure this project set out to avoid.

    Sweeping the chain inside a single thread solves the chain almost
    directly, because a forward pass followed by a backward pass over a
    tridiagonal system propagates corrections end to end. Parallelism then
    comes from the number of plants -- which is the right axis anyway: an
    agricultural scene has hundreds or thousands of plants and only a few
    dozen nodes each.

    The index arrays here are plant-local (the template's own numbering) and
    are offset by the plant's base; per-constraint rest lengths and compliance
    are per-plant, since stems vary in size across the field.
    """
    p = wp.tid()
    base = p * nodes_per_plant
    n_s = s_ia.shape[0]
    n_b = b_ia.shape[0]
    s_off = p * n_s
    b_off = p * n_b

    for _it in range(iterations):
        for pass_id in range(2):
            for k in range(n_s + n_b):
                # Forward on pass 0, reverse on pass 1. The symmetric pair is
                # what makes the sweep converge from both ends of the chain.
                if pass_id == 0:
                    j = k
                else:
                    j = n_s + n_b - 1 - k

                if j < n_s:
                    c = j
                    a = base + s_ia[c]
                    b = base + s_ib[c]
                    wa = inv_mass[a]
                    wb = inv_mass[b]
                    w_sum = wa + wb
                    if w_sum > 0.0:
                        d = x[b] - x[a]
                        dist = wp.length(d)
                        if dist > EPS:
                            n = d / dist
                            at = s_alpha[s_off + c] * inv_h2
                            lam = s_lambda[s_off + c]
                            dl = (-(dist - s_rest[s_off + c]) - at * lam) / (w_sum + at)
                            s_lambda[s_off + c] = lam + dl
                            x[a] = x[a] - n * (wa * dl)
                            x[b] = x[b] + n * (wb * dl)
                else:
                    c = j - n_s
                    a = base + b_ia[c]
                    b = base + b_ib[c]
                    cc = base + b_ic[c]
                    wa = inv_mass[a]
                    wb = inv_mass[b]
                    wc = inv_mass[cc]
                    w_sum = wb + 0.25 * wa + 0.25 * wc
                    if w_sum > 0.0:
                        d = x[b] - (x[a] + x[cc]) * 0.5
                        dist = wp.length(d)
                        if dist > EPS:
                            n = d / dist
                            at = b_alpha[b_off + c] * inv_h2
                            lam = b_lambda[b_off + c]
                            dl = (-(dist - b_rest[b_off + c]) - at * lam) / (w_sum + at)
                            b_lambda[b_off + c] = lam + dl
                            x[b] = x[b] + n * (wb * dl)
                            x[a] = x[a] - n * (0.5 * wa * dl)
                            x[cc] = x[cc] - n * (0.5 * wc * dl)


@wp.kernel
def solve_contact(
    x: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    node_radius: wp.array(dtype=float),
    centres: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=float),
    contact_lambda: wp.array(dtype=float),
    contact_sphere: wp.array(dtype=wp.int32),
    contact_normal: wp.array(dtype=wp.vec3),
    compliance: float,
    inv_h2: float,
):
    """XPBD non-penetration constraint against the manipulator's spheres.

    Why this is a constraint with a multiplier rather than a direct positional
    push: the force a contact exerts cannot be recovered from how far the
    solver displaced the particle. That displacement is mostly the solver
    undoing the elastic correction applied moments earlier in the same
    substep, and it barely shrinks as the timestep shrinks -- so a force read
    from it diverges under refinement. Measured on this model, the naive
    estimator grew from 2.4 N to 25 N over a 16x timestep refinement.

    The XPBD multiplier does not have that problem. At convergence
    C + (alpha/h^2) * lambda = 0, so lambda = -C * h^2 / alpha and the force
    lambda/h^2 = -C/alpha is a property of the contact, not of the timestep.

    Contact is one-sided: lambda is clamped at zero so the constraint can push
    a particle out but never pull it in.
    """
    i = wp.tid()
    w = inv_mass[i]
    contact_sphere[i] = -1
    if w == 0.0:
        return

    p = x[i]

    # Deepest penetration wins. A particle squeezed between two spheres is
    # resolved against the one it has violated most, which is the stable
    # choice when the proxies overlap (and they are built to overlap).
    best = int(-1)
    best_depth = float(0.0)
    best_normal = wp.vec3(0.0, 0.0, 0.0)

    for s in range(centres.shape[0]):
        d = p - centres[s]
        dist = wp.length(d)
        min_dist = radii[s] + node_radius[i]
        if dist < min_dist and dist > EPS:
            depth = min_dist - dist
            if depth > best_depth:
                best_depth = depth
                best = s
                best_normal = d / dist

    if best < 0:
        return

    c_val = -best_depth                       # negative while penetrating
    alpha_tilde = compliance * inv_h2
    lam = contact_lambda[i]
    d_lambda = (-c_val - alpha_tilde * lam) / (w + alpha_tilde)

    # One-sided: never let the accumulated multiplier go negative.
    if lam + d_lambda < 0.0:
        d_lambda = -lam

    contact_lambda[i] = lam + d_lambda
    contact_sphere[i] = best
    contact_normal[i] = best_normal
    x[i] = p + best_normal * (w * d_lambda)


@wp.kernel
def accumulate_contact_force(
    contact_lambda: wp.array(dtype=float),
    contact_sphere: wp.array(dtype=wp.int32),
    contact_normal: wp.array(dtype=wp.vec3),
    reaction: wp.array(dtype=wp.vec3),
    inv_h2: float,
):
    """Reaction on each sphere: -lambda * n / h^2, summed over particles."""
    i = wp.tid()
    s = contact_sphere[i]
    if s < 0:
        return
    wp.atomic_add(reaction, s, contact_normal[i] * (-contact_lambda[i] * inv_h2))
