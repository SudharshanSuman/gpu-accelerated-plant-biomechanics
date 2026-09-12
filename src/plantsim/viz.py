"""Matplotlib figures for the report: bending panels, canopy views, benchmarks."""

from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")           # headless: these run over SSH and in CI
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

STEM_COLOURS = ["#1D9E75", "#0F6E56", "#085041", "#04342C", "#5DCAA5", "#9FE1CB"]
SOIL = "#8B5E3C"
LEAF = "#3FBF7F"


def _draw_plant(ax, pos, template, colour, plane=(0, 1), lw=2.4, leaves=True):
    """Draw one plant projected onto two axes (default X-Y)."""
    i, j = plane
    stem = pos[template.stem_path]
    ax.plot(stem[:, i], stem[:, j], "-", color=colour, linewidth=lw,
            solid_capstyle="round", zorder=3)
    ax.plot(stem[:, i], stem[:, j], "o", color=colour, markersize=2.6, zorder=4)
    if leaves:
        for path in template.leaf_paths:
            blade = pos[path]
            ax.plot(blade[:, i], blade[:, j], "-", color=LEAF,
                    linewidth=1.5, alpha=0.85, zorder=2)
    ax.plot(stem[0, i], stem[0, j], "o", color=SOIL, markersize=7, zorder=5)


def _style_ground(ax, xlim, ylim):
    ax.axhspan(ylim[0], 0.0, color=SOIL, alpha=0.18, zorder=0)
    ax.axhline(0.0, color=SOIL, linewidth=2.0, alpha=0.9, zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.18)


def plot_bending_panels(frames, template, out_path, title, stem_length=0.9):
    """Time series of a single plant bending, one panel per sampled instant."""
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 4.6))
    axes = np.atleast_1d(axes)

    all_x = np.concatenate([p[template.stem_path][:, 0] for _, p in frames])
    xlim = (min(-0.15, all_x.min() - 0.1), max(0.45, all_x.max() + 0.12))
    ylim = (-0.12, stem_length * 1.18)

    for k, (ax, (t, pos)) in enumerate(zip(axes, frames)):
        _draw_plant(ax, pos, template, STEM_COLOURS[k % len(STEM_COLOURS)])
        _style_ground(ax, xlim, ylim)
        ax.set_title(f"t = {t:.2f} s", fontsize=10)
        if k:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("height (m)")
        ax.set_xlabel("x (m)")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_stiffness_profile(template, out_path, n_stem):
    """Show that rigidity falls off with height, and by how much."""
    s = template.height_param[:n_stem]
    r = template.radius[:n_stem]
    # Bending compliance is stored per hinge (interior stem nodes 1..n-2).
    alpha = template.bend_compliance[: n_stem - 2]
    stiffness = 1.0 / alpha
    h = s[1:n_stem - 1]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(r * 1000, s * 100, color=STEM_COLOURS[0], linewidth=2)
    a1.set_xlabel("stem radius (mm)")
    a1.set_ylabel("height along stem (%)")
    a1.set_title("Tapered geometry")
    a1.grid(alpha=0.2)

    a2.semilogy(h * 100, stiffness, color=STEM_COLOURS[1], linewidth=2)
    a2.set_xlabel("height along stem (%)")
    a2.set_ylabel("bending stiffness  1/$\\alpha$  (N/m)")
    a2.set_title(f"Rigidity vs height — base is {stiffness[0] / stiffness[-1]:.0f}x stiffer")
    a2.grid(alpha=0.2, which="both")

    fig.suptitle("Rigidity falls with height because $EI \\propto r^4$", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_field(positions, scene, out_path, title, max_plants=120):
    """Side elevation and plan view of a whole simulated plot."""
    tpl = scene.template
    n = min(scene.n_plants, max_plants)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for p in range(n):
        pos = positions[scene.plant_slice(p)]
        _draw_plant(a1, pos, tpl, STEM_COLOURS[p % len(STEM_COLOURS)], lw=1.4)
        _draw_plant(a2, pos, tpl, STEM_COLOURS[p % len(STEM_COLOURS)],
                    plane=(0, 2), lw=1.0, leaves=False)

    a1.axhspan(-0.2, 0.0, color=SOIL, alpha=0.18)
    a1.axhline(0.0, color=SOIL, linewidth=2)
    a1.set_aspect("equal")
    a1.set_xlabel("x — downwind (m)")
    a1.set_ylabel("height (m)")
    a1.set_title("Side elevation")
    a1.grid(alpha=0.18)

    a2.set_aspect("equal")
    a2.set_xlabel("x — downwind (m)")
    a2.set_ylabel("z — across rows (m)")
    a2.set_title("Plan view (wind blows +x)")
    a2.grid(alpha=0.18)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def animate_plant(frames, template, out_path, stem_length=0.9, fps=30):
    """Write a GIF of one plant responding to the wind."""
    fig, ax = plt.subplots(figsize=(4.2, 5.0))
    all_x = np.concatenate([p[template.stem_path][:, 0] for _, p in frames])
    xlim = (min(-0.15, all_x.min() - 0.1), max(0.45, all_x.max() + 0.12))
    ylim = (-0.12, stem_length * 1.18)

    def draw(k):
        ax.clear()
        t, pos = frames[k]
        _draw_plant(ax, pos, template, STEM_COLOURS[0])
        _style_ground(ax, xlim, ylim)
        ax.set_title(f"t = {t:.2f} s")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("height (m)")

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def plot_benchmark(results, out_path):
    """Scaling curves and speed-up from the CPU/GPU benchmark."""
    particles = np.array([r["particles"] for r in results], dtype=float)
    cpu = np.array([r["cpu_ms"] for r in results], dtype=float)
    gpu = np.array([r["gpu_ms"] for r in results], dtype=float)
    speedup = cpu / np.maximum(gpu, 1e-9)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))

    a1.loglog(particles, cpu, "o-", color="#C2553B", linewidth=2, label="CPU (NumPy)")
    a1.loglog(particles, gpu, "o-", color=STEM_COLOURS[0], linewidth=2, label="GPU (Warp)")
    a1.axhline(1000 / 60, color="#777", linestyle="--", linewidth=1,
               label="60 FPS budget (16.7 ms)")
    a1.set_xlabel("particles")
    a1.set_ylabel("time per frame (ms)")
    a1.set_title("Frame cost vs scene size")
    a1.grid(alpha=0.25, which="both")
    a1.legend(fontsize=9)

    a2.semilogx(particles, speedup, "o-", color=STEM_COLOURS[1], linewidth=2)
    a2.axhline(1.0, color="#777", linestyle="--", linewidth=1)
    a2.set_xlabel("particles")
    a2.set_ylabel("speed-up  (CPU / GPU)")
    a2.set_title(f"GPU speed-up — peak {speedup.max():.1f}x")
    a2.grid(alpha=0.25, which="both")

    fig.suptitle("GPU vs CPU plant biomechanics throughput", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_robot_interaction(snapshots, force_log, template, out_path,
                           stem_length=0.85):
    """Contact snapshots above, reaction force against time below."""
    n = len(snapshots)
    fig = plt.figure(figsize=(3.0 * n, 7.4))
    gs = fig.add_gridspec(2, n, height_ratios=[2.0, 1.0], hspace=0.28)

    all_x = np.concatenate([p[template.stem_path][:, 0] for _, p, _ in snapshots])
    all_e = np.array([e[0] for _, _, e in snapshots])
    xlim = (min(-0.35, all_x.min() - 0.1, all_e.min() - 0.12),
            max(0.45, all_x.max() + 0.12, all_e.max() + 0.12))
    ylim = (-0.12, stem_length * 1.18)

    for k, (t, pos, eff) in enumerate(snapshots):
        ax = fig.add_subplot(gs[0, k])
        _draw_plant(ax, pos, template, STEM_COLOURS[k % len(STEM_COLOURS)])
        ax.add_patch(plt.Circle((eff[0], eff[1]), eff[3], color="#3E6DB5",
                                alpha=0.85, zorder=6))
        _style_ground(ax, xlim, ylim)
        ax.set_title(f"t = {t:.2f} s", fontsize=10)
        ax.set_xlabel("x (m)")
        if k == 0:
            ax.set_ylabel("height (m)")
        else:
            ax.set_yticklabels([])

    ax = fig.add_subplot(gs[1, :])
    times = np.array([f[0] for f in force_log])
    mags = np.array([f[1] for f in force_log])
    ax.plot(times, mags, color="#3E6DB5", linewidth=1.8)
    ax.fill_between(times, 0, mags, color="#3E6DB5", alpha=0.18)
    for t, _, _ in snapshots:
        ax.axvline(t, color="#999", linestyle=":", linewidth=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("contact force (N)")
    ax.set_title("Reaction force on the end effector")
    ax.grid(alpha=0.25)

    fig.suptitle("Robot end effector pushing through a deformable plant", fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_field_validation(results, out_path, mode, curve=None):
    """Measured vs predicted tip angle, with per-condition spread."""
    conditions = [r.condition for r in results]
    measured = np.array([r.measured_mean for r in results])
    predicted = np.array([r.predicted_deg for r in results])
    lo = np.array([r.measured_range[0] for r in results])
    hi = np.array([r.measured_range[1] for r in results])
    x = np.arange(len(results))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))

    a1.errorbar(x, measured, yerr=[measured - lo, hi - measured],
                fmt="o", color="#C2553B", capsize=5, markersize=8,
                linewidth=2, label="measured (mean, range)")
    a1.plot(x, predicted, "s--", color=STEM_COLOURS[0], markersize=8,
            linewidth=2, label="simulated")
    a1.set_xticks(x)
    a1.set_xticklabels(conditions)
    a1.set_xlabel("condition")
    a1.set_ylabel("tip angle from vertical (deg)")
    a1.set_title(f"Deflection response ({mode} mode)")
    a1.grid(alpha=0.25)
    a1.legend(fontsize=9)

    limit = max(measured.max(), predicted.max()) * 1.15 + 1
    a2.plot([0, limit], [0, limit], "--", color="#777", linewidth=1,
            label="perfect agreement")
    a2.scatter(measured, predicted, s=90, color=STEM_COLOURS[1], zorder=3)
    for k, r in enumerate(results):
        a2.annotate(r.condition, (measured[k], predicted[k]),
                    textcoords="offset points", xytext=(7, 5), fontsize=9)
    a2.set_xlim(0, limit)
    a2.set_ylim(0, limit)
    a2.set_aspect("equal")
    a2.set_xlabel("measured tip angle (deg)")
    a2.set_ylabel("simulated tip angle (deg)")
    a2.set_title("Agreement")
    a2.grid(alpha=0.25)
    a2.legend(fontsize=9)

    fig.suptitle("Simulated vs measured stem deflection", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


ARM_COLOUR = "#3E6DB5"
TOOL_COLOUR = "#D96C33"


def _draw_arm(ax, joints, spheres, plane=(0, 1), show_spheres=True):
    """Draw the manipulator projected onto two axes."""
    i, j = plane
    origin, elbow, tool = joints

    if show_spheres:
        centres, radii = spheres
        for centre, radius in zip(centres, radii):
            ax.add_patch(plt.Circle((centre[i], centre[j]), radius,
                                    color=ARM_COLOUR, alpha=0.13, zorder=2))

    ax.plot([origin[i], elbow[i]], [origin[j], elbow[j]], "-",
            color=ARM_COLOUR, linewidth=7, solid_capstyle="round", zorder=6)
    ax.plot([elbow[i], tool[i]], [elbow[j], tool[j]], "-",
            color=ARM_COLOUR, linewidth=5.5, solid_capstyle="round", zorder=6)
    ax.plot([origin[i], elbow[i], tool[i]], [origin[j], elbow[j], tool[j]],
            "o", color="#24456F", markersize=5, zorder=7)
    ax.plot(tool[i], tool[j], "o", color=TOOL_COLOUR, markersize=11, zorder=8)


def plot_scene_panels(snapshots, scene, out_path, title):
    """Time panels of the arm driving through the canopy (side elevation)."""
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.8))
    axes = np.atleast_1d(axes)
    tpl = scene.template

    for k, (ax, snap) in enumerate(zip(axes, snapshots)):
        t, positions, joints, spheres, force = snap
        for p in range(scene.n_plants):
            _draw_plant(ax, positions[scene.plant_slice(p)], tpl,
                        STEM_COLOURS[p % len(STEM_COLOURS)], lw=1.8)
        _draw_arm(ax, joints, spheres)

        ax.axhspan(-0.2, 0.0, color=SOIL, alpha=0.18, zorder=0)
        ax.axhline(0.0, color=SOIL, linewidth=2, zorder=1)
        ax.set_xlim(-1.1, 2.0)
        ax.set_ylim(-0.15, 1.15)
        ax.set_aspect("equal")
        ax.grid(alpha=0.18)
        ax.set_title(f"t = {t:.1f} s   contact {force:.2f} N", fontsize=10)
        ax.set_xlabel("x — direction of travel (m)")
        if k == 0:
            ax.set_ylabel("height (m)")
        else:
            ax.set_yticklabels([])

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def animate_scene(frames, scene, out_path, fps=20, force_limit=5.0):
    """GIF of the canopy interaction: side view, plan view, and force trace."""
    tpl = scene.template
    fig = plt.figure(figsize=(11.5, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.3, 1.0], hspace=0.33, wspace=0.18)
    ax_side = fig.add_subplot(gs[0, 0])
    ax_plan = fig.add_subplot(gs[0, 1])
    ax_force = fig.add_subplot(gs[1, :])

    times = np.array([f[0] for f in frames])
    forces = np.array([f[4] for f in frames])

    def draw(k):
        t, positions, joints, spheres, force = frames[k]

        ax_side.clear()
        for p in range(scene.n_plants):
            _draw_plant(ax_side, positions[scene.plant_slice(p)], tpl,
                        STEM_COLOURS[p % len(STEM_COLOURS)], lw=1.8)
        _draw_arm(ax_side, joints, spheres)
        ax_side.axhspan(-0.2, 0.0, color=SOIL, alpha=0.18, zorder=0)
        ax_side.axhline(0.0, color=SOIL, linewidth=2, zorder=1)
        ax_side.set_xlim(-1.1, 2.0)
        ax_side.set_ylim(-0.15, 1.15)
        ax_side.set_aspect("equal")
        ax_side.grid(alpha=0.18)
        ax_side.set_title("Side elevation", fontsize=10)
        ax_side.set_xlabel("x — direction of travel (m)")
        ax_side.set_ylabel("height (m)")

        ax_plan.clear()
        for p in range(scene.n_plants):
            _draw_plant(ax_plan, positions[scene.plant_slice(p)], tpl,
                        STEM_COLOURS[p % len(STEM_COLOURS)],
                        plane=(0, 2), lw=1.4, leaves=False)
        _draw_arm(ax_plan, joints, spheres, plane=(0, 2))
        ax_plan.set_xlim(-1.1, 2.0)
        ax_plan.set_ylim(-1.0, 0.9)
        ax_plan.set_aspect("equal")
        ax_plan.grid(alpha=0.18)
        ax_plan.set_title("Plan view — vehicle drives beside the row", fontsize=10)
        ax_plan.set_xlabel("x (m)")
        ax_plan.set_ylabel("z (m)")

        ax_force.clear()
        ax_force.plot(times[:k + 1], forces[:k + 1], color=ARM_COLOUR, linewidth=2)
        ax_force.fill_between(times[:k + 1], 0, forces[:k + 1],
                              color=ARM_COLOUR, alpha=0.18)
        ax_force.axhline(force_limit, color="#C2553B", linestyle="--",
                         linewidth=1.5, label=f"{force_limit:.0f} N stem-damage threshold")
        ax_force.set_xlim(times[0], times[-1])
        ax_force.set_ylim(0, max(force_limit * 1.25, forces.max() * 1.2, 1.0))
        ax_force.set_xlabel("time (s)")
        ax_force.set_ylabel("contact force (N)")
        ax_force.grid(alpha=0.25)
        ax_force.legend(fontsize=9, loc="upper left")
        ax_force.set_title(f"Whole-arm reaction — now {force:.2f} N", fontsize=10)

        fig.suptitle("Manipulator driving through a deformable canopy "
                     "(Isaac Sim scene, solver-accurate)", fontsize=12)

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def plot_speed_damage(rows, out_path, failure_low=5.0, failure_high=15.0,
                      crossing=None, exponent=None):
    """Peak contact force against driving speed, with the damage band."""
    speeds = np.array([r["speed"] for r in rows])
    forces = np.array([r["force"] for r in rows])
    deflections = np.array([r["deflection"] for r in rows])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    a1.axhspan(failure_low, failure_high, color="#C2553B", alpha=0.13,
               label=f"assumed failure band ({failure_low:.0f}–{failure_high:.0f} MPa)")
    a1.axhline(failure_low, color="#C2553B", linestyle="--", linewidth=1.4)
    a1.plot(speeds, forces, "o-", color=ARM_COLOUR, linewidth=2.2,
            markersize=7, label="peak bending stress")

    if crossing:
        a1.axvline(crossing, color="#7A4FA3", linestyle=":", linewidth=1.8)
        a1.annotate(f"damage onset\n{crossing:.2f} m/s",
                    xy=(crossing, failure_low),
                    xytext=(crossing * 1.08, failure_low * 1.9),
                    fontsize=9, color="#7A4FA3",
                    arrowprops=dict(arrowstyle="->", color="#7A4FA3", lw=1.2))

    a1.set_xlabel("driving speed (m/s)")
    a1.set_ylabel("peak bending stress in stem (MPa)")
    title = "Stem stress vs driving speed"
    if exponent is not None:
        title += f"   (fit: $\\sigma \\propto$ v$^{{{exponent:.2f}}}$)"
    a1.set_title(title, fontsize=11)
    a1.grid(alpha=0.25)
    a1.legend(fontsize=9, loc="upper left")

    a2.plot(speeds, 100 * deflections, "s-", color=STEM_COLOURS[0],
            linewidth=2.2, markersize=7)
    a2.set_xlabel("driving speed (m/s)")
    a2.set_ylabel("peak stem deflection (cm)")
    a2.set_title("How far the plant is pushed", fontsize=11)
    a2.grid(alpha=0.25)

    fig.suptitle("How fast can a manipulator work the row before it damages the crop?",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
