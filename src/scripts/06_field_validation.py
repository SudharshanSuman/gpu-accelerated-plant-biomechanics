"""Compare the simulator against real measured plants.

Fill in `Research/field_measurements.csv` first. This script will not run
without real data in it, and will not invent any.

Produces:
    results/field_validation.md    the Results subsection, ready to adapt
    results/field_validation.png   measured vs predicted figure
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from _bootstrap import RESULTS, ROOT
from plantsim.field import load_measurements, run_direct, run_trend
from plantsim.viz import plot_field_validation

CSV = ROOT / "Research" / "field_measurements.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV))
    ap.add_argument("--mode", choices=["auto", "direct", "trend"], default="auto")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    try:
        rows = load_measurements(args.csv)
    except (FileNotFoundError, ValueError) as exc:
        print(f"{exc}\n")
        print("Once you have photographs and calliper readings, one row per")
        print("photo, this script writes the whole Results subsection for you.")
        return 1

    have_speeds = all(m.wind_speed_ms is not None for m in rows)
    mode = args.mode
    if mode == "auto":
        mode = "direct" if have_speeds else "trend"
    if mode == "direct" and not have_speeds:
        print("direct mode needs wind_speed_ms on every row; some are blank.")
        return 1

    species = sorted({m.species for m in rows})
    heights = sorted({m.stem_height_m for m in rows})
    print(f"{len(rows)} measurements, {len(set(m.condition for m in rows))} conditions")
    print(f"species: {', '.join(species)}")
    print(f"mode: {mode}")
    print()

    fitted_k, curve = None, None
    if mode == "direct":
        results = run_direct(rows, device=args.device)
    else:
        results, fitted_k, curve = run_trend(rows, device=args.device)

    errors = np.array([r.error_deg for r in results])
    measured = np.array([r.measured_mean for r in results])
    mae = float(np.mean(np.abs(errors)))
    rel = float(np.mean(np.abs(errors) / np.maximum(measured, 1e-6))) * 100

    header = f"{'condition':>10} {'n':>3} {'measured':>18} {'simulated':>10} {'error':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        lo, hi = r.measured_range
        print(f"{r.condition:>10} {r.n:>3} "
              f"{r.measured_mean:8.1f} ({lo:.1f}-{hi:.1f}) "
              f"{r.predicted_deg:9.1f} {r.error_deg:+8.1f}")
    print()
    print(f"mean absolute error: {mae:.1f} deg  ({rel:.0f}% of measured)")

    figure = plot_field_validation(results, RESULTS / "field_validation.png",
                                   mode, curve)
    print(f"wrote {figure}")

    # ---- write the Results subsection ---------------------------------
    n_total = len(rows)
    n_conditions = len(results)

    if mode == "direct":
        claim = (
            "Airspeed was measured for every condition, so the simulation has "
            "**no free parameters fitted to this data**: stem geometry comes "
            "from calliper readings and airspeed from the anemometer."
        )
        caveat = ""
    else:
        claim = (
            f"Airspeed was **not** measured. A single scale factor "
            f"(k = {fitted_k:.2f} m/s per fan setting) was fitted across all "
            f"{n_conditions} conditions simultaneously to map fan setting to "
            f"airspeed. With one free parameter and {n_conditions} conditions, "
            f"the fit can absorb the overall magnitude of the response but not "
            f"its shape."
        )
        caveat = (
            "\n> **This is calibration plus a trend test, not an absolute "
            "validation.** It must be described that way in the paper. An "
            "anemometer reading for each fan setting would upgrade it to a "
            "zero-free-parameter comparison.\n"
        )

    lines = [
        "# Field validation — simulated vs measured stem deflection",
        "",
        f"- **{n_total} measurements** across **{n_conditions} conditions**",
        f"- species: {', '.join(species)}",
        f"- stem height(s): {', '.join(f'{h:.2f} m' for h in heights)}",
        f"- mode: `{mode}`",
        "",
        claim,
        caveat,
        "## Results",
        "",
        "| condition | n | measured mean (deg) | measured range | simulated (deg) | error (deg) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lo, hi = r.measured_range
        lines.append(
            f"| {r.condition} | {r.n} | {r.measured_mean:.1f} | "
            f"{lo:.1f}–{hi:.1f} | {r.predicted_deg:.1f} | {r.error_deg:+.1f} |"
        )
    lines += [
        "",
        f"Mean absolute error **{mae:.1f}°** ({rel:.0f}% of the measured angle).",
        "",
        "## Limitations",
        "",
        f"This is a **preliminary validation** and is reported as such. "
        f"n = {n_total} across {n_conditions} conditions on "
        f"{'a single plant' if len(heights) == 1 else f'{len(heights)} plants'} "
        f"of {'one species' if len(species) == 1 else f'{len(species)} species'}. "
        "It establishes that the model reproduces the magnitude and trend of "
        "real stem deflection; it does not establish accuracy across species, "
        "growth stages, or wind regimes.",
        "",
        "Specific threats to validity:",
        "",
        "- **Material constants are assumed, not measured.** Young's modulus "
        "and drag coefficients come from published ranges for herbaceous "
        "tissue, not from bending tests on this plant. Agreement therefore "
        "partly reflects those assumptions being reasonable for this species.",
        "- **Tip angle is a coarse summary of shape.** Two stems with "
        "different curvature can share a tip angle. Matching the full "
        "centreline would be a stronger claim and needs camera calibration.",
        "- **Leaf area was not measured**, and leaves carry most of the "
        "aerodynamic load. Leaf geometry remains at library defaults.",
    ]
    if mode == "trend":
        lines.append(
            "- **Airspeed was uncalibrated**, so the airspeed scale is fitted "
            "rather than measured (see above)."
        )
    lines += [
        "- **Single-plant aerodynamics.** The real plant stood in still room "
        "air in front of a fan; the model assumes the prescribed flow reaches "
        "the whole plant with no wake or sheltering.",
    ]

    out = RESULTS / "field_validation.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
