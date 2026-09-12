"""Compare simulated stem deflection against real measured plants.

The comparison is deliberately built around a quantity that can be measured
from a straight-on photograph with a protractor: the **tip angle**, the angle
between vertical and the straight line from the stem base to the stem tip.
Curvature along the stem would be a richer comparison, but it needs careful
camera calibration, and a measurement that cannot be made reliably in an
afternoon is not a measurement.

Two modes, because a household fan is not a calibrated wind tunnel:

  direct     airspeed was measured for each condition. The simulation is run
             at that airspeed and the predicted angle is compared to the
             measured one. This is a genuine validation: zero free parameters.

  trend      airspeed was not measured, only fan settings. A single scale
             factor mapping setting index to airspeed is fitted across *all*
             conditions at once, and the residuals are reported. With one free
             parameter and three or more conditions this still tests whether
             the model reproduces the shape of the response, but it is
             calibration plus a trend test, not an absolute validation, and
             it must be labelled that way in the paper.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np

from .config import SimConfig
from .solver import PlantSolver

REQUIRED_COLUMNS = [
    "trial", "species", "stem_height_m", "base_diameter_mm",
    "tip_diameter_mm", "condition", "wind_speed_ms", "tip_angle_deg",
]


@dataclass
class Measurement:
    trial: str
    species: str
    stem_height_m: float
    base_diameter_mm: float
    tip_diameter_mm: float
    condition: str
    wind_speed_ms: float | None
    tip_angle_deg: float
    notes: str = ""


@dataclass
class ConditionResult:
    condition: str
    measured: list = field(default_factory=list)
    predicted_deg: float = float("nan")
    wind_speed_ms: float = float("nan")

    @property
    def measured_mean(self) -> float:
        return float(np.mean(self.measured))

    @property
    def measured_range(self) -> tuple:
        return float(np.min(self.measured)), float(np.max(self.measured))

    @property
    def n(self) -> int:
        return len(self.measured)

    @property
    def error_deg(self) -> float:
        return self.predicted_deg - self.measured_mean


def load_measurements(path) -> list:
    """Read the measurement CSV, refusing anything that is not real data.

    The template ships with no data rows on purpose. Fabricating a row here
    to make the pipeline run would defeat the entire point of the exercise,
    so an empty file is an error with instructions, never a silent default.
    """
    rows = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(
            line for line in handle if not line.lstrip().startswith("#")
        )
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}")

        for raw in reader:
            if not (raw.get("trial") or "").strip():
                continue
            speed = (raw.get("wind_speed_ms") or "").strip()
            rows.append(Measurement(
                trial=raw["trial"].strip(),
                species=raw["species"].strip(),
                stem_height_m=float(raw["stem_height_m"]),
                base_diameter_mm=float(raw["base_diameter_mm"]),
                tip_diameter_mm=float(raw["tip_diameter_mm"]),
                condition=raw["condition"].strip(),
                wind_speed_ms=float(speed) if speed else None,
                tip_angle_deg=float(raw["tip_angle_deg"]),
                notes=(raw.get("notes") or "").strip(),
            ))

    if not rows:
        raise ValueError(
            f"{path} contains no measurement rows.\n"
            "This file is a template and ships empty by design -- it is meant to be\n"
            "filled in with real measurements from a real plant. Nothing in this\n"
            "project will invent values for it."
        )
    return rows


def config_from_measurement(m: Measurement, steady: bool = True) -> SimConfig:
    """Build a simulation of *this specific plant*, from its real geometry.

    The point of measuring the plant is that the simulation is then not of a
    generic plant but of the one that was photographed. Only the material
    constants remain assumed.
    """
    cfg = SimConfig()
    cfg.stem.length = m.stem_height_m
    cfg.stem.radius_base = m.base_diameter_mm / 2000.0     # mm diameter -> m radius
    cfg.stem.radius_tip = m.tip_diameter_mm / 2000.0
    cfg.scene.n_plants = 1
    cfg.scene.height_jitter = 0.0

    if steady:
        # A household fan is close to a steady jet. Leaving the atmospheric
        # gust model on would add sway the real plant did not experience.
        cfg.wind.gust_amplitude = 0.0
        cfg.wind.turbulence_amplitude = 0.0
        cfg.wind.shear_exponent = 0.0
    return cfg


def simulate_tip_angle(cfg: SimConfig, wind_speed: float,
                       settle_frames: int = 420, average_frames: int = 120,
                       device: str | None = None) -> float:
    """Mean tip angle from vertical, in degrees, after the plant settles."""
    cfg.wind.speed = float(wind_speed)
    solver = PlantSolver(cfg, device=device)
    solver.run(settle_frames)

    tpl = solver.scene.template
    angles = []
    for _ in range(average_frames):
        solver.step()
        stem = solver.positions()[solver.scene.plant_slice(0)][tpl.stem_path]
        axis = stem[-1] - stem[0]
        horizontal = float(np.hypot(axis[0], axis[2]))
        angles.append(np.degrees(np.arctan2(horizontal, float(axis[1]))))
    solver.sync()
    return float(np.mean(angles))


def group_by_condition(rows: list) -> dict:
    grouped: dict = {}
    for m in rows:
        grouped.setdefault(m.condition, []).append(m)
    return grouped


def run_direct(rows: list, device: str | None = None) -> list:
    """Every condition has a measured airspeed: zero free parameters."""
    results = []
    for condition, group in group_by_condition(rows).items():
        speeds = [m.wind_speed_ms for m in group]
        if any(s is None for s in speeds):
            raise ValueError(
                f"condition '{condition}' has rows without wind_speed_ms; "
                "use trend mode instead, or fill in every airspeed"
            )
        speed = float(np.mean(speeds))
        cfg = config_from_measurement(group[0])
        result = ConditionResult(
            condition=condition,
            measured=[m.tip_angle_deg for m in group],
            predicted_deg=simulate_tip_angle(cfg, speed, device=device),
            wind_speed_ms=speed,
        )
        results.append(result)
    return results


def run_trend(rows: list, device: str | None = None,
              speed_grid=np.arange(1.0, 9.01, 0.5)) -> tuple:
    """Fit one airspeed-per-condition scale across all conditions at once.

    Conditions are assumed ordered (fan setting 1 < 2 < 3) and airspeed is
    taken proportional to the setting index, with a single unknown scale `k`
    shared by every condition. `k` is chosen to minimise total squared angle
    error. One free parameter for N conditions: with N >= 3 the fit cannot
    absorb the shape of the response, only its overall magnitude.
    """
    grouped = group_by_condition(rows)
    conditions = sorted(grouped)
    measured = np.array([np.mean([m.tip_angle_deg for m in grouped[c]])
                         for c in conditions])

    # Precompute the simulated angle at every candidate airspeed once, so the
    # fit is a table lookup rather than thousands of simulations.
    cfg_template = config_from_measurement(grouped[conditions[0]][0])
    curve = np.array([
        simulate_tip_angle(config_from_measurement(grouped[conditions[0]][0]),
                           float(v), device=device)
        for v in speed_grid
    ])

    best_k, best_error = None, np.inf
    for k in np.arange(0.25, 4.01, 0.05):
        speeds = k * np.arange(1, len(conditions) + 1)
        predicted = np.interp(speeds, speed_grid, curve)
        error = float(np.sum((predicted - measured) ** 2))
        if error < best_error:
            best_k, best_error = float(k), error

    speeds = best_k * np.arange(1, len(conditions) + 1)
    predicted = np.interp(speeds, speed_grid, curve)

    results = []
    for c, speed, pred in zip(conditions, speeds, predicted):
        results.append(ConditionResult(
            condition=c,
            measured=[m.tip_angle_deg for m in grouped[c]],
            predicted_deg=float(pred),
            wind_speed_ms=float(speed),
        ))
    return results, best_k, (speed_grid, curve)
