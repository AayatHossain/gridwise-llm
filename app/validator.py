"""Final validator: replays a plan hour by hour the way the judge does.

Used twice: as the service's own final check before a response leaves, and
by the public-sample runner / tests to replay plans against the organizer
ground-truth directives (Problem Statement section 11).
"""
from __future__ import annotations

import math

from .directives import HOURS, Constraints

ACTIONS = ("charge", "discharge", "idle")


def _num(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v)


def validate_plan(cons: Constraints, plan: list[dict], tol: float = 1e-3) -> list[str]:
    """Return a list of violations (empty list == valid)."""
    if not isinstance(plan, list) or len(plan) != HOURS:
        return [f"hourly_plan must contain exactly {HOURS} entries"]
    try:
        by_hour = {int(p["hour"]): p for p in plan}
    except (KeyError, TypeError, ValueError):
        return ["every hourly_plan entry needs an integer hour"]
    if sorted(by_hour) != list(range(HOURS)):
        return ["hourly_plan must contain hours 0 through 23 exactly once"]

    errors: list[str] = []
    energy = cons.initial_energy
    for h in range(HOURS):
        p = by_hour[h]
        g = _num(p.get("grid_kwh"))
        s = _num(p.get("solar_used_kwh"))
        k = _num(p.get("battery_kwh"))
        e_after = _num(p.get("battery_energy_after_kwh"))
        action = p.get("battery_action")
        if None in (g, s, k, e_after):
            errors.append(f"hour {h}: numeric fields must be finite numbers")
            continue
        if action not in ACTIONS:
            errors.append(f"hour {h}: battery_action must be charge, discharge or idle")
            continue
        for name, v in (
            ("grid_kwh", g),
            ("solar_used_kwh", s),
            ("battery_kwh", k),
            ("battery_energy_after_kwh", e_after),
        ):
            if v < -tol:
                errors.append(f"hour {h}: {name} is negative")
        if action == "idle" and abs(k) > tol:
            errors.append(f"hour {h}: battery_kwh must be 0 when idle")
        charge = k if action == "charge" else 0.0
        discharge = k if action == "discharge" else 0.0
        if charge > cons.charge_max[h] + tol:
            errors.append(f"hour {h}: charge {charge:g} exceeds allowed {cons.charge_max[h]:g}")
        if discharge > cons.discharge_max[h] + tol:
            errors.append(f"hour {h}: discharge {discharge:g} exceeds allowed {cons.discharge_max[h]:g}")
        if s > cons.effective_solar[h] + tol:
            errors.append(f"hour {h}: solar_used {s:g} exceeds effective solar {cons.effective_solar[h]:g}")
        if g > cons.grid_max[h] + tol:
            errors.append(f"hour {h}: grid {g:g} exceeds cap {cons.grid_max[h]:g}")
        if abs(g + s + discharge - (cons.demand[h] + charge)) > tol:
            errors.append(f"hour {h}: energy balance violated")
        expected = energy + charge - discharge
        if abs(e_after - expected) > tol:
            errors.append(f"hour {h}: battery_energy_after_kwh {e_after:g} != {expected:g}")
        if e_after < cons.min_energy[h] - tol:
            errors.append(f"hour {h}: battery {e_after:g} below minimum {cons.min_energy[h]:g}")
        if e_after > cons.capacity + tol:
            errors.append(f"hour {h}: battery {e_after:g} above capacity {cons.capacity:g}")
        energy = e_after

    if abs(energy - cons.initial_energy) > tol:
        errors.append(f"end-of-day battery {energy:g} != initial {cons.initial_energy:g}")
    return errors


def validate_totals(
    plan: list[dict],
    tariff: list[float],
    total_grid: float,
    total_cost: float,
    peak: float,
    tol: float = 1e-2,
) -> list[str]:
    errors: list[str] = []
    grid = [float(p["grid_kwh"]) for p in sorted(plan, key=lambda p: p["hour"])]
    if abs(sum(grid) - total_grid) > tol:
        errors.append("total_grid_kwh does not match hourly_plan")
    if abs(sum(g * t for g, t in zip(grid, tariff)) - total_cost) > tol:
        errors.append("total_cost_bdt does not match hourly_plan")
    if abs(max(grid) - peak) > tol:
        errors.append("peak_grid_kwh does not match hourly_plan")
    return errors
