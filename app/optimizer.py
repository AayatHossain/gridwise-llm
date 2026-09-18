"""Exact linear-programming optimizer (scipy / HiGHS).

Decision variables per hour h (72 in total):

    g[h]  grid import (kWh)        0 <= g[h] <= grid_max[h]
    s[h]  solar used (kWh)         0 <= s[h] <= effective_solar[h]
    b[h]  net battery flow (kWh)  -discharge_max[h] <= b[h] <= charge_max[h]
          b > 0 charges, b < 0 discharges. A no-charge hour sets the upper
          bound to 0, a no-discharge hour sets the lower bound to 0, so one
          variable also guarantees exactly one battery action per hour.

Constraints (Problem Statement section 09 and 5.3):

    g[h] + s[h] - b[h] = demand[h]                       energy balance
    min_energy[h] <= E0 + b[0] + ... + b[h] <= capacity   battery bounds / reserve
    b[0] + ... + b[23] = 0                                end-of-day neutrality

Objective: minimize sum(tariff[h] * g[h]).

There is no round-trip loss and no export, so the problem is exactly linear
and the LP optimum is the true minimum cost. It matches the organizer
reference cost on all 10 public cases.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .directives import HOURS, Constraints

DECIMALS = 6


@dataclass
class RawSolution:
    grid: list[float]
    solar_used: list[float]
    net_battery: list[float]
    cost: float


def solve_lp(cons: Constraints) -> RawSolution | None:
    """Return the LP optimum, or None when the constraints are infeasible."""
    H = HOURS
    n = 3 * H
    c = np.zeros(n)
    c[:H] = cons.tariff

    A_eq = np.zeros((H + 1, n))
    b_eq = np.zeros(H + 1)
    for h in range(H):  # g + s - b = demand
        A_eq[h, h] = 1.0
        A_eq[h, H + h] = 1.0
        A_eq[h, 2 * H + h] = -1.0
        b_eq[h] = cons.demand[h]
    A_eq[H, 2 * H :] = 1.0  # sum(b) = 0 -> battery ends where it started

    A_ub = np.zeros((2 * H, n))
    b_ub = np.zeros(2 * H)
    for h in range(H):  # min_energy[h] <= E0 + cumsum(b) <= capacity
        A_ub[2 * h, 2 * H : 2 * H + h + 1] = 1.0
        b_ub[2 * h] = cons.capacity - cons.initial_energy
        A_ub[2 * h + 1, 2 * H : 2 * H + h + 1] = -1.0
        b_ub[2 * h + 1] = cons.initial_energy - cons.min_energy[h]

    bounds: list[tuple[float, float | None]] = []
    for h in range(H):
        cap = cons.grid_max[h]
        bounds.append((0.0, None if math.isinf(cap) else max(0.0, cap)))
    for h in range(H):
        bounds.append((0.0, max(0.0, cons.effective_solar[h])))
    for h in range(H):
        bounds.append((-max(0.0, cons.discharge_max[h]), max(0.0, cons.charge_max[h])))

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0 or res.x is None:
        return None
    x = res.x
    return RawSolution(
        grid=x[:H].tolist(),
        solar_used=x[H : 2 * H].tolist(),
        net_battery=x[2 * H :].tolist(),
        cost=float(res.fun),
    )


def _r(x: float) -> float:
    v = round(float(x), DECIMALS)
    return 0.0 if abs(v) < 1e-9 else v


def build_plan(cons: Constraints, sol: RawSolution) -> list[dict]:
    """Turn the LP solution into the 24 hourly_plan entries.

    Values are rounded to 6 decimals, then grid is recomputed from the rounded
    solar and battery values so the energy balance holds exactly on the numbers
    we return, and the last hour absorbs any rounding drift so the battery ends
    exactly at its initial level.
    """
    net = [_r(v) for v in sol.net_battery]
    drift = _r(sum(net))
    if drift != 0.0:
        net[-1] = _r(net[-1] - drift)

    plan: list[dict] = []
    energy = cons.initial_energy
    for h in range(HOURS):
        b = net[h]
        s = _r(min(max(sol.solar_used[h], 0.0), cons.effective_solar[h]))
        g = _r(cons.demand[h] + b - s)
        if g < 0.0:  # only possible through rounding noise; repair without breaking the balance
            if s + g >= 0.0:
                s = _r(s + g)
            else:
                b = _r(s - cons.demand[h])
            g = 0.0
        energy = _r(energy + b)
        action = "charge" if b > 0 else "discharge" if b < 0 else "idle"
        plan.append(
            {
                "hour": h,
                "grid_kwh": g,
                "solar_used_kwh": s,
                "battery_action": action,
                "battery_kwh": abs(b),
                "battery_energy_after_kwh": energy,
            }
        )
    return plan


def compute_totals(plan: list[dict], tariff: list[float]) -> tuple[float, float, float]:
    """total_grid_kwh, total_cost_bdt, peak_grid_kwh recomputed from the plan itself."""
    total_grid = _r(sum(p["grid_kwh"] for p in plan))
    total_cost = _r(sum(p["grid_kwh"] * tariff[p["hour"]] for p in plan))
    peak = _r(max(p["grid_kwh"] for p in plan))
    return total_grid, total_cost, peak
