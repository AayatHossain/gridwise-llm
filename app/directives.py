"""Directive domain model and merging of directives into per-hour constraints.

Everything in this module is deterministic. The language model never touches
these structures directly: its raw JSON goes through
``guardrails.validate_llm_output`` first, and only validated ``Directive``
objects reach ``build_constraints``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

HOURS = 24

DIRECTIVE_TYPES = frozenset(
    {
        "solar_reduction",
        "minimum_battery_reserve",
        "no_charge_window",
        "no_discharge_window",
        "max_grid_window",
        "no_op",
    }
)


@dataclass(frozen=True)
class Directive:
    """One validated interpretation of one operator note."""

    note_index: int
    directive_type: str
    hours: tuple[int, ...] = ()
    factor: float | None = None
    minimum_energy_kwh: float | None = None
    max_grid_kwh: float | None = None
    explanation: str = ""

    @property
    def applies(self) -> bool:
        return self.directive_type != "no_op"

    def structured_adjustment(self) -> dict[str, Any] | None:
        t = self.directive_type
        if t == "no_op":
            return None
        hours = list(self.hours)
        if t == "solar_reduction":
            return {"hours": hours, "factor": self.factor}
        if t == "minimum_battery_reserve":
            return {"hours": hours, "minimum_energy_kwh": self.minimum_energy_kwh}
        if t == "max_grid_window":
            return {"hours": hours, "max_grid_kwh": self.max_grid_kwh}
        return {"hours": hours}  # no_charge_window / no_discharge_window

    def to_interpretation(self) -> dict[str, Any]:
        return {
            "note_index": self.note_index,
            "applies": self.applies,
            "directive_type": self.directive_type,
            "structured_adjustment": self.structured_adjustment(),
            "explanation": self.explanation,
        }

    def describe(self) -> str:
        """Short human-readable description used in plan_summary."""
        if not self.applies:
            return "no_op"
        span = hour_span(self.hours)
        t = self.directive_type
        if t == "solar_reduction":
            return f"solar reduced to {self.factor:g}x in {span}"
        if t == "minimum_battery_reserve":
            return f"battery reserve >= {self.minimum_energy_kwh:g} kWh in {span}"
        if t == "no_charge_window":
            return f"no charging in {span}"
        if t == "no_discharge_window":
            return f"no discharging in {span}"
        return f"grid import <= {self.max_grid_kwh:g} kWh in {span}"


def hour_span(hours: tuple[int, ...] | list[int]) -> str:
    hs = sorted(hours)
    if not hs:
        return "no hours"
    if hs == list(range(hs[0], hs[-1] + 1)):
        return f"hour {hs[0]}" if len(hs) == 1 else f"hours {hs[0]}-{hs[-1]}"
    return "hours " + ",".join(str(h) for h in hs)


def no_op(note_index: int, explanation: str) -> Directive:
    return Directive(note_index=note_index, directive_type="no_op", explanation=explanation)


def directive_from_interpretation(entry: dict[str, Any]) -> Directive:
    """Build a Directive from a response-shaped interpretation entry.

    Used by the sample runner / tests to replay a plan against the organizer
    ground truth, exactly as the judge does.
    """
    t = entry["directive_type"]
    adj = entry.get("structured_adjustment") or {}
    return Directive(
        note_index=int(entry["note_index"]),
        directive_type=t,
        hours=tuple(int(h) for h in adj.get("hours", [])),
        factor=adj.get("factor"),
        minimum_energy_kwh=adj.get("minimum_energy_kwh"),
        max_grid_kwh=adj.get("max_grid_kwh"),
        explanation=str(entry.get("explanation", "")),
    )


@dataclass
class Constraints:
    """Per-hour limits after merging the base battery rules with directives."""

    demand: list[float]
    tariff: list[float]
    effective_solar: list[float]  # kWh usable in each hour
    min_energy: list[float]  # battery floor after each hour
    capacity: float
    initial_energy: float
    charge_max: list[float]  # 0 inside a no-charge window
    discharge_max: list[float]  # 0 inside a no-discharge window
    grid_max: list[float]  # inf unless a max_grid_window covers the hour


def build_constraints(scenario, directives: list[Directive]) -> Constraints:
    """Apply validated directives deterministically (Problem Statement 5.3)."""
    hours = sorted(scenario.hours, key=lambda h: h.hour)
    b = scenario.battery
    cons = Constraints(
        demand=[float(h.demand_kwh) for h in hours],
        tariff=[float(h.tariff_bdt_per_kwh) for h in hours],
        effective_solar=[float(h.solar_kwh) for h in hours],
        min_energy=[float(b.minimum_energy_kwh)] * HOURS,
        capacity=float(b.capacity_kwh),
        initial_energy=float(b.initial_energy_kwh),
        charge_max=[float(b.max_charge_kwh_per_hour)] * HOURS,
        discharge_max=[float(b.max_discharge_kwh_per_hour)] * HOURS,
        grid_max=[float("inf")] * HOURS,
    )
    for d in directives:
        if not d.applies:
            continue
        for h in d.hours:
            if d.directive_type == "solar_reduction":
                # Overlapping reductions multiply: the most conservative reading,
                # so the plan can never use more solar than the judge allows.
                cons.effective_solar[h] *= float(d.factor)
            elif d.directive_type == "minimum_battery_reserve":
                cons.min_energy[h] = max(cons.min_energy[h], float(d.minimum_energy_kwh))
            elif d.directive_type == "no_charge_window":
                cons.charge_max[h] = 0.0
            elif d.directive_type == "no_discharge_window":
                cons.discharge_max[h] = 0.0
            elif d.directive_type == "max_grid_window":
                cons.grid_max[h] = min(cons.grid_max[h], float(d.max_grid_kwh))
    return cons
