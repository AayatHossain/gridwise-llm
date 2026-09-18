"""Deterministic guardrails: raw LLM JSON -> validated ``Directive`` objects.

The model output is treated as untrusted data (Problem Statement section 08).
Every check here is plain code. A note whose entry fails any check is
reported as an error for that note; the caller decides whether to retry the
model or fall back to ``no_op``. Nothing in this module ever invents a
directive the model did not produce, and nothing here changes demand, tariff
or battery parameters.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from .directives import DIRECTIVE_TYPES, Directive, hour_span
from .schemas import Battery

MAX_EXPLANATION_CHARS = 300


class NoteError(Exception):
    """Validation failure attributable to one note."""

    def __init__(self, note_index: int, message: str):
        super().__init__(f"note {note_index}: {message}")
        self.note_index = note_index
        self.message = message


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if s.lstrip("-").isdigit():
            return int(s)
    raise ValueError(f"{name} must be an integer")


def _as_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    if isinstance(value, (int, float)):
        v = float(value)
    elif isinstance(value, str):
        try:
            v = float(value.strip())
        except ValueError:
            raise ValueError(f"{name} must be a number") from None
    else:
        raise ValueError(f"{name} is missing or not a number")
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return v


def expand_window(start: int, end: int) -> list[int]:
    """Whole-hour window, start inclusive, end exclusive (1 PM-3 PM -> [13, 14]).

    ``end`` may be 24 (window runs to midnight). A window that crosses midnight
    (start 22, end 2) expands to [22, 23, 0, 1].
    """
    if not 0 <= start <= 23:
        raise ValueError("start_hour must be within 0..23")
    if not 1 <= end <= 24:
        raise ValueError("end_hour must be within 1..24")
    if end > start:
        return list(range(start, end))
    if end == start:
        raise ValueError("start_hour equals end_hour: empty window")
    return list(range(start, 24)) + list(range(0, end))


def normalize_hours(hours_field: Any, start: Any, end: Any) -> tuple[int, ...]:
    """Unique integers 0..23 in ascending order (Problem Statement 5.1)."""
    if isinstance(hours_field, list) and hours_field:
        hours = sorted({_as_int(h, "hours[]") for h in hours_field})
        if hours[0] < 0 or hours[-1] > 23:
            raise ValueError("hours must be within 0..23")
        return tuple(hours)
    if start is None or end is None:
        raise ValueError("time window missing (start_hour / end_hour)")
    window = expand_window(_as_int(start, "start_hour"), _as_int(end, "end_hour"))
    return tuple(sorted(set(window)))


def _clean_explanation(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return " ".join(value.split())[:MAX_EXPLANATION_CHARS]
    return ""


def default_explanation(d: Directive) -> str:
    span = hour_span(d.hours)
    t = d.directive_type
    if t == "no_op":
        return "This note does not affect the 24-hour energy schedule."
    if t == "solar_reduction":
        return f"Usable solar is reduced to {d.factor:g} of the forecast during {span}."
    if t == "minimum_battery_reserve":
        return f"Battery energy must stay at or above {d.minimum_energy_kwh:g} kWh during {span}."
    if t == "no_charge_window":
        return f"Battery charging is not allowed during {span}."
    if t == "no_discharge_window":
        return f"Battery discharging is not allowed during {span}."
    return f"Grid import is capped at {d.max_grid_kwh:g} kWh per hour during {span}."


def validate_entry(item: Any, n_notes: int, battery: Battery) -> Directive:
    """Validate one interpretation entry. Raises NoteError / ValueError."""
    if not isinstance(item, dict):
        raise ValueError("interpretation entry is not an object")
    idx = _as_int(item.get("note_index"), "note_index")
    if not 0 <= idx < n_notes:
        raise ValueError(f"note_index {idx} does not match any operator note")

    dtype = item.get("directive_type")
    if not isinstance(dtype, str):
        raise NoteError(idx, "directive_type is missing")
    dtype = dtype.strip().lower()
    if dtype not in DIRECTIVE_TYPES:
        raise NoteError(idx, f"unsupported directive_type '{dtype}'")

    explanation = _clean_explanation(item.get("explanation"))
    if dtype == "no_op":
        d = Directive(note_index=idx, directive_type="no_op", explanation=explanation)
        return d if d.explanation else replace(d, explanation=default_explanation(d))

    try:
        hours = normalize_hours(item.get("hours"), item.get("start_hour"), item.get("end_hour"))
        factor = minimum = max_grid = None
        if dtype == "solar_reduction":
            factor = _as_number(item.get("factor"), "factor")
            if 1.0 < factor <= 100.0:
                # The model wrote a percentage; the contract wants the remaining fraction.
                factor = factor / 100.0
            if not 0.0 <= factor <= 1.0:
                raise ValueError("factor must be within 0..1")
            factor = round(factor, 6)
        elif dtype == "minimum_battery_reserve":
            minimum = _as_number(item.get("minimum_energy_kwh"), "minimum_energy_kwh")
            if minimum < 0:
                raise ValueError("minimum_energy_kwh must be non-negative")
            if minimum > battery.capacity_kwh + 1e-9:
                raise ValueError(
                    f"minimum_energy_kwh {minimum:g} exceeds battery capacity {battery.capacity_kwh:g}"
                )
            minimum = round(minimum, 6)
        elif dtype == "max_grid_window":
            max_grid = _as_number(item.get("max_grid_kwh"), "max_grid_kwh")
            if max_grid < 0:
                raise ValueError("max_grid_kwh must be non-negative")
            max_grid = round(max_grid, 6)
    except ValueError as exc:
        raise NoteError(idx, str(exc)) from None

    d = Directive(
        note_index=idx,
        directive_type=dtype,
        hours=hours,
        factor=factor,
        minimum_energy_kwh=minimum,
        max_grid_kwh=max_grid,
        explanation=explanation,
    )
    return d if d.explanation else replace(d, explanation=default_explanation(d))


def validate_llm_output(
    raw: Any,
    notes: list[str],
    battery: Battery,
    daylight_hours: tuple[int, ...] | None = None,
) -> tuple[dict[int, Directive], dict[int, str]]:
    """Validate a whole model response.

    Returns ``(valid, errors)``: validated directives keyed by note index, and
    an error message for every note that has no valid entry. Every note ends
    up in exactly one of the two dictionaries.

    When ``daylight_hours`` is given, a solar_reduction whose window has no
    solar at all is reported as an error so the caller can ask the model to
    re-check AM/PM; the code never moves the window itself.
    """
    n = len(notes)
    valid: dict[int, Directive] = {}
    errors: dict[int, str] = {}

    entries = raw.get("directives") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        return valid, {i: "model output is not an object with a 'directives' array" for i in range(n)}

    seen: set[int] = set()
    for item in entries:
        try:
            d = validate_entry(item, n, battery)
        except NoteError as exc:
            errors[exc.note_index] = exc.message
            continue
        except (ValueError, TypeError):
            continue  # cannot even tell which note this entry belongs to
        if (
            daylight_hours is not None
            and d.directive_type == "solar_reduction"
            and not set(d.hours) & set(daylight_hours)
        ):
            errors[d.note_index] = (
                f"the solar window {hour_span(d.hours)} has no solar in this forecast "
                f"(daylight is {hour_span(daylight_hours)}); re-check AM/PM"
            )
            continue
        if d.note_index in seen:
            errors[d.note_index] = "more than one interpretation entry returned for this note"
            valid.pop(d.note_index, None)
            continue
        seen.add(d.note_index)
        valid[d.note_index] = d
        errors.pop(d.note_index, None)

    for i in range(n):
        if i not in valid and i not in errors:
            errors[i] = "no interpretation entry returned for this note"
    return valid, errors
