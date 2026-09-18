"""GridWise LLM-assisted energy optimizer - HTTP API.

    GET  /health            -> {"status": "ok"}
    POST /optimize-energy   -> directive interpretation + optimal 24-hour plan

Pipeline: request validation -> LLM interpreter -> deterministic guardrails
-> constraint builder -> LP optimizer -> plan builder -> final validator.
"""
from __future__ import annotations

import logging
import time
from itertools import combinations
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import load_settings
from .directives import Directive, build_constraints, hour_span
from .llm import NoteInterpreter
from .optimizer import build_plan, compute_totals, solve_lp
from .schemas import HealthResponse, OptimizeResponse, Scenario
from .validator import validate_plan

settings = load_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gridwise.api")

app = FastAPI(title="GridWise LLM Energy Optimizer", version="1.0.0", docs_url=None, redoc_url=None)
interpreter = NoteInterpreter(settings)

if settings.llm_enabled:
    log.info("LLM provider=%s model=%s key=%s", settings.llm_provider, settings.llm_model,
             "configured" if settings.llm_api_key else "MISSING")
else:
    log.warning("LLM_PROVIDER=none: every operator note will be treated as no_op")


# --------------------------------------------------------------------------- errors
def _compact(errors: list[dict[str, Any]]) -> list[str]:
    out = []
    for e in errors[:20]:
        loc = ".".join(str(p) for p in e.get("loc", ()) if p != "body")
        out.append(f"{loc}: {e.get('msg')}" if loc else str(e.get("msg")))
    return out


@app.exception_handler(RequestValidationError)
async def _bad_request(_: Request, exc: RequestValidationError) -> JSONResponse:
    # Malformed JSON or a structurally invalid request (Problem Statement 6.1 -> 400)
    return JSONResponse(status_code=400, content={"error": "bad_request", "detail": _compact(exc.errors())})


@app.exception_handler(Exception)
async def _internal_error(_: Request, exc: Exception) -> JSONResponse:
    # Controlled 500: type + message in the server log, no stack trace or secrets anywhere.
    log.error("unhandled error: %s: %s", type(exc).__name__, str(exc)[:300])
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "The service hit an unexpected error while building the plan."},
    )


# --------------------------------------------------------------------------- helpers
def _semantic_problems(scenario: Scenario) -> list[str]:
    b = scenario.battery
    problems = []
    if b.initial_energy_kwh > b.capacity_kwh + 1e-9:
        problems.append("battery.initial_energy_kwh exceeds capacity_kwh")
    if b.minimum_energy_kwh > b.capacity_kwh + 1e-9:
        problems.append("battery.minimum_energy_kwh exceeds capacity_kwh")
    if b.minimum_energy_kwh > b.initial_energy_kwh + 1e-9:
        problems.append(
            "battery.minimum_energy_kwh exceeds initial_energy_kwh: the battery would start below its reserve "
            "and could never end the day at its initial level"
        )
    return problems


def _solve_with_relaxation(scenario: Scenario, directives: list[Directive]):
    """Solve with every applicable directive; if infeasible, drop the fewest directives possible.

    Organizer scenarios are feasible under the ground-truth directives, so an
    infeasible model means an interpretation is wrong. Dropping the smallest
    set of directives keeps the plan valid instead of failing the request.
    """
    active = [d for d in directives if d.applies]
    for k in range(len(active), -1, -1):
        for subset in combinations(active, k):
            cons = build_constraints(scenario, list(subset))
            sol = solve_lp(cons)
            if sol is not None:
                dropped = [d for d in active if d not in subset]
                return cons, sol, dropped
    return None, None, active


def _plan_summary(directives, dropped, plan, cons, total_grid, total_cost, peak) -> str:
    applied = [d for d in directives if d.applies and d not in dropped]
    parts = []
    if applied:
        parts.append("Applied " + "; ".join(d.describe() for d in applied) + ".")
    else:
        parts.append("No operator directive changed the schedule.")
    n_noop = sum(1 for d in directives if not d.applies)
    if n_noop:
        parts.append(f"{n_noop} note(s) treated as no_op.")
    if dropped:
        parts.append(
            "Could not satisfy simultaneously and did not apply: "
            + "; ".join(d.describe() for d in dropped) + "."
        )
    charge = [p for p in plan if p["battery_action"] == "charge"]
    discharge = [p for p in plan if p["battery_action"] == "discharge"]
    if charge or discharge:
        parts.append(
            f"Charged {sum(p['battery_kwh'] for p in charge):g} kWh in {hour_span([p['hour'] for p in charge])} "
            f"and discharged {sum(p['battery_kwh'] for p in discharge):g} kWh in "
            f"{hour_span([p['hour'] for p in discharge])}, using solar first and returning the battery to "
            f"{cons.initial_energy:g} kWh."
        )
    else:
        parts.append("Battery stays idle; solar is used first and the grid covers the rest.")
    parts.append(f"Total grid {total_grid:g} kWh, cost {total_cost:g} BDT, peak {peak:g} kWh.")
    return " ".join(parts)


# --------------------------------------------------------------------------- endpoints
@app.get("/health", response_model=HealthResponse)
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/optimize-energy", response_model=OptimizeResponse)
async def optimize_energy(scenario: Scenario):
    t0 = time.perf_counter()

    problems = _semantic_problems(scenario)
    if problems:
        return JSONResponse(status_code=422, content={"error": "unprocessable_scenario", "detail": problems})

    daylight = tuple(h.hour for h in scenario.hours if h.solar_kwh > 0)
    directives, llm_info = await interpreter.interpret(scenario.operator_notes, scenario.battery, daylight)
    t_llm = time.perf_counter()

    cons, sol, dropped = _solve_with_relaxation(scenario, directives)
    if sol is None:
        return JSONResponse(
            status_code=422,
            content={"error": "unprocessable_scenario", "detail": ["no feasible 24-hour schedule exists for this input"]},
        )

    plan = build_plan(cons, sol)
    total_grid, total_cost, peak = compute_totals(plan, cons.tariff)
    violations = validate_plan(cons, plan, tol=5e-3)
    if violations:  # should never happen; logged so it is visible in testing
        log.warning("final validator flagged %s: %s", scenario.scenario_id, violations[:5])

    response = {
        "scenario_id": scenario.scenario_id,
        "directive_interpretation": [d.to_interpretation() for d in directives],
        "hourly_plan": plan,
        "total_grid_kwh": total_grid,
        "total_cost_bdt": total_cost,
        "peak_grid_kwh": peak,
        "plan_summary": _plan_summary(directives, dropped, plan, cons, total_grid, total_cost, peak),
    }
    log.info(
        "scenario=%s notes=%d applied=%d dropped=%d unresolved=%s cost=%.2f llm_ms=%d total_ms=%d attempts=%s cached=%s",
        scenario.scenario_id, len(directives), sum(d.applies for d in directives) - len(dropped), len(dropped),
        llm_info.get("unresolved", []), total_cost, (t_llm - t0) * 1000, (time.perf_counter() - t0) * 1000,
        llm_info.get("attempts"), llm_info.get("cached"),
    )
    return response
