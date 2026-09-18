"""Offline tests: no API key, no network, no LLM cost.

A fake interpreter returns the organizer's expected interpretation for each
public case, so everything downstream of the model (guardrails, constraint
builder, LP optimizer, plan builder, final validator, API contract) is
exercised exactly as the judge would.

Run:  python -m pytest
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ["LLM_PROVIDER"] = "none"  # must be set before app.main is imported

from fastapi.testclient import TestClient  # noqa: E402

import app.main as main  # noqa: E402
from app.directives import build_constraints, directive_from_interpretation  # noqa: E402
from app.guardrails import expand_window, validate_llm_output  # noqa: E402
from app.schemas import Battery, Scenario  # noqa: E402
from app.validator import validate_plan, validate_totals  # noqa: E402

CASES = json.loads((Path(__file__).resolve().parent.parent / "data" / "public_sample_cases.json").read_text("utf-8"))["cases"]
TOL = 0.01


class FakeInterpreter:
    """Stands in for the LLM: returns the expected interpretation for known notes."""

    def __init__(self):
        self.by_notes = {tuple(c["input"]["operator_notes"]): c["expected_output"]["directive_interpretation"] for c in CASES}

    async def interpret(self, notes, battery, daylight_hours=()):
        expected = self.by_notes[tuple(notes)]
        return [directive_from_interpretation(e) for e in expected], {"cached": False, "attempts": 0}


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(main, "interpreter", FakeInterpreter())
    with TestClient(main.app) as c:
        yield c


def _interp_matches(got: dict, exp: dict) -> bool:
    if got["applies"] != exp["applies"] or got["directive_type"] != exp["directive_type"]:
        return False
    ga, ea = got["structured_adjustment"], exp["structured_adjustment"]
    if ea is None:
        return ga is None
    if ga is None or ga.get("hours") != ea.get("hours"):
        return False
    return all(abs(float(ga.get(k, 1e9)) - float(v)) <= TOL for k, v in ea.items() if k != "hours")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_public_case_is_valid_and_optimal(client, case):
    r = client.post("/optimize-energy", json=case["input"])
    assert r.status_code == 200, r.text
    body = r.json()
    exp = case["expected_output"]

    assert body["scenario_id"] == case["input"]["scenario_id"]
    assert set(body) >= {"scenario_id", "directive_interpretation", "hourly_plan", "total_grid_kwh",
                         "total_cost_bdt", "peak_grid_kwh", "plan_summary"}

    # interpretation: one entry per note, in order, matching ground truth
    assert [d["note_index"] for d in body["directive_interpretation"]] == list(range(len(case["input"]["operator_notes"])))
    for got, want in zip(body["directive_interpretation"], exp["directive_interpretation"]):
        assert _interp_matches(got, want), (got, want)
        assert got["applies"] == (got["directive_type"] != "no_op")
        assert (got["structured_adjustment"] is None) == (got["directive_type"] == "no_op")

    # plan: replay against ground-truth directives, the way the judge does
    scenario = Scenario.model_validate(case["input"])
    truth = [directive_from_interpretation(e) for e in exp["directive_interpretation"]]
    cons = build_constraints(scenario, truth)
    assert validate_plan(cons, body["hourly_plan"], tol=TOL) == []
    assert validate_totals(body["hourly_plan"], cons.tariff, body["total_grid_kwh"], body["total_cost_bdt"], body["peak_grid_kwh"]) == []

    # cost: equal to the organizer optimum
    assert abs(body["total_cost_bdt"] - exp["total_cost_bdt"]) <= TOL
    assert all(p["battery_action"] in ("charge", "discharge", "idle") for p in body["hourly_plan"])
    assert isinstance(body["plan_summary"], str) and body["plan_summary"]


def test_malformed_json_is_400(client):
    r = client.post("/optimize-energy", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json()["error"] == "bad_request"


@pytest.mark.parametrize("mutate", [
    lambda req: req.pop("hours"),
    lambda req: req["hours"].pop(),
    lambda req: req["hours"].__setitem__(0, {**req["hours"][0], "hour": 5}),  # duplicate hour
    lambda req: req.__setitem__("operator_notes", []),
    lambda req: req.__setitem__("operator_notes", ["a", "b", "c", "d"]),
    lambda req: req.__setitem__("operator_notes", ["   "]),
    lambda req: req["hours"].__setitem__(3, {**req["hours"][3], "demand_kwh": -1}),
    lambda req: req["battery"].pop("capacity_kwh"),
    lambda req: req.__setitem__("scenario_id", ""),
])
def test_structurally_invalid_is_400(client, mutate):
    req = json.loads(json.dumps(CASES[0]["input"]))
    mutate(req)
    assert client.post("/optimize-energy", json=req).status_code == 400


def test_semantically_impossible_battery_is_422(client):
    req = json.loads(json.dumps(CASES[0]["input"]))
    req["battery"]["minimum_energy_kwh"] = req["battery"]["initial_energy_kwh"] + 10
    r = client.post("/optimize-energy", json=req)
    assert r.status_code == 422 and r.json()["error"] == "unprocessable_scenario"


def test_no_llm_configured_still_returns_valid_plan():
    # main.interpreter is the real NoteInterpreter with provider=none -> every note is no_op
    with TestClient(main.app) as c:
        r = c.post("/optimize-energy", json=CASES[0]["input"])
    assert r.status_code == 200
    body = r.json()
    assert all(d["directive_type"] == "no_op" and d["applies"] is False and d["structured_adjustment"] is None
               for d in body["directive_interpretation"])
    scenario = Scenario.model_validate(CASES[0]["input"])
    assert validate_plan(build_constraints(scenario, []), body["hourly_plan"], tol=TOL) == []


# ------------------------------------------------------------------ guardrails
BATTERY = Battery(capacity_kwh=220, initial_energy_kwh=110, minimum_energy_kwh=40,
                  max_charge_kwh_per_hour=50, max_discharge_kwh_per_hour=50)


def test_window_expansion_is_end_exclusive_and_wraps():
    assert expand_window(13, 15) == [13, 14]
    assert expand_window(22, 2) == [22, 23, 0, 1]
    assert expand_window(0, 24) == list(range(24))
    with pytest.raises(ValueError):
        expand_window(5, 5)


def test_guardrails_accept_valid_entries_and_normalise_percent_factor():
    raw = {"directives": [
        {"note_index": 0, "directive_type": "solar_reduction", "start_hour": 13, "end_hour": 15, "factor": 20},
        {"note_index": 1, "directive_type": "no_charge_window", "hours": [3, 1, 2]},
        {"note_index": 2, "directive_type": "no_op"},
    ]}
    valid, errors = validate_llm_output(raw, ["a", "b", "c"], BATTERY)
    assert errors == {}
    assert valid[0].factor == 0.2 and valid[0].hours == (13, 14)
    assert valid[1].hours == (1, 2, 3)
    assert valid[2].applies is False and valid[2].structured_adjustment() is None


@pytest.mark.parametrize("entry", [
    {"note_index": 0, "directive_type": "curtail_load", "start_hour": 1, "end_hour": 2},          # unsupported type
    {"note_index": 0, "directive_type": "minimum_battery_reserve", "start_hour": 1, "end_hour": 2,
     "minimum_energy_kwh": 9999},                                                                  # reserve > capacity
    {"note_index": 0, "directive_type": "max_grid_window", "start_hour": 1, "end_hour": 2, "max_grid_kwh": -5},
    {"note_index": 0, "directive_type": "solar_reduction", "start_hour": 1, "end_hour": 2, "factor": 250},
    {"note_index": 0, "directive_type": "no_charge_window"},                                        # no window
    {"note_index": 0, "directive_type": "no_discharge_window", "start_hour": 30, "end_hour": 31},
])
def test_guardrails_reject_bad_entries_without_inventing(entry):
    valid, errors = validate_llm_output({"directives": [entry]}, ["x"], BATTERY)
    assert valid == {} and 0 in errors


def test_guardrails_flag_duplicates_missing_and_garbage():
    valid, errors = validate_llm_output({"directives": [{"note_index": 0, "directive_type": "no_op"},
                                                        {"note_index": 0, "directive_type": "no_op"}]}, ["x", "y"], BATTERY)
    assert valid == {} and set(errors) == {0, 1}
    valid, errors = validate_llm_output("garbage", ["x"], BATTERY)
    assert valid == {} and 0 in errors


def test_guardrails_send_dark_solar_window_back_for_repair():
    entry = {"note_index": 0, "directive_type": "solar_reduction", "start_hour": 1, "end_hour": 3, "factor": 0.2}
    valid, errors = validate_llm_output({"directives": [entry]}, ["x"], BATTERY, daylight_hours=tuple(range(6, 18)))
    assert valid == {} and "AM/PM" in errors[0]
    valid, errors = validate_llm_output({"directives": [entry]}, ["x"], BATTERY, daylight_hours=None)
    assert errors == {} and valid[0].hours == (1, 2)  # repair attempt: the model's answer stands
