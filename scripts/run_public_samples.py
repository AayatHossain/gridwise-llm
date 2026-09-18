#!/usr/bin/env python
"""Run the public sample cases against a running service and score them like the judge.

    python scripts/run_public_samples.py --base-url http://localhost:8000
    python scripts/run_public_samples.py --base-url https://your-deployment.example.com --only SAMPLE-03

For every case it POSTs case.input to /optimize-energy and reports:
  * HTTP status and latency
  * interpretation vs the public reference (applies, type, hours, numbers within 0.01)
  * hourly_plan replayed against the reference (ground-truth) directives and the GridWise rules
  * cost quality  = min(1, reference_cost / your_cost)   (0 when the plan is invalid)
Each case costs one LLM call (about USD 0.004 with gpt-4.1), so a full run is roughly USD 0.04.
Only the standard library is needed on the client side; the app package is imported for the replay checks.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.directives import build_constraints, directive_from_interpretation  # noqa: E402
from app.schemas import Scenario  # noqa: E402
from app.validator import validate_plan, validate_totals  # noqa: E402

TOL = 0.01


def post_json(url: str, payload: dict, timeout: float) -> tuple[int, dict | str, float]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            status = resp.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        status = exc.code
    except (urllib.error.URLError, TimeoutError) as exc:
        return 0, f"connection error: {exc}", (time.perf_counter() - t0) * 1000
    ms = (time.perf_counter() - t0) * 1000
    try:
        return status, json.loads(body), ms
    except json.JSONDecodeError:
        return status, body, ms


def interpretation_mismatches(got: list[dict], expected: list[dict]) -> list[str]:
    problems = []
    if [g.get("note_index") for g in got] != list(range(len(expected))):
        problems.append(f"note_index order {[g.get('note_index') for g in got]} != {list(range(len(expected)))}")
    for g, e in zip(got, expected):
        i = e["note_index"]
        if g.get("applies") != e["applies"] or g.get("directive_type") != e["directive_type"]:
            problems.append(f"note {i}: got {g.get('directive_type')} (applies={g.get('applies')}), "
                            f"expected {e['directive_type']}")
            continue
        ga, ea = g.get("structured_adjustment"), e["structured_adjustment"]
        if ea is None:
            if ga is not None:
                problems.append(f"note {i}: no_op must carry structured_adjustment null")
            continue
        if not isinstance(ga, dict):
            problems.append(f"note {i}: structured_adjustment missing")
            continue
        if ga.get("hours") != ea.get("hours"):
            problems.append(f"note {i}: hours {ga.get('hours')} != {ea.get('hours')}")
        for k, v in ea.items():
            if k == "hours":
                continue
            try:
                if abs(float(ga.get(k)) - float(v)) > TOL:
                    problems.append(f"note {i}: {k} {ga.get(k)} != {v}")
            except (TypeError, ValueError):
                problems.append(f"note {i}: {k} missing or not a number")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--cases", default=str(ROOT / "data" / "public_sample_cases.json"))
    ap.add_argument("--only", help="run a single case id, e.g. SAMPLE-03")
    ap.add_argument("--timeout", type=float, default=30.0, help="per-request timeout in seconds (judge uses 30)")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    try:
        with urllib.request.urlopen(f"{base}/health", timeout=args.timeout) as resp:
            health = json.loads(resp.read().decode("utf-8"))
        print(f"GET /health -> {resp.status} {health}")
        if health.get("status") != "ok":
            print("health check did not return status=ok"); return 2
    except Exception as exc:  # noqa: BLE001
        print(f"GET /health failed: {exc}"); return 2

    cases = json.loads(Path(args.cases).read_text("utf-8"))["cases"]
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]
        if not cases:
            print(f"no case with id {args.only}"); return 2

    latencies, quality = [], []
    n_interp = n_valid = n_http = 0
    for case in cases:
        exp = case["expected_output"]
        status, body, ms = post_json(f"{base}/optimize-energy", case["input"], args.timeout)
        latencies.append(ms)
        if status != 200 or not isinstance(body, dict):
            print(f"{case['id']}  HTTP {status}  {ms:6.0f} ms  FAIL  {str(body)[:120]}")
            quality.append(0.0)
            continue
        n_http += 1

        problems = interpretation_mismatches(body.get("directive_interpretation", []), exp["directive_interpretation"])
        scenario = Scenario.model_validate(case["input"])
        truth = [directive_from_interpretation(e) for e in exp["directive_interpretation"]]
        cons = build_constraints(scenario, truth)
        violations = validate_plan(cons, body.get("hourly_plan", []), tol=TOL)
        try:
            violations += validate_totals(body["hourly_plan"], cons.tariff, float(body["total_grid_kwh"]),
                                          float(body["total_cost_bdt"]), float(body["peak_grid_kwh"]))
        except (KeyError, TypeError, ValueError):
            violations.append("totals missing or not numeric")
        if body.get("scenario_id") != case["input"]["scenario_id"]:
            violations.append("scenario_id not echoed")

        cost = float(body.get("total_cost_bdt", 0) or 0)
        ref = float(exp["total_cost_bdt"])
        if violations:
            q = 0.0
        elif ref <= TOL and cost <= TOL:
            q = 1.0
        elif cost <= TOL:
            q = 0.0
        else:
            q = min(1.0, ref / cost)
        quality.append(q)
        n_interp += not problems
        n_valid += not violations
        flag = "OK  " if not problems and not violations else "FAIL"
        print(f"{case['id']}  HTTP {status}  {ms:6.0f} ms  {flag}  interp={'ok' if not problems else 'MISS'}  "
              f"plan={'valid' if not violations else 'INVALID'}  cost={cost:.2f}  ref={ref:.2f}  quality={q:.3f}")
        for p in problems:
            print(f"      interpretation: {p}")
        for v in violations[:5]:
            print(f"      plan: {v}")

    n = len(cases)
    latencies.sort()
    p95 = latencies[max(0, int(round(0.95 * n)) - 1)]
    print("\nSUMMARY")
    print(f"  HTTP 200            {n_http}/{n}")
    print(f"  interpretation      {n_interp}/{n}")
    print(f"  valid plans         {n_valid}/{n}")
    print(f"  optimization score  {10 * sum(quality) / n:.2f} / 10   (mean quality {sum(quality) / n:.3f})")
    print(f"  latency             p50 {latencies[n // 2]:.0f} ms   p95 {p95:.0f} ms   max {latencies[-1]:.0f} ms"
          f"   -> {'3/3' if p95 <= 5000 else '2/3' if p95 <= 15000 else '1/3' if p95 <= 30000 else '0/3'} latency points")
    return 0 if n_interp == n and n_valid == n else 1


if __name__ == "__main__":
    sys.exit(main())
