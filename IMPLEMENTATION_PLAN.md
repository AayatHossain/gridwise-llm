# GridWise LLM — Implementation Plan

BUP CSE Fest 2026 · Online Preliminary · LLM-Assisted Operator Directive Interpretation

Goal: a deployed HTTP API (`GET /health`, `POST /optimize-energy`) where an LLM reads operator notes, deterministic guardrails validate them, an exact LP optimizer produces the cheapest valid 24-hour schedule, and a final validator re-checks everything before the response leaves.

---

## 1. Architecture (mirrors the diagram in the Problem Statement)

```
HTTP request
   │
   ▼
[1] Request validation (Pydantic)        → 400 malformed / 422 semantically impossible
   │
   ▼
[2] LLM Interpreter (1 call, all notes)  → raw JSON, treated as untrusted
   │
   ▼
[3] Guardrail Validator (pure Python)    → validated Directive objects, or per-note errors
   │        └─ errors → 1 repair call → still bad → that note becomes no_op (controlled failure)
   ▼
[4] Constraint builder                   → per-hour arrays: effective_solar, min_energy, charge_max, discharge_max, grid_max
   │
   ▼
[5] LP Optimizer (scipy HiGHS)           → exact minimum-cost schedule
   │        └─ infeasible → drop directive subsets (max 8 LPs) → physics-only → else 422
   ▼
[6] Plan builder                         → round, recompute grid from rounded values, force end-of-day = initial
   │
   ▼
[7] Final Validator (same code the judge runs)  → log if anything fails
   │
   ▼
[8] Response JSON (totals recomputed from the plan itself)
```

Stack: Python 3.12, FastAPI + uvicorn, Pydantic v2, scipy (`linprog`, HiGHS), numpy, `anthropic` SDK (optionally `openai` SDK for any OpenAI-compatible provider: OpenAI, Groq, Gemini, OpenRouter, Ollama).

---

## 2. Module layout

```
gridwise-llm/
├── app/
│   ├── main.py         # FastAPI app, 2 endpoints, error handlers, pipeline orchestration
│   ├── config.py       # env vars → Settings
│   ├── schemas.py      # Pydantic request/response models
│   ├── llm.py          # prompt, JSON schema, provider backends, retry/deadline, cache
│   ├── guardrails.py   # deterministic validation of LLM JSON → Directive
│   ├── directives.py   # Directive dataclass + build_constraints()
│   ├── optimizer.py    # LP + plan builder + totals
│   └── validator.py    # replay checker (used by service AND test runner)
├── scripts/run_public_samples.py   # POSTs all 10 cases, scores like the judge
├── tests/test_pipeline.py          # offline tests with a fake LLM (no key needed)
├── data/public_sample_cases.json, data/sample_request.json
├── Dockerfile, .dockerignore, requirements.txt
├── .env.example, .gitignore
├── .github/workflows/docker-publish.yml   # builds & pushes image to GHCR (no local Docker needed)
└── README.md
```

---

## 3. Request validation (`schemas.py`) — API contract points

| Check | Response |
|---|---|
| Body not JSON / not an object | **400** `{"error":"bad_request","detail":[...]}` (override FastAPI's default 422 for validation errors) |
| `scenario_id` missing/empty; `operator_notes` not 1–3 non-empty strings; `hours` not exactly 24 entries covering 0–23 once; missing battery fields; negative/NaN/inf numbers | **400** |
| `initial_energy_kwh > capacity_kwh` or `minimum_energy_kwh > initial_energy_kwh` (battery could never end the day at its start level) | **422** `unprocessable_scenario` |
| Unknown extra fields | ignored |
| Any uncaught exception | **500** `{"error":"internal_error","detail":"..."}` — no stack trace, no secrets. Logs record only exception type + message |

Hours are sorted by `hour` after validation so the rest of the code indexes `[0..23]`.

---

## 4. LLM interpreter (`llm.py`) — 25 points live here

One call per request carrying all 1–3 notes (latency = 1 round trip). Temperature 0 / low effort. Response constrained to a JSON schema.

### What the model receives
- System prompt (frozen, cacheable): the 6 directive types with definitions and synonyms; conversion rules; ~6 few-shot examples in original wording (not the public sample sentences — the pack forbids hard-coding them).
- User message: the battery object (needed for "50% of capacity" → 100 kWh) + numbered notes + the required output shape.

### What the model returns (per note)
```json
{"note_index":0, "directive_type":"solar_reduction", "start_hour":13, "end_hour":15,
 "factor":0.2, "minimum_energy_kwh":null, "max_grid_kwh":null, "explanation":"..."}
```
Design choice: the model gives `start_hour` + `end_hour` (exclusive), not the hours list. Converting "6 PM" → 18 is easy for a model; applying the end-exclusive convention is where models slip, so our code expands `range(start, end)`. Handles `end=24` (whole day) and windows crossing midnight (22→2 = `[0,1,22,23]`, sorted ascending as required).

### Key prompt rules (each maps to a scored item)
- factor = fraction that remains: "drops to 20%" → 0.2, "80% reduction" → 0.2, "reduced by 30%" → 0.7, "half" → 0.5, "one-fifth" → 0.2, "a quarter" → 0.25
- reserve given as % of capacity → multiply by `capacity_kwh` from the request
- synonym lists: charger isolated/offline/disabled/circuit unavailable → `no_charge_window`; feeder/transformer/substation limit, intake/import/draw cap → `max_grid_window`; kW ≡ kWh per hour
- `no_op` for anything that isn't a solar/battery/grid constraint on this 24-hour schedule (bookings, menus, deadlines, next week/month, staffing) — and for energy remarks with no usable time window
- exactly 1 entry per note, same order, never merge/split notes, never invent numbers

### Backends
- `anthropic`: official SDK, `messages.create` with `output_config.format = json_schema` and `output_config.effort = low`; check `stop_reason == "refusal"` → treat as failure. If the API rejects the schema or effort param (400), automatically retry without it and parse JSON from text.
- `openai_compatible`: official `openai` SDK with `base_url` (OpenAI, Groq, Gemini's OpenAI endpoint, OpenRouter, Ollama), `response_format=json_object`, same fallback.
- Chosen by env: `LLM_PROVIDER`, `LLM_MODEL`, `ANTHROPIC_API_KEY` / `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_EFFORT`, `LLM_TIMEOUT_SECONDS`, `LLM_TOTAL_BUDGET_SECONDS`, optional `LLM_FALLBACK_MODEL`.

### Retry & time budget (judge timeout is 30 s)
```
deadline = now + 22 s
attempt 1: primary model
   → guardrails; if all notes valid → done
attempt 2 (only if ≥ 3 s left): same model, user message appended with
   "Your previous answer had these problems: note 1: reserve exceeds capacity ..."
   → keep attempt-1 notes that were already valid; only fill the gaps
attempt 3 (optional): LLM_FALLBACK_MODEL if configured
still invalid / provider down → that note = no_op with explanation
   "Could not be mapped to a supported directive; treated as no_op"
```
This is the "SAFE FAILURE" rule: never crash, never invent a directive type, always 200 with a valid plan.

### Cache
Dict keyed by `(tuple(notes), capacity_kwh)` → validated directives, LRU 512. Repeated hidden notes cost 0 ms.

### Model choice
Default `claude-opus-5` at effort `low` for accuracy; the sample runner prints per-request latency — if p95 > 5 s, switch to `claude-sonnet-5` (env var only, no code change). Haiku 4.5 works too (set `LLM_EFFORT=none`).

---

## 5. Guardrails (`guardrails.py`) — deterministic, per Problem Statement §08

For each returned entry:
1. `note_index` is an integer, in range, appears once. Duplicates → error for that note. Notes with no entry → error.
2. `directive_type` ∈ the 6 allowed strings (case-normalized). Anything else → error (never mapped to something "close").
3. `no_op` → `applies=false`, `structured_adjustment=null`.
4. Otherwise hours: prefer explicit `hours` list if the model gave one; else expand `start_hour`/`end_hour`. Result must be unique ints 0–23, ascending.
5. Numeric fields by type:
   - `factor`: finite; if in (1, 100] divide by 100 (model wrote a percent); must end in [0, 1]
   - `minimum_energy_kwh`: finite, ≥ 0, ≤ `capacity_kwh` (else error — never clamped, clamping would be inventing)
   - `max_grid_kwh`: finite, ≥ 0
6. `explanation`: whitespace-collapsed, ≤ 300 chars; if empty, generate one from the validated fields.
7. Returns `(valid: {idx: Directive}, errors: {idx: message})` — every note lands in exactly one.

Nothing here reads or changes demand/tariff/battery ("No invention" rule).

---

## 6. Constraint builder (`directives.py`) — §5.3 exactly

Start from base arrays, then per applied directive, per listed hour:

| Directive | Effect |
|---|---|
| `solar_reduction` | `effective_solar[h] *= factor` (overlaps multiply = most conservative, can never overuse solar) |
| `minimum_battery_reserve` | `min_energy[h] = max(base_min, directive_min)` |
| `no_charge_window` | `charge_max[h] = 0` |
| `no_discharge_window` | `discharge_max[h] = 0` |
| `max_grid_window` | `grid_max[h] = min(existing, cap)` |
| `no_op` | nothing |

---

## 7. Optimizer (`optimizer.py`) — exact LP

72 variables: for each hour `g[h]` grid, `s[h]` solar used, `b[h]` net battery flow (+charge / −discharge).

```
bounds:   0 ≤ g[h] ≤ grid_max[h]
          0 ≤ s[h] ≤ effective_solar[h]
          −discharge_max[h] ≤ b[h] ≤ charge_max[h]      ← no-charge/no-discharge windows are just 0 bounds
equal:    g[h] + s[h] − b[h] = demand[h]                  (24 rows, energy balance)
          Σ b[h] = 0                                      (end-of-day neutrality)
ineq:     min_energy[h] ≤ E0 + Σ_{k≤h} b[k] ≤ capacity   (48 rows)
min:      Σ tariff[h] · g[h]
```
`scipy.optimize.linprog(method="highs")`, ~5 ms. One net variable per hour guarantees exactly one battery action per hour (no charge+discharge in the same hour). Verified: matches the reference cost on all 10 public cases to 0.0000.

### Infeasible?
Judge cases are feasible under ground truth, so infeasibility means an interpretation is wrong. Try subsets of the applied directives from largest to smallest (≤ 8 LPs with 3 notes), take the first feasible; last resort physics-only (always feasible after the 422 checks). The interpretation array is still reported as extracted; `plan_summary` says which directive couldn't be applied.

### Plan builder
1. Round `b` to 6 decimals, kill |b| < 1e-9 → `idle`.
2. Absorb rounding drift so Σb = 0 exactly → final `battery_energy_after_kwh == initial`.
3. Clamp `s` to `[0, effective_solar]`, round.
4. Recompute `g = demand + b − s` from the rounded values, so the balance holds on the exact numbers returned; if a rounding artifact makes g < 0 by 1e-6, reduce `s` (or `b`) instead and set g = 0.
5. `battery_action` / `battery_kwh = |b|` / running `battery_energy_after_kwh`.
6. Totals from the returned plan: `total_grid = Σg`, `total_cost = Σ g·tariff`, `peak = max g`.

---

## 8. Final validator (`validator.py`)

Replays exactly what §11.3 lists, with tolerance 1e-3 (tighter than the judge's 0.01): 24 unique hours; finite non-negative values; action ∈ {charge, discharge, idle}; `battery_kwh = 0` when idle; charge/discharge ≤ per-hour limits (which are 0 inside windows); solar_used ≤ effective solar; grid ≤ cap; energy balance; battery transition consistency; min reserve ≤ E ≤ capacity; end-of-day = initial; totals match plan. Runs on every response (log a warning if it ever fails — it shouldn't) and is reused by the sample runner with the expected directives, exactly like the judge.

---

## 9. Endpoint behaviour summary

| | |
|---|---|
| `GET /health` | `{"status":"ok"}` immediately; does not depend on the LLM (must be ready < 60 s after start) |
| `POST /optimize-energy` | pipeline above; ~LLM latency + 10 ms |
| Concurrency | async FastAPI; the LLM call is awaited so repeated judge requests don't block each other |
| Logging | scenario_id, note count, applied/dropped counts, cost, ms, model name. Never the API key, never the raw prompt, never tracebacks |
| `plan_summary` | generated deterministically: applied directives, no_op count, charge/discharge hours and kWh, end battery level, totals |

---

## 10. Testing

- `tests/test_pipeline.py` (pytest, no API key): fake interpreter returns the expected interpretation for each of the 10 cases → assert cost within 0.01 of reference, plan valid under ground truth, totals consistent; guardrail unit tests (percent factor, bad type, wrap-around, duplicate index, reserve > capacity); 400 on malformed JSON; 422 on min > initial; LLM-unavailable path returns 200 with all `no_op`.
- `scripts/run_public_samples.py --base-url http://localhost:8000`: real end-to-end run with your LLM. Per case prints latency, interpretation ✔/✘ (type, hours, numbers within 0.01), plan validity under expected directives, cost ratio `min(1, ref/yours)`. Summary line = your approximate rubric score. Non-zero exit if anything fails.

---

## 11. Deployment & deliverables (Participant Guide compliance)

| Item | Plan |
|---|---|
| Repo | New private GitHub repo created after reveal; make public after deadline. `.gitignore` excludes `.env`; `.env.example` lists names only |
| Secrets | Only via env vars. Never in code, README, image, logs or responses |
| Docker | `python:3.12-slim`, `pip install`, copies `app/ data/ scripts/`, `EXPOSE 8000`, `CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}`, HEALTHCHECK on `/health` |
| Image without local Docker | GitHub Actions workflow builds on push and pushes `ghcr.io/<you>/gridwise-llm:latest` + `:sha`. After the deadline set the package visibility to public so judges can pull |
| Hosting | Railway or Render deploy straight from the Dockerfile (Render free tier sleeps — use a paid instance or an external uptime pinger); HF Spaces (Docker) also works (set `PORT=7860`). Set `ANTHROPIC_API_KEY` (or your provider's key) + `LLM_MODEL` in the platform's env settings. Test both endpoints from a phone/other network before submitting |
| README | Quickstart (clone → env names → `pip install -r requirements.txt` → `uvicorn ...` → curl `/health` → curl `/optimize-energy` with `data/sample_request.json` → `python scripts/run_public_samples.py`), Docker pull/run, model/provider, LLM role, guardrails, solver, dependencies credited (FastAPI, scipy, anthropic/openai SDKs, AI coding assistance), known limitations |
| Video (≤ 3 min) | 0:00 problem in 2 sentences → 0:30 the 6-box pipeline → 1:15 show the prompt + guardrail code → 1:45 LP formulation on one slide → 2:15 run sample script live, show cost matches → 2:45 how to run/Docker |

---

## 12. Rule-compliance checklist

- LLM is in the interpretation path and produces the structured directives the optimizer consumes — not just `plan_summary` ✔
- Deterministic guardrails after the LLM, never replacing it; no regex-only interpreter ✔
- Only the 6 published directive types; unsupported output → controlled `no_op`, never invented ✔
- `applies=false` only with `no_op` + `null`; every other directive `applies=true` with the exact shape ✔
- Hours unique/ascending 0–23; end-exclusive windows; factor = remaining fraction ✔
- All §09 physics + neutrality enforced in the LP and re-checked by the validator ✔
- Totals recomputed from the returned plan ✔
- 400/422/500 semantics; no secrets/stack traces exposed ✔
- Public phrases/IDs/values not hard-coded (few-shot examples are original) ✔
- No training/fine-tuning; no live data; dependencies credited ✔

---

## 13. Build order

1. Core modules: `directives.py`, `schemas.py`, `guardrails.py`, `optimizer.py`, `validator.py`
2. Offline tests with a fake LLM against all 10 public cases
3. `config.py`, `llm.py`, `main.py`
4. `scripts/run_public_samples.py` (real LLM end-to-end + latency)
5. Dockerfile, `.dockerignore`, GitHub Actions workflow, `.env.example`, `.gitignore`
6. README
7. Deploy, test from outside, record video
