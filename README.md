# GridWise LLM — Smart Campus Energy Optimizer

BUP CSE Fest 2026 · Hackathon · Online Preliminary — *LLM-Assisted Operator Directive Interpretation*

One HTTP service that reads a 24-hour campus energy scenario plus 1–3 natural-language operator notes, interprets the notes with a language model, validates that interpretation with deterministic guardrails, and returns the **minimum-cost valid 24-hour schedule** computed by an exact linear-programming optimizer.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Readiness: `{"status": "ok"}` |
| `POST /optimize-energy` | Directive interpretation + optimal hourly plan (schema exactly as in the Problem Statement) |

## Contents

1. [Quickstart (local, from a clean machine)](#1-quickstart-local-from-a-clean-machine)
2. [Configuration (environment variables)](#2-configuration-environment-variables)
3. [Docker fallback](#3-docker-fallback)
4. [Testing](#4-testing)
5. [Architecture](#5-architecture)
6. [API examples](#6-api-examples)
7. [Dependencies and credits](#7-dependencies-and-credits)
8. [Known limitations](#8-known-limitations)
9. [Security and secret handling](#9-security-and-secret-handling)

---

## 1. Quickstart (local, from a clean machine)

Requirements: Python 3.11+ (3.12 recommended), `pip`, an OpenAI API key.

```bash
git clone https://github.com/AayatHossain/gridwise-llm.git gridwise-llm
cd gridwise-llm

python -m venv .venv
# Linux/macOS:            source .venv/bin/activate
# Windows (PowerShell):   .venv\Scripts\Activate.ps1

pip install -r requirements.txt

# language model credentials (never committed; see .env.example for every variable)
export OPENAI_API_KEY=sk-...        # PowerShell:  $env:OPENAI_API_KEY="sk-..."

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
curl -s http://localhost:8000/health
# {"status":"ok"}

curl -s -X POST http://localhost:8000/optimize-energy \
     -H "Content-Type: application/json" \
     -d @data/sample_request.json
```

Run every public sample case against the running service and score it the way the judge does:

```bash
python scripts/run_public_samples.py --base-url http://localhost:8000
```

Expected result (with `OPENAI_API_KEY` set):

```
SAMPLE-01  HTTP 200    2544 ms  OK    interp=ok  plan=valid  cost=38365.00  ref=38365.00  quality=1.000
...
SUMMARY
  HTTP 200            10/10
  interpretation      10/10
  valid plans         10/10
  optimization score  10.00 / 10   (mean quality 1.000)
  latency             p50 1814 ms   p95 2267 ms   max 2544 ms   -> 3/3 latency points
```

The script exits with code 0 only when every case is interpreted correctly and every plan is valid.

## 2. Configuration (environment variables)

All configuration is read from environment variables. Copy `.env.example` to `.env` for local use (`.env` is git-ignored) or set the variables in your hosting platform.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` (OpenAI or any OpenAI-compatible endpoint), `anthropic`, or `none` (offline: every note becomes `no_op`) |
| `LLM_MODEL` | `gpt-4.1` | Model id. `gpt-4.1-mini` is ~5× cheaper and also passed every public and paraphrase check |
| `OPENAI_API_KEY` | — | **Required** for `LLM_PROVIDER=openai`. `LLM_API_KEY` is accepted as an alias |
| `LLM_BASE_URL` | — | Optional OpenAI-compatible base URL (Groq, Gemini, OpenRouter, Ollama, …) |
| `LLM_FALLBACK_MODEL` | — | Optional second model tried when the primary call fails |
| `ANTHROPIC_API_KEY` | — | Only for `LLM_PROVIDER=anthropic` (default model `claude-opus-5`) |
| `LLM_EFFORT` | `low` | Anthropic only: `low` / `medium` / `high` / `none` |
| `LLM_TIMEOUT_SECONDS` | `20` | Timeout of one model call |
| `LLM_TOTAL_BUDGET_SECONDS` | `22` | Total interpretation budget per request (judge timeout is 30 s) |
| `LLM_CACHE_SIZE` | `512` | Cached interpretations keyed by note text + battery capacity (`0` disables) |
| `PORT` | `8000` | Listening port |
| `LOG_LEVEL` | `INFO` | Python log level |

**Model / provider used for the submission:** OpenAI `gpt-4.1` via the official `openai` Python SDK (chat completions with a strict JSON schema response format).

## 3. Docker fallback

The image contains no credentials; the API key is injected at run time.

```bash
docker pull ghcr.io/aayathossain/gridwise-llm:latest
docker run --rm -p 8000:8000 -e OPENAI_API_KEY=sk-... ghcr.io/aayathossain/gridwise-llm:latest

curl -s http://localhost:8000/health        # {"status":"ok"}
python scripts/run_public_samples.py --base-url http://localhost:8000
```

Build locally instead:

```bash
docker build -t gridwise-llm .
docker run --rm -p 8000:8000 -e OPENAI_API_KEY=sk-... gridwise-llm
```

Details: base image `python:3.12-slim`, runs as a non-root user, binds `0.0.0.0`, exposes port `8000` (override with `-e PORT=...` and matching `-p`), built-in `HEALTHCHECK` on `/health`. `.github/workflows/docker-publish.yml` builds and pushes the image to GitHub Container Registry on every push to `main` and smoke-tests `/health` on the pushed image.

## 4. Testing

**Offline tests — no API key, no network, no cost (33 tests):**

```bash
pip install -r requirements-dev.txt
python -m pytest
```

They feed the organizer's expected interpretation for each public case through a fake model, so guardrails, constraint building, the LP optimizer, plan building, the final validator and the HTTP contract are all checked against the reference cost and the GridWise rules; they also cover 400/422 handling, the no-LLM safe path and guardrail rejection cases.

**End-to-end with the real model:** `python scripts/run_public_samples.py --base-url <URL>` (section 1). Each case is one model call (≈ USD 0.004 with `gpt-4.1`).

## 5. Architecture

```
POST /optimize-energy
   │
   ▼
[1] Request validation (Pydantic)        400 malformed / structurally invalid · 422 semantically impossible
   │
   ▼
[2] LLM interpreter  (app/llm.py)        one chat-completion call carrying all notes, strict JSON schema
   │
   ▼
[3] Guardrail validator (app/guardrails.py)   deterministic; rejects anything outside the contract
   │       └── rejected note → one repair call → still invalid → that note is reported as no_op
   ▼
[4] Constraint builder (app/directives.py)    directives → per-hour limits (Problem Statement 5.3)
   │
   ▼
[5] LP optimizer (app/optimizer.py)      scipy / HiGHS, exact minimum grid cost
   │       └── infeasible → drop the fewest directives possible → physics-only → 422
   ▼
[6] Plan builder                         rounding, energy balance recomputed on returned numbers, end-of-day = initial
   │
   ▼
[7] Final validator (app/validator.py)   replays the plan the way the judge does
   │
   ▼
Response JSON (totals recomputed from the returned hourly_plan)
```

**LLM role.** The model is the mandatory interpretation step: it receives the battery parameters and the numbered notes and must return, per note, `directive_type` (one of the 6 published values), `start_hour`, `end_hour` (exclusive), the numeric field for that type and a short explanation. It converts phrasing such as "80% reduction" → `factor 0.2`, "half of capacity" → absolute kWh, "6 PM until 9 PM" → hours 18–20, and marks unrelated notes `no_op`. It never sees demand or tariff data and cannot change them.

**Guardrails (deterministic, `app/guardrails.py`).** Every model answer is untrusted until it passes: exactly one entry per note, valid `note_index`, `directive_type` in the allowed set, hours expanded by code from `start_hour`/`end_hour` (end-exclusive, midnight wrap, `24` = end of day) into unique ascending integers 0–23, `factor` in [0, 1] (a percentage written as `20` is normalised to `0.2`), reserve finite and ≤ battery capacity, grid cap finite and non-negative. Failures are sent back to the model once with the reason; if still invalid the note becomes `no_op` with an explanation — the service never invents a directive and never crashes on bad model output. A solar window that contains no daylight (an AM/PM slip such as "from one until three") is also sent back for re-interpretation together with the daylight hours taken from the request's own solar forecast; the code never moves the window itself.

**Optimizer (`app/optimizer.py`).** Per hour: `g` grid import, `s` solar used, `b` net battery flow (positive = charge, negative = discharge, so exactly one action per hour). Constraints: `g + s − b = demand`; `min_energy[h] ≤ E0 + Σb ≤ capacity` (reserve directives raise `min_energy`); `Σb = 0` (end-of-day neutrality); bounds `0 ≤ g ≤ grid cap`, `0 ≤ s ≤ effective solar`, `−discharge_max ≤ b ≤ charge_max` (no-charge / no-discharge windows set the bound to 0). Objective: minimise `Σ tariff·g`. Because there is no round-trip loss and no export the problem is exactly linear, so the LP optimum is the true minimum; it matches the organizer reference cost on all 10 public cases. Solve time ≈ 2 ms.

**Reliability.** Async FastAPI; one model call per request (p95 ≈ 2 s measured); 20 s per-call timeout inside a 22 s interpretation budget so every response stays under the 30 s judge limit; authentication errors are not retried; interpretations are cached; the LP and validator never raise into the response path; unexpected exceptions become a controlled 500 without stack traces.

## 6. API examples

Request (`data/sample_request.json`, abbreviated):

```json
{
  "scenario_id": "SAMPLE-01",
  "operator_notes": [
    "Facilities will wash the rooftop solar panels from noon until 2 PM. During cleaning, usable solar should be treated as roughly 25% of the forecast.",
    "The sports office moved next month's registration deadline."
  ],
  "hours": [ {"hour": 0, "demand_kwh": 90, "solar_kwh": 0, "tariff_bdt_per_kwh": 6}, "... 23 more ..." ],
  "battery": { "capacity_kwh": 220, "initial_energy_kwh": 110, "minimum_energy_kwh": 40,
               "max_charge_kwh_per_hour": 50, "max_discharge_kwh_per_hour": 50 }
}
```

Response (abbreviated):

```json
{
  "scenario_id": "SAMPLE-01",
  "directive_interpretation": [
    { "note_index": 0, "applies": true, "directive_type": "solar_reduction",
      "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
      "explanation": "Panel cleaning reduces usable solar to 25% from noon to 2 PM." },
    { "note_index": 1, "applies": false, "directive_type": "no_op",
      "structured_adjustment": null,
      "explanation": "Registration deadline change does not affect today's energy schedule." }
  ],
  "hourly_plan": [
    { "hour": 0, "grid_kwh": 90.0, "solar_used_kwh": 0.0, "battery_action": "idle",
      "battery_kwh": 0.0, "battery_energy_after_kwh": 110.0 },
    "... 23 more ..."
  ],
  "total_grid_kwh": 2692.5,
  "total_cost_bdt": 38365.0,
  "peak_grid_kwh": 187.5,
  "plan_summary": "Applied solar reduced to 0.25x in hours 12-13. 1 note(s) treated as no_op. Charged ... returning the battery to 110 kWh. Total grid 2692.5 kWh, cost 38365 BDT, peak 187.5 kWh."
}
```

HTTP codes: `200` success · `400` malformed JSON / structurally invalid request (`{"error":"bad_request","detail":[...]}`) · `422` well-formed but impossible scenario, e.g. reserve above initial energy (`{"error":"unprocessable_scenario"}`) · `500` controlled internal error (`{"error":"internal_error"}`, no stack trace).

## 7. Dependencies and credits

| Component | Use |
|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/) | HTTP service |
| [Pydantic v2](https://docs.pydantic.dev/) | Request/response validation |
| [SciPy](https://scipy.org/) (`linprog`, HiGHS) + [NumPy](https://numpy.org/) | Linear-programming optimizer |
| [openai](https://github.com/openai/openai-python) Python SDK | Operator-note interpretation (OpenAI `gpt-4.1`) |
| [anthropic](https://github.com/anthropics/anthropic-sdk-python) Python SDK | Optional alternative provider |
| pytest, httpx | Offline tests |
| Docker, GitHub Actions, GitHub Container Registry | Fallback image build and publishing |

AI coding assistance (Claude Code) was used while developing this repository; the architecture, prompt design, guardrails, optimisation model and tests were designed and verified by the team. Public sample cases are the organizer's `BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json` (copied to `data/`).

## 8. Known limitations

- Each operator note maps to exactly one directive, as the Problem Statement specifies; a single note that combines two constraints would be interpreted as one of them.
- Overlapping directives of the same type are merged conservatively: solar factors multiply, reserves take the maximum, grid caps take the minimum.
- If the interpreted directives are mutually infeasible (only possible when an interpretation is wrong — organizer scenarios are feasible under the ground truth), the service drops the fewest directives needed to produce a valid plan and says so in `plan_summary`; `directive_interpretation` still reports what the model extracted.
- The service depends on the configured model provider being reachable; if every model call fails within the time budget the request still returns `200` with a valid physics-only plan and `no_op` interpretations, which loses interpretation credit for that case but never fails the request.
- Battery efficiency losses, grid export and sub-hour scheduling are outside the challenge and not modelled.

## 9. Security and secret handling

- No API keys, tokens or `.env` files are committed; `.gitignore` and `.dockerignore` exclude them and `.env.example` lists variable names only.
- The Docker image contains no baked-in credentials.
- Logs record scenario id, counts, cost, timings and exception types/messages — never the API key, the raw prompt or stack traces. API error responses contain no internal details.
- Only the synthetic challenge data supplied by the harness is used; no live campus, utility, billing or personal data.
