"""LLM interpreter: operator notes -> structured directives.

The language model is the mandatory interpretation step: it reads every
operator note and emits one structured entry per note. Everything it returns
is treated as untrusted until ``guardrails.validate_llm_output`` accepts it.

Backends (selected by LLM_PROVIDER):
  * openai     - official ``openai`` SDK; works with OpenAI or any OpenAI-compatible
                 chat-completions endpoint via LLM_BASE_URL (Groq, Gemini, OpenRouter, Ollama)
  * anthropic  - official ``anthropic`` SDK with structured JSON output
  * none       - no model configured; every note becomes no_op (used by offline tests)

Failure policy (Problem Statement "SAFE FAILURE"): a bad or missing model
answer never crashes the service and never invents a directive. After the
retry budget is exhausted, the affected note is reported as ``no_op``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import OrderedDict
from typing import Any

from .config import Settings
from .directives import Directive, no_op
from .guardrails import validate_llm_output
from .schemas import Battery

log = logging.getLogger("gridwise.llm")

MIN_ATTEMPT_SECONDS = 3.0

SYSTEM_PROMPT = """You are the operator-note interpreter inside GridWise, a campus energy scheduling service.
The scheduler already has the next 24 hours of demand, solar forecast and grid tariff (hours 0-23).
Campus operators add short natural-language notes. Your only job is to convert EACH note into exactly one
structured directive, or no_op when the note does not change this 24-hour energy schedule.

DIRECTIVE TYPES (use these exact strings)
1. solar_reduction - usable rooftop solar is reduced during a time window.
   fields: start_hour, end_hour, factor
   factor = fraction of the normal solar forecast that REMAINS usable, between 0 and 1.
   "drops to 20%" -> 0.2 | "80% reduction" -> 0.2 | "reduced by 30%" -> 0.7 | "about half" -> 0.5 |
   "one-fifth of normal" -> 0.2 | "roughly a quarter" -> 0.25 | "no solar / panels offline" -> 0.0
2. minimum_battery_reserve - battery energy must stay at or above a level during a window.
   fields: start_hour, end_hour, minimum_energy_kwh (absolute kWh)
   A percentage or fraction "of capacity" / "of the battery" must be converted with the capacity_kwh
   given in the request: 50% of a 200 kWh battery -> 100.
3. no_charge_window - the battery must not be charged during a window
   (charger offline / isolated / disabled / charging circuit unavailable / do not charge).
   fields: start_hour, end_hour
4. no_discharge_window - the battery must not be discharged during a window
   (do not discharge / must not discharge / no battery draw / hold the battery / relay or protection testing).
   fields: start_hour, end_hour
5. max_grid_window - grid import in each hour must not exceed a limit during a window
   (feeder, transformer or substation limit; grid intake / import / draw cap). kW and kWh per hour mean the same here.
   fields: start_hour, end_hour, max_grid_kwh
6. no_op - the note does not change this 24-hour energy schedule: announcements, bookings, menus,
   deadlines, events, staffing, things scheduled for next week / next month, general information,
   or an energy remark with no supported constraint or no usable time window.

TIME RULES
- Hours are integers in 24-hour time: midnight = 0, 1 AM = 1, noon = 12, 1 PM = 13, 11 PM = 23.
- A window "from A until/to/through B", "between A and B", "A-B" -> start_hour = A, end_hour = B.
  end_hour is EXCLUSIVE (the clock hour at which the window ends): "1 PM to 3 PM" -> 13, 15 (covers hours 13 and 14);
  "6 PM until 9 PM" -> 18, 21; "noon until 2 PM" -> 12, 14; "13:00-15:00" -> 13, 15; "2-4 PM" -> 14, 16.
- A single hour ("at 3 PM", "during the 3 PM hour") -> 15, 16.
- "all day" / "for the whole day" -> 0, 24.   A window crossing midnight ("10 PM to 2 AM") -> 22, 2.
- A clock time without AM/PM or a leading zero ("from one until three", "between 2 and 4") is ambiguous:
  solar work happens in daylight, so read it as the daytime hour ("one until three" for panel work -> 13, 15);
  for other notes prefer the reading inside the campus operating day (06-22) unless the note says night,
  overnight, early morning or similar.
- Never shrink or move a window the note states clearly: "all day" / "the whole day" / "today" is always
  0, 24 even for a solar note, and "9 AM to 6 PM" stays 9, 18.
- Use only times stated in the note. Never guess a window that is not given.

STRICT RULES
- Exactly one entry per note, in note order, note_index starting at 0. Never merge or split notes.
- Use only the six directive_type values. Never invent demand, solar, tariff or battery limits.
- Fields that do not apply must be null. Numbers must be plain numbers (0.2, not "20%").
- explanation: one short sentence.

EXAMPLES (only the non-null fields are shown)
note: "PV output expected at roughly 30% of forecast between 09:00 and 11:00 because of haze."
-> {"directive_type":"solar_reduction","start_hour":9,"end_hour":11,"factor":0.3}
note: "Hold at least 40% of battery capacity from 5 PM until 8 PM." (capacity_kwh = 250)
-> {"directive_type":"minimum_battery_reserve","start_hour":17,"end_hour":20,"minimum_energy_kwh":100}
note: "Charger offline 1 AM-4 AM for firmware work."
-> {"directive_type":"no_charge_window","start_hour":1,"end_hour":4}
note: "Keep the battery from discharging between noon and 1 PM."
-> {"directive_type":"no_discharge_window","start_hour":12,"end_hour":13}
note: "Feeder limit: grid draw capped at 120 kWh per hour from 7 PM to 10 PM."
-> {"directive_type":"max_grid_window","start_hour":19,"end_hour":22,"max_grid_kwh":120}
note: "Guest lecture in Hall B moved to Thursday."
-> {"directive_type":"no_op"}
"""

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "directives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "note_index": {"type": "integer"},
                    "directive_type": {
                        "type": "string",
                        "enum": [
                            "solar_reduction",
                            "minimum_battery_reserve",
                            "no_charge_window",
                            "no_discharge_window",
                            "max_grid_window",
                            "no_op",
                        ],
                    },
                    "start_hour": {"type": ["integer", "null"]},
                    "end_hour": {"type": ["integer", "null"]},
                    "factor": {"type": ["number", "null"]},
                    "minimum_energy_kwh": {"type": ["number", "null"]},
                    "max_grid_kwh": {"type": ["number", "null"]},
                    "explanation": {"type": "string"},
                },
                "required": [
                    "note_index",
                    "directive_type",
                    "start_hour",
                    "end_hour",
                    "factor",
                    "minimum_energy_kwh",
                    "max_grid_kwh",
                    "explanation",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["directives"],
    "additionalProperties": False,
}


class LLMError(Exception):
    """A model call failed or returned unusable output."""


def build_user_message(
    notes: list[str],
    battery: Battery,
    daylight_hours: tuple[int, ...],
    previous_errors: dict[int, str] | None,
) -> str:
    lines = [
        "Battery for this scenario: "
        f"capacity_kwh={battery.capacity_kwh:g}, initial_energy_kwh={battery.initial_energy_kwh:g}, "
        f"minimum_energy_kwh={battery.minimum_energy_kwh:g}, "
        f"max_charge_kwh_per_hour={battery.max_charge_kwh_per_hour:g}, "
        f"max_discharge_kwh_per_hour={battery.max_discharge_kwh_per_hour:g}",
        "",
        "Operator notes:",
    ]
    for i, note in enumerate(notes):
        lines.append(f"[{i}] {note.strip()}")
    if previous_errors:
        # Repair attempt: the daylight hours (from the request's own solar forecast) are given only
        # here, so they can resolve an AM/PM slip without biasing clearly stated windows.
        daylight = (
            f"{daylight_hours[0]}-{daylight_hours[-1]} (24-hour clock)" if daylight_hours
            else "none (no solar forecast today)"
        )
        lines.append("")
        lines.append(f"Daylight hours with solar forecast > 0 in this scenario: {daylight}")
        lines.append("Your previous interpretation had these problems; re-interpret ALL notes carefully and fix them:")
        for i in sorted(previous_errors):
            lines.append(f"- note {i}: {previous_errors[i]}")
    lines.append("")
    shape = (
        '{"directives": [{"note_index": 0, "directive_type": "...", "start_hour": 18, "end_hour": 21, '
        '"factor": null, "minimum_energy_kwh": 100, "max_grid_kwh": null, "explanation": "..."}, ...]}'
    )
    lines.append(
        f"Return ONLY a JSON object of the form {shape} with exactly {len(notes)} entries, one per note, in order."
    )
    return "\n".join(lines)


def extract_json(text: str) -> dict:
    """Parse the first JSON object in a model reply (tolerates code fences / chatter)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise LLMError("model reply contained no JSON object") from None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMError(f"model reply is not valid JSON: {exc.msg}") from None
    if not isinstance(data, dict):
        raise LLMError("model reply is not a JSON object")
    return data


class OpenAIBackend:
    """Official ``openai`` SDK against OpenAI or any OpenAI-compatible endpoint."""

    def __init__(self, settings: Settings):
        import openai

        self._openai = openai
        self._kwargs: dict[str, Any] = {"timeout": settings.llm_timeout_s, "max_retries": 1}
        if settings.llm_api_key:
            self._kwargs["api_key"] = settings.llm_api_key
        elif settings.llm_base_url:
            self._kwargs["api_key"] = "not-needed"  # local servers such as Ollama
        if settings.llm_base_url:
            self._kwargs["base_url"] = settings.llm_base_url
        self._client = None
        self._loop = None
        self._mode = "json_schema"  # degrades to json_object, then plain text, if the endpoint rejects it
        self._temperature = True

    def _get_client(self):
        # The async client owns a connection pool bound to the event loop that first used it;
        # build one per running loop so the service (and test clients) never reuse a closed loop.
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:
            self._client = self._openai.AsyncOpenAI(**self._kwargs)
            self._loop = loop
        return self._client

    async def complete(self, model: str, system: str, user: str) -> dict:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self._temperature:
            kwargs["temperature"] = 0
        if self._mode == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "directive_interpretation", "strict": True, "schema": OUTPUT_SCHEMA},
            }
        elif self._mode == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = await self._get_client().chat.completions.create(**kwargs)
        except self._openai.BadRequestError as exc:
            message = str(exc).lower()
            if self._temperature and "temperature" in message:
                self._temperature = False
                return await self.complete(model, system, user)
            if self._mode == "json_schema":
                log.warning("endpoint rejected json_schema response_format; falling back to json_object")
                self._mode = "json_object"
                return await self.complete(model, system, user)
            if self._mode == "json_object":
                log.warning("endpoint rejected json_object response_format; falling back to plain text")
                self._mode = "text"
                return await self.complete(model, system, user)
            raise LLMError(f"bad request: {type(exc).__name__}") from None
        choice = resp.choices[0] if resp.choices else None
        if choice is None or choice.message is None or not choice.message.content:
            raise LLMError("empty model reply")
        if getattr(choice.message, "refusal", None):
            raise LLMError("model refused")
        return extract_json(choice.message.content)


class AnthropicBackend:
    """Official ``anthropic`` SDK with structured JSON output."""

    def __init__(self, settings: Settings):
        import anthropic

        self._anthropic = anthropic
        self._kwargs: dict[str, Any] = {"timeout": settings.llm_timeout_s, "max_retries": 1}
        if settings.llm_api_key:
            self._kwargs["api_key"] = settings.llm_api_key
        self._client = None
        self._loop = None
        self._use_format = True
        self._effort = settings.llm_effort

    def _get_client(self):
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:
            self._client = self._anthropic.AsyncAnthropic(**self._kwargs)
            self._loop = loop
        return self._client

    async def complete(self, model: str, system: str, user: str) -> dict:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        output_config: dict[str, Any] = {}
        if self._effort:
            output_config["effort"] = self._effort
        if self._use_format:
            output_config["format"] = {"type": "json_schema", "schema": OUTPUT_SCHEMA}
        if output_config:
            kwargs["output_config"] = output_config
        try:
            resp = await self._get_client().messages.create(**kwargs)
        except self._anthropic.BadRequestError as exc:
            if self._use_format:
                log.warning("structured output rejected; falling back to plain JSON parsing")
                self._use_format = False
                return await self.complete(model, system, user)
            if self._effort:
                log.warning("effort parameter rejected by this model; retrying without it")
                self._effort = None
                return await self.complete(model, system, user)
            raise LLMError(f"bad request: {type(exc).__name__}") from None
        if resp.stop_reason == "refusal":
            raise LLMError("model refused")
        text = "".join(block.text for block in resp.content if block.type == "text")
        if not text.strip():
            raise LLMError("empty model reply")
        return extract_json(text)


def make_backend(settings: Settings):
    if settings.llm_provider == "openai":
        return OpenAIBackend(settings)
    if settings.llm_provider == "anthropic":
        return AnthropicBackend(settings)
    return None


class NoteInterpreter:
    """LLM call + guardrails + retry budget + cache. Never raises."""

    def __init__(self, settings: Settings, backend=None):
        self.settings = settings
        self.backend = backend if backend is not None else make_backend(settings)
        self._cache: OrderedDict[tuple, list[Directive]] = OrderedDict()

    def _models(self) -> list[str]:
        models = [self.settings.llm_model, self.settings.llm_model]  # first try + repair try
        if self.settings.llm_fallback_model:
            models.append(self.settings.llm_fallback_model)
        return models

    async def interpret(
        self, notes: list[str], battery: Battery, daylight_hours: tuple[int, ...] = ()
    ) -> tuple[list[Directive], dict[str, Any]]:
        """Return one Directive per note (in order) plus diagnostics for logging.

        ``daylight_hours`` are the hours whose solar forecast is > 0; they come
        straight from the request and only help the model resolve AM/PM.
        """
        n = len(notes)
        key = (tuple(notes), float(battery.capacity_kwh), tuple(daylight_hours))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return list(cached), {"cached": True, "attempts": 0}

        info: dict[str, Any] = {"cached": False, "attempts": 0, "failures": []}
        valid: dict[int, Directive] = {}
        errors: dict[int, str] = {i: "not interpreted" for i in range(n)}

        if self.backend is None:
            info["failures"].append("no LLM configured")
        else:
            deadline = time.monotonic() + self.settings.llm_total_budget_s
            previous_errors: dict[int, str] | None = None
            for model in self._models():
                remaining = deadline - time.monotonic()
                if remaining < MIN_ATTEMPT_SECONDS:
                    info["failures"].append("time budget exhausted")
                    break
                info["attempts"] += 1
                user = build_user_message(notes, battery, daylight_hours, previous_errors)
                try:
                    raw = await asyncio.wait_for(
                        self.backend.complete(model, SYSTEM_PROMPT, user), timeout=remaining
                    )
                except asyncio.TimeoutError:
                    info["failures"].append(f"{model}: timeout")
                    log.warning("LLM attempt %d (%s) timed out", info["attempts"], model)
                    previous_errors = None
                    continue
                except Exception as exc:  # provider/SDK errors: never crash the request
                    name = type(exc).__name__
                    info["failures"].append(f"{model}: {name}")
                    log.warning("LLM attempt %d (%s) failed: %s: %s", info["attempts"], model, name, str(exc)[:200])
                    previous_errors = None
                    if name in ("AuthenticationError", "PermissionDeniedError"):
                        break  # a bad key never succeeds on retry; do not burn the time budget
                    continue
                # First attempt: a solar window with no daylight is sent back for re-interpretation
                # (an AM/PM slip); on the repair attempt the model's answer stands.
                new_valid, new_errors = validate_llm_output(
                    raw, notes, battery, daylight_hours if previous_errors is None else None
                )
                for i, d in new_valid.items():
                    valid.setdefault(i, d)  # keep earlier valid answers, only fill gaps
                errors = {i: new_errors.get(i, "missing") for i in range(n) if i not in valid}
                if not errors:
                    break
                log.info("guardrails rejected %d note(s): %s", len(errors), errors)
                previous_errors = errors

        directives: list[Directive] = []
        for i in range(n):
            if i in valid:
                directives.append(valid[i])
            else:
                reason = errors.get(i, "unavailable")
                directives.append(
                    no_op(i, f"Could not be mapped to a supported directive ({reason}); treated as no_op.")
                )
        info["unresolved"] = sorted(errors)

        if not errors and self.settings.llm_cache_size > 0:
            self._cache[key] = list(directives)
            while len(self._cache) > self.settings.llm_cache_size:
                self._cache.popitem(last=False)
        return directives, info
