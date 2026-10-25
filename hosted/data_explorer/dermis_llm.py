"""ASI:One LLM for understanding requests and writing the interpretation.

The model never supplies numbers: it reads facts the analysis computed. Every call has a rule-based
fallback, so agents keep working with no API key or when the API is down.
"""

from __future__ import annotations

import json
import logging
import re

import httpx

import dermis_config as config

log = logging.getLogger("dermis.llm")

OBJECTS = ("can", "egg", "ball", "cup", "bottle", "sponge")
SCENARIOS = ("nominal", "crush", "light_grip", "drop")


async def complete(system: str, user: str, json_mode: bool = False, max_tokens: int = 500) -> str | None:
    if not config.ASI1_API_KEY:
        return None
    body = {
        "model": config.ASI1_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            r = await client.post(
                f"{config.ASI1_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {config.ASI1_API_KEY}"},
                json=body,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
        log.warning("ASI:One call failed, using fallback: %s", e)
        return None


async def chat(messages: list[dict], tools: list[dict], max_tokens: int = 1200) -> dict | None:
    """One tool-calling turn. Returns the assistant message (content and/or tool_calls), or None."""
    if not config.ASI1_API_KEY:
        return None
    body = {"model": config.ASI1_AGENT_MODEL, "messages": messages, "tools": tools,
            "temperature": 0.3, "max_tokens": max_tokens}
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            r = await client.post(
                f"{config.ASI1_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {config.ASI1_API_KEY}"},
                json=body,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
        log.warning("ASI:One agent call failed: %s", e)
        return None


def _json(text: str | None) -> dict:
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def parse_rules(text: str) -> dict:
    t = text.lower()
    out: dict = {}
    m = re.search(r"(?:episode|session|ep|run|recording)[\s_#-]*(\d{1,4})", t)
    if m:
        out["episode"] = f"episode_{int(m.group(1)):04d}"
    for o in OBJECTS:
        if re.search(rf"\b{o}s?\b", t):
            out["object"] = o
            break
    for s in SCENARIOS:
        if s.replace("_", " ") in t or s in t:
            out["scenario"] = s
            break
    if re.search(r"safe ?grip (on|enabled)|with safe ?grip", t):
        out["safe_grip"] = True
    elif re.search(r"safe ?grip (off|disabled)|without safe ?grip|no safe ?grip", t):
        out["safe_grip"] = False
    if re.search(r"\b(fleet|overall|across|all (episodes|runs|data)|summary|summari[sz]e|yield|"
                 r"which (objects?|items?)|most|rate|dataset|collect|statistics|stats)\b", t):
        out["mode"] = "fleet"
    if re.search(r"\b(help|what can you do|capabilities|who are you)\b", t):
        out["mode"] = "help"
    return out


async def parse_request(text: str) -> dict:
    rules = parse_rules(text)
    if "episode" in rules or not config.ASI1_API_KEY:
        return rules
    reply = await complete(
        "Extract fields from a request about robot touch-sensor recordings. Reply with JSON only, keys: "
        f"object (one of {list(OBJECTS)} or null), scenario (one of {list(SCENARIOS)} or null), "
        "safe_grip (true/false/null), mode ('episode' to analyze one recording, 'fleet' for statistics "
        "across many, 'help' for capabilities). Objects may be described indirectly: a soda can is 'can', "
        "a paper cup is 'cup', a water bottle is 'bottle', fruit or produce maps to 'egg'.",
        text, json_mode=True, max_tokens=120,
    )
    llm = _json(reply)
    merged = {k: v for k, v in llm.items() if v is not None and k in ("object", "scenario", "safe_grip", "mode")}
    if merged.get("mode") not in ("episode", "fleet"):  # only explicit words ("help") ask for help
        merged.pop("mode", None)
    if merged.get("object") not in OBJECTS:
        merged.pop("object", None)
    if merged.get("scenario") not in SCENARIOS:
        merged.pop("scenario", None)
    merged.update(rules)
    return merged


async def interpret(domain: str, priorities: tuple[str, ...], question: str, facts: dict, findings: list[dict]) -> str | None:
    payload = {k: v for k, v in facts.items() if not k.startswith("_")}
    return await complete(
        f"You are a robotics touch-sensing engineer advising a team that builds {domain}. "
        f"They care most about: {', '.join(priorities)}. Using ONLY the facts and findings given, write "
        "3 to 4 sentences: what physically happened in this grasp, what it means for this application, "
        "and the single most important thing to change. Do not invent numbers; quote only numbers present "
        "in the input. If the data is synthetic (source sim) or force is an uncalibrated estimate, say so "
        "briefly. Plain text, no headings.",
        json.dumps({"question": question, "facts": payload, "findings": findings}, default=str),
        max_tokens=260,
    )


async def route(text: str, catalog: list[dict]) -> list[str]:
    reply = await complete(
        "Pick which specialist agents should answer a request about robot touch sensing. Reply with JSON "
        '{"agents": [keys]} using 1 to 3 keys from the catalog, most relevant first. Use "all" as the only '
        "key if the user asks to compare across every application or industry.",
        json.dumps({"request": text, "catalog": catalog}),
        json_mode=True, max_tokens=60,
    )
    keys = _json(reply).get("agents") or []
    return [k for k in keys if isinstance(k, str)]
