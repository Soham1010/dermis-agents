"""The agent loop: an ASI:One LLM that talks to the user and calls tools over the Dermis data.

Every number comes from a tool (the same analysis the reports use), never from the model. Charts are
returned to the model as short [[chart:N]] tokens and swapped back in afterwards, because the
QuickChart URLs are thousands of characters and a model would garble them. If the LLM is unavailable
the caller falls back to the deterministic report, so an agent always answers.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Awaitable, Callable

import dermis_analysis as analysis, dermis_charts as charts, dermis_llm as llm, dermis_specialist as specialist, dermis_viz as viz
from dermis_profiles import BY_KEY, PROFILES, Profile
from dermis_store import DataUnavailable, snapshot

MAX_ROUNDS = 6
HISTORY_TURNS = 8
TOOL_RESULT_CHARS = 6000
IMAGE_RE = re.compile(r"!\[[^\]]*\]\((https?://[^)]+)\)")
APPS = [p.key for p in PROFILES if p.kind == "touch"]

# Agent-to-agent call the hub can make: (agent key, question) -> markdown answer.
AskAgent = Callable[[str, str], Awaitable[str]]


class Charts:
    """Holds chart images out of the model's context and puts them back into the final answer."""

    def __init__(self):
        self.images: list[str] = []

    def stash(self, md: str) -> str:
        def repl(m: re.Match) -> str:
            self.images.append(m.group(0))
            return f"[[chart:{len(self.images)}]]"
        return IMAGE_RE.sub(repl, md)

    def restore(self, text: str) -> str:
        used = set()

        def repl(m: re.Match) -> str:
            i = int(m.group(1))
            if 1 <= i <= len(self.images):
                used.add(i)
                return self.images[i - 1]
            return ""
        text = re.sub(r"\[\[chart:(\d+)\]\]", repl, text)
        text = re.sub(r"^\s*\[chart\]\s*$", "", text, flags=re.M)  # placeholder echoed from memory
        # If the model placed no chart at all, show the first one it looked at.
        if not used and self.images:
            text += "\n\n" + self.images[0]
        return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------- tools

def _fn(name: str, description: str, props: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": props or {}, "required": required or []}}}


EPISODE = {"type": "string", "description": "recording id, e.g. episode_0004 (0-191)"}
OBJECT = {"type": "string", "enum": ["can", "egg", "ball", "cup", "bottle", "sponge"]}
SCENARIO = {"type": "string", "enum": ["nominal", "crush", "light_grip", "drop"]}
APP = {"type": "string", "enum": APPS, "description": "application whose limits to judge against"}

DATA_TOOLS = [
    _fn("list_recordings", "Find recordings. Returns id, object, scenario, SafeGrip, outcome for each match.",
        {"object": OBJECT, "scenario": SCENARIO, "safe_grip": {"type": "boolean"},
         "outcome": {"type": "string", "enum": ["success", "fracture", "slip", "impact", "early_release", "incomplete"]},
         "limit": {"type": "integer", "description": "max rows, default 15"}}),
    _fn("analyze_recording", "Analyze one recording for an application: per-pod force, contact, slip vibration, "
        "sensor saturation and noise, touch latency, faults, findings, readiness score, the SafeGrip on/off twin "
        "outcome, and a force chart.", {"episode": EPISODE, "application": APP}, ["episode"]),
    _fn("fleet_stats", "Success and failure rates across recordings, by object, scenario and SafeGrip, optionally "
        "limited to the objects an application handles.", {"application": APP, "object": OBJECT}),
    _fn("compare_applications", "Score one recording against every application's limits (humanoid, industrial, "
        "quadruped, prosthetic, surgical, warehouse, food, eldercare, teleop, sensor QA).", {"episode": EPISODE},
        ["episode"]),
    _fn("sensor_health", "Scan all recordings for sensor problems: magnetometer saturation, noise vs spec, dropped "
        "frames, touch detection latency."),
    _fn("dataset_overview", "What the dataset contains: size, outcome counts, coverage gaps, and how much touch "
        "improves a baseline model over IMU only."),
    _fn("make_chart", "Draw a chart and get its numbers. Describe it in words, e.g. 'dashboard', 'plot episode 4', "
        "'plot the raw field for episode 4', 'compare episode 4 with safegrip on', 'compare episode 4 vs episode 0', "
        "'histogram of peak force', 'scatter latency vs peak force', 'pie of outcomes for the bottle', "
        "'heatmap of success', 'radar of episode 40'.", {"request": {"type": "string"}}, ["request"]),
    _fn("make_table", "Build a pandas-style table. Describe it, e.g. 'overview', 'success by scenario', "
        "'peak force by object and scenario', 'latency by safe grip', 'stats for episode 4'.",
        {"request": {"type": "string"}}, ["request"]),
]
ASK_TOOL = _fn("ask_agent", "Ask one of the specialist agents (a separate Fetch.ai agent) and get its full answer. "
               "Use it when a question is about a specific application.",
               {"agent": {"type": "string", "enum": [p.key for p in PROFILES]}, "question": {"type": "string"}},
               ["agent", "question"])


def _parsed(args: dict) -> dict:
    return {k: v for k, v in args.items() if k in ("object", "scenario", "safe_grip") and v is not None}


def t_list_recordings(args: dict) -> dict:
    eps = snapshot().episodes
    rows = [e for e in eps
            if (not args.get("object") or e["object"] == args["object"])
            and (not args.get("scenario") or e["scenario"] == args["scenario"])
            and (args.get("safe_grip") is None or bool(e["safe_grip"]) == args["safe_grip"])
            and (not args.get("outcome") or e["outcome"] == args["outcome"])]
    limit = int(args.get("limit") or 15)
    return {"matches": len(rows), "recordings": [
        {k: e[k] for k in ("episode", "object", "scenario", "safe_grip", "outcome", "seconds")} for e in rows[:limit]]}


def t_analyze_recording(args: dict, default_app: str) -> dict:
    snap = snapshot()
    eid = _episode_id(args.get("episode", ""))
    meta = snap.meta(eid)
    if meta is None:
        return {"error": f"no recording {eid}; ids run episode_0000 to episode_{len(snap.episodes) - 1:04d}"}
    p = BY_KEY.get(args.get("application") or default_app) or BY_KEY["humanoid_arm"]
    f = snap.facts(eid)
    findings = analysis.evaluate(p, f)
    twin = analysis.counterfactual(meta, snap.episodes)
    key = "peak_force_gt_n" if f["has_ground_truth_force"] else "peak_force_est_n"
    return {
        "episode": eid, "application": p.key, "object": meta["object"], "mass_g": meta["mass_g"],
        "scenario": meta["scenario"], "safe_grip": meta["safe_grip"], "outcome": meta["outcome"],
        "seconds": f["seconds"], "force_limit_n": p.limit_for(meta["object"]),
        "readiness_score": analysis.score(findings),
        "findings": [{k: x[k] for k in ("severity", "message", "action")} for x in findings],
        "pods": {p.pod_labels[k]: {"peak_force_n": v[key], "contact_s": v["contact_s"], "slip_vibration_s": v["slip_signal_s"],
                                   "rail_frames": v["rail_frames"], "noise_vs_spec": v["noise_vs_spec"]}
                 for k, v in f["pods"].items()},
        "touch_detect_latency_ms": f.get("touch_detect_latency_ms"), "fault": f.get("fault"),
        "fault_at_s": f.get("fault_at_s"), "time_to_hold_s": f.get("time_to_hold_s"),
        "min_friction_margin": f.get("min_friction_margin"), "max_lift_cm": f.get("max_lift_cm"),
        "safegrip_twin": ({"episode": twin["episode"], "safe_grip": twin["safe_grip"], "outcome": twin["outcome"]}
                          if twin else None),
        "force_is": "simulated ground truth" if f["has_ground_truth_force"] else "uncalibrated field estimate",
        "proxy_note": p.proxy_note or None,
        "chart": charts.force_chart(f, p.pod_labels, p.limit_for(meta["object"])),
    }


def t_fleet_stats(args: dict) -> dict:
    eps = snapshot().episodes
    p = BY_KEY.get(args.get("application") or "")
    objects = (args["object"],) if args.get("object") else (p.objects if p else ())
    f = analysis.fleet(eps, objects)
    f["chart"] = charts.fleet_chart(f, "Success rate by object")
    return f


def t_compare_applications(args: dict) -> dict:
    snap = snapshot()
    eid = _episode_id(args.get("episode", ""))
    if snap.meta(eid) is None:
        return {"error": f"no recording {eid}"}
    f = snap.facts(eid)
    rows = []
    for p in PROFILES:
        if not p.in_matrix:
            continue
        fs = analysis.evaluate(p, f)
        rows.append({"application": p.title.replace(" Agent", ""), "score": analysis.score(fs),
                     "top_finding": fs[0]["message"] if fs else "no problems", "first_action": fs[0]["action"] if fs else ""})
    return {"episode": eid, "outcome": snap.meta(eid)["outcome"], "applications": rows,
            "chart": charts.score_chart([(r["application"], r["score"]) for r in rows], f"Readiness by application · {eid}")}


def t_sensor_health(args: dict) -> dict:
    rows, src = specialist._sensor_scan()
    railed = [r for r in rows if r["rail"]]
    lat = sorted(r["latency"] for r in rows if r["latency"] is not None)
    return {"recordings": len(rows), "frames": sum(r["frames"] for r in rows),
            "railed_recordings": len(railed), "noisy_recordings": sum(r["noise"] > 3 for r in rows),
            "recordings_with_dropped_frames": sum(bool(r["dropped"]) for r in rows),
            "latency_ms_median": lat[len(lat) // 2] if lat else None, "latency_ms_worst": lat[-1] if lat else None,
            "worst_railed": [{"episode": r["episode"], "object": r["object"], "railed_frames": r["rail"],
                              "also_grip_fault": r["fault"]} for r in sorted(railed, key=lambda r: -r["rail"])[:5]],
            "calibration": "field-to-force is an assumed 1.5 mT/N until load-cell calibration"}


def t_dataset_overview(args: dict) -> dict:
    snap = snapshot()
    f = analysis.fleet(snap.episodes)
    out = {"recordings": len(snap.episodes), "frames": sum(e.get("frames") or 0 for e in snap.episodes),
           "minutes": round(sum(e.get("seconds") or 0 for e in snap.episodes) / 60, 1),
           "source": snap.source, "outcomes": f["outcomes"], "by_scenario": f["by_scenario"],
           "hardware_recordings": sum(1 for e in snap.episodes if e.get("source") == "hardware")}
    b = snap.baseline or {}
    try:
        out["baseline_macro_f1"] = {
            "imu_only": {"grasp": b["IMU only"]["session"]["grasp"]["macro_f1"], "fault": b["IMU only"]["session"]["fault"]["macro_f1"]},
            "imu_plus_touch": {"grasp": b["IMU + touch"]["session"]["grasp"]["macro_f1"], "fault": b["IMU + touch"]["session"]["fault"]["macro_f1"]}}
    except KeyError:
        pass
    return out


def _episode_id(s: str) -> str:
    m = re.search(r"(\d{1,4})", s or "")
    return f"episode_{int(m.group(1)):04d}" if m else s


async def run_tool(name: str, args: dict, default_app: str, stash: Charts, ask: AskAgent | None) -> str:
    try:
        if name == "make_chart":
            res = await asyncio.to_thread(viz.visualize, args.get("request", ""), _parsed(args))
            text = res["markdown"]
        elif name == "make_table":
            res = await asyncio.to_thread(viz.explore, args.get("request", ""), _parsed(args))
            text = res["markdown"]
        elif name == "ask_agent" and ask:
            text = await ask(args.get("agent", ""), args.get("question", ""))
        else:
            fn = {"list_recordings": t_list_recordings, "fleet_stats": t_fleet_stats,
                  "compare_applications": t_compare_applications, "sensor_health": t_sensor_health,
                  "dataset_overview": t_dataset_overview}.get(name)
            if name == "analyze_recording":
                out = await asyncio.to_thread(t_analyze_recording, args, default_app)
            elif fn:
                out = await asyncio.to_thread(fn, args)
            else:
                return json.dumps({"error": f"unknown tool {name}"})
            text = json.dumps(out, default=str)
    except DataUnavailable as e:
        text = json.dumps({"error": f"data unavailable: {e}"})
    except Exception as e:  # a broken tool must not kill the conversation
        text = json.dumps({"error": f"{name} failed: {e}"})
    text = stash.stash(text)
    return text if len(text) <= TOOL_RESULT_CHARS else text[:TOOL_RESULT_CHARS] + " …(truncated)"


# ---------------------------------------------------------------- personas

def system_prompt(title: str, role: str, priorities: str, extra: str = "") -> str:
    return (
        f"You are {title}, a Fetch.ai agent built on Dermis, a magnetic tactile glove (4 touch pods + IMUs at "
        f"350 Hz). {role} You care most about: {priorities}.\n\n"
        "Data: 192 recorded grasps of 6 objects (can, egg, ball, cup, bottle, sponge) in 4 scenarios (nominal, "
        "crush, light_grip, drop), each with SafeGrip force limiting on or off. The recordings are simulated "
        "(hardware sessions come later); say so once when you quote forces.\n\n"
        "How to work:\n"
        "- Use tools for every number. Never invent or estimate figures the tools didn't return.\n"
        "- Call several tools if the question needs it; ask a short clarifying question only if you truly can't "
        "pick a sensible default (otherwise pick one and say which).\n"
        "- Tool results may contain tokens like [[chart:3]]. Put the ones that help on their own line where the "
        "chart belongs; never write image links yourself.\n"
        "- Answer first in 1-3 plain sentences, then the evidence (short bullets or a small markdown table), then "
        "1-3 concrete next steps. Keep it tight; no headings for short answers.\n"
        "- You remember this conversation; resolve 'it', 'that one', 'the egg' from earlier turns.\n"
        "- If asked something unrelated to robot touch data, answer briefly and offer what you can do."
        + (f"\n\n{extra}" if extra else ""))


def persona(key: str) -> tuple[str, str, str]:
    """(system prompt, default application, short intro) for an agent key."""
    if key == "hub":
        return (system_prompt(
            "Dermis Hub", "You are the front door: answer directly with your tools, and use ask_agent to bring in "
                          "a specialist agent when a question is about a specific application (you may ask more "
                          "than one and combine their answers).",
            "getting people the right answer fast, picking the right specialist, comparing applications",
            "When the user doesn't name an application, judge recordings as a humanoid arm (application "
            "humanoid_arm) and say so. "
            "Specialist agents: " + "; ".join(f"{p.key} = {p.title}" for p in PROFILES)), "humanoid_arm", "")
    p = BY_KEY[key]
    if p.kind == "visualizer":
        role = "You turn questions into charts: pick the chart that answers the question, then explain what it shows."
    elif p.kind == "explorer":
        role = "You answer with tables like a data analyst using pandas, then explain what they show."
    else:
        role = f"You analyze touch data for {p.domain}, judging it against that application's limits."
    extra = p.proxy_note
    default = p.key if p.kind == "touch" else "humanoid_arm"
    return system_prompt(p.title, role, ", ".join(p.priorities), extra), default, ""


# ---------------------------------------------------------------- the loop

async def converse(key: str, text: str, history: list[dict] | None = None,
                   ask: AskAgent | None = None) -> str | None:
    """One user turn. Returns the answer markdown, or None if the LLM is unavailable (caller falls back)."""
    system, default_app, _ = persona(key)
    tools = DATA_TOOLS + ([ASK_TOOL] if ask else [])
    messages = [{"role": "system", "content": system}, *(history or [])[-2 * HISTORY_TURNS:],
                {"role": "user", "content": text}]
    stash = Charts()
    for _ in range(MAX_ROUNDS):
        msg = await llm.chat(messages, tools)
        if msg is None:
            return None
        calls = msg.get("tool_calls") or []
        if not calls:
            answer = (msg.get("content") or "").strip()
            return stash.restore(answer) if answer else None
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        results = await asyncio.gather(*(
            run_tool(c["function"]["name"], _args(c), default_app, stash, ask) for c in calls))
        for c, res in zip(calls, results):
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": res})
    # Out of rounds: ask for a final answer with what it has.
    messages.append({"role": "user", "content": "Answer now with what you have."})
    msg = await llm.chat(messages, [])
    return stash.restore((msg or {}).get("content") or "") or None


def _args(call: dict) -> dict:
    try:
        a = json.loads(call["function"].get("arguments") or "{}")
        return a if isinstance(a, dict) else {}
    except json.JSONDecodeError:
        return {}


def remember(history: list[dict], user: str, answer: str) -> list[dict]:
    """Keep the last turns as plain text (no tool traces, no chart URLs) so storage stays small."""
    short = IMAGE_RE.sub("", answer)
    return (history + [{"role": "user", "content": user[:1500]},
                       {"role": "assistant", "content": short[:2500]}])[-2 * HISTORY_TURNS:]
