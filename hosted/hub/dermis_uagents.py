"""The uAgents layer: 13 specialist agents + the Dermis Hub, all speaking the Agent Chat Protocol.

Every agent is reachable from ASI:One on its own. The hub also fans questions out to specialists over a
typed internal protocol (TouchQuery -> TouchReport), and if a specialist doesn't answer in time it runs
that specialist's analysis itself, so a dead agent degrades the answer instead of breaking it.
"""

from __future__ import annotations

import json
import secrets
from uagents import Agent, Context, Model, Protocol
from uagents_core.identity import Identity
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    TextContent,
    chat_protocol_spec,
)

import dermis_brain as brain, dermis_config as config, dermis_hub as hub, dermis_specialist as specialist
from dermis_profiles import BY_KEY, PROFILES, Profile

HUB_KEY = "hub"
SPECIALIST_TIMEOUT_S = 30
BADGES = ("![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)\n"
          "![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)")


class TouchQuery(Model):
    text: str
    parsed_json: str = "{}"
    brief: bool = False
    agent: bool = False  # True: answer as the LLM agent (ask_agent); False: the structured report


class TouchReport(Model):
    agent: str
    result_json: str


def seed_secret() -> str:
    """AGENT_SEED_SECRET, or a random one saved to .agent_seed so addresses stay stable across runs."""
    if config.AGENT_SEED_SECRET:
        return config.AGENT_SEED_SECRET
    path = config.ROOT / ".agent_seed"
    if not path.exists():
        path.write_text(secrets.token_hex(32))
        path.chmod(0o600)
    return path.read_text().strip()


def seed_for(key: str) -> str:
    return f"{seed_secret()}:dermis:{key}"


def address_for(key: str) -> str:
    # Addresses are derived from seeds, so the hub knows every specialist without importing it.
    return Identity.from_seed(seed_for(key), 0).address


def write_readme(key: str, title: str, body: str) -> str:
    """Agentverse indexes this README; ASI:One uses it to decide which agent answers a question."""
    folder = config.ROOT / "agent_readmes"
    folder.mkdir(exist_ok=True)
    path = folder / f"{key}.md"
    path.write_text(f"# {title}\n\n{BADGES}\n\n{body}\n")
    return str(path)


def profile_readme(p: Profile) -> str:
    ex = "\n".join(f"- {e}" for e in p.examples)
    return (f"Analyzes Dermis tactile-sensor recordings (magnetic touch pods + IMU at 350 Hz) for "
            f"**{p.domain}**.\n\nFocus: {', '.join(p.priorities)}.\n\n"
            "Reads recordings from SpacetimeDB (local dataset fallback), computes contact force, slip "
            "vibration, sensor saturation and noise, judges them against this application's limits, "
            "charts the result and recommends what to change.\n\n"
            f"Keywords: {', '.join(p.keywords)}\n\nExample questions:\n{ex}\n"
            + (f"\nNote: {p.proxy_note}\n" if p.proxy_note else ""))


def chat_protocol(answer) -> Protocol:
    """ACP handler: ack, answer the text, reply with markdown and end the session."""
    proto = Protocol(spec=chat_protocol_spec)

    @proto.on_message(ChatMessage)
    async def on_chat(ctx: Context, sender: str, msg: ChatMessage):
        await ctx.send(sender, ChatAcknowledgement(acknowledged_msg_id=msg.msg_id))
        text = " ".join(c.text for c in msg.content if isinstance(c, TextContent)).strip()
        if not text:
            return  # start-session or metadata only
        ctx.logger.info(f"chat from {sender[:16]}…: {text[:120]}")
        try:
            reply = await answer(ctx, sender, text)
        except Exception as e:  # never leave ASI:One hanging
            ctx.logger.exception("answer failed")
            reply = f"Sorry, that request failed inside the agent: {e}"
        await ctx.send(sender, ChatMessage(content=[
            TextContent(type="text", text=reply),
            EndSessionContent(type="end-session"),
        ]))

    @proto.on_message(ChatAcknowledgement)
    async def on_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
        pass

    return proto


def port_for(key: str) -> int:
    """Each agent runs as its own process: hub on DERMIS_PORT, specialists on the ports after it."""
    return config.BUREAU_PORT + all_keys().index(key)


HUB_TITLE = "Dermis Hub: Robot Touch Intelligence"
HUB_README = (
    "The front door to Dermis touch intelligence. Routes questions about robot touch sensing to "
    f"{len(PROFILES)} specialist agents (humanoid arms, industrial manipulators, quadrupeds, "
    "prosthetics, surgical, warehouse, food & agriculture, care robots, teleoperation, sensor "
    "health, robot-learning data), plus a Data Explorer for pandas-style tables and a Visualizer for "
    "charts (time series, overlays, histograms, scatter, pie, radar, heatmaps), and combines their answers. "
    "Can compare one recording across every application at once.\n\nExample questions:\n- Analyze episode 4\n"
    "- Compare all applications on episode 4\n- Show me a dashboard chart\n"
    "- Which items do we drop most in the warehouse?\n- Are any sensor readings unreliable?\n"
    "- What data should we collect next?\n")


def identity_of(key: str) -> tuple[str, str, str]:
    """(agent name, title, README body) for an agent key."""
    if key == HUB_KEY:
        return "dermis_hub", HUB_TITLE, HUB_README
    p = BY_KEY[key]
    return p.name, p.title, profile_readme(p)


async def converse(ctx: Context, sender: str, key: str, text: str, fallback, ask=None) -> str:
    """LLM agent turn with this sender's recent history; the structured report if the LLM is down."""
    storage_key = f"history:{sender}"
    history = ctx.storage.get(storage_key) or []
    answer = await brain.converse(key, text, history, ask)
    if answer is None:
        ctx.logger.warning("ASI:One unavailable; answering with the structured report")
        answer = await fallback()
    ctx.storage.set(storage_key, brain.remember(history, text, answer))
    return answer


def specialist_protocols(p: Profile) -> list[Protocol]:
    async def answer(ctx: Context, sender: str, text: str) -> str:
        async def report() -> str:
            return (await specialist.respond(p, text))["markdown"]
        return await converse(ctx, sender, p.key, text, report)

    internal = Protocol(name="dermis_touch", version="0.1.0")

    @internal.on_message(TouchQuery, replies=TouchReport)
    async def on_query(ctx: Context, sender: str, q: TouchQuery):
        ctx.logger.info(f"{'agent question' if q.agent else 'query'} from {sender[:16]}…: {q.text[:100]}")
        try:
            if q.agent:
                md = await brain.converse(p.key, q.text) or (await specialist.respond(p, q.text))["markdown"]
                result = {"markdown": md, "score": None, "headline": "", "episode": None}
            else:
                result = await specialist.respond(p, q.text, json.loads(q.parsed_json), q.brief)
        except Exception as e:
            ctx.logger.exception("query failed")
            result = {"markdown": f"**{p.title}** failed: {e}", "score": None, "headline": str(e), "episode": None}
        await ctx.send(sender, TouchReport(agent=p.key, result_json=json.dumps(result, default=str)))

    return [chat_protocol(answer), internal]


def hub_protocols(addresses: dict[str, str]) -> list[Protocol]:
    """`addresses` maps specialist keys to agents to ask. Keys missing from it run inside the hub."""

    async def answer(ctx: Context, sender: str, text: str) -> str:
        async def ask(key: str, q: str, parsed: dict, brief: bool, agent: bool = False) -> dict:
            if key not in addresses:  # not deployed as its own agent: same code, run here
                if agent:
                    md = await brain.converse(key, q) or (await specialist.respond(BY_KEY[key], q))["markdown"]
                    return {"markdown": md}
                return await specialist.respond(BY_KEY[key], q, parsed, brief)
            reason = ""
            try:
                reply, status = await ctx.send_and_receive(
                    addresses[key], TouchQuery(text=q, parsed_json=json.dumps(parsed), brief=brief, agent=agent),
                    response_type=TouchReport, timeout=SPECIALIST_TIMEOUT_S,
                )
                if isinstance(reply, TouchReport):
                    return json.loads(reply.result_json)
                reason = getattr(status, "detail", None) or str(status)
            except Exception as e:
                reason = str(e)
            # Recovery: same code, run here, and say so.
            ctx.logger.warning(f"{key} didn't answer ({reason}); running it in the hub")
            r = await specialist.respond(BY_KEY[key], q, parsed, brief)
            if r.get("markdown"):
                r["markdown"] = (f"_{BY_KEY[key].title} didn't answer ({reason}); the hub ran its "
                                 "analysis directly._\n\n" + r["markdown"])
            return r

        async def ask_agent(key: str, question: str) -> str:
            if key not in BY_KEY:
                return f"No agent called {key}."
            r = await ask(key, question, {}, False, agent=True)
            return f"**{BY_KEY[key].title}** answered:\n\n{r.get('markdown', '')}"

        async def report() -> str:
            return await hub.respond(text, ask)

        return await converse(ctx, sender, "hub", text, report, ask_agent)

    return [chat_protocol(answer)]


def protocols_for(key: str, addresses: dict[str, str] | None = None) -> list[Protocol]:
    return hub_protocols(addresses or {}) if key == HUB_KEY else specialist_protocols(BY_KEY[key])


def make_agent(key: str) -> Agent:
    name, title, body = identity_of(key)
    port = port_for(key)
    return Agent(
        name=name,
        seed=seed_for(key),
        port=port,
        endpoint=None if config.MAILBOX else [f"http://127.0.0.1:{port}/submit"],
        mailbox=config.MAILBOX,
        readme_path=write_readme(key, title, body),
        description=title,
        publish_agent_details=True,
        handle_messages_concurrently=True,
    )


def all_keys() -> list[str]:
    return [HUB_KEY] + [p.key for p in PROFILES]


def build(key: str) -> Agent:
    """A local agent (run.py). Locally every specialist runs as its own agent, so the hub asks them all."""
    agent = make_agent(key)
    addresses = {p.key: address_for(p.key) for p in PROFILES} if key == HUB_KEY else None
    for proto in protocols_for(key, addresses):
        agent.include(proto, publish_manifest=True)
    return agent

