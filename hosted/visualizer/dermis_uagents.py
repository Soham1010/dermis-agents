"""The uAgents layer: 13 specialist agents + the Dermis Hub, all speaking the Agent Chat Protocol.

Every agent is reachable from ASI:One on its own. The hub also fans questions out to specialists over a
typed internal protocol (TouchQuery -> TouchReport), and if a specialist doesn't answer in time it runs
that specialist's analysis itself, so a dead agent degrades the answer instead of breaking it.
"""

from __future__ import annotations

import asyncio
import json
import re
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

import dermis_brain as brain, dermis_cards as cards, dermis_config as config, dermis_hub as hub, dermis_payments as payments, dermis_specialist as specialist
from dermis_profiles import BY_KEY, PROFILES, Profile

HUB_KEY = "hub"
GREETING = re.compile(r"^\s*(hi|hello|hey|yo|help|start|get started|what can you do|menu)\b[\s!?.]*$", re.I)
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
        reply = reply if isinstance(reply, dict) else {"text": reply}
        content = [TextContent(type="text", text=reply["text"])]
        if reply.get("card"):
            try:
                content.append(cards.content(reply["card"]))
            except ValueError:
                ctx.logger.warning("card too large; sending text only")
        if len(content) == 1:  # a card waits for a button press, so only plain answers end the session
            content.append(EndSessionContent(type="end-session"))
        await ctx.send(sender, ChatMessage(content=content))

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


async def converse(ctx: Context, sender: str, key: str, text: str, fallback, ask=None) -> dict:
    """LLM agent turn with this sender's history and session; the structured report if the LLM is down."""
    if key == HUB_KEY:
        sel = cards.parse_selection(text)
        if sel and sel["action"] in ("confirm_purchase", "cancel_purchase"):
            return await purchase_step(ctx, sender, sel)
    history = ctx.storage.get(f"history:{sender}") or []
    session = ctx.storage.get(f"session:{sender}") or {}
    result = await brain.converse(key, text, history, ask, session)
    if result is None:
        ctx.logger.warning("ASI:One unavailable; answering with the structured report")
        result = {"text": await fallback(), "card": None, "purchase": None}
    if result.get("purchase") and key == HUB_KEY:
        result["card"] = cards.purchase_review(result["purchase"], payments.PRICE_FET, payments.NETWORK)
    elif key == HUB_KEY and not result.get("card") and GREETING.match(text):
        result["card"] = cards.start_form()  # first impressions shouldn't depend on the model's mood
    ctx.storage.set(f"history:{sender}", brain.remember(history, text, result["text"]))
    ctx.storage.set(f"session:{sender}", session)
    return result


async def purchase_step(ctx: Context, sender: str, sel: dict) -> dict:
    """Pay / Not now on the review card. Handled without the LLM: money paths stay deterministic."""
    episode = sel.get("episode")
    if sel["action"] == "cancel_purchase":
        return {"text": "No problem, nothing was charged. Ask me anything else."}
    if not episode:
        return {"text": "Which recording should the certified report cover? (e.g. episode 4)"}
    if not await payments.request_payment(ctx, sender, episode):
        return {"text": "Payments aren't available on this agent right now (no wallet). Everything else still works."}
    return {"text": f"Payment request sent: {payments.PRICE_FET} FET on {payments.NETWORK} for the certified report on "
                    f"{episode}. Approve it in your wallet; I verify the transaction on-chain and send the report."}


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
                r = await brain.converse(p.key, q.text)
                md = r["text"] if r else (await specialist.respond(p, q.text))["markdown"]
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
                    r = await brain.converse(key, q)
                    return {"markdown": r["text"] if r else (await specialist.respond(BY_KEY[key], q))["markdown"]}
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

        return await converse(ctx, sender, HUB_KEY, text, report, ask_agent)

    async def deliver(episode: str, receipt: str) -> str:
        return await asyncio.to_thread(specialist.certified_report, episode, receipt)

    return [chat_protocol(answer), payments.payment_protocol(deliver)]


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
    if key == HUB_KEY:
        payments.set_wallet(agent.wallet)
    addresses = {p.key: address_for(p.key) for p in PROFILES} if key == HUB_KEY else None
    for proto in protocols_for(key, addresses):
        agent.include(proto, publish_manifest=True)
    return agent

