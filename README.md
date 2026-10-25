# Dermis Agents: robot touch intelligence on Fetch.ai

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

Fourteen uAgents that turn Dermis tactile-glove recordings into decisions for specific robotics
applications. Ask in ASI:One, for example “compare all applications on episode 4”. The hub then
has ten specialists judge the same recording against their own force limits and failure tolerances,
and returns a chart, a readiness score per application and the first thing to change.

Built for the Fetch.ai track at Cal Hacks 13.0
([hackpack](https://www.fetch.ai/events/hackathons/calhacks-13-0/hackpack)).

## The agents

| Key | Agent name | What it judges |
|---|---|---|
| `hub` | dermis_hub | Front door. Routes to specialists, fans out, combines, compares across all applications |
| `humanoid_arm` | dermis_humanoid_arm | Dexterous humanoid hands: grasp stability, fragile objects, finger timing |
| `industrial_arm` | dermis_industrial_arm | Production grippers: yield, repeatability, headroom (fleet view first) |
| `quadruped` | dermis_quadruped | Robot dogs: foot contact timing, foot slip, touchdown impact |
| `prosthetic_hand` | dermis_prosthetics | Prosthetics: never crush, never drop, a grip state the wearer can be told |
| `surgical` | dermis_surgical | Delicate handling: 50% safety factor on yield, noise floor |
| `warehouse` | dermis_warehouse | Pick and place: drops, slip on heavy items, impact (fleet view first) |
| `food_agri` | dermis_food_agri | Produce: crushing and bruising |
| `eldercare` | dermis_eldercare | Care robots: hand-over safety, over-pressure, predictable release |
| `teleop` | dermis_teleop | The glove itself: touch detection latency, what the operator feels |
| `sensor_qa` | dermis_sensor_qa | Sensor health: saturation, noise vs spec, dropped frames. Separates sensor faults from grip failures |
| `data_curator` | dermis_data_curator | Training data: label balance, coverage gaps, what to record next, baseline F1 |
| `data_explorer` | dermis_data_explorer | Pandas-style tables: overview, crosstab, pivot, group-by any metric, per-episode stats and calibration fit. Shows the pandas expression behind each table |
| `visualizer` | dermis_visualizer | Charts: dashboards, signal time series, SafeGrip overlays, histograms, scatter with Pearson r, pie, radar, heatmap grid. Numbers under every chart |

Print the addresses with `python run.py --addresses`. They are derived from `AGENT_SEED_SECRET`
(or `.agent_seed`), so share that secret privately within the team to get the same addresses everywhere.

<!-- Fill in after the first mailbox run, for judges: -->
**Hub address:** `agent1q…` · **Agentverse profile:** https://agentverse.ai/agents/details/agent1q…

## How it works

```
ASI:One ──ACP──▶ Dermis Hub ──TouchQuery──▶ 10 specialists (parallel) ──TouchReport──▶ Hub ──▶ ASI:One
                   │                              │
                   │ ASI:One LLM: parse + route   └─ data.py: SpacetimeDB, else local dataset
                   │ (rule-based fallback)           analysis.py: numpy on the 350 Hz frames
                   └─ if a specialist is down, the hub runs its analysis itself and says so
```

- **Every agent speaks the Agent Chat Protocol**, so each one is usable on its own from ASI:One,
  not only through the hub.
- **The analysis is real computation on the recordings, not LLM text.** It covers per-pod force,
  touch-onset latency against true contact, slip-band vibration (high-passed field energy),
  magnetometer rail, robust noise against the MLX90393 spec, dropped frames and friction margin. It
  also finds the SafeGrip on/off twin of the same run to show what would have changed.
- **The LLM (ASI:One `asi1-mini`) only parses requests, routes them and writes the interpretation**
  from the computed facts. It is told never to add numbers. Without `ASI1_API_KEY`, rules take over.
- **Charts** are QuickChart image URLs, so they render inline in ASI:One. The chart URL contains
  the downsampled force series. Set `DERMIS_CHARTS=0` to leave the charts out.
- **The profiles** in `dermis_agents/profiles.py` set each domain's force safety factor, friction
  margin floor, critical faults, check priority, latency tolerance and recommended actions. To add an
  application, add one `Profile`.

## Run it

Use Python 3.10 to 3.13. uagents doesn't run on 3.14.

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env                       # optional: SpacetimeDB, ASI1_API_KEY

# 1. No network at all: hub + specialists in-process
.venv/bin/python -m dermis_agents.cli "compare all applications on episode 4"
.venv/bin/python -m dermis_agents.cli quadruped "check slip for the robot dog"

# 2. Real agents, real Chat Protocol, locally
.venv/bin/python run.py                    # terminal 1: all 14, one process each (hub :8000, others :8001-8013)
.venv/bin/python ask.py "which items do we drop most in the warehouse?"   # terminal 2
.venv/bin/python ask.py --to sensor_qa "sensor health check across the fleet"

# 3. Live on Agentverse / ASI:One
DERMIS_MAILBOX=1 .venv/bin/python run.py
#    first run: open each printed inspector link → Connect → Mailbox
#    then find "Dermis Hub" on agentverse.ai or ASI:One (asi1.ai) and chat

.venv/bin/python -m pytest -q tests
```

Questions to try in ASI:One:

- Analyze episode 4 for a humanoid arm
- Compare all applications on episode 4
- Which items do we drop most in the warehouse?
- Are any sensor readings unreliable? Sensor health check across the fleet
- Would a prosthetic hand drop the cup?
- What data should we collect next for training?
- Pivot table of success by scenario · Peak force by object and scenario
- Dashboard · Histogram of peak force · Scatter latency vs peak force · Heatmap of success
- Compare episode 4 with SafeGrip on · Plot the raw field for episode 4 · Radar of episode 40
- Help

## SpacetimeDB contract

`data.py` reads over SpacetimeDB's HTTP SQL API (`POST /v1/database/{db}/sql`). It uses the local
dataset whenever `SPACETIME_DB` is unset or the query fails, and the reply says which source it used.

| Table | Columns |
|---|---|
| `episode` | `episode_id` (pk), `application` (`hand`, `quadruped`, …), `source` (`sim`/`hardware`), `object`, `mass_g`, `mu`, `scenario`, `safe_grip`, `seed`, `frames`, `seconds`, `outcome`, `faults` (JSON text), `first_fault_frame` |
| `frame` | `episode_id`, `frame`, then the dermis-sim-v1 column names (`t_device_us`, `index_tip_bx_ut` … `label_fault`) |

Hardware sessions can leave out the ground-truth columns (`*_normal_n`, labels, `object_lift_m`).
The analysis then falls back to the field-based force estimate and marks it uncalibrated.

## What the numbers are

The local data is `dermis-sim-v1`: 192 synthetic episodes in the real hub's column layout. Force is
simulated ground truth. Field-to-force uses an assumed 1.5 mT/N until the load-cell calibration
(FORCE_REF) is done. Object yield forces are engineering estimates (can 18 N, egg 25 N, cup 4 N).
The agents state all of this in their replies. The quadruped agent reads hand recordings with pods
mapped to feet until quadruped data exists, and it says so.

## Layout

```
dermis_agents/
  profiles.py    the 13 agents' domains, limits and keywords
  data.py        SpacetimeDB + local dataset, same shapes
  analysis.py    numpy analysis, per-profile evaluation, fleet stats, SafeGrip twin
  specialist.py  episode / fleet / sensor-scan / data-curation reports
  hub.py         routing, fan-out, cross-application matrix
  llm.py         ASI:One LLM client with rule fallbacks
  viz.py         Data Explorer (pandas tables) and Visualizer (chart picker)
  charts.py      QuickChart URLs + text sparklines
  agents.py      uAgents: Chat Protocol, internal TouchQuery protocol, recovery
run.py           starts one process per agent (local or mailbox)
ask.py           Chat Protocol test client
examples/        the original two-agent starter
```
