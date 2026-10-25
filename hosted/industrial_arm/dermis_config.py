"""Runtime configuration, all from environment variables (see .env.example)."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# Local copy of the Dermis sim dataset. Used when SpacetimeDB is not configured or unreachable.
DATA_DIR = Path(
    os.getenv("DERMIS_DATA_DIR", ROOT.parent / "dermis-teleop" / "dataset" / "dermis_sim_v1")
)

# SpacetimeDB (HTTP SQL API). Leave SPACETIME_DB empty to use the local dataset only.
SPACETIME_HOST = os.getenv("SPACETIME_HOST", "https://maincloud.spacetimedb.com").rstrip("/")
SPACETIME_DB = os.getenv("SPACETIME_DB", "")
SPACETIME_TOKEN = os.getenv("SPACETIME_TOKEN", "")
SPACETIME_EPISODE_TABLE = os.getenv("SPACETIME_EPISODE_TABLE", "episode")
SPACETIME_FRAME_TABLE = os.getenv("SPACETIME_FRAME_TABLE", "frame")

# ASI:One LLM API (OpenAI-compatible). Without a key, agents use rule-based parsing and summaries.
ASI1_API_KEY = os.getenv("ASI1_API_KEY", "")
ASI1_BASE_URL = os.getenv("ASI1_BASE_URL", "https://api.asi1.ai/v1").rstrip("/")
ASI1_MODEL = os.getenv("ASI1_MODEL", "asi1-mini")

# Agent identity. Seeds derive the agents' addresses, so keep AGENT_SEED_SECRET private and stable.
AGENT_SEED_SECRET = os.getenv("AGENT_SEED_SECRET", "")
MAILBOX = os.getenv("DERMIS_MAILBOX", "0") == "1"
BUREAU_PORT = int(os.getenv("DERMIS_PORT", "8000"))

# Charts are rendered by QuickChart from a URL that carries the (downsampled) series.
CHARTS = os.getenv("DERMIS_CHARTS", "1") == "1"
QUICKCHART_URL = os.getenv("QUICKCHART_URL", "https://quickchart.io/chart")

# Precomputed analysis every agent reads (built by `python -m dermis_agents.snapshot`). Hosted agents
# have no local file and fetch SNAPSHOT_URL instead.
SNAPSHOT_PATH = Path(os.getenv("DERMIS_SNAPSHOT_PATH", ROOT / "snapshot" / "dermis_snapshot.json"))
SNAPSHOT_URL = os.getenv("DERMIS_SNAPSHOT_URL", "")
