import json
import os

from src.config import config

_STATE_FILE = os.path.join(config.workspace_state_dir, "ingest_state.json")

_DEFAULT_STATE = {
    "slack": {},   # channel_id -> latest message ts already ingested
}


def load_state() -> dict:
    if not os.path.exists(_STATE_FILE):
        return json.loads(json.dumps(_DEFAULT_STATE))
    with open(_STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict) -> None:
    os.makedirs(config.workspace_state_dir, exist_ok=True)
    with open(_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
