import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass
class Config:
    slack_bot_token: str = os.environ.get("SLACK_BOT_TOKEN", "")
    slack_app_token: str = os.environ.get("SLACK_APP_TOKEN", "")
    slack_channel_ids: list[str] = field(
        default_factory=lambda: _split_csv(os.environ.get("SLACK_CHANNEL_IDS", ""))
    )

    voyage_api_key: str = os.environ.get("VOYAGE_API_KEY", "")
    voyage_rerank_model: str = os.environ.get("VOYAGE_RERANK_MODEL", "rerank-2.5")

    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
    # Used for the cheap classification/rewriting steps (not final answer
    # synthesis) -- those don't need Opus-level reasoning, and a smaller
    # model cuts several seconds of latency off each of those calls.
    anthropic_fast_model: str = os.environ.get("ANTHROPIC_FAST_MODEL", "claude-haiku-4-5")

    chroma_persist_dir: str = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    chroma_collection_name: str = os.environ.get("CHROMA_COLLECTION_NAME", "team_knowledge")

    state_dir: str = os.environ.get("STATE_DIR", "./state")

    # Identifies this deployment's tenant (Slack workspace/client). Every
    # per-tenant data path -- the Chroma collection and all state files --
    # is namespaced by this so two clients can never silently share data,
    # even if their other env vars happen to collide.
    workspace_id: str = os.environ.get("WORKSPACE_ID", "default")

    ingest_interval_seconds: int = int(os.environ.get("INGEST_INTERVAL_SECONDS", "300"))

    rate_limit_burst_max: int = int(os.environ.get("RATE_LIMIT_BURST_MAX", "5"))
    rate_limit_burst_window_seconds: int = int(os.environ.get("RATE_LIMIT_BURST_WINDOW_SECONDS", "60"))
    rate_limit_daily_max: int = int(os.environ.get("RATE_LIMIT_DAILY_MAX", "50"))

    health_check_port: int = int(os.environ.get("HEALTH_CHECK_PORT", "8080"))

    @property
    def workspace_collection_name(self) -> str:
        return f"{self.chroma_collection_name}__{self.workspace_id}"

    @property
    def workspace_state_dir(self) -> str:
        return os.path.join(self.state_dir, self.workspace_id)


config = Config()
