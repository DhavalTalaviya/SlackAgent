import time

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from src.config import config
from src.ingestion.state import load_state
from src.slack.ingest import _call

_MEMBERSHIP_TTL_SECONDS = 300

_membership_cache: dict[str, tuple[float, set[str]]] = {}


def _get_channel_members(client: WebClient, channel_id: str) -> set[str]:
    cached = _membership_cache.get(channel_id)
    if cached and time.time() - cached[0] < _MEMBERSHIP_TTL_SECONDS:
        return cached[1]

    members: set[str] = set()
    cursor = None
    while True:
        resp = _call(client, client.conversations_members, channel=channel_id, cursor=cursor, limit=200)
        members.update(resp["members"])
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    _membership_cache[channel_id] = (time.time(), members)
    return members


def get_allowed_channel_ids(user_id: str) -> list[str]:
    """Channels the given Slack user is currently a member of, restricted to
    channels we've actually ingested. Fails closed: any channel we can't
    verify membership for is excluded, never included by default."""
    if not user_id:
        return []

    ingested_channel_ids = list(load_state().get("slack", {}).keys())
    if not ingested_channel_ids:
        return []

    client = WebClient(token=config.slack_bot_token)
    allowed = []
    for channel_id in ingested_channel_ids:
        try:
            if user_id in _get_channel_members(client, channel_id):
                allowed.append(channel_id)
        except SlackApiError:
            continue

    return allowed
