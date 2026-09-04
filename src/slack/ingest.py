import time
from datetime import datetime, timezone

from langchain_core.documents import Document
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from src.config import config


def _call(client: WebClient, method, **kwargs):
    """Call a Slack Web API method, retrying once on rate limits."""
    while True:
        try:
            return method(**kwargs)
        except SlackApiError as e:
            if e.response.status_code == 429:
                delay = int(e.response.headers.get("Retry-After", "5"))
                time.sleep(delay)
                continue
            raise


def _list_channels(client: WebClient) -> list[dict]:
    if config.slack_channel_ids:
        return [{"id": cid, "name": cid} for cid in config.slack_channel_ids]

    channels = []
    cursor = None
    while True:
        resp = _call(
            client,
            client.conversations_list,
            types="public_channel",
            exclude_archived=True,
            limit=200,
            cursor=cursor,
        )
        channels.extend(resp["channels"])
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return [c for c in channels if c.get("is_member")]


def _user_name_cache(client: WebClient) -> dict:
    cache: dict[str, str] = {}

    def resolve(user_id: str) -> str:
        if not user_id:
            return "unknown"
        if user_id not in cache:
            try:
                info = _call(client, client.users_info, user=user_id)
                profile = info["user"]
                cache[user_id] = profile.get("real_name") or profile.get("name") or user_id
            except SlackApiError:
                cache[user_id] = user_id
        return cache[user_id]

    return resolve


def _fetch_thread_replies(client: WebClient, channel_id: str, thread_ts: str, resolve_user) -> str:
    lines = []
    cursor = None
    while True:
        resp = _call(
            client,
            client.conversations_replies,
            channel=channel_id,
            ts=thread_ts,
            cursor=cursor,
            limit=200,
        )
        for msg in resp["messages"]:
            if msg.get("ts") == thread_ts or msg.get("bot_id"):
                continue
            lines.append(f"{resolve_user(msg.get('user'))}: {msg.get('text', '')}")
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return "\n".join(lines)


def fetch_slack_documents(state: dict) -> tuple[list[Document], dict]:
    """Fetch new Slack messages since the last recorded cursor per channel.

    Returns the documents plus the updated slack portion of the ingest state.
    """
    if not config.slack_bot_token:
        return [], state.get("slack", {})

    client = WebClient(token=config.slack_bot_token)
    resolve_user = _user_name_cache(client)
    slack_state = dict(state.get("slack", {}))

    documents: list[Document] = []
    for channel in _list_channels(client):
        channel_id = channel["id"]
        channel_name = channel.get("name", channel_id)
        oldest = slack_state.get(channel_id, "0")
        latest_seen = oldest
        cursor = None

        while True:
            resp = _call(
                client,
                client.conversations_history,
                channel=channel_id,
                oldest=oldest,
                cursor=cursor,
                limit=200,
            )
            for msg in resp["messages"]:
                # Advance the cursor past every message we see, skipped or
                # not -- otherwise a skipped message sitting at the tail of
                # a channel (very likely for a bot reply) gets re-fetched
                # and re-skipped on every future ingestion run forever.
                ts = msg["ts"]
                if ts > latest_seen:
                    latest_seen = ts

                if msg.get("subtype") in {"channel_join", "channel_leave"}:
                    continue
                if msg.get("bot_id"):
                    # Never ingest the bot's own replies (or other bots')
                    # back into the knowledge base -- otherwise a past "I
                    # couldn't find anything" answer becomes a retrievable
                    # "source" that can even get flagged as contradicting
                    # real data.
                    continue

                text = msg.get("text", "")
                if msg.get("thread_ts") == ts and msg.get("reply_count", 0) > 0:
                    replies = _fetch_thread_replies(client, channel_id, ts, resolve_user)
                    if replies:
                        text = f"{text}\n{replies}"

                if not text.strip():
                    continue

                when = datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
                documents.append(
                    Document(
                        page_content=f"[#{channel_name}] {resolve_user(msg.get('user'))} ({when}):\n{text}",
                        metadata={
                            "source": "slack",
                            "channel_id": channel_id,
                            "channel_name": channel_name,
                            "ts": ts,
                            "ts_epoch": float(ts),
                            "user": resolve_user(msg.get("user")),
                            "permalink": f"https://slack.com/archives/{channel_id}/p{ts.replace('.', '')}",
                        },
                    )
                )

            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        slack_state[channel_id] = latest_seen

    return documents, slack_state
