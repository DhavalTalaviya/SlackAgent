from datetime import datetime, timezone

from langchain_core.documents import Document
from slack_sdk.errors import SlackApiError

from src.ingestion.chunking import chunk_document
from src.slack.ingest import _call, _user_name_cache
from src.vectorstore import get_vectorstore

_channel_name_cache: dict[str, str] = {}


def _get_channel_name(client, channel_id: str) -> str:
    if channel_id not in _channel_name_cache:
        try:
            info = _call(client, client.conversations_info, channel=channel_id)
            _channel_name_cache[channel_id] = info["channel"].get("name", channel_id)
        except SlackApiError:
            _channel_name_cache[channel_id] = channel_id
    return _channel_name_cache[channel_id]


def sync_thread(client, channel_id: str, root_ts: str) -> None:
    """Re-sync the ingested document for one thread (or standalone message)
    after an edit or delete, so a stale/removed version never lingers in the
    vector store. Always deletes the old version first; re-adds the current
    version only if the thread still has content."""
    vectorstore = get_vectorstore()
    vectorstore.delete(where={"$and": [{"channel_id": {"$eq": channel_id}}, {"ts": {"$eq": root_ts}}]})

    try:
        resp = _call(client, client.conversations_replies, channel=channel_id, ts=root_ts, limit=200)
    except SlackApiError as e:
        print(f"[slack_sync] {channel_id}/{root_ts} no longer accessible ({e}), leaving it deleted")
        return

    messages = resp.get("messages", [])
    if not messages:
        print(f"[slack_sync] {channel_id}/{root_ts} deleted, removed from the index")
        return

    resolve_user = _user_name_cache(client)
    root_msg = messages[0]
    text = root_msg.get("text", "")
    if len(messages) > 1:
        text += "\n" + "\n".join(
            f"{resolve_user(m.get('user'))}: {m.get('text', '')}" for m in messages[1:]
        )

    if not text.strip():
        print(f"[slack_sync] {channel_id}/{root_ts} is now empty, removed from the index")
        return

    channel_name = _get_channel_name(client, channel_id)
    when = datetime.fromtimestamp(float(root_ts), tz=timezone.utc).isoformat()
    doc = Document(
        page_content=f"[#{channel_name}] {resolve_user(root_msg.get('user'))} ({when}):\n{text}",
        metadata={
            "source": "slack",
            "channel_id": channel_id,
            "channel_name": channel_name,
            "ts": root_ts,
            "ts_epoch": float(root_ts),
            "user": resolve_user(root_msg.get("user")),
            "permalink": f"https://slack.com/archives/{channel_id}/p{root_ts.replace('.', '')}",
        },
    )
    chunks = chunk_document(doc)
    if chunks:
        vectorstore.add_documents(chunks)
    print(f"[slack_sync] re-synced {channel_id}/{root_ts} -> {len(chunks)} chunks")
