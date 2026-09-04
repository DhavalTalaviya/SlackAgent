import re

from slack_bolt import App

from src.agent.qa_graph import ask
from src.config import config
from src.slack.sync import sync_thread

app = App(token=config.slack_bot_token)

_MENTION_RE = re.compile(r"<@[A-Z0-9]+>\s*")
_THINKING_REACTION = "hourglass_flowing_sand"


def _format_reply(result: dict) -> str:
    text = result["answer"]
    if result["citations"]:
        sources = "\n".join(
            f"<{c['permalink']}|#{c['channel']} - {c['user']}>" for c in result["citations"]
        )
        text += f"\n\n*Sources:*\n{sources}"
    return text


def _handle_question(user_id: str, question: str, channel: str, thread_ts: str, client) -> None:
    try:
        client.reactions_add(channel=channel, timestamp=thread_ts, name=_THINKING_REACTION)
    except Exception:
        pass

    result = ask(question, requesting_user_id=user_id, thread_id=f"user:{user_id}")

    try:
        client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=_format_reply(result))
    except Exception as e:
        print(f"[slack_bot] failed to post answer, retrying with a plain message: {e!r}")
        try:
            client.chat_postMessage(
                channel=channel, thread_ts=thread_ts, text="Sorry, something went wrong posting my answer."
            )
        except Exception as e2:
            print(f"[slack_bot] retry also failed: {e2!r}")

    try:
        client.reactions_remove(channel=channel, timestamp=thread_ts, name=_THINKING_REACTION)
    except Exception:
        pass


@app.event("app_mention")
def handle_mention(event: dict, client) -> None:
    question = _MENTION_RE.sub("", event.get("text", "")).strip()
    if not question:
        return
    thread_ts = event.get("thread_ts", event["ts"])
    _handle_question(event["user"], question, event["channel"], thread_ts, client)


@app.event("message")
def handle_message_event(event: dict, client) -> None:
    subtype = event.get("subtype")

    if subtype == "message_changed":
        new_msg = event.get("message", {})
        root_ts = new_msg.get("thread_ts", new_msg.get("ts"))
        if root_ts:
            try:
                sync_thread(client, event["channel"], root_ts)
            except Exception as e:
                print(f"[slack_bot] sync on edit failed for {event['channel']}/{root_ts}: {e!r}")
        return

    if subtype == "message_deleted":
        prev_msg = event.get("previous_message", {})
        root_ts = prev_msg.get("thread_ts", prev_msg.get("ts"))
        if root_ts:
            try:
                sync_thread(client, event["channel"], root_ts)
            except Exception as e:
                print(f"[slack_bot] sync on delete failed for {event['channel']}/{root_ts}: {e!r}")
        return

    if subtype or event.get("bot_id"):
        return  # other subtypes (channel_join, etc.) and bot messages aren't questions

    if event.get("channel_type") != "im":
        return  # only treat DMs as questions; channel messages need an @mention

    question = event.get("text", "").strip()
    if not question:
        return
    thread_ts = event.get("thread_ts", event["ts"])
    _handle_question(event["user"], question, event["channel"], thread_ts, client)
