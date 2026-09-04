from slack_bolt.adapter.socket_mode import SocketModeHandler

from src.config import config
from src.infra.health import start_health_server
from src.slack.bot import app

if __name__ == "__main__":
    # A Socket Mode bot has no inbound HTTP of its own; this gives container
    # orchestrators something to poll for liveness. Long-lived, so a large
    # max_heartbeat_age is fine -- the process being up is the health signal.
    start_health_server(config.health_check_port, max_heartbeat_age=3600)

    print("Starting Slack bot (Socket Mode). Mention the bot in a channel or DM it to ask a question.")
    SocketModeHandler(app, config.slack_app_token).start()
