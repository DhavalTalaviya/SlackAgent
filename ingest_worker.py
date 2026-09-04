import time

from src.config import config
from src.infra.health import heartbeat, start_health_server
from src.ingestion.graph import run_ingest

if __name__ == "__main__":
    # Heartbeat records once per loop pass, success or failure, so the check
    # reflects "the loop is alive" (restart-worthy if it stalls), not "the
    # last ingest succeeded" (a bad credential shouldn't trigger a crash-loop
    # restart -- it needs a human, not a restart).
    start_health_server(config.health_check_port, max_heartbeat_age=2 * config.ingest_interval_seconds + 60)

    print(f"Starting ingestion worker. Polling every {config.ingest_interval_seconds}s. Ctrl+C to stop.")
    while True:
        try:
            count = run_ingest()
            print(f"[ingest_worker] {count} chunks embedded and stored this run.")
        except Exception as e:
            print(f"[ingest_worker] run failed, will retry next cycle: {e!r}")
        heartbeat()
        time.sleep(config.ingest_interval_seconds)
