import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.infra.metrics import LAST_HEARTBEAT_TIMESTAMP

_last_heartbeat = time.time()
_lock = threading.Lock()


def heartbeat() -> None:
    """Call this periodically from a long-running process's main loop.
    A process that hangs (deadlock, stuck call) stops calling this, so its
    health check goes stale and the orchestrator knows to restart it -- a
    plain "is the port open" check wouldn't catch that."""
    global _last_heartbeat
    with _lock:
        _last_heartbeat = time.time()
    LAST_HEARTBEAT_TIMESTAMP.set(_last_heartbeat)


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/healthz":
            with _lock:
                age = time.time() - _last_heartbeat
            healthy = age < self.server.max_heartbeat_age  # type: ignore[attr-defined]
            self.send_response(200 if healthy else 503)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok" if healthy else f"stale ({age:.0f}s)".encode())
            return

        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(generate_latest())
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        pass  # keep container logs free of health-check/scrape noise


def start_health_server(port: int, max_heartbeat_age: float = 120.0) -> None:
    """Starts a background HTTP server exposing GET /healthz (for container
    orchestrator liveness/readiness probes) and GET /metrics (for Prometheus
    to scrape) on one port. Records an initial heartbeat so the process is
    healthy from the moment it comes up, even before its main loop runs once."""
    heartbeat()
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    server.max_heartbeat_age = max_heartbeat_age  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
