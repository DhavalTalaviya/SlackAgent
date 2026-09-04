from langchain_core.callbacks import BaseCallbackHandler
from prometheus_client import Counter, Gauge, Histogram

# --- liveness --------------------------------------------------------------
# Separate from the /healthz HTTP check the orchestrator polls: that check's
# own HTTP server thread can stay up even if the main loop hangs, so this
# gives Prometheus/Alertmanager the same heartbeat signal for human alerting,
# not just container restarts.
LAST_HEARTBEAT_TIMESTAMP = Gauge(
    "process_last_heartbeat_timestamp_seconds", "Unix timestamp of the last heartbeat"
)

# --- latency -----------------------------------------------------------
ASK_LATENCY = Histogram(
    "ask_latency_seconds",
    "End-to-end latency of one ask() turn",
    buckets=(1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120),
)
NODE_LATENCY = Histogram(
    "qa_node_latency_seconds", "Latency of each QA graph node", ["node"]
)

# --- error rate ----------------------------------------------------------
NODE_FAILURES = Counter(
    "qa_node_failures_total",
    "Node-level failures (includes ones that degraded gracefully, not just fatal ones)",
    ["node"],
)
REQUESTS_TOTAL = Counter(
    "ask_requests_total", "Total ask() calls by final outcome", ["outcome"]
)
RATE_LIMIT_BLOCKED = Counter(
    "rate_limit_blocked_total", "Requests blocked by the per-user rate limiter", ["reason"]
)

# --- retrieval quality (proxies -- no ground-truth relevance labels here) ---
RETRIEVAL_CANDIDATES = Histogram(
    "retrieval_candidates_count",
    "Number of candidate chunks retrieved per turn",
    buckets=(0, 1, 2, 5, 10, 15, 20, 30),
)
RETRIEVAL_EMPTY = Counter(
    "retrieval_empty_total", "Turns where retrieval found nothing at all"
)
CONFLICT_FLAGGED = Counter(
    "conflict_flagged_total", "Turns where conflict_check found disagreeing sources"
)

# --- spend -----------------------------------------------------------------
ANTHROPIC_TOKENS = Counter(
    "anthropic_tokens_total", "Anthropic tokens consumed", ["model", "kind"]
)
ANTHROPIC_COST_USD = Counter(
    "anthropic_cost_usd_total", "Estimated Anthropic spend in USD (list price, not invoice-accurate)", ["model"]
)
VOYAGE_RERANK_TOKENS = Counter(
    "voyage_rerank_tokens_total", "Tokens consumed by the Voyage rerank API", ["model"]
)
# No dollar-cost counter for rerank tokens -- unlike the Anthropic table
# above, sourced from a verified live pricing reference this session, current
# Voyage rerank pricing wasn't independently confirmed here. Check Voyage's
# pricing page and add a _RERANK_PRICING_PER_MTOK table + a
# voyage_rerank_cost_usd_total counter the same way if you want it tracked.

# USD per 1M tokens (input, output). Update if pricing changes -- this is an
# estimate for dashboards/alerting, not a substitute for your actual invoice.
_PRICING_PER_MTOK = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
_DEFAULT_PRICING = (5.00, 25.00)


def _record_llm_usage(model: str, usage: dict) -> None:
    input_tokens = usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0
    if not input_tokens and not output_tokens:
        return

    ANTHROPIC_TOKENS.labels(model=model, kind="input").inc(input_tokens)
    ANTHROPIC_TOKENS.labels(model=model, kind="output").inc(output_tokens)

    in_price, out_price = _PRICING_PER_MTOK.get(model, _DEFAULT_PRICING)
    cost = (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price
    ANTHROPIC_COST_USD.labels(model=model).inc(cost)


class UsageMetricsCallback(BaseCallbackHandler):
    """Attach to every ChatAnthropic instance so token spend is captured
    transparently -- this fires at the underlying chat-model call, before any
    with_structured_output() wrapping strips the raw message away, so it
    works identically for plain and structured-output calls without either
    call site needing to know about metrics."""

    def __init__(self, model: str):
        self._model = model

    def on_llm_end(self, response, **kwargs) -> None:
        for generation_list in response.generations:
            for generation in generation_list:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None)
                if usage:
                    _record_llm_usage(self._model, usage)
