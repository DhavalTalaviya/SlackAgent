import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from typing import Annotated, Literal, Optional, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langgraph.checkpoint.base import BaseCheckpointSaver

from src.agent.prompts import (
    ANALYZE_AND_REWRITE,
    CONTEXTUALIZE_PRIOR_RANGE,
    CONTEXTUALIZE_QUESTION,
    SYNTHESIZE_ANSWER,
)
from src.agent.reranker import rerank
from src.config import config
from src.infra.metrics import (
    ASK_LATENCY,
    CONFLICT_FLAGGED,
    NODE_FAILURES,
    NODE_LATENCY,
    RATE_LIMIT_BLOCKED,
    REQUESTS_TOTAL,
    RETRIEVAL_CANDIDATES,
    RETRIEVAL_EMPTY,
    UsageMetricsCallback,
)
from src.infra.rate_limiter import check_and_record
from src.slack.access_control import get_allowed_channel_ids
from src.vectorstore import get_vectorstore

_RETRIEVE_K = 8
_MAX_CANDIDATES = 20
_RERANK_KEEP = 6
_CHECKPOINT_DB = os.path.join(config.workspace_state_dir, "conversations.sqlite")

_usage_callback = UsageMetricsCallback(config.anthropic_model)
_fast_usage_callback = UsageMetricsCallback(config.anthropic_fast_model)


def _llm(effort: str = "high", fast: bool = False) -> ChatAnthropic:
    """fast=True routes to the smaller ANTHROPIC_FAST_MODEL (Haiku by
    default) for classification/rewriting steps that don't need Opus-level
    reasoning -- meaningfully lower latency per call. Final answer synthesis
    stays on the main model."""
    kwargs = dict(
        model=config.anthropic_fast_model if fast else config.anthropic_model,
        api_key=config.anthropic_api_key,
        max_tokens=4096,
        callbacks=[_fast_usage_callback if fast else _usage_callback],
    )
    if not fast:
        # The default fast model (Haiku 4.5) doesn't support the effort
        # parameter at all -- Anthropic returns a 400 if it's sent. Only
        # newer models (Opus 5, Sonnet 5, ...) support effort tuning.
        kwargs["effort"] = effort
    return ChatAnthropic(**kwargs)


def _timed_node(name: str):
    """Records per-node latency uniformly, without touching what each node
    returns or how it handles its own errors internally."""

    def decorator(fn):
        @wraps(fn)
        def wrapper(state):
            start = time.monotonic()
            try:
                return fn(state)
            finally:
                NODE_LATENCY.labels(node=name).observe(time.monotonic() - start)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------


class QueryIntent(BaseModel):
    query_type: Literal["person_filter", "time_filter", "summarize", "general"] = Field(
        description="What kind of question this is."
    )
    person: Optional[str] = Field(
        default=None, description="The Slack display name of the person asked about, if any."
    )
    channel: Optional[str] = Field(default=None, description="Slack channel name mentioned, if any.")
    time_start: Optional[str] = Field(
        default=None, description="ISO 8601 date/time for the start of a mentioned time range, if any."
    )
    time_end: Optional[str] = Field(
        default=None, description="ISO 8601 date/time for the end of a mentioned time range, if any."
    )
    needs_full_context: bool = Field(
        description=(
            "True if answering well requires a full transcript/thread rather than a handful of "
            "similarity-matched snippets (e.g. 'summarize the meeting')."
        )
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description=(
            "2-4 search-friendly rewrites of the question for a vector search over Slack messages -- "
            "synonyms and related terminology a teammate might have actually used."
        ),
    )


def _as_intent_dict(value) -> dict | None:
    """last_intent is read back from a checkpoint written on a *previous*
    turn -- possibly before intent was switched to storing a plain dict.
    Accept either shape so an in-place code upgrade doesn't crash on
    conversation state that was already on disk; one successful turn
    naturally overwrites it with the dict form via record_turn_node."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, QueryIntent):
        return value.model_dump()
    return None


class Conflict(BaseModel):
    topic: str
    statement_a: str
    source_a: str
    statement_b: str
    source_b: str


class AnswerWithConflicts(BaseModel):
    answer: str = Field(
        description=(
            "A direct, concise answer (a sentence or two) to the question. Cite inline with bracket "
            "markers like [1] ONLY for the specific excerpt(s) that actually support the answer -- do "
            "not mention, list, or explain excerpts that turned out to be irrelevant. If the excerpts "
            "don't contain the answer, say so in one sentence instead of guessing."
        )
    )
    has_conflict: bool = Field(
        description=(
            "True if two or more excerpts genuinely disagree about the same fact or decision relevant "
            "to the question -- not just different topics."
        )
    )
    conflicts: list[Conflict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class QAState(TypedDict, total=False):
    question: str
    resolved_question: str
    requesting_user_id: str
    allowed_channel_ids: list[str]
    messages: Annotated[list[AnyMessage], add_messages]
    last_intent: Optional[dict]  # a QueryIntent.model_dump() -- see analyze_and_rewrite_node
    intent: dict  # likewise
    search_queries: list[str]
    candidates: list[Document]
    reranked: list[Document]
    answer: str
    citations: list[dict]
    final_answer: str
    error: Optional[str]


_GENERIC_ERROR_MESSAGE = (
    "Sorry, I ran into a problem answering that just now. Please try again in a moment."
)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def contextualize_question_node(state: QAState) -> dict:
    print("[contextualize] starting")
    history = state.get("messages", [])
    if not history:
        print("[contextualize] no prior history, using the question as-is")
        return {"resolved_question": state["question"]}

    today = datetime.now(timezone.utc).date().isoformat()
    transcript = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.text}" for m in history
    )
    last_intent = _as_intent_dict(state.get("last_intent"))
    prior_range = ""
    if last_intent and (last_intent.get("time_start") or last_intent.get("time_end")):
        prior_range = "\n" + CONTEXTUALIZE_PRIOR_RANGE.format(
            time_start=last_intent.get("time_start") or "(open start)",
            time_end=last_intent.get("time_end") or "(open end)",
        )

    try:
        llm = _llm(effort="low", fast=True)
        response = llm.invoke(
            CONTEXTUALIZE_QUESTION.format(
                today=today, transcript=transcript, prior_range=prior_range, question=state["question"]
            )
        )
        # Claude Opus 5 thinks by default, so a response can carry multiple
        # content blocks (thinking + text) instead of a plain string -- .text
        # extracts just the text blocks, correctly, unlike .content.
        resolved = response.text.strip()
        print(f"[contextualize] {state['question']!r} -> {resolved!r}")
    except Exception as e:
        print(f"[contextualize] failed, falling back to raw question: {e!r}")
        NODE_FAILURES.labels(node="contextualize_question").inc()
        resolved = state["question"]
    return {"resolved_question": resolved}


def check_permissions_node(state: QAState) -> dict:
    try:
        allowed = get_allowed_channel_ids(state.get("requesting_user_id", ""))
        print(f"[permissions] user={state.get('requesting_user_id')!r} allowed_channels={allowed}")
    except Exception as e:
        print(f"[permissions] lookup failed, failing closed: {e!r}")
        NODE_FAILURES.labels(node="check_permissions").inc()
        allowed = []
    return {"allowed_channel_ids": allowed}


def gate_permissions_node(state: QAState) -> dict:
    """Pure join point: contextualize_question and check_permissions run in
    parallel from START, and both feed this node via unconditional edges so
    it only runs once both are done. The routing decision then happens here,
    not on check_permissions directly -- a conditional edge on one parallel
    branch doesn't block a sibling branch's own unconditional edge from
    firing its target anyway, so making the decision post-join is what
    actually guarantees analyze_and_rewrite never starts before permissions
    are known.

    Also resets "error" for this turn. State is checkpointed per
    conversation thread, and a node only ever *sets* "error" on failure --
    nothing ever clears it back on success. Without this reset, an error
    from an earlier turn on the same thread stays in the checkpoint forever
    and silently routes every later turn straight to error_response, even
    when that turn's own nodes all succeed."""
    return {"error": None}


def route_after_permissions(state: QAState) -> str:
    return "analyze_and_rewrite" if state.get("allowed_channel_ids") else "no_access"


def no_access_node(state: QAState) -> dict:
    answer = (
        "I don't have access to any Slack channels for you right now, so I can't answer this. "
        "Make sure you're a member of the channels that have been ingested."
    )
    return {"answer": answer, "citations": [], "final_answer": answer}


def analyze_and_rewrite_node(state: QAState) -> dict:
    """Classifies intent AND generates search-query rewrites in a single
    call -- these used to be two separate nodes/API calls; one structured
    call gets both at once."""
    print("[analyze] starting")
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        llm = _llm(effort="low", fast=True).with_structured_output(QueryIntent)
        intent = llm.invoke(ANALYZE_AND_REWRITE.format(today=today, question=state["resolved_question"]))
        print(
            f"[analyze] {intent.query_type} person={intent.person} channel={intent.channel} "
            f"rewrites={len(intent.search_queries)}"
        )
    except Exception as e:
        print(f"[analyze] failed: {e!r}")
        NODE_FAILURES.labels(node="analyze_and_rewrite").inc()
        return {"error": "analyze_and_rewrite failed"}

    queries = [state["resolved_question"]] + [q for q in intent.search_queries if q.strip()]

    # Store as a plain dict, not the live QueryIntent instance -- state gets
    # checkpointed after every node, and langgraph's msgpack serializer warns
    # (and will eventually block) deserializing custom types like a Pydantic
    # model unless explicitly allow-listed. A plain dict sidesteps that.
    return {"intent": intent.model_dump(), "search_queries": queries}


def _parse_epoch(value: Optional[str], label: str) -> float | None:
    """intent.time_start/time_end come from the model's free-form output --
    QueryIntent only declares them as Optional[str], not a validated date
    format, so a malformed value (e.g. "next week" instead of an ISO date)
    is expected occasionally. Drop it instead of crashing retrieval over
    one bad field."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        print(f"[retrieve] ignoring unparseable {label}={value!r}")
        return None


def _build_filter(intent: QueryIntent, allowed_channel_ids: list[str]) -> dict:
    # Permission clause is mandatory and always included -- everything else is
    # an optional refinement on top of it, never a substitute for it.
    clauses = [{"channel_id": {"$in": allowed_channel_ids}}]
    if intent.person:
        clauses.append({"user": {"$eq": intent.person}})
    if intent.channel:
        clauses.append({"channel_name": {"$eq": intent.channel}})
    start_epoch = _parse_epoch(intent.time_start, "time_start")
    if start_epoch is not None:
        clauses.append({"ts_epoch": {"$gte": start_epoch}})
    end_epoch = _parse_epoch(intent.time_end, "time_end")
    if end_epoch is not None:
        clauses.append({"ts_epoch": {"$lte": end_epoch}})

    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _search_one(vectorstore, query: str, where: dict):
    return vectorstore.similarity_search_with_relevance_scores(query, k=_RETRIEVE_K, filter=where)


def retrieve_node(state: QAState) -> dict:
    print("[retrieve] starting")
    try:
        intent = QueryIntent(**state["intent"])
        where = _build_filter(intent, state["allowed_channel_ids"])
        vectorstore = get_vectorstore()

        if intent.needs_full_context:
            raw = vectorstore.get(where=where, limit=200)
            docs = [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(raw["documents"], raw["metadatas"])
            ]
            docs.sort(key=lambda d: d.metadata.get("ts_epoch", 0))
            print(f"[retrieve] full-context fetch -> {len(docs)} chunks")
            RETRIEVAL_CANDIDATES.observe(len(docs))
            if not docs:
                RETRIEVAL_EMPTY.inc()
            return {"candidates": docs}

        queries = state["search_queries"]
        best: dict[str, tuple[float, Document]] = {}
        # Each rewritten query is an independent Voyage API call -- running
        # them concurrently instead of in a sequential loop turns N x
        # round-trip latency into ~1 x round-trip.
        with ThreadPoolExecutor(max_workers=max(len(queries), 1)) as executor:
            futures = [executor.submit(_search_one, vectorstore, q, where) for q in queries]
            for future in futures:
                for doc, score in future.result():
                    key = f"{doc.metadata.get('permalink', '')}|{doc.page_content[:80]}"
                    if key not in best or score > best[key][0]:
                        best[key] = (score, doc)
    except Exception as e:
        print(f"[retrieve] failed: {e!r}")
        NODE_FAILURES.labels(node="retrieve").inc()
        return {"error": "retrieve failed"}

    ranked = sorted(best.values(), key=lambda pair: pair[0], reverse=True)
    candidates = [doc for _, doc in ranked[:_MAX_CANDIDATES]]
    print(f"[retrieve] hybrid search -> {len(candidates)} candidates")
    RETRIEVAL_CANDIDATES.observe(len(candidates))
    if not candidates:
        RETRIEVAL_EMPTY.inc()
    return {"candidates": candidates}


def rerank_node(state: QAState) -> dict:
    print("[rerank] starting")
    candidates = state["candidates"]
    if len(candidates) <= _RERANK_KEEP:
        return {"reranked": candidates}

    try:
        documents = [doc.page_content for doc in candidates]
        order = rerank(state["resolved_question"], documents, top_k=_RERANK_KEEP)
        top = [candidates[i] for i in order if 0 <= i < len(candidates)]
        print(f"[rerank] kept {len(top)}/{len(candidates)} via Voyage rerank")
    except Exception as e:
        print(f"[rerank] failed, using unranked candidates: {e!r}")
        NODE_FAILURES.labels(node="rerank").inc()
        top = []
    return {"reranked": top or candidates[:_RERANK_KEEP]}


def synthesize_answer_node(state: QAState) -> dict:
    """Answers the question AND checks retrieved sources for disagreement in
    one call -- these used to be two sequential nodes/API calls."""
    print("[synthesize_answer] starting")
    docs = state["reranked"]
    if not docs:
        msg = "I couldn't find anything in the Slack history about that."
        return {"answer": msg, "citations": [], "final_answer": msg}

    context_lines = []
    citations = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        context_lines.append(f"[{i}] ({meta.get('channel_name')}, {meta.get('user')}): {doc.page_content}")
        citations.append(
            {
                "marker": i,
                "channel": meta.get("channel_name"),
                "user": meta.get("user"),
                "permalink": meta.get("permalink"),
                "ts": meta.get("ts"),
            }
        )

    try:
        llm = _llm(effort="high").with_structured_output(AnswerWithConflicts)
        result = llm.invoke(
            SYNTHESIZE_ANSWER.format(
                question=state["resolved_question"], excerpts="\n\n".join(context_lines)
            )
        )
    except Exception as e:
        print(f"[synthesize_answer] failed: {e!r}")
        NODE_FAILURES.labels(node="synthesize_answer").inc()
        return {"error": "synthesize_answer failed"}

    answer = result.answer
    # Only surface sources the answer actually cited -- otherwise every
    # retrieved candidate shows up under "Sources" even when most of them
    # weren't relevant enough to mention in the answer itself.
    cited_markers = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    cited = [c for c in citations if c["marker"] in cited_markers]

    final_answer = answer
    if result.has_conflict and result.conflicts:
        CONFLICT_FLAGGED.inc()
        note = ["\n\n⚠️ Sources disagree:"]
        for c in result.conflicts:
            note.append(f'- On {c.topic}: {c.source_a} said "{c.statement_a}" vs {c.source_b} said "{c.statement_b}"')
        final_answer = answer + "\n".join(note)
        print(f"[synthesize_answer] flagged {len(result.conflicts)} conflict(s)")

    return {"answer": answer, "citations": cited, "final_answer": final_answer}


def error_response_node(state: QAState) -> dict:
    return {"answer": _GENERIC_ERROR_MESSAGE, "citations": [], "final_answer": _GENERIC_ERROR_MESSAGE}


def _route_on_error(next_ok: str):
    def _router(state: QAState) -> str:
        return "error_response" if state.get("error") else next_ok

    return _router


def record_turn_node(state: QAState) -> dict:
    return {
        "messages": [HumanMessage(content=state["question"]), AIMessage(content=state["final_answer"])],
        "last_intent": state.get("intent"),
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_qa_graph(checkpointer=None):
    builder = StateGraph(QAState)
    builder.add_node("contextualize_question", _timed_node("contextualize_question")(contextualize_question_node))
    builder.add_node("check_permissions", _timed_node("check_permissions")(check_permissions_node))
    builder.add_node("gate_permissions", gate_permissions_node)
    builder.add_node("no_access", no_access_node)
    builder.add_node("analyze_and_rewrite", _timed_node("analyze_and_rewrite")(analyze_and_rewrite_node))
    builder.add_node("retrieve", _timed_node("retrieve")(retrieve_node))
    builder.add_node("rerank", _timed_node("rerank")(rerank_node))
    builder.add_node("synthesize_answer", _timed_node("synthesize_answer")(synthesize_answer_node))
    builder.add_node("error_response", error_response_node)
    builder.add_node("record_turn", record_turn_node)

    # contextualize_question and check_permissions don't depend on each
    # other -- run them in parallel, both fed into gate_permissions via
    # unconditional edges so it only fires once BOTH are done. The
    # conditional routing happens strictly after that join, not on either
    # parallel branch directly -- doing it on check_permissions directly
    # would let contextualize_question's own unconditional edge race ahead
    # and run analyze_and_rewrite even on a no-access result.
    builder.add_edge(START, "contextualize_question")
    builder.add_edge(START, "check_permissions")
    builder.add_edge("contextualize_question", "gate_permissions")
    builder.add_edge("check_permissions", "gate_permissions")
    builder.add_conditional_edges(
        "gate_permissions", route_after_permissions, ["analyze_and_rewrite", "no_access"]
    )

    builder.add_conditional_edges(
        "analyze_and_rewrite", _route_on_error("retrieve"), ["retrieve", "error_response"]
    )
    builder.add_conditional_edges("retrieve", _route_on_error("rerank"), ["rerank", "error_response"])
    builder.add_edge("rerank", "synthesize_answer")
    builder.add_conditional_edges(
        "synthesize_answer", _route_on_error("record_turn"), ["record_turn", "error_response"]
    )
    builder.add_edge("no_access", "record_turn")
    builder.add_edge("error_response", "record_turn")
    builder.add_edge("record_turn", END)
    
    if not isinstance(checkpointer, (BaseCheckpointSaver, bool)):
        checkpointer = None

    return builder.compile(checkpointer=checkpointer)


@contextmanager
def qa_session():
    """Yields a QA graph whose conversation memory persists across process runs (SQLite-backed)."""
    os.makedirs(config.workspace_state_dir, exist_ok=True)
    with SqliteSaver.from_conn_string(_CHECKPOINT_DB) as checkpointer:
        checkpointer.setup()
        yield build_qa_graph(checkpointer=checkpointer)


def run_turn(graph, question: str, requesting_user_id: str, thread_id: str) -> dict:
    """Runs one turn against an already-open graph session: rate-limits,
    invokes, and records metrics. Shared by ask() (which opens its own
    short-lived session) and ask.py's interactive mode (which keeps one
    session open across turns) so every entry point gets the same rate
    limiting and metrics -- calling graph.invoke() directly would bypass
    both."""
    start = time.monotonic()

    allowed, block_message, reason = check_and_record(requesting_user_id)
    if not allowed:
        print(f"[rate_limit] blocked user={requesting_user_id!r}: {block_message}")
        RATE_LIMIT_BLOCKED.labels(reason=reason).inc()
        REQUESTS_TOTAL.labels(outcome="rate_limited").inc()
        return {"answer": block_message, "citations": []}

    try:
        result = graph.invoke(
            {"question": question, "requesting_user_id": requesting_user_id},
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as e:
        # Last-resort safety net: callers (CLI, Slack bot) should never see a
        # raw exception -- that's a worse experience than a plain apology.
        print(f"[ask] graph invocation failed outright: {e!r}")
        REQUESTS_TOTAL.labels(outcome="error").inc()
        ASK_LATENCY.observe(time.monotonic() - start)
        return {"answer": _GENERIC_ERROR_MESSAGE, "citations": []}

    ASK_LATENCY.observe(time.monotonic() - start)
    if result.get("error"):
        outcome = "error"
    elif not result.get("allowed_channel_ids"):
        outcome = "no_access"
    else:
        outcome = "answered"
    REQUESTS_TOTAL.labels(outcome=outcome).inc()

    return {
        "answer": result.get("final_answer", result.get("answer", "")),
        "citations": result.get("citations", []),
    }


def ask(question: str, requesting_user_id: str, thread_id: str | None = None) -> dict:
    thread_id = thread_id or f"user:{requesting_user_id}"
    with qa_session() as graph:
        return run_turn(graph, question, requesting_user_id, thread_id)
