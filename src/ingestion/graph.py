from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from src.ingestion.chunking import chunk_documents
from src.ingestion.state import load_state, save_state
from src.slack.ingest import fetch_slack_documents
from src.vectorstore import get_vectorstore

_BATCH_SIZE = 100


class IngestState(TypedDict, total=False):
    prior_state: dict
    slack_docs: list[Document]
    slack_state: dict
    chunks: list[Document]
    stored_count: int


def fetch_slack_node(state: IngestState) -> dict:
    docs, slack_state = fetch_slack_documents(state["prior_state"])
    print(f"[slack] fetched {len(docs)} new messages/threads")
    return {"slack_docs": docs, "slack_state": slack_state}


def chunk_node(state: IngestState) -> dict:
    all_docs = state.get("slack_docs", [])
    chunks = chunk_documents(all_docs)
    print(f"[chunk] {len(all_docs)} documents -> {len(chunks)} chunks")
    return {"chunks": chunks}


def store_node(state: IngestState) -> dict:
    chunks = state.get("chunks", [])
    if chunks:
        vectorstore = get_vectorstore()
        for i in range(0, len(chunks), _BATCH_SIZE):
            vectorstore.add_documents(chunks[i : i + _BATCH_SIZE])
        print(f"[store] embedded and persisted {len(chunks)} chunks")
    else:
        print("[store] nothing new to embed")

    save_state({"slack": state.get("slack_state", {})})
    return {"stored_count": len(chunks)}


def build_graph():
    builder = StateGraph(IngestState)
    builder.add_node("fetch_slack", fetch_slack_node)
    builder.add_node("chunk", chunk_node)
    builder.add_node("store", store_node)

    builder.add_edge(START, "fetch_slack")
    builder.add_edge("fetch_slack", "chunk")
    builder.add_edge("chunk", "store")
    builder.add_edge("store", END)

    return builder.compile()


def run_ingest() -> int:
    graph = build_graph()
    result = graph.invoke({"prior_state": load_state()})
    return result.get("stored_count", 0)
