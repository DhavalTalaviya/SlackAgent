import voyageai

from src.config import config
from src.infra.metrics import VOYAGE_RERANK_TOKENS

_client = voyageai.Client(api_key=config.voyage_api_key)


def rerank(query: str, documents: list[str], top_k: int) -> list[int]:
    """Returns document indices in relevance order (best first), via Voyage's
    dedicated rerank model. Far cheaper and faster than asking an LLM to
    score each passage's relevance one at a time -- one small-model API call
    instead of one Claude call, at a fraction of the cost and latency."""
    result = _client.rerank(
        query=query, documents=documents, model=config.voyage_rerank_model, top_k=top_k
    )
    VOYAGE_RERANK_TOKENS.labels(model=config.voyage_rerank_model).inc(result.total_tokens)
    return [r.index for r in result.results]
