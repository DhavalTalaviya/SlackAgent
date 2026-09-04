from langchain_chroma import Chroma
from langchain_voyageai import VoyageAIEmbeddings

from src.config import config

_embeddings = VoyageAIEmbeddings(model="voyage-3.5", voyage_api_key=config.voyage_api_key)


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=config.workspace_collection_name,
        embedding_function=_embeddings,
        persist_directory=config.chroma_persist_dir,
    )
