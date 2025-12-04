import os

from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from .data_sources import KaggleCSVDataSource
from .indexer import MovieIndexer

load_dotenv()


if __name__ == "__main__":
    data_source = KaggleCSVDataSource.from_env()
    loader = data_source.get_loader()
    embedding = HuggingFaceEmbeddings(
        model_name=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    indexer = MovieIndexer(
        loader=loader,
        vector_store=InMemoryVectorStore(embedding=embedding),
    )
    indexer.index()
