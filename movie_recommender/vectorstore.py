import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from movie_recommender import config
from movie_recommender.data_sources import KaggleCSVDataSource
from movie_recommender.indexer import MovieIndexer

load_dotenv()


def get_vector_store(
    vector_store_name: str, embedding: Embeddings, **kwargs: Any
) -> VectorStore:
    """Factory method to create a vector store based on the given name.

    Supports both in-memory and PostgreSQL-based vector stores.

    Args:
        vector_store_name: Name of the vector store to create
            ("in_memory" or "pgvectorstore").
        embedding: The embeddings model to use for the vector store.
        **kwargs: Additional keyword arguments. For "pgvectorstore",
            accepts "initialize_table" (bool) to control table creation.

    Returns:
        VectorStore: An instance of the specified vector store.

    Raises:
        ValueError: If the specified vector store is not supported.
    """
    if vector_store_name == "in_memory":
        from langchain_core.vectorstores import InMemoryVectorStore

        return InMemoryVectorStore(embedding=embedding)
    if vector_store_name == "pgvectorstore":
        from langchain_postgres import PGEngine, PGVectorStore

        connection_string = (
            f"postgresql+asyncpg://{config.PG_USER}:{config.PG_PASSWORD}@{config.PG_HOST}"
            f":{config.PG_PORT}/{config.PG_DB}"
        )

        pg_engine = PGEngine.from_connection_string(connection_string)
        if kwargs.get("initialize_table", True):
            pg_engine.init_vectorstore_table(
                table_name=config.TABLE_NAME,
                vector_size=config.VECTOR_SIZE,
                overwrite_existing=True,
            )

        return PGVectorStore.create_sync(
            engine=pg_engine,
            table_name=config.TABLE_NAME,
            embedding_service=embedding,
        )
    else:
        raise ValueError(f"Unsupported vector store: {vector_store_name}")


def index() -> None:
    """Index movie data from Kaggle dataset into a vector store.

    Loads configuration from environment variables, creates a data source,
    initializes embeddings, and indexes the movie data into the specified
    vector store.
    """
    data_source = KaggleCSVDataSource.from_env()
    loader = data_source.get_loader()
    embedding = HuggingFaceEmbeddings(
        model_name=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    indexer = MovieIndexer(
        loader=loader,
        vector_store=get_vector_store(
            os.getenv("VECTOR_STORE", "in_memory"), embedding
        ),
    )
    indexer.index()


def search() -> None:
    """Search for movies using similarity search in the vector store.

    Initializes embeddings and connects to the configured vector store
    to perform a similarity search for movies matching the query.
    Prints the search results to stdout.
    """
    embedding = HuggingFaceEmbeddings(
        model_name=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    store = get_vector_store(
        os.getenv("VECTOR_STORE", "in_memory"),
        embedding=embedding,
        initialize_table=False,
    )
    res = store.similarity_search(
        "hard science fiction space exploration realistic grounded"
    )
    print(res)


if __name__ == "__main__":
    search()
