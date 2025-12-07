import os
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings


def create_vector_store(
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
            f"postgresql+asyncpg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}"
            f":{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
        )

        pg_engine = PGEngine.from_connection_string(connection_string)
        if kwargs.get("initialize_table", True):
            pg_engine.init_vectorstore_table(
                table_name=os.getenv("TABLE_NAME"),
                vector_size=int(os.getenv("VECTOR_SIZE")),
                overwrite_existing=True,
            )

        return PGVectorStore.create_sync(
            engine=pg_engine,
            table_name=os.getenv("TABLE_NAME"),
            embedding_service=embedding,
        )
    else:
        raise ValueError(f"Unsupported vector store: {vector_store_name}")


embedding = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
)
store = create_vector_store(
    os.getenv("VECTOR_STORE", "in_memory"),
    embedding=embedding,
    initialize_table=False,
)


@tool()
def movie_recommendation(query: str) -> list[Document]:
    """Tool to recommend movies based on a user query.

    Searches the vector store for movies that match the user's query
    using semantic similarity search.

    Args:
        query: The user's search query describing their movie preferences.

    Returns:
        list[Document]: A list of Document objects representing matching movies.
    """
    res = store.similarity_search(query)
    return res
