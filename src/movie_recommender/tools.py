import os

from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings


def create_vector_store(
    vector_store_name: str, embedding: Embeddings, **kwargs
) -> VectorStore:
    """Factory method to create a vector store based on the given name.

        Args:
            vector_store_name: Name of the vector store     embedding = HuggingFaceEmbeddings(
            model_name=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
        )
        store = vector_store(
            os.getenv("VECTOR_STORE", "in_memory"),
            embedding=embedding,
            initialize_table=False,
        )
        res = store.similarity_search("A movie about space exploration.")
        print(res)
    to create.

        Returns:
            VectorStore : An instance of the specified vector store.

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
def movie_recommendation(query: str):
    """Tool to recommend movies based on a user query."""
    res = store.similarity_search(query)
    return res
