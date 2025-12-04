from langchain_core.document_loaders import BaseLoader
from langchain_core.vectorstores import VectorStore


class MovieIndexer:
    """Indexes movie data using a provided loader and vector store."""

    def __init__(self, loader: BaseLoader, vector_store: VectorStore):
        """Initializes the MovieIndexer.

        Args:
            loader: The data loader to load movie data.
            vector_store: The vector store to index the data into.
        """
        self._loader = loader
        self._vector_store = vector_store

    def index(self) -> None:
        """Indexes the movie data.

        Loads documents using the loader and adds them to the vector store.
        """
        documents = self._loader.load()
        self._vector_store.add_documents(documents)
