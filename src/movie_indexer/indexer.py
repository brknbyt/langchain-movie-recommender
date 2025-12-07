from langchain_core.document_loaders import BaseLoader
from langchain_core.vectorstores import VectorStore


class MovieIndexer:
    """Indexes movie data using a provided loader and vector store.

    This class takes a document loader and vector store, loads movie documents,
    and indexes them into the vector store for similarity search.
    """

    def __init__(self, loader: BaseLoader, vector_store: VectorStore) -> None:
        """Initialize the MovieIndexer.

        Args:
            loader: The data loader to load movie data.
            vector_store: The vector store to index the data into.
        """
        self._loader = loader
        self._vector_store = vector_store

    def index(self) -> None:
        """Index the movie data.

        Loads documents using the loader and adds them to the vector store.
        """
        documents = self._loader.load()
        self._vector_store.add_documents(documents)
