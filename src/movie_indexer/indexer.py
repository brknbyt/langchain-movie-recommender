from langchain_core.document_loaders import BaseLoader
from langchain_core.vectorstores import VectorStore


class MovieIndexer:
    def __init__(self, loader: BaseLoader, vector_store: VectorStore):
        self._loader = loader
        self._vector_store = vector_store

    def index(self) -> None:
        documents = self._loader.load()
        self._vector_store.add_documents(documents)
