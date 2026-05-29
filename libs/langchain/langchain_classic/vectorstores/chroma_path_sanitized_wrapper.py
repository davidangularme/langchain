import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

from langchain_classic.vectorstores.path_sanitizer import PathSanitizer

logger = logging.getLogger(__name__)


class ChromaPathSanitizedWrapper:
    """
    A secure wrapper around ChromaDB operations that sanitizes all user-supplied paths
    before they reach the filesystem. This prevents path traversal attacks.
    """

    def __init__(
        self,
        embedding_function: Any,
        persist_directory: Optional[Union[str, Path]] = None,
        collection_name: str = "langchain",
        **kwargs: Any,
    ):
        """
        Initialize the sanitized Chroma wrapper.

        Args:
            embedding_function: The embedding function to use.
            persist_directory: The base directory for storing ChromaDB data.
                              If provided, it must be a directory path (will be created if needed).
            collection_name: The name of the collection.
            **kwargs: Additional arguments for the underlying Chroma client.
        """
        self.collection_name = collection_name
        self._embedding_function = embedding_function
        self._extra_kwargs = kwargs

        # If persist_directory is provided, sanitize and set it as base
        if persist_directory is not None:
            # Temporarily use current working directory as base to resolve persist_directory
            temp_sanitizer = PathSanitizer(base_directory=Path.cwd())
            resolved_persist = temp_sanitizer.sanitize(persist_directory)
            # The resolved path becomes the base for all further operations
            self._base_directory = resolved_persist
            self._path_sanitizer = PathSanitizer(base_directory=self._base_directory)
            # Ensure the persist directory exists
            self._base_directory.mkdir(parents=True, exist_ok=True)
        else:
            self._base_directory = None
            self._path_sanitizer = None

        self._chroma_collection = None

    def _get_or_create_collection(self, collection_name: Optional[str] = None) -> Any:
        """Internal method to get or create the ChromaDB collection."""
        if self._chroma_collection is not None and collection_name is None:
            return self._chroma_collection

        import chromadb

        if self._base_directory is not None:
            client = chromadb.PersistentClient(
                path=str(self._base_directory), **self._extra_kwargs
            )
        else:
            client = chromadb.Client(**self._extra_kwargs)

        col_name = collection_name or self.collection_name
        self._chroma_collection = client.get_or_create_collection(
            name=col_name, embedding_function=self._embedding_function
        )
        return self._chroma_collection

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the collection."""
        collection = self._get_or_create_collection()
        return collection.add_texts(
            texts=texts, metadatas=metadatas, ids=ids, **kwargs
        )

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Any]:
        """Perform similarity search."""
        collection = self._get_or_create_collection()
        return collection.similarity_search(query, k=k, **kwargs)

    def delete_collection(self, collection_name: Optional[str] = None) -> None:
        """Delete a collection."""
        client = self._get_or_create_collection()
        col_name = collection_name or self.collection_name
        client.delete_collection(col_name)

    def delete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> None:
        """Delete entries by IDs."""
        collection = self._get_or_create_collection()
        collection.delete(ids=ids, **kwargs)
