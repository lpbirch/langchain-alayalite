from __future__ import annotations
__all__ = ["AlayaLite"]

from itertools import cycle
import logging
import uuid
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Sequence,
)
import numpy as np


from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from langchain_alayalite.alayalite import Client


logger = logging.getLogger(__name__)

class AlayaLite(VectorStore):

    def __init__(
        self,
        embedding_function: Embeddings,
        collection_name: str = "DefaultCollection",
        index_params: Optional[Dict[str, Any]] = None,
        search_params: Optional[Dict[str, Any]] = None,
        drop_old: bool = False,
        url: str = ".alayalite_data",
        **kwargs: Any,
    ):
        
        
        try:
            import langchain_alayalite.alayalite as alayalite
        except ImportError:
            raise ImportError(
                "Could not import alayalite python package. "
                "Please install it with `pip install alayalite`."
            )
            
            
        self._embedding_function = embedding_function
        self._collection_name = collection_name
        self._index_params = index_params or {}
        self._search_params = search_params or {}
        self._drop_old = drop_old
        self._url = url


        self._client = alayalite.Client(url=self._url)


        if drop_old:
            if collection_name in self._client.list_collections():
                self._client.delete_collection(collection_name=collection_name)

        self._collection = self._client.get_or_create_collection(name=collection_name)



    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding_function
    @property
    def client(self) -> Client:
        return self._client
    @property
    def collection(self):
        return self._collection
    @property
    def collection_name(self) -> str:
        return self._collection_name
    @property
    def index_params(self) -> Dict[str, Any]:
        return self._index_params
    @property
    def search_params(self) -> Dict[str, Any]:
        return self._search_params
    @property
    def persist_directory(self) -> str:
        return self._url
    
    
    
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        # This is not yet enforced in the type signature for backwards compatibility
        # with existing implementations.
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            **kwargs: vectorstore specific parameters.
                One of the kwargs should be `ids` which is a list of ids
                associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.

        Raises:
            ValueError: If the number of metadatas does not match the number of texts.
            ValueError: If the number of ids does not match the number of texts.
        """
        texts = list(texts)
        if len(texts) == 0:
            logger.debug("No texts to add, skipping.")
            return []


        # Get the ids
        ids = kwargs.get("ids", None)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        else:
            if len(ids) != len(texts):
                raise ValueError("Length of `ids` must be equal to length of `texts`.")


        # Embedding the texts
        try:
            embeddings = self._embedding_function.embed_documents(texts)
        except (NotImplementedError, AttributeError) as e:
            # Catch more possible exceptions
            # AttributeError: method does not exist
            # NotImplementedError: method exists but is not implemented
            logger.warning(f"Batch embedding failed, falling back to single text embedding: {e}")
            embeddings = [self._embedding_function.embed_query(text) for text in texts]


        # Process the metadatas
        if metadatas and len(metadatas) != len(texts):
            raise ValueError(
                "Length of `metadatas` must be equal to length of `texts`."
            )
        metadatas = iter(metadatas) if metadatas else cycle([{}])

        # Add to the collection
        # Tuple(id, document, embedding, metadata)
        self._collection.insert(list(zip(ids, texts, embeddings, metadatas)))
        self._client.save_collection(collection_name=self._collection_name)

    
        return ids
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> AlayaLite:
        """Return VectorStore initialized from texts and embeddings.

        Args:
            texts: Texts to add to the vectorstore.
            embedding: Embedding function to use.
            metadatas: Optional list of metadatas associated with the texts.
                Default is None.
            kwargs: Additional keyword arguments.

        Returns:
            VectorStore: VectorStore initialized from texts and embeddings.
        """
        vector_db = cls(
            embedding_function=embedding,
            **kwargs,
        )

        vector_db.add_texts(
            texts=texts,
            metadatas=metadatas,
            **kwargs,
        )

        return vector_db    
    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> "AlayaLite":
        """
        Return AlayaLite vector store initialized from documents and embeddings.
        
        This method extracts text content and metadata from Document objects,
        then uses the from_texts method to create and populate the vector store.
        
        Args:
            documents: List of Document objects to add to the vector store.
                      Each Document has:
                      - page_content: the text content
                      - metadata: associated metadata (dict)
                      - id: optional unique identifier
            embedding: Embedding function to use for vectorizing the text content.
            **kwargs: Additional keyword arguments passed to from_texts.
                     Common kwargs for AlayaLite:
                     - collection_name: name of the collection (default: "DefaultCollection")
                     - index_params: index configuration parameters
                     - search_params: search configuration parameters
                     - drop_old: whether to delete existing collection (default: False)
                     - url: data storage path (default: ".alayalite_data")
        
        Returns:
            AlayaLite: Initialized and populated vector store instance.
        """
        # Extract texts and metadatas from Document objects
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Check if any Document has an id attribute
        # Some Document implementations might have an id field
        document_ids = []
        for doc in documents:
            # Check if Document has id attribute (not all implementations do)
            if hasattr(doc, 'id') and doc.id is not None:
                document_ids.append(doc.id)
            else:
                document_ids.append(None)
        
        # Only pass ids if at least one document has a valid id
        # This follows the same logic as the base class implementation
        if any(id_val is not None for id_val in document_ids):
            kwargs["ids"] = [
                doc_id if doc_id is not None else str(uuid.uuid4())
                for doc_id in document_ids
            ]
        
        # Use from_texts to create and populate the vector store
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            **kwargs
        )

    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Delete vectors from the vector store.

        Supported:
        - delete()  # delete all vectors in the current collection
        - delete(ids=[...])
        - delete(filter={...})

        Args:
            ids: Optional list of ids to delete
            **kwargs: Additional parameters including 'filter'

        Returns:
            bool: True if deletion was successful
        """
        try:
            has_filter = "filter" in kwargs and kwargs["filter"] is not None

            if ids is not None and has_filter:
                raise ValueError(
                    "delete() received both `ids` and `filter`. "
                    "Please specify only one."
                )

            # Case 1: Delete all vectors in current collection
            if ids is None and not has_filter:
                logger.info(f"Deleting all vectors from collection: {self._collection_name}")
                
                # Method 1: Delete and recreate collection
                self._client.delete_collection(collection_name=self._collection_name)
                self._collection = self._client.get_or_create_collection(
                    name=self._collection_name
                )
                return True

            # Case 2: Delete by filter
            elif ids is None and has_filter:
                metadata_filter = kwargs["filter"]
                if not isinstance(metadata_filter, dict):
                    raise ValueError("`filter` must be a dict")
                self._collection.delete_by_filter(metadata_filter)

            # Case 3: Delete by ids
            elif ids is not None:
                if not ids:
                    return True
                self._collection.delete_by_id(ids)

            # Save changes
            if hasattr(self._client, 'save_collection'):
                self._client.save_collection(
                    collection_name=self._collection_name
                )
            return True

        except Exception:
            logger.exception("Error deleting vectors from AlayaLite")
            raise

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query vector.
        """
        ef_search = kwargs.pop("ef_search", 10)
        num_threads = kwargs.pop("num_threads", 1)
        res = self._collection.batch_query(
            [embedding], limit=k, ef_search=ef_search, num_threads=num_threads
        )

        docs = [
            Document(page_content=doc, metadata=metadata)
            for doc, metadata in zip(res["document"][0], res["metadata"][0])
        ]
        return docs    

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query.
        """
        if self._collection is None:
            logger.debug("No collection found, returning empty list.")
            return []

        embeddings = self._embedding_function.embed_query(query)
        return self.similarity_search_by_vector(embedding=embeddings, k=k, **kwargs)
    
    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            ef_search: Number of nearest neighbors to search for, optimal value
            num_threads: Number of threads to use for the search, optimal value
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Tuple(Document, float) most similar to the query vector.
        """
        
        ef_search = kwargs.pop("ef_search", 10)
        num_threads = kwargs.pop("num_threads", 1)
        res = self._collection.batch_query(
            [embedding], limit=k, ef_search=ef_search, num_threads=num_threads
        )

        docs = [
            Document(page_content=doc, metadata=metadata)
            for doc, metadata in zip(res["document"][0], res["metadata"][0])
        ]

        scores = [self._euclidean_relevance_score_fn(dis) for dis in res["distance"][0]]
        return list(zip(docs, scores))
    
    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        
        """Run similarity search with distance.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            ef_search: Number of nearest neighbors to search for, optimal value
            num_threads: Number of threads to use for the search, optimal value
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Tuples of (doc, similarity_score).
        """
        
        if self._collection is None:
            logger.debug("No collection found, returning empty list.")
            return []

        embedding = self._embedding_function.embed_query(query)
        
        return self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, **kwargs
        )
   
    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        """Get documents by their IDs.

        Args:
            ids: List of IDs to retrieve.

        Returns:
            List of `Document` objects.
        """
        if not ids:
            return []

        try:
            res = self._collection.get_by_id(list(ids))
        except Exception as e:
            logger.warning(f"get_by_ids failed: {e}")
            return []

        documents: List[Document] = []

        for doc_id, content, metadata in zip(
            res.get("id", []),
            res.get("document", []),
            res.get("metadata", []),
        ):
            meta = dict(metadata) if metadata else {}
            meta["id"] = doc_id

            documents.append(
                Document(
                    page_content=content,
                    metadata=meta,
                )
            )

        return documents



    @staticmethod
    def _async_not_supported(self, method_name: str) -> None:
        """Raise NotImplementedError for async methods."""
        raise NotImplementedError(
            f"Async method '{method_name}' is not currently supported by AlayaLite. "
            f"Please use the synchronous version: '{method_name[1:]}()'. "
            f"Async support may be added in future versions."
        )
    @classmethod
    async def afrom_texts(cls, *args, **kwargs) -> "AlayaLite":
        """Async from_texts - Not supported."""
        cls._async_not_supported("afrom_texts")
    
    async def aadd_texts(self, *args, **kwargs) -> List[str]:
        """Async add_texts - Not supported."""
        self._async_not_supported("aadd_texts")
    
    async def asimilarity_search(self, *args, **kwargs) -> List[Document]:
        """Async similarity_search - Not supported."""
        self._async_not_supported("asimilarity_search")
    
    async def asimilarity_search_by_vector(self, *args, **kwargs) -> List[Document]:
        """Async similarity_search_by_vector - Not supported."""
        self._async_not_supported("asimilarity_search_by_vector")
        
    async def asimilarity_search_with_score(self, *args, **kwargs) -> List[Tuple[Document, float]]:
        """Async similarity_search_with_score - Not supported."""
        self._async_not_supported("asimilarity_search_with_score")
        
    async def asimilarity_search_with_score_by_vector(self, *args, **kwargs) -> List[Tuple[Document, float]]:
        """Async similarity_search_with_score_by_vector - Not supported."""
        self._async_not_supported("asimilarity_search_with_score_by_vector")
    
    async def aget_by_ids(self, *args, **kwargs) -> List[Document]:
        """Async get_by_ids - Not supported."""
        self._async_not_supported("aget_by_ids")
        
    async def adelete(self, *args, **kwargs) -> bool:   
        """Async delete - Not supported."""
        self._async_not_supported("adelete")
        
    async def afrom_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> "AlayaLite":
        """Async from_documents - Not supported."""
        cls._async_not_supported("afrom_documents")
        
    