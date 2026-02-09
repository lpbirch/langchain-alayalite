from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable, Sequence
from itertools import cycle
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance

__all__ = ["AlayaLite"]

logger = logging.getLogger(__name__)


class AlayaLite(VectorStore):
    def __init__(
        self,
        embedding_function: Embeddings,
        collection_name: str = "DefaultCollection",
        index_params: dict[str, Any] | None = None,
        search_params: dict[str, Any] | None = None,
        drop_old: bool = False,
        url: str = ".alayalite_data",
        **kwargs: Any,
    ):
        try:
            import alayalite
        except ImportError as e:
            raise ImportError(
                "alayalite is required. Install it with `pip install alayalite`."
            ) from e

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
    def embeddings(self) -> Embeddings | None:
        """Access the query embedding object if available."""
        return self._embedding_function

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        # This is not yet enforced in the type signature for backwards compatibility
        # with existing implementations.
        **kwargs: Any,
    ) -> list[str]:
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
            logger.warning(
                f"Batch embedding failed, falling back to single text embedding: {e}"
            )
            embeddings = [self._embedding_function.embed_query(text) for text in texts]

        # Process the metadatas
        if metadatas and len(metadatas) != len(texts):
            raise ValueError(
                "Length of `metadatas` must be equal to length of `texts`."
            )
        metadatas_iter = iter(metadatas) if metadatas else cycle([{}])

        # Add to the collection
        # Tuple(id, document, embedding, metadata)
        self._collection.upsert(
            list(zip(ids, texts, embeddings, metadatas_iter, strict=False))
        )
        self._client.save_collection(collection_name=self._collection_name)

        return list(ids)

    def add_documents(
        self,
        documents: list[Document],
        **kwargs: Any,
    ) -> list[str]:
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        ids = kwargs.get("ids", None)

        if ids is None:
            ids = [
                doc.id if doc.id is not None else str(uuid.uuid4()) for doc in documents
            ]

        return self.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
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
        documents: list[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> AlayaLite:
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
        document_ids: list[str | None] = []
        for doc in documents:
            # Check if Document has id attribute (not all implementations do)
            if hasattr(doc, "id") and doc.id is not None:
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
            texts=texts, embedding=embedding, metadatas=metadatas, **kwargs
        )

    def delete(
        self,
        ids: list[str] | None = None,
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
                logger.info(
                    f"Deleting all vectors from collection: {self._collection_name}"
                )

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
            # Save changes (only if index exists)
            if (
                hasattr(self._client, "save_collection")
                and getattr(self._collection, "_Collection__index_py", None) is not None
            ):
                self._client.save_collection(collection_name=self._collection_name)

            return True

        except Exception:
            logger.exception("Error deleting vectors from AlayaLite")
            raise

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query vector.
        """
        if self._collection is None:
            return []

        if not getattr(self._collection, "_Collection__index_py", None):
            return []

        ef_search = kwargs.pop("ef_search", 10)
        num_threads = kwargs.pop("num_threads", 1)
        res = self._collection.batch_query(
            [embedding], limit=k, ef_search=ef_search, num_threads=num_threads
        )

        docs = [
            Document(page_content=doc, metadata=metadata, id=id_)
            for id_, doc, metadata in zip(
                res["id"][0],
                res["document"][0],
                res["metadata"][0],
                strict=True,
            )
        ]

        return docs

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
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
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
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
            Document(page_content=doc, metadata=metadata, id=id_)
            for id_, doc, metadata in zip(
                res["id"][0],
                res["document"][0],
                res["metadata"][0],
                strict=True,
            )
        ]

        scores = [self._euclidean_relevance_score_fn(dis) for dis in res["distance"][0]]
        return list(zip(docs, scores, strict=True))

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
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

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
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

        documents: list[Document] = []

        for doc_id, content, metadata in zip(
            res.get("id", []),
            res.get("document", []),
            res.get("metadata", []),
            strict=True,
        ):
            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                    id=doc_id,
                )
            )

        return documents

    def delete_collection(self) -> None:
        """
        Delete the entire collection from the database.

        This method completely removes the collection and all its data.
        After calling this method, the internal collection reference will be None.

        Note: The collection object will need to be recreated if you want to use it again.
        """
        try:
            if self._collection_name in self._client.list_collections():
                self._client.delete_collection(collection_name=self._collection_name)
                logger.info(
                    f"Collection '{self._collection_name}' deleted successfully."
                )

            # Reset the internal collection reference
            self._collection = None

        except Exception as e:
            logger.error(f"Failed to delete collection '{self._collection_name}': {e}")
            raise

    @classmethod
    async def afrom_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> AlayaLite:
        def _create() -> AlayaLite:
            return cls.from_texts(
                texts=texts,
                embedding=embedding,
                metadatas=metadatas,
                **kwargs,
            )

        return await run_in_executor(None, _create)

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        if ids is not None:
            kwargs["ids"] = ids
        return await run_in_executor(None, self.add_texts, texts, metadatas, **kwargs)

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        return await run_in_executor(None, self.similarity_search, query, k, **kwargs)

    async def asimilarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:
        return await run_in_executor(
            None, self.similarity_search_by_vector, embedding, k, **kwargs
        )

    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        return await run_in_executor(
            None, self.similarity_search_with_score, query, k, **kwargs
        )

    async def asimilarity_search_with_score_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        return await run_in_executor(
            None, self.similarity_search_with_score_by_vector, embedding, k, **kwargs
        )

    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        return await run_in_executor(None, self.get_by_ids, ids)

    async def adelete(self, ids: list[str] | None = None, **kwargs: Any) -> bool:
        return await run_in_executor(None, self.delete, ids, **kwargs)

    async def aadd_documents(
        self,
        documents: list[Document],
        **kwargs: Any,
    ) -> list[str]:
        return await run_in_executor(None, self.add_documents, documents, **kwargs)

    @classmethod
    async def afrom_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> AlayaLite:
        def _create() -> AlayaLite:
            return cls.from_documents(
                documents=documents,
                embedding=embedding,
                **kwargs,
            )

        return await run_in_executor(None, _create)

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using maximal marginal relevance (MMR)."""
        if k <= 0:
            return []

        if self._collection is None:
            return []

        # collection 内部 index 未初始化时，batch_query 会 assert，这里提前拦一下
        if not getattr(self._collection, "_Collection__index_py", None):
            return []

        # 1) embed query
        query_embedding = self._embedding_function.embed_query(query)

        # 2) 先做一次普通相似检索，取 fetch_k 个候选
        #    注意：这里调用 by_vector，避免重复 embed
        candidates = self.similarity_search_by_vector(
            embedding=query_embedding,
            k=fetch_k,
            **kwargs,
        )
        if not candidates:
            return []

        # 3) 拉取候选向量（按 candidates 顺序对齐）
        candidate_ids = [d.id for d in candidates if d.id is not None]
        if not candidate_ids:
            return []

        try:
            candidate_embeddings = self._collection.get_embeddings_by_id(candidate_ids)
        except Exception as e:
            logger.warning(f"get_embeddings_by_id failed: {e}")
            return []

        if not candidate_embeddings:
            return []

        # 4) MMR 选择（返回的是 candidate_embeddings 的下标）
        mmr_indices = maximal_marginal_relevance(
            query_embedding=np.asarray(query_embedding, dtype=np.float32),
            embedding_list=candidate_embeddings,
            lambda_mult=lambda_mult,
            k=min(k, len(candidate_embeddings)),
        )
        # 5) 按 MMR 顺序返回文档
        return [candidates[i] for i in mmr_indices]

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        if k <= 0:
            return []
        if self._collection is None:
            return []
        if not getattr(self._collection, "_Collection__index_py", None):
            return []

        candidates = self.similarity_search_by_vector(
            embedding=embedding,
            k=fetch_k,
            **kwargs,
        )
        if not candidates:
            return []

        candidate_ids = [d.id for d in candidates if d.id is not None]
        if not candidate_ids:
            return []

        try:
            candidate_embeddings = self._collection.get_embeddings_by_id(candidate_ids)
        except Exception as e:
            logger.warning(f"get_embeddings_by_id failed: {e}")
            return []

        mmr_indices = maximal_marginal_relevance(
            query_embedding=np.asarray(embedding, dtype=np.float32),
            embedding_list=candidate_embeddings,
            lambda_mult=lambda_mult,
            k=min(k, len(candidate_embeddings)),
        )
        return [candidates[i] for i in mmr_indices]
