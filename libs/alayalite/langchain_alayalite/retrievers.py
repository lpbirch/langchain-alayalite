from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever, LangSmithRetrieverParams
from langchain_core.runnables.config import run_in_executor
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self, override

from langchain_alayalite.vectorstores import AlayaLite

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )

SearchType = Literal["similarity", "mmr", "filter"]


class AlayaLiteRetriever(BaseRetriever):
    """Retriever backed by an AlayaLite vector store.

    The retriever can wrap an existing `AlayaLite` vector store or create one from
    texts/documents and an embedding function. It exposes AlayaLite search knobs
    such as `ef_search`, `num_threads`, and metadata-only filtering.
    """

    vectorstore: AlayaLite
    """AlayaLite vector store used for retrieval."""

    search_type: SearchType = "similarity"
    """Retrieval mode. One of `similarity`, `mmr`, or `filter`."""

    k: int = 4
    """Number of documents to return."""

    search_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Additional keyword arguments passed to the selected search method."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        *,
        vectorstore: AlayaLite | None = None,
        embedding_function: Embeddings | None = None,
        texts: list[str] | None = None,
        documents: list[Document] | None = None,
        metadatas: list[dict] | None = None,
        collection_name: str = "DefaultCollection",
        index_params: dict[str, Any] | None = None,
        search_params: dict[str, Any] | None = None,
        drop_old: bool = False,
        url: str = ".alayalite_data",
        search_type: SearchType = "similarity",
        k: int = 4,
        search_kwargs: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if vectorstore is None:
            if embedding_function is None:
                msg = (
                    "`embedding_function` is required when `vectorstore` is not "
                    "provided."
                )
                raise ValueError(msg)
            vectorstore = AlayaLite(
                embedding_function=embedding_function,
                collection_name=collection_name,
                index_params=index_params,
                search_params=search_params,
                drop_old=drop_old,
                url=url,
            )
            if documents is not None and texts is not None:
                msg = "Pass either `documents` or `texts`, not both."
                raise ValueError(msg)
            if documents is not None:
                vectorstore.add_documents(documents, **kwargs)
            elif texts is not None:
                vectorstore.add_texts(texts, metadatas=metadatas, **kwargs)
        elif any(
            value is not None
            for value in (embedding_function, texts, documents, metadatas)
        ):
            msg = (
                "`embedding_function`, `texts`, `documents`, and `metadatas` are "
                "only valid when creating a retriever without an existing "
                "`vectorstore`."
            )
            raise ValueError(msg)

        super().__init__(  # type: ignore[call-arg]
            vectorstore=vectorstore,
            search_type=search_type,
            k=k,
            search_kwargs=search_kwargs or {},
            tags=tags,
            metadata=metadata,
        )

    @model_validator(mode="after")
    def validate_search_type(self) -> Self:
        if self.search_type not in ("similarity", "mmr", "filter"):
            msg = "`search_type` must be one of 'similarity', 'mmr', or 'filter'."
            raise ValueError(msg)
        return self

    def _get_ls_params(self, **kwargs: Any) -> LangSmithRetrieverParams:
        ls_params = super()._get_ls_params(**kwargs)
        ls_params["ls_vector_store_provider"] = "AlayaLite"
        if self.vectorstore.embeddings:
            ls_params["ls_embedding_provider"] = (
                self.vectorstore.embeddings.__class__.__name__
            )
        return ls_params

    def _filter_search(self, k: int, **kwargs: Any) -> list[Document]:
        metadata_filter = kwargs.pop("filter", None)
        if metadata_filter is None:
            msg = "`filter` search requires a metadata filter dictionary."
            raise ValueError(msg)
        if not isinstance(metadata_filter, dict):
            msg = "`filter` must be a dictionary."
            raise ValueError(msg)

        result = self.vectorstore._collection.filter_query(metadata_filter, limit=k)
        return [
            Document(page_content=doc, metadata=metadata, id=id_)
            for id_, doc, metadata in zip(
                result.get("id", []),
                result.get("document", []),
                result.get("metadata", []),
                strict=True,
            )
        ]

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        k = kwargs.pop("k", self.k)
        kwargs_ = self.search_kwargs | kwargs

        if self.search_type == "similarity":
            return self.vectorstore.similarity_search(query, k=k, **kwargs_)
        if self.search_type == "mmr":
            return self.vectorstore.max_marginal_relevance_search(query, k=k, **kwargs_)
        return self._filter_search(k=k, **kwargs_)

    @override
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        return await run_in_executor(
            None,
            self._get_relevant_documents,
            query,
            run_manager=run_manager.get_sync(),
            **kwargs,
        )
