from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import pytest
from langchain_core.retrievers import BaseRetriever
from langchain_tests.integration_tests import RetrieversIntegrationTests

from langchain_alayalite import AlayaLite, AlayaLiteRetriever
from tests.integration_tests.fake_embeddings import ConsistentFakeEmbeddings


class TestAlayaLiteRetrieverStandard(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> type[BaseRetriever]:
        return AlayaLiteRetriever

    @property
    def retriever_constructor_params(self) -> dict[str, Any]:
        return {
            "embedding_function": ConsistentFakeEmbeddings(),
            "texts": ["alpha", "beta", "gamma"],
            "metadatas": [
                {"group": "a"},
                {"group": "b"},
                {"group": "a"},
            ],
            "drop_old": True,
            "url": mkdtemp(prefix="alayalite_retriever_"),
        }

    @property
    def retriever_query_example(self) -> str:
        return "alpha"


def test_retriever_wraps_existing_vectorstore(tmp_path: Path) -> None:
    vectorstore = AlayaLite(
        embedding_function=ConsistentFakeEmbeddings(),
        url=str(tmp_path / "alayalite_data"),
    )
    vectorstore.add_texts(["alpha", "beta"], metadatas=[{"group": "a"}, {"group": "b"}])

    retriever = AlayaLiteRetriever(vectorstore=vectorstore, k=1)

    docs = retriever.invoke("alpha")

    assert len(docs) == 1
    assert docs[0].page_content == "alpha"
    assert docs[0].metadata == {"group": "a"}


def test_vectorstore_as_retriever_returns_alayalite_retriever(tmp_path: Path) -> None:
    vectorstore = AlayaLite(
        embedding_function=ConsistentFakeEmbeddings(),
        url=str(tmp_path / "alayalite_data"),
    )
    vectorstore.add_texts(["alpha", "beta"])

    retriever = vectorstore.as_retriever(k=1)

    assert isinstance(retriever, AlayaLiteRetriever)
    assert len(retriever.invoke("alpha")) == 1


def test_retriever_filter_search(tmp_path: Path) -> None:
    retriever = AlayaLiteRetriever(
        embedding_function=ConsistentFakeEmbeddings(),
        texts=["alpha", "beta", "gamma"],
        metadatas=[{"group": "a"}, {"group": "b"}, {"group": "a"}],
        search_type="filter",
        search_kwargs={"filter": {"group": "a"}},
        url=str(tmp_path / "alayalite_data"),
    )

    docs = retriever.invoke("ignored", k=2)

    assert [doc.page_content for doc in docs] == ["alpha", "gamma"]


@pytest.mark.asyncio
async def test_retriever_async_invoke(tmp_path: Path) -> None:
    retriever = AlayaLiteRetriever(
        embedding_function=ConsistentFakeEmbeddings(),
        texts=["alpha", "beta"],
        url=str(tmp_path / "alayalite_data"),
        k=1,
    )

    docs = await retriever.ainvoke("alpha")

    assert len(docs) == 1
    assert docs[0].page_content == "alpha"
