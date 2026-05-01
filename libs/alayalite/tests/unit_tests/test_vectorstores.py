from pathlib import Path

from langchain_core.embeddings.fake import (
    FakeEmbeddings,
)

from langchain_alayalite.vectorstores import AlayaLite
from tests.integration_tests.fake_embeddings import ConsistentFakeEmbeddings


def test_initialization(tmp_path: Path) -> None:
    """Test integration vectorstore initialization."""
    texts = ["foo", "bar", "baz"]
    AlayaLite.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(size=10),
        url=str(tmp_path / "alayalite_data"),
    )


def test_similarity_search(tmp_path: Path) -> None:
    """Test similarity search by AlayaLite."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = AlayaLite.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(size=10),
        metadatas=metadatas,
        url=str(tmp_path / "alayalite_data"),
    )
    output = docsearch.similarity_search("foo", k=1)
    docsearch.delete()
    assert len(output) == 1


def test_filter_search(tmp_path: Path) -> None:
    """Test metadata filter search through the vector store API."""
    docsearch = AlayaLite.from_texts(
        collection_name="test_collection",
        texts=["alpha", "beta", "gamma"],
        embedding=ConsistentFakeEmbeddings(),
        metadatas=[{"group": "a"}, {"group": "b"}, {"group": "a"}],
        url=str(tmp_path / "alayalite_data"),
    )

    output = docsearch.filter_search({"group": "a"}, k=2)

    assert [doc.page_content for doc in output] == ["alpha", "gamma"]


def test_max_marginal_relevance_search(tmp_path: Path) -> None:
    """Test MMR can read stored embeddings from AlayaLite's index."""
    docsearch = AlayaLite.from_texts(
        collection_name="test_collection",
        texts=["alpha", "beta", "gamma"],
        embedding=ConsistentFakeEmbeddings(),
        url=str(tmp_path / "alayalite_data"),
    )

    output = docsearch.max_marginal_relevance_search("alpha", k=2, fetch_k=3)

    assert len(output) == 2
    assert output[0].page_content == "alpha"


def test_similarity_search_uses_configured_search_params(tmp_path: Path) -> None:
    """Test constructor search params are passed into AlayaLite queries."""
    docsearch = AlayaLite.from_texts(
        collection_name="test_collection",
        texts=["alpha", "beta", "gamma"],
        embedding=ConsistentFakeEmbeddings(),
        search_params={"ef_search": 1, "num_threads": 2},
        url=str(tmp_path / "alayalite_data"),
    )
    calls: list[dict] = []
    batch_query = docsearch._collection.batch_query

    def capture_batch_query(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return batch_query(*args, **kwargs)

    docsearch._collection.batch_query = capture_batch_query

    docsearch.similarity_search("alpha", k=3)

    assert calls[0]["ef_search"] == 3
    assert calls[0]["num_threads"] == 2
