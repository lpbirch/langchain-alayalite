from langchain_core.embeddings.fake import (
    FakeEmbeddings,
)

from langchain_alayalite.vectorstores import AlayaLite


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    texts = ["foo", "bar", "baz"]
    AlayaLite.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(size=10),
    )


def test_similarity_search() -> None:
    """Test similarity search by Chroma."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = AlayaLite.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(size=10),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search("foo", k=1)
    docsearch.delete()
    assert len(output) == 1
