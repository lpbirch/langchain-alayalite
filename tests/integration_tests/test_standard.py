import uuid
from typing import Generator, List

import pytest

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import VectorStore
from langchain_alayalite.vectorstores import AlayaLite

EMBEDDING_SIZE = 6

@pytest.fixture()
def embeddings():
    return DeterministicFakeEmbedding(size=EMBEDDING_SIZE)


@pytest.fixture()
def vectorstore(embeddings) -> Generator[VectorStore, None, None]:
    """
    每个 test 拿到一个全新的 AlayaLite 实例，保证测试隔离
    """
    collection_name = f"test_{uuid.uuid4().hex}"

    store = AlayaLite(
        embedding_function=embeddings,
        collection_name=collection_name,
        drop_old=True,
    )
    try:
        yield store
    finally:
        try:
            store.delete_collection()
        except Exception:
            pass

def test_vectorstore_is_empty(vectorstore: VectorStore):
    assert vectorstore.similarity_search("foo", k=1) == []
    
def test_add_documents(vectorstore: VectorStore) -> None:
    original_documents = [
        Document(page_content="foo", metadata={"id": 1}),
        Document(page_content="bar", metadata={"id": 2}),
    ]
    ids = vectorstore.add_documents(original_documents)
    documents = vectorstore.similarity_search("bar", k=2)
    assert documents == [
        Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
        Document(page_content="foo", metadata={"id": 1}, id=ids[0]),
    ]
        # Verify that the original document object does not get mutated!
        # (e.g., an ID is added to the original document object)
    assert original_documents == [
        Document(page_content="foo", metadata={"id": 1}),
        Document(page_content="bar", metadata={"id": 2}),
    ]

def test_vectorstore_still_empty(vectorstore: VectorStore) -> None:
    assert vectorstore.similarity_search("foo", k=1) == []
    
def test_deleting_documents(vectorstore: VectorStore) -> None:
    """Test deleting documents from the `VectorStore`.

    ??? note "Troubleshooting"

        If this test fails, check that `add_documents` preserves identifiers
        passed in through `ids`, and that `delete` correctly removes
        documents.
    """
    documents = [
        Document(page_content="foo", metadata={"id": 1}),
        Document(page_content="bar", metadata={"id": 2}),
    ]
    ids = vectorstore.add_documents(documents, ids=["1", "2"])
    assert ids == ["1", "2"]
    vectorstore.delete(["1"])
    documents = vectorstore.similarity_search("foo", k=1)
    assert documents == [Document(page_content="bar", metadata={"id": 2}, id="2")]
    

def test_deleting_bulk_documents(vectorstore: VectorStore) -> None:
    """Test that we can delete several documents at once.

    ??? note "Troubleshooting"

        If this test fails, check that `delete` correctly removes multiple
        documents when given a list of IDs.
    """
    documents = [
        Document(page_content="foo", metadata={"id": 1}),
        Document(page_content="bar", metadata={"id": 2}),
        Document(page_content="baz", metadata={"id": 3}),
    ]

    vectorstore.add_documents(documents, ids=["1", "2", "3"])
    vectorstore.delete(["1", "2"])
    documents = vectorstore.similarity_search("foo", k=1)
    assert documents == [Document(page_content="baz", metadata={"id": 3}, id="3")]
    

def test_delete_missing_content(vectorstore: VectorStore) -> None:
    """Deleting missing content should not raise an exception.

    ??? note "Troubleshooting"

        If this test fails, check that `delete` does not raise an exception
        when deleting IDs that do not exist.
    """

    vectorstore.delete(["1"])
    vectorstore.delete(["1", "2", "3"])
    

def test_add_documents_with_ids_is_idempotent(vectorstore: VectorStore) -> None:
    """Adding by ID should be idempotent.

    ??? note "Troubleshooting"

        If this test fails, check that adding the same document twice with the
        same IDs has the same effect as adding it once (i.e., it does not
        duplicate the documents).
    """
    documents = [
        Document(page_content="foo", metadata={"id": 1}),
        Document(page_content="bar", metadata={"id": 2}),
    ]
    vectorstore.add_documents(documents, ids=["1", "2"])
    vectorstore.add_documents(documents, ids=["1", "2"])
    documents = vectorstore.similarity_search("bar", k=2)
    assert documents == [
        Document(page_content="bar", metadata={"id": 2}, id="2"),
        Document(page_content="foo", metadata={"id": 1}, id="1"),
    ]
    

def test_add_documents_by_id_with_mutation(vectorstore: VectorStore) -> None:
    """Test that we can overwrite by ID using `add_documents`.

    ??? note "Troubleshooting"

        If this test fails, check that when `add_documents` is called with an
        ID that already exists in the vector store, the content is updated
        rather than duplicated.
    """
    documents = [
        Document(page_content="foo", metadata={"id": 1}),
        Document(page_content="bar", metadata={"id": 2}),
    ]

    vectorstore.add_documents(documents=documents, ids=["1", "2"])

    # Now over-write content of ID 1
    new_documents = [
        Document(
            page_content="new foo", metadata={"id": 1, "some_other_field": "foo"}
        ),
    ]

    vectorstore.add_documents(documents=new_documents, ids=["1"])

    # Check that the content has been updated
    documents = vectorstore.similarity_search("new foo", k=2)
    assert documents == [
        Document(
            id="1",
            page_content="new foo",
            metadata={"id": 1, "some_other_field": "foo"},
        ),
        Document(id="2", page_content="bar", metadata={"id": 2}),
    ]

def test_get_by_ids(vectorstore: VectorStore) -> None:
    """Test get by IDs.

    This test requires that `get_by_ids` be implemented on the vector store.

    ??? note "Troubleshooting"

        If this test fails, check that `get_by_ids` is implemented and returns
        documents in the same order as the IDs passed in.

        !!! note
            `get_by_ids` was added to the `VectorStore` interface in
            `langchain-core` version 0.2.11. If difficult to implement, this
            test can be skipped by setting the `has_get_by_ids` property to
            `False`.

            ```python
            @property
            def has_get_by_ids(self) -> bool:
                return False
            ```
    """
    documents = [
        Document(page_content="foo", metadata={"id": 1}),
        Document(page_content="bar", metadata={"id": 2}),
    ]
    ids = vectorstore.add_documents(documents, ids=["1", "2"])
    retrieved_documents = vectorstore.get_by_ids(ids)
    assert _sort_by_id(retrieved_documents) == _sort_by_id(
        [
            Document(page_content="foo", metadata={"id": 1}, id=ids[0]),
            Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
        ]
    )

def _sort_by_id(docs: List[Document]) -> List[Document]:
    return sorted(docs, key=lambda d: d.id)



def test_get_by_ids_missing(vectorstore: VectorStore) -> None:
    """Test get by IDs with missing IDs.

    ??? note "Troubleshooting"

        If this test fails, check that `get_by_ids` is implemented and does not
        raise an exception when given IDs that do not exist.

        !!! note
            `get_by_ids` was added to the `VectorStore` interface in
            `langchain-core` version 0.2.11. If difficult to implement, this
            test can be skipped by setting the `has_get_by_ids` property to
            `False`.

            ```python
            @property
            def has_get_by_ids(self) -> bool:
                return False
            ```
    """
    # This should not raise an exception
    documents = vectorstore.get_by_ids(["1", "2", "3"])
    assert documents == []


def test_add_documents_documents( vectorstore: VectorStore) -> None:
    """Run `add_documents` tests.

    ??? note "Troubleshooting"

        If this test fails, check that `get_by_ids` is implemented and returns
        documents in the same order as the IDs passed in.

        Check also that `add_documents` will correctly generate string IDs if
        none are provided.

        !!! note
            `get_by_ids` was added to the `VectorStore` interface in
            `langchain-core` version 0.2.11. If difficult to implement, this
            test can be skipped by setting the `has_get_by_ids` property to
            `False`.

            ```python
            @property
            def has_get_by_ids(self) -> bool:
                return False
            ```
    """
    documents = [
        Document(page_content="foo", metadata={"id": 1}),
        Document(page_content="bar", metadata={"id": 2}),
    ]
    ids = vectorstore.add_documents(documents)
    assert _sort_by_id(vectorstore.get_by_ids(ids)) == _sort_by_id(
        [
            Document(page_content="foo", metadata={"id": 1}, id=ids[0]),
            Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
        ]
    )


def test_add_documents_with_existing_ids(vectorstore: VectorStore) -> None:
    """Test that `add_documents` with existing IDs is idempotent.

    ??? note "Troubleshooting"

        If this test fails, check that `get_by_ids` is implemented and returns
        documents in the same order as the IDs passed in.

        This test also verifies that:

        1. IDs specified in the `Document.id` field are assigned when adding
            documents.
        2. If some documents include IDs and others don't string IDs are generated
            for the latter.

        !!! note
            `get_by_ids` was added to the `VectorStore` interface in
            `langchain-core` version 0.2.11. If difficult to implement, this
            test can be skipped by setting the `has_get_by_ids` property to
            `False`.

            ```python
            @property
            def has_get_by_ids(self) -> bool:
                return False
            ```
    """

    documents = [
        Document(id="foo", page_content="foo", metadata={"id": 1}),
        Document(page_content="bar", metadata={"id": 2}),
    ]
    ids = vectorstore.add_documents(documents)
    assert "foo" in ids
    assert _sort_by_id(vectorstore.get_by_ids(ids)) == _sort_by_id(
        [
            Document(page_content="foo", metadata={"id": 1}, id="foo"),
            Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
        ]
    )






