# /root/projects/langchain-community/libs/community/tests/unit_tests/vectorstores/test_alayalite.py

import os
import shutil
import sys
from typing import Any, Generator
import pytest
from langchain_core.documents import Document

from langchain_alayalite import AlayaLite
from tests.integration_tests.fake_embeddings import FakeEmbeddings

pytestmark = pytest.mark.requires("alayalite")

from tests.integration_tests.fake_embeddings import (
    ConsistentFakeEmbeddings,
)
@pytest.fixture(scope="function", autouse=True)
def cleanup_test_data() -> Generator[None, None, None]:
    yield
    test_dir = "./test_alayalite_data"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

@pytest.fixture
def test_embeddings() -> ConsistentFakeEmbeddings:
    return ConsistentFakeEmbeddings()


@pytest.fixture
def alayalite_store(test_embeddings: ConsistentFakeEmbeddings) -> AlayaLite:
    return AlayaLite(
        embedding_function=test_embeddings,
        collection_name="test_collection",
        drop_old=True,
        url="./test_alayalite_data"
    )


def test_alayalite_initialization(test_embeddings: ConsistentFakeEmbeddings) -> None:
    store = AlayaLite(
        embedding_function=test_embeddings,
        collection_name="test_init",
        drop_old=True
    )
    assert store is not None
    assert store.embeddings == test_embeddings
    assert store.collection_name == "test_init"


def test_alayalite_add_texts(alayalite_store: AlayaLite) -> None:
    texts = ["Hello world", "Machine learning", "Python programming"]
    ids = alayalite_store.add_texts(texts)
    
    assert len(ids) == 3
    # 验证返回的 ID 是字符串
    assert all(isinstance(id_, str) for id_ in ids)


def test_alayalite_add_texts_with_metadata(alayalite_store: AlayaLite) -> None:
    texts = ["Document 1", "Document 2"]
    metadatas = [{"source": "test1", "page": 1}, {"source": "test2", "page": 2}]
    
    ids = alayalite_store.add_texts(texts, metadatas=metadatas)
    assert len(ids) == 2


def test_alayalite_add_texts_with_custom_ids(alayalite_store: AlayaLite) -> None:
    texts = ["Custom doc 1", "Custom doc 2"]
    custom_ids = ["id_001", "id_002"]
    
    ids = alayalite_store.add_texts(texts, ids=custom_ids)
    assert ids == custom_ids


def test_alayalite_similarity_search(alayalite_store: AlayaLite) -> None:
    texts = [
        "The weather is sunny today",
        "Machine learning algorithms",
        "Python is a programming language",
        "Artificial intelligence is fascinating"
    ]
    alayalite_store.add_texts(texts)

    results = alayalite_store.similarity_search("programming", k=2)
    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)


def test_alayalite_similarity_search_with_score(alayalite_store: AlayaLite) -> None:
    texts = ["Apple fruit", "Banana fruit", "Car vehicle", "Dog animal"]
    alayalite_store.add_texts(texts)
    
    results = alayalite_store.similarity_search_with_score("fruit", k=3)
    assert len(results) == 3
    assert all(isinstance(item, tuple) and len(item) == 2 for item in results)
    assert all(isinstance(item[0], Document) for item in results)
    assert all(isinstance(item[1], float) for item in results)


def test_alayalite_similarity_search_by_vector(alayalite_store: AlayaLite) -> None:
    texts = ["Test document 1", "Test document 2", "Test document 3"]
    alayalite_store.add_texts(texts)
    
    test_vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    results = alayalite_store.similarity_search_by_vector(test_vector, k=2)
    assert len(results) == 2


def test_alayalite_get_by_ids(alayalite_store: AlayaLite) -> None:
    texts = ["Doc A", "Doc B", "Doc C"]
    ids = alayalite_store.add_texts(texts)
    
    retrieved_docs = alayalite_store.get_by_ids([ids[0], ids[2]])
    assert len(retrieved_docs) == 2
    assert all(isinstance(doc, Document) for doc in retrieved_docs)


def test_alayalite_delete_by_ids(alayalite_store: AlayaLite) -> None:
    texts = ["To delete 1", "To delete 2", "To keep"]
    ids = alayalite_store.add_texts(texts)
    
    result = alayalite_store.delete(ids=[ids[0], ids[1]])
    assert result is True

    remaining_docs = alayalite_store.get_by_ids([ids[2]])
    assert len(remaining_docs) == 1
    assert remaining_docs[0].page_content == "To keep"


def test_alayalite_delete_all(alayalite_store: AlayaLite) -> None:
    pass


def test_alayalite_from_texts(test_embeddings: ConsistentFakeEmbeddings) -> None:
    """测试 from_texts 类方法。"""
    texts = ["Text 1", "Text 2", "Text 3"]
    metadatas = [{"id": 1}, {"id": 2}, {"id": 3}]
    
    store = AlayaLite.from_texts(
        texts=texts,
        embedding=test_embeddings,
        metadatas=metadatas,
        collection_name="from_texts_test",
        drop_old=True
    )
    
    assert isinstance(store, AlayaLite)
    results = store.similarity_search("Text", k=3)
    assert len(results) == 3


def test_alayalite_from_documents(test_embeddings: ConsistentFakeEmbeddings) -> None:
    from langchain_core.documents import Document
    
    documents = [
        Document(page_content="Document A", metadata={"source": "src1"}),
        Document(page_content="Document B", metadata={"source": "src2"}),
        Document(page_content="Document C", metadata={"source": "src3"})
    ]
    
    store = AlayaLite.from_documents(
        documents=documents,
        embedding=test_embeddings,
        collection_name="from_docs_test",
        drop_old=True
    )
    
    assert isinstance(store, AlayaLite)
    results = store.similarity_search("Document", k=3)
    assert len(results) == 3


def test_alayalite_async_methods_not_implemented(alayalite_store: AlayaLite) -> None:
    pass

