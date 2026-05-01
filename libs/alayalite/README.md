# langchain-alayalite

`langchain-alayalite` is the LangChain partner integration for
[AlayaLite](https://github.com/AlayaDB-AI/AlayaLite), a lightweight local vector
database. It provides a `langchain-core` compatible vector store and retriever
for storing embeddings, running similarity search, applying metadata filters,
and building retrieval workflows.

## Installation

```bash
pip install langchain-alayalite
```

For development from this repository:

```bash
cd libs/alayalite
pip install -e .
```

If you are developing against local checkouts of AlayaLite or LangChain Core,
install those packages into the same environment before installing this package:

```bash
pip install -e ../../../alayalite/python
pip install -e ../../../langchain/libs/core
pip install -e .
```

## What This Package Provides

- `AlayaLite`, a `VectorStore` implementation backed by AlayaLite collections.
- `AlayaLiteRetriever`, a `BaseRetriever` implementation for direct retrieval use.
- Standard LangChain vector store methods, including `add_texts`,
  `add_documents`, `delete`, `get_by_ids`, and async equivalents.
- Similarity search by text or vector, with optional relevance scores.
- Maximal marginal relevance search through LangChain's MMR utility.
- Metadata filter search for exact-match AlayaLite filters.
- LangSmith retriever metadata for vector store and embedding provider tracing.

## Quick Start

```python
from langchain_alayalite import AlayaLite
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = AlayaLite(
    embedding_function=embeddings,
    collection_name="demo",
    url="./alayalite_data",
)

documents = [
    Document(
        page_content="AlayaLite is a lightweight vector database.",
        metadata={"source": "alayalite", "topic": "database"},
        id="doc-1",
    ),
    Document(
        page_content="LangChain provides standard interfaces for retrieval.",
        metadata={"source": "langchain", "topic": "retrieval"},
        id="doc-2",
    ),
    Document(
        page_content="Vector stores support semantic similarity search.",
        metadata={"source": "notes", "topic": "database"},
        id="doc-3",
    ),
]

vectorstore.add_documents(documents)

results = vectorstore.similarity_search("local vector database", k=2)
for doc in results:
    print(doc.id, doc.page_content)
```

## Vector Store Usage

Create a vector store from texts:

```python
from langchain_alayalite import AlayaLite

vectorstore = AlayaLite.from_texts(
    texts=["alpha", "beta", "gamma"],
    embedding=embeddings,
    metadatas=[
        {"group": "a"},
        {"group": "b"},
        {"group": "a"},
    ],
    collection_name="example",
    url="./alayalite_data",
)
```

Run similarity search:

```python
docs = vectorstore.similarity_search("alpha", k=2)
docs_with_scores = vectorstore.similarity_search_with_score("alpha", k=2)
```

Run metadata filter search:

```python
docs = vectorstore.filter_search({"group": "a"}, k=2)
```

Run maximal marginal relevance search:

```python
docs = vectorstore.max_marginal_relevance_search(
    "alpha",
    k=2,
    fetch_k=10,
    lambda_mult=0.5,
)
```

Delete records:

```python
vectorstore.delete(ids=["doc-1"])
vectorstore.delete(filter={"group": "a"})
vectorstore.delete()  # delete all records in the current collection
```

## Retriever Usage

Use the vector store as a retriever:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    k=4,
    search_kwargs={"ef_search": 40, "num_threads": 1},
)

docs = retriever.invoke("local vector database")
```

Use MMR retrieval:

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    k=4,
    search_kwargs={"fetch_k": 20, "lambda_mult": 0.5},
)
```

Use metadata-only filter retrieval:

```python
retriever = vectorstore.as_retriever(
    search_type="filter",
    k=2,
    search_kwargs={"filter": {"topic": "database"}},
)

docs = retriever.invoke("ignored for filter search")
```

You can also construct a retriever directly:

```python
from langchain_alayalite import AlayaLiteRetriever

retriever = AlayaLiteRetriever(
    embedding_function=embeddings,
    texts=["alpha", "beta", "gamma"],
    metadatas=[{"group": "a"}, {"group": "b"}, {"group": "a"}],
    search_type="similarity",
    k=2,
    url="./alayalite_data",
)
```

## AlayaLite Parameters

`AlayaLite` accepts the following integration-specific parameters:

| Parameter | Description |
| --- | --- |
| `embedding_function` | Required LangChain `Embeddings` object. |
| `collection_name` | AlayaLite collection name. Defaults to `DefaultCollection`. |
| `url` | Local AlayaLite data directory. Defaults to `.alayalite_data`. |
| `drop_old` | Delete the existing collection before creating this instance. |
| `index_params` | Keyword arguments forwarded to AlayaLite collection creation. |
| `search_params` | Default query parameters such as `ef_search` and `num_threads`. |

Per-call search keyword arguments override constructor-level `search_params`.

## Async API

The integration exposes async counterparts for the main vector store operations:

```python
vectorstore = await AlayaLite.afrom_texts(
    texts=["alpha", "beta"],
    embedding=embeddings,
    url="./alayalite_data",
)

docs = await vectorstore.asimilarity_search("alpha", k=1)
ids = await vectorstore.aadd_texts(["gamma"])
```

## Development

This repository follows the LangChain partner package layout. The package source
lives in `libs/alayalite`.

Install development dependencies:

```bash
cd libs/alayalite
uv sync --group lint --group typing --group test --group test_integration
```

Run checks:

```bash
make lint
make test
make integration_tests
```

Useful direct commands:

```bash
uv run ruff check .
uv run ruff format .
uv run mypy langchain_alayalite
uv run pytest tests -q
```

## Project Links

- AlayaLite: <https://github.com/AlayaDB-AI/AlayaLite>
- LangChain: <https://github.com/langchain-ai/langchain>
- This integration: <https://github.com/lpbirch/langchain-alayalite>

## License

MIT
