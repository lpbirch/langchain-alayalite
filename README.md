# langchain-alayalite

`langchain-alayalite` is a LangChain integration package that connects
[AlayaLite](https://github.com/AlayaDB-AI/AlayaLite), a lightweight vector
database, with the LangChain ecosystem.

It provides a `VectorStore` implementation compatible with `langchain-core` for
embedding storage, similarity search, deletion, ID lookup, retrieval workflows,
and maximal marginal relevance search.

## Installation

```bash
pip install langchain-alayalite
```

For local development:

```bash
cd libs/alayalite
pip install -e .
```

## Features

- Native LangChain `VectorStore` implementation
- Native LangChain `BaseRetriever` implementation
- Synchronous and asynchronous APIs
- `add_texts` and `add_documents`
- `delete` and `get_by_ids`
- `similarity_search`, `similarity_search_by_vector`, and score variants
- `max_marginal_relevance_search` and `max_marginal_relevance_search_by_vector`
- `from_texts` and `from_documents` constructors
- `AlayaLite.as_retriever()` with AlayaLite-specific search options
- LangChain standard vector store test coverage

## Quick Start

```python
from langchain_alayalite import AlayaLite
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

vectorstore = AlayaLite(
    embedding_function=embeddings,
    url="./alayalite_data",
    collection_name="demo",
)

docs = [
    Document(page_content="LangChain makes building LLM applications easier."),
    Document(page_content="AlayaLite is a lightweight vector database."),
    Document(page_content="Vector databases enable semantic similarity search."),
]

vectorstore.add_documents(docs)

results = vectorstore.similarity_search("lightweight vector database", k=2)

for doc in results:
    print(doc.page_content)
```

## Retriever

```python
retriever = vectorstore.as_retriever(
    k=4,
    search_type="similarity",
    search_kwargs={"ef_search": 20, "num_threads": 1},
)

docs = retriever.invoke("lightweight vector database")
```

You can also construct a retriever directly:

```python
from langchain_alayalite import AlayaLiteRetriever

retriever = AlayaLiteRetriever(
    embedding_function=embeddings,
    texts=["alpha", "beta", "gamma"],
    metadatas=[{"group": "a"}, {"group": "b"}, {"group": "a"}],
    search_type="filter",
    search_kwargs={"filter": {"group": "a"}},
)
```

## Development

Run the test suite from `libs/alayalite`:

```bash
pytest tests
```

Run static checks:

```bash
ruff check .
mypy langchain_alayalite
```

## License

MIT
