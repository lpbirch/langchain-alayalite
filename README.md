# langchain-alayalite

`langchain-alayalite` is an official LangChain partner integration that connects **AlayaLite**, a high-performance lightweight vector database, with the **LangChain** ecosystem.

It provides a native **VectorStore** implementation fully compatible with **langchain-core**, enabling seamless usage of **AlayaLite** for vector storage, similarity search, and retrieval workflows in modern LLM-powered applications.

This project is officially maintained by members of the **AlayaLite team**, with the primary goal of promoting **AlayaLite** as a production-ready, lightweight, and high-performance vector database solution.

---

## About AlayaLite

**AlayaLite** is a lightweight vector database designed for efficient embedding storage, similarity search, and retrieval workloads, optimized for simplicity, performance, and developer friendliness.

**GitHub Repository:** https://github.com/AlayaDB-AI/AlayaLite

### Key Features

- **High Performance**: Optimized vector indexing and search pipeline.
- **Elastic Scalability**: Multi-threaded design powered by C++20 coroutines.
- **Adaptive Flexibility**: Pluggable quantization strategies, metrics, and data types.
- **Ease of Use**: Intuitive Python APIs with minimal configuration.
- **Low Resource Overhead**: Designed for lightweight deployments.

---

## About LangChain

**LangChain** is a widely used framework for building applications powered by large language models (LLMs), providing modular components for prompt engineering, chains and agents, memory systems, Retrieval-Augmented Generation (RAG), and vector database integrations.

**Official Website:** https://www.langchain.com/

**GitHub Repository:** https://github.com/langchain-ai/langchain

---

## Features

- Native **LangChain VectorStore** implementation
- Fully compatible with **langchain-core**
- Supports both **synchronous and asynchronous APIs**
- Production-ready, high-performance backend powered by AlayaLite

### Data Management

- `add_documents` 
- `add_texts` 
- `delete`
- `get_by_ids`

### Similarity Search

- `similarity_search`
- `similarity_search_with_score`
- `similarity_search_by_vector`

### Advanced Retrieval

- `max_marginal_relevance_search`
- `max_marginal_relevance_search_by_vector` (MMR)

### Construction Helpers

- `from_texts`
- `from_documents`

- Passes **LangChain official standard integration test suite**

---

## Current Status & Limitations

- ✅ All **synchronous APIs** are fully supported.
- ✅ Passes **LangChain official standard test suite** (except async tests).
- ✅ **Asynchronous APIs are fully supported**
- ✅ **MMR (Maximal Marginal Relevance) retrieval is available**
---

## Installation

Install from PyPI:

```bash
pip install langchain-alayalite
```

For development mode:

```bash
pip install -e .
```

---

## Quick Start

Below is a complete working example using **OpenAI embeddings**.

```python
from langchain_alayalite import AlayaLite
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Initialize embedding model
embeddings = OpenAIEmbeddings()

# Create vector store
vectorstore = AlayaLite(
    embedding=embeddings,
    persist_directory="./alayalite_data"  # Optional: persist data to disk
)

# Prepare documents
docs = [
    Document(page_content="LangChain makes building LLM applications easy."),
    Document(page_content="AlayaLite is a high-performance lightweight vector database."),
    Document(page_content="Vector databases enable fast semantic similarity search.")
]

# Add documents
vectorstore.add_documents(docs)

# Perform similarity search
results = vectorstore.similarity_search("lightweight vector database", k=2)

for doc in results:
    print(doc.page_content)
```

### Example Output

```
AlayaLite is a high-performance lightweight vector database.
Vector databases enable fast semantic similarity search.
```

---

## Contributing

Contributions, issues, and feature requests are welcome!

Please open issues or pull requests at: https://github.com/AlayaDB-AI/AlayaLite

---

## License

MIT
