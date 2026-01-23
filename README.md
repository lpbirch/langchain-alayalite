# langchain-alayalite

`langchain-alayalite` is a LangChain partner package that integrates **AlayaLite** as a lightweight vector database backend.

It provides a native `VectorStore` implementation fully compatible with **LangChain Core**, enabling seamless usage of **AlayaLite** inside the LangChain ecosystem for vector storage, similarity search, and retrieval workflows.

This project is officially maintained by members of the **AlayaLite team**, with the primary goal of promoting **AlayaLite** as a high-performance, lightweight vector database solution for AI applications.

---

## About AlayaLite

**AlayaLite** is a lightweight vector database designed for efficient embedding storage, similarity search, and retrieval workloads, optimized for simplicity, performance, and developer friendliness.

* GitHub Repository:
  [https://github.com/AlayaDB-AI/AlayaLite.git](https://github.com/AlayaDB-AI/AlayaLite.git)

* Key characteristics:

  * Lightweight deployment
  * Fast vector indexing & search
  * Simple API design
  * Optimized for LLM & RAG scenarios
  * Easy integration with LangChain

---

## About LangChain

**LangChain** is a widely used framework for building applications powered by large language models (LLMs), providing modular components for:

* Prompt engineering
* Chains and agents
* Memory systems
* Retrieval-Augmented Generation (RAG)
* Vector database integrations

Official website:
[https://www.langchain.com/](https://www.langchain.com/)

---

## Features

* Native **LangChain VectorStore** implementation
* Fully compatible with `langchain-core`
* Supports:

  * `add_documents`
  * `add_texts`
  * `similarity_search`
  * `similarity_search_with_score`
* Passes **LangChain official standard integration tests**
* Simple API, easy configuration
* Designed for high performance and low overhead

---

## Current Status

* Passed **LangChain official standard test suite**
* Core synchronous APIs fully supported
* Asynchronous APIs are **not fully supported yet**

> Async method support is currently under development and will be released in upcoming versions.

---

## Installation

Install from PyPI:

```bash
pip install langchain-alayalite
```

---

## Quick Start

```python
from langchain_alayalite import AlayaLite
from langchain_core.documents import Document

vectorstore = AlayaLite(
    persist_path="./alayalite_data"
)

docs = [
    Document(page_content="LangChain makes LLM applications easy."),
    Document(page_content="AlayaLite is a lightweight vector database.")
]

vectorstore.add_documents(docs)

results = vectorstore.similarity_search("vector database", k=2)

for doc in results:
    print(doc.page_content)
```

---

## Why AlayaLite?

Compared with traditional vector databases, **AlayaLite focuses on lightweight design and simplicity**, making it ideal for:

* Local development
* Edge deployment
* Lightweight RAG systems
* Rapid prototyping
* Small to medium-scale vector workloads

---

## Roadmap

* [ ] Full async API support
* [ ] Advanced indexing strategies
* [ ] Distributed mode
* [ ] Hybrid search (vector + keyword)
* [ ] Performance benchmarking release

---

## Contributing

Contributions are welcome!
Feel free to open issues or pull requests on GitHub:

[https://github.com/AlayaDB-AI/AlayaLite.git](https://github.com/AlayaDB-AI/AlayaLite.git)

