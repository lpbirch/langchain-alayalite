# AlayaLite Metadata Filtering Experiment

This report validates AlayaLite metadata filtering through the
`langchain-alayalite` integration.

## Configuration

- Corpus size: 1200
- Embedding model: deterministic local test embeddings
- Vector store: `langchain_alayalite.AlayaLite`
- Retriever: `AlayaLite.as_retriever(search_type="filter")`
- Build time: 73.912 ms

## Filter Query Results

| Case | Filter | Expected total | Returned | Precision | Recall@limit | Latency ms | Passed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| single_metadata_key | `{"topic": "vector-db"}` | 300 | 20 | 1.000 | 1.000 | 2.982 | True |
| two_metadata_keys | `{"topic": "rag", "source": "manual"}` | 100 | 20 | 1.000 | 1.000 | 1.469 | True |
| three_metadata_keys | `{"topic": "ann", "source": "blog", "year": 2025}` | 33 | 20 | 1.000 | 1.000 | 1.527 | True |
| limit_is_respected | `{"access": "public"}` | 600 | 7 | 1.000 | 1.000 | 1.299 | True |
| no_match_returns_empty | `{"topic": "missing-topic"}` | 0 | 0 | 1.000 | 1.000 | 1.202 | True |
| langchain_retriever_filter | `{"topic": "langchain", "access": "public"}` | 300 | 15 | 1.000 | 1.000 | 21.254 | True |

## Delete By Filter Result

- Filter: `{"access": "internal"}`
- Expected deleted documents: 600
- Remaining matches for deleted filter: 0
- Remaining public documents: 600
- Passed: True

## Conclusion

All metadata filtering checks passed: True

The experiment verifies exact-match metadata filtering for single-key and
multi-key predicates, empty-result behavior, limit handling, LangChain retriever
integration, and deletion by metadata filter.
