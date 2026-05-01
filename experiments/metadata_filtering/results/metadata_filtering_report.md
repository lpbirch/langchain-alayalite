# AlayaLite Metadata Filtering Experiment

This report validates AlayaLite metadata filtering through the
`langchain-alayalite` integration.

## Configuration

- Corpus size: 50000
- Embedding model: deterministic local test embeddings
- Vector store: `langchain_alayalite.AlayaLite`
- Retriever: `AlayaLite.as_retriever(search_type="filter")`
- Build time: 2918.888 ms

## Filter Query Results

| Case | Filter | Expected total | Selectivity | Returned | Precision | Recall@limit | Latency ms | Passed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| single_metadata_key | `{"topic": "vector-db"}` | 12500 | 25.000% | 20 | 1.000 | 1.000 | 32.520 | True |
| two_metadata_keys | `{"topic": "rag", "source": "manual"}` | 4167 | 8.334% | 20 | 1.000 | 1.000 | 32.456 | True |
| three_metadata_keys | `{"topic": "ann", "source": "blog", "year": 2025}` | 1388 | 2.776% | 20 | 1.000 | 1.000 | 33.276 | True |
| limit_is_respected | `{"access": "public"}` | 25000 | 50.000% | 7 | 1.000 | 1.000 | 30.739 | True |
| no_match_returns_empty | `{"topic": "missing-topic"}` | 0 | 0.000% | 0 | 1.000 | 1.000 | 31.663 | True |
| langchain_retriever_filter | `{"topic": "langchain", "access": "public"}` | 12500 | 25.000% | 15 | 1.000 | 1.000 | 50.497 | True |

## Delete By Filter Result

- Filter: `{"access": "internal"}`
- Expected deleted documents: 25000
- Remaining matches for deleted filter: 0
- Remaining public documents: 25000
- Passed: True

## Conclusion

All metadata filtering checks passed: True

Precision and recall@limit are expected to be 1.0 for a correct exact-match
metadata filter. The scale-sensitive evidence is the selectivity and latency
columns: the experiment scans filters with 0% to 50% selectivity over
50000 documents while preserving exact results.

The experiment verifies exact-match metadata filtering for single-key and
multi-key predicates, empty-result behavior, limit handling, LangChain retriever
integration, and deletion by metadata filter.
