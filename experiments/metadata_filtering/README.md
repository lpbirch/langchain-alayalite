# Metadata Filtering Experiment

This experiment validates the metadata filtering requirement from the thesis
task book and midterm self-check form.

It uses the local `langchain-alayalite` integration to:

- Build a deterministic corpus with at least 1000 documents.
- Attach structured metadata fields: `topic`, `source`, `year`, `access`, and
  `shard`.
- Verify `AlayaLite.filter_search()` for single-key and multi-key exact-match
  filters.
- Verify limit handling and empty-result behavior.
- Verify LangChain retriever integration with
  `AlayaLite.as_retriever(search_type="filter")`.
- Verify deletion by metadata filter through `AlayaLite.delete(filter=...)`.

Run from the repository root:

```bash
..\.venv\Scripts\python.exe experiments\metadata_filtering\metadata_filtering_experiment.py
```

The script writes:

- `experiments/metadata_filtering/results/metadata_filtering_results.json`
- `experiments/metadata_filtering/results/metadata_filtering_report.md`

The generated report can be used directly as the experiment record for the
thesis section about AlayaLite metadata filtering.
