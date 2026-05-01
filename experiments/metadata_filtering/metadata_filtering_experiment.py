"""Validate AlayaLite metadata filtering through the LangChain integration.

This experiment is designed for the thesis requirement:
"verify basic similarity search and optional metadata filtering functionality."

It builds a deterministic corpus with structured metadata, stores it through
`langchain_alayalite.AlayaLite`, and checks exact-match filtering through both
the vector store API and the retriever API.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_alayalite import AlayaLite


TOPICS = ("vector-db", "rag", "langchain", "ann")
SOURCES = ("paper", "manual", "blog")
YEARS = (2023, 2024, 2025)
ACCESS_LEVELS = ("public", "internal")


class DeterministicMetadataEmbeddings(Embeddings):
    """Small deterministic embeddings for repeatable local experiments."""

    def __init__(self, size: int = 16) -> None:
        self.size = size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.size
        for index, byte in enumerate(text.encode("utf-8")):
            vector[index % self.size] += float(byte % 31) / 31.0
        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            return vector
        return [value / norm for value in vector]


@dataclass(frozen=True)
class CaseResult:
    name: str
    filter: dict[str, Any]
    limit: int
    expected_total: int
    selectivity: float
    returned: int
    precision: float
    recall_at_limit: float
    latency_ms: float
    passed: bool


@dataclass(frozen=True)
class ExperimentResult:
    corpus_size: int
    build_time_ms: float
    vectorstore_cases: list[CaseResult]
    retriever_case: CaseResult
    delete_by_filter: dict[str, Any]
    all_passed: bool


def build_documents(count: int) -> list[Document]:
    documents = []
    for index in range(count):
        topic = TOPICS[index % len(TOPICS)]
        source = SOURCES[(index // len(TOPICS)) % len(SOURCES)]
        year = YEARS[(index // (len(TOPICS) * len(SOURCES))) % len(YEARS)]
        access = ACCESS_LEVELS[index % len(ACCESS_LEVELS)]
        shard = f"shard-{index % 10}"
        documents.append(
            Document(
                id=f"doc-{index:05d}",
                page_content=(
                    f"Document {index} discusses {topic} in a {source} "
                    f"from {year}. It belongs to {access} material in {shard}."
                ),
                metadata={
                    "topic": topic,
                    "source": source,
                    "year": year,
                    "access": access,
                    "shard": shard,
                },
            )
        )
    return documents


def matches_filter(document: Document, metadata_filter: dict[str, Any]) -> bool:
    return all(
        document.metadata.get(key) == value for key, value in metadata_filter.items()
    )


def evaluate_case(
    *,
    name: str,
    documents: list[Document],
    metadata_filter: dict[str, Any],
    limit: int,
    actual: list[Document],
    latency_ms: float,
) -> CaseResult:
    expected_all = [
        document for document in documents if matches_filter(document, metadata_filter)
    ]
    expected_limited = expected_all[:limit]
    expected_ids = [document.id for document in expected_limited]
    actual_ids = [document.id for document in actual]
    expected_id_set = set(expected_ids)
    actual_id_set = set(actual_ids)
    true_positive = len(expected_id_set & actual_id_set)
    precision = (
        true_positive / len(actual_ids) if actual_ids else float(not expected_ids)
    )
    recall_at_limit = (
        true_positive / len(expected_ids) if expected_ids else float(not actual_ids)
    )

    return CaseResult(
        name=name,
        filter=metadata_filter,
        limit=limit,
        expected_total=len(expected_all),
        selectivity=len(expected_all) / len(documents),
        returned=len(actual),
        precision=precision,
        recall_at_limit=recall_at_limit,
        latency_ms=latency_ms,
        passed=actual_ids == expected_ids,
    )


def timed_filter_search(
    vectorstore: AlayaLite, metadata_filter: dict[str, Any], limit: int
) -> tuple[list[Document], float]:
    start = time.perf_counter()
    docs = vectorstore.filter_search(metadata_filter, k=limit)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return docs, elapsed_ms


def timed_retriever_filter_search(
    vectorstore: AlayaLite, metadata_filter: dict[str, Any], limit: int
) -> tuple[list[Document], float]:
    retriever = vectorstore.as_retriever(
        search_type="filter",
        k=limit,
        search_kwargs={"filter": metadata_filter},
    )
    start = time.perf_counter()
    docs = retriever.invoke("metadata-only filter query")
    elapsed_ms = (time.perf_counter() - start) * 1000
    return docs, elapsed_ms


def run_experiment(corpus_size: int, data_dir: Path) -> ExperimentResult:
    documents = build_documents(corpus_size)
    if data_dir.exists():
        if "metadata_filtering" not in data_dir.name:
            raise ValueError(
                "Refusing to delete an existing data directory whose name does not "
                "contain 'metadata_filtering'."
            )
        shutil.rmtree(data_dir)

    vectorstore = AlayaLite(
        embedding_function=DeterministicMetadataEmbeddings(),
        collection_name="metadata_filtering_experiment",
        url=str(data_dir),
        drop_old=True,
    )

    start = time.perf_counter()
    vectorstore.add_documents(documents)
    build_time_ms = (time.perf_counter() - start) * 1000

    cases = [
        ("single_metadata_key", {"topic": "vector-db"}, 20),
        ("two_metadata_keys", {"topic": "rag", "source": "manual"}, 20),
        ("three_metadata_keys", {"topic": "ann", "source": "blog", "year": 2025}, 20),
        ("limit_is_respected", {"access": "public"}, 7),
        ("no_match_returns_empty", {"topic": "missing-topic"}, 10),
    ]

    vectorstore_cases = []
    for name, metadata_filter, limit in cases:
        actual, latency_ms = timed_filter_search(vectorstore, metadata_filter, limit)
        vectorstore_cases.append(
            evaluate_case(
                name=name,
                documents=documents,
                metadata_filter=metadata_filter,
                limit=limit,
                actual=actual,
                latency_ms=latency_ms,
            )
        )

    retriever_filter = {"topic": "langchain", "access": "public"}
    retriever_limit = 15
    retriever_docs, retriever_latency_ms = timed_retriever_filter_search(
        vectorstore, retriever_filter, retriever_limit
    )
    retriever_case = evaluate_case(
        name="langchain_retriever_filter",
        documents=documents,
        metadata_filter=retriever_filter,
        limit=retriever_limit,
        actual=retriever_docs,
        latency_ms=retriever_latency_ms,
    )

    delete_filter = {"access": "internal"}
    expected_deleted = len(
        [document for document in documents if matches_filter(document, delete_filter)]
    )
    vectorstore.delete(filter=delete_filter)
    remaining_internal = vectorstore.filter_search(delete_filter, k=1)
    remaining_public = vectorstore.filter_search({"access": "public"}, k=corpus_size)
    delete_by_filter = {
        "filter": delete_filter,
        "expected_deleted": expected_deleted,
        "remaining_deleted_matches": len(remaining_internal),
        "remaining_public_matches": len(remaining_public),
        "passed": len(remaining_internal) == 0
        and len(remaining_public) == corpus_size - expected_deleted,
    }

    all_passed = (
        all(case.passed for case in vectorstore_cases)
        and retriever_case.passed
        and bool(delete_by_filter["passed"])
    )

    return ExperimentResult(
        corpus_size=corpus_size,
        build_time_ms=build_time_ms,
        vectorstore_cases=vectorstore_cases,
        retriever_case=retriever_case,
        delete_by_filter=delete_by_filter,
        all_passed=all_passed,
    )


def write_report(result: ExperimentResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dict = asdict(result)
    (output_dir / "metadata_filtering_results.json").write_text(
        json.dumps(result_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rows = [
        "| Case | Filter | Expected total | Selectivity | Returned | Precision | Recall@limit | Latency ms | Passed |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in result.vectorstore_cases + [result.retriever_case]:
        rows.append(
            "| "
            f"{case.name} | `{json.dumps(case.filter, ensure_ascii=False)}` | "
            f"{case.expected_total} | {case.selectivity:.3%} | {case.returned} | "
            f"{case.precision:.3f} | {case.recall_at_limit:.3f} | "
            f"{case.latency_ms:.3f} | {case.passed} |"
        )

    delete_result = result.delete_by_filter
    report = f"""# AlayaLite Metadata Filtering Experiment

This report validates AlayaLite metadata filtering through the
`langchain-alayalite` integration.

## Configuration

- Corpus size: {result.corpus_size}
- Embedding model: deterministic local test embeddings
- Vector store: `langchain_alayalite.AlayaLite`
- Retriever: `AlayaLite.as_retriever(search_type="filter")`
- Build time: {result.build_time_ms:.3f} ms

## Filter Query Results

{chr(10).join(rows)}

## Delete By Filter Result

- Filter: `{json.dumps(delete_result["filter"], ensure_ascii=False)}`
- Expected deleted documents: {delete_result["expected_deleted"]}
- Remaining matches for deleted filter: {delete_result["remaining_deleted_matches"]}
- Remaining public documents: {delete_result["remaining_public_matches"]}
- Passed: {delete_result["passed"]}

## Conclusion

All metadata filtering checks passed: {result.all_passed}

Precision and recall@limit are expected to be 1.0 for a correct exact-match
metadata filter. The scale-sensitive evidence is the selectivity and latency
columns: the experiment scans filters with 0% to 50% selectivity over
{result.corpus_size} documents while preserving exact results.

The experiment verifies exact-match metadata filtering for single-key and
multi-key predicates, empty-result behavior, limit handling, LangChain retriever
integration, and deletion by metadata filter.
"""
    (output_dir / "metadata_filtering_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-size", type=int, default=50000)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(".alayalite_metadata_filtering_data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/metadata_filtering/results"),
    )
    args = parser.parse_args()

    if args.corpus_size < 1000:
        raise ValueError("corpus size must be at least 1000 for the thesis experiment")

    result = run_experiment(args.corpus_size, args.data_dir)
    write_report(result, args.output_dir)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    if not result.all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
