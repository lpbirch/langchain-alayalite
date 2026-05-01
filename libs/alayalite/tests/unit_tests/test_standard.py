from collections.abc import Generator
from pathlib import Path

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

from langchain_alayalite import AlayaLite


class TestAlayaLiteStandard(VectorStoreIntegrationTests):
    @property
    def has_async(self) -> bool:
        return True

    @pytest.fixture
    def vectorstore(self, tmp_path: Path) -> Generator[VectorStore, None, None]:  # type: ignore[override]
        """Get an empty vectorstore for unit tests."""
        store = AlayaLite(
            embedding_function=self.get_embeddings(),
            drop_old=True,
            url=str(tmp_path / "alayalite_data"),
        )
        try:
            yield store
        finally:
            store.delete_collection()
