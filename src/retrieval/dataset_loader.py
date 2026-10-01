from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator


class DatasetValidationError(ValueError):
    """Raised when the dataset does not match the expected schema."""


class ComplexTableDatasetLoader:
    """Load and validate the complex-table RAG dataset."""

    REQUIRED_ROOT_KEYS = {"schema", "dataset", "pages"}
    REQUIRED_PAGE_KEYS = {
        "page",
        "table_id",
        "title",
        "equipment",
        "row_count",
        "rows",
        "notes",
    }
    REQUIRED_ROW_KEYS = {"no", "parameter"}

    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)
        self.data: dict[str, Any] | None = None

    def load(self) -> dict[str, Any]:
        """Load JSON from disk and validate its structure."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.dataset_path}"
            )

        if self.dataset_path.suffix.lower() != ".json":
            raise ValueError(
                f"Dataset must be a JSON file: {self.dataset_path}"
            )

        try:
            with self.dataset_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(
                f"Invalid JSON: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise DatasetValidationError(
                "Dataset root must be a JSON object."
            )

        self._validate(data)
        self.data = data

        return data

    def _validate(self, data: dict[str, Any]) -> None:
        """Validate the dataset without modifying its content."""
        missing = self.REQUIRED_ROOT_KEYS - data.keys()
        if missing:
            raise DatasetValidationError(
                f"Missing root keys: {sorted(missing)}"
            )

        schema = data["schema"]

        if not isinstance(schema, dict):
            raise DatasetValidationError(
                "'schema' must be an object."
            )

        if schema.get("name") != "complex-table-rag-dataset":
            raise DatasetValidationError(
                f"Unexpected schema name: {schema.get('name')!r}"
            )

        if not isinstance(data["pages"], list):
            raise DatasetValidationError(
                "'pages' must be a list."
            )

        if not data["pages"]:
            raise DatasetValidationError(
                "Dataset contains no pages."
            )

        for page_index, page in enumerate(data["pages"], start=1):
            self._validate_page(page, page_index)

        # Validate declared aggregate metadata when present.
        dataset_meta = data["dataset"]

        if isinstance(dataset_meta, dict):
            declared_pages = dataset_meta.get("page_count")
            declared_rows = dataset_meta.get("total_rows")

            actual_pages = len(data["pages"])
            actual_rows = sum(
                len(page["rows"])
                for page in data["pages"]
            )

            if (
                declared_pages is not None
                and declared_pages != actual_pages
            ):
                raise DatasetValidationError(
                    f"page_count mismatch: declared={declared_pages}, "
                    f"actual={actual_pages}"
                )

            if (
                declared_rows is not None
                and declared_rows != actual_rows
            ):
                raise DatasetValidationError(
                    f"total_rows mismatch: declared={declared_rows}, "
                    f"actual={actual_rows}"
                )

    def _validate_page(
        self,
        page: Any,
        page_index: int,
    ) -> None:
        if not isinstance(page, dict):
            raise DatasetValidationError(
                f"Page {page_index} must be an object."
            )

        missing = self.REQUIRED_PAGE_KEYS - page.keys()
        if missing:
            raise DatasetValidationError(
                f"Page {page_index} missing keys: {sorted(missing)}"
            )

        if not isinstance(page["rows"], list):
            raise DatasetValidationError(
                f"Page {page_index}: 'rows' must be a list."
            )

        if page["row_count"] != len(page["rows"]):
            raise DatasetValidationError(
                f"Page {page_index}: row_count mismatch."
            )

        for row_index, row in enumerate(page["rows"], start=1):
            if not isinstance(row, dict):
                raise DatasetValidationError(
                    f"Page {page_index}, row {row_index} must be an object."
                )

            missing_row_keys = self.REQUIRED_ROW_KEYS - row.keys()
            if missing_row_keys:
                raise DatasetValidationError(
                    f"Page {page_index}, row {row_index} "
                    f"missing keys: {sorted(missing_row_keys)}"
                )

    def require_loaded(self) -> dict[str, Any]:
        """Return loaded data or raise a clear error."""
        if self.data is None:
            raise RuntimeError(
                "Dataset has not been loaded. Call load() first."
            )

        return self.data

    def get_pages(self) -> list[dict[str, Any]]:
        """Return the original page records."""
        return self.require_loaded()["pages"]

    def get_page(self, page_number: int) -> dict[str, Any]:
        """Return one page by its document page number."""
        for page in self.get_pages():
            if page["page"] == page_number:
                return page

        raise KeyError(
            f"Page {page_number} not found."
        )

    def iter_rows(self) -> Iterator[dict[str, Any]]:
        """
        Iterate over original rows while attaching minimal provenance.

        The row content itself is not altered.
        """
        for page in self.get_pages():
            for row in page["rows"]:
                yield {
                    "page": page["page"],
                    "table_id": page["table_id"],
                    "equipment": page["equipment"],
                    "row": row,
                }

    def statistics(self) -> dict[str, int]:
        """Return basic dataset statistics."""
        pages = self.get_pages()

        return {
            "pages": len(pages),
            "tables": len(pages),
            "rows": sum(len(page["rows"]) for page in pages),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and validate the complex-table RAG dataset."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the dataset JSON file.",
    )

    args = parser.parse_args()

    loader = ComplexTableDatasetLoader(args.input)
    loader.load()

    stats = loader.statistics()

    print("=" * 72)
    print("RAG DATASET LOADER")
    print("=" * 72)
    print(f"Dataset : {loader.dataset_path}")
    print(f"Pages   : {stats['pages']}")
    print(f"Tables  : {stats['tables']}")
    print(f"Rows    : {stats['rows']}")
    print("-" * 72)
    print("VALIDATION : PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()