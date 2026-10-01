from pathlib import Path
import argparse
import json
from collections import Counter


class NormalizedDocumentValidator:

    def __init__(self, input_path: str):
        self.input_path = Path(input_path)

        self.data = None

        self.errors = []
        self.warnings = []

        self.stats = {
            "pages": 0,
            "blocks": 0,
            "tables": 0,
            "table_rows": 0,
            "table_cells": 0,
            "table_images": 0,
            "rowspan_cells": 0,
            "colspan_cells": 0,
            "empty_tables": 0,
            "dsh_hits": 0,
        }

        self.block_types = Counter()
        self.dsh_hits = []

    # ---------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------

    def load(self):

        if not self.input_path.exists():
            self.errors.append(
                f"Input file does not exist: {self.input_path}"
            )
            return False

        if self.input_path.suffix.lower() != ".json":
            self.errors.append(
                "Input file must be a JSON file."
            )
            return False

        try:

            with open(
                self.input_path,
                "r",
                encoding="utf-8"
            ) as file:

                self.data = json.load(file)

            return True

        except json.JSONDecodeError as exc:

            self.errors.append(
                f"Invalid JSON: {exc}"
            )

            return False

        except Exception as exc:

            self.errors.append(
                f"Failed to read JSON: {exc}"
            )

            return False

    # ---------------------------------------------------------
    # BASIC SCHEMA VALIDATION
    # ---------------------------------------------------------

    def validate_schema(self):

        if not isinstance(self.data, dict):

            self.errors.append(
                "Root JSON must be an object."
            )

            return

        required_root_keys = [
            "schema",
            "source",
            "extraction",
            "document"
        ]

        for key in required_root_keys:

            if key not in self.data:

                self.errors.append(
                    f"Missing root key: {key}"
                )

        schema = self.data.get("schema", {})

        if schema.get("name") != "pdf-ingestion-normalized":

            self.warnings.append(
                "Unexpected schema name."
            )

        if "version" not in schema:

            self.warnings.append(
                "Schema version is missing."
            )

        document = self.data.get("document")

        if not isinstance(document, dict):

            self.errors.append(
                "'document' must be an object."
            )

            return

        if "page_count" not in document:

            self.errors.append(
                "'document.page_count' is missing."
            )

        if "pages" not in document:

            self.errors.append(
                "'document.pages' is missing."
            )

    # ---------------------------------------------------------
    # DOCUMENT VALIDATION
    # ---------------------------------------------------------

    def validate_document(self):

        document = self.data.get("document", {})

        pages = document.get("pages", [])

        if not isinstance(pages, list):

            self.errors.append(
                "'document.pages' must be a list."
            )

            return

        self.stats["pages"] = len(pages)

        declared_page_count = document.get(
            "page_count"
        )

        if declared_page_count != len(pages):

            self.errors.append(
                f"Page count mismatch: "
                f"declared={declared_page_count}, "
                f"actual={len(pages)}"
            )

        page_numbers = []

        for page in pages:

            page_number = page.get("page")

            page_numbers.append(page_number)

            if "blocks" not in page:

                self.errors.append(
                    f"Page {page_number}: missing blocks."
                )

                continue

            if not isinstance(
                page["blocks"],
                list
            ):

                self.errors.append(
                    f"Page {page_number}: blocks "
                    f"must be a list."
                )

    # ---------------------------------------------------------
    # BLOCK VALIDATION
    # ---------------------------------------------------------

    def validate_blocks(self):

        pages = self.data.get(
            "document",
            {}
        ).get(
            "pages",
            []
        )

        for page in pages:

            page_number = page.get(
                "page",
                "UNKNOWN"
            )

            blocks = page.get(
                "blocks",
                []
            )

            for block_index, block in enumerate(
                blocks
            ):

                self.stats["blocks"] += 1

                block_type = block.get(
                    "type",
                    "UNKNOWN"
                )

                self.block_types[
                    block_type
                ] += 1

                if "type" not in block:

                    self.errors.append(
                        f"Page {page_number}, "
                        f"block {block_index}: "
                        f"missing type."
                    )

                if "page" not in block:

                    self.warnings.append(
                        f"Page {page_number}, "
                        f"block {block_index}: "
                        f"page field missing."
                    )

                if "bbox" not in block:

                    self.warnings.append(
                        f"Page {page_number}, "
                        f"block {block_index}: "
                        f"bbox missing."
                    )

    # ---------------------------------------------------------
    # TABLE VALIDATION
    # ---------------------------------------------------------

    def validate_tables(self):

        pages = self.data.get(
            "document",
            {}
        ).get(
            "pages",
            []
        )

        for page in pages:

            page_number = page.get(
                "page",
                "UNKNOWN"
            )

            for block_index, block in enumerate(
                page.get("blocks", [])
            ):

                if block.get("type") != "table":
                    continue

                self.stats["tables"] += 1

                parts = block.get(
                    "parts",
                    []
                )

                table_found = False

                for part in parts:

                    if part.get("type") != "table_body":
                        continue

                    table = part.get(
                        "table"
                    )

                    if not isinstance(
                        table,
                        dict
                    ):
                        continue

                    table_found = True

                    rows = table.get(
                        "rows",
                        []
                    )

                    row_count = table.get(
                        "row_count"
                    )

                    column_count = table.get(
                        "column_count"
                    )

                    if not rows:

                        self.stats[
                            "empty_tables"
                        ] += 1

                        self.warnings.append(
                            f"Page {page_number}, "
                            f"table {block_index}: "
                            f"empty rows."
                        )

                        continue

                    self.stats[
                        "table_rows"
                    ] += len(rows)

                    if row_count != len(rows):

                        self.errors.append(
                            f"Page {page_number}, "
                            f"table {block_index}: "
                            f"row_count mismatch. "
                            f"Declared={row_count}, "
                            f"Actual={len(rows)}"
                        )

                    for row_index, row in enumerate(
                        rows
                    ):

                        if not isinstance(
                            row,
                            list
                        ):
                            self.errors.append(
                                f"Page {page_number}, "
                                f"table {block_index}, "
                                f"row {row_index}: "
                                f"row is not a list."
                            )
                            continue

                        for cell in row:

                            self.stats[
                                "table_cells"
                            ] += 1

                            if not isinstance(
                                cell,
                                dict
                            ):
                                self.errors.append(
                                    f"Page {page_number}, "
                                    f"table {block_index}: "
                                    f"invalid cell."
                                )
                                continue

                            self.validate_cell(
                                cell,
                                page_number,
                                block_index,
                                column_count
                            )

                    image_path = part.get(
                        "image_path"
                    )

                    if image_path:

                        self.stats[
                            "table_images"
                        ] += 1

                if not table_found:

                    self.warnings.append(
                        f"Page {page_number}, "
                        f"table {block_index}: "
                        f"table block has no "
                        f"table_body."
                    )

    # ---------------------------------------------------------
    # CELL VALIDATION
    # ---------------------------------------------------------

    def validate_cell(
        self,
        cell,
        page_number,
        table_index,
        column_count
    ):

        required_fields = [
            "text",
            "row",
            "column",
            "rowspan",
            "colspan"
        ]

        for field in required_fields:

            if field not in cell:

                self.errors.append(
                    f"Page {page_number}, "
                    f"table {table_index}: "
                    f"cell missing '{field}'."
                )

        rowspan = cell.get(
            "rowspan",
            1
        )

        colspan = cell.get(
            "colspan",
            1
        )

        column = cell.get(
            "column",
            0
        )

        if rowspan > 1:

            self.stats[
                "rowspan_cells"
            ] += 1

        if colspan > 1:

            self.stats[
                "colspan_cells"
            ] += 1

        # Check horizontal table bounds
        if (
            column_count is not None
            and isinstance(column, int)
            and isinstance(colspan, int)
        ):

            if column + colspan > column_count:

                self.errors.append(
                    f"Page {page_number}, "
                    f"table {table_index}: "
                    f"cell exceeds column boundary. "
                    f"column={column}, "
                    f"colspan={colspan}, "
                    f"columns={column_count}"
                )

        # Check invalid span values
        if not isinstance(
            rowspan,
            int
        ) or rowspan < 1:

            self.errors.append(
                f"Page {page_number}, "
                f"table {table_index}: "
                f"invalid rowspan={rowspan}"
            )

        if not isinstance(
            colspan,
            int
        ) or colspan < 1:

            self.errors.append(
                f"Page {page_number}, "
                f"table {table_index}: "
                f"invalid colspan={colspan}"
            )

    # ---------------------------------------------------------
    # DSH VALIDATION
    # ---------------------------------------------------------

    def validate_dsh_content(self):

        pages = self.data.get(
            "document",
            {}
        ).get(
            "pages",
            []
        )

        search_terms = [
            "DSH 4",
            'DSH 4"',
            "DSH 4”"
        ]

        for page in pages:

            page_number = page.get(
                "page",
                "UNKNOWN"
            )

            for block_index, block in enumerate(
                page.get("blocks", [])
            ):

                # Normal block text
                block_text = block.get(
                    "text",
                    ""
                )

                if self.contains_dsh(
                    block_text,
                    search_terms
                ):

                    self.stats[
                        "dsh_hits"
                    ] += 1

                    self.dsh_hits.append({
                        "page": page_number,
                        "block": block_index,
                        "location": "block",
                        "text": block_text[:300]
                    })

                # Table cells
                if block.get("type") != "table":
                    continue

                for part in block.get(
                    "parts",
                    []
                ):

                    table = part.get(
                        "table"
                    )

                    if not table:
                        continue

                    for row in table.get(
                        "rows",
                        []
                    ):

                        for cell in row:

                            cell_text = cell.get(
                                "text",
                                ""
                            )

                            if self.contains_dsh(
                                cell_text,
                                search_terms
                            ):

                                self.stats[
                                    "dsh_hits"
                                ] += 1

                                self.dsh_hits.append({
                                    "page": page_number,
                                    "block": block_index,
                                    "location": "table_cell",
                                    "text": cell_text[:300]
                                })

    # ---------------------------------------------------------
    # HELPER
    # ---------------------------------------------------------

    @staticmethod
    def contains_dsh(
        text,
        search_terms
    ):

        if not text:
            return False

        normalized = (
            str(text)
            .upper()
            .replace("”", '"')
            .replace("“", '"')
        )

        return any(
            term.upper()
                .replace("”", '"')
                .replace("“", '"')
            in normalized
            for term in search_terms
        )

    # ---------------------------------------------------------
    # REPORT
    # ---------------------------------------------------------

    def print_report(self):

        print()
        print("=" * 80)
        print("NORMALIZED DOCUMENT VALIDATION REPORT")
        print("=" * 80)

        print()
        print("INPUT")
        print("-" * 80)
        print(self.input_path)

        # -----------------------------------------------------
        # METADATA
        # -----------------------------------------------------

        print()
        print("DOCUMENT METADATA")
        print("-" * 80)

        schema = self.data.get(
            "schema",
            {}
        )

        extraction = self.data.get(
            "extraction",
            {}
        )

        source = self.data.get(
            "source",
            {}
        )

        print(
            f"Schema       : "
            f"{schema.get('name')}"
        )

        print(
            f"Version      : "
            f"{schema.get('version')}"
        )

        print(
            f"Source       : "
            f"{source.get('filename', 'N/A')}"
        )

        print(
            f"Pages        : "
            f"{self.stats['pages']}"
        )

        print(
            f"Extraction   : "
            f"{extraction.get('engine', 'N/A')}"
        )

        mineru = extraction.get(
            "mineru",
            {}
        )

        print(
            f"MinerU Tier  : "
            f"{mineru.get('tier', 'N/A')}"
        )

        print(
            f"Parse Mode   : "
            f"{mineru.get('parse_mode', 'N/A')}"
        )

        # -----------------------------------------------------
        # BLOCK STATISTICS
        # -----------------------------------------------------

        print()
        print("BLOCK STATISTICS")
        print("-" * 80)

        print(
            f"Total Blocks : "
            f"{self.stats['blocks']}"
        )

        for block_type, count in sorted(
            self.block_types.items()
        ):

            print(
                f"  {block_type:<20} "
                f"{count}"
            )

        # -----------------------------------------------------
        # TABLE STATISTICS
        # -----------------------------------------------------

        print()
        print("TABLE STATISTICS")
        print("-" * 80)

        print(
            f"Tables              : "
            f"{self.stats['tables']}"
        )

        print(
            f"Table Rows          : "
            f"{self.stats['table_rows']}"
        )

        print(
            f"Table Cells         : "
            f"{self.stats['table_cells']}"
        )

        print(
            f"Table Images        : "
            f"{self.stats['table_images']}"
        )

        print(
            f"Rowspan Cells       : "
            f"{self.stats['rowspan_cells']}"
        )

        print(
            f"Colspan Cells       : "
            f"{self.stats['colspan_cells']}"
        )

        print(
            f"Empty Tables        : "
            f"{self.stats['empty_tables']}"
        )

        # -----------------------------------------------------
        # DSH TEST
        # -----------------------------------------------------

        print()
        print("CONTENT VALIDATION")
        print("-" * 80)

        print(
            f"DSH 4 occurrences : "
            f"{self.stats['dsh_hits']}"
        )

        if self.dsh_hits:

            print()
            print("DSH 4 LOCATIONS")
            print("-" * 80)

            # Avoid printing hundreds of duplicates
            shown = set()

            for hit in self.dsh_hits:

                key = (
                    hit["page"],
                    hit["block"],
                    hit["location"],
                    hit["text"]
                )

                if key in shown:
                    continue

                shown.add(key)

                print(
                    f"Page {hit['page']}, "
                    f"Block {hit['block']}, "
                    f"{hit['location']}"
                )

                print(
                    f"  {hit['text']}"
                )

        # -----------------------------------------------------
        # ERRORS
        # -----------------------------------------------------

        print()
        print("ERRORS")
        print("-" * 80)

        if not self.errors:

            print("No structural errors detected.")

        else:

            for error in self.errors:

                print(
                    f"[ERROR] {error}"
                )

        # -----------------------------------------------------
        # WARNINGS
        # -----------------------------------------------------

        print()
        print("WARNINGS")
        print("-" * 80)

        if not self.warnings:

            print("No warnings.")

        else:

            for warning in self.warnings[:50]:

                print(
                    f"[WARNING] {warning}"
                )

            if len(self.warnings) > 50:

                print(
                    f"... "
                    f"{len(self.warnings) - 50} "
                    f"additional warnings."
                )

        # -----------------------------------------------------
        # FINAL STATUS
        # -----------------------------------------------------

        print()
        print("=" * 80)

        if self.errors:

            print(
                "VALIDATION STATUS : FAILED"
            )

        elif self.warnings:

            print(
                "VALIDATION STATUS : PASSED WITH WARNINGS"
            )

        else:

            print(
                "VALIDATION STATUS : PASSED"
            )

        print("=" * 80)

    # ---------------------------------------------------------
    # RUN
    # ---------------------------------------------------------

    def run(self):

        if not self.load():

            self.print_report()
            return False

        self.validate_schema()
        self.validate_document()
        self.validate_blocks()
        self.validate_tables()
        self.validate_dsh_content()

        self.print_report()

        return not self.errors


# =============================================================
# MAIN
# =============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Validate MinerU normalized "
            "PDF ingestion output."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to normalized_document.json"
    )

    args = parser.parse_args()

    validator = NormalizedDocumentValidator(
        input_path=args.input
    )

    success = validator.run()

    raise SystemExit(
        0 if success else 1
    )


if __name__ == "__main__":
    main()