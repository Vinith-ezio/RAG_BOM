import json
import argparse
from pathlib import Path
from typing import Any, Dict, List


class StructuredRecordBuilder:
    """
    Step 2:
    Convert the complex-table dataset into retrieval-ready
    structured records.

    No embeddings.
    No LLM.
    No search text generation.
    """

    def __init__(self, dataset: Dict[str, Any]):

        self.dataset = dataset
        self.records: List[Dict[str, Any]] = []

        self.stats = {
            "pages_processed": 0,
            "tables_processed": 0,
            "source_rows": 0,
            "records_created": 0,
            "blank_rows_skipped": 0,
            "header_rows_skipped": 0,
            "side_records_created": 0,
            "scalar_records_created": 0,
        }

    # ================================================================
    # Utility
    # ================================================================

    @staticmethod
    def is_empty(value: Any) -> bool:

        if value is None:
            return True

        if isinstance(value, str):
            return value.strip() == ""

        if isinstance(value, dict):
            return all(
                StructuredRecordBuilder.is_empty(v)
                for v in value.values()
            )

        if isinstance(value, list):
            return len(value) == 0

        return False

    # ================================================================
    # Check whether row contains actual data
    # ================================================================

    @staticmethod
    def has_meaningful_content(row: Dict[str, Any]) -> bool:

        ignored_keys = {
            "no",
            "type",
        }

        for key, value in row.items():

            if key in ignored_keys:
                continue

            if not StructuredRecordBuilder.is_empty(value):
                return True

        return False

    # ================================================================
    # Record ID
    # ================================================================

    @staticmethod
    def make_record_id(
        page_number: int,
        row_number: int,
        side: str | None = None,
    ) -> str:

        record_id = (
            f"p{page_number:02d}"
            f"_r{row_number}"
        )

        if side:
            record_id += f"_{side}"

        return record_id

    # ================================================================
    # Extract non-side values
    # ================================================================

    @staticmethod
    def extract_scalar_values(
        row: Dict[str, Any]
    ) -> Dict[str, Any]:

        excluded_keys = {
            "no",
            "type",
            "parameter",
            "sub_parameter",
            "sub_sub_parameter",
            "inlet",
            "outlet",
        }

        values = {}

        for key, value in row.items():

            if key in excluded_keys:
                continue

            if StructuredRecordBuilder.is_empty(value):
                continue

            values[key] = value

        return values

    # ================================================================
    # Build side record
    # ================================================================

    def build_side_record(
        self,
        page_number: int,
        table_id: str,
        row: Dict[str, Any],
        section: str | None,
        subsection: str | None,
        side: str,
    ) -> Dict[str, Any]:

        row_number = row.get("no")

        return {
            "record_id": self.make_record_id(
                page_number,
                row_number,
                side,
            ),

            "page": page_number,

            "table_id": table_id,

            "row_number": row_number,

            "section": section,

            "subsection": subsection,

            "parameter": row.get("parameter"),

            "sub_parameter": row.get("sub_parameter"),

            "sub_sub_parameter": row.get(
                "sub_sub_parameter"
            ),

            "side": side,

            "values": row.get(side, {}),
        }

    # ================================================================
    # Build normal/scalar record
    # ================================================================

    def build_scalar_record(
        self,
        page_number: int,
        table_id: str,
        row: Dict[str, Any],
        section: str | None,
        subsection: str | None,
    ) -> Dict[str, Any]:

        row_number = row.get("no")

        return {
            "record_id": self.make_record_id(
                page_number,
                row_number,
            ),

            "page": page_number,

            "table_id": table_id,

            "row_number": row_number,

            "section": section,

            "subsection": subsection,

            "parameter": row.get("parameter"),

            "sub_parameter": row.get(
                "sub_parameter"
            ),

            "sub_sub_parameter": row.get(
                "sub_sub_parameter"
            ),

            "side": None,

            "values": self.extract_scalar_values(row),
        }

    # ================================================================
    # Process one page/table
    # ================================================================

    def process_page(
        self,
        page: Dict[str, Any],
    ):

        page_number = page.get("page")

        table_id = page.get(
            "table_id",
            f"instrument_specification_page_{page_number}"
        )

        rows = page.get("rows", [])

        self.stats["tables_processed"] += 1

        section = None
        subsection = None

        # ------------------------------------------------------------
        # Process rows
        # ------------------------------------------------------------

        for row in rows:

            self.stats["source_rows"] += 1

            if not isinstance(row, dict):
                continue

            row_type = row.get("type")

            # --------------------------------------------------------
            # Section Header
            # --------------------------------------------------------

            if row_type == "section_header":

                section = row.get("parameter")

                subsection = None

                self.stats["header_rows_skipped"] += 1

                continue

            # --------------------------------------------------------
            # Subsection Header
            # --------------------------------------------------------

            if row_type == "subsection_header":

                subsection = row.get("parameter")

                self.stats["header_rows_skipped"] += 1

                continue

            # --------------------------------------------------------
            # Empty row
            # --------------------------------------------------------

            if not self.has_meaningful_content(row):

                self.stats["blank_rows_skipped"] += 1

                continue

            # --------------------------------------------------------
            # Inlet / Outlet records
            # --------------------------------------------------------

            has_inlet = (
                "inlet" in row
                and not self.is_empty(
                    row.get("inlet")
                )
            )

            has_outlet = (
                "outlet" in row
                and not self.is_empty(
                    row.get("outlet")
                )
            )

            if has_inlet or has_outlet:

                if has_inlet:

                    record = self.build_side_record(
                        page_number=page_number,
                        table_id=table_id,
                        row=row,
                        section=section,
                        subsection=subsection,
                        side="inlet",
                    )

                    self.records.append(record)

                    self.stats[
                        "records_created"
                    ] += 1

                    self.stats[
                        "side_records_created"
                    ] += 1

                if has_outlet:

                    record = self.build_side_record(
                        page_number=page_number,
                        table_id=table_id,
                        row=row,
                        section=section,
                        subsection=subsection,
                        side="outlet",
                    )

                    self.records.append(record)

                    self.stats[
                        "records_created"
                    ] += 1

                    self.stats[
                        "side_records_created"
                    ] += 1

                continue

            # --------------------------------------------------------
            # Normal scalar record
            # --------------------------------------------------------

            record = self.build_scalar_record(
                page_number=page_number,
                table_id=table_id,
                row=row,
                section=section,
                subsection=subsection,
            )

            self.records.append(record)

            self.stats[
                "records_created"
            ] += 1

            self.stats[
                "scalar_records_created"
            ] += 1

    # ================================================================
    # Build complete dataset
    # ================================================================

    def build(self) -> Dict[str, Any]:

        pages = self.dataset.get(
            "pages",
            []
        )

        self.stats[
            "pages_processed"
        ] = len(pages)

        for page in pages:

            self.process_page(page)

        # ------------------------------------------------------------
        # Output structure
        # ------------------------------------------------------------

        dataset_metadata = self.dataset.get(
            "dataset",
            {}
        )

        return {

            "schema": {
                "name": "complex-table-retrieval-records",
                "version": "1.0",
            },

            "source_dataset": {

                "name": dataset_metadata.get(
                    "dataset_name"
                ),

                "version": self.dataset.get(
                    "schema",
                    {}
                ).get(
                    "version"
                ),

                "document": dataset_metadata.get(
                    "document_name"
                ),
            },

            "statistics": self.stats,

            "records": self.records,
        }


# ====================================================================
# MAIN
# ====================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build structured retrieval records "
            "from complex table dataset."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input complex table JSON dataset.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output retrieval records JSON.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    output_path = Path(args.output)

    print("=" * 72)
    print("STRUCTURED RECORD BUILDER")
    print("=" * 72)

    print(f"Input  : {input_path}")
    print(f"Output : {output_path}")

    print("-" * 72)

    # ================================================================
    # Load JSON
    # ================================================================

    with input_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        dataset = json.load(file)

    # ================================================================
    # Build records
    # ================================================================

    builder = StructuredRecordBuilder(
        dataset
    )

    result = builder.build()

    # ================================================================
    # Save
    # ================================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ================================================================
    # Print result
    # ================================================================

    stats = result["statistics"]

    print("-" * 72)

    print(
        f"Pages processed        : "
        f"{stats['pages_processed']}"
    )

    print(
        f"Tables processed       : "
        f"{stats['tables_processed']}"
    )

    print(
        f"Source rows            : "
        f"{stats['source_rows']}"
    )

    print(
        f"Headers skipped        : "
        f"{stats['header_rows_skipped']}"
    )

    print(
        f"Blank rows skipped     : "
        f"{stats['blank_rows_skipped']}"
    )

    print(
        f"Side records created   : "
        f"{stats['side_records_created']}"
    )

    print(
        f"Scalar records created : "
        f"{stats['scalar_records_created']}"
    )

    print(
        f"Total records created  : "
        f"{stats['records_created']}"
    )

    print("-" * 72)

    print("BUILD : PASSED")

    print("=" * 72)


if __name__ == "__main__":
    main()