from __future__ import annotations

import argparse
import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_text(value: Any) -> str:
    """
    Normalize extracted text without changing its meaning.
    """

    if value is None:
        return ""

    text = str(value)

    text = html.unescape(text)

    # Normalize common whitespace
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces while preserving newlines
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_bbox(bbox: Any) -> list[float] | None:
    """
    MinerU middle_json bboxes are normally normalized coordinates.
    Preserve them exactly as numeric values.
    """

    if not bbox:
        return None

    try:
        return [float(x) for x in bbox]
    except (TypeError, ValueError):
        return None


# ============================================================
# HTML TABLE PARSER
# ============================================================

class TableCell:
    """
    Internal representation of one HTML table cell.
    """

    def __init__(
        self,
        text: str,
        row: int,
        col: int,
        rowspan: int = 1,
        colspan: int = 1,
    ):
        self.text = clean_text(text)
        self.row = row
        self.col = col
        self.rowspan = rowspan
        self.colspan = colspan

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "row": self.row,
            "column": self.col,
            "rowspan": self.rowspan,
            "colspan": self.colspan,
        }


class MinerUTableParser(HTMLParser):
    """
    Parses the HTML table stored inside MinerU's table_body.content.

    Handles:
        <table>
        <tr>
        <td>
        rowspan
        colspan
    """

    def __init__(self):
        super().__init__(
            convert_charrefs=True
        )

        self.rows: list[list[TableCell]] = []

        self.current_row: list[TableCell] | None = None

        self.current_cell_text: list[str] = []

        self.current_cell_attrs: dict[str, str] = {}

        self.current_row_index = -1

        self.current_column_index = 0

        self.inside_cell = False

        self.pending_rowspans: dict[int, tuple[TableCell, int]] = {}

    # --------------------------------------------------------

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ):
        tag = tag.lower()

        attributes = {
            key.lower(): value or ""
            for key, value in attrs
        }

        if tag == "tr":

            self.current_row_index += 1

            self.current_row = []

            self.current_column_index = 0

            self.rows.append(
                self.current_row
            )

            self._fill_pending_rowspans()

        elif tag == "td" or tag == "th":

            self.inside_cell = True

            self.current_cell_text = []

            self.current_cell_attrs = attributes

            self._move_to_free_column()

    # --------------------------------------------------------

    def handle_endtag(self, tag: str):

        tag = tag.lower()

        if tag == "td" or tag == "th":

            if not self.inside_cell:
                return

            text = "".join(
                self.current_cell_text
            )

            rowspan = self._safe_int(
                self.current_cell_attrs.get(
                    "rowspan",
                    "1"
                )
            )

            colspan = self._safe_int(
                self.current_cell_attrs.get(
                    "colspan",
                    "1"
                )
            )

            cell = TableCell(
                text=text,
                row=self.current_row_index,
                col=self.current_column_index,
                rowspan=rowspan,
                colspan=colspan,
            )

            self.current_row.append(cell)

            # Reserve current cell position for rowspan/colspan
            for r in range(
                self.current_row_index,
                self.current_row_index + rowspan
            ):
                for c in range(
                    self.current_column_index,
                    self.current_column_index + colspan
                ):
                    if (
                        r > self.current_row_index
                        or c > self.current_column_index
                    ):
                        self.pending_rowspans[(r, c)] = (
                            cell,
                            r,
                        )

            self.current_column_index += colspan

            self.inside_cell = False

            self.current_cell_text = []

    # --------------------------------------------------------

    def handle_data(self, data: str):

        if self.inside_cell:
            self.current_cell_text.append(
                data
            )

    # --------------------------------------------------------

    @staticmethod
    def _safe_int(
        value: str
    ) -> int:

        try:
            result = int(value)
            return max(result, 1)
        except (TypeError, ValueError):
            return 1

    # --------------------------------------------------------

    def _move_to_free_column(self):

        while (
            self.current_row is not None
            and any(
                cell.col <= self.current_column_index
                < cell.col + cell.colspan
                for cell in self.current_row
            )
        ):
            self.current_column_index += 1

    # --------------------------------------------------------

    def _fill_pending_rowspans(self):

        if self.current_row is None:
            return

        reserved_columns = []

        for (
            (row, col),
            (cell, _),
        ) in list(
            self.pending_rowspans.items()
        ):

            if row == self.current_row_index:

                reserved_columns.append(col)

                del self.pending_rowspans[
                    (row, col)
                ]

        # The actual row representation retains original cells.
        # We don't insert duplicated cells here because rowspan /
        # colspan are preserved in each cell's metadata.


# ============================================================
# TABLE PROCESSING
# ============================================================

def parse_table_html(
    table_html: str,
) -> dict:

    parser = MinerUTableParser()

    parser.feed(table_html)

    cells: list[dict] = []

    max_column = 0

    for row in parser.rows:

        for cell in row:

            cell_dict = cell.to_dict()

            cells.append(
                cell_dict
            )

            max_column = max(
                max_column,
                cell.col + cell.colspan
            )

    # --------------------------------------------------------
    # Create a simple row-oriented representation.
    # --------------------------------------------------------

    rows: list[list[dict]] = []

    for row_index in range(
        len(parser.rows)
    ):

        row_cells = [
            cell
            for cell in cells
            if cell["row"] == row_index
        ]

        row_cells.sort(
            key=lambda x: x["column"]
        )

        rows.append(
            row_cells
        )

    return {
        "row_count": len(rows),
        "column_count": max_column,
        "rows": rows,
        "cells": cells,
        "raw_html": table_html,
    }


# ============================================================
# CONTENT EXTRACTION
# ============================================================

def extract_inline_content(
    content: Any,
) -> str:

    if isinstance(content, str):
        return clean_text(content)

    if not isinstance(content, list):
        return ""

    parts = []

    for item in content:

        if not isinstance(item, dict):
            continue

        value = item.get("content")

        if isinstance(value, str):
            parts.append(value)

    return clean_text(
        " ".join(parts)
    )


# ============================================================
# BLOCK NORMALIZATION
# ============================================================

def normalize_text_block(
    block: dict,
    page_idx: int,
) -> dict:

    block_type = block.get(
        "type",
        "unknown"
    )

    text = extract_inline_content(
        block.get("content")
    )

    result = {
        "type": block_type,
        "page": page_idx + 1,
        "page_index": page_idx,
        "index": block.get("index"),
        "bbox": normalize_bbox(
            block.get("bbox")
        ),
        "text": text,
    }

    return result


def normalize_table_block(
    block: dict,
    page_idx: int,
) -> dict:

    table_parts = block.get(
        "content",
        []
    )

    normalized_parts = []

    image_paths = []

    for part in table_parts:

        if not isinstance(part, dict):
            continue

        part_type = part.get(
            "type"
        )

        part_result = {
            "type": part_type,
            "bbox": normalize_bbox(
                part.get("bbox")
            ),
        }

        if part_type == "table_caption":

            caption = extract_inline_content(
                part.get("content")
            )

            part_result[
                "text"
            ] = caption

        elif part_type == "table_body":

            table_html = part.get(
                "content",
                ""
            )

            parsed_table = parse_table_html(
                table_html
            )

            part_result[
                "table"
            ] = parsed_table

            image_path = part.get(
                "image_path"
            )

            if image_path:

                image_paths.append(
                    image_path
                )

                part_result[
                    "image_path"
                ] = image_path

        else:

            part_result[
                "content"
            ] = part.get(
                "content"
            )

        normalized_parts.append(
            part_result
        )

    return {
        "type": "table",
        "page": page_idx + 1,
        "page_index": page_idx,
        "index": block.get("index"),
        "bbox": normalize_bbox(
            block.get("bbox")
        ),
        "parts": normalized_parts,
        "images": image_paths,
    }


def normalize_block(
    block: dict,
    page_idx: int,
) -> dict:

    block_type = block.get(
        "type",
        "unknown"
    )

    if block_type == "table":

        return normalize_table_block(
            block,
            page_idx
        )

    return normalize_text_block(
        block,
        page_idx
    )


# ============================================================
# PAGE NORMALIZATION
# ============================================================

def get_page_dimensions(
    data: dict,
    page_idx: int,
) -> dict | None:

    try:

        layout_pages = (
            data
            .get("extensions", {})
            .get("docvortex_layout", {})
            .get("pages", [])
        )

        for page in layout_pages:

            if page.get(
                "page_idx"
            ) == page_idx:

                return {
                    "width_pt": page.get(
                        "width_pt"
                    ),
                    "height_pt": page.get(
                        "height_pt"
                    ),
                }

    except Exception:
        pass

    return None


def normalize_page(
    data: dict,
    page: dict,
) -> dict:

    page_idx = page.get(
        "page_idx",
        0
    )

    blocks = page.get(
        "blocks",
        []
    )

    normalized_blocks = []

    for block in blocks:

        if not isinstance(
            block,
            dict
        ):
            continue

        normalized_blocks.append(
            normalize_block(
                block,
                page_idx
            )
        )

    return {
        "page": page_idx + 1,
        "page_index": page_idx,
        "dimensions": get_page_dimensions(
            data,
            page_idx
        ),
        "block_count": len(
            normalized_blocks
        ),
        "blocks": normalized_blocks,
    }


# ============================================================
# DOCUMENT NORMALIZATION
# ============================================================

class DocumentNormalizer:

    def __init__(
        self,
        input_path: str,
        output_path: str,
    ):

        self.input_path = Path(
            input_path
        )

        self.output_path = Path(
            output_path
        )

    # --------------------------------------------------------

    def validate(self):

        if not self.input_path.exists():

            raise FileNotFoundError(
                f"Input file not found: "
                f"{self.input_path}"
            )

        if (
            self.input_path.suffix.lower()
            != ".json"
        ):

            raise ValueError(
                "Input must be a JSON file."
            )

    # --------------------------------------------------------

    def load(self) -> dict:

        with self.input_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    # --------------------------------------------------------

    def normalize(
        self,
        data: dict,
    ) -> dict:

        pages = data.get(
            "pages",
            []
        )

        normalized_pages = []

        for page in pages:

            normalized_pages.append(
                normalize_page(
                    data,
                    page
                )
            )

        metadata = data.get(
            "metadata",
            {}
        )

        mineru_info = (
            data
            .get("extensions", {})
            .get("mineru", {})
        )

        result = {

            "schema": {
                "name": "pdf-ingestion-normalized",
                "version": "1.0"
            },

            "source": {
                "file_suffix": metadata.get(
                    "file_suffix"
                ),

                "producer": metadata.get(
                    "producer"
                ),

                "document": metadata.get(
                    "document"
                ),
            },

            "extraction": {
                "engine": "MinerU",
                "mineru": mineru_info,
                "source_schema": data.get(
                    "schema"
                ),
                "source_schema_version": data.get(
                    "schema_version"
                ),
            },

            "document": {
                "page_count": len(
                    normalized_pages
                ),

                "pages": normalized_pages
            }
        }

        return result

    # --------------------------------------------------------

    def save(
        self,
        result: dict,
    ):

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with self.output_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                result,
                file,
                ensure_ascii=False,
                indent=2
            )

    # --------------------------------------------------------

    def run(self):

        self.validate()

        print("=" * 80)
        print("DOCUMENT NORMALIZATION")
        print("=" * 80)

        print(
            f"\nInput : {self.input_path}"
        )

        print(
            f"Output: {self.output_path}"
        )

        print(
            "\nLoading MinerU Middle JSON..."
        )

        data = self.load()

        print(
            f"Pages found: "
            f"{len(data.get('pages', []))}"
        )

        print(
            "\nNormalizing document..."
        )

        result = self.normalize(
            data
        )

        self.save(
            result
        )

        print(
            "\nNormalization completed."
        )

        print(
            f"Output written to:\n"
            f"{self.output_path}"
        )

        return result


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Normalize MinerU middle_json.json "
            "into a RAG-ready structural representation."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to MinerU middle_json.json"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to normalized JSON output"
    )

    args = parser.parse_args()

    normalizer = DocumentNormalizer(
        input_path=args.input,
        output_path=args.output,
    )

    normalizer.run()


if __name__ == "__main__":
    main()