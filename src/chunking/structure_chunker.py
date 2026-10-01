from pathlib import Path
import argparse
import json
import re
from collections import Counter


class StructureAwareChunkerV3:

    def __init__(
        self,
        input_path: str,
        output_path: str,
        max_chars: int = 1800,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.max_chars = max_chars

        self.document = None
        self.chunks = []

        self.current_document = None
        self.current_section = None
        self.current_subsection = None
        self.current_clause = None

        self.chunk_counter = 0

        self.stats = {
            "pages_processed": 0,
            "source_blocks_processed": 0,

            "real_tables": 0,
            "layout_wrappers": 0,
            "semantic_text_blocks": 0,

            "ignored_headers": 0,
            "ignored_footers": 0,
            "ignored_page_numbers": 0,
            "ignored_layout_noise": 0,

            "headings_detected": 0,
            "sections_detected": 0,
            "subsections_detected": 0,
            "clauses_detected": 0,

            "table_chunks": 0,
            "text_chunks": 0,
            "image_chunks": 0,
        }

    # ============================================================
    # LOAD
    # ============================================================

    def load(self):

        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {self.input_path}"
            )

        with open(
            self.input_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.document = json.load(file)

    # ============================================================
    # TEXT CLEANING
    # ============================================================

    @staticmethod
    def clean_text(text):

        if text is None:
            return ""

        text = str(text)

        text = text.replace(
            "\u00a0",
            " "
        )

        text = text.replace(
            "\r",
            "\n"
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        text = re.sub(
            r"\n[ \t]+",
            "\n",
            text
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        return text.strip()

    # ============================================================
    # CELL TEXT
    # ============================================================

    @staticmethod
    def cell_text(cell):

        if not isinstance(cell, dict):
            return ""

        return StructureAwareChunkerV3.clean_text(
            cell.get(
                "text",
                ""
            )
        )

    # ============================================================
    # NORMALIZE OCR SPACING
    # ============================================================

    @staticmethod
    def normalize_ocr_text(text):

        text = StructureAwareChunkerV3.clean_text(
            text
        )

        if not text:
            return ""

        # Common MinerU/OCR spacing artifacts
        text = re.sub(
            r"\b([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])\b",
            lambda m: m.group(0),
            text
        )

        return text

    # ============================================================
    # DOCUMENT HEADER DETECTION
    # ============================================================

    @staticmethod
    def is_repeated_document_header(text):

        text = text.upper()

        patterns = [
            r"PRODUCT STANDARD",
            r"HYDERABAD",
            r"TC\s*65132",
            r"REV\s*NO",
            r"PAGE\s+\d+\s+OF\s+\d+",
        ]

        matches = sum(
            bool(re.search(pattern, text))
            for pattern in patterns
        )

        return matches >= 2

    # ============================================================
    # PAGE LABEL DETECTION
    # ============================================================

    @staticmethod
    def is_page_label(text):

        text = StructureAwareChunkerV3.clean_text(
            text
        )

        if not text:
            return False

        patterns = [
            r"^Page\s+\d+\s+of\s+\d+$",
            r"^Rev\.?\s*No\.?\s*\d+$",
            r"^TC\s*\d+$",
            r"^HYDERABAD$",
            r"^PRODUCT STANDARD$",
        ]

        return any(
            re.match(
                pattern,
                text,
                re.IGNORECASE
            )
            for pattern in patterns
        )

    # ============================================================
    # LAYOUT NOISE DETECTION
    # ============================================================

    @staticmethod
    def is_layout_noise(text):

        text = StructureAwareChunkerV3.clean_text(
            text
        )

        if not text:
            return True

        # Very short OCR fragments
        if len(text) <= 8:
            letters = re.findall(
                r"[A-Za-z]",
                text
            )

            if len(letters) <= 3:
                return True

        # Known OCR artifacts from the document
        noise_patterns = [
            r"^fe\.?\s*co\s*R\s*D$",
            r"^R\s*00\s*T",
            r"^10\s+00o",
            r"^D,\s*E",
        ]

        for pattern in noise_patterns:

            if re.search(
                pattern,
                text,
                re.IGNORECASE
            ):
                return True

        return False

    # ============================================================
    # HEADING DETECTION
    # ============================================================

    @staticmethod
    def detect_numbered_heading(text):

        text = StructureAwareChunkerV3.clean_text(
            text
        )

        if not text:
            return None

        match = re.match(
            r"^(\d+(?:\.\d+){0,4})\.?\s+(.+)$",
            text
        )

        if not match:
            return None

        number = match.group(1)
        title = match.group(2).strip()

        # Prevent long normal sentences
        if len(text) > 180:
            return None

        # Clause-like text normally ends with punctuation
        if title.endswith(
            (
                ".",
                ":",
                ";"
            )
        ) and len(title) > 40:

            return None

        return {
            "number": number,
            "title": title,
            "text": text
        }

    # ============================================================
    # LETTER HEADING
    # ============================================================

    @staticmethod
    def detect_letter_heading(text):

        text = StructureAwareChunkerV3.clean_text(
            text
        )

        if not text:
            return None

        pattern = (
            r"^[A-Z]\.\s+"
            r"[A-Z][A-Z0-9 &/\-(),.'']{2,}$"
        )

        if re.match(
            pattern,
            text
        ):
            return text

        return None

    # ============================================================
    # NUMBER LEVEL
    # ============================================================

    @staticmethod
    def number_level(number):

        if not number:
            return None

        return number.count(".") + 1

    # ============================================================
    # HIERARCHY UPDATE
    # ============================================================

    def update_hierarchy(
        self,
        heading
    ):

        number = heading.get(
            "number"
        )

        title = heading.get(
            "title"
        )

        text = heading.get(
            "text"
        )

        level = self.number_level(
            number
        )

        if level == 1:

            self.current_section = text
            self.current_subsection = None
            self.current_clause = None

            self.stats[
                "sections_detected"
            ] += 1

        elif level == 2:

            self.current_subsection = text
            self.current_clause = None

            self.stats[
                "subsections_detected"
            ] += 1

        else:

            self.current_clause = text

            self.stats[
                "clauses_detected"
            ] += 1

        self.stats[
            "headings_detected"
        ] += 1

    # ============================================================
    # RESET / DOCUMENT CONTEXT
    # ============================================================

    def reset_hierarchy_for_new_document_section(
        self,
        document_name
    ):

        if (
            self.current_document
            != document_name
        ):

            self.current_document = (
                document_name
            )

            self.current_section = None
            self.current_subsection = None
            self.current_clause = None

    # ============================================================
    # CHUNK ID
    # ============================================================

    def make_chunk_id(
        self,
        page,
        block_index
    ):

        chunk_id = (
            f"chunk_{self.chunk_counter:06d}_"
            f"page_{page}_"
            f"block_{block_index}"
        )

        self.chunk_counter += 1

        return chunk_id

    # ============================================================
    # ADD CHUNK
    # ============================================================

    def add_chunk(
        self,
        page,
        block_index,
        content_type,
        content,
        bbox=None,
        metadata=None,
        section=None,
        subsection=None,
        clause=None
    ):

        content = self.clean_text(
            content
        )

        if not content:
            return

        chunk = {

            "chunk_id":
                self.make_chunk_id(
                    page,
                    block_index
                ),

            "document_page":
                page,

            "block_index":
                block_index,

            "content_type":
                content_type,

            "hierarchy": {

                "document":
                    self.current_document,

                "section":
                    section
                    if section is not None
                    else self.current_section,

                "subsection":
                    subsection
                    if subsection is not None
                    else self.current_subsection,

                "clause":
                    clause
                    if clause is not None
                    else self.current_clause,
            },

            "content":
                content,

            "bbox":
                bbox,

            "metadata":
                metadata or {},
        }

        self.chunks.append(
            chunk
        )

    # ============================================================
    # SPLIT TEXT
    # ============================================================

    def split_text(
        self,
        text
    ):

        text = self.clean_text(
            text
        )

        if not text:
            return []

        if len(text) <= self.max_chars:
            return [text]

        # Prefer paragraph boundaries
        paragraphs = re.split(
            r"\n\s*\n",
            text
        )

        result = []

        current = ""

        for paragraph in paragraphs:

            paragraph = self.clean_text(
                paragraph
            )

            if not paragraph:
                continue

            candidate = (
                paragraph
                if not current
                else current
                + "\n\n"
                + paragraph
            )

            if (
                current
                and len(candidate)
                > self.max_chars
            ):

                result.append(
                    current
                )

                current = paragraph

            else:

                current = candidate

        if current:
            result.append(
                current
            )

        # Safety split for a single
        # extremely large paragraph
        final = []

        for part in result:

            if len(part) <= self.max_chars:

                final.append(
                    part
                )

                continue

            sentences = re.split(
                r"(?<=[.!?])\s+",
                part
            )

            current = ""

            for sentence in sentences:

                if not current:

                    current = sentence

                    continue

                candidate = (
                    current
                    + " "
                    + sentence
                )

                if (
                    len(candidate)
                    > self.max_chars
                ):

                    final.append(
                        current
                    )

                    current = sentence

                else:

                    current = candidate

            if current:
                final.append(
                    current
                )

        return final

    # ============================================================
    # TABLE ANALYSIS
    # ============================================================

    def extract_table_rows(
        self,
        table
    ):

        rows = table.get(
            "rows",
            []
        )

        result = []

        for row_index, row in enumerate(
            rows
        ):

            if not isinstance(
                row,
                list
            ):
                continue

            cells = []

            for cell in row:

                text = self.cell_text(
                    cell
                )

                if not text:
                    continue

                rowspan = cell.get(
                    "rowspan",
                    1
                )

                colspan = cell.get(
                    "colspan",
                    1
                )

                cells.append(
                    {
                        "text": text,
                        "rowspan": rowspan,
                        "colspan": colspan
                    }
                )

            if cells:

                result.append(
                    {
                        "row_index":
                            row_index,

                        "cells":
                            cells
                    }
                )

        return result

    # ============================================================
    # REAL TABLE DETECTION
    # ============================================================

    def looks_like_real_table(
        self,
        table
    ):

        rows = self.extract_table_rows(
            table
        )

        if len(rows) < 2:
            return False

        all_cells = [
            cell
            for row in rows
            for cell in row["cells"]
        ]

        if len(all_cells) < 2:
            return False

        all_text = " ".join(
            cell["text"]
            for cell in all_cells
        )

        # --------------------------------------------------------
        # Strong evidence of document prose
        # --------------------------------------------------------

        heading_hits = re.findall(
            r"\b\d+(?:\.\d+){0,4}\.?\s+[A-Z][A-Za-z]",
            all_text
        )

        long_cells = sum(
            1
            for cell in all_cells
            if len(cell["text"]) > 120
        )

        long_cell_ratio = (
            long_cells
            / max(
                1,
                len(all_cells)
            )
        )

        if (
            len(heading_hits) >= 2
            and long_cell_ratio >= 0.20
        ):
            return False

        # --------------------------------------------------------
        # Product Standard wrapper detection
        # --------------------------------------------------------

        upper_text = all_text.upper()

        product_standard = (
            "PRODUCT STANDARD"
            in upper_text
        )

        page_reference = bool(
            re.search(
                r"PAGE\s+\d+\s+OF\s+\d+",
                upper_text
            )
        )

        if (
            product_standard
            and page_reference
        ):
            return False

        # --------------------------------------------------------
        # Typical real table indicators
        # --------------------------------------------------------

        row_count = table.get(
            "row_count",
            len(rows)
        )

        column_count = table.get(
            "column_count",
            1
        )

        short_cells = sum(
            1
            for cell in all_cells
            if len(cell["text"]) <= 80
        )

        short_ratio = (
            short_cells
            / max(
                1,
                len(all_cells)
            )
        )

        # A conventional table
        if (
            row_count >= 2
            and column_count >= 2
            and short_ratio >= 0.35
        ):
            return True

        # Tables with multiple compact rows
        if (
            row_count >= 3
            and column_count >= 2
        ):
            return True

        return False

    # ============================================================
    # TABLE CHUNKING
    # ============================================================

    def process_real_table(
        self,
        page,
        block_index,
        block
    ):

        parts = block.get(
            "parts",
            []
        )

        for table_index, part in enumerate(
            parts
        ):

            if part.get(
                "type"
            ) != "table_body":

                continue

            table = part.get(
                "table"
            )

            if not isinstance(
                table,
                dict
            ):

                continue

            rows = self.extract_table_rows(
                table
            )

            if not rows:
                continue

            current_rows = []
            current_length = 0

            for row in rows:

                row_parts = []

                for cell in row["cells"]:

                    text = cell[
                        "text"
                    ]

                    rowspan = cell[
                        "rowspan"
                    ]

                    colspan = cell[
                        "colspan"
                    ]

                    if rowspan > 1:
                        text += (
                            f" [rowspan={rowspan}]"
                        )

                    if colspan > 1:
                        text += (
                            f" [colspan={colspan}]"
                        )

                    row_parts.append(
                        text
                    )

                row_text = " | ".join(
                    row_parts
                )

                if not row_text:
                    continue

                if (
                    current_rows
                    and
                    current_length
                    + len(row_text)
                    + 1
                    > self.max_chars
                ):

                    self.add_chunk(
                        page=page,
                        block_index=block_index,
                        content_type="table",
                        content="\n".join(
                            current_rows
                        ),
                        bbox=block.get(
                            "bbox"
                        ),
                        section=self.current_section,
                        subsection=self.current_subsection,
                        clause=self.current_clause,
                        metadata={
                            "source_block_type":
                                "table",

                            "classification":
                                "real_table",

                            "table_index":
                                table_index,

                            "table_chunk_index":
                                self.stats[
                                    "table_chunks"
                                ],

                            "row_count":
                                table.get(
                                    "row_count"
                                ),

                            "column_count":
                                table.get(
                                    "column_count"
                                ),

                            "table_image":
                                part.get(
                                    "image_path"
                                ),
                        }
                    )

                    self.stats[
                        "table_chunks"
                    ] += 1

                    current_rows = []
                    current_length = 0

                current_rows.append(
                    row_text
                )

                current_length += (
                    len(row_text)
                    + 1
                )

            if current_rows:

                self.add_chunk(
                    page=page,
                    block_index=block_index,
                    content_type="table",
                    content="\n".join(
                        current_rows
                    ),
                    bbox=block.get(
                        "bbox"
                    ),
                    section=self.current_section,
                    subsection=self.current_subsection,
                    clause=self.current_clause,
                    metadata={
                        "source_block_type":
                            "table",

                        "classification":
                            "real_table",

                        "table_index":
                            table_index,

                        "table_chunk_index":
                            self.stats[
                                "table_chunks"
                            ],

                        "row_count":
                            table.get(
                                "row_count"
                            ),

                        "column_count":
                            table.get(
                                "column_count"
                            ),

                        "table_image":
                            part.get(
                                "image_path"
                            ),
                    }
                )

                self.stats[
                    "table_chunks"
                ] += 1

    # ============================================================
    # FLATTEN LAYOUT WRAPPER
    # ============================================================

    def flatten_layout_wrapper(
        self,
        block
    ):

        parts = block.get(
            "parts",
            []
        )

        texts = []

        for part in parts:

            if part.get(
                "type"
            ) != "table_body":

                continue

            table = part.get(
                "table"
            )

            if not isinstance(
                table,
                dict
            ):
                continue

            rows = self.extract_table_rows(
                table
            )

            for row in rows:

                for cell in row[
                    "cells"
                ]:

                    text = cell[
                        "text"
                    ]

                    if not text:
                        continue

                    if self.is_page_label(
                        text
                    ):
                        continue

                    if self.is_layout_noise(
                        text
                    ):
                        continue

                    texts.append(
                        text
                    )

        return "\n".join(
            texts
        )

    # ============================================================
    # PARSE SEMANTIC TEXT
    # ============================================================

    def parse_semantic_text(
        self,
        text
    ):

        text = self.clean_text(
            text
        )

        if not text:
            return

        # --------------------------------------------------------
        # Remove repeated document header
        # --------------------------------------------------------

        if self.is_repeated_document_header(
            text
        ):

            self.stats[
                "ignored_headers"
            ] += 1

            # Do not immediately return because
            # the block can contain useful content
            text = re.sub(
                r"PRODUCT STANDARD",
                "",
                text,
                flags=re.IGNORECASE
            )

            text = re.sub(
                r"HYDERABAD",
                "",
                text,
                flags=re.IGNORECASE
            )

            text = re.sub(
                r"TC\s*65132",
                "",
                text,
                flags=re.IGNORECASE
            )

            text = re.sub(
                r"Rev\s*No\.?\s*\d+",
                "",
                text,
                flags=re.IGNORECASE
            )

            text = re.sub(
                r"Page\s+\d+\s+of\s+\d+",
                "",
                text,
                flags=re.IGNORECASE
            )

            text = self.clean_text(
                text
            )

        if not text:
            return

        # --------------------------------------------------------
        # Detect headings anywhere in the text
        # --------------------------------------------------------

        lines = re.split(
            r"\n+",
            text
        )

        semantic_buffer = []

        for line in lines:

            line = self.clean_text(
                line
            )

            if not line:
                continue

            heading = self.detect_numbered_heading(
                line
            )

            # ----------------------------------------------------
            # Numbered heading
            # ----------------------------------------------------

            if heading:

                if semantic_buffer:

                    self.emit_semantic_text(
                        "\n".join(
                            semantic_buffer
                        )
                    )

                    semantic_buffer = []

                self.update_hierarchy(
                    heading
                )

                self.add_chunk(
                    page=self.current_page,
                    block_index=self.current_block_index,
                    content_type="heading",
                    content=line,
                    bbox=self.current_bbox,
                    section=self.current_section,
                    subsection=self.current_subsection,
                    clause=self.current_clause,
                    metadata={
                        "source_block_type":
                            self.current_source_type,

                        "classification":
                            "semantic_heading",

                        "heading_number":
                            heading[
                                "number"
                            ],
                    }
                )

                continue

            # ----------------------------------------------------
            # Letter heading
            # ----------------------------------------------------

            letter_heading = (
                self.detect_letter_heading(
                    line
                )
            )

            if letter_heading:

                if semantic_buffer:

                    self.emit_semantic_text(
                        "\n".join(
                            semantic_buffer
                        )
                    )

                    semantic_buffer = []

                self.add_chunk(
                    page=self.current_page,
                    block_index=self.current_block_index,
                    content_type="heading",
                    content=line,
                    bbox=self.current_bbox,
                    section=self.current_section,
                    subsection=self.current_subsection,
                    clause=self.current_clause,
                    metadata={
                        "source_block_type":
                            self.current_source_type,

                        "classification":
                            "letter_heading",
                    }
                )

                continue

            # ----------------------------------------------------
            # Normal content
            # ----------------------------------------------------

            semantic_buffer.append(
                line
            )

        if semantic_buffer:

            self.emit_semantic_text(
                "\n".join(
                    semantic_buffer
                )
            )

    # ============================================================
    # EMIT SEMANTIC TEXT
    # ============================================================

    def emit_semantic_text(
        self,
        text
    ):

        parts = self.split_text(
            text
        )

        for part in parts:

            if not part:
                continue

            self.add_chunk(
                page=self.current_page,
                block_index=self.current_block_index,
                content_type="text",
                content=part,
                bbox=self.current_bbox,
                section=self.current_section,
                subsection=self.current_subsection,
                clause=self.current_clause,
                metadata={
                    "source_block_type":
                        self.current_source_type,

                    "classification":
                        "semantic_text",
                }
            )

            self.stats[
                "text_chunks"
            ] += 1

    # ============================================================
    # PROCESS LAYOUT WRAPPER
    # ============================================================

    def process_layout_wrapper(
        self,
        page,
        block_index,
        block
    ):

        self.stats[
            "layout_wrappers"
        ] += 1

        text = self.flatten_layout_wrapper(
            block
        )

        if not text:
            return

        self.current_page = page
        self.current_block_index = block_index
        self.current_bbox = block.get(
            "bbox"
        )
        self.current_source_type = (
            "table"
        )

        self.parse_semantic_text(
            text
        )

    # ============================================================
    # PROCESS TEXT BLOCK
    # ============================================================

    def process_text(
        self,
        page,
        block_index,
        block
    ):

        text = self.clean_text(
            block.get(
                "text",
                ""
            )
        )

        if not text:
            return

        block_type = block.get(
            "type",
            "text"
        )

        if block_type == "page_number":

            self.stats[
                "ignored_page_numbers"
            ] += 1

            return

        if block_type == "footer":

            self.stats[
                "ignored_footers"
            ] += 1

            return

        if block_type == "header":

            self.stats[
                "ignored_headers"
            ] += 1

            return

        self.stats[
            "semantic_text_blocks"
        ] += 1

        self.current_page = page
        self.current_block_index = (
            block_index
        )
        self.current_bbox = block.get(
            "bbox"
        )
        self.current_source_type = (
            block_type
        )

        # Explicit MinerU heading
        if block_type in (
            "paragraph_title",
            "doc_title"
        ):

            heading = self.detect_numbered_heading(
                text
            )

            if heading:

                self.update_hierarchy(
                    heading
                )

            else:

                # Non-numbered title
                self.current_section = (
                    text
                )

                self.current_subsection = None
                self.current_clause = None

                self.stats[
                    "headings_detected"
                ] += 1

            self.add_chunk(
                page=page,
                block_index=block_index,
                content_type="heading",
                content=text,
                bbox=block.get(
                    "bbox"
                ),
                section=self.current_section,
                subsection=self.current_subsection,
                clause=self.current_clause,
                metadata={
                    "source_block_type":
                        block_type,

                    "classification":
                        "explicit_heading",
                }
            )

            return

        self.parse_semantic_text(
            text
        )

    # ============================================================
    # IMAGE
    # ============================================================

    def process_image(
        self,
        page,
        block_index,
        block
    ):

        image_path = (
            block.get(
                "image_path"
            )
            or block.get(
                "path"
            )
            or block.get(
                "src"
            )
            or "image_reference"
        )

        self.add_chunk(
            page=page,
            block_index=block_index,
            content_type="image",
            content=(
                f"Image reference: "
                f"{image_path}"
            ),
            bbox=block.get(
                "bbox"
            ),
            section=self.current_section,
            subsection=self.current_subsection,
            clause=self.current_clause,
            metadata={
                "source_block_type":
                    "image",

                "classification":
                    "image_reference",

                "image_path":
                    image_path,
            }
        )

        self.stats[
            "image_chunks"
        ] += 1

    # ============================================================
    # PROCESS TABLE BLOCK
    # ============================================================

    def process_table(
        self,
        page,
        block_index,
        block
    ):

        parts = block.get(
            "parts",
            []
        )

        for part in parts:

            if part.get(
                "type"
            ) != "table_body":

                continue

            table = part.get(
                "table"
            )

            if not isinstance(
                table,
                dict
            ):
                continue

            if self.looks_like_real_table(
                table
            ):

                self.stats[
                    "real_tables"
                ] += 1

                self.process_real_table(
                    page,
                    block_index,
                    block
                )

            else:

                self.process_layout_wrapper(
                    page,
                    block_index,
                    block
                )

    # ============================================================
    # OTHER BLOCK TYPES
    # ============================================================

    def process_other(
        self,
        page,
        block_index,
        block
    ):

        block_type = block.get(
            "type",
            "unknown"
        )

        if block_type == "page_number":

            self.stats[
                "ignored_page_numbers"
            ] += 1

            return

        if block_type == "footer":

            self.stats[
                "ignored_footers"
            ] += 1

            return

        if block_type == "header":

            self.stats[
                "ignored_headers"
            ] += 1

            return

        text = self.clean_text(
            block.get(
                "text",
                ""
            )
        )

        if not text:
            return

        if self.is_layout_noise(
            text
        ):

            self.stats[
                "ignored_layout_noise"
            ] += 1

            return

        self.current_page = page
        self.current_block_index = (
            block_index
        )
        self.current_bbox = block.get(
            "bbox"
        )
        self.current_source_type = (
            block_type
        )

        self.parse_semantic_text(
            text
        )

    # ============================================================
    # PROCESS PAGE
    # ============================================================

    def process_page(
        self,
        page
    ):

        page_number = page.get(
            "page"
        )

        self.stats[
            "pages_processed"
        ] += 1

        blocks = page.get(
            "blocks",
            []
        )

        for block_index, block in enumerate(
            blocks
        ):

            self.stats[
                "source_blocks_processed"
            ] += 1

            block_type = block.get(
                "type",
                "unknown"
            )

            if block_type in (
                "text",
                "paragraph_title",
                "doc_title",
                "header",
                "footer",
                "page_number"
            ):

                self.process_text(
                    page_number,
                    block_index,
                    block
                )

            elif block_type == "table":

                self.process_table(
                    page_number,
                    block_index,
                    block
                )

            elif block_type == "image":

                self.process_image(
                    page_number,
                    block_index,
                    block
                )

            else:

                self.process_other(
                    page_number,
                    block_index,
                    block
                )

    # ============================================================
    # BUILD
    # ============================================================

    def build(self):

        document = self.document.get(
            "document",
            {}
        )

        pages = document.get(
            "pages",
            []
        )

        for page in pages:

            self.process_page(
                page
            )

    # ============================================================
    # VALIDATION
    # ============================================================

    def validate_chunks(self):

        errors = []

        for index, chunk in enumerate(
            self.chunks
        ):

            required_fields = [
                "chunk_id",
                "document_page",
                "block_index",
                "content_type",
                "hierarchy",
                "content",
            ]

            for field in required_fields:

                if field not in chunk:

                    errors.append(
                        f"Chunk {index}: "
                        f"missing {field}"
                    )

            if not chunk.get(
                "content"
            ):

                errors.append(
                    f"Chunk {index}: "
                    f"empty content"
                )

            if not chunk.get(
                "document_page"
            ):

                errors.append(
                    f"Chunk {index}: "
                    f"missing page"
                )

        return errors

    # ============================================================
    # SEMANTIC VALIDATION
    # ============================================================

    def semantic_validation(self):

        problems = []

        # --------------------------------------------------------
        # Check Product Standard hierarchy
        # --------------------------------------------------------

        product_standard_chunks = [
            chunk
            for chunk in self.chunks
            if (
                chunk[
                    "document_page"
                ] >= 4
                and
                (
                    "PRODUCT STANDARD"
                    in chunk[
                        "content"
                    ].upper()
                    or
                    (
                        chunk[
                            "hierarchy"
                        ].get(
                            "section"
                        )
                        and
                        (
                            "GENERAL"
                            in chunk[
                                "hierarchy"
                            ][
                                "section"
                            ].upper()
                        )
                    )
                )
            )
        ]

        # --------------------------------------------------------
        # Detect SCC carry-over
        # --------------------------------------------------------

        for chunk in self.chunks:

            page = chunk[
                "document_page"
            ]

            hierarchy = chunk[
                "hierarchy"
            ]

            section = hierarchy.get(
                "section"
            )

            if (
                page >= 4
                and
                section
                and
                "SPECIAL CONTRACT CONDITIONS"
                in section.upper()
            ):

                content = chunk[
                    "content"
                ].upper()

                # Ignore explicit SCC references
                # in actual SCC-like text.
                if (
                    "PRODUCT STANDARD"
                    in content
                    or
                    re.search(
                        r"\b\d+\.\d+\.",
                        content
                    )
                ):

                    problems.append(
                        (
                            "Possible SCC "
                            "carry-over: "
                            f"page={page}, "
                            f"chunk={chunk['chunk_id']}"
                        )
                    )

        return problems

    # ============================================================
    # SAVE
    # ============================================================

    def save(self):

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        structural_errors = (
            self.validate_chunks()
        )

        semantic_problems = (
            self.semantic_validation()
        )

        status = (
            "passed"
            if not structural_errors
            else "failed"
        )

        output = {

            "schema": {
                "name":
                    "pdf-ingestion-chunks",

                "version":
                    "3.0"
            },

            "source":
                self.document.get(
                    "source",
                    {}
                ),

            "extraction":
                self.document.get(
                    "extraction",
                    {}
                ),

            "chunking": {

                "strategy":
                    "semantic-layout-aware-v3",

                "max_chars":
                    self.max_chars,

                "chunk_count":
                    len(
                        self.chunks
                    ),

                "statistics":
                    self.stats,

                "validation": {

                    "status":
                        status,

                    "structural_errors":
                        structural_errors,

                    "semantic_warnings":
                        semantic_problems,
                }
            },

            "chunks":
                self.chunks,
        }

        with open(
            self.output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                output,
                file,
                indent=2,
                ensure_ascii=False
            )

    # ============================================================
    # RUN
    # ============================================================

    def run(self):

        print("=" * 80)
        print(
            "PHASE 3 — SEMANTIC "
            "LAYOUT-AWARE CHUNKING V3"
        )
        print("=" * 80)

        print()

        print(
            f"Input      : "
            f"{self.input_path}"
        )

        print(
            f"Output     : "
            f"{self.output_path}"
        )

        print(
            f"Max chars  : "
            f"{self.max_chars}"
        )

        print()

        self.load()

        print(
            "Building semantic chunks..."
        )

        self.build()

        print()
        print("-" * 80)
        print("CHUNKING STATISTICS")
        print("-" * 80)

        print(
            f"Pages processed       : "
            f"{self.stats['pages_processed']}"
        )

        print(
            f"Source blocks         : "
            f"{self.stats['source_blocks_processed']}"
        )

        print(
            f"Real tables           : "
            f"{self.stats['real_tables']}"
        )

        print(
            f"Layout wrappers       : "
            f"{self.stats['layout_wrappers']}"
        )

        print(
            f"Semantic text blocks  : "
            f"{self.stats['semantic_text_blocks']}"
        )

        print(
            f"Headers ignored       : "
            f"{self.stats['ignored_headers']}"
        )

        print(
            f"Footers ignored       : "
            f"{self.stats['ignored_footers']}"
        )

        print(
            f"Page numbers ignored  : "
            f"{self.stats['ignored_page_numbers']}"
        )

        print(
            f"Layout noise ignored  : "
            f"{self.stats['ignored_layout_noise']}"
        )

        print(
            f"Headings detected     : "
            f"{self.stats['headings_detected']}"
        )

        print(
            f"Sections detected     : "
            f"{self.stats['sections_detected']}"
        )

        print(
            f"Subsections detected  : "
            f"{self.stats['subsections_detected']}"
        )

        print(
            f"Clauses detected      : "
            f"{self.stats['clauses_detected']}"
        )

        print(
            f"Table chunks          : "
            f"{self.stats['table_chunks']}"
        )

        print(
            f"Text chunks           : "
            f"{self.stats['text_chunks']}"
        )

        print(
            f"Image chunks          : "
            f"{self.stats['image_chunks']}"
        )

        print(
            f"Total chunks          : "
            f"{len(self.chunks)}"
        )

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------

        structural_errors = (
            self.validate_chunks()
        )

        semantic_warnings = (
            self.semantic_validation()
        )

        print()
        print("-" * 80)
        print("VALIDATION")
        print("-" * 80)

        if structural_errors:

            print(
                f"Structural errors     : "
                f"{len(structural_errors)}"
            )

            for error in structural_errors[
                :20
            ]:

                print(
                    f"  [ERROR] {error}"
                )

        else:

            print(
                "Structural validation : PASSED"
            )

        if semantic_warnings:

            print(
                f"Semantic warnings     : "
                f"{len(semantic_warnings)}"
            )

            for warning in semantic_warnings[
                :20
            ]:

                print(
                    f"  [WARNING] {warning}"
                )

        else:

            print(
                "Semantic validation   : PASSED"
            )

        self.save()

        print()
        print("=" * 80)
        print(
            "CHUNKING V3 COMPLETED"
        )
        print("=" * 80)

        print()
        print(
            f"Output: "
            f"{self.output_path}"
        )


# ==================================================================
# MAIN
# ==================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Semantic layout-aware "
            "chunking V3 for MinerU "
            "normalized documents."
        )
    )

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=1800
    )

    args = parser.parse_args()

    chunker = StructureAwareChunkerV3(
        input_path=args.input,
        output_path=args.output,
        max_chars=args.max_chars
    )

    chunker.run()


if __name__ == "__main__":
    main()