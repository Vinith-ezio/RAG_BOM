import json
import math
from collections import Counter
from pathlib import Path

import pymupdf


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = Path(
    r"E:\PDF Ingestion\data\input\BHEL Spec..pdf"
)

OUTPUT_DIR = Path(
    r"E:\PDF Ingestion\data\analysis\page_investigation"
)

SUMMARY_PATH = OUTPUT_DIR / "all_pages_investigation_summary.json"


# ============================================================
# HELPERS
# ============================================================

def round_value(value):
    return round(float(value), 3)


def rect_to_dict(rect):
    return {
        "x0": round_value(rect.x0),
        "y0": round_value(rect.y0),
        "x1": round_value(rect.x1),
        "y1": round_value(rect.y1),
        "width": round_value(rect.width),
        "height": round_value(rect.height),
    }


def point_to_dict(point):
    return {
        "x": round_value(point.x),
        "y": round_value(point.y),
    }


def distance(p1, p2):
    return math.sqrt(
        (p2.x - p1.x) ** 2
        + (p2.y - p1.y) ** 2
    )


# ============================================================
# PAGE GEOMETRY
# ============================================================

def analyze_page_geometry(page):

    rect = page.rect

    return {
        "width": round_value(rect.width),
        "height": round_value(rect.height),
        "area": round_value(
            rect.width * rect.height
        ),
        "orientation": (
            "landscape"
            if rect.width > rect.height
            else "portrait"
        ),
    }


# ============================================================
# TEXT ANALYSIS
# ============================================================

def analyze_text(page):

    raw_text = page.get_text("text")

    words = page.get_text("words")

    blocks = page.get_text("blocks")

    raw_dict = page.get_text("dict")

    block_results = []

    for block_index, block in enumerate(blocks):

        if len(block) < 5:
            continue

        x0, y0, x1, y1, block_text = block[:5]

        block_results.append(
            {
                "block_index": block_index,
                "bbox": {
                    "x0": round_value(x0),
                    "y0": round_value(y0),
                    "x1": round_value(x1),
                    "y1": round_value(y1),
                    "width": round_value(x1 - x0),
                    "height": round_value(y1 - y0),
                },
                "text": block_text,
                "characters": len(block_text),
            }
        )

    spans = []
    line_count = 0
    font_counter = Counter()

    for block_index, block in enumerate(
        raw_dict.get("blocks", [])
    ):

        if block.get("type") != 0:
            continue

        for line_index, line in enumerate(
            block.get("lines", [])
        ):

            line_count += 1

            for span_index, span in enumerate(
                line.get("spans", [])
            ):

                bbox = span.get("bbox")

                font = span.get("font")

                if font:
                    font_counter[font] += 1

                spans.append(
                    {
                        "block_index": block_index,
                        "line_index": line_index,
                        "span_index": span_index,
                        "text": span.get(
                            "text",
                            ""
                        ),
                        "font": font,
                        "size": span.get("size"),
                        "flags": span.get(
                            "flags"
                        ),
                        "color": span.get(
                            "color"
                        ),
                        "bbox": (
                            {
                                "x0": round_value(
                                    bbox[0]
                                ),
                                "y0": round_value(
                                    bbox[1]
                                ),
                                "x1": round_value(
                                    bbox[2]
                                ),
                                "y1": round_value(
                                    bbox[3]
                                ),
                                "width": round_value(
                                    bbox[2] - bbox[0]
                                ),
                                "height": round_value(
                                    bbox[3] - bbox[1]
                                ),
                            }
                            if bbox
                            else None
                        ),
                    }
                )

    return {
        "characters": len(raw_text),
        "words": len(words),
        "block_count": len(
            block_results
        ),
        "line_count": line_count,
        "span_count": len(spans),
        "raw_text_sample": raw_text[
            :1000
        ],
        "blocks": block_results,
        "spans": spans,
        "fonts": dict(font_counter),
    }


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_images(page):

    images = page.get_images(
        full=True
    )

    results = []

    for index, image in enumerate(images):

        xref = image[0]
        smask = image[1]
        width = image[2]
        height = image[3]
        colorspace = image[5]
        name = image[7]

        results.append(
            {
                "index": index,
                "xref": xref,
                "smask": smask,
                "width": width,
                "height": height,
                "colorspace": colorspace,
                "name": name,
            }
        )

    return results


# ============================================================
# DRAWING ANALYSIS
# ============================================================

def analyze_drawings(page):

    drawings = page.get_drawings()

    drawing_type_counter = Counter()

    horizontal_lines = []
    vertical_lines = []
    diagonal_lines = []

    rectangles = []
    curves = []

    # Important:
    # We keep ALL drawing objects for evidence,
    # but separately calculate meaningful geometry.

    meaningful_horizontal = []
    meaningful_vertical = []

    for drawing_index, drawing in enumerate(
        drawings
    ):

        items = drawing.get(
            "items",
            []
        )

        for item_index, item in enumerate(
            items
        ):

            if not item:
                continue

            operation = item[0]

            drawing_type_counter[
                operation
            ] += 1

            # ------------------------------------------------
            # LINE
            # ------------------------------------------------

            if operation == "l":

                if len(item) < 3:
                    continue

                p1 = item[1]
                p2 = item[2]

                line_length = distance(
                    p1,
                    p2
                )

                dx = abs(
                    p2.x - p1.x
                )

                dy = abs(
                    p2.y - p1.y
                )

                # More reliable orientation:
                if dx >= dy:
                    orientation = (
                        "horizontal"
                    )
                else:
                    orientation = (
                        "vertical"
                    )

                record = {
                    "drawing_index":
                        drawing_index,
                    "item_index":
                        item_index,
                    "start":
                        point_to_dict(p1),
                    "end":
                        point_to_dict(p2),
                    "length":
                        round_value(
                            line_length
                        ),
                    "orientation":
                        orientation,
                }

                if orientation == "horizontal":

                    horizontal_lines.append(
                        record
                    )

                    if line_length >= 20:
                        meaningful_horizontal.append(
                            record
                        )

                elif orientation == "vertical":

                    vertical_lines.append(
                        record
                    )

                    if line_length >= 20:
                        meaningful_vertical.append(
                            record
                        )

                else:

                    diagonal_lines.append(
                        record
                    )

            # ------------------------------------------------
            # RECTANGLE
            # ------------------------------------------------

            elif operation == "re":

                if len(item) >= 2:

                    rect = item[1]

                    record = {
                        "drawing_index":
                            drawing_index,
                        "item_index":
                            item_index,
                        "bbox":
                            rect_to_dict(
                                rect
                            ),
                    }

                    rectangles.append(
                        record
                    )

            # ------------------------------------------------
            # CURVE
            # ------------------------------------------------

            elif operation == "c":

                curves.append(
                    {
                        "drawing_index":
                            drawing_index,
                        "item_index":
                            item_index,
                    }
                )

    return {
        "drawing_count":
            len(drawings),

        "drawing_type_summary":
            dict(drawing_type_counter),

        "horizontal_line_count":
            len(horizontal_lines),

        "vertical_line_count":
            len(vertical_lines),

        "diagonal_line_count":
            len(diagonal_lines),

        "rectangle_count":
            len(rectangles),

        "curve_count":
            len(curves),

        "meaningful_horizontal_line_count":
            len(meaningful_horizontal),

        "meaningful_vertical_line_count":
            len(meaningful_vertical),

        "horizontal_lines":
            horizontal_lines,

        "vertical_lines":
            vertical_lines,

        "meaningful_horizontal_lines":
            meaningful_horizontal,

        "meaningful_vertical_lines":
            meaningful_vertical,

        "rectangles":
            rectangles,

        "curves":
            curves,
    }


# ============================================================
# TEXT / RECTANGLE RELATIONSHIP
# ============================================================

def find_text_inside_rectangles(
    spans,
    rectangles
):

    matches = []

    for span in spans:

        bbox = span.get(
            "bbox"
        )

        text = span.get(
            "text",
            ""
        ).strip()

        if not bbox or not text:
            continue

        sx0 = bbox["x0"]
        sy0 = bbox["y0"]
        sx1 = bbox["x1"]
        sy1 = bbox["y1"]

        for rectangle_index, rectangle in enumerate(
            rectangles
        ):

            rbbox = rectangle[
                "bbox"
            ]

            rx0 = rbbox["x0"]
            ry0 = rbbox["y0"]
            rx1 = rbbox["x1"]
            ry1 = rbbox["y1"]

            inside = (
                sx0 >= rx0
                and sy0 >= ry0
                and sx1 <= rx1
                and sy1 <= ry1
            )

            if inside:

                matches.append(
                    {
                        "text":
                            text,
                        "rectangle_index":
                            rectangle_index,
                        "text_bbox":
                            bbox,
                        "rectangle_bbox":
                            rbbox,
                    }
                )

    return matches


# ============================================================
# TABLE / GRID EVIDENCE
# ============================================================

def analyze_grid_evidence(
    vectors,
    text_rectangle_matches
):

    meaningful_horizontal = vectors[
        "meaningful_horizontal_line_count"
    ]

    meaningful_vertical = vectors[
        "meaningful_vertical_line_count"
    ]

    rectangle_count = vectors[
        "rectangle_count"
    ]

    evidence = []

    if meaningful_horizontal >= 3:

        evidence.append(
            "multiple_meaningful_horizontal_lines"
        )

    if meaningful_vertical >= 3:

        evidence.append(
            "multiple_meaningful_vertical_lines"
        )

    if rectangle_count >= 3:

        evidence.append(
            "multiple_rectangles"
        )

    if text_rectangle_matches:

        evidence.append(
            "text_inside_vector_rectangles"
        )

    possible_grid = (
        meaningful_horizontal >= 3
        and meaningful_vertical >= 3
    )

    strong_grid_evidence = (
        possible_grid
        and (
            rectangle_count >= 3
            or len(
                text_rectangle_matches
            ) >= 3
        )
    )

    return {
        "possible_grid":
            possible_grid,

        "strong_grid_evidence":
            strong_grid_evidence,

        "evidence":
            evidence,
    }


# ============================================================
# PAGE INVESTIGATION
# ============================================================

def investigate_page(
    document,
    page_number
):

    page = document[
        page_number - 1
    ]

    geometry = (
        analyze_page_geometry(
            page
        )
    )

    text = analyze_text(
        page
    )

    images = analyze_images(
        page
    )

    vectors = analyze_drawings(
        page
    )

    text_rectangle_matches = (
        find_text_inside_rectangles(
            text["spans"],
            vectors["rectangles"]
        )
    )

    grid = analyze_grid_evidence(
        vectors,
        text_rectangle_matches
    )

    return {
        "page_number":
            page_number,

        "geometry":
            geometry,

        "text":
            text,

        "images": {
            "count":
                len(images),
            "objects":
                images,
        },

        "vectors":
            vectors,

        "relationships": {
            "text_inside_vector_rectangles":
                {
                    "count":
                        len(
                            text_rectangle_matches
                        ),
                    "matches":
                        text_rectangle_matches,
                }
        },

        "semantic_evidence": {
            "grid":
                grid
        }
    }


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    page_results,
    total_pages
):

    page_types = Counter()

    pages_with_images = []
    image_only_pages = []

    high_vector_pages = []
    extreme_vector_pages = []

    possible_grid_pages = []
    strong_grid_pages = []

    low_text_pages = []

    total_images = 0
    total_drawings = 0

    for result in page_results:

        page_number = result[
            "page_number"
        ]

        text = result[
            "text"
        ]

        image_count = result[
            "images"
        ]["count"]

        drawing_count = result[
            "vectors"
        ]["drawing_count"]

        grid = result[
            "semantic_evidence"
        ]["grid"]

        # ----------------------------------------------------
        # Physical page type
        # ----------------------------------------------------

        if (
            text["characters"] == 0
            and image_count == 0
        ):

            page_type = "EMPTY"

        elif (
            text["characters"] == 0
            and image_count > 0
        ):

            page_type = "IMAGE_ONLY"

            image_only_pages.append(
                page_number
            )

        elif (
            text["characters"] > 0
            and image_count == 0
        ):

            page_type = "TEXT_ONLY"

        else:

            page_type = (
                "MIXED_TEXT_IMAGE"
            )

        page_types[
            page_type
        ] += 1

        # ----------------------------------------------------
        # Images
        # ----------------------------------------------------

        if image_count > 0:

            pages_with_images.append(
                page_number
            )

        total_images += image_count

        # ----------------------------------------------------
        # Vectors
        # ----------------------------------------------------

        total_drawings += drawing_count

        if drawing_count >= 500:

            high_vector_pages.append(
                page_number
            )

        if drawing_count >= 2000:

            extreme_vector_pages.append(
                page_number
            )

        # ----------------------------------------------------
        # Grid evidence
        # ----------------------------------------------------

        if grid[
            "possible_grid"
        ]:

            possible_grid_pages.append(
                page_number
            )

        if grid[
            "strong_grid_evidence"
        ]:

            strong_grid_pages.append(
                page_number
            )

        # ----------------------------------------------------
        # Low text
        # ----------------------------------------------------

        if text[
            "characters"
        ] < 1000:

            low_text_pages.append(
                page_number
            )

    return {
        "total_pages":
            total_pages,

        "investigated_pages":
            len(page_results),

        "page_types":
            dict(page_types),

        "pages_with_images":
            pages_with_images,

        "image_only_pages":
            image_only_pages,

        "high_vector_pages":
            high_vector_pages,

        "extreme_vector_pages":
            extreme_vector_pages,

        "possible_grid_pages":
            possible_grid_pages,

        "strong_grid_pages":
            strong_grid_pages,

        "low_text_pages":
            low_text_pages,

        "total_image_objects":
            total_images,

        "total_drawing_objects":
            total_drawings,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print(
        "PDF INVESTIGATION - ALL 114 PAGES"
    )
    print("=" * 70)

    print(
        f"PDF: {PDF_PATH}"
    )

    document = pymupdf.open(
        PDF_PATH
    )

    total_pages = len(
        document
    )

    print(
        f"Pages found: {total_pages}"
    )

    print()

    page_results = []

    for page_number in range(
        1,
        total_pages + 1
    ):

        print(
            f"[{page_number:03d}/{total_pages:03d}] "
            "Investigating...",
            end=""
        )

        try:

            result = investigate_page(
                document,
                page_number
            )

            page_results.append(
                result
            )

            output_path = (
                OUTPUT_DIR
                / f"page_{page_number:03d}_investigation.json"
            )

            with open(
                output_path,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    result,
                    file,
                    indent=2,
                    ensure_ascii=False
                )

            text_chars = result[
                "text"
            ]["characters"]

            image_count = result[
                "images"
            ]["count"]

            drawing_count = result[
                "vectors"
            ]["drawing_count"]

            grid = result[
                "semantic_evidence"
            ]["grid"]

            print(
                f" PASS | "
                f"text={text_chars} | "
                f"images={image_count} | "
                f"drawings={drawing_count} | "
                f"grid={grid['possible_grid']}"
            )

        except Exception as exc:

            print(
                f" ERROR: {exc}"
            )

    document.close()

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = build_summary(
        page_results,
        total_pages
    )

    summary_document = {
        "investigator": {
            "name":
                "PDF Page Investigator",
            "version":
                "2.0",
            "engine":
                "PyMuPDF",
        },

        "source": {
            "pdf":
                str(PDF_PATH),
        },

        "summary":
            summary,
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary_document,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 70)
    print(
        "INVESTIGATION SUMMARY"
    )
    print("=" * 70)

    print(
        f"Investigated pages : "
        f"{summary['investigated_pages']}"
    )

    print(
        f"Total pages        : "
        f"{summary['total_pages']}"
    )

    print(
        f"Total images       : "
        f"{summary['total_image_objects']}"
    )

    print(
        f"Total drawings     : "
        f"{summary['total_drawing_objects']}"
    )

    print()
    print(
        "Page types:"
    )

    for page_type, count in (
        summary["page_types"]
        .items()
    ):

        print(
            f"  {page_type:<20} "
            f"{count}"
        )

    print()
    print(
        f"High vector pages  : "
        f"{len(summary['high_vector_pages'])}"
    )

    print(
        f"Extreme vector     : "
        f"{len(summary['extreme_vector_pages'])}"
    )

    print(
        f"Possible grids     : "
        f"{len(summary['possible_grid_pages'])}"
    )

    print(
        f"Strong grid pages  : "
        f"{len(summary['strong_grid_pages'])}"
    )

    print()
    print(
        f"Summary saved to:"
    )

    print(
        SUMMARY_PATH
    )


if __name__ == "__main__":
    main()