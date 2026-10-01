import json
import hashlib
import statistics
from collections import Counter
from pathlib import Path

import pymupdf


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = Path(
    r"E:\PDF Ingestion\data\input\BHEL Spec..pdf"
)

OUTPUT_PATH = Path(
    r"E:\PDF Ingestion\data\analysis\phase_1\bhel_spec_phase1.json"
)

ANALYZER_VERSION = "0.2.0"

# Phase 1 is an observation layer.
# These values are used only for basic page-state classification
# stored as derived information; Phase 2 should own routing policy.
MIN_TEXT_CHARS = 1

# Long line threshold used only for aggregate diagnostics.
MEANINGFUL_LINE_MIN_LENGTH = 20.0

# Font/image/vector diagnostic limits.
MAX_TEXT_SAMPLE_CHARS = 500


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_mean(values):
    return round(statistics.mean(values), 4) if values else 0.0


def safe_median(values):
    return round(statistics.median(values), 4) if values else 0.0


def safe_min(values):
    return min(values) if values else 0


def safe_max(values):
    return max(values) if values else 0


def count_words(text):
    return len(text.split()) if text else 0


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bbox_to_list(rect):
    return [
        round(float(rect.x0), 2),
        round(float(rect.y0), 2),
        round(float(rect.x1), 2),
        round(float(rect.y1), 2),
    ]


def rect_area(rect):
    return max(0.0, float(rect.width)) * max(0.0, float(rect.height))


def normalize_text(text):
    return text.strip() if text else ""


# ============================================================
# PAGE GEOMETRY
# ============================================================

def analyze_page_dimensions(page):
    rect = page.rect

    width = float(rect.width)
    height = float(rect.height)

    if width > height:
        orientation = "landscape"
    elif height > width:
        orientation = "portrait"
    else:
        orientation = "square"

    return {
        "width": round(width, 2),
        "height": round(height, 2),
        "rotation": int(page.rotation),
        "orientation": orientation,
        "page_rect": bbox_to_list(rect),
        "page_area": round(width * height, 2),
    }


# ============================================================
# TEXT ANALYSIS
# ============================================================

def analyze_text(page):
    text = page.get_text("text")
    stripped_text = normalize_text(text)

    lines = [
        line.strip()
        for line in stripped_text.splitlines()
        if line.strip()
    ]

    return {
        "characters": len(stripped_text),
        "words": count_words(stripped_text),
        "lines": len(lines),
        "has_text": bool(stripped_text),
        "text_sample": (
            stripped_text[:MAX_TEXT_SAMPLE_CHARS]
            if stripped_text
            else ""
        ),
    }


# ============================================================
# TEXT BLOCK ANALYSIS
# ============================================================

def analyze_text_blocks(page):
    blocks = page.get_text("blocks")

    text_blocks = []

    for block in blocks:
        if len(block) < 5:
            continue

        x0, y0, x1, y1, text = block[:5]
        text = normalize_text(text)

        if not text:
            continue

        text_blocks.append(
            {
                "bbox": [
                    round(float(x0), 2),
                    round(float(y0), 2),
                    round(float(x1), 2),
                    round(float(y1), 2),
                ],
                "text": text,
                "characters": len(text),
                "words": count_words(text),
                "area": round(
                    max(0.0, float(x1 - x0))
                    * max(0.0, float(y1 - y0)),
                    2,
                ),
            }
        )

    block_character_count = sum(
        block["characters"] for block in text_blocks
    )
    block_word_count = sum(
        block["words"] for block in text_blocks
    )

    return {
        "count": len(text_blocks),
        "characters": block_character_count,
        "words": block_word_count,
        "records": text_blocks,
    }


# ============================================================
# SPAN / FONT ANALYSIS
# ============================================================

def analyze_text_spans(page):
    raw_dict = page.get_text("dict")

    spans = []
    font_counter = Counter()

    for block_index, block in enumerate(
        raw_dict.get("blocks", [])
    ):
        if block.get("type") != 0:
            continue

        for line_index, line in enumerate(
            block.get("lines", [])
        ):
            for span_index, span in enumerate(
                line.get("spans", [])
            ):
                text = normalize_text(span.get("text", ""))

                if not text:
                    continue

                font = span.get("font", "UNKNOWN")

                font_counter[font] += 1

                bbox = span.get("bbox")
                spans.append(
                    {
                        "block_index": block_index,
                        "line_index": line_index,
                        "span_index": span_index,
                        "text": text,
                        "characters": len(text),
                        "words": count_words(text),
                        "font": font,
                        "size": round(
                            float(span.get("size", 0.0)),
                            3,
                        ),
                        "flags": int(span.get("flags", 0)),
                        "color": span.get("color"),
                        "bbox": (
                            [
                                round(float(v), 2)
                                for v in bbox
                            ]
                            if bbox
                            else None
                        ),
                    }
                )

    return {
        "count": len(spans),
        "font_unique_count": len(font_counter),
        "font_usage_by_span": dict(font_counter),
        "records": spans,
    }


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_images(page):
    images = page.get_images(full=True)

    records = []
    unique_xrefs = set()

    # page.get_image_rects() gives actual placement rectangles
    # for an image xref on this page.
    for image_index, image in enumerate(images):
        xref = int(image[0])
        smask = int(image[1])
        width = int(image[2])
        height = int(image[3])
        colorspace = image[5]
        name = image[7]

        unique_xrefs.add(xref)

        try:
            placement_rects = page.get_image_rects(xref)
        except Exception:
            placement_rects = []

        placements = []
        for rect in placement_rects:
            placements.append(
                {
                    "bbox": bbox_to_list(rect),
                    "area": round(rect_area(rect), 2),
                }
            )

        records.append(
            {
                "reference_index": image_index,
                "xref": xref,
                "smask": smask,
                "width_px": width,
                "height_px": height,
                "colorspace": colorspace,
                "name": name,
                "placement_count": len(placements),
                "placements": placements,
            }
        )

    return {
        "reference_count": len(records),
        "unique_xref_count": len(unique_xrefs),
        "records": records,
    }


# ============================================================
# VECTOR / DRAWING ANALYSIS
# ============================================================

def analyze_drawings(page):
    drawings = page.get_drawings()

    type_counter = Counter()

    line_count = 0
    rectangle_count = 0
    curve_count = 0
    total_items = 0
    filled_drawing_count = 0
    stroked_drawing_count = 0

    all_bboxes = []
    meaningful_horizontal_lines = []
    meaningful_vertical_lines = []

    for drawing in drawings:
        rect = drawing.get("rect")
        if rect is not None:
            all_bboxes.append(bbox_to_list(rect))

        items = drawing.get("items", [])
        total_items += len(items)

        if drawing.get("fill") is not None:
            filled_drawing_count += 1

        if drawing.get("color") is not None:
            stroked_drawing_count += 1

        for item in items:
            kind = item[0] if item else "unknown"
            type_counter[kind] += 1

            if kind == "l":
                line_count += 1

                if len(item) >= 3:
                    p1 = item[1]
                    p2 = item[2]

                    dx = abs(float(p2.x) - float(p1.x))
                    dy = abs(float(p2.y) - float(p1.y))

                    if dx >= MEANINGFUL_LINE_MIN_LENGTH and dy <= 1.0:
                        meaningful_horizontal_lines.append(
                            {
                                "p1": [
                                    round(float(p1.x), 2),
                                    round(float(p1.y), 2),
                                ],
                                "p2": [
                                    round(float(p2.x), 2),
                                    round(float(p2.y), 2),
                                ],
                                "length": round(dx, 2),
                            }
                        )

                    elif dy >= MEANINGFUL_LINE_MIN_LENGTH and dx <= 1.0:
                        meaningful_vertical_lines.append(
                            {
                                "p1": [
                                    round(float(p1.x), 2),
                                    round(float(p1.y), 2),
                                ],
                                "p2": [
                                    round(float(p2.x), 2),
                                    round(float(p2.y), 2),
                                ],
                                "length": round(dy, 2),
                            }
                        )

            elif kind == "re":
                rectangle_count += 1

            elif kind in {"c", "v", "y"}:
                curve_count += 1

    return {
        "drawing_count": len(drawings),
        "drawing_item_count": total_items,
        "drawing_type_summary": dict(type_counter),
        "line_count": line_count,
        "rectangle_count": rectangle_count,
        "curve_count": curve_count,
        "filled_drawing_count": filled_drawing_count,
        "stroked_drawing_count": stroked_drawing_count,
        "meaningful_horizontal_line_count": len(
            meaningful_horizontal_lines
        ),
        "meaningful_vertical_line_count": len(
            meaningful_vertical_lines
        ),
        "meaningful_horizontal_lines": meaningful_horizontal_lines,
        "meaningful_vertical_lines": meaningful_vertical_lines,
        "drawing_bboxes": all_bboxes,
    }


# ============================================================
# DERIVED PAGE STATE
# ============================================================

def derive_page_state(text, images):
    has_text = text["has_text"]
    image_count = images["reference_count"]

    if not has_text and image_count == 0:
        page_state = "EMPTY"
    elif not has_text and image_count > 0:
        page_state = "IMAGE_ONLY"
    elif has_text and image_count > 0:
        page_state = "MIXED_TEXT_IMAGE"
    else:
        page_state = "TEXT_ONLY"

    return {
        "page_state": page_state,
        "has_native_text": has_text,
        "has_image_references": image_count > 0,
    }


# ============================================================
# PAGE ANALYSIS
# ============================================================

def analyze_page(page, page_number):
    dimensions = analyze_page_dimensions(page)
    text = analyze_text(page)
    text_blocks = analyze_text_blocks(page)
    spans = analyze_text_spans(page)
    images = analyze_images(page)
    drawings = analyze_drawings(page)

    page_area = dimensions["page_area"]
    text_characters = text["characters"]

    text_density = (
        text_characters / page_area
        if page_area > 0
        else 0.0
    )

    block_text_density = (
        text_blocks["characters"] / page_area
        if page_area > 0
        else 0.0
    )

    image_placement_area = sum(
        placement["area"]
        for image in images["records"]
        for placement in image["placements"]
    )

    image_coverage_ratio = (
        image_placement_area / page_area
        if page_area > 0
        else 0.0
    )

    return {
        "page_number": page_number,

        "dimensions": dimensions,

        "text": {
            **text,
            "block_count": text_blocks["count"],
            "span_count": spans["count"],
            "block_text_characters": text_blocks["characters"],
            "block_text_words": text_blocks["words"],
            "text_density": round(text_density, 8),
            "block_text_density": round(
                block_text_density,
                8,
            ),
        },

        "text_blocks": text_blocks["records"],

        "text_spans": spans["records"],

        "fonts": {
            "unique_font_count": spans["font_unique_count"],
            "usage_by_span": spans["font_usage_by_span"],
        },

        "images": {
            "reference_count": images["reference_count"],
            "unique_xref_count": images["unique_xref_count"],
            "records": images["records"],
            "placement_area": round(
                image_placement_area,
                2,
            ),
            "coverage_ratio": round(
                image_coverage_ratio,
                8,
            ),
        },

        "vectors": drawings,

        "derived": derive_page_state(
            text=text,
            images=images,
        ),
    }


# ============================================================
# DOCUMENT SUMMARY
# ============================================================

def build_document_summary(page_results):
    total_pages = len(page_results)

    page_states = Counter()

    text_counts = []
    word_counts = []
    block_counts = []
    span_counts = []
    image_reference_counts = []
    unique_image_xref_counts = []
    drawing_counts = []
    image_coverage_ratios = []

    orientations = Counter()

    for page in page_results:
        text = page["text"]
        images = page["images"]
        vectors = page["vectors"]
        dimensions = page["dimensions"]
        derived = page["derived"]

        page_states[derived["page_state"]] += 1

        text_counts.append(text["characters"])
        word_counts.append(text["words"])
        block_counts.append(text["block_count"])
        span_counts.append(text["span_count"])

        image_reference_counts.append(
            images["reference_count"]
        )

        unique_image_xref_counts.append(
            images["unique_xref_count"]
        )

        drawing_counts.append(
            vectors["drawing_count"]
        )

        image_coverage_ratios.append(
            images["coverage_ratio"]
        )

        orientations[
            dimensions["orientation"]
        ] += 1

    unique_document_image_xrefs = set()

    for page in page_results:
        for image in page["images"]["records"]:
            unique_document_image_xrefs.add(
                image["xref"]
            )

    # Internal consistency invariants.
    total_image_references = sum(
        image_reference_counts
    )

    total_drawings = sum(
        drawing_counts
    )

    pages_with_text = sum(
        1
        for page in page_results
        if page["derived"]["has_native_text"]
    )

    pages_with_images = sum(
        1
        for count in image_reference_counts
        if count > 0
    )

    max_images = (
        max(image_reference_counts)
        if image_reference_counts
        else 0
    )

    max_drawings = (
        max(drawing_counts)
        if drawing_counts
        else 0
    )

    return {
        "total_pages": total_pages,

        "page_states": dict(page_states),

        "text_statistics": {
            "characters": {
                "min": safe_min(text_counts),
                "max": safe_max(text_counts),
                "mean": safe_mean(text_counts),
                "median": safe_median(text_counts),
            },
            "words": {
                "min": safe_min(word_counts),
                "max": safe_max(word_counts),
                "mean": safe_mean(word_counts),
                "median": safe_median(word_counts),
            },
            "text_blocks_per_page": {
                "min": safe_min(block_counts),
                "max": safe_max(block_counts),
                "mean": safe_mean(block_counts),
                "median": safe_median(block_counts),
            },
            "text_spans_per_page": {
                "min": safe_min(span_counts),
                "max": safe_max(span_counts),
                "mean": safe_mean(span_counts),
                "median": safe_median(span_counts),
            },
        },

        "image_statistics": {
            "pages_with_images": pages_with_images,
            "total_image_references": total_image_references,
            "unique_image_xrefs_in_document": len(
                unique_document_image_xrefs
            ),
            "max_image_references_on_single_page": max_images,
            "mean_image_coverage_ratio": safe_mean(
                image_coverage_ratios
            ),
            "max_image_coverage_ratio": safe_max(
                image_coverage_ratios
            ),
        },

        "drawing_statistics": {
            "total_drawings": total_drawings,
            "max_drawings_on_single_page": max_drawings,
        },

        "orientation_distribution": dict(
            orientations
        ),

        "invariants": {
            "page_count_matches_page_records": (
                total_pages == len(page_results)
            ),
            "pages_with_text_matches_page_states": (
                pages_with_text
                == (
                    page_states["TEXT_ONLY"]
                    + page_states["MIXED_TEXT_IMAGE"]
                )
            ),
            "image_pages_match_page_states": (
                pages_with_images
                == (
                    page_states["IMAGE_ONLY"]
                    + page_states["MIXED_TEXT_IMAGE"]
                )
            ),
            "total_image_references_matches_page_sum": (
                total_image_references
                == sum(image_reference_counts)
            ),
            "total_drawings_matches_page_sum": (
                total_drawings
                == sum(drawing_counts)
            ),
        },
    }


# ============================================================
# PHASE 1 VALIDATION
# ============================================================

def validate_result(result):
    issues = []
    warnings = []

    pages = result["pages"]
    summary = result["summary"]

    # Page count.
    expected_page_count = result["document"]["page_count"]

    if expected_page_count != len(pages):
        issues.append(
            {
                "severity": "CRITICAL",
                "code": "PAGE_COUNT_MISMATCH",
                "expected": expected_page_count,
                "actual": len(pages),
            }
        )

    # Page numbering.
    expected_numbers = list(
        range(1, len(pages) + 1)
    )

    actual_numbers = [
        page["page_number"]
        for page in pages
    ]

    if actual_numbers != expected_numbers:
        issues.append(
            {
                "severity": "CRITICAL",
                "code": "PAGE_NUMBER_SEQUENCE_INVALID",
                "expected": expected_numbers,
                "actual": actual_numbers,
            }
        )

    # Page-level invariants.
    for page in pages:
        page_no = page["page_number"]

        state = page["derived"]["page_state"]
        has_text = page["text"]["has_text"]
        image_count = page["images"]["reference_count"]

        expected_state = (
            "EMPTY"
            if not has_text and image_count == 0
            else "IMAGE_ONLY"
            if not has_text and image_count > 0
            else "MIXED_TEXT_IMAGE"
            if has_text and image_count > 0
            else "TEXT_ONLY"
        )

        if state != expected_state:
            issues.append(
                {
                    "severity": "CRITICAL",
                    "code": "PAGE_STATE_INCONSISTENT",
                    "page": page_no,
                    "expected": expected_state,
                    "actual": state,
                }
            )

        if (
            page["images"]["unique_xref_count"]
            > page["images"]["reference_count"]
        ):
            issues.append(
                {
                    "severity": "CRITICAL",
                    "code": "IMAGE_XREF_COUNT_INVALID",
                    "page": page_no,
                    "expected": (
                        "<= reference_count"
                    ),
                    "actual": (
                        page["images"]["unique_xref_count"]
                    ),
                }
            )

        if (
            page["images"]["coverage_ratio"] < 0
            or page["images"]["coverage_ratio"] > 1.0 + 1e-9
        ):
            warnings.append(
                {
                    "severity": "WARNING",
                    "code": "IMAGE_COVERAGE_OUT_OF_RANGE",
                    "page": page_no,
                    "value": page["images"]["coverage_ratio"],
                }
            )

        if (
            page["text"]["block_count"]
            != len(page["text_blocks"])
        ):
            issues.append(
                {
                    "severity": "CRITICAL",
                    "code": "TEXT_BLOCK_COUNT_MISMATCH",
                    "page": page_no,
                }
            )

        if (
            page["text"]["span_count"]
            != len(page["text_spans"])
        ):
            issues.append(
                {
                    "severity": "CRITICAL",
                    "code": "TEXT_SPAN_COUNT_MISMATCH",
                    "page": page_no,
                }
            )

        if (
            page["vectors"]["drawing_count"] < 0
        ):
            issues.append(
                {
                    "severity": "CRITICAL",
                    "code": "NEGATIVE_DRAWING_COUNT",
                    "page": page_no,
                }
            )

    # Summary invariants.
    for name, value in summary["invariants"].items():
        if not value:
            issues.append(
                {
                    "severity": "CRITICAL",
                    "code": "SUMMARY_INVARIANT_FAILED",
                    "invariant": name,
                }
            )

    status = (
        "PASS"
        if not issues
        else "FAIL"
    )

    return {
        "status": status,
        "critical_issue_count": len(issues),
        "warning_count": len(warnings),
        "issues": issues,
        "warnings": warnings,
    }


# ============================================================
# MAIN ANALYZER
# ============================================================

def analyze_pdf(pdf_path: Path):
    document = pymupdf.open(pdf_path)

    metadata = document.metadata

    page_results = []

    for index, page in enumerate(
        document,
        start=1,
    ):
        page_results.append(
            analyze_page(
                page=page,
                page_number=index,
            )
        )

    summary = build_document_summary(
        page_results
    )

    result = {
        "analyzer": {
            "name": "PDF Ingestion Engine - Phase 1 Analyzer",
            "version": ANALYZER_VERSION,
            "engine": "PyMuPDF",
        },

        "source": {
            "file": str(pdf_path),
            "file_name": pdf_path.name,
            "file_size_bytes": pdf_path.stat().st_size,
            "sha256": file_sha256(pdf_path),
        },

        "metadata": metadata,

        "document": {
            "page_count": len(page_results),
        },

        "summary": summary,

        "pages": page_results,
    }

    document.close()

    result["validation"] = validate_result(result)

    return result


# ============================================================
# CLI
# ============================================================

def main():
    pdf_path = PDF_PATH
    output_path = OUTPUT_PATH

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    print("=" * 70)
    print("PDF INGESTION ENGINE")
    print("PHASE 1 — PDF ANALYZER")
    print("=" * 70)

    print(f"Input : {pdf_path}")
    print("Analyzing PDF...")

    result = analyze_pdf(pdf_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = result["summary"]
    validation = result["validation"]

    print()
    print("=" * 70)
    print("PHASE 1 ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Pages                    : "
        f"{summary['total_pages']}"
    )

    print(
        f"TEXT_ONLY                : "
        f"{summary['page_states'].get('TEXT_ONLY', 0)}"
    )

    print(
        f"MIXED_TEXT_IMAGE        : "
        f"{summary['page_states'].get('MIXED_TEXT_IMAGE', 0)}"
    )

    print(
        f"IMAGE_ONLY              : "
        f"{summary['page_states'].get('IMAGE_ONLY', 0)}"
    )

    print(
        f"EMPTY                   : "
        f"{summary['page_states'].get('EMPTY', 0)}"
    )

    print(
        f"Pages with images       : "
        f"{summary['image_statistics']['pages_with_images']}"
    )

    print(
        f"Image references        : "
        f"{summary['image_statistics']['total_image_references']}"
    )

    print(
        f"Unique image xrefs      : "
        f"{summary['image_statistics']['unique_image_xrefs_in_document']}"
    )

    print(
        f"Total vector drawings   : "
        f"{summary['drawing_statistics']['total_drawings']}"
    )

    print()
    print(
        f"Phase 1 validation      : "
        f"{validation['status']}"
    )

    print(
        f"Critical issues         : "
        f"{validation['critical_issue_count']}"
    )

    print(
        f"Warnings                : "
        f"{validation['warning_count']}"
    )

    print()
    print("Output:")
    print(output_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
