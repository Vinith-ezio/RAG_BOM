import json
from collections import Counter
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PHASE1_INPUT = Path(
    r"E:\PDF Ingestion\data\analysis\phase_1\bhel_spec_phase1.json"
)

PHASE2_OUTPUT = Path(
    r"E:\PDF Ingestion\data\analysis\phase_2\bhel_spec_phase2.json"
)

CLASSIFIER_VERSION = "0.2.0"


# ============================================================
# TEXT THRESHOLDS
# ============================================================

LOW_TEXT_CHARS = 1000
HIGH_TEXT_CHARS = 3000

LOW_TEXT_BLOCKS = 15
HIGH_TEXT_BLOCKS = 40
EXTREME_TEXT_BLOCKS = 70


# ============================================================
# VECTOR THRESHOLDS
# ============================================================

LOW_DRAWINGS = 100
HIGH_DRAWINGS = 500
EXTREME_DRAWINGS = 2000

LOW_RECTANGLES = 5
HIGH_RECTANGLES = 25
EXTREME_RECTANGLES = 75

LOW_CURVES = 100
HIGH_CURVES = 1000
EXTREME_CURVES = 10000


# ============================================================
# GRID / TABLE GEOMETRY THRESHOLDS
# ============================================================

MIN_HORIZONTAL_LINES_FOR_GRID = 2
MIN_VERTICAL_LINES_FOR_GRID = 2

STRONG_HORIZONTAL_LINES = 3
STRONG_VERTICAL_LINES = 2

STRONG_RECTANGLE_COUNT = 5


# ============================================================
# IMAGE THRESHOLDS
# ============================================================

MEDIUM_IMAGES = 3
HIGH_IMAGES = 5

LARGE_IMAGE_COVERAGE = 0.50
FULL_PAGE_IMAGE_COVERAGE = 0.90


# ============================================================
# TEXT DENSITY THRESHOLDS
# ============================================================

LOW_TEXT_DENSITY = 0.01
HIGH_TEXT_DENSITY = 0.05


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_int(value, default=0):
    """
    Safely convert a value to int.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value, default=0.0):
    """
    Safely convert a value to float.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_nested(data, *keys, default=None):
    """
    Safely retrieve nested dictionary values.

    Example:
        get_nested(page, "text", "characters")
    """

    current = data

    for key in keys:

        if not isinstance(current, dict):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def unique_preserve_order(values):
    """
    Remove duplicates while preserving order.
    """

    return list(
        dict.fromkeys(values)
    )


# ============================================================
# PAGE STATE
# ============================================================

def classify_page_state(page):
    """
    Use Phase 1 derived.page_state as the authoritative
    page-state classification.
    """

    page_state = get_nested(
        page,
        "derived",
        "page_state",
        default=None,
    )

    valid_states = {
        "EMPTY",
        "IMAGE_ONLY",
        "TEXT_ONLY",
        "MIXED_TEXT_IMAGE",
    }

    if page_state not in valid_states:

        raise ValueError(
            f"Invalid or missing page_state "
            f"for page {page.get('page_number')}: "
            f"{page_state}"
        )

    return page_state


# ============================================================
# TEXT COMPLEXITY
# ============================================================

def classify_text_complexity(characters):
    """
    Classify textual content volume.

    LOW:
        < 1000 characters

    MEDIUM:
        1000 - 2999 characters

    HIGH:
        >= 3000 characters
    """

    if characters < LOW_TEXT_CHARS:
        return "LOW"

    if characters < HIGH_TEXT_CHARS:
        return "MEDIUM"

    return "HIGH"


# ============================================================
# LAYOUT COMPLEXITY
# ============================================================

def classify_layout_complexity(text_blocks):
    """
    Classify layout density using native text block count.
    """

    if text_blocks < LOW_TEXT_BLOCKS:
        return "SIMPLE"

    if text_blocks < HIGH_TEXT_BLOCKS:
        return "MODERATE"

    if text_blocks < EXTREME_TEXT_BLOCKS:
        return "COMPLEX"

    return "EXTREME"


# ============================================================
# VECTOR COMPLEXITY
# ============================================================

def classify_vector_complexity(drawing_count):
    """
    Classify vector/drawing complexity.
    """

    if drawing_count < LOW_DRAWINGS:
        return "LOW"

    if drawing_count < HIGH_DRAWINGS:
        return "MEDIUM"

    if drawing_count < EXTREME_DRAWINGS:
        return "HIGH"

    return "EXTREME"


# ============================================================
# RECTANGLE COMPLEXITY
# ============================================================

def classify_rectangle_complexity(rectangle_count):
    """
    Classify rectangle geometry complexity.
    """

    if rectangle_count < LOW_RECTANGLES:
        return "LOW"

    if rectangle_count < HIGH_RECTANGLES:
        return "MEDIUM"

    if rectangle_count < EXTREME_RECTANGLES:
        return "HIGH"

    return "EXTREME"


# ============================================================
# CURVE COMPLEXITY
# ============================================================

def classify_curve_complexity(curve_count):
    """
    Classify curve geometry complexity.
    """

    if curve_count < LOW_CURVES:
        return "LOW"

    if curve_count < HIGH_CURVES:
        return "MEDIUM"

    if curve_count < EXTREME_CURVES:
        return "HIGH"

    return "EXTREME"


# ============================================================
# IMAGE COMPLEXITY
# ============================================================

def classify_image_complexity(image_count):
    """
    Classify image-reference count.
    """

    if image_count == 0:
        return "NONE"

    if image_count < MEDIUM_IMAGES:
        return "LOW"

    if image_count < HIGH_IMAGES:
        return "MEDIUM"

    return "HIGH"


# ============================================================
# TEXT SIGNALS
# ============================================================

def build_text_signals(
    characters,
    words,
    lines,
    spans,
    blocks,
    text_density,
):
    """
    Generate deterministic textual evidence.
    """

    signals = []

    if characters == 0:
        signals.append(
            "no_native_text"
        )

    elif characters < LOW_TEXT_CHARS:
        signals.append(
            "low_text_content"
        )

    elif characters >= HIGH_TEXT_CHARS:
        signals.append(
            "high_text_content"
        )

    if blocks >= HIGH_TEXT_BLOCKS:
        signals.append(
            "dense_text_blocks"
        )

    if blocks >= EXTREME_TEXT_BLOCKS:
        signals.append(
            "extreme_text_block_density"
        )

    if lines >= 100:
        signals.append(
            "many_text_lines"
        )

    if spans >= 100:
        signals.append(
            "many_text_spans"
        )

    if text_density >= HIGH_TEXT_DENSITY:
        signals.append(
            "high_text_density"
        )

    elif (
        text_density <= LOW_TEXT_DENSITY
        and characters > 0
    ):
        signals.append(
            "low_text_density"
        )

    return signals


# ============================================================
# VECTOR SIGNALS
# ============================================================

def build_vector_signals(
    drawing_count,
    rectangle_count,
    curve_count,
    horizontal_lines,
    vertical_lines,
):
    """
    Generate deterministic vector/geometry evidence.
    """

    signals = []

    if drawing_count >= HIGH_DRAWINGS:
        signals.append(
            "high_vector_complexity"
        )

    if drawing_count >= EXTREME_DRAWINGS:
        signals.append(
            "extreme_vector_complexity"
        )

    if rectangle_count >= HIGH_RECTANGLES:
        signals.append(
            "many_rectangles"
        )

    if rectangle_count >= EXTREME_RECTANGLES:
        signals.append(
            "extreme_rectangle_density"
        )

    if curve_count >= HIGH_CURVES:
        signals.append(
            "many_curves"
        )

    if curve_count >= EXTREME_CURVES:
        signals.append(
            "extreme_curve_density"
        )

    if horizontal_lines >= STRONG_HORIZONTAL_LINES:
        signals.append(
            "multiple_horizontal_lines"
        )

    if vertical_lines >= STRONG_VERTICAL_LINES:
        signals.append(
            "multiple_vertical_lines"
        )

    return signals


# ============================================================
# GRID / TABLE GEOMETRY EVIDENCE
# ============================================================

def detect_grid_evidence(
    horizontal_lines,
    vertical_lines,
    rectangle_count,
):
    """
    Detect geometric evidence that may represent a table/grid.

    IMPORTANT:
    This does NOT declare that the page contains a table.

    It only identifies geometry that is consistent with
    rows, columns, cells, or grid-like structures.
    """

    possible_grid = (
        horizontal_lines
        >= MIN_HORIZONTAL_LINES_FOR_GRID
        and
        vertical_lines
        >= MIN_VERTICAL_LINES_FOR_GRID
    )

    strong_grid_evidence = (
        horizontal_lines
        >= STRONG_HORIZONTAL_LINES
        and
        vertical_lines
        >= STRONG_VERTICAL_LINES
        and
        rectangle_count
        >= STRONG_RECTANGLE_COUNT
    )

    evidence = []

    if horizontal_lines >= MIN_HORIZONTAL_LINES_FOR_GRID:

        evidence.append(
            "multiple_horizontal_lines"
        )

    if vertical_lines >= MIN_VERTICAL_LINES_FOR_GRID:

        evidence.append(
            "multiple_vertical_lines"
        )

    if rectangle_count >= STRONG_RECTANGLE_COUNT:

        evidence.append(
            "multiple_rectangles"
        )

    return {
        "possible_grid": possible_grid,
        "strong_grid_evidence": strong_grid_evidence,
        "evidence": evidence,
    }


# ============================================================
# IMAGE EVIDENCE
# ============================================================

def classify_image_evidence(
    image_count,
    unique_image_xrefs,
    image_coverage_ratio,
):
    """
    Classify image geometry.

    Distinguishes:
        - no images
        - embedded images
        - large images
        - full-page images
    """

    signals = []

    if image_count == 0:

        return {
            "full_page_image": False,
            "large_image": False,
            "signals": signals,
        }

    if unique_image_xrefs > 0:

        signals.append(
            "image_references_present"
        )

    if image_coverage_ratio >= FULL_PAGE_IMAGE_COVERAGE:

        signals.append(
            "full_page_image"
        )

        return {
            "full_page_image": True,
            "large_image": True,
            "signals": signals,
        }

    if image_coverage_ratio >= LARGE_IMAGE_COVERAGE:

        signals.append(
            "large_image"
        )

        return {
            "full_page_image": False,
            "large_image": True,
            "signals": signals,
        }

    signals.append(
        "embedded_image"
    )

    return {
        "full_page_image": False,
        "large_image": False,
        "signals": signals,
    }


# ============================================================
# LAYOUT SIGNALS
# ============================================================

def build_layout_signals(
    page_state,
    text_blocks,
    image_count,
    grid_evidence,
):
    """
    Generate deterministic layout evidence.
    """

    signals = []

    if text_blocks >= HIGH_TEXT_BLOCKS:

        signals.append(
            "dense_layout"
        )

    if text_blocks >= EXTREME_TEXT_BLOCKS:

        signals.append(
            "extreme_layout_density"
        )

    if page_state == "MIXED_TEXT_IMAGE":

        signals.append(
            "mixed_content_layout"
        )

    if image_count > 0:

        signals.append(
            "contains_images"
        )

    if grid_evidence["possible_grid"]:

        signals.append(
            "grid_evidence"
        )

    if grid_evidence["strong_grid_evidence"]:

        signals.append(
            "strong_grid_evidence"
        )

    return signals


# ============================================================
# REVIEW SIGNALS
# ============================================================

def build_review_signals(
    page_state,
    characters,
    text_blocks,
    image_coverage_ratio,
    drawing_count,
    grid_evidence,
):
    """
    Identify pages that require special attention.

    REVIEW_REQUIRED means that the page has structural
    characteristics that may require specialized extraction
    or additional validation.

    It does NOT mean extraction has failed.
    """

    reasons = []

    # Strong geometric grid
    if grid_evidence["strong_grid_evidence"]:

        reasons.append(
            "strong_grid_evidence"
        )

    # Extreme vector complexity
    if drawing_count >= EXTREME_DRAWINGS:

        reasons.append(
            "extreme_vector_complexity"
        )

    # Very little text with significant geometry
    if (
        characters < LOW_TEXT_CHARS
        and
        drawing_count >= HIGH_DRAWINGS
    ):

        reasons.append(
            "low_text_with_complex_geometry"
        )

    # Very dense textual layout
    if text_blocks >= EXTREME_TEXT_BLOCKS:

        reasons.append(
            "extreme_layout_density"
        )

    # Full-page image
    if image_coverage_ratio >= FULL_PAGE_IMAGE_COVERAGE:

        reasons.append(
            "full_page_image"
        )

    # Mixed page with substantial image
    if (
        page_state == "MIXED_TEXT_IMAGE"
        and
        image_coverage_ratio >= LARGE_IMAGE_COVERAGE
    ):

        reasons.append(
            "large_image_with_native_text"
        )

    return {
        "required": len(reasons) > 0,
        "reasons": unique_preserve_order(
            reasons
        ),
    }


# ============================================================
# ROUTING EVIDENCE
# ============================================================

def build_routing_signals(
    page_state,
    characters,
    text_blocks,
    image_count,
    image_evidence,
    drawing_count,
    grid_evidence,
):
    """
    Generate extraction-routing evidence.

    IMPORTANT:

    This function does NOT select an extraction engine.

    It only produces evidence that the future
    Extraction Router will consume.
    """

    signals = []

    # --------------------------------------------------------
    # Page state
    # --------------------------------------------------------

    if page_state == "IMAGE_ONLY":

        signals.append(
            "requires_ocr"
        )

    if page_state == "MIXED_TEXT_IMAGE":

        signals.append(
            "mixed_content"
        )

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    if image_count > 0:

        signals.append(
            "contains_images"
        )

    if image_evidence["full_page_image"]:

        signals.append(
            "full_page_image"
        )

    if image_evidence["large_image"]:

        signals.append(
            "large_image"
        )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    if characters < LOW_TEXT_CHARS:

        signals.append(
            "low_text_content"
        )

    if text_blocks >= HIGH_TEXT_BLOCKS:

        signals.append(
            "dense_layout"
        )

    if text_blocks >= EXTREME_TEXT_BLOCKS:

        signals.append(
            "extreme_layout_density"
        )

    # --------------------------------------------------------
    # Vector
    # --------------------------------------------------------

    if drawing_count >= HIGH_DRAWINGS:

        signals.append(
            "high_vector_complexity"
        )

    if drawing_count >= EXTREME_DRAWINGS:

        signals.append(
            "extreme_vector_complexity"
        )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    if grid_evidence["possible_grid"]:

        signals.append(
            "grid_evidence"
        )

    if grid_evidence["strong_grid_evidence"]:

        signals.append(
            "strong_grid_evidence"
        )

    # --------------------------------------------------------
    # Layout analysis
    # --------------------------------------------------------

    if (
        page_state == "MIXED_TEXT_IMAGE"
        or
        grid_evidence["possible_grid"]
        or
        text_blocks >= HIGH_TEXT_BLOCKS
    ):

        signals.append(
            "requires_layout_analysis"
        )

    return unique_preserve_order(
        signals
    )


# ============================================================
# SINGLE PAGE CLASSIFICATION
# ============================================================

def classify_page(page):
    """
    Classify one page using the exact Phase 1 v0.2 schema.
    """

    page_number = safe_int(
        page.get("page_number")
    )

    # ========================================================
    # PAGE STATE
    # ========================================================

    page_state = classify_page_state(
        page
    )

    # ========================================================
    # DIMENSIONS
    # ========================================================

    orientation = get_nested(
        page,
        "dimensions",
        "orientation",
        default="unknown",
    )

    page_width = safe_float(
        get_nested(
            page,
            "dimensions",
            "width",
            default=0.0,
        )
    )

    page_height = safe_float(
        get_nested(
            page,
            "dimensions",
            "height",
            default=0.0,
        )
    )

    page_area = safe_float(
        get_nested(
            page,
            "dimensions",
            "page_area",
            default=0.0,
        )
    )

    # ========================================================
    # TEXT
    # ========================================================

    characters = safe_int(
        get_nested(
            page,
            "text",
            "characters",
        )
    )

    words = safe_int(
        get_nested(
            page,
            "text",
            "words",
        )
    )

    lines = safe_int(
        get_nested(
            page,
            "text",
            "lines",
        )
    )

    text_blocks = safe_int(
        get_nested(
            page,
            "text",
            "block_count",
        )
    )

    spans = safe_int(
        get_nested(
            page,
            "text",
            "span_count",
        )
    )

    text_density = safe_float(
        get_nested(
            page,
            "text",
            "text_density",
            default=0.0,
        )
    )

    block_text_density = safe_float(
        get_nested(
            page,
            "text",
            "block_text_density",
            default=0.0,
        )
    )

    # ========================================================
    # IMAGES
    # ========================================================

    image_count = safe_int(
        get_nested(
            page,
            "images",
            "reference_count",
        )
    )

    unique_image_xrefs = safe_int(
        get_nested(
            page,
            "images",
            "unique_xref_count",
            default=0,
        )
    )

    image_coverage_ratio = safe_float(
        get_nested(
            page,
            "images",
            "coverage_ratio",
            default=0.0,
        )
    )

    image_placement_area = safe_float(
        get_nested(
            page,
            "images",
            "placement_area",
            default=0.0,
        )
    )

    # ========================================================
    # VECTORS
    # ========================================================

    drawing_count = safe_int(
        get_nested(
            page,
            "vectors",
            "drawing_count",
        )
    )

    drawing_item_count = safe_int(
        get_nested(
            page,
            "vectors",
            "drawing_item_count",
            default=0,
        )
    )

    line_count = safe_int(
        get_nested(
            page,
            "vectors",
            "line_count",
            default=0,
        )
    )

    rectangle_count = safe_int(
        get_nested(
            page,
            "vectors",
            "rectangle_count",
            default=0,
        )
    )

    curve_count = safe_int(
        get_nested(
            page,
            "vectors",
            "curve_count",
            default=0,
        )
    )

    meaningful_horizontal_lines = safe_int(
        get_nested(
            page,
            "vectors",
            "meaningful_horizontal_line_count",
            default=0,
        )
    )

    meaningful_vertical_lines = safe_int(
        get_nested(
            page,
            "vectors",
            "meaningful_vertical_line_count",
            default=0,
        )
    )

    filled_drawing_count = safe_int(
        get_nested(
            page,
            "vectors",
            "filled_drawing_count",
            default=0,
        )
    )

    stroked_drawing_count = safe_int(
        get_nested(
            page,
            "vectors",
            "stroked_drawing_count",
            default=0,
        )
    )

    # ========================================================
    # COMPLEXITY
    # ========================================================

    text_complexity = (
        classify_text_complexity(
            characters
        )
    )

    layout_complexity = (
        classify_layout_complexity(
            text_blocks
        )
    )

    vector_complexity = (
        classify_vector_complexity(
            drawing_count
        )
    )

    rectangle_complexity = (
        classify_rectangle_complexity(
            rectangle_count
        )
    )

    curve_complexity = (
        classify_curve_complexity(
            curve_count
        )
    )

    image_complexity = (
        classify_image_complexity(
            image_count
        )
    )

    # ========================================================
    # TEXT SIGNALS
    # ========================================================

    text_signals = build_text_signals(
        characters=characters,
        words=words,
        lines=lines,
        spans=spans,
        blocks=text_blocks,
        text_density=text_density,
    )

    # ========================================================
    # VECTOR SIGNALS
    # ========================================================

    vector_signals = build_vector_signals(
        drawing_count=drawing_count,
        rectangle_count=rectangle_count,
        curve_count=curve_count,
        horizontal_lines=(
            meaningful_horizontal_lines
        ),
        vertical_lines=(
            meaningful_vertical_lines
        ),
    )

    # ========================================================
    # GRID EVIDENCE
    # ========================================================

    grid_evidence = detect_grid_evidence(
        horizontal_lines=(
            meaningful_horizontal_lines
        ),
        vertical_lines=(
            meaningful_vertical_lines
        ),
        rectangle_count=rectangle_count,
    )

    # ========================================================
    # IMAGE EVIDENCE
    # ========================================================

    image_evidence = classify_image_evidence(
        image_count=image_count,
        unique_image_xrefs=(
            unique_image_xrefs
        ),
        image_coverage_ratio=(
            image_coverage_ratio
        ),
    )

    # ========================================================
    # LAYOUT SIGNALS
    # ========================================================

    layout_signals = build_layout_signals(
        page_state=page_state,
        text_blocks=text_blocks,
        image_count=image_count,
        grid_evidence=grid_evidence,
    )

    # ========================================================
    # ROUTING SIGNALS
    # ========================================================

    routing_signals = build_routing_signals(
        page_state=page_state,
        characters=characters,
        text_blocks=text_blocks,
        image_count=image_count,
        image_evidence=image_evidence,
        drawing_count=drawing_count,
        grid_evidence=grid_evidence,
    )

    # ========================================================
    # REVIEW
    # ========================================================

    review = build_review_signals(
        page_state=page_state,
        characters=characters,
        text_blocks=text_blocks,
        image_coverage_ratio=(
            image_coverage_ratio
        ),
        drawing_count=drawing_count,
        grid_evidence=grid_evidence,
    )

    # ========================================================
    # RESULT
    # ========================================================

    return {
        "page_number": page_number,

        "page_state": page_state,

        "orientation": orientation,

        "dimensions": {
            "width": page_width,
            "height": page_height,
            "page_area": page_area,
        },

        "text": {
            "characters": characters,
            "words": words,
            "lines": lines,
            "blocks": text_blocks,
            "spans": spans,
            "text_density": text_density,
            "block_text_density": (
                block_text_density
            ),
            "complexity": text_complexity,
            "signals": text_signals,
        },

        "layout": {
            "complexity": layout_complexity,
            "signals": layout_signals,
        },

        "images": {
            "count": image_count,
            "unique_xrefs": unique_image_xrefs,
            "placement_area": (
                image_placement_area
            ),
            "coverage_ratio": (
                image_coverage_ratio
            ),
            "complexity": image_complexity,
            "evidence": image_evidence,
        },

        "vectors": {
            "drawings": drawing_count,
            "drawing_items": drawing_item_count,
            "lines": line_count,
            "rectangles": rectangle_count,
            "curves": curve_count,
            "filled_drawings": (
                filled_drawing_count
            ),
            "stroked_drawings": (
                stroked_drawing_count
            ),
            "meaningful_horizontal_lines": (
                meaningful_horizontal_lines
            ),
            "meaningful_vertical_lines": (
                meaningful_vertical_lines
            ),
            "complexity": vector_complexity,
            "rectangle_complexity": (
                rectangle_complexity
            ),
            "curve_complexity": (
                curve_complexity
            ),
            "signals": vector_signals,
        },

        "semantic_evidence": {
            "grid": grid_evidence,
        },

        "routing_signals": routing_signals,

        "review": review,
    }


# ============================================================
# DOCUMENT CLASSIFICATION
# ============================================================

def classify_document(phase1_data):
    """
    Classify every Phase 1 page.
    """

    pages = phase1_data.get(
        "pages"
    )

    if not isinstance(
        pages,
        list,
    ):

        raise ValueError(
            "Invalid Phase 1 JSON: "
            "'pages' must be a list."
        )

    classified_pages = []

    for page in pages:

        classified_pages.append(
            classify_page(
                page
            )
        )

    return classified_pages


# ============================================================
# COUNTER HELPER
# ============================================================

def count_values(
    classified_pages,
    path,
):
    """
    Count nested classification values.

    Example:
        count_values(
            pages,
            ("text", "complexity")
        )
    """

    counter = Counter()

    for page in classified_pages:

        value = page

        for key in path:

            if not isinstance(
                value,
                dict,
            ):

                value = None
                break

            value = value.get(
                key
            )

        if value is not None:

            counter[value] += 1

    return dict(
        counter
    )


# ============================================================
# DOCUMENT SUMMARY
# ============================================================

def build_summary(
    classified_pages,
):
    """
    Build document-level Phase 2 statistics.
    """

    review_pages = [
        page["page_number"]
        for page in classified_pages
        if page["review"]["required"]
    ]

    possible_grid_pages = [
        page["page_number"]
        for page in classified_pages
        if (
            page[
                "semantic_evidence"
            ][
                "grid"
            ][
                "possible_grid"
            ]
        )
    ]

    strong_grid_pages = [
        page["page_number"]
        for page in classified_pages
        if (
            page[
                "semantic_evidence"
            ][
                "grid"
            ][
                "strong_grid_evidence"
            ]
        )
    ]

    full_page_image_pages = [
        page["page_number"]
        for page in classified_pages
        if (
            page[
                "images"
            ][
                "evidence"
            ][
                "full_page_image"
            ]
        )
    ]

    large_image_pages = [
        page["page_number"]
        for page in classified_pages
        if (
            page[
                "images"
            ][
                "evidence"
            ][
                "large_image"
            ]
        )
    ]

    high_vector_pages = [
        page["page_number"]
        for page in classified_pages
        if (
            page[
                "vectors"
            ][
                "complexity"
            ]
            in {
                "HIGH",
                "EXTREME",
            }
        )
    ]

    extreme_vector_pages = [
        page["page_number"]
        for page in classified_pages
        if (
            page[
                "vectors"
            ][
                "complexity"
            ]
            == "EXTREME"
        )
    ]

    routing_signal_counts = Counter()

    for page in classified_pages:

        routing_signal_counts.update(
            page[
                "routing_signals"
            ]
        )

    return {

        "total_pages": len(
            classified_pages
        ),

        "page_states": count_values(
            classified_pages,
            ("page_state",),
        ),

        "text_complexity": count_values(
            classified_pages,
            (
                "text",
                "complexity",
            ),
        ),

        "layout_complexity": count_values(
            classified_pages,
            (
                "layout",
                "complexity",
            ),
        ),

        "vector_complexity": count_values(
            classified_pages,
            (
                "vectors",
                "complexity",
            ),
        ),

        "image_complexity": count_values(
            classified_pages,
            (
                "images",
                "complexity",
            ),
        ),

        "orientation_distribution": (
            count_values(
                classified_pages,
                ("orientation",),
            )
        ),

        "grid_evidence": {

            "possible_grid_pages": len(
                possible_grid_pages
            ),

            "possible_grid_page_numbers": (
                possible_grid_pages
            ),

            "strong_grid_pages": len(
                strong_grid_pages
            ),

            "strong_grid_page_numbers": (
                strong_grid_pages
            ),
        },

        "image_evidence": {

            "large_image_pages": len(
                large_image_pages
            ),

            "large_image_page_numbers": (
                large_image_pages
            ),

            "full_page_image_pages": len(
                full_page_image_pages
            ),

            "full_page_image_page_numbers": (
                full_page_image_pages
            ),
        },

        "vector_evidence": {

            "high_vector_pages": len(
                high_vector_pages
            ),

            "high_vector_page_numbers": (
                high_vector_pages
            ),

            "extreme_vector_pages": len(
                extreme_vector_pages
            ),

            "extreme_vector_page_numbers": (
                extreme_vector_pages
            ),
        },

        "review": {

            "pages_requiring_review": len(
                review_pages
            ),

            "page_numbers": review_pages,
        },

        "routing_signal_counts": dict(
            routing_signal_counts
        ),
    }


# ============================================================
# PHASE 1 INPUT VALIDATION
# ============================================================

def validate_phase1_input(
    phase1_data,
):
    """
    Validate the exact Phase 1 v0.2 schema required
    by Phase 2.

    This is a schema validation only.

    It does NOT validate semantic PDF accuracy.
    """

    errors = []

    # --------------------------------------------------------
    # Root
    # --------------------------------------------------------

    if not isinstance(
        phase1_data,
        dict,
    ):

        return [
            "Phase 1 root must be a JSON object."
        ]

    # --------------------------------------------------------
    # Pages
    # --------------------------------------------------------

    pages = phase1_data.get(
        "pages"
    )

    if not isinstance(
        pages,
        list,
    ):

        return [
            "Phase 1 'pages' must be a list."
        ]

    # --------------------------------------------------------
    # Required page fields
    # --------------------------------------------------------

    required_page_fields = {
        "page_number",
        "dimensions",
        "text",
        "images",
        "vectors",
        "derived",
    }

    required_dimension_fields = {
        "width",
        "height",
        "orientation",
        "page_area",
    }

    required_text_fields = {
        "characters",
        "words",
        "lines",
        "block_count",
        "span_count",
        "text_density",
        "block_text_density",
    }

    required_image_fields = {
        "reference_count",
        "unique_xref_count",
        "coverage_ratio",
    }

    required_vector_fields = {
        "drawing_count",
        "drawing_item_count",
        "line_count",
        "rectangle_count",
        "curve_count",
        "meaningful_horizontal_line_count",
        "meaningful_vertical_line_count",
    }

    valid_page_states = {
        "EMPTY",
        "IMAGE_ONLY",
        "TEXT_ONLY",
        "MIXED_TEXT_IMAGE",
    }

    # --------------------------------------------------------
    # Page-by-page validation
    # --------------------------------------------------------

    for index, page in enumerate(
        pages,
        start=1,
    ):

        if not isinstance(
            page,
            dict,
        ):

            errors.append(
                f"Page record {index} "
                "is not an object."
            )

            continue

        page_number = page.get(
            "page_number",
            index,
        )

        # ----------------------------------------------------
        # Top-level
        # ----------------------------------------------------

        for field in required_page_fields:

            if field not in page:

                errors.append(
                    f"Page {page_number} "
                    f"missing '{field}'."
                )

        # ----------------------------------------------------
        # Dimensions
        # ----------------------------------------------------

        dimensions = page.get(
            "dimensions"
        )

        if not isinstance(
            dimensions,
            dict,
        ):

            errors.append(
                f"Page {page_number} "
                "'dimensions' is not an object."
            )

        else:

            for field in required_dimension_fields:

                if field not in dimensions:

                    errors.append(
                        f"Page {page_number} "
                        f"dimensions missing "
                        f"'{field}'."
                    )

        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        text = page.get(
            "text"
        )

        if not isinstance(
            text,
            dict,
        ):

            errors.append(
                f"Page {page_number} "
                "'text' is not an object."
            )

        else:

            for field in required_text_fields:

                if field not in text:

                    errors.append(
                        f"Page {page_number} "
                        f"text missing "
                        f"'{field}'."
                    )

        # ----------------------------------------------------
        # Images
        # ----------------------------------------------------

        images = page.get(
            "images"
        )

        if not isinstance(
            images,
            dict,
        ):

            errors.append(
                f"Page {page_number} "
                "'images' is not an object."
            )

        else:

            for field in required_image_fields:

                if field not in images:

                    errors.append(
                        f"Page {page_number} "
                        f"images missing "
                        f"'{field}'."
                    )

        # ----------------------------------------------------
        # Vectors
        # ----------------------------------------------------

        vectors = page.get(
            "vectors"
        )

        if not isinstance(
            vectors,
            dict,
        ):

            errors.append(
                f"Page {page_number} "
                "'vectors' is not an object."
            )

        else:

            for field in required_vector_fields:

                if field not in vectors:

                    errors.append(
                        f"Page {page_number} "
                        f"vectors missing "
                        f"'{field}'."
                    )

        # ----------------------------------------------------
        # Derived
        # ----------------------------------------------------

        derived = page.get(
            "derived"
        )

        if not isinstance(
            derived,
            dict,
        ):

            errors.append(
                f"Page {page_number} "
                "'derived' is not an object."
            )

        else:

            page_state = derived.get(
                "page_state"
            )

            if page_state not in valid_page_states:

                errors.append(
                    f"Page {page_number} "
                    f"has invalid page_state: "
                    f"{page_state}"
                )

    return errors


# ============================================================
# PHASE 2 PROCESSOR
# ============================================================

def process_phase2():

    print("=" * 72)
    print(
        "PDF INGESTION ENGINE"
    )
    print(
        "PHASE 2 — PAGE / LAYOUT CLASSIFIER"
    )
    print("=" * 72)

    print()

    print(
        f"Input : {PHASE1_INPUT}"
    )

    print(
        f"Output: {PHASE2_OUTPUT}"
    )

    print()

    # ========================================================
    # INPUT CHECK
    # ========================================================

    if not PHASE1_INPUT.exists():

        raise FileNotFoundError(
            f"Phase 1 JSON not found:\n"
            f"{PHASE1_INPUT}"
        )

    # ========================================================
    # LOAD PHASE 1
    # ========================================================

    print(
        "Loading Phase 1 analysis..."
    )

    phase1_data = json.loads(
        PHASE1_INPUT.read_text(
            encoding="utf-8"
        )
    )

    # ========================================================
    # VALIDATE PHASE 1
    # ========================================================

    print(
        "Validating Phase 1 schema..."
    )

    validation_errors = (
        validate_phase1_input(
            phase1_data
        )
    )

    if validation_errors:

        print()

        print(
            "PHASE 1 INPUT VALIDATION FAILED"
        )

        for error in validation_errors:

            print(
                f"  ERROR: {error}"
            )

        raise ValueError(
            "Phase 1 input structure is invalid."
        )

    print(
        "Phase 1 schema validation : PASS"
    )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    print()

    print(
        "Classifying pages..."
    )

    classified_pages = (
        classify_document(
            phase1_data
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = build_summary(
        classified_pages
    )

    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "classifier": {

            "name": (
                "PDF Ingestion Engine - "
                "Phase 2 Page/Layout Classifier"
            ),

            "version": CLASSIFIER_VERSION,
        },

        "source": {

            "phase1_file": str(
                PHASE1_INPUT
            ),

        },

        "classification_policy": {

            "page_state_source": (
                "phase1.derived.page_state"
            ),

            "text_measurements": [
                "characters",
                "words",
                "lines",
                "block_count",
                "span_count",
                "text_density",
                "block_text_density",
            ],

            "image_measurements": [
                "reference_count",
                "unique_xref_count",
                "placement_area",
                "coverage_ratio",
            ],

            "vector_measurements": [
                "drawing_count",
                "drawing_item_count",
                "line_count",
                "rectangle_count",
                "curve_count",
                "meaningful_horizontal_line_count",
                "meaningful_vertical_line_count",
            ],

            "routing_policy": (
                "Evidence only. "
                "No extraction engine is selected "
                "in Phase 2."
            ),
        },

        "summary": summary,

        "pages": classified_pages,
    }

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    PHASE2_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # WRITE JSON
    # ========================================================

    PHASE2_OUTPUT.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # CONSOLE REPORT
    # ========================================================

    print()

    print("=" * 72)
    print(
        "PHASE 2 CLASSIFICATION COMPLETE"
    )
    print("=" * 72)

    print()

    print(
        f"Total pages : "
        f"{summary['total_pages']}"
    )

    # --------------------------------------------------------
    # Page states
    # --------------------------------------------------------

    print()
    print(
        "Page States:"
    )

    for name, count in (
        summary[
            "page_states"
        ].items()
    ):

        print(
            f"  {name:<20}: {count}"
        )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    print()
    print(
        "Text Complexity:"
    )

    for name, count in (
        summary[
            "text_complexity"
        ].items()
    ):

        print(
            f"  {name:<20}: {count}"
        )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    print()
    print(
        "Layout Complexity:"
    )

    for name, count in (
        summary[
            "layout_complexity"
        ].items()
    ):

        print(
            f"  {name:<20}: {count}"
        )

    # --------------------------------------------------------
    # Vector
    # --------------------------------------------------------

    print()
    print(
        "Vector Complexity:"
    )

    for name, count in (
        summary[
            "vector_complexity"
        ].items()
    ):

        print(
            f"  {name:<20}: {count}"
        )

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    print()
    print(
        "Image Complexity:"
    )

    for name, count in (
        summary[
            "image_complexity"
        ].items()
    ):

        print(
            f"  {name:<20}: {count}"
        )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    print()
    print(
        "Grid Evidence:"
    )

    print(
        "  Possible grid pages : "
        f"{summary['grid_evidence']['possible_grid_pages']}"
    )

    print(
        "  Possible grid nums  : "
        f"{summary['grid_evidence']['possible_grid_page_numbers']}"
    )

    print(
        "  Strong grid pages   : "
        f"{summary['grid_evidence']['strong_grid_pages']}"
    )

    print(
        "  Strong grid nums    : "
        f"{summary['grid_evidence']['strong_grid_page_numbers']}"
    )

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    print()
    print(
        "Image Evidence:"
    )

    print(
        "  Large image pages   : "
        f"{summary['image_evidence']['large_image_pages']}"
    )

    print(
        "  Large image nums    : "
        f"{summary['image_evidence']['large_image_page_numbers']}"
    )

    print(
        "  Full-page images    : "
        f"{summary['image_evidence']['full_page_image_pages']}"
    )

    print(
        "  Full-page image nums: "
        f"{summary['image_evidence']['full_page_image_page_numbers']}"
    )

    # --------------------------------------------------------
    # Vector evidence
    # --------------------------------------------------------

    print()
    print(
        "Vector Evidence:"
    )

    print(
        "  High+ vector pages  : "
        f"{summary['vector_evidence']['high_vector_pages']}"
    )

    print(
        "  High+ vector nums   : "
        f"{summary['vector_evidence']['high_vector_page_numbers']}"
    )

    print(
        "  Extreme vector pages: "
        f"{summary['vector_evidence']['extreme_vector_pages']}"
    )

    print(
        "  Extreme vector nums : "
        f"{summary['vector_evidence']['extreme_vector_page_numbers']}"
    )

    # --------------------------------------------------------
    # Review
    # --------------------------------------------------------

    print()
    print(
        "Review Required:"
    )

    print(
        "  Pages : "
        f"{summary['review']['pages_requiring_review']}"
    )

    print(
        "  Numbers: "
        f"{summary['review']['page_numbers']}"
    )

    # --------------------------------------------------------
    # Routing signal counts
    # --------------------------------------------------------

    print()
    print(
        "Routing Signal Counts:"
    )

    for signal, count in (
        summary[
            "routing_signal_counts"
        ].items()
    ):

        print(
            f"  {signal:<35}: {count}"
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print(
        "Output:"
    )

    print(
        PHASE2_OUTPUT
    )

    print()
    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    process_phase2()