import json
import sys
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PHASE1_INPUT = Path(
    r"E:\PDF Ingestion\data\analysis\phase_1\bhel_spec_phase1.json"
)

PHASE2_INPUT = Path(
    r"E:\PDF Ingestion\data\analysis\phase_2\bhel_spec_phase2.json"
)

OUTPUT_PATH = Path(
    r"E:\PDF Ingestion\data\analysis\phase_2\phase2_validation.json"
)


# ============================================================
# THRESHOLDS
# These MUST match phase2_classifier.py
# ============================================================

LOW_TEXT_CHARS = 1000
HIGH_TEXT_CHARS = 3000

LOW_TEXT_BLOCKS = 15
HIGH_TEXT_BLOCKS = 40
EXTREME_TEXT_BLOCKS = 70

LOW_DRAWINGS = 100
HIGH_DRAWINGS = 500
EXTREME_DRAWINGS = 2000

MEDIUM_IMAGES = 3
HIGH_IMAGES = 5


# ============================================================
# VALIDATION HELPERS
# ============================================================

def add_issue(
    issues,
    page_number,
    category,
    field,
    expected,
    actual,
    severity="ERROR",
):
    issues.append(
        {
            "page_number": page_number,
            "category": category,
            "field": field,
            "expected": expected,
            "actual": actual,
            "severity": severity,
        }
    )


# ============================================================
# TEXT COMPLEXITY VALIDATION
# ============================================================

def expected_text_complexity(characters):

    if characters < LOW_TEXT_CHARS:
        return "LOW"

    if characters >= HIGH_TEXT_CHARS:
        return "HIGH"

    return "MEDIUM"


# ============================================================
# LAYOUT COMPLEXITY VALIDATION
# ============================================================

def expected_layout_complexity(text_blocks):

    if text_blocks < LOW_TEXT_BLOCKS:
        return "SIMPLE"

    if text_blocks >= EXTREME_TEXT_BLOCKS:
        return "EXTREME"

    if text_blocks >= HIGH_TEXT_BLOCKS:
        return "COMPLEX"

    return "MODERATE"


# ============================================================
# VECTOR COMPLEXITY VALIDATION
# ============================================================

def expected_vector_complexity(drawing_count):

    if drawing_count < LOW_DRAWINGS:
        return "LOW"

    if drawing_count >= EXTREME_DRAWINGS:
        return "EXTREME"

    if drawing_count >= HIGH_DRAWINGS:
        return "HIGH"

    return "MEDIUM"


# ============================================================
# IMAGE COMPLEXITY VALIDATION
# ============================================================

def expected_image_complexity(image_count):

    if image_count == 0:
        return "NONE"

    if image_count < MEDIUM_IMAGES:
        return "LOW"

    if image_count < HIGH_IMAGES:
        return "MEDIUM"

    return "HIGH"


# ============================================================
# PAGE TYPE VALIDATION
# ============================================================

def expected_page_type(characters, image_count):

    if characters == 0 and image_count > 0:
        return "IMAGE_ONLY"

    if characters > 0 and image_count == 0:
        return "TEXT_ONLY"

    if characters > 0 and image_count > 0:
        return "MIXED_TEXT_IMAGE"

    return "EMPTY"


# ============================================================
# ROUTING SIGNAL VALIDATION
# ============================================================

def expected_routing_signals(
    characters,
    text_blocks,
    image_count,
    drawing_count,
):

    signals = []

    if characters == 0 and image_count > 0:
        signals.append("requires_ocr")

    if image_count > 0:
        signals.append("contains_images")

    if drawing_count >= HIGH_DRAWINGS:
        signals.append("high_vector_complexity")

    if drawing_count >= EXTREME_DRAWINGS:
        signals.append("extreme_vector_complexity")

    if characters < LOW_TEXT_CHARS:
        signals.append("low_text_content")

    if text_blocks >= HIGH_TEXT_BLOCKS:
        signals.append("dense_layout")

    if text_blocks >= EXTREME_TEXT_BLOCKS:
        signals.append("extreme_layout_density")

    if characters > 0 and image_count > 0:
        signals.append("requires_layout_analysis")

    return signals


# ============================================================
# PHASE 1 / PHASE 2 PAGE CONSISTENCY
# ============================================================

def validate_page(
    phase1_page,
    phase2_page,
    issues,
):

    page_number = phase1_page["page_number"]

    # --------------------------------------------------------
    # Page number
    # --------------------------------------------------------

    if phase2_page.get("page_number") != page_number:

        add_issue(
            issues,
            page_number,
            "PAGE_NUMBER",
            "page_number",
            page_number,
            phase2_page.get("page_number"),
        )

        return

    # --------------------------------------------------------
    # Read Phase 1 measurements
    # --------------------------------------------------------

    characters = phase1_page["text"]["characters"]

    text_blocks = len(
        phase1_page["text_blocks"]
    )

    image_count = len(
        phase1_page["images"]
    )

    drawing_count = len(
        phase1_page["drawings"]
    )

    # --------------------------------------------------------
    # Read Phase 2 classifications
    # --------------------------------------------------------

    actual_page_type = phase2_page["page_type"]

    actual_text_complexity = (
        phase2_page["text"]["complexity"]
    )

    actual_layout_complexity = (
        phase2_page["layout"]["complexity"]
    )

    actual_vector_complexity = (
        phase2_page["vectors"]["complexity"]
    )

    actual_image_complexity = (
        phase2_page["images"]["complexity"]
    )

    actual_signals = sorted(
        phase2_page.get("routing_signals", [])
    )

    # --------------------------------------------------------
    # Expected classifications
    # --------------------------------------------------------

    expected_type = expected_page_type(
        characters,
        image_count,
    )

    expected_text = expected_text_complexity(
        characters
    )

    expected_layout = expected_layout_complexity(
        text_blocks
    )

    expected_vector = expected_vector_complexity(
        drawing_count
    )

    expected_image = expected_image_complexity(
        image_count
    )

    expected_signals = sorted(
        expected_routing_signals(
            characters,
            text_blocks,
            image_count,
            drawing_count,
        )
    )

    # --------------------------------------------------------
    # Validate page type
    # --------------------------------------------------------

    if actual_page_type != expected_type:

        add_issue(
            issues,
            page_number,
            "PAGE_TYPE",
            "page_type",
            expected_type,
            actual_page_type,
        )

    # --------------------------------------------------------
    # Validate text complexity
    # --------------------------------------------------------

    if actual_text_complexity != expected_text:

        add_issue(
            issues,
            page_number,
            "TEXT_COMPLEXITY",
            "text.complexity",
            expected_text,
            actual_text_complexity,
        )

    # --------------------------------------------------------
    # Validate layout complexity
    # --------------------------------------------------------

    if actual_layout_complexity != expected_layout:

        add_issue(
            issues,
            page_number,
            "LAYOUT_COMPLEXITY",
            "layout.complexity",
            expected_layout,
            actual_layout_complexity,
        )

    # --------------------------------------------------------
    # Validate vector complexity
    # --------------------------------------------------------

    if actual_vector_complexity != expected_vector:

        add_issue(
            issues,
            page_number,
            "VECTOR_COMPLEXITY",
            "vectors.complexity",
            expected_vector,
            actual_vector_complexity,
        )

    # --------------------------------------------------------
    # Validate image complexity
    # --------------------------------------------------------

    if actual_image_complexity != expected_image:

        add_issue(
            issues,
            page_number,
            "IMAGE_COMPLEXITY",
            "images.complexity",
            expected_image,
            actual_image_complexity,
        )

    # --------------------------------------------------------
    # Validate routing signals
    # --------------------------------------------------------

    if actual_signals != expected_signals:

        add_issue(
            issues,
            page_number,
            "ROUTING_SIGNALS",
            "routing_signals",
            expected_signals,
            actual_signals,
        )


# ============================================================
# DOCUMENT VALIDATION
# ============================================================

def validate_document(phase1_data, phase2_data):

    issues = []

    phase1_pages = phase1_data["pages"]
    phase2_pages = phase2_data["pages"]

    # --------------------------------------------------------
    # Page count
    # --------------------------------------------------------

    if len(phase1_pages) != len(phase2_pages):

        add_issue(
            issues,
            0,
            "DOCUMENT",
            "page_count",
            len(phase1_pages),
            len(phase2_pages),
        )

    # --------------------------------------------------------
    # Validate every page
    # --------------------------------------------------------

    phase2_by_number = {
        page["page_number"]: page
        for page in phase2_pages
    }

    for phase1_page in phase1_pages:

        page_number = phase1_page["page_number"]

        phase2_page = phase2_by_number.get(
            page_number
        )

        if phase2_page is None:

            add_issue(
                issues,
                page_number,
                "DOCUMENT",
                "page",
                "page exists in Phase 2",
                "missing",
            )

            continue

        validate_page(
            phase1_page,
            phase2_page,
            issues,
        )

    return issues


# ============================================================
# SUMMARY
# ============================================================

def build_report(
    phase1_data,
    phase2_data,
    issues,
):

    total_pages = len(phase1_data["pages"])

    failed_pages = sorted(
        {
            issue["page_number"]
            for issue in issues
            if issue["page_number"] != 0
        }
    )

    passed_pages = total_pages - len(
        failed_pages
    )

    return {
        "validator": "phase2_validator",
        "version": "1.0",
        "phase1_source": str(PHASE1_INPUT),
        "phase2_source": str(PHASE2_INPUT),

        "document": {
            "source_file": phase1_data["source"]["file_name"],
            "total_pages": total_pages,
        },

        "validation": {
            "passed_pages": passed_pages,
            "failed_pages": len(failed_pages),
            "failed_page_numbers": failed_pages,
            "total_issues": len(issues),
            "overall_status": (
                "PASS"
                if len(issues) == 0
                else "FAIL"
            ),
        },

        "issues": issues,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 2 VALIDATION")
    print("=" * 70)

    print(f"Phase 1: {PHASE1_INPUT}")
    print(f"Phase 2: {PHASE2_INPUT}")
    print(f"Output : {OUTPUT_PATH}")
    print()

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not PHASE1_INPUT.exists():

        print("ERROR: Phase 1 JSON not found.")
        sys.exit(1)

    if not PHASE2_INPUT.exists():

        print("ERROR: Phase 2 JSON not found.")
        sys.exit(1)

    # --------------------------------------------------------
    # Load JSON
    # --------------------------------------------------------

    with PHASE1_INPUT.open(
        "r",
        encoding="utf-8",
    ) as f:

        phase1_data = json.load(f)

    with PHASE2_INPUT.open(
        "r",
        encoding="utf-8",
    ) as f:

        phase2_data = json.load(f)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    issues = validate_document(
        phase1_data,
        phase2_data,
    )

    # --------------------------------------------------------
    # Build report
    # --------------------------------------------------------

    report = build_report(
        phase1_data,
        phase2_data,
        issues,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    validation = report["validation"]

    print(
        f"Total Pages  : {total_pages if 'total_pages' in locals() else len(phase1_data['pages'])}"
    )

    print(
        f"Passed Pages : {validation['passed_pages']}"
    )

    print(
        f"Failed Pages : {validation['failed_pages']}"
    )

    print(
        f"Total Issues : {validation['total_issues']}"
    )

    print(
        f"Status       : {validation['overall_status']}"
    )

    if issues:

        print()
        print("Issues:")
        print("-" * 70)

        for issue in issues:

            print(
                f"Page {issue['page_number']:03d} | "
                f"{issue['category']} | "
                f"{issue['field']} | "
                f"Expected={issue['expected']} | "
                f"Actual={issue['actual']}"
            )

    print()
    print(
        f"Validation report written to:\n{OUTPUT_PATH}"
    )

    if issues:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()