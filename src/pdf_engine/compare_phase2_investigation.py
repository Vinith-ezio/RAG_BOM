import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PHASE2_INPUT = Path(
    r"E:\PDF Ingestion\data\analysis\phase_2\bhel_spec_phase2.json"
)

INVESTIGATION_DIR = Path(
    r"E:\PDF Ingestion\data\analysis\page_investigation"
)

OUTPUT_PATH = Path(
    r"E:\PDF Ingestion\data\analysis\page_investigation"
    r"\phase2_investigation_comparison.json"
)


# ============================================================
# STATUS
# ============================================================

CONSISTENT = "CONSISTENT"
WARNING = "WARNING"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
CONTRADICTION = "CONTRADICTION"


# ============================================================
# TOLERANCES
# ============================================================

# Character counts should normally be identical because both
# measurements originate from PyMuPDF text extraction.
CHARACTER_TOLERANCE = 0

# Word counts are useful but are treated as diagnostic because
# different tokenization logic can produce small differences.
WORD_TOLERANCE = 0

# Text block counts are NOT required to match.
#
# Phase 2 and Investigator use different structural concepts.
# Therefore this is diagnostic only.
BLOCK_COUNT_WARNING_THRESHOLD = 10


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path: Path):

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


# ============================================================
# LOAD PHASE 2
# ============================================================

def load_phase2():

    data = load_json(
        PHASE2_INPUT
    )

    required = {
        "classifier",
        "source",
        "summary",
        "pages",
    }

    missing = required - set(
        data.keys()
    )

    if missing:
        raise ValueError(
            "Phase 2 JSON is missing required "
            f"top-level keys: {sorted(missing)}"
        )

    return data


# ============================================================
# LOAD INVESTIGATION
# ============================================================

def load_investigation(page_number: int):

    path = (
        INVESTIGATION_DIR
        / f"page_{page_number:03d}_investigation.json"
    )

    if not path.exists():
        return None

    data = load_json(path)

    required = {
        "page_number",
        "geometry",
        "text",
        "images",
        "vectors",
        "relationships",
        "semantic_evidence",
    }

    missing = required - set(
        data.keys()
    )

    if missing:
        raise ValueError(
            f"Investigation page "
            f"{page_number:03d} is missing: "
            f"{sorted(missing)}"
        )

    return data


# ============================================================
# PHASE 2 PAGE MAP
# ============================================================

def build_phase2_page_map(
    phase2_data
):

    return {
        page["page_number"]: page
        for page in phase2_data["pages"]
    }


# ============================================================
# SAFE INTEGER
# ============================================================

def to_int(
    value,
    default=0
):

    if isinstance(
        value,
        bool
    ):
        return int(value)

    if isinstance(
        value,
        int
    ):
        return value

    if isinstance(
        value,
        float
    ):
        return int(value)

    return default


# ============================================================
# INVESTIGATOR TEXT METRICS
# ============================================================

def investigator_text_metrics(
    investigation
):

    text = investigation["text"]

    return {
        "characters": to_int(
            text.get("characters")
        ),

        "words": to_int(
            text.get("words")
        ),

        "block_count": to_int(
            text.get("block_count")
        ),

        "line_count": to_int(
            text.get("line_count")
        ),

        "span_count": to_int(
            text.get("span_count")
        ),
    }


# ============================================================
# INVESTIGATOR IMAGE METRICS
# ============================================================

def investigator_image_metrics(
    investigation
):

    images = investigation["images"]

    return {
        "count": to_int(
            images.get("count")
        ),

        "object_count": len(
            images.get(
                "objects",
                []
            )
        ),
    }


# ============================================================
# INVESTIGATOR VECTOR METRICS
# ============================================================

def investigator_vector_metrics(
    investigation
):

    vectors = investigation["vectors"]

    return {
        "drawing_count": to_int(
            vectors.get(
                "drawing_count"
            )
        ),

        "horizontal_line_count": to_int(
            vectors.get(
                "horizontal_line_count"
            )
        ),

        "vertical_line_count": to_int(
            vectors.get(
                "vertical_line_count"
            )
        ),

        "diagonal_line_count": to_int(
            vectors.get(
                "diagonal_line_count"
            )
        ),

        "rectangle_count": to_int(
            vectors.get(
                "rectangle_count"
            )
        ),

        "curve_count": to_int(
            vectors.get(
                "curve_count"
            )
        ),

        "meaningful_horizontal_line_count":
            to_int(
                vectors.get(
                    "meaningful_horizontal_line_count"
                )
            ),

        "meaningful_vertical_line_count":
            to_int(
                vectors.get(
                    "meaningful_vertical_line_count"
                )
            ),
    }


# ============================================================
# PAGE TYPE
# ============================================================

def derive_page_type(
    investigation
):

    text = investigator_text_metrics(
        investigation
    )

    images = investigator_image_metrics(
        investigation
    )

    characters = text["characters"]
    image_count = images["count"]

    if (
        characters == 0
        and image_count > 0
    ):
        return "IMAGE_ONLY"

    if (
        characters > 0
        and image_count > 0
    ):
        return "MIXED_TEXT_IMAGE"

    if (
        characters > 0
        and image_count == 0
    ):
        return "TEXT_ONLY"

    return "EMPTY"


# ============================================================
# PAGE TYPE COMPARISON
# ============================================================

def compare_page_type(
    phase2_page,
    investigation
):

    phase2_type = phase2_page[
        "page_type"
    ]

    investigation_type = derive_page_type(
        investigation
    )

    if (
        phase2_type
        == investigation_type
    ):

        return {
            "status": CONSISTENT,
            "phase2": phase2_type,
            "investigation": investigation_type,
        }

    return {
        "status": CONTRADICTION,
        "phase2": phase2_type,
        "investigation": investigation_type,
        "reason": (
            "Phase 2 and independent "
            "investigation disagree on "
            "physical page type."
        ),
    }


# ============================================================
# CHARACTER COUNT
# ============================================================

def compare_characters(
    phase2_page,
    investigation
):

    phase2_value = to_int(
        phase2_page[
            "text"
        ].get(
            "characters"
        )
    )

    investigation_value = (
        investigator_text_metrics(
            investigation
        )["characters"]
    )

    difference = (
        investigation_value
        - phase2_value
    )

    if abs(
        difference
    ) <= CHARACTER_TOLERANCE:

        status = CONSISTENT

    else:

        status = CONTRADICTION

    return {
        "status": status,
        "phase2": phase2_value,
        "investigation": investigation_value,
        "difference": difference,
    }


# ============================================================
# WORD COUNT
# ============================================================

def compare_words(
    phase2_page,
    investigation
):

    phase2_value = to_int(
        phase2_page[
            "text"
        ].get(
            "words"
        )
    )

    investigation_value = (
        investigator_text_metrics(
            investigation
        )["words"]
    )

    difference = (
        investigation_value
        - phase2_value
    )

    if abs(
        difference
    ) <= WORD_TOLERANCE:

        status = CONSISTENT

    else:

        status = WARNING

    return {
        "status": status,
        "phase2": phase2_value,
        "investigation": investigation_value,
        "difference": difference,
    }


# ============================================================
# TEXT BLOCK COUNT
# ============================================================

def compare_block_counts(
    phase2_page,
    investigation
):

    phase2_value = to_int(
        phase2_page[
            "text"
        ].get(
            "text_blocks"
        )
    )

    investigation_value = (
        investigator_text_metrics(
            investigation
        )["block_count"]
    )

    difference = (
        investigation_value
        - phase2_value
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Block counts are NOT treated as a hard equality check.
    #
    # The two systems collect blocks for different purposes.
    # --------------------------------------------------------

    if abs(
        difference
    ) <= BLOCK_COUNT_WARNING_THRESHOLD:

        status = CONSISTENT

    else:

        status = WARNING

    return {
        "status": status,
        "phase2": phase2_value,
        "investigation": investigation_value,
        "difference": difference,
        "comparison_type": "diagnostic_only",
    }


# ============================================================
# IMAGE COUNT
# ============================================================

def compare_image_count(
    phase2_page,
    investigation
):

    phase2_value = to_int(
        phase2_page[
            "images"
        ].get(
            "count"
        )
    )

    investigation_value = (
        investigator_image_metrics(
            investigation
        )["count"]
    )

    if (
        phase2_value
        == investigation_value
    ):

        status = CONSISTENT

    else:

        status = CONTRADICTION

    return {
        "status": status,
        "phase2": phase2_value,
        "investigation": investigation_value,
        "difference": (
            investigation_value
            - phase2_value
        ),
    }


# ============================================================
# DRAWING COUNT
# ============================================================

def compare_drawing_count(
    phase2_page,
    investigation
):

    phase2_value = to_int(
        phase2_page[
            "vectors"
        ].get(
            "drawing_count"
        )
    )

    investigation_value = (
        investigator_vector_metrics(
            investigation
        )["drawing_count"]
    )

    if (
        phase2_value
        == investigation_value
    ):

        status = CONSISTENT

    else:

        status = CONTRADICTION

    return {
        "status": status,
        "phase2": phase2_value,
        "investigation": investigation_value,
        "difference": (
            investigation_value
            - phase2_value
        ),
    }


# ============================================================
# VECTOR COMPLEXITY
# ============================================================

def derive_vector_complexity(
    drawing_count
):

    if drawing_count >= 2000:
        return "EXTREME"

    if drawing_count >= 500:
        return "HIGH"

    if drawing_count >= 100:
        return "MEDIUM"

    return "LOW"


def compare_vector_complexity(
    phase2_page,
    investigation
):

    phase2_value = (
        phase2_page[
            "vectors"
        ].get(
            "complexity"
        )
    )

    drawing_count = (
        investigator_vector_metrics(
            investigation
        )["drawing_count"]
    )

    expected = derive_vector_complexity(
        drawing_count
    )

    if phase2_value == expected:

        status = CONSISTENT

    else:

        status = CONTRADICTION

    return {
        "status": status,
        "phase2": phase2_value,
        "expected_from_investigation":
            expected,
        "drawing_count":
            drawing_count,
    }


# ============================================================
# IMAGE COMPLEXITY
# ============================================================

def derive_image_complexity(
    image_count
):

    # EXACT Phase 2 thresholds
    #
    # 0       -> NONE
    # 1-3     -> LOW
    # 4-5     -> MEDIUM
    # >5      -> HIGH

    if image_count == 0:
        return "NONE"

    if image_count <= 3:
        return "LOW"

    if image_count <= 5:
        return "MEDIUM"

    return "HIGH"


def compare_image_complexity(
    phase2_page,
    investigation
):

    phase2_value = (
        phase2_page[
            "images"
        ].get(
            "complexity"
        )
    )

    image_count = (
        investigator_image_metrics(
            investigation
        )["count"]
    )

    expected = derive_image_complexity(
        image_count
    )

    if phase2_value == expected:

        status = CONSISTENT

    else:

        status = CONTRADICTION

    return {
        "status": status,
        "phase2": phase2_value,
        "expected_from_investigation":
            expected,
        "image_count":
            image_count,
    }


# ============================================================
# GRID EVIDENCE
# ============================================================

def get_grid_evidence(
    investigation
):

    semantic = investigation.get(
        "semantic_evidence",
        {}
    )

    grid = semantic.get(
        "grid",
        {}
    )

    return {
        "possible_grid": bool(
            grid.get(
                "possible_grid",
                False
            )
        ),

        "strong_grid_evidence": bool(
            grid.get(
                "strong_grid_evidence",
                False
            )
        ),

        "evidence": grid.get(
            "evidence",
            []
        ),
    }


# ============================================================
# GRID ANALYSIS
# ============================================================

def analyze_grid(
    investigation
):

    grid = get_grid_evidence(
        investigation
    )

    if grid[
        "strong_grid_evidence"
    ]:

        return {
            "status": REVIEW_REQUIRED,
            **grid,
            "reason": (
                "Strong vector-grid evidence "
                "exists. This page requires "
                "semantic table/layout review."
            ),
        }

    if grid[
        "possible_grid"
    ]:

        return {
            "status": REVIEW_REQUIRED,
            **grid,
            "reason": (
                "Possible vector-grid evidence "
                "exists. This page requires "
                "semantic review."
            ),
        }

    return {
        "status": CONSISTENT,
        **grid,
        "reason": (
            "No vector-grid evidence "
            "was detected."
        ),
    }


# ============================================================
# ROUTING EVIDENCE
# ============================================================

def analyze_routing_evidence(
    phase2_page,
    investigation
):

    phase2_signals = set(
        phase2_page.get(
            "routing_signals",
            []
        )
    )

    text = investigator_text_metrics(
        investigation
    )

    images = investigator_image_metrics(
        investigation
    )

    vectors = investigator_vector_metrics(
        investigation
    )

    characters = text[
        "characters"
    ]

    blocks = text[
        "block_count"
    ]

    image_count = images[
        "count"
    ]

    drawing_count = vectors[
        "drawing_count"
    ]

    evidence = get_grid_evidence(
        investigation
    )

    # --------------------------------------------------------
    # Expected physical/routing evidence
    #
    # These are NOT treated as required exact matches.
    # --------------------------------------------------------

    physical_evidence = set()

    if image_count > 0:

        physical_evidence.add(
            "contains_images"
        )

    if characters < 1000:

        physical_evidence.add(
            "low_text_content"
        )

    if drawing_count >= 500:

        physical_evidence.add(
            "high_vector_complexity"
        )

    if drawing_count >= 2000:

        physical_evidence.add(
            "extreme_vector_complexity"
        )

    if blocks >= 40:

        physical_evidence.add(
            "dense_layout"
        )

    if blocks >= 70:

        physical_evidence.add(
            "extreme_layout_density"
        )

    # --------------------------------------------------------
    # Semantic routing evidence
    # --------------------------------------------------------

    if (
        evidence["possible_grid"]
        or evidence["strong_grid_evidence"]
    ):

        physical_evidence.add(
            "possible_vector_grid"
        )

    # --------------------------------------------------------
    # Compare only Phase 2's known signals.
    #
    # We do NOT call missing semantic evidence a
    # contradiction.
    # --------------------------------------------------------

    recognized_phase2_signals = (
        physical_evidence
        & phase2_signals
    )

    unsupported_phase2_signals = (
        phase2_signals
        - physical_evidence
    )

    # --------------------------------------------------------
    # Important:
    #
    # A semantic grid does not mean Phase 2 is wrong.
    # It means the classifier has evidence that may
    # deserve an additional routing category later.
    # --------------------------------------------------------

    if evidence[
        "strong_grid_evidence"
    ]:

        status = REVIEW_REQUIRED

    elif evidence[
        "possible_grid"
    ]:

        status = REVIEW_REQUIRED

    elif unsupported_phase2_signals:

        status = WARNING

    else:

        status = CONSISTENT

    return {
        "status": status,

        "phase2_signals":
            sorted(
                phase2_signals
            ),

        "physical_evidence":
            sorted(
                physical_evidence
            ),

        "recognized_phase2_signals":
            sorted(
                recognized_phase2_signals
            ),

        "unsupported_phase2_signals":
            sorted(
                unsupported_phase2_signals
            ),

        "grid_evidence":
            evidence,
    }


# ============================================================
# PAGE RECONCILIATION
# ============================================================

def compare_page(
    phase2_page,
    investigation
):

    page_number = phase2_page[
        "page_number"
    ]

    if investigation is None:

        return {
            "page_number": page_number,
            "overall_status":
                CONTRADICTION,
            "reason":
                "Investigation file missing.",
        }

    checks = {

        "page_type":
            compare_page_type(
                phase2_page,
                investigation
            ),

        "characters":
            compare_characters(
                phase2_page,
                investigation
            ),

        "words":
            compare_words(
                phase2_page,
                investigation
            ),

        "block_counts":
            compare_block_counts(
                phase2_page,
                investigation
            ),

        "image_count":
            compare_image_count(
                phase2_page,
                investigation
            ),

        "drawing_count":
            compare_drawing_count(
                phase2_page,
                investigation
            ),

        "vector_complexity":
            compare_vector_complexity(
                phase2_page,
                investigation
            ),

        "image_complexity":
            compare_image_complexity(
                phase2_page,
                investigation
            ),

        "grid_analysis":
            analyze_grid(
                investigation
            ),

        "routing_analysis":
            analyze_routing_evidence(
                phase2_page,
                investigation
            ),
    }

    # ========================================================
    # OVERALL STATUS
    # ========================================================

    statuses = []

    for check in checks.values():

        if isinstance(
            check,
            dict
        ):

            status = check.get(
                "status"
            )

            if status:
                statuses.append(
                    status
                )

    if CONTRADICTION in statuses:

        overall_status = CONTRADICTION

    elif REVIEW_REQUIRED in statuses:

        overall_status = REVIEW_REQUIRED

    elif WARNING in statuses:

        overall_status = WARNING

    else:

        overall_status = CONSISTENT

    # ========================================================
    # RESULT
    # ========================================================

    return {

        "page_number":
            page_number,

        "overall_status":
            overall_status,

        "phase2": {

            "page_type":
                phase2_page.get(
                    "page_type"
                ),

            "text_complexity":
                phase2_page[
                    "text"
                ].get(
                    "complexity"
                ),

            "layout_complexity":
                phase2_page[
                    "layout"
                ].get(
                    "complexity"
                ),

            "vector_complexity":
                phase2_page[
                    "vectors"
                ].get(
                    "complexity"
                ),

            "image_complexity":
                phase2_page[
                    "images"
                ].get(
                    "complexity"
                ),

            "routing_signals":
                phase2_page.get(
                    "routing_signals",
                    []
                ),
        },

        "investigation": {

            "page_type":
                derive_page_type(
                    investigation
                ),

            "text":
                investigator_text_metrics(
                    investigation
                ),

            "images":
                investigator_image_metrics(
                    investigation
                ),

            "vectors":
                investigator_vector_metrics(
                    investigation
                ),

            "grid":
                get_grid_evidence(
                    investigation
                ),
        },

        "checks":
            checks,
    }


# ============================================================
# DOCUMENT SUMMARY
# ============================================================

def build_summary(
    results
):

    counts = {
        CONSISTENT: 0,
        WARNING: 0,
        REVIEW_REQUIRED: 0,
        CONTRADICTION: 0,
    }

    page_lists = {
        CONSISTENT: [],
        WARNING: [],
        REVIEW_REQUIRED: [],
        CONTRADICTION: [],
    }

    for result in results:

        status = result[
            "overall_status"
        ]

        page = result[
            "page_number"
        ]

        counts[
            status
        ] += 1

        page_lists[
            status
        ].append(
            page
        )

    return {
        "total_pages":
            len(results),

        "status_counts":
            counts,

        "consistent_page_numbers":
            page_lists[
                CONSISTENT
            ],

        "warning_page_numbers":
            page_lists[
                WARNING
            ],

        "review_required_page_numbers":
            page_lists[
                REVIEW_REQUIRED
            ],

        "contradiction_page_numbers":
            page_lists[
                CONTRADICTION
            ],
    }


# ============================================================
# DOCUMENT COMPARISON
# ============================================================

def compare_document():

    phase2_data = load_phase2()

    phase2_pages = (
        build_phase2_page_map(
            phase2_data
        )
    )

    total_pages = (
        phase2_data[
            "summary"
        ][
            "total_pages"
        ]
    )

    results = []

    for page_number in range(
        1,
        total_pages + 1
    ):

        phase2_page = (
            phase2_pages.get(
                page_number
            )
        )

        if phase2_page is None:

            result = {
                "page_number":
                    page_number,

                "overall_status":
                    CONTRADICTION,

                "reason":
                    "Page missing from Phase 2.",
            }

        else:

            investigation = (
                load_investigation(
                    page_number
                )
            )

            result = compare_page(
                phase2_page,
                investigation
            )

        results.append(
            result
        )

        print(
            f"Page {page_number:03d} : "
            f"{result['overall_status']}"
        )

    summary = build_summary(
        results
    )

    output = {

        "comparison": {

            "name":
                "Phase 2 vs Page Investigation",

            "version":
                "3.0",

            "purpose":
                (
                    "Independent reconciliation of "
                    "Phase 2 classification against "
                    "PDF page investigation evidence."
                ),

            "principles": [

                (
                    "Investigator is an evidence source, "
                    "not ground truth."
                ),

                (
                    "Derived structural measurements "
                    "are not forced to be identical."
                ),

                (
                    "Physical contradictions are "
                    "separated from semantic review."
                ),

                (
                    "Grid evidence produces "
                    "REVIEW_REQUIRED rather than "
                    "automatic classification failure."
                ),

                (
                    "No Phase 2 classification is "
                    "modified by this comparison."
                ),
            ],
        },

        "source": {

            "phase2":
                str(
                    PHASE2_INPUT
                ),

            "investigation_directory":
                str(
                    INVESTIGATION_DIR
                ),
        },

        "summary":
            summary,

        "pages":
            results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "PHASE 2 vs PAGE INVESTIGATION COMPARISON"
    )
    print("=" * 70)

    output = compare_document()

    summary = output[
        "summary"
    ]

    counts = summary[
        "status_counts"
    ]

    print()
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(
        f"Total pages       : "
        f"{summary['total_pages']}"
    )

    print(
        f"Consistent        : "
        f"{counts[CONSISTENT]}"
    )

    print(
        f"Warnings          : "
        f"{counts[WARNING]}"
    )

    print(
        f"Review required   : "
        f"{counts[REVIEW_REQUIRED]}"
    )

    print(
        f"Contradictions    : "
        f"{counts[CONTRADICTION]}"
    )

    print()
    print(
        "Review pages      :",
        summary[
            "review_required_page_numbers"
        ]
    )

    print(
        "Contradictions    :",
        summary[
            "contradiction_page_numbers"
        ]
    )

    print()
    print(
        f"Output            : "
        f"{OUTPUT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()