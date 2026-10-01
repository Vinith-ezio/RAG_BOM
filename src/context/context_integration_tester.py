"""
Phase 4 — Context Integration Tester
====================================

Complete integration/regression test for:

    Query
      ↓
    Phase 3.2 Intent Retrieval Adapter
      ↓
    Phase 4 Context Builder
      ↓
    RAG Context

Purpose
-------
Stress-test the retrieval-to-context boundary across many
different query types.

This test suite validates:

1. Entity-scoped queries
2. Identifier -> ITEM queries
3. Identifier -> parameter queries
4. Side-specific queries
5. Hierarchical parameter queries
6. Section queries
7. Global schema queries
8. Multi-intent queries
9. Identifier + multi-intent queries
10. Ambiguous queries
11. Negative / unsupported queries
12. Unknown identifiers
13. Case / formatting variations
14. Provenance preservation
15. Context integrity
16. Deduplication
17. Large-context preservation

No LLM
-------
No Qwen
No embeddings
No external retrieval
No answer generation

The purpose of this file is testing only.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


# ================================================================
# PATH SETUP
# ================================================================

ROOT = Path(__file__).resolve().parents[2]

QUERY_DIR = ROOT / "src" / "query"
RETRIEVAL_DIR = ROOT / "src" / "retrieval"
CONTEXT_DIR = ROOT / "src" / "context"

for directory in (
    QUERY_DIR,
    RETRIEVAL_DIR,
    CONTEXT_DIR,
):
    directory_string = str(directory)

    if directory_string not in sys.path:
        sys.path.insert(0, directory_string)


# ================================================================
# IMPORTS
# ================================================================

from intent_retrieval_adapter import (
    IntentRetrievalAdapter,
)

from context_builder import (
    ContextBuilder,
)


# ================================================================
# PATHS
# ================================================================

OUTPUT_DIR = ROOT / "data" / "output"

SCHEMA_CATALOG_PATH = (
    OUTPUT_DIR / "schema_catalog.json"
)

ENTITY_INDEX_PATH = (
    OUTPUT_DIR / "entity_index.json"
)

RETRIEVAL_RECORDS_PATH = (
    OUTPUT_DIR / "retrieval_enriched.json"
)


# ================================================================
# TEST CASES
# ================================================================

TEST_CASES: List[Dict[str, Any]] = [

    # ============================================================
    # 1. ENTITY-SCOPED QUERIES
    # ============================================================

    {
        "id": "E001",
        "category": "entity_scoped",
        "query": (
            'What is the actuator type of '
            'DSH 6"300RF-INTEG TCV 1"600RF-HART?'
        ),
        "expected_mode": "entity_scoped_lookup",
        "expected_ids": ["p02_r40"],
        "expected_count": 1,
    },

    {
        "id": "E002",
        "category": "entity_scoped",
        "query": (
            'What is the actuator type for '
            'DSH 6"300RF-INTEG TCV 1"600RF-HART?'
        ),
        "expected_mode": "entity_scoped_lookup",
        "expected_ids": ["p02_r40"],
        "expected_count": 1,
    },

    {
        "id": "E003",
        "category": "entity_scoped",
        "query": (
            'What is the body material of '
            'DSH 6"300RF-INTEG TCV 1"600RF-HART?'
        ),
        "expected_mode": "entity_scoped_lookup",
        "expected_contains": ["p02_r17_inlet"],
        "min_count": 1,
    },

    {
        "id": "E004",
        "category": "entity_scoped",
        "query": (
            'What is the line size of '
            'DSH 6"300RF-INTEG TCV 1"600RF-HART?'
        ),
        "expected_mode": "entity_scoped_lookup",
        "min_count": 1,
    },

    {
        "id": "E005",
        "category": "entity_scoped",
        "query": (
            'What is the material code of '
            'DSH 6"300RF-INTEG TCV 1"600RF-HART?'
        ),
        "expected_mode": "entity_scoped_lookup",
        "expected_ids": ["p02_r58"],
        "expected_count": 1,
    },


    # ============================================================
    # 2. IDENTIFIER -> ITEM
    # ============================================================

    {
        "id": "I001",
        "category": "identifier_to_item",
        "query": (
            "What is the item of "
            "TC9765132581?"
        ),
        "expected_mode": "entity_scoped_lookup",
        "expected_ids": ["p02_r4"],
        "expected_count": 1,
    },


    # ============================================================
    # 3. IDENTIFIER -> PARAMETER
    # ============================================================

    {
        "id": "I002",
        "category": "identifier_parameter",
        "query": (
            "What is the actuator type for "
            "material code TC9765132581?"
        ),
        "expected_mode": "entity_scoped_lookup",
        "expected_ids": ["p02_r40"],
        "expected_count": 1,
    },

    {
        "id": "I003",
        "category": "identifier_parameter",
        "query": (
            "What is the body material for "
            "material code TC9765132581?"
        ),
        "expected_mode": "entity_scoped_lookup",
        "expected_contains": ["p02_r17_inlet"],
        "min_count": 1,
    },

    {
        "id": "I004",
        "category": "identifier_parameter",
        "query": (
            "What is the line size for "
            "material code TC9765132581?"
        ),
        "expected_mode": "entity_scoped_lookup",
        "min_count": 1,
    },

    {
        "id": "I005",
        "category": "identifier_parameter",
        "query": (
            "What is the inlet line size for "
            "material code TC9765132581?"
        ),
        "expected_mode": "entity_scoped_lookup",
        "expected_ids": ["p02_r8_inlet"],
        "expected_count": 1,
    },

    {
        "id": "I006",
        "category": "identifier_parameter",
        "query": (
            "What is the outlet line size for "
            "material code TC9765132581?"
        ),
        "expected_mode": "entity_scoped_lookup",
        "expected_ids": ["p02_r8_outlet"],
        "expected_count": 1,
    },

    {
        "id": "I007",
        "category": "identifier_parameter",
        "query": (
            "What is the service for "
            "material code TC9765132581?"
        ),
        "expected_mode": "entity_scoped_lookup",
        "min_count": 1,
    },


    # ============================================================
    # 4. SIDE-SPECIFIC QUERIES
    # ============================================================

    {
        "id": "S001",
        "category": "side_specific",
        "query": "What is the inlet steam flow?",
        "expected_mode": "global_hybrid_v3",
        "expected_contains": ["p01_r12_inlet"],
    },

    {
        "id": "S002",
        "category": "side_specific",
        "query": "What is the outlet steam flow?",
        "expected_mode": "global_hybrid_v3",
        "expected_contains": ["p01_r12_outlet"],
    },

    {
        "id": "S003",
        "category": "side_specific",
        "query": "What is the inlet line size?",
        "expected_mode": "global_hybrid_v3",
        "expected_contains": ["p01_r8_inlet"],
    },

    {
        "id": "S004",
        "category": "side_specific",
        "query": "What is the outlet line size?",
        "expected_mode": "global_hybrid_v3",
        "expected_contains": ["p01_r8_outlet"],
    },

    {
        "id": "S005",
        "category": "side_specific",
        "query": (
            "What is the piping class "
            "of the outlet steam?"
        ),
        "expected_mode": "global_hybrid_v3",
        "expected_contains": ["p01_r8_outlet"],
    },

    {
        "id": "S006",
        "category": "side_specific",
        "query": (
            "What is the piping class "
            "of the inlet steam?"
        ),
        "expected_mode": "global_hybrid_v3",
        "expected_contains": ["p01_r8_inlet"],
    },


    # ============================================================
    # 5. HIERARCHICAL PARAMETERS
    # ============================================================

    {
        "id": "H001",
        "category": "hierarchy",
        "query": "What is the piping class?",
        "expected_mode": "global_hybrid_v3",
        "expected_contains": ["p01_r8_outlet"],
    },

    {
        "id": "H002",
        "category": "hierarchy",
        "query": "What is the stem plug material?",
        "expected_mode": "global_hybrid_v3",
        "expected_contains": ["p01_r17_inlet"],
    },

    {
        "id": "H003",
        "category": "hierarchy",
        "query": "What is the cooling fins information?",
        "expected_mode": "global_hybrid_v3",
        "min_count": 1,
    },

    {
        "id": "H004",
        "category": "hierarchy",
        "query": "What is the spring range?",
        "expected_mode": "global_hybrid_v3",
        "min_count": 1,
    },

    {
        "id": "H005",
        "category": "hierarchy",
        "query": "What is the pressure gauge?",
        "expected_mode": "global_hybrid_v3",
        "min_count": 1,
    },

    {
        "id": "H006",
        "category": "hierarchy",
        "query": "What is the enclosure protection?",
        "expected_mode": "global_hybrid_v3",
        "min_count": 1,
    },


    # ============================================================
    # 6. SECTION QUERIES
    # ============================================================

    {
        "id": "SEC001",
        "category": "section",
        "query": "What are the water conditions?",
        "expected_mode": "section_lookup",
        "expected_count": 238,
    },

    {
        "id": "SEC002",
        "category": "section",
        "query": "Show me the steam conditions.",
        "expected_mode": "section_lookup",
        "min_count": 1,
    },

    {
        "id": "SEC003",
        "category": "section",
        "query": (
            "Give me all information "
            "under water conditions."
        ),
        "expected_mode": "section_lookup",
        "expected_count": 238,
    },

    {
        "id": "SEC004",
        "category": "section",
        "query": (
            "List the parameters "
            "in water conditions."
        ),
        "expected_mode": "section_lookup",
        "expected_count": 238,
    },


    # ============================================================
    # 7. GLOBAL SCHEMA QUERIES
    # ============================================================

    {
        "id": "G001",
        "category": "global_schema",
        "query": "What is the material code?",
        "expected_mode": "global_hybrid_v3",
        "expected_contains": ["p01_r58"],
    },

    {
        "id": "G002",
        "category": "global_schema",
        "query": "What is the body material?",
        "expected_mode": "global_hybrid_v3",
        "expected_contains": ["p01_r17_inlet"],
    },

    {
        "id": "G003",
        "category": "global_schema",
        "query": "What is the actuator type?",
        "expected_mode": "global_hybrid_v3",
        "min_count": 1,
    },

    {
        "id": "G004",
        "category": "global_schema",
        "query": "What is the service?",
        "expected_mode": "global_hybrid_v3",
        "min_count": 1,
    },

    {
        "id": "G005",
        "category": "global_schema",
        "query": "What is the manufacturer?",
        "expected_mode": "global_hybrid_v3",
        "min_count": 1,
    },

    {
        "id": "G006",
        "category": "global_schema",
        "query": "What is the temperature?",
        "expected_mode": "global_hybrid_v3",
        "min_count": 1,
    },

    {
        "id": "G007",
        "category": "global_schema",
        "query": "What is the pressure?",
        "expected_mode": "global_hybrid_v3",
        "min_count": 1,
    },

    {
        "id": "G008",
        "category": "global_schema",
        "query": "What is the line material?",
        "expected_mode": "global_hybrid_v3",
        "min_count": 1,
    },


    # ============================================================
    # 8. MULTI-INTENT
    # ============================================================

    {
        "id": "M001",
        "category": "multi_intent",
        "query": (
            "What are the inlet flow "
            "and outlet flow?"
        ),
        "expected_mode": "multi",
        "min_count": 2,
    },

    {
        "id": "M002",
        "category": "multi_intent",
        "query": (
            "What are the inlet line size "
            "and outlet piping class?"
        ),
        "expected_mode": "multi",
        "min_count": 2,
    },

    {
        "id": "M003",
        "category": "multi_intent",
        "query": (
            "Give me the inlet flow, "
            "outlet flow, outlet piping class "
            "and material code."
        ),
        "expected_mode": "multi",
        "min_count": 4,
    },

    {
        "id": "M004",
        "category": "multi_intent",
        "query": (
            "What are the actuator type "
            "and body material?"
        ),
        "expected_mode": "multi",
        "min_count": 2,
    },

    {
        "id": "M005",
        "category": "multi_intent",
        "query": (
            "Give me the line size, "
            "piping class and actuator type."
        ),
        "expected_mode": "multi",
        "min_count": 3,
    },


    # ============================================================
    # 9. IDENTIFIER + MULTI-INTENT
    # ============================================================

    {
        "id": "IM001",
        "category": "identifier_multi_intent",
        "query": (
            "For material code TC9765132581, "
            "what are the actuator type "
            "and body material?"
        ),
        "expected_mode": "multi",
        "min_count": 2,
    },

    {
        "id": "IM002",
        "category": "identifier_multi_intent",
        "query": (
            "For material code TC9765132581, "
            "give me the inlet line size "
            "and outlet line size."
        ),
        "expected_mode": "multi",
        "min_count": 2,
    },


    # ============================================================
    # 10. AMBIGUOUS / SHORT
    # ============================================================

    {
        "id": "A001",
        "category": "ambiguous",
        "query": "actuator",
        "allow_empty": True,
    },

    {
        "id": "A002",
        "category": "ambiguous",
        "query": "size",
        "allow_empty": True,
    },

    {
        "id": "A003",
        "category": "ambiguous",
        "query": "type",
        "allow_empty": True,
    },

    {
        "id": "A004",
        "category": "ambiguous",
        "query": "material",
        "allow_empty": True,
    },

    {
        "id": "A005",
        "category": "ambiguous",
        "query": "what is the size?",
        "allow_empty": True,
    },


    # ============================================================
    # 11. NEGATIVE / UNSUPPORTED
    # ============================================================

    {
        "id": "N001",
        "category": "negative",
        "query": (
            "What is the turbine shaft diameter?"
        ),
        "expected_empty": True,
    },

    {
        "id": "N002",
        "category": "negative",
        "query": (
            "What is the motor bearing temperature?"
        ),
        "expected_empty": True,
    },

    {
        "id": "N003",
        "category": "negative",
        "query": (
            "What is the gearbox oil pressure?"
        ),
        "expected_empty": True,
    },

    {
        "id": "N004",
        "category": "negative",
        "query": (
            "What is the compressor vibration level?"
        ),
        "expected_empty": True,
    },

    {
        "id": "N005",
        "category": "negative",
        "query": (
            "What is the pump shaft diameter?"
        ),
        "expected_empty": True,
    },


    # ============================================================
    # 12. UNKNOWN IDENTIFIERS
    # ============================================================

    {
        "id": "U001",
        "category": "unknown_identifier",
        "query": (
            "What is the actuator type for "
            "material code UNKNOWN123?"
        ),
        "expected_empty": True,
    },

    {
        "id": "U002",
        "category": "unknown_identifier",
        "query": (
            "What is the item of "
            "material code INVALID999?"
        ),
        "expected_empty": True,
    },


    # ============================================================
    # 13. FORMAT VARIATIONS
    # ============================================================

    {
        "id": "F001",
        "category": "format_variation",
        "query": "WHAT IS THE MATERIAL CODE?",
        "expected_mode": "global_hybrid_v3",
        "expected_contains": ["p01_r58"],
    },

    {
        "id": "F002",
        "category": "format_variation",
        "query": "what is the actuator type?",
        "expected_mode": "global_hybrid_v3",
        "min_count": 1,
    },

    {
        "id": "F003",
        "category": "format_variation",
        "query": (
            "What is the ITEM "
            "of TC9765132581?"
        ),
        "expected_mode": "entity_scoped_lookup",
        "expected_ids": ["p02_r4"],
        "expected_count": 1,
    },


    # ============================================================
    # 14. PROVENANCE
    # ============================================================

    {
        "id": "P001",
        "category": "provenance",
        "query": (
            "What is the actuator type for "
            "material code TC9765132581?"
        ),
        "expected_mode": "entity_scoped_lookup",
        "expected_ids": ["p02_r40"],
        "expected_count": 1,
        "check_provenance": True,
    },

    {
        "id": "P002",
        "category": "provenance",
        "query": (
            "What is the inlet line size for "
            "material code TC9765132581?"
        ),
        "expected_mode": "entity_scoped_lookup",
        "expected_ids": ["p02_r8_inlet"],
        "expected_count": 1,
        "check_provenance": True,
    },
]


# ================================================================
# JSON LOADER
# ================================================================


def load_json_file(
    path: Path,
) -> Any:
    """
    Load a JSON artifact with a useful error message.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Required artifact not found:\n{path}"
        )

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except json.JSONDecodeError as exc:

        raise ValueError(
            f"Invalid JSON file:\n"
            f"{path}\n"
            f"Error: {exc}"
        ) from exc


# ================================================================
# ARTIFACT NORMALIZATION
# ================================================================


def normalize_retrieval_records(
    data: Any,
) -> Any:
    """
    Normalize retrieval_enriched.json.

    Supports either:

        [
            {...},
            {...}
        ]

    or:

        {
            "records": [...]
        }

    without changing the adapter's expected record structure.
    """

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        if isinstance(
            data.get("records"),
            list,
        ):
            return data["records"]

        if isinstance(
            data.get("retrieval_records"),
            list,
        ):
            return data[
                "retrieval_records"
            ]

    raise ValueError(
        "retrieval_enriched.json does not contain "
        "a supported record list."
    )


# ================================================================
# RESULT HELPERS
# ================================================================


def get_result_records(
    context_result: Dict[str, Any],
) -> List[Dict[str, Any]]:

    records = context_result.get(
        "records",
        []
    )

    if not isinstance(
        records,
        list,
    ):
        return []

    return records


def get_record_ids(
    context_result: Dict[str, Any],
) -> List[str]:

    return [
        record.get("record_id")
        for record in get_result_records(
            context_result
        )
        if record.get("record_id")
    ]


def check_unique_ids(
    context_result: Dict[str, Any],
) -> bool:

    ids = get_record_ids(
        context_result
    )

    return len(ids) == len(set(ids))


def check_context_blocks(
    context_result: Dict[str, Any],
) -> List[str]:

    errors = []

    records = get_result_records(
        context_result
    )

    blocks = context_result.get(
        "context_blocks",
        []
    )

    if not isinstance(
        blocks,
        list,
    ):

        errors.append(
            "context_blocks is not a list"
        )

        return errors

    if len(blocks) != len(records):

        errors.append(
            f"context_blocks count "
            f"({len(blocks)}) != "
            f"records count "
            f"({len(records)})"
        )

    return errors


def check_context_text(
    context_result: Dict[str, Any],
) -> bool:

    context_text = context_result.get(
        "context_text",
        ""
    )

    return (
        isinstance(context_text, str)
        and bool(context_text.strip())
    )


def check_provenance(
    context_result: Dict[str, Any],
) -> List[str]:

    errors = []

    for record in get_result_records(
        context_result
    ):

        record_id = record.get(
            "record_id"
        )

        if not record_id:

            errors.append(
                "missing record_id"
            )

        if record.get("page") is None:

            errors.append(
                f"{record_id}: missing page"
            )

        if not record.get("parameter"):

            errors.append(
                f"{record_id}: missing parameter"
            )

        if "values" not in record:

            errors.append(
                f"{record_id}: missing values"
            )

    return errors


def check_context_block_provenance(
    context_result: Dict[str, Any],
) -> List[str]:

    errors = []

    for block in context_result.get(
        "context_blocks",
        []
    ):

        record_id = block.get(
            "record_id"
        )

        if not record_id:

            errors.append(
                "context block missing record_id"
            )

        if block.get("page") is None:

            errors.append(
                f"{record_id}: block missing page"
            )

        if not block.get("parameter"):

            errors.append(
                f"{record_id}: block missing parameter"
            )

        if "values" not in block:

            errors.append(
                f"{record_id}: block missing values"
            )

        if not block.get("text"):

            errors.append(
                f"{record_id}: block missing text"
            )

    return errors


# ================================================================
# SINGLE TEST
# ================================================================


def run_test(
    adapter: IntentRetrievalAdapter,
    builder: ContextBuilder,
    test_case: Dict[str, Any],
) -> Dict[str, Any]:

    test_id = test_case["id"]
    query = test_case["query"]

    failures = []

    retrieval_result = None
    context_result = None

    try:

        # --------------------------------------------------------
        # Phase 3.2
        # --------------------------------------------------------

        retrieval_result = adapter.search(
            query
        )

    except Exception as exc:

        return {
            "id": test_id,
            "category": test_case[
                "category"
            ],
            "query": query,
            "passed": False,
            "stage": "retrieval",
            "record_count": 0,
            "record_ids": [],
            "retrieval_mode": None,
            "failures": [
                f"{type(exc).__name__}: {exc}"
            ],
        }

    # ------------------------------------------------------------
    # Phase 4
    # ------------------------------------------------------------

    try:

        context_result = builder.build(
            retrieval_result
        )

    except Exception as exc:

        return {
            "id": test_id,
            "category": test_case[
                "category"
            ],
            "query": query,
            "passed": False,
            "stage": "context",
            "record_count": 0,
            "record_ids": [],
            "retrieval_mode": None,
            "failures": [
                f"{type(exc).__name__}: {exc}"
            ],
        }

    # ------------------------------------------------------------
    # Extract result information
    # ------------------------------------------------------------

    records = get_result_records(
        context_result
    )

    record_ids = get_record_ids(
        context_result
    )

    retrieval_metadata = context_result.get(
        "retrieval",
        {}
    )

    actual_mode = retrieval_metadata.get(
        "retrieval_mode"
    )

    actual_intent_type = retrieval_metadata.get(
        "intent_type"
    )

    actual_count = len(records)

    # ------------------------------------------------------------
    # Expected mode
    # ------------------------------------------------------------

    expected_mode = test_case.get(
        "expected_mode"
    )

    if (
        expected_mode
        and expected_mode != "multi"
        and actual_mode != expected_mode
    ):

        failures.append(
            f"retrieval mode: "
            f"expected={expected_mode}, "
            f"actual={actual_mode}"
        )

    # ------------------------------------------------------------
    # Expected IDs
    # ------------------------------------------------------------

    expected_ids = test_case.get(
        "expected_ids"
    )

    if expected_ids is not None:

        if set(record_ids) != set(
            expected_ids
        ):

            failures.append(
                f"record IDs: "
                f"expected={expected_ids}, "
                f"actual={record_ids}"
            )

    # ------------------------------------------------------------
    # Expected contains
    # ------------------------------------------------------------

    expected_contains = test_case.get(
        "expected_contains",
        []
    )

    for expected_id in expected_contains:

        if expected_id not in record_ids:

            failures.append(
                f"missing expected record: "
                f"{expected_id}"
            )

    # ------------------------------------------------------------
    # Exact count
    # ------------------------------------------------------------

    expected_count = test_case.get(
        "expected_count"
    )

    if (
        expected_count is not None
        and actual_count != expected_count
    ):

        failures.append(
            f"record count: "
            f"expected={expected_count}, "
            f"actual={actual_count}"
        )

    # ------------------------------------------------------------
    # Minimum count
    # ------------------------------------------------------------

    min_count = test_case.get(
        "min_count"
    )

    if (
        min_count is not None
        and actual_count < min_count
    ):

        failures.append(
            f"record count below minimum: "
            f"expected>={min_count}, "
            f"actual={actual_count}"
        )

    # ------------------------------------------------------------
    # Expected empty
    # ------------------------------------------------------------

    if test_case.get(
        "expected_empty"
    ):

        if records:

            failures.append(
                f"expected empty context, "
                f"received {actual_count} records"
            )

    # ------------------------------------------------------------
    # Ambiguous query
    # ------------------------------------------------------------

    if test_case.get(
        "allow_empty"
    ):

        # No failure is generated solely because
        # the query returns zero records.

        # However, if records exist, they must still
        # satisfy basic context integrity.

        pass

    # ------------------------------------------------------------
    # Unique IDs
    # ------------------------------------------------------------

    if not check_unique_ids(
        context_result
    ):

        failures.append(
            "duplicate record IDs found"
        )

    # ------------------------------------------------------------
    # Context blocks
    # ------------------------------------------------------------

    failures.extend(
        check_context_blocks(
            context_result
        )
    )

    # ------------------------------------------------------------
    # Context text
    # ------------------------------------------------------------

    if records:

        if not check_context_text(
            context_result
        ):

            failures.append(
                "records exist but context_text "
                "is empty"
            )

    # ------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------

    if test_case.get(
        "check_provenance"
    ):

        failures.extend(
            check_provenance(
                context_result
            )
        )

        failures.extend(
            check_context_block_provenance(
                context_result
            )
        )

    # ------------------------------------------------------------
    # Statistics consistency
    # ------------------------------------------------------------

    statistics = context_result.get(
        "statistics",
        {}
    )

    unique_count = statistics.get(
        "unique"
    )

    if (
        unique_count is not None
        and unique_count != actual_count
    ):

        failures.append(
            f"statistics.unique={unique_count} "
            f"does not match actual count={actual_count}"
        )

    # ------------------------------------------------------------
    # Result
    # ------------------------------------------------------------

    return {
        "id": test_id,
        "category": test_case[
            "category"
        ],
        "query": query,
        "passed": len(failures) == 0,
        "stage": (
            "passed"
            if not failures
            else "validation"
        ),
        "record_count": actual_count,
        "record_ids": record_ids,
        "retrieval_mode": actual_mode,
        "intent_type": actual_intent_type,
        "failures": failures,
    }


# ================================================================
# ARTIFACT VALIDATION
# ================================================================


def validate_artifacts(
    schema_catalog: Any,
    entity_index: Any,
    retrieval_records: Any,
) -> None:
    """
    Basic sanity checks before running 60+ tests.
    """

    if not isinstance(
        schema_catalog,
        dict,
    ):

        raise ValueError(
            "schema_catalog.json must contain a JSON object."
        )

    if not isinstance(
        entity_index,
        dict,
    ):

        raise ValueError(
            "entity_index.json must contain a JSON object."
        )

    if not isinstance(
        retrieval_records,
        list,
    ):

        raise ValueError(
            "retrieval_enriched.json must resolve "
            "to a list of records."
        )

    if not retrieval_records:

        raise ValueError(
            "retrieval_enriched.json contains zero records."
        )


# ================================================================
# MAIN
# ================================================================


def main():

    print()
    print("=" * 100)
    print("PHASE 4 — DETAILED CONTEXT INTEGRATION TEST")
    print("=" * 100)

    # ============================================================
    # ARTIFACT CHECK
    # ============================================================

    print()
    print("Checking Phase 3.2 artifacts...")
    print()

    print(
        f"Schema catalog    : "
        f"{SCHEMA_CATALOG_PATH}"
    )

    print(
        f"Entity index      : "
        f"{ENTITY_INDEX_PATH}"
    )

    print(
        f"Retrieval records : "
        f"{RETRIEVAL_RECORDS_PATH}"
    )

    try:

        schema_catalog = load_json_file(
            SCHEMA_CATALOG_PATH
        )

        entity_index = load_json_file(
            ENTITY_INDEX_PATH
        )

        retrieval_records_raw = (
            load_json_file(
                RETRIEVAL_RECORDS_PATH
            )
        )

        retrieval_records = (
            normalize_retrieval_records(
                retrieval_records_raw
            )
        )

        validate_artifacts(
            schema_catalog,
            entity_index,
            retrieval_records,
        )

    except Exception as exc:

        print()
        print("=" * 100)
        print("ARTIFACT INITIALIZATION FAILED")
        print("=" * 100)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()

        return 1

    print()
    print(
        f"Retrieval records loaded : "
        f"{len(retrieval_records)}"
    )

    # ============================================================
    # INITIALIZE ADAPTER
    # ============================================================

    print()
    print("Initializing Phase 3.2 retrieval adapter...")

    try:

        adapter = IntentRetrievalAdapter(
            schema_catalog=schema_catalog,
            entity_index=entity_index,
            retrieval_records=retrieval_records,
        )

    except Exception as exc:

        print()
        print("=" * 100)
        print("RETRIEVAL ADAPTER INITIALIZATION FAILED")
        print("=" * 100)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()

        return 1

    print(
        "Phase 3.2 adapter initialized."
    )

    # ============================================================
    # INITIALIZE CONTEXT BUILDER
    # ============================================================

    builder = ContextBuilder()

    print(
        "Phase 4 context builder initialized."
    )

    # ============================================================
    # TEST INFORMATION
    # ============================================================

    total = len(
        TEST_CASES
    )

    print()
    print("=" * 100)
    print(
        f"TOTAL TEST CASES: {total}"
    )
    print("=" * 100)

    # ============================================================
    # RUN TESTS
    # ============================================================

    results = []

    for index, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):

        result = run_test(
            adapter=adapter,
            builder=builder,
            test_case=test_case,
        )

        results.append(
            result
        )

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print(
            f"[{index:02d}/{total:02d}] "
            f"{status:<4} "
            f"{result['id']:<6} "
            f"{result['category']:<25} "
            f"records={result['record_count']:<4} "
            f"mode={str(result['retrieval_mode']):<24}"
        )

        if not result["passed"]:

            for failure in result[
                "failures"
            ]:

                print(
                    f"       └─ {failure}"
                )

    # ============================================================
    # SUMMARY
    # ============================================================

    passed = sum(
        1
        for result in results
        if result["passed"]
    )

    failed = total - passed

    print()
    print("=" * 100)
    print("TEST SUMMARY")
    print("=" * 100)

    print(
        f"Total tests : {total}"
    )

    print(
        f"Passed      : {passed}"
    )

    print(
        f"Failed      : {failed}"
    )

    if total > 0:

        pass_rate = (
            passed / total
        ) * 100

        print(
            f"Pass rate   : {pass_rate:.2f}%"
        )

    # ============================================================
    # CATEGORY SUMMARY
    # ============================================================

    print()
    print("-" * 100)
    print("CATEGORY SUMMARY")
    print("-" * 100)

    categories: Dict[
        str,
        Dict[str, int]
    ] = {}

    for result in results:

        category = result[
            "category"
        ]

        if category not in categories:

            categories[
                category
            ] = {
                "total": 0,
                "passed": 0,
            }

        categories[
            category
        ]["total"] += 1

        if result["passed"]:

            categories[
                category
            ]["passed"] += 1

    for category, stats in categories.items():

        category_total = stats[
            "total"
        ]

        category_passed = stats[
            "passed"
        ]

        category_rate = (
            category_passed
            / category_total
            * 100
        )

        print(
            f"{category:<30} "
            f"{category_passed:>3}/"
            f"{category_total:<3} "
            f"{category_rate:>6.2f}%"
        )

    # ============================================================
    # FAILED TEST DETAILS
    # ============================================================

    failed_results = [
        result
        for result in results
        if not result["passed"]
    ]

    if failed_results:

        print()
        print("=" * 100)
        print("FAILED TEST DETAILS")
        print("=" * 100)

        for result in failed_results:

            print()
            print(
                f"Test ID       : "
                f"{result['id']}"
            )

            print(
                f"Category      : "
                f"{result['category']}"
            )

            print(
                f"Stage         : "
                f"{result['stage']}"
            )

            print(
                f"Query         : "
                f"{result['query']}"
            )

            print(
                f"Retrieval mode: "
                f"{result['retrieval_mode']}"
            )

            print(
                f"Record count  : "
                f"{result['record_count']}"
            )

            print(
                f"Record IDs    : "
                f"{result['record_ids']}"
            )

            for failure in result[
                "failures"
            ]:

                print(
                    f"  - {failure}"
                )

    # ============================================================
    # FINAL STATUS
    # ============================================================

    print()
    print("=" * 100)

    if failed == 0:

        print(
            "PHASE 4 INTEGRATION TEST: PASS"
        )

        print()
        print(
            "Phase 3.2 Retrieval Adapter "
            "and Phase 4 Context Builder "
            "are compatible."
        )

    else:

        print(
            "PHASE 4 INTEGRATION TEST: FAIL"
        )

        print()
        print(
            "Do not freeze Phase 4 yet."
        )

        print(
            "Review whether each failure belongs "
            "to retrieval/query understanding "
            "or context construction."
        )

    print("=" * 100)
    print()

    return 0 if failed == 0 else 1


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )