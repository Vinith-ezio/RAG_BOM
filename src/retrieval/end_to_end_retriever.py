import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_INPUT = (
    r"E:\PDF Ingestion\data\output\retrieval_enriched.json"
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(value: Any) -> str:
    """
    Normalize text only for comparison.

    Source values are never modified.
    """

    if value is None:
        return ""

    text = str(value).strip().lower()

    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# DATASET LOADER
# ============================================================

def load_records(path: str) -> List[Dict[str, Any]]:
    """
    Load retrieval_enriched.json.
    """

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}"
        )

    with file_path.open(
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    records = data.get("records", [])

    if not isinstance(records, list):
        raise ValueError(
            "Invalid dataset: 'records' must be a list."
        )

    return records


# ============================================================
# QUERY PARSER
# ============================================================

def normalize_query_text(value: Any) -> str:
    """
    Normalize natural-language query.
    """

    if value is None:
        return ""

    text = str(value).strip().lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# DATASET VOCABULARY
# ============================================================

def collect_unique_values(
    records: List[Dict[str, Any]],
    field: str
) -> List[str]:

    values = set()

    for record in records:

        value = record.get(field)

        if value is None:
            continue

        value = str(value).strip()

        if value:
            values.add(value)

    return sorted(
        values,
        key=lambda x: len(x),
        reverse=True
    )


# ============================================================
# ITEM DETECTION
# ============================================================

def detect_item(
    query: str,
    records: List[Dict[str, Any]]
) -> Optional[str]:

    normalized_query = normalize_query_text(
        query
    )

    items = collect_unique_values(
        records,
        "item"
    )

    for item in items:

        normalized_item = normalize(
            item
        )

        if (
            normalized_item
            and normalized_item in normalized_query
        ):
            return item

    return None


# ============================================================
# SIDE DETECTION
# ============================================================

def detect_side(
    query: str
) -> Optional[str]:

    text = normalize_query_text(
        query
    )

    if re.search(
        r"\binlet\s+steam\b",
        text
    ):
        return "inlet"

    if re.search(
        r"\boutlet\s+steam\b",
        text
    ):
        return "outlet"

    if re.search(
        r"\binlet\b",
        text
    ):
        return "inlet"

    if re.search(
        r"\boutlet\b",
        text
    ):
        return "outlet"

    return None


# ============================================================
# PARAMETER ALIASES
# ============================================================

PARAMETER_ALIASES = {

    "steam flow": "Flow",
    "water flow": "Flow",
    "flow rate": "Flow",
    "flow": "Flow",

    "line diameter": "Line Size",
    "pipe size": "Line Size",
    "piping size": "Line Size",
    "line size": "Line Size",

    "body material": "Body Material",
    "body": "Body Material",

    "material code": "Material Code",

    "steam pressure": "Pressure",
    "water pressure": "Pressure",
    "pressure": "Pressure",

    "steam temperature": "Temperature",
    "water temperature": "Temperature",
    "temperature": "Temperature",

    "actuator": "Actuator",
    "positioner": "Positioner",

    "tag number": "Tag Number",
    "tag": "Tag Number",

    "insulation code": "Insulation Code",
}


def detect_parameter(
    query: str,
    records: List[Dict[str, Any]]
) -> Optional[str]:

    text = normalize_query_text(
        query
    )

    # --------------------------------------------------------
    # Alias matching
    # --------------------------------------------------------

    aliases = sorted(
        PARAMETER_ALIASES.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )

    for alias, parameter in aliases:

        if re.search(
            rf"\b{re.escape(alias)}\b",
            text
        ):
            return parameter

    # --------------------------------------------------------
    # Dataset parameter matching
    # --------------------------------------------------------

    parameters = collect_unique_values(
        records,
        "parameter"
    )

    for parameter in parameters:

        normalized_parameter = normalize(
            parameter
        )

        if (
            normalized_parameter
            and normalized_parameter in text
        ):
            return parameter

    return None


# ============================================================
# SUB-PARAMETER
# ============================================================

SUB_PARAMETER_ALIASES = {

    "piping class": "Piping Class",

    "stem plug material":
        "Stem / Plug Material",

    "stem / plug material":
        "Stem / Plug Material",

    "minimum normal maximum unit":
        "Min. / Norm. / Max. / Unit",

    "min norm max unit":
        "Min. / Norm. / Max. / Unit",
}


# ============================================================
# SUB-PARAMETER → PARENT PARAMETER
# ============================================================

SUB_PARAMETER_PARENT = {

    "Piping Class": "Line Size",

    "Stem / Plug Material":
        "Body Material",
}


def detect_sub_parameter(
    query: str,
    records: List[Dict[str, Any]]
) -> Optional[str]:

    text = normalize_query_text(
        query
    )

    # --------------------------------------------------------
    # Alias detection
    # --------------------------------------------------------

    aliases = sorted(
        SUB_PARAMETER_ALIASES.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )

    for alias, sub_parameter in aliases:

        if alias in text:
            return sub_parameter

    # --------------------------------------------------------
    # Dataset matching
    # --------------------------------------------------------

    sub_parameters = collect_unique_values(
        records,
        "sub_parameter"
    )

    for sub_parameter in sub_parameters:

        normalized = normalize(
            sub_parameter
        )

        if (
            normalized
            and normalized in text
        ):
            return sub_parameter

    return None


# ============================================================
# SECTION
# ============================================================

def detect_section(
    query: str,
    records: List[Dict[str, Any]]
) -> Optional[str]:

    text = normalize_query_text(
        query
    )

    sections = collect_unique_values(
        records,
        "section"
    )

    for section in sections:

        normalized = normalize(
            section
        )

        if (
            normalized
            and normalized in text
        ):
            return section

    # Natural-language aliases

    if "steam" in text:

        for section in sections:

            if normalize(section) == (
                "steam conditions"
            ):
                return section

    if "water" in text:

        for section in sections:

            if normalize(section) == (
                "water conditions"
            ):
                return section

    return None


# ============================================================
# SUBSECTION
# ============================================================

def detect_subsection(
    query: str,
    records: List[Dict[str, Any]]
) -> Optional[str]:

    text = normalize_query_text(
        query
    )

    subsections = collect_unique_values(
        records,
        "subsection"
    )

    for subsection in subsections:

        normalized = normalize(
            subsection
        )

        if (
            normalized
            and normalized in text
        ):
            return subsection

    if (
        "inlet steam" in text
        or "outlet steam" in text
    ):

        for subsection in subsections:

            if normalize(subsection) == (
                "inlet steam / outlet steam"
            ):
                return subsection

    return None


# ============================================================
# VALUE FIELD
# ============================================================

VALUE_FIELD_ALIASES = {

    "piping class": "class",

    "minimum": "min",
    "min": "min",

    "normal": "normal",
    "norm": "normal",

    "maximum": "max",
    "max": "max",

    "unit": "unit",

    "class": "class",

    "value": "value",
}


def detect_value_field(
    query: str
) -> Optional[str]:

    text = normalize_query_text(
        query
    )

    aliases = sorted(
        VALUE_FIELD_ALIASES.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )

    for alias, field in aliases:

        if re.search(
            rf"\b{re.escape(alias)}\b",
            text
        ):
            return field

    return None


# ============================================================
# QUERY PARSER
# ============================================================

def parse_query(
    query: str,
    records: List[Dict[str, Any]]
) -> Dict[str, Any]:

    item = detect_item(
        query,
        records
    )

    section = detect_section(
        query,
        records
    )

    subsection = detect_subsection(
        query,
        records
    )

    parameter = detect_parameter(
        query,
        records
    )

    sub_parameter = detect_sub_parameter(
        query,
        records
    )

    side = detect_side(
        query
    )

    value_field = detect_value_field(
        query
    )

    # --------------------------------------------------------
    # Resolve parent parameter
    # --------------------------------------------------------

    if sub_parameter and not parameter:

        parent_parameter = (
            SUB_PARAMETER_PARENT.get(
                sub_parameter
            )
        )

        if parent_parameter:
            parameter = parent_parameter

    return {

        "original_query": query,

        "item": item,

        "section": section,

        "subsection": subsection,

        "parameter": parameter,

        "sub_parameter": sub_parameter,

        "side": side,

        "value_field": value_field
    }


# ============================================================
# STRUCTURED RETRIEVAL
# ============================================================

def field_matches(
    record_value: Any,
    query_value: Any
) -> bool:

    if query_value is None:
        return True

    if record_value is None:
        return False

    return (
        normalize(record_value)
        == normalize(query_value)
    )


def retrieve_records(
    records: List[Dict[str, Any]],
    structured_query: Dict[str, Any],
    limit: int = 10
) -> List[Dict[str, Any]]:

    results = []

    for record in records:

        # ----------------------------------------------------
        # Item
        # ----------------------------------------------------

        if not field_matches(
            record.get("item"),
            structured_query.get("item")
        ):
            continue

        # ----------------------------------------------------
        # Section
        # ----------------------------------------------------

        if not field_matches(
            record.get("section"),
            structured_query.get("section")
        ):
            continue

        # ----------------------------------------------------
        # Subsection
        # ----------------------------------------------------

        if not field_matches(
            record.get("subsection"),
            structured_query.get("subsection")
        ):
            continue

        # ----------------------------------------------------
        # Parameter
        # ----------------------------------------------------

        if not field_matches(
            record.get("parameter"),
            structured_query.get("parameter")
        ):
            continue

        # ----------------------------------------------------
        # Sub-parameter
        # ----------------------------------------------------

        if not field_matches(
            record.get("sub_parameter"),
            structured_query.get("sub_parameter")
        ):
            continue

        # ----------------------------------------------------
        # Side
        # ----------------------------------------------------

        if not field_matches(
            record.get("side"),
            structured_query.get("side")
        ):
            continue

        results.append(record)

        if len(results) >= limit:
            break

    return results


# ============================================================
# VALUE EXTRACTION
# ============================================================

def extract_requested_value(
    record: Dict[str, Any],
    value_field: Optional[str]
) -> Any:

    values = record.get(
        "values",
        {}
    )

    if not isinstance(values, dict):
        return None

    # If query explicitly requested a field
    if value_field:

        return values.get(
            value_field
        )

    # Otherwise return complete structured values
    return values


# ============================================================
# RESULT DISPLAY
# ============================================================

def print_result(
    query: str,
    parsed_query: Dict[str, Any],
    results: List[Dict[str, Any]]
) -> None:

    print()
    print("=" * 72)
    print("END-TO-END RETRIEVAL RESULT")
    print("=" * 72)

    print(
        f"Query : {query}"
    )

    print("-" * 72)

    print("PARSED QUERY")
    print()

    print(
        f"Item          : "
        f"{parsed_query.get('item')}"
    )

    print(
        f"Section       : "
        f"{parsed_query.get('section')}"
    )

    print(
        f"Subsection    : "
        f"{parsed_query.get('subsection')}"
    )

    print(
        f"Parameter     : "
        f"{parsed_query.get('parameter')}"
    )

    print(
        f"Sub-parameter : "
        f"{parsed_query.get('sub_parameter')}"
    )

    print(
        f"Side          : "
        f"{parsed_query.get('side')}"
    )

    print(
        f"Value Field   : "
        f"{parsed_query.get('value_field')}"
    )

    print("-" * 72)

    print(
        f"Matches       : {len(results)}"
    )

    print("-" * 72)

    if not results:

        print("NO MATCHING RECORDS FOUND.")

        print("=" * 72)

        return

    for index, record in enumerate(
        results,
        start=1
    ):

        print()
        print(f"[{index}]")

        print(
            f"Record ID     : "
            f"{record.get('record_id')}"
        )

        print(
            f"Page          : "
            f"{record.get('page')}"
        )

        print(
            f"Row           : "
            f"{record.get('row_number')}"
        )

        print(
            f"Item          : "
            f"{record.get('item')}"
        )

        print(
            f"Section       : "
            f"{record.get('section')}"
        )

        print(
            f"Subsection    : "
            f"{record.get('subsection')}"
        )

        print(
            f"Parameter     : "
            f"{record.get('parameter')}"
        )

        print(
            f"Sub-parameter : "
            f"{record.get('sub_parameter')}"
        )

        print(
            f"Side          : "
            f"{record.get('side')}"
        )

        requested_value = (
            extract_requested_value(
                record,
                parsed_query.get(
                    "value_field"
                )
            )
        )

        print(
            f"Requested Value: "
            f"{requested_value}"
        )

        print(
            f"All Values     : "
            f"{record.get('values')}"
        )

        print(
            f"Text           : "
            f"{record.get('retrieval_text')}"
        )

    print()
    print("=" * 72)


# ============================================================
# TEST CASES
# ============================================================

TEST_QUERIES = [

    (
        "Inlet Steam Flow",

        'What is the inlet steam flow of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),

    (
        "Outlet Steam Flow",

        'What is the outlet steam flow of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),

    (
        "Inlet Steam Line Size",

        'What is the line size of the inlet steam of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),

    (
        "Outlet Piping Class",

        'What is the piping class of the outlet steam of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),

    (
        "Inlet Body Material",

        'What is the inlet body material of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),

    (
        "Material Code",

        'What is the material code of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),

    (
        "Page 2 Inlet Steam Flow",

        'What is the inlet steam flow of '
        'DSH 6"300RF-INTEG TCV 1"600RF-HART?'
    ),

    (
        "Page 5 Material Code",

        'What is the material code of '
        'DSH 12"300RF-INTEG TCV 1"300RF-HART?'
    ),
]


# ============================================================
# TEST SUITE
# ============================================================

def run_test_suite(
    records: List[Dict[str, Any]]
) -> None:

    print()
    print("#" * 72)
    print("STEP 4.4 END-TO-END RETRIEVAL TEST SUITE")
    print("#" * 72)

    passed = 0

    for test_name, query in TEST_QUERIES:

        print()
        print()
        print(
            f"TEST: {test_name}"
        )

        print("-" * 72)

        parsed_query = parse_query(
            query,
            records
        )

        results = retrieve_records(
            records,
            parsed_query,
            limit=10
        )

        print_result(
            query,
            parsed_query,
            results
        )

        # ----------------------------------------------------
        # Basic success condition
        # ----------------------------------------------------

        if len(results) >= 1:
            passed += 1

        else:
            print(
                f"\nTEST RESULT: FAILED"
            )

    print()
    print("=" * 72)
    print("STEP 4.4 TEST SUMMARY")
    print("=" * 72)

    print(
        f"Tests  : {len(TEST_QUERIES)}"
    )

    print(
        f"Passed : {passed}"
    )

    print(
        f"Failed : "
        f"{len(TEST_QUERIES) - passed}"
    )

    if passed == len(TEST_QUERIES):

        print(
            "STATUS : PASSED"
        )

    else:

        print(
            "STATUS : NEEDS REVIEW"
        )

    print("=" * 72)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Step 4.4 - End-to-End "
            "Natural Language Structured Retrieval"
        )
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to retrieval_enriched.json"
    )

    parser.add_argument(
        "--query",
        type=str,
        help="Natural language query"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run Step 4.4 test suite"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of retrieval results"
    )

    args = parser.parse_args()

    print("=" * 72)
    print("END-TO-END STRUCTURED RETRIEVER")
    print("STEP 4.4")
    print("=" * 72)

    print(
        f"Dataset : {args.input}"
    )

    records = load_records(
        args.input
    )

    print(
        f"Records : {len(records)}"
    )

    # --------------------------------------------------------
    # Test suite
    # --------------------------------------------------------

    if args.test:

        run_test_suite(
            records
        )

        return

    # --------------------------------------------------------
    # Single query
    # --------------------------------------------------------

    if args.query:

        parsed_query = parse_query(
            args.query,
            records
        )

        results = retrieve_records(
            records,
            parsed_query,
            limit=args.limit
        )

        print_result(
            args.query,
            parsed_query,
            results
        )

        return

    # --------------------------------------------------------
    # No arguments
    # --------------------------------------------------------

    print()
    print("No query supplied.")
    print()

    print("Run test suite:")
    print()

    print(
        "python src\\retrieval\\end_to_end_retriever.py "
        "--test"
    )

    print()

    print("Run a single query:")
    print()

    print(
        'python src\\retrieval\\end_to_end_retriever.py '
        '--query "What is the inlet steam flow of DSH ...?"'
    )


if __name__ == "__main__":
    main()