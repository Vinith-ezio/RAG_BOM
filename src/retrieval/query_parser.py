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

def normalize_text(value: Any) -> str:
    """
    Normalize text for comparison.

    Important:
    - Does NOT modify the actual source value.
    - Only used internally for matching.
    """
    if value is None:
        return ""

    text = str(value).strip().lower()

    # Normalize repeated whitespace
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
# DATASET VOCABULARY
# ============================================================

def collect_unique_values(
    records: List[Dict[str, Any]],
    field: str
) -> List[str]:
    """
    Collect unique values from a record field.
    """

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
    """
    Detect ITEM by matching known dataset ITEM values.

    Longest values are checked first so that a more
    specific item wins over a partial match.
    """

    normalized_query = normalize_text(query)

    items = collect_unique_values(
        records,
        "item"
    )

    for item in items:

        normalized_item = normalize_text(item)

        if normalized_item and normalized_item in normalized_query:
            return item

    return None


# ============================================================
# SIDE DETECTION
# ============================================================

def detect_side(query: str) -> Optional[str]:
    """
    Detect inlet / outlet from the query.
    """

    text = normalize_text(query)

    # More specific phrases first
    if re.search(r"\binlet\s+steam\b", text):
        return "inlet"

    if re.search(r"\boutlet\s+steam\b", text):
        return "outlet"

    if re.search(r"\binlet\b", text):
        return "inlet"

    if re.search(r"\boutlet\b", text):
        return "outlet"

    return None


# ============================================================
# PARAMETER ALIASES
# ============================================================

PARAMETER_ALIASES = {

    # Flow
    "flow": "Flow",
    "steam flow": "Flow",
    "water flow": "Flow",
    "flow rate": "Flow",

    # Line size
    "line size": "Line Size",
    "line diameter": "Line Size",
    "pipe size": "Line Size",
    "piping size": "Line Size",

    # Materials
    "body material": "Body Material",
    "body": "Body Material",

    # Material code
    "material code": "Material Code",

    # Pressure
    "pressure": "Pressure",
    "steam pressure": "Pressure",
    "water pressure": "Pressure",

    # Temperature
    "temperature": "Temperature",
    "steam temperature": "Temperature",
    "water temperature": "Temperature",

    # Actuator / positioner
    "actuator": "Actuator",
    "positioner": "Positioner",

    # Tag
    "tag": "Tag Number",
    "tag number": "Tag Number",

    # Insulation
    "insulation code": "Insulation Code",
}


def detect_parameter(
    query: str,
    records: List[Dict[str, Any]]
) -> Optional[str]:
    """
    Detect parameter from:

    1. Known aliases
    2. Actual parameters present in dataset
    """

    text = normalize_text(query)

    # --------------------------------------------------------
    # 1. Explicit aliases
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
    # 2. Dataset parameters
    # --------------------------------------------------------

    parameters = collect_unique_values(
        records,
        "parameter"
    )

    for parameter in parameters:

        normalized_parameter = normalize_text(parameter)

        if normalized_parameter in text:
            return parameter

    return None


# ============================================================
# SUB-PARAMETER DETECTION
# ============================================================

SUB_PARAMETER_ALIASES = {

    "piping class": "Piping Class",
    "stem plug material": "Stem / Plug Material",
    "stem / plug material": "Stem / Plug Material",

    "minimum normal maximum unit":
        "Min. / Norm. / Max. / Unit",

    "min norm max unit":
        "Min. / Norm. / Max. / Unit",
}

# ============================================================
# SUB-PARAMETER → PARENT PARAMETER RELATIONSHIP
# ============================================================

SUB_PARAMETER_PARENT = {
    "Piping Class": "Line Size",
    "Stem / Plug Material": "Body Material",
}


def detect_sub_parameter(
    query: str,
    records: List[Dict[str, Any]]
) -> Optional[str]:

    text = normalize_text(query)

    # --------------------------------------------------------
    # 1. Alias detection
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
    # 2. Dataset values
    # --------------------------------------------------------

    sub_parameters = collect_unique_values(
        records,
        "sub_parameter"
    )

    for sub_parameter in sub_parameters:

        normalized = normalize_text(sub_parameter)

        if normalized and normalized in text:
            return sub_parameter

    return None


# ============================================================
# SECTION DETECTION
# ============================================================

def detect_section(
    query: str,
    records: List[Dict[str, Any]]
) -> Optional[str]:

    text = normalize_text(query)

    sections = collect_unique_values(
        records,
        "section"
    )

    for section in sections:

        normalized = normalize_text(section)

        if normalized and normalized in text:
            return section

    # Natural language aliases
    if "steam" in text:
        for section in sections:
            if normalize_text(section) == "steam conditions":
                return section

    if "water" in text:
        for section in sections:
            if normalize_text(section) == "water conditions":
                return section

    return None


# ============================================================
# SUBSECTION DETECTION
# ============================================================

def detect_subsection(
    query: str,
    records: List[Dict[str, Any]]
) -> Optional[str]:

    text = normalize_text(query)

    subsections = collect_unique_values(
        records,
        "subsection"
    )

    for subsection in subsections:

        normalized = normalize_text(subsection)

        if normalized and normalized in text:
            return subsection

    # Natural language mapping
    if "inlet steam" in text or "outlet steam" in text:

        for subsection in subsections:

            if normalize_text(subsection) == (
                "inlet steam / outlet steam"
            ):
                return subsection

    return None


# ============================================================
# VALUE FIELD DETECTION
# ============================================================

VALUE_FIELD_ALIASES = {

    "minimum": "min",
    "min": "min",

    "normal": "normal",
    "norm": "normal",

    "maximum": "max",
    "max": "max",

    "unit": "unit",

    "class": "class",
    "piping class": "class",

    "value": "value",
}


def detect_value_field(
    query: str
) -> Optional[str]:

    text = normalize_text(query)

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
# SINGLE QUERY PARSER
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
    # Resolve parent parameter from sub-parameter
    # --------------------------------------------------------
    #
    # Example:
    #
    # "piping class"
    #       ↓
    # sub_parameter = "Piping Class"
    #       ↓
    # parent parameter = "Line Size"
    #
    # This matches the actual source table structure.
    # --------------------------------------------------------

    if sub_parameter and not parameter:

        parent_parameter = SUB_PARAMETER_PARENT.get(
            sub_parameter
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
# VALIDATION
# ============================================================

def validate_parsed_query(
    parsed: Dict[str, Any]
) -> Dict[str, Any]:

    missing = []

    if not parsed.get("item"):
        missing.append("item")

    if not parsed.get("parameter"):
        missing.append("parameter")

    return {
        "valid": len(missing) == 0,
        "missing_required_fields": missing
    }


# ============================================================
# DISPLAY
# ============================================================

def print_parsed_query(
    parsed: Dict[str, Any]
) -> None:

    validation = validate_parsed_query(
        parsed
    )

    print()
    print("=" * 72)
    print("PARSED STRUCTURED QUERY")
    print("=" * 72)

    print(f"Original Query : {parsed['original_query']}")
    print("-" * 72)

    print(f"Item           : {parsed['item']}")
    print(f"Section        : {parsed['section']}")
    print(f"Subsection     : {parsed['subsection']}")
    print(f"Parameter      : {parsed['parameter']}")
    print(f"Sub-parameter  : {parsed['sub_parameter']}")
    print(f"Side           : {parsed['side']}")
    print(f"Value Field    : {parsed['value_field']}")

    print("-" * 72)

    if validation["valid"]:
        print("VALIDATION     : PASSED")
    else:
        print("VALIDATION     : FAILED")
        print(
            "Missing        : "
            + ", ".join(
                validation["missing_required_fields"]
            )
        )

    print("=" * 72)


# ============================================================
# TEST QUERIES
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
]


def run_test_suite(
    records: List[Dict[str, Any]]
) -> None:

    print()
    print("#" * 72)
    print("QUERY PARSER TEST SUITE")
    print("#" * 72)

    passed = 0

    for name, query in TEST_QUERIES:

        print()
        print()
        print(f"TEST: {name}")
        print("-" * 72)

        parsed = parse_query(
            query,
            records
        )

        print_parsed_query(
            parsed
        )

        validation = validate_parsed_query(
            parsed
        )

        if validation["valid"]:
            passed += 1

    print()
    print("=" * 72)
    print("TEST SUMMARY")
    print("=" * 72)
    print(f"Tests : {len(TEST_QUERIES)}")
    print(f"Passed: {passed}")
    print(
        f"Failed: {len(TEST_QUERIES) - passed}"
    )
    print("=" * 72)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Step 4.3 - Natural Language "
            "Structured Query Parser"
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
        help="Run parser test suite"
    )

    args = parser.parse_args()

    print("=" * 72)
    print("NATURAL LANGUAGE QUERY PARSER")
    print("STEP 4.3")
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

    if args.test:

        run_test_suite(
            records
        )

        return

    if args.query:

        parsed = parse_query(
            args.query,
            records
        )

        print_parsed_query(
            parsed
        )

        print()
        print("JSON OUTPUT")
        print("-" * 72)

        print(
            json.dumps(
                parsed,
                indent=4,
                ensure_ascii=False
            )
        )

        return

    print()
    print("No query supplied.")
    print()
    print("Use:")
    print(
        'python src\\retrieval\\query_parser.py '
        '--test'
    )
    print()
    print(
        'python src\\retrieval\\query_parser.py '
        '--query "What is the inlet steam flow of DSH ...?"'
    )


if __name__ == "__main__":
    main()