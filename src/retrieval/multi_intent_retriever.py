import argparse
import json
import re
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_INPUT = (
    r"E:\PDF Ingestion\data\output\retrieval_enriched.json"
)


# ============================================================
# DATASET LOADER
# ============================================================

def load_records(path):
    """
    Load enriched retrieval records from JSON.
    """

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        records = data.get("records", [])

        if not records:
            # Support alternative wrapper structures
            for key in ["data", "items", "retrieval_records"]:
                if isinstance(data.get(key), list):
                    records = data[key]
                    break

    elif isinstance(data, list):
        records = data

    else:
        raise ValueError("Unsupported JSON structure.")

    if not records:
        raise ValueError("No retrieval records found.")

    return records


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Normalize text for comparison.
    """

    if value is None:
        return ""

    value = str(value).strip().lower()

    # Normalize quotes
    value = value.replace("“", '"')
    value = value.replace("”", '"')
    value = value.replace("’", "'")

    # Normalize multiple spaces
    value = re.sub(r"\s+", " ", value)

    return value


# ============================================================
# FIELD MATCHING
# ============================================================

def field_matches(record_value, query_value):
    """
    Match a structured field.

    None query means:
        do not filter by this field.
    """

    if query_value is None:
        return True

    if record_value is None:
        return False

    return normalize_text(record_value) == normalize_text(query_value)


# ============================================================
# ITEM DETECTION
# ============================================================

def detect_item(text, records):
    """
    Detect ITEM from the query.

    ITEM is treated as shared document context because a
    multi-intent query normally refers to one instrument/item.
    """

    normalized_query = normalize_text(text)

    items = []

    for record in records:
        item = record.get("item")

        if not item:
            continue

        if normalize_text(item) not in items:
            items.append(normalize_text(item))

    # Prefer longest match
    items.sort(key=len, reverse=True)

    for item_normalized in items:
        if item_normalized in normalized_query:
            for record in records:
                item = record.get("item")

                if (
                    item
                    and normalize_text(item) == item_normalized
                ):
                    return item

    return None


# ============================================================
# SIDE DETECTION
# ============================================================

def detect_side(text):
    """
    Detect inlet/outlet side.
    """

    normalized = normalize_text(text)

    # Check more specific phrases first
    if "inlet steam" in normalized:
        return "inlet"

    if "outlet steam" in normalized:
        return "outlet"

    if re.search(r"\binlet\b", normalized):
        return "inlet"

    if re.search(r"\boutlet\b", normalized):
        return "outlet"

    return None


# ============================================================
# SECTION DETECTION
# ============================================================

def detect_section(text, records):
    """
    Detect section ONLY from the individual intent text.

    IMPORTANT:
    Do not pass the original multi-intent query here.

    This prevents:
        "inlet steam flow ... material code"

    from incorrectly assigning:
        STEAM CONDITIONS

    to the material-code intent.
    """

    normalized = normalize_text(text)

    sections = []

    for record in records:
        section = record.get("section")

        if not section:
            continue

        if normalize_text(section) not in [
            normalize_text(x) for x in sections
        ]:
            sections.append(section)

    # Longest section names first
    sections.sort(
        key=lambda x: len(normalize_text(x)),
        reverse=True
    )

    for section in sections:
        section_normalized = normalize_text(section)

        # Direct section-name match
        if section_normalized in normalized:
            return section

    # Dataset-aware semantic section detection
    if "steam" in normalized:
        for section in sections:
            if "steam" in normalize_text(section):
                return section

    if "water" in normalized:
        for section in sections:
            if "water" in normalize_text(section):
                return section

    return None


# ============================================================
# SUBSECTION DETECTION
# ============================================================

def detect_subsection(text, records):
    """
    Detect subsection from the individual intent text.
    """

    normalized = normalize_text(text)

    subsections = []

    for record in records:
        subsection = record.get("subsection")

        if not subsection:
            continue

        if normalize_text(subsection) not in [
            normalize_text(x) for x in subsections
        ]:
            subsections.append(subsection)

    subsections.sort(
        key=lambda x: len(normalize_text(x)),
        reverse=True
    )

    for subsection in subsections:

        subsection_normalized = normalize_text(subsection)

        if subsection_normalized in normalized:
            return subsection

        # Handle individual sides inside a combined subsection
        if (
            "inlet" in normalized
            and "inlet" in subsection_normalized
        ):
            return subsection

        if (
            "outlet" in normalized
            and "outlet" in subsection_normalized
        ):
            return subsection

    return None


# ============================================================
# PARAMETER DETECTION
# ============================================================

PARAMETER_ALIASES = {
    "flow": "Flow",
    "line size": "Line Size",
    "body material": "Body Material",
    "material code": "Material Code",
    "tag number": "Tag Number",
    "service": "Service",
    "p/n": "P/N No",
    "p/n no": "P/N No",
    "item": "ITEM",
    "esd": "ESD",
    "actuator": "Actuator",
    "positioner": "Positioner",
    "pressure": "Pressure",
    "temperature": "Temperature",
}


def detect_parameter(text):
    """
    Detect parameter from intent text.
    """

    normalized = normalize_text(text)

    # Longest aliases first
    aliases = sorted(
        PARAMETER_ALIASES.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )

    for alias, parameter in aliases:

        if alias in normalized:
            return parameter

    return None


# ============================================================
# SUB-PARAMETER DETECTION
# ============================================================

SUB_PARAMETER_ALIASES = {
    "piping class": "Piping Class",
    "stem / plug material": "Stem / Plug Material",
    "stem plug material": "Stem / Plug Material",
}


def detect_sub_parameter(text):
    """
    Detect sub-parameter.
    """

    normalized = normalize_text(text)

    aliases = sorted(
        SUB_PARAMETER_ALIASES.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )

    for alias, sub_parameter in aliases:

        if alias in normalized:
            return sub_parameter

    return None


# ============================================================
# SUB-PARAMETER -> PARENT PARAMETER
# ============================================================

SUB_PARAMETER_PARENT = {
    "Piping Class": "Line Size",
    "Stem / Plug Material": "Body Material",
}


# ============================================================
# VALUE FIELD DETECTION
# ============================================================

VALUE_FIELD_ALIASES = {
    "minimum": "min",
    "minimum value": "min",
    "min": "min",

    "normal": "normal",
    "normal value": "normal",

    "maximum": "max",
    "maximum value": "max",
    "max": "max",

    "unit": "unit",

    "class": "class",
    "piping class": "class",

    "value": "value",
}


def detect_value_field(text, parameter=None):
    """
    Detect requested value field.

    Examples:
        normal flow -> normal
        minimum flow -> min
        maximum flow -> max
        piping class -> class
    """

    normalized = normalize_text(text)

    # Piping Class has a special structured value
    if "piping class" in normalized:
        return "class"

    # Explicit aliases
    aliases = sorted(
        VALUE_FIELD_ALIASES.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )

    for alias, field in aliases:

        if re.search(
            rf"\b{re.escape(alias)}\b",
            normalized
        ):
            return field

    return None


# ============================================================
# QUERY SPLITTING
# ============================================================

def split_multi_intent_query(query):
    """
    Split a natural-language query into individual intents.

    Example:

    What is the inlet steam flow,
    outlet steam flow,
    outlet piping class,
    and material code?

    becomes:

    1. inlet steam flow
    2. outlet steam flow
    3. outlet piping class
    4. material code
    """

    text = query.strip()

    # Remove common leading question phrases
    text = re.sub(
        r"^(what\s+is|what\s+are|tell\s+me|show\s+me)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove trailing question mark
    text = text.rstrip("?").strip()

    # Replace " as well as " with " and "
    text = re.sub(
        r"\bas\s+well\s+as\b",
        "and",
        text,
        flags=re.IGNORECASE,
    )

    # Split comma-separated parts first
    parts = re.split(r"\s*,\s*", text)

    expanded = []

    for part in parts:

        part = part.strip()

        if not part:
            continue

        # Split "X and Y" only when both sides look like
        # separate attributes.
        and_parts = re.split(
            r"\s+\band\b\s+",
            part,
            flags=re.IGNORECASE,
        )

        if len(and_parts) > 1:

            for item in and_parts:

                item = item.strip()

                if item:
                    expanded.append(item)

        else:
            expanded.append(part)

    return expanded


# ============================================================
# QUERY PARSER
# ============================================================

def parse_intent(intent_text):
    """
    Parse one individual intent into structured fields.
    """

    parameter = detect_parameter(intent_text)

    sub_parameter = detect_sub_parameter(intent_text)

    # If a sub-parameter is detected but parent parameter
    # is absent, recover the parent.
    if sub_parameter and not parameter:

        parent_parameter = SUB_PARAMETER_PARENT.get(
            sub_parameter
        )

        if parent_parameter:
            parameter = parent_parameter

    parsed = {
        "query": intent_text,
        "side": detect_side(intent_text),
        "section": None,
        "subsection": None,
        "parameter": parameter,
        "sub_parameter": sub_parameter,
        "value_field": detect_value_field(
            intent_text,
            parameter,
        ),
    }

    return parsed


# ============================================================
# INTENT ENRICHMENT
# ============================================================

def enrich_intent(intent, original_query, records):
    """
    Add context to an individual intent.

    IMPORTANT DESIGN RULE:

    ITEM can safely be inherited from the original query
    because it identifies the document/entity.

    SECTION MUST NOT be inherited from the original query.

    Example:

        Original query:
        "inlet steam flow, outlet steam flow,
         outlet piping class, material code"

    The word "steam" belongs to the first three intents.

    It must NOT force:

        material code -> STEAM CONDITIONS
    """

    enriched = dict(intent)

    intent_text = intent["query"]

    # --------------------------------------------------------
    # ITEM
    # --------------------------------------------------------
    #
    # ITEM is global/shared context.
    #
    if not detect_item(intent_text, records):
        enriched["item"] = detect_item(
            original_query,
            records
        )
    else:
        enriched["item"] = detect_item(
            intent_text,
            records
        )

    # --------------------------------------------------------
    # SECTION
    # --------------------------------------------------------
    #
    # CRITICAL FIX:
    #
    # Detect section ONLY from the individual intent.
    #
    enriched["section"] = detect_section(
        intent_text,
        records
    )

    # --------------------------------------------------------
    # SUBSECTION
    # --------------------------------------------------------

    enriched["subsection"] = detect_subsection(
        intent_text,
        records
    )

    return enriched


# ============================================================
# EXACT STRUCTURED RETRIEVAL
# ============================================================

def retrieve_records(
    records,
    item=None,
    section=None,
    subsection=None,
    parameter=None,
    sub_parameter=None,
    side=None,
    value_field=None,
    limit=10,
):
    """
    Retrieve records using exact structured filtering.
    """

    matches = []

    for record in records:

        if not field_matches(
            record.get("item"),
            item
        ):
            continue

        if not field_matches(
            record.get("section"),
            section
        ):
            continue

        if not field_matches(
            record.get("subsection"),
            subsection
        ):
            continue

        if not field_matches(
            record.get("parameter"),
            parameter
        ):
            continue

        if not field_matches(
            record.get("sub_parameter"),
            sub_parameter
        ):
            continue

        if not field_matches(
            record.get("side"),
            side
        ):
            continue

        matches.append(record)

        if len(matches) >= limit:
            break

    return matches


# ============================================================
# SINGLE INTENT RETRIEVAL
# ============================================================

def retrieve_intent(
    intent,
    records,
    limit=10,
):
    """
    Retrieve records for one parsed intent.

    If the query does not identify a known structured
    parameter, abstain instead of performing broad retrieval.
    """

    parameter = intent.get("parameter")

    # --------------------------------------------------------
    # RETRIEVAL ABSTENTION
    # --------------------------------------------------------
    if parameter is None:
        return []

    matches = retrieve_records(
        records=records,

        item=intent.get("item"),

        section=intent.get("section"),

        subsection=intent.get("subsection"),

        parameter=parameter,

        sub_parameter=intent.get("sub_parameter"),

        side=intent.get("side"),

        value_field=intent.get("value_field"),

        limit=limit,
    )

    return matches


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_record(record):
    """
    Print a compact representation of one record.
    """

    print(f"Record ID      : {record.get('record_id')}")
    print(f"Page           : {record.get('page')}")
    print(f"Row            : {record.get('row_number')}")
    print(f"Section        : {record.get('section')}")
    print(f"Subsection     : {record.get('subsection')}")
    print(f"Parameter      : {record.get('parameter')}")
    print(f"Sub-parameter  : {record.get('sub_parameter')}")
    print(f"Side           : {record.get('side')}")
    print(f"Values         : {record.get('values')}")


def print_intent_result(
    index,
    intent,
    matches,
):
    """
    Print one intent and its retrieval results.
    """

    print()
    print("-" * 72)
    print(f"INTENT {index}")
    print("-" * 72)

    print(f"Query          : {intent.get('query')}")
    print(f"Item           : {intent.get('item')}")
    print(f"Section        : {intent.get('section')}")
    print(f"Subsection     : {intent.get('subsection')}")
    print(f"Parameter      : {intent.get('parameter')}")
    print(f"Sub-parameter  : {intent.get('sub_parameter')}")
    print(f"Side           : {intent.get('side')}")
    print(f"Value Field    : {intent.get('value_field')}")
    print(f"Matches        : {len(matches)}")

    if matches:

        print()

        for record in matches:

            print_record(record)

    else:

        print("NO MATCH FOUND")


# ============================================================
# MULTI-INTENT RETRIEVAL
# ============================================================

def multi_intent_retrieve(
    query,
    records,
    limit=10,
):
    """
    Complete multi-intent retrieval pipeline.
    """

    intent_texts = split_multi_intent_query(query)

    results = []

    for intent_text in intent_texts:

        parsed = parse_intent(intent_text)

        enriched = enrich_intent(
            parsed,
            query,
            records,
        )

        matches = retrieve_intent(
            enriched,
            records,
            limit=limit,
        )

        results.append(
            {
                "intent": enriched,
                "matches": matches,
            }
        )

    return results


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [

    {
        "name": "Two Attributes",
        "query": (
            'What is the line size of the inlet steam '
            'and the piping class of the outlet steam '
            'of DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected_intents": 2,
        "expected_matches": 2,
    },

    {
        "name": "Three Attributes",
        "query": (
            'What is the inlet steam flow, '
            'outlet steam line size, '
            'and inlet body material '
            'of DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected_intents": 3,
        "expected_matches": 3,
    },

    {
        "name": "Four Attributes",
        "query": (
            'What is the inlet steam flow, '
            'outlet steam flow, '
            'outlet piping class, '
            'and material code '
            'of DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected_intents": 4,
        "expected_matches": 4,
    },

    {
        "name": "Page 2 Multiple Attributes",
        "query": (
            'What is the inlet steam flow '
            'and outlet piping class '
            'of DSH 6"300RF-INTEG TCV 1"600RF-HART?'
        ),
        "expected_intents": 2,
        "expected_matches": 2,
    },

    {
        "name": "Page 5 Multiple Attributes",
        "query": (
            'What is the inlet line size '
            'and material code '
            'of DSH 12"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected_intents": 2,
        "expected_matches": 2,
    },
]


# ============================================================
# TEST RUNNER
# ============================================================

def run_tests(records):
    """
    Run the Step 4.5 validation suite.
    """

    print()
    print("=" * 72)
    print("STEP 4.5 — MULTI-INTENT STRUCTURED RETRIEVAL")
    print("=" * 72)

    total_tests = len(TEST_CASES)
    passed_tests = 0

    for test_index, test in enumerate(
        TEST_CASES,
        start=1,
    ):

        print()
        print("=" * 72)
        print(
            f"TEST {test_index}/{total_tests} : "
            f"{test['name']}"
        )
        print("=" * 72)

        print()
        print(f"Query:")
        print(test["query"])

        results = multi_intent_retrieve(
            test["query"],
            records,
        )

        total_matches = sum(
            len(result["matches"])
            for result in results
        )

        intent_count = len(results)

        expected_intents = test[
            "expected_intents"
        ]

        expected_matches = test[
            "expected_matches"
        ]

        test_passed = (
            intent_count == expected_intents
            and total_matches == expected_matches
            and all(
                len(result["matches"]) > 0
                for result in results
            )
        )

        for index, result in enumerate(
            results,
            start=1,
        ):

            print_intent_result(
                index,
                result["intent"],
                result["matches"],
            )

        print()
        print("-" * 72)

        print(
            f"Expected intents : {expected_intents}"
        )

        print(
            f"Actual intents   : {intent_count}"
        )

        print(
            f"Expected matches : {expected_matches}"
        )

        print(
            f"Actual matches   : {total_matches}"
        )

        if test_passed:

            print("STATUS           : PASSED")

            passed_tests += 1

        else:

            print("STATUS           : FAILED")

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 4.5 SUMMARY")
    print("=" * 72)

    print(
        f"Total Tests      : {total_tests}"
    )

    print(
        f"Passed           : {passed_tests}"
    )

    print(
        f"Failed           : "
        f"{total_tests - passed_tests}"
    )

    if passed_tests == total_tests:

        print(
            "STATUS           : PASSED"
        )

    else:

        print(
            "STATUS           : NEEDS REVIEW"
        )

    print("=" * 72)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Step 4.5 Multi-Intent "
            "Structured Retrieval"
        )
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to retrieval_enriched.json",
    )

    parser.add_argument(
        "--query",
        default=None,
        help=(
            "Run a custom multi-intent query "
            "instead of the test suite."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum matches per intent.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("LOADING RETRIEVAL DATASET")
    print("=" * 72)

    print(f"Input : {args.input}")

    records = load_records(
        Path(args.input)
    )

    print(
        f"Records loaded : {len(records)}"
    )

    # --------------------------------------------------------
    # Custom query
    # --------------------------------------------------------

    if args.query:

        print()
        print("=" * 72)
        print("CUSTOM MULTI-INTENT QUERY")
        print("=" * 72)

        print(
            f"Query : {args.query}"
        )

        results = multi_intent_retrieve(
            args.query,
            records,
            limit=args.limit,
        )

        for index, result in enumerate(
            results,
            start=1,
        ):

            print_intent_result(
                index,
                result["intent"],
                result["matches"],
            )

        return

    # --------------------------------------------------------
    # Test suite
    # --------------------------------------------------------

    run_tests(records)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()