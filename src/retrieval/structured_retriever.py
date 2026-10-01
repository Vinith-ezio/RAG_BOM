import json
import argparse
import re
from pathlib import Path


def normalize(text):
    """
    Normalize text only for matching.
    Original source values remain unchanged.
    """

    if text is None:
        return ""

    text = str(text).strip().upper()

    # Normalize curly quotes to normal quotes
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("’", "'")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text


def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])

    if not records:
        raise ValueError("No retrieval records found.")

    return records


def exact_match(value, target):
    """
    Case-insensitive exact match.
    """

    return normalize(value) == normalize(target)


def contains_match(value, target):
    """
    Useful for flexible user input.

    Example:
        target = "FLOW"
        value  = "FLOW"
    """

    value = normalize(value)
    target = normalize(target)

    return target in value


def retrieve(
    records,
    item=None,
    section=None,
    subsection=None,
    parameter=None,
    sub_parameter=None,
    side=None,
    limit=10
):
    """
    Structured metadata retrieval.

    Every supplied filter is applied.

    Example:

        item       = DSH 6"300RF-INTEG TCV 1"300RF-HART
        section    = STEAM CONDITIONS
        parameter  = FLOW
        side       = INLET
    """

    candidates = []

    for record in records:

        metadata = record.get(
            "search_metadata",
            {}
        )

        # -----------------------------------------------------
        # ITEM
        # -----------------------------------------------------

        if item:
            record_item = metadata.get("item")

            if not exact_match(record_item, item):
                continue

        # -----------------------------------------------------
        # SECTION
        # -----------------------------------------------------

        if section:
            record_section = metadata.get("section")

            if not exact_match(record_section, section):
                continue

        # -----------------------------------------------------
        # SUBSECTION
        # -----------------------------------------------------

        if subsection:
            record_subsection = metadata.get("subsection")

            if not exact_match(
                record_subsection,
                subsection
            ):
                continue

        # -----------------------------------------------------
        # PARAMETER
        # -----------------------------------------------------

        if parameter:
            record_parameter = metadata.get(
                "parameter"
            )

            if not contains_match(
                record_parameter,
                parameter
            ):
                continue

        # -----------------------------------------------------
        # SUB-PARAMETER
        # -----------------------------------------------------

        if sub_parameter:
            record_sub_parameter = metadata.get(
                "sub_parameter"
            )

            if not contains_match(
                record_sub_parameter,
                sub_parameter
            ):
                continue

        # -----------------------------------------------------
        # SIDE
        # -----------------------------------------------------

        if side:
            record_side = metadata.get("side")

            if not exact_match(
                record_side,
                side
            ):
                continue

        candidates.append(record)

        if len(candidates) >= limit:
            break

    return candidates


def print_result(records):

    print("\n" + "=" * 72)
    print("RETRIEVAL RESULTS")
    print("=" * 72)

    if not records:
        print("No matching records found.")
        print("=" * 72)
        return

    print(f"Matches : {len(records)}")
    print("-" * 72)

    for index, record in enumerate(
        records,
        start=1
    ):

        print(f"\n[{index}]")
        print(f"Record ID   : {record.get('record_id')}")
        print(f"Page        : {record.get('page')}")
        print(f"Row         : {record.get('row_number')}")
        print(f"Item        : {record.get('item')}")
        print(f"Section     : {record.get('section')}")
        print(f"Subsection  : {record.get('subsection')}")
        print(f"Parameter   : {record.get('parameter')}")
        print(f"Sub-param   : {record.get('sub_parameter')}")
        print(f"Side        : {record.get('side')}")
        print(f"Values      : {record.get('values')}")
        print(
            f"Text        : "
            f"{record.get('retrieval_text')}"
        )

    print("\n" + "=" * 72)


def run_test_queries(records):

    tests = [

        {
            "name": "Inlet Steam Flow",
            "filters": {
                "item":
                    'DSH 6"300RF-INTEG TCV 1"300RF-HART',
                "section":
                    "STEAM CONDITIONS",
                "parameter":
                    "FLOW",
                "side":
                    "INLET"
            }
        },

        {
            "name": "Outlet Steam Flow",
            "filters": {
                "item":
                    'DSH 6"300RF-INTEG TCV 1"300RF-HART',
                "section":
                    "STEAM CONDITIONS",
                "parameter":
                    "FLOW",
                "side":
                    "OUTLET"
            }
        },

        {
            "name": "Inlet Line Size",
            "filters": {
                "item":
                    'DSH 6"300RF-INTEG TCV 1"300RF-HART',
                "section":
                    "STEAM CONDITIONS",
                "parameter":
                    "LINE SIZE",
                "side":
                    "INLET"
            }
        },

        {
            "name": "Outlet Line Size",
            "filters": {
                "item":
                    'DSH 6"300RF-INTEG TCV 1"300RF-HART',
                "section":
                    "STEAM CONDITIONS",
                "parameter":
                    "LINE SIZE",
                "side":
                    "OUTLET"
            }
        },

        {
            "name": "Inlet Body Material",
            "filters": {
                "item":
                    'DSH 6"300RF-INTEG TCV 1"300RF-HART',
                "section":
                    "STEAM CONDITIONS",
                "parameter":
                    "BODY MATERIAL",
                "side":
                    "INLET"
            }
        }
    ]

    print("\n")
    print("#" * 72)
    print("STRUCTURED RETRIEVAL TEST SUITE")
    print("#" * 72)

    for test in tests:

        print(
            f"\n\nTEST: {test['name']}"
        )

        print("-" * 72)

        results = retrieve(
            records,
            **test["filters"]
        )

        print_result(results)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Exact structured retrieval "
            "for complex engineering tables."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to retrieval_enriched.json"
    )

    parser.add_argument(
        "--item",
        help="Exact ITEM"
    )

    parser.add_argument(
        "--section",
        help="Section"
    )

    parser.add_argument(
        "--subsection",
        help="Subsection"
    )

    parser.add_argument(
        "--parameter",
        help="Parameter"
    )

    parser.add_argument(
        "--sub-parameter",
        dest="sub_parameter",
        help="Sub-parameter"
    )

    parser.add_argument(
        "--side",
        help="inlet / outlet"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run predefined retrieval tests"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    records = load_records(input_path)

    print("=" * 72)
    print("STRUCTURED RETRIEVER")
    print("=" * 72)

    print(f"Dataset : {input_path}")
    print(f"Records : {len(records)}")

    if args.test:

        run_test_queries(records)
        return

    results = retrieve(
        records=records,
        item=args.item,
        section=args.section,
        subsection=args.subsection,
        parameter=args.parameter,
        sub_parameter=args.sub_parameter,
        side=args.side,
        limit=args.limit
    )

    print_result(results)


if __name__ == "__main__":
    main()