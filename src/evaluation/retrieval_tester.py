import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RECORDS_PATH = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "retrieval_enriched.json"
)

GOLDEN_PATH = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "retrieval_evaluation.json"
)

SRC_RETRIEVAL = PROJECT_ROOT / "src" / "retrieval"

if str(SRC_RETRIEVAL) not in sys.path:
    sys.path.insert(0, str(SRC_RETRIEVAL))


# ---------------------------------------------------------------------
# IMPORT FROZEN V3 RETRIEVER
# ---------------------------------------------------------------------

from hybrid_retriever_v3 import NativeMultiIntentHybridRetriever


# ---------------------------------------------------------------------
# LOAD RECORDS
# ---------------------------------------------------------------------

def load_records():
    with open(RECORDS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # -------------------------------------------------------------
    # Expected format:
    #
    # {
    #     "records": [...]
    # }
    #
    # But also support a direct list:
    #
    # [
    #     {...},
    #     {...}
    # ]
    # -------------------------------------------------------------

    if isinstance(data, list):
        records = data

    elif isinstance(data, dict):
        if isinstance(data.get("records"), list):
            records = data["records"]

        elif isinstance(data.get("data"), list):
            records = data["data"]

        else:
            raise ValueError(
                "Unsupported retrieval_enriched.json structure. "
                "Expected a list or a dictionary containing "
                "'records' or 'data'."
            )

    else:
        raise ValueError(
            f"Unsupported JSON root type: {type(data).__name__}"
        )

    # -------------------------------------------------------------
    # Validate individual records
    # -------------------------------------------------------------

    valid_records = []

    for index, record in enumerate(records):

        if not isinstance(record, dict):
            print(
                f"WARNING: Skipping record at index {index}: "
                f"expected object, got {type(record).__name__}"
            )
            continue

        if not record.get("record_id"):
            print(
                f"WARNING: Skipping record at index {index}: "
                "missing record_id"
            )
            continue

        valid_records.append(record)

    return valid_records

# ---------------------------------------------------------------------
# BUILD RECORD LOOKUP
# ---------------------------------------------------------------------

def build_record_lookup(records):
    return {
        record["record_id"]: record
        for record in records
    }


# ---------------------------------------------------------------------
# VALUE DISPLAY
# ---------------------------------------------------------------------

def print_values(values):
    if not values:
        print("  Values      : None")
        return

    for key, value in values.items():
        label = str(key).replace("_", " ").title()
        print(f"  {label:<12}: {value}")


# ---------------------------------------------------------------------
# RECORD DISPLAY
# ---------------------------------------------------------------------

def print_record(record, rank=None):
    if rank is not None:
        print(f"\n  Rank         : {rank}")

    print(f"  Record ID    : {record.get('record_id')}")
    print(f"  Page         : {record.get('page')}")
    print(f"  Table        : {record.get('table_id')}")
    print(f"  Item         : {record.get('item')}")
    print(f"  Section      : {record.get('section')}")
    print(f"  Subsection   : {record.get('subsection')}")
    print(f"  Parameter    : {record.get('parameter')}")
    print(f"  Sub-parameter: {record.get('sub_parameter')}")
    print(f"  Side         : {record.get('side')}")

    print("  Values       :")
    print_values(record.get("values"))


# ---------------------------------------------------------------------
# INTENT DISPLAY
# ---------------------------------------------------------------------

def print_intent(intent_data, record_lookup):
    intent_number = intent_data.get("intent_number")
    intent_query = intent_data.get("query")
    parsed = intent_data.get("parsed", {})
    abstained = intent_data.get("abstained", False)

    print()
    print("=" * 90)
    print(f"INTENT {intent_number}")
    print("=" * 90)

    print(f"Query        : {intent_query}")
    print(f"Abstained    : {abstained}")

    print("\nParsed Intent")
    print("-" * 90)

    if parsed:
        for key, value in parsed.items():
            if value is not None:
                print(f"{key:<16}: {value}")
    else:
        print("No parsed fields.")

    results = intent_data.get("results", [])

    print("\nRetrieved Records")
    print("-" * 90)

    if not results:
        print("No records retrieved.")
        return

    for rank, result in enumerate(results, start=1):
        record_id = result.get("record_id")

        print(f"\nResult #{rank}")
        print(f"  Record ID    : {record_id}")

        if record_id not in record_lookup:
            print("  WARNING      : Record not found in dataset.")
            continue

        record = record_lookup[record_id]

        print(f"  Page         : {record.get('page')}")
        print(f"  Section      : {record.get('section')}")
        print(f"  Parameter    : {record.get('parameter')}")
        print(f"  Sub-parameter: {record.get('sub_parameter')}")
        print(f"  Side         : {record.get('side')}")

        print("  Values:")
        print_values(record.get("values"))


# ---------------------------------------------------------------------
# RETRIEVAL TEST
# ---------------------------------------------------------------------

def test_query(retriever, record_lookup, query):
    print()
    print("#" * 90)
    print("RETRIEVAL TEST")
    print("#" * 90)

    print(f"\nQuery:\n{query}")

    # -------------------------------------------------------------
    # RUN FROZEN V3
    # -------------------------------------------------------------

    output = retriever.search(
        query,
        top_k=5,
        structured_k=10,
        bm25_k=10
    )

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------

    print("\n" + "-" * 90)
    print("RETRIEVAL SUMMARY")
    print("-" * 90)

    print(f"Intent count       : {output.get('intent_count', 0)}")
    print(f"Successful intents : {output.get('successful_intents', 0)}")
    print(f"Abstained intents  : {output.get('abstained_intents', 0)}")

    flattened = output.get("flattened_results", [])

    print(f"Retrieved records  : {len(flattened)}")

    if flattened:
        print("\nRetrieved Record IDs:")
        for rank, result in enumerate(flattened, start=1):
            record_id = result.get("record_id")
            print(f"  {rank}. {record_id}")

    # -------------------------------------------------------------
    # INTENT DETAILS
    # -------------------------------------------------------------

    intents = output.get("intents", [])

    for intent in intents:
        print_intent(intent, record_lookup)

    # -------------------------------------------------------------
    # FINAL UNIQUE RECORDS
    # -------------------------------------------------------------

    print()
    print("=" * 90)
    print("FINAL RETRIEVED RECORDS")
    print("=" * 90)

    if not flattened:
        print("\n[NO RECORDS RETRIEVED]")
        print("\nRetrieval Status: ABSTAINED / NO MATCH")
        return output

    for rank, result in enumerate(flattened, start=1):
        record_id = result.get("record_id")

        if record_id not in record_lookup:
            print(f"\n{rank}. {record_id}")
            print("   WARNING: Record not found.")
            continue

        record = record_lookup[record_id]

        print()
        print(f"[RESULT {rank}]")
        print_record(record, rank=rank)

    # -------------------------------------------------------------
    # STATUS
    # -------------------------------------------------------------

    print()
    print("-" * 90)

    if flattened:
        print("Retrieval Status : RECORDS FOUND")
    else:
        print("Retrieval Status : NO RECORDS")

    print("-" * 90)

    return output


# ---------------------------------------------------------------------
# GOLDEN DATASET LOOKUP
# ---------------------------------------------------------------------

def load_golden_dataset():
    if not GOLDEN_PATH.exists():
        return []

    try:
        with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            for key in ["queries", "results", "tests", "evaluation"]:
                if key in data and isinstance(data[key], list):
                    return data[key]

    except Exception:
        pass

    return []


# ---------------------------------------------------------------------
# GOLDEN QUERY SEARCH
# ---------------------------------------------------------------------

def find_golden_query(golden_data, query):
    normalized = query.strip().lower()

    for item in golden_data:
        candidate = item.get("query", "")

        if candidate.strip().lower() == normalized:
            return item

    return None


# ---------------------------------------------------------------------
# GOLDEN COMPARISON
# ---------------------------------------------------------------------

def compare_with_golden(output, golden):
    if not golden:
        return

    expected_ids = golden.get("expected_record_ids")

    if expected_ids is None:
        expected_ids = golden.get("expected_ids")

    if expected_ids is None:
        expected = golden.get("expected")

        if isinstance(expected, list):
            expected_ids = expected

    if expected_ids is None:
        return

    retrieved_ids = [
        result.get("record_id")
        for result in output.get("flattened_results", [])
    ]

    expected_ids = list(expected_ids)

    print()
    print("=" * 90)
    print("GOLDEN DATASET COMPARISON")
    print("=" * 90)

    print(f"Expected records : {expected_ids}")
    print(f"Retrieved records: {retrieved_ids}")

    expected_set = set(expected_ids)
    retrieved_set = set(retrieved_ids)

    missing = expected_set - retrieved_set
    unexpected = retrieved_set - expected_set

    print()

    if not missing and not unexpected:
        print("Result           : PASS")
        print("All expected records were retrieved.")
    else:
        print("Result           : FAIL")

        if missing:
            print(f"Missing records  : {sorted(missing)}")

        if unexpected:
            print(f"Unexpected       : {sorted(unexpected)}")

    # -------------------------------------------------------------
    # RANK CHECK
    # -------------------------------------------------------------

    if expected_ids:
        first_expected = expected_ids[0]

        if retrieved_ids and retrieved_ids[0] == first_expected:
            print("Top-1            : PASS")
        else:
            print("Top-1            : FAIL")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 90)
    print("RETRIEVAL TESTER")
    print("=" * 90)

    print("\nLoading retrieval records...")

    records = load_records()

    print(f"Records loaded: {len(records)}")

    record_lookup = build_record_lookup(records)

    print("Initializing Hybrid V3...")

    retriever = NativeMultiIntentHybridRetriever(
        records,
        rrf_k=60
    )

    golden_data = load_golden_dataset()

    print("Hybrid V3 initialized.")

    print()
    print("=" * 90)
    print("INTERACTIVE RETRIEVAL TEST")
    print("=" * 90)

    print("\nEnter a natural-language query.")
    print("Type 'exit' or 'quit' to stop.")

    while True:

        print()
        query = input("Query: ").strip()

        if not query:
            continue

        if query.lower() in {"exit", "quit"}:
            print("\nExiting retrieval tester.")
            break

        output = test_query(
            retriever,
            record_lookup,
            query
        )

        # ---------------------------------------------------------
        # OPTIONAL GOLDEN COMPARISON
        # ---------------------------------------------------------

        golden = find_golden_query(
            golden_data,
            query
        )

        if golden:
            compare_with_golden(
                output,
                golden
            )


if __name__ == "__main__":
    main()