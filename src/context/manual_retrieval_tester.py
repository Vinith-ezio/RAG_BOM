import json
import sys
from pathlib import Path


# =====================================================================
# PATH SETUP
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RECORDS_PATH = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "retrieval_enriched.json"
)

SCHEMA_PATH = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "schema_catalog.json"
)

ENTITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "entity_index.json"
)

SRC_RETRIEVAL = PROJECT_ROOT / "src" / "retrieval"
SRC_QUERY = PROJECT_ROOT / "src" / "query"
SRC_CONTEXT = PROJECT_ROOT / "src" / "context"

for directory in (
    SRC_RETRIEVAL,
    SRC_QUERY,
    SRC_CONTEXT,
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


# =====================================================================
# IMPORT
# =====================================================================

from intent_retrieval_adapter import IntentRetrievalAdapter
from context_builder import ContextBuilder


# =====================================================================
# JSON LOADER
# =====================================================================

def load_json(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# =====================================================================
# LOAD RETRIEVAL RECORDS
# =====================================================================

def load_records():

    data = load_json(RECORDS_PATH)

    if isinstance(data, list):

        records = data

    elif isinstance(data, dict):

        if isinstance(data.get("records"), list):
            records = data["records"]

        elif isinstance(data.get("data"), list):
            records = data["data"]

        else:
            raise ValueError(
                "Unsupported retrieval_enriched.json structure."
            )

    else:

        raise ValueError(
            f"Unsupported JSON root type: "
            f"{type(data).__name__}"
        )

    valid_records = []

    for index, record in enumerate(records):

        if not isinstance(record, dict):

            print(
                f"WARNING: Skipping record {index}"
            )

            continue

        if not record.get("record_id"):

            print(
                f"WARNING: Skipping record {index}: "
                f"missing record_id"
            )

            continue

        valid_records.append(record)

    return valid_records


# =====================================================================
# VALUE DISPLAY
# =====================================================================

def print_values(values):

    if not values:

        print("  Values        : None")
        return

    for key, value in values.items():

        label = (
            str(key)
            .replace("_", " ")
            .title()
        )

        print(
            f"  {label:<14}: {value}"
        )


# =====================================================================
# RECORD DISPLAY
# =====================================================================

def print_record(record, rank=None):

    if rank is not None:

        print()
        print(
            f"[RESULT {rank}]"
        )

    print(
        f"  Record ID     : "
        f"{record.get('record_id')}"
    )

    print(
        f"  Page          : "
        f"{record.get('page')}"
    )

    print(
        f"  Table         : "
        f"{record.get('table_id')}"
    )

    print(
        f"  Row           : "
        f"{record.get('row_number')}"
    )

    print(
        f"  Item          : "
        f"{record.get('item')}"
    )

    print(
        f"  Section       : "
        f"{record.get('section')}"
    )

    print(
        f"  Subsection    : "
        f"{record.get('subsection')}"
    )

    print(
        f"  Parameter     : "
        f"{record.get('parameter')}"
    )

    print(
        f"  Sub-parameter : "
        f"{record.get('sub_parameter')}"
    )

    print(
        f"  Side          : "
        f"{record.get('side')}"
    )

    print("  Values        :")

    print_values(
        record.get("values")
    )


# =====================================================================
# INTENT DISPLAY
# =====================================================================

def print_intent(intent, title="INTENT"):

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)

    if not isinstance(intent, dict):

        print(
            f"Invalid intent object: "
            f"{type(intent).__name__}"
        )

        return

    print(
        f"Type           : "
        f"{intent.get('type')}"
    )

    print(
        f"Parameter      : "
        f"{intent.get('parameter')}"
    )

    print(
        f"Sub-parameter  : "
        f"{intent.get('sub_parameter')}"
    )

    print(
        f"Section        : "
        f"{intent.get('section')}"
    )

    print(
        f"Side           : "
        f"{intent.get('side')}"
    )

    print(
        f"Identifier     : "
        f"{intent.get('identifier')}"
    )


# =====================================================================
# PHASE 3.2 RETRIEVAL DISPLAY
# =====================================================================

def print_retrieval_result(output):

    print()
    print("=" * 90)
    print("PHASE 3.2 — RETRIEVAL RESULT")
    print("=" * 90)

    print(
        f"Retrieval mode       : "
        f"{output.get('retrieval_mode')}"
    )

    print(
        f"Result count         : "
        f"{output.get('result_count', 0)}"
    )

    print(
        f"Abstained            : "
        f"{output.get('abstained', False)}"
    )

    print(
        f"Entity scoped        : "
        f"{output.get('entity_scoped', False)}"
    )

    print(
        f"Structured records   : "
        f"{len(output.get('structured_records', []))}"
    )

    # -------------------------------------------------------------
    # MAIN INTENT
    # -------------------------------------------------------------

    intent = output.get("intent")

    if intent:

        print_intent(
            intent,
            "PARSED INTENT"
        )

    # -------------------------------------------------------------
    # MULTI-INTENT DETAILS
    # -------------------------------------------------------------

    intents = output.get("intents")

    if isinstance(intents, list) and intents:

        print()
        print("=" * 90)
        print("MULTI-INTENT DETAILS")
        print("=" * 90)

        for index, intent_data in enumerate(
            intents,
            start=1
        ):

            print(
                f"\nIntent #{index}"
            )

            if isinstance(intent_data, dict):

                print(
                    f"  Query        : "
                    f"{intent_data.get('query')}"
                )

                print(
                    f"  Result count : "
                    f"{intent_data.get('result_count', 0)}"
                )

                print(
                    f"  Abstained    : "
                    f"{intent_data.get('abstained', False)}"
                )

                print(
                    f"  Record IDs   : "
                    f"{intent_data.get('record_ids', [])}"
                )

                inner_intent = (
                    intent_data.get("intent")
                )

                if inner_intent:

                    print(
                        f"  Type         : "
                        f"{inner_intent.get('type')}"
                    )

                    print(
                        f"  Parameter    : "
                        f"{inner_intent.get('parameter')}"
                    )

                    print(
                        f"  Sub-param    : "
                        f"{inner_intent.get('sub_parameter')}"
                    )

                    print(
                        f"  Side         : "
                        f"{inner_intent.get('side')}"
                    )


# =====================================================================
# PHASE 4 CONTEXT DISPLAY
# =====================================================================

def print_context_result(context_result):

    retrieval = (
        context_result.get(
            "retrieval",
            {}
        )
    )

    records = (
        context_result.get(
            "records",
            []
        )
    )

    blocks = (
        context_result.get(
            "context_blocks",
            []
        )
    )

    print()
    print("=" * 90)
    print("PHASE 4 — CONTEXT RESULT")
    print("=" * 90)

    print(
        f"Retrieval mode    : "
        f"{retrieval.get('retrieval_mode')}"
    )

    print(
        f"Intent type       : "
        f"{retrieval.get('intent_type')}"
    )

    print(
        f"Records           : "
        f"{len(records)}"
    )

    print(
        f"Context blocks    : "
        f"{len(blocks)}"
    )

    # -------------------------------------------------------------
    # NO CONTEXT
    # -------------------------------------------------------------

    if not records:

        print()
        print(
            "[EMPTY CONTEXT]"
        )

        print(
            "Context status   : "
            "NO RECORDS"
        )

        return

    # -------------------------------------------------------------
    # RECORDS
    # -------------------------------------------------------------

    print()
    print(
        "CONTEXT RECORDS"
    )

    print("-" * 90)

    for rank, record in enumerate(
        records,
        start=1
    ):

        print_record(
            record,
            rank
        )

    # -------------------------------------------------------------
    # CONTEXT BLOCKS
    # -------------------------------------------------------------

    print()
    print(
        "STRUCTURED CONTEXT BLOCKS"
    )

    print("-" * 90)

    for rank, block in enumerate(
        blocks,
        start=1
    ):

        print()

        print(
            f"[CONTEXT BLOCK {rank}]"
        )

        print(
            f"  Record ID     : "
            f"{block.get('record_id')}"
        )

        print(
            f"  Page          : "
            f"{block.get('page')}"
        )

        print(
            f"  Parameter     : "
            f"{block.get('parameter')}"
        )

        print(
            f"  Sub-parameter : "
            f"{block.get('sub_parameter')}"
        )

        print(
            f"  Side          : "
            f"{block.get('side')}"
        )

        print("  Values        :")

        print_values(
            block.get("values")
        )

    # -------------------------------------------------------------
    # CONTEXT TEXT
    # -------------------------------------------------------------

    context_text = (
        context_result.get(
            "context_text"
        )
    )

    if context_text:

        print()
        print(
            "GENERATED CONTEXT TEXT"
        )

        print("-" * 90)

        print(
            context_text
        )


# =====================================================================
# SINGLE QUERY
# =====================================================================

def test_query(
    adapter,
    context_builder,
    query
):

    print()
    print("#" * 90)
    print("MANUAL RETRIEVAL TEST")
    print("#" * 90)

    print()
    print(
        f"Query:\n{query}"
    )

    # =============================================================
    # PHASE 3.2
    # =============================================================

    output = adapter.search(
        query,
        top_k=5
    )

    # =============================================================
    # IMPORTANT:
    # Phase 3.2 uses `records` / `results`.
    # DO NOT use `flattened_results` here.
    # =============================================================

    records = output.get(
        "records",
        []
    )

    # =============================================================
    # RETRIEVAL SUMMARY
    # =============================================================

    print()
    print("-" * 90)
    print("RETRIEVAL SUMMARY")
    print("-" * 90)

    print(
        f"Intent count       : "
        f"{output.get('intent_count', 1 if output.get('intent') else 0)}"
    )

    print(
        f"Successful intents : "
        f"{output.get('successful_intents', 1 if records else 0)}"
    )

    print(
        f"Abstained intents  : "
        f"{output.get('abstained_intents', 1 if output.get('abstained') else 0)}"
    )

    print(
        f"Retrieved records  : "
        f"{len(records)}"
    )

    print(
        f"Adapter result_count: "
        f"{output.get('result_count', len(records))}"
    )

    # =============================================================
    # RETRIEVED RECORD IDs
    # =============================================================

    if records:

        print()
        print(
            "RETRIEVED RECORD IDs"
        )

        print("-" * 90)

        for rank, record in enumerate(
            records,
            start=1
        ):

            print(
                f"{rank}. "
                f"{record.get('record_id')}"
            )

    else:

        print()
        print(
            "[NO RECORDS FROM PHASE 3.2]"
        )

    # =============================================================
    # DETAILED PHASE 3.2 RESULT
    # =============================================================

    print_retrieval_result(
        output
    )

    # =============================================================
    # PHASE 4
    # =============================================================

    context_result = (
        context_builder.build(
            output
        )
    )

    print_context_result(
        context_result
    )

    # =============================================================
    # FINAL STATUS
    # =============================================================

    print()
    print("-" * 90)

    context_records = (
        context_result.get(
            "records",
            []
        )
    )

    if context_records:

        print(
            "FINAL STATUS : "
            "RETRIEVAL SUCCESS"
        )

    elif records:

        print(
            "FINAL STATUS : "
            "RETRIEVAL SUCCESS / "
            "CONTEXT EMPTY"
        )

    else:

        print(
            "FINAL STATUS : "
            "NO MATCH / ABSTAINED"
        )

    print("-" * 90)


# =====================================================================
# HELP
# =====================================================================

def show_help():

    print()
    print("=" * 90)
    print("EXAMPLE QUERIES")
    print("=" * 90)

    queries = [

        "What is the actuator type?",

        "What is the actuator type for material code TC9765132581?",

        "What is the inlet steam flow?",

        "What is the outlet steam flow?",

        "What is the inlet line size?",

        "What is the outlet line size?",

        "What is the piping class of the outlet steam?",

        "What is the body material?",

        "What is the stem plug material?",

        "What is the cooling fins information?",

        "What are the water conditions?",

        "What is the item of material code TC9765132581?",

        "What is the inlet line size for material code TC9765132581?",

        "What is the outlet line size for material code TC9765132581?",

        "What is the turbine shaft diameter?",

        "What is the motor bearing temperature?",

        "What is the gearbox oil pressure?",

        "What is the actuator type for material code UNKNOWN123?",
    ]

    for index, query in enumerate(
        queries,
        start=1
    ):

        print(
            f"{index:2}. {query}"
        )

    print()


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 90)
    print(
        "PHASE 4 — MANUAL RETRIEVAL TESTER"
    )
    print("=" * 90)

    # =============================================================
    # LOAD ARTIFACTS
    # =============================================================

    print()
    print(
        "Loading Phase 4 artifacts..."
    )

    records = load_records()

    schema_catalog = load_json(
        SCHEMA_PATH
    )

    entity_index = load_json(
        ENTITY_PATH
    )

    print(
        f"Retrieval records : "
        f"{len(records)}"
    )

    print(
        f"Schema catalog    : "
        f"{SCHEMA_PATH}"
    )

    print(
        f"Entity index      : "
        f"{ENTITY_PATH}"
    )

    # =============================================================
    # INITIALIZE ADAPTER
    # =============================================================

    print()
    print(
        "Initializing Phase 3.2 "
        "Intent Retrieval Adapter..."
    )

    adapter = IntentRetrievalAdapter(
        schema_catalog=schema_catalog,
        entity_index=entity_index,
        retrieval_records=records,
    )

    print(
        "Phase 3.2 adapter initialized."
    )

    # =============================================================
    # INITIALIZE CONTEXT BUILDER
    # =============================================================

    print()
    print(
        "Initializing Phase 4 "
        "Context Builder..."
    )

    context_builder = ContextBuilder()

    print(
        "Phase 4 context builder initialized."
    )

    # =============================================================
    # INTERACTIVE LOOP
    # =============================================================

    print()
    print("=" * 90)
    print(
        "INTERACTIVE MANUAL RETRIEVAL"
    )
    print("=" * 90)

    print()
    print(
        "Enter a natural-language query."
    )

    print(
        "Commands:"
    )

    print(
        "  help  -> show example queries"
    )

    print(
        "  exit  -> stop"
    )

    print(
        "  quit  -> stop"
    )

    while True:

        print()

        query = input(
            "Query: "
        ).strip()

        if not query:
            continue

        if query.lower() in {
            "exit",
            "quit"
        }:

            print()
            print(
                "Exiting manual retrieval tester."
            )

            break

        if query.lower() == "help":

            show_help()

            continue

        try:

            test_query(
                adapter,
                context_builder,
                query
            )

        except Exception as error:

            print()
            print("!" * 90)
            print("ERROR")
            print("!" * 90)

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

            print()
            print(
                "You can continue with another query."
            )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()