"""
Native Multi-Intent Hybrid Retrieval V3

Architecture:

User Query
    ↓
Multi-Intent Splitter
    ↓
Query Parser per Intent
    ↓
Hybrid V2 per Intent
    ↓
Grouped Intent Results
    ↓
Deduplicated Results

No embeddings.
No LLM.
No heuristic score boosting.

V3 reuses the existing Hybrid V2 retrieval engine.
"""

import argparse
import json
import sys
from pathlib import Path


# =====================================================================
# PATH SETUP
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETRIEVAL_DIR = (
    PROJECT_ROOT
    / "src"
    / "retrieval"
)

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "retrieval_enriched.json"
)

sys.path.insert(
    0,
    str(RETRIEVAL_DIR)
)


# =====================================================================
# EXISTING COMPONENTS
# =====================================================================

from structured_retriever import load_records
from query_parser import parse_query

from hybrid_retriever_v2 import (
    HybridRetrieverV2
)

from multi_intent_retriever import (
    split_multi_intent_query,
    detect_item
)


# =====================================================================
# CONFIGURATION
# =====================================================================

DEFAULT_TOP_K = 5
DEFAULT_STRUCTURED_K = 10
DEFAULT_BM25_K = 10
DEFAULT_RRF_K = 60


# =====================================================================
# NATIVE MULTI-INTENT HYBRID RETRIEVER
# =====================================================================

class NativeMultiIntentHybridRetriever:

    def __init__(
        self,
        records,
        rrf_k=DEFAULT_RRF_K
    ):

        self.records = records

        self.hybrid = HybridRetrieverV2(
            records,
            rrf_k=rrf_k
        )


    # -----------------------------------------------------------------
    # PREPARE INTENT QUERY
    # -----------------------------------------------------------------

    def _prepare_intent_query(
        self,
        intent_query,
        original_query
    ):
        """
        Some intent fragments may not contain the item identifier.

        Recover the item from the original query and append it only
        when the intent fragment does not already contain it.
        """

        item = detect_item(
            original_query,
            self.records
        )

        if not item:
            return intent_query

        normalized_intent = (
            intent_query.lower()
        )

        normalized_item = (
            item.lower()
        )

        if normalized_item not in normalized_intent:

            return (
                f"{intent_query} "
                f"{item}"
            )

        return intent_query


    # -----------------------------------------------------------------
    # SEARCH
    # -----------------------------------------------------------------

    def search(
        self,
        query,
        top_k=DEFAULT_TOP_K,
        structured_k=DEFAULT_STRUCTURED_K,
        bm25_k=DEFAULT_BM25_K
    ):
        """
        Native multi-intent retrieval.

        The evaluator does NOT split the query.

        This method:

            1. Splits the query
            2. Prepares each intent
            3. Runs Hybrid V2 for each intent
            4. Uses the actual V2 result dictionary
            5. Groups results by intent
            6. Creates flattened unique results
        """

        # =============================================================
        # MULTI-INTENT SPLITTING
        # =============================================================

        raw_intents = split_multi_intent_query(
            query
        )

        if not raw_intents:

            raw_intents = [
                query
            ]


        intents = []


        # =============================================================
        # PROCESS EACH INTENT
        # =============================================================

        for index, raw_intent in enumerate(
            raw_intents,
            start=1
        ):

            raw_intent = (
                raw_intent.strip()
            )

            # ---------------------------------------------------------
            # PREPARE QUERY
            # ---------------------------------------------------------

            search_query = (
                self._prepare_intent_query(
                    raw_intent,
                    query
                )
            )


            # ---------------------------------------------------------
            # HYBRID V2
            # ---------------------------------------------------------

            v2_output = self.hybrid.search(

                search_query,

                top_k=top_k,

                structured_k=structured_k,

                bm25_k=bm25_k
            )


            # ---------------------------------------------------------
            # USE V2 OUTPUT DIRECTLY
            # ---------------------------------------------------------

            parsed = v2_output.get(
                "parsed",
                {}
            )

            hybrid_results = v2_output.get(
                "results",
                []
            )

            structured_results = v2_output.get(
                "structured_results",
                []
            )

            bm25_results = v2_output.get(
                "bm25_results",
                []
            )

            abstained = v2_output.get(
                "abstained",
                False
            )


            # ---------------------------------------------------------
            # STORE INTENT
            # ---------------------------------------------------------

            intents.append({

                "intent_index":
                    index,

                "query":
                    raw_intent,

                "search_query":
                    search_query,

                "parsed":
                    parsed,

                "abstained":
                    abstained,

                "results":
                    hybrid_results,

                "structured_results":
                    structured_results,

                "bm25_results":
                    bm25_results

            })


        # =============================================================
        # FLATTEN UNIQUE RESULTS
        # =============================================================

        flattened = []

        seen = set()


        for intent in intents:

            for result in intent[
                "results"
            ]:

                record_id = (
                    result[
                        "record_id"
                    ]
                )


                if record_id in seen:
                    continue


                seen.add(
                    record_id
                )


                flattened.append({

                    "record_id":
                        record_id,

                    "intent_index":
                        intent[
                            "intent_index"
                        ],

                    "intent_query":
                        intent[
                            "query"
                        ],

                    "hybrid_result":
                        result

                })


        # =============================================================
        # SUMMARY
        # =============================================================

        successful_intents = sum(

            1

            for intent in intents

            if (
                not intent[
                    "abstained"
                ]

                and intent[
                    "results"
                ]
            )
        )


        abstained_intents = sum(

            1

            for intent in intents

            if intent[
                "abstained"
            ]
        )


        # =============================================================
        # FINAL OUTPUT
        # =============================================================

        return {

            "query":
                query,

            "intent_count":
                len(intents),

            "successful_intents":
                successful_intents,

            "abstained_intents":
                abstained_intents,

            "intents":
                intents,

            "flattened_results":
                flattened[
                    :top_k
                ]

        }


# =====================================================================
# DATASET LOADER
# =====================================================================

def load_v3_records():

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{DATA_FILE}"
        )


    records = load_records(
        str(DATA_FILE)
    )


    if not records:

        raise ValueError(
            "No records loaded from dataset."
        )


    return records


# =====================================================================
# DISPLAY
# =====================================================================

def print_result(
    output
):

    print(
        "=" * 90
    )

    print(
        "NATIVE MULTI-INTENT "
        "HYBRID RETRIEVAL V3"
    )

    print(
        "=" * 90
    )


    print(
        f"Query        : "
        f"{output['query']}"
    )

    print(
        f"Intent count : "
        f"{output['intent_count']}"
    )

    print(
        f"Successful   : "
        f"{output['successful_intents']}"
    )

    print(
        f"Abstained    : "
        f"{output['abstained_intents']}"
    )


    # ================================================================
    # INTENTS
    # ================================================================

    for intent in output[
        "intents"
    ]:

        print()

        print(
            "-" * 90
        )

        print(
            f"INTENT "
            f"{intent['intent_index']}"
        )

        print(
            "-" * 90
        )


        print(
            f"Query        : "
            f"{intent['query']}"
        )

        print(
            f"Search query : "
            f"{intent['search_query']}"
        )


        parsed = intent[
            "parsed"
        ]


        print(
            f"Item         : "
            f"{parsed.get('item')}"
        )

        print(
            f"Section      : "
            f"{parsed.get('section')}"
        )

        print(
            f"Subsection   : "
            f"{parsed.get('subsection')}"
        )

        print(
            f"Parameter    : "
            f"{parsed.get('parameter')}"
        )

        print(
            f"Sub-parameter: "
            f"{parsed.get('sub_parameter')}"
        )

        print(
            f"Side         : "
            f"{parsed.get('side')}"
        )

        print(
            f"Value field  : "
            f"{parsed.get('value_field')}"
        )


        # ------------------------------------------------------------
        # ABSTENTION
        # ------------------------------------------------------------

        if intent[
            "abstained"
        ]:

            print()

            print(
                "ABSTENTION"
            )

            print(
                "No supported "
                "parameter detected."
            )

            continue


        # ------------------------------------------------------------
        # RESULTS
        # ------------------------------------------------------------

        print()

        print(
            f"Results      : "
            f"{len(intent['results'])}"
        )


        for rank, result in enumerate(
            intent[
                "results"
            ],
            start=1
        ):

            record = result[
                "record"
            ]


            print(
                f"  {rank}. "
                f"{result['record_id']} "
                f"(RRF="
                f"{result['hybrid_score']:.6f})"
            )


            print(
                f"     page="
                f"{record.get('page')} "
                f"row="
                f"{record.get('row_number')} "
                f"parameter="
                f"{record.get('parameter')} "
                f"sub_parameter="
                f"{record.get('sub_parameter')} "
                f"side="
                f"{record.get('side')}"
            )


            print(
                f"     values="
                f"{record.get('values')}"
            )


    # ================================================================
    # FLATTENED RESULTS
    # ================================================================

    print()

    print(
        "=" * 90
    )

    print(
        "FLATTENED UNIQUE RESULTS"
    )

    print(
        "=" * 90
    )


    if not output[
        "flattened_results"
    ]:

        print(
            "No results."
        )

    else:

        for rank, result in enumerate(
            output[
                "flattened_results"
            ],
            start=1
        ):

            print(
                f"{rank}. "
                f"{result['record_id']} "
                f"(intent="
                f"{result['intent_index']})"
            )


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Native multi-intent "
            "hybrid retrieval V3"
        )
    )


    parser.add_argument(
        "--query",
        type=str,
        default=None
    )


    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K
    )


    parser.add_argument(
        "--structured-k",
        type=int,
        default=DEFAULT_STRUCTURED_K
    )


    parser.add_argument(
        "--bm25-k",
        type=int,
        default=DEFAULT_BM25_K
    )


    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K
    )


    parser.add_argument(
        "--json",
        action="store_true"
    )


    args = parser.parse_args()


    # ================================================================
    # LOAD DATA
    # ================================================================

    records = load_v3_records()


    # ================================================================
    # CREATE RETRIEVER
    # ================================================================

    retriever = (
        NativeMultiIntentHybridRetriever(
            records,
            rrf_k=args.rrf_k
        )
    )


    # ================================================================
    # TEST QUERIES
    # ================================================================

    if args.query:

        queries = [
            args.query
        ]

    else:

        queries = [

            (
                'What is the line size of '
                'the inlet steam and the '
                'piping class of the outlet '
                'steam of DSH 6"300RF-INTEG '
                'TCV 1"300RF-HART?'
            ),

            (
                'What is the inlet steam flow, '
                'outlet steam line size, and '
                'inlet body material of '
                'DSH 6"300RF-INTEG '
                'TCV 1"300RF-HART?'
            ),

            (
                'What is the inlet steam flow, '
                'outlet steam flow, outlet '
                'piping class, and material '
                'code of DSH 6"300RF-INTEG '
                'TCV 1"300RF-HART?'
            ),

            (
                'What is the turbine shaft '
                'diameter of DSH 6"300RF-INTEG '
                'TCV 1"300RF-HART?'
            )

        ]


    # ================================================================
    # EXECUTE
    # ================================================================

    outputs = []


    for query in queries:

        output = retriever.search(

            query,

            top_k=args.top_k,

            structured_k=args.structured_k,

            bm25_k=args.bm25_k

        )


        outputs.append(
            output
        )


        if not args.json:

            print_result(
                output
            )


    # ================================================================
    # JSON OUTPUT
    # ================================================================

    if args.json:

        print(
            json.dumps(
                outputs,
                indent=2,
                default=str
            )
        )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()