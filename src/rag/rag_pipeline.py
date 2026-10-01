"""
RAG Pipeline - Phase 6

Architecture:

    User Query
        ↓
    Hybrid V3
        ↓
    Retrieved Record IDs
        ↓
    Full Record Resolution
        ↓
    Context Builder
        ↓
    Structured Context

No LLM yet.
"""

import sys
from pathlib import Path


# ================================================================
# PATH SETUP
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETRIEVAL_DIR = (
    PROJECT_ROOT
    / "src"
    / "retrieval"
)

CONTEXT_DIR = (
    PROJECT_ROOT
    / "src"
    / "context"
)

sys.path.insert(
    0,
    str(RETRIEVAL_DIR)
)

sys.path.insert(
    0,
    str(CONTEXT_DIR)
)


# ================================================================
# IMPORTS
# ================================================================

from hybrid_retriever_v3 import (
    NativeMultiIntentHybridRetriever,
    load_v3_records
)

from context_builder import (
    ContextBuilder
)


# ================================================================
# CONFIGURATION
# ================================================================

TOP_K = 5
STRUCTURED_K = 10
BM25_K = 10
RRF_K = 60

DOCUMENT_NAME = (
    "Instrument Specification - Desuperheater"
)


# ================================================================
# RAG PIPELINE
# ================================================================

class RAGPipeline:

    def __init__(
        self,
        top_k=TOP_K,
        structured_k=STRUCTURED_K,
        bm25_k=BM25_K,
        rrf_k=RRF_K
    ):

        self.top_k = top_k
        self.structured_k = structured_k
        self.bm25_k = bm25_k
        self.rrf_k = rrf_k

        # --------------------------------------------------------
        # Load complete records
        # --------------------------------------------------------

        print(
            "Loading retrieval records..."
        )

        records = load_v3_records()

        print(
            f"Records loaded: {len(records)}"
        )

        # --------------------------------------------------------
        # Create record lookup
        # --------------------------------------------------------

        self.record_lookup = {
            record["record_id"]: record
            for record in records
            if record.get("record_id")
        }

        # --------------------------------------------------------
        # Frozen Hybrid V3
        # --------------------------------------------------------

        self.retriever = (
            NativeMultiIntentHybridRetriever(
                records,
                rrf_k=rrf_k
            )
        )

        # --------------------------------------------------------
        # Context Builder
        # --------------------------------------------------------

        self.context_builder = (
            ContextBuilder(
                document_name=DOCUMENT_NAME
            )
        )


    # ============================================================
    # RESOLVE FULL RECORDS
    # ============================================================

    def resolve_records(
        self,
        retrieval_output
    ):

        flattened_results = (
            retrieval_output.get(
                "flattened_results",
                []
            )
        )

        full_records = []

        unresolved_ids = []

        for result in flattened_results:

            record_id = result.get(
                "record_id"
            )

            if not record_id:
                continue

            record = self.record_lookup.get(
                record_id
            )

            if record is None:

                unresolved_ids.append(
                    record_id
                )

                continue

            full_records.append(
                record
            )

        return (
            full_records,
            unresolved_ids
        )


    # ============================================================
    # RETRIEVAL
    # ============================================================

    def retrieve(
        self,
        query
    ):

        return self.retriever.search(
            query,
            top_k=self.top_k,
            structured_k=self.structured_k,
            bm25_k=self.bm25_k
        )


    # ============================================================
    # RUN PIPELINE
    # ============================================================

    def run(
        self,
        query
    ):

        if not isinstance(
            query,
            str
        ) or not query.strip():

            raise ValueError(
                "Query must be a non-empty string."
            )

        # --------------------------------------------------------
        # Step 1: Hybrid V3
        # --------------------------------------------------------

        retrieval_output = (
            self.retrieve(
                query
            )
        )

        # --------------------------------------------------------
        # Step 2: Resolve IDs
        # --------------------------------------------------------

        (
            full_records,
            unresolved_ids
        ) = self.resolve_records(
            retrieval_output
        )

        # --------------------------------------------------------
        # Step 3: Build context
        # --------------------------------------------------------

        context_output = (
            self.context_builder.build(
                full_records
            )
        )

        # --------------------------------------------------------
        # Step 4: Final response
        # --------------------------------------------------------

        return {

            "query":
                query,

            "pipeline":
                "Hybrid V3 → "
                "Record Resolution → "
                "Context Builder",

            "retrieval":
                retrieval_output,

            "resolved_records":
                full_records,

            "unresolved_record_ids":
                unresolved_ids,

            "context":
                context_output,

            "answer":
                None,

            "generation":
                {

                    "enabled":
                        False,

                    "model":
                        None,

                    "reason":
                        "LLM generation is "
                        "not enabled in this phase."

                }

        }


# ================================================================
# PRINT RESULT
# ================================================================

def print_response(
    response
):

    print()
    print("=" * 90)
    print("RAG PIPELINE RESULT")
    print("=" * 90)

    print()
    print("Query:")
    print(
        response["query"]
    )

    # ------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------

    retrieval = response[
        "retrieval"
    ]

    print()
    print("-" * 90)
    print("RETRIEVAL")
    print("-" * 90)

    print(
        f"Intent count       : "
        f"{retrieval.get('intent_count')}"
    )

    print(
        f"Successful intents : "
        f"{retrieval.get('successful_intents')}"
    )

    print(
        f"Abstained intents  : "
        f"{retrieval.get('abstained_intents')}"
    )

    flattened = retrieval.get(
        "flattened_results",
        []
    )

    print(
        f"Retrieved records  : "
        f"{len(flattened)}"
    )

    for index, result in enumerate(
        flattened,
        start=1
    ):

        print(
            f"  {index}. "
            f"{result.get('record_id')}"
        )


    # ------------------------------------------------------------
    # Record Resolution
    # ------------------------------------------------------------

    resolved = response[
        "resolved_records"
    ]

    unresolved = response[
        "unresolved_record_ids"
    ]

    print()
    print("-" * 90)
    print("RECORD RESOLUTION")
    print("-" * 90)

    print(
        f"Resolved records   : "
        f"{len(resolved)}"
    )

    print(
        f"Unresolved IDs     : "
        f"{len(unresolved)}"
    )

    if unresolved:

        print(
            f"Unresolved: "
            f"{unresolved}"
        )


    # ------------------------------------------------------------
    # Context
    # ------------------------------------------------------------

    context = response[
        "context"
    ]

    print()
    print("-" * 90)
    print("CONTEXT")
    print("-" * 90)

    print(
        f"Retrieved records : "
        f"{context['retrieved_count']}"
    )

    print(
        f"Valid records     : "
        f"{context['valid_count']}"
    )

    print(
        f"Unique records    : "
        f"{context['unique_count']}"
    )

    print(
        f"Invalid records   : "
        f"{context['invalid_count']}"
    )

    print()

    if context["context_text"]:

        print(
            context["context_text"]
        )

    else:

        print(
            "[EMPTY CONTEXT]"
        )


    # ------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------

    print()
    print("-" * 90)
    print("ANSWER GENERATION")
    print("-" * 90)

    generation = response[
        "generation"
    ]

    print(
        f"Enabled : "
        f"{generation['enabled']}"
    )

    print(
        f"Model   : "
        f"{generation['model']}"
    )

    print(
        f"Reason  : "
        f"{generation['reason']}"
    )

    print()
    print("=" * 90)


# ================================================================
# TEST QUERIES
# ================================================================

TEST_QUERIES = [

    (
        'What is the inlet steam flow of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),

    (
        'What is the line size of the inlet steam '
        'and the piping class of the outlet steam of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),

    (
        'What is the inlet steam flow, outlet steam flow, '
        'outlet piping class, and material code of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),

    (
        'What is the turbine shaft diameter of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    )

]


# ================================================================
# MAIN
# ================================================================

def main():

    print()
    print("=" * 90)
    print("LLM-FREE RAG PIPELINE")
    print("=" * 90)

    pipeline = RAGPipeline()

    for query in TEST_QUERIES:

        response = pipeline.run(
            query
        )

        print_response(
            response
        )


if __name__ == "__main__":
    main()