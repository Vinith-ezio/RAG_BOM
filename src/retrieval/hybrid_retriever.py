"""
Hybrid Retrieval Engine

Combines:

1. Structured Retrieval
2. BM25 Retrieval

using Reciprocal Rank Fusion (RRF).

Pipeline:

    Natural Language Query
            |
            +----------------------+
            |                      |
            v                      v
      Query Parser              BM25
            |                      |
            v                      v
   Structured Retrieval       BM25 Retrieval
            |                      |
            +----------+-----------+
                       |
                       v
                  RRF Fusion
                       |
                       v
                 Hybrid Top-K

No LLM.
No embeddings.
No heuristic score boosting.
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
# IMPORTS
# =====================================================================

from bm25_retriever import BM25Retriever

from structured_retriever import (
    retrieve,
    load_records,
)

from query_parser import parse_query


# =====================================================================
# CONFIGURATION
# =====================================================================

DEFAULT_TOP_K = 5

DEFAULT_STRUCTURED_K = 10

DEFAULT_BM25_K = 10

DEFAULT_RRF_K = 60


# =====================================================================
# DATA LOADER
# =====================================================================

def load_hybrid_records():
    """
    Load retrieval-enriched records.

    Uses the same load_records() function as the
    existing structured retrieval implementation.
    """

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    records = load_records(
        str(DATA_FILE)
    )

    if not records:

        raise ValueError(
            "No records loaded from retrieval dataset."
        )

    return records


# =====================================================================
# RRF SCORE
# =====================================================================

def rrf_score(
    rank,
    k=DEFAULT_RRF_K
):
    """
    Reciprocal Rank Fusion contribution.

    Formula:

        RRF = 1 / (k + rank)
    """

    return 1.0 / (
        k + rank
    )


# =====================================================================
# HYBRID RETRIEVER
# =====================================================================

class HybridRetriever:

    def __init__(
        self,
        records,
        rrf_k=DEFAULT_RRF_K
    ):

        self.records = records

        self.rrf_k = rrf_k

        # -------------------------------------------------------------
        # BM25 ENGINE
        # -------------------------------------------------------------

        self.bm25 = BM25Retriever(
            records
        )

        # -------------------------------------------------------------
        # RECORD LOOKUP
        # -------------------------------------------------------------

        self.record_lookup = {
            record["record_id"]: record
            for record in records
        }


    # =================================================================
    # STRUCTURED RETRIEVAL
    # =================================================================

    def structured_search(
        self,
        query,
        top_k
    ):
        """
        Structured retrieval pipeline.

        Step 1:
            Parse natural-language query.

        Step 2:
            Convert parsed fields into structured filters.

        Step 3:
            Call the existing structured retriever.
        """

        # -------------------------------------------------------------
        # QUERY PARSING
        # -------------------------------------------------------------

        parsed = parse_query(
            query,
            self.records
        )

        if parsed.get("parameter") is None:
            return []

        results = retrieve(
            records=self.records,
            item=parsed.get("item"),
            section=parsed.get("section"),
            subsection=parsed.get("subsection"),
            parameter=parsed.get("parameter"),
            sub_parameter=parsed.get("sub_parameter"),
            side=parsed.get("side"),
            limit=top_k
        )

        return results


    # =================================================================
    # BM25 RETRIEVAL
    # =================================================================

    def bm25_search(
        self,
        query,
        top_k
    ):
        """
        BM25 lexical retrieval.
        """

        return self.bm25.search(
            query,
            top_k=top_k
        )


    # =================================================================
    # RRF FUSION
    # =================================================================

    def fuse(
        self,
        structured_results,
        bm25_results
    ):
        """
        Fuse structured and BM25 rankings using RRF.

        A record can receive:

            Structured contribution
            +
            BM25 contribution

        if it appears in both result lists.
        """

        fused_scores = {}

        source_ranks = {}


        # =============================================================
        # STRUCTURED RESULTS
        # =============================================================

        for rank, result in enumerate(
            structured_results,
            start=1
        ):

            record_id = result[
                "record_id"
            ]

            contribution = rrf_score(
                rank,
                self.rrf_k
            )

            fused_scores[
                record_id
            ] = (
                fused_scores.get(
                    record_id,
                    0.0
                )
                + contribution
            )

            source_ranks.setdefault(
                record_id,
                {}
            )

            source_ranks[
                record_id
            ][
                "structured_rank"
            ] = rank


        # =============================================================
        # BM25 RESULTS
        # =============================================================

        for rank, result in enumerate(
            bm25_results,
            start=1
        ):

            record_id = result[
                "record_id"
            ]

            contribution = rrf_score(
                rank,
                self.rrf_k
            )

            fused_scores[
                record_id
            ] = (
                fused_scores.get(
                    record_id,
                    0.0
                )
                + contribution
            )

            source_ranks.setdefault(
                record_id,
                {}
            )

            source_ranks[
                record_id
            ][
                "bm25_rank"
            ] = rank


        # =============================================================
        # SORT
        # =============================================================

        ranked = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )


        # =============================================================
        # BUILD FINAL RESULTS
        # =============================================================

        results = []

        for final_rank, (
            record_id,
            score
        ) in enumerate(
            ranked,
            start=1
        ):

            record = self.record_lookup[
                record_id
            ]

            ranks = source_ranks[
                record_id
            ]

            results.append({

                "record_id":
                    record_id,

                "hybrid_rank":
                    final_rank,

                "hybrid_score":
                    score,

                "structured_rank":
                    ranks.get(
                        "structured_rank"
                    ),

                "bm25_rank":
                    ranks.get(
                        "bm25_rank"
                    ),

                "record":
                    record
            })


        return results


    # =================================================================
    # HYBRID SEARCH
    # =================================================================

    def search(
        self,
        query,
        top_k=DEFAULT_TOP_K,
        structured_k=DEFAULT_STRUCTURED_K,
        bm25_k=DEFAULT_BM25_K
    ):
        """
        Execute:

            Query
              |
              +---- Structured Retrieval
              |
              +---- BM25
              |
              +---- RRF
              |
              +---- Final Top-K
        """

        """
    Run structured + BM25 retrieval and fuse using RRF.
    """

    # -------------------------------------------------------------
    # PARSE FIRST
    # -------------------------------------------------------------

        parsed = parse_query(
            query,
            self.records
        )

        # -------------------------------------------------------------
        # UNKNOWN PARAMETER = ABSTAIN
        # -------------------------------------------------------------

        if parsed.get("parameter") is None:

            return [], [], []

        # -------------------------------------------------------------
        # STRUCTURED
        # -------------------------------------------------------------

        structured_results = retrieve(
            records=self.records,
            item=parsed.get("item"),
            section=parsed.get("section"),
            subsection=parsed.get("subsection"),
            parameter=parsed.get("parameter"),
            sub_parameter=parsed.get("sub_parameter"),
            side=parsed.get("side"),
            limit=structured_k
        )

        # -------------------------------------------------------------
        # BM25
        # -------------------------------------------------------------

        bm25_results = self.bm25_search(
            query,
            bm25_k
        )

        # -------------------------------------------------------------
        # FUSION
        # -------------------------------------------------------------

        fused_results = self.fuse(
            structured_results,
            bm25_results
        )

        return (
            fused_results[:top_k],
            structured_results,
            bm25_results
        )

# =====================================================================
# DISPLAY
# =====================================================================

def print_results(
    query,
    hybrid_results,
    structured_results,
    bm25_results
):

    print()
    print("=" * 80)
    print("HYBRID RETRIEVAL")
    print("=" * 80)

    print(
        f"Query: {query}"
    )


    # =================================================================
    # STRUCTURED
    # =================================================================

    print()
    print(
        "STRUCTURED RETRIEVAL"
    )

    if not structured_results:

        print(
            "  No structured results."
        )

    else:

        for rank, result in enumerate(
            structured_results,
            start=1
        ):

            print(
                f"  {rank}. "
                f"{result['record_id']}"
            )


    # =================================================================
    # BM25
    # =================================================================

    print()
    print(
        "BM25 RETRIEVAL"
    )

    if not bm25_results:

        print(
            "  No BM25 results."
        )

    else:

        for rank, result in enumerate(
            bm25_results,
            start=1
        ):

            score = result.get(
                "_bm25_score",
                result.get(
                    "bm25_score",
                    0.0
                )
            )

            print(
                f"  {rank}. "
                f"{result['record_id']} "
                f"(score={score:.4f})"
            )


    # =================================================================
    # HYBRID
    # =================================================================

    print()
    print(
        "HYBRID / RRF RESULTS"
    )

    if not hybrid_results:

        print(
            "  No hybrid results."
        )

        return


    for result in hybrid_results:

        print()

        print(
            f"  Rank            : "
            f"{result['hybrid_rank']}"
        )

        print(
            f"  Record ID       : "
            f"{result['record_id']}"
        )

        print(
            f"  Hybrid Score    : "
            f"{result['hybrid_score']:.6f}"
        )

        print(
            f"  Structured Rank : "
            f"{result['structured_rank']}"
        )

        print(
            f"  BM25 Rank       : "
            f"{result['bm25_rank']}"
        )


        record = result[
            "record"
        ]


        print(
            f"  Page            : "
            f"{record.get('page')}"
        )

        print(
            f"  Section         : "
            f"{record.get('section')}"
        )

        print(
            f"  Parameter       : "
            f"{record.get('parameter')}"
        )

        print(
            f"  Sub-parameter   : "
            f"{record.get('sub_parameter')}"
        )

        print(
            f"  Side            : "
            f"{record.get('side')}"
        )

        print(
            f"  Values          : "
            f"{record.get('values')}"
        )

        print(
            "-" * 80
        )


# =====================================================================
# TEST QUERIES
# =====================================================================

TEST_QUERIES = [

    # -------------------------------------------------------------
    # Q1
    # -------------------------------------------------------------

    (
        'What is the inlet line size of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),


    # -------------------------------------------------------------
    # Q2
    # -------------------------------------------------------------

    (
        'What is the inlet steam flow of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),


    # -------------------------------------------------------------
    # Q3
    # -------------------------------------------------------------

    (
        'What is the inlet body material of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),


    # -------------------------------------------------------------
    # Q4
    # -------------------------------------------------------------

    (
        'What is the material code of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),


    # -------------------------------------------------------------
    # Q5
    # -------------------------------------------------------------

    (
        'What is the outlet piping class of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),


    # -------------------------------------------------------------
    # Q6
    # -------------------------------------------------------------

    (
        'What is the inlet steam flow of '
        'DSH 6"300RF-INTEG TCV 1"600RF-HART?'
    ),


    # -------------------------------------------------------------
    # Q7
    # -------------------------------------------------------------

    (
        'What is the material code of '
        'DSH 12"300RF-INTEG TCV 1"300RF-HART?'
    ),


    # -------------------------------------------------------------
    # Q8
    # -------------------------------------------------------------

    (
        'What is the stem plug material of '
        'the inlet steam of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),


    # -------------------------------------------------------------
    # Q9 - NEGATIVE
    # -------------------------------------------------------------

    (
        'What is the turbine shaft diameter of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),
]


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Hybrid Structured + BM25 Retrieval"
        )
    )


    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Final hybrid result count"
    )


    parser.add_argument(
        "--structured-k",
        type=int,
        default=DEFAULT_STRUCTURED_K,
        help="Structured retrieval candidate count"
    )


    parser.add_argument(
        "--bm25-k",
        type=int,
        default=DEFAULT_BM25_K,
        help="BM25 candidate count"
    )


    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help="RRF constant"
    )


    args = parser.parse_args()


    # =================================================================
    # START
    # =================================================================

    print("=" * 80)
    print("HYBRID RETRIEVAL ENGINE")
    print("=" * 80)

    print()

    print(
        f"Dataset        : "
        f"{DATA_FILE}"
    )


    # =================================================================
    # LOAD
    # =================================================================

    records = load_hybrid_records()

    print(
        f"Records        : "
        f"{len(records)}"
    )

    print(
        f"Structured Top : "
        f"{args.structured_k}"
    )

    print(
        f"BM25 Top       : "
        f"{args.bm25_k}"
    )

    print(
        f"Final Top      : "
        f"{args.top_k}"
    )

    print(
        f"RRF K          : "
        f"{args.rrf_k}"
    )


    # =================================================================
    # BUILD
    # =================================================================

    print()

    print(
        "Building hybrid retriever..."
    )


    retriever = HybridRetriever(
        records,
        rrf_k=args.rrf_k
    )


    print(
        "Hybrid retriever ready."
    )


    # =================================================================
    # TEST
    # =================================================================

    for query in TEST_QUERIES:

        (
            hybrid_results,
            structured_results,
            bm25_results
        ) = retriever.search(

            query,

            top_k=args.top_k,

            structured_k=args.structured_k,

            bm25_k=args.bm25_k
        )


        print_results(

            query,

            hybrid_results,

            structured_results,

            bm25_results
        )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()