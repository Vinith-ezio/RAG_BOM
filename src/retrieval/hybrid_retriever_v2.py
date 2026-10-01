"""
Hybrid Retrieval Engine V2
==========================

Field-Aware Hybrid Retrieval

Combines:

1. Structured Retrieval
2. BM25 Retrieval
3. Field-aware candidate filtering
4. Reciprocal Rank Fusion (RRF)

Important:
- Does NOT modify structured_retriever.py
- Does NOT modify bm25_retriever.py
- No LLM
- No embeddings
- No heuristic score boosting
- No hardcoded record IDs
"""

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# EXISTING COMPONENTS
# ---------------------------------------------------------------------

from bm25_retriever import BM25Retriever

from structured_retriever import (
    retrieve,
    load_records,
)

from query_parser import parse_query


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

DEFAULT_TOP_K = 5

DEFAULT_STRUCTURED_K = 10

DEFAULT_BM25_K = 10

DEFAULT_RRF_K = 60


# ---------------------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------------------

def normalize(value):
    """
    Normalize text for field comparison.
    """

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace("–", "-")
        .replace("—", "-")
    )


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

def load_hybrid_records():

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    records = load_records(
        str(DATA_FILE)
    )

    if not records:

        raise ValueError(
            "No records loaded from dataset."
        )

    return records


# ---------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------

def rrf_score(
    rank,
    k=DEFAULT_RRF_K
):
    """
    Reciprocal Rank Fusion.

    RRF(rank) = 1 / (k + rank)
    """

    return 1.0 / (
        k + rank
    )


# ---------------------------------------------------------------------
# FIELD COMPATIBILITY
# ---------------------------------------------------------------------

def field_compatible(
    record,
    parsed
):
    """
    Determine whether a record satisfies
    the structured fields explicitly detected
    in the query.

    Only fields present in the parsed query
    are used as constraints.

    This is NOT a score boost.
    It is candidate filtering.
    """

    # -------------------------------------------------------------
    # ITEM
    # -------------------------------------------------------------

    query_item = parsed.get("item")

    if query_item:

        record_item = record.get("item")

        if normalize(record_item) != normalize(query_item):

            return False

    # -------------------------------------------------------------
    # SECTION
    # -------------------------------------------------------------

    query_section = parsed.get("section")

    if query_section:

        record_section = record.get("section")

        if normalize(record_section) != normalize(query_section):

            return False

    # -------------------------------------------------------------
    # SUBSECTION
    # -------------------------------------------------------------

    query_subsection = parsed.get("subsection")

    if query_subsection:

        record_subsection = record.get("subsection")

        if normalize(record_subsection) != normalize(query_subsection):

            return False

    # -------------------------------------------------------------
    # PARAMETER
    # -------------------------------------------------------------

    query_parameter = parsed.get("parameter")

    if query_parameter:

        record_parameter = record.get("parameter")

        if normalize(record_parameter) != normalize(query_parameter):

            return False

    # -------------------------------------------------------------
    # SUB-PARAMETER
    # -------------------------------------------------------------

    query_sub_parameter = parsed.get(
        "sub_parameter"
    )

    if query_sub_parameter:

        record_sub_parameter = record.get(
            "sub_parameter"
        )

        if normalize(record_sub_parameter) != normalize(
            query_sub_parameter
        ):

            return False

    # -------------------------------------------------------------
    # SIDE
    # -------------------------------------------------------------

    query_side = parsed.get("side")

    if query_side:

        record_side = record.get("side")

        if normalize(record_side) != normalize(query_side):

            return False

    return True


# ---------------------------------------------------------------------
# HYBRID RETRIEVER V2
# ---------------------------------------------------------------------

class HybridRetrieverV2:

    def __init__(
        self,
        records,
        rrf_k=DEFAULT_RRF_K
    ):

        self.records = records

        self.rrf_k = rrf_k

        # ---------------------------------------------------------
        # BM25
        # ---------------------------------------------------------

        self.bm25 = BM25Retriever(
            records
        )

        # ---------------------------------------------------------
        # Record lookup
        # ---------------------------------------------------------

        self.record_lookup = {

            record["record_id"]: record

            for record in records

        }

    # -----------------------------------------------------------------
    # QUERY PARSING
    # -----------------------------------------------------------------

    def parse(self, query):

        return parse_query(
            query,
            self.records
        )

    # -----------------------------------------------------------------
    # STRUCTURED SEARCH
    # -----------------------------------------------------------------

    def structured_search(
        self,
        parsed,
        top_k
    ):

        results = retrieve(

            records=self.records,

            item=parsed.get(
                "item"
            ),

            section=parsed.get(
                "section"
            ),

            subsection=parsed.get(
                "subsection"
            ),

            parameter=parsed.get(
                "parameter"
            ),

            sub_parameter=parsed.get(
                "sub_parameter"
            ),

            side=parsed.get(
                "side"
            ),

            limit=top_k
        )

        return results

    # -----------------------------------------------------------------
    # BM25 SEARCH
    # -----------------------------------------------------------------

    def bm25_search(
        self,
        query,
        top_k
    ):

        return self.bm25.search(
            query,
            top_k=top_k
        )

    # -----------------------------------------------------------------
    # CANDIDATE UNION
    # -----------------------------------------------------------------

    def build_candidate_pool(
        self,
        structured_results,
        bm25_results
    ):
        """
        Combine candidate IDs from both retrievers.
        """

        candidate_ids = []

        seen = set()

        # Structured first
        for result in structured_results:

            record_id = result["record_id"]

            if record_id not in seen:

                seen.add(record_id)

                candidate_ids.append(
                    record_id
                )

        # BM25
        for result in bm25_results:

            record_id = result["record_id"]

            if record_id not in seen:

                seen.add(record_id)

                candidate_ids.append(
                    record_id
                )

        return candidate_ids

    # -----------------------------------------------------------------
    # FIELD-AWARE FILTERING
    # -----------------------------------------------------------------

    def filter_candidates(
        self,
        candidate_ids,
        parsed
    ):
        """
        Keep only candidates compatible with
        explicitly parsed query fields.
        """

        compatible = []

        rejected = []

        for record_id in candidate_ids:

            record = self.record_lookup[
                record_id
            ]

            if field_compatible(
                record,
                parsed
            ):

                compatible.append(
                    record_id
                )

            else:

                rejected.append(
                    record_id
                )

        return compatible, rejected

    # -----------------------------------------------------------------
    # RANK MAP
    # -----------------------------------------------------------------

    @staticmethod
    def build_rank_map(results):

        rank_map = {}

        for rank, result in enumerate(
            results,
            start=1
        ):

            rank_map[
                result["record_id"]
            ] = rank

        return rank_map

    # -----------------------------------------------------------------
    # FIELD-AWARE RRF
    # -----------------------------------------------------------------

    def fuse(
        self,
        structured_results,
        bm25_results,
        compatible_ids
    ):
        """
        Apply RRF only to field-compatible
        candidates.
        """

        compatible_set = set(
            compatible_ids
        )

        structured_rank_map = (
            self.build_rank_map(
                structured_results
            )
        )

        bm25_rank_map = (
            self.build_rank_map(
                bm25_results
            )
        )

        fused_scores = {}

        source_ranks = {}

        # ---------------------------------------------------------
        # STRUCTURED CONTRIBUTION
        # ---------------------------------------------------------

        for record_id in compatible_set:

            if record_id in structured_rank_map:

                rank = structured_rank_map[
                    record_id
                ]

                score = rrf_score(
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

                    + score
                )

                source_ranks.setdefault(
                    record_id,
                    {}
                )

                source_ranks[
                    record_id
                ]["structured_rank"] = rank

        # ---------------------------------------------------------
        # BM25 CONTRIBUTION
        # ---------------------------------------------------------

        for record_id in compatible_set:

            if record_id in bm25_rank_map:

                rank = bm25_rank_map[
                    record_id
                ]

                score = rrf_score(
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

                    + score
                )

                source_ranks.setdefault(
                    record_id,
                    {}
                )

                source_ranks[
                    record_id
                ]["bm25_rank"] = rank

        # ---------------------------------------------------------
        # SORT
        # ---------------------------------------------------------

        ranked = sorted(

            fused_scores.items(),

            key=lambda item: item[1],

            reverse=True

        )

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
        Full Hybrid V2 pipeline.
        """

        # ---------------------------------------------------------
        # PARSE QUERY
        # ---------------------------------------------------------

        parsed = self.parse(
            query
        )

        # ---------------------------------------------------------
        # ABSTENTION
        # ---------------------------------------------------------

        if parsed.get(
            "parameter"
        ) is None:

            return {

                "query": query,

                "parsed": parsed,

                "structured_results": [],

                "bm25_results": [],

                "candidate_ids": [],

                "compatible_ids": [],

                "rejected_ids": [],

                "results": [],

                "abstained": True

            }

        # ---------------------------------------------------------
        # STRUCTURED
        # ---------------------------------------------------------

        structured_results = (
            self.structured_search(
                parsed,
                structured_k
            )
        )

        # ---------------------------------------------------------
        # BM25
        # ---------------------------------------------------------

        bm25_results = (
            self.bm25_search(
                query,
                bm25_k
            )
        )

        # ---------------------------------------------------------
        # CANDIDATE POOL
        # ---------------------------------------------------------

        candidate_ids = (
            self.build_candidate_pool(
                structured_results,
                bm25_results
            )
        )

        # ---------------------------------------------------------
        # FIELD FILTER
        # ---------------------------------------------------------

        (
            compatible_ids,
            rejected_ids
        ) = self.filter_candidates(
            candidate_ids,
            parsed
        )

        # ---------------------------------------------------------
        # FUSION
        # ---------------------------------------------------------

        fused_results = self.fuse(
            structured_results,
            bm25_results,
            compatible_ids
        )

        # ---------------------------------------------------------
        # FINAL TOP-K
        # ---------------------------------------------------------

        final_results = (
            fused_results[:top_k]
        )

        return {

            "query": query,

            "parsed": parsed,

            "structured_results":
                structured_results,

            "bm25_results":
                bm25_results,

            "candidate_ids":
                candidate_ids,

            "compatible_ids":
                compatible_ids,

            "rejected_ids":
                rejected_ids,

            "results":
                final_results,

            "abstained":
                False

        }


# ---------------------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------------------

def print_results(
    output
):

    query = output["query"]

    parsed = output["parsed"]

    structured_results = (
        output["structured_results"]
    )

    bm25_results = (
        output["bm25_results"]
    )

    candidate_ids = (
        output["candidate_ids"]
    )

    compatible_ids = (
        output["compatible_ids"]
    )

    rejected_ids = (
        output["rejected_ids"]
    )

    results = output["results"]

    print()
    print("=" * 90)
    print("HYBRID RETRIEVAL V2")
    print("=" * 90)

    print()
    print(
        f"Query:\n{query}"
    )

    print()
    print("PARSED QUERY")
    print("-" * 90)

    print(
        f"Item          : "
        f"{parsed.get('item')}"
    )

    print(
        f"Section       : "
        f"{parsed.get('section')}"
    )

    print(
        f"Subsection    : "
        f"{parsed.get('subsection')}"
    )

    print(
        f"Parameter     : "
        f"{parsed.get('parameter')}"
    )

    print(
        f"Sub-parameter : "
        f"{parsed.get('sub_parameter')}"
    )

    print(
        f"Side          : "
        f"{parsed.get('side')}"
    )

    print(
        f"Value Field   : "
        f"{parsed.get('value_field')}"
    )

    # -------------------------------------------------------------
    # ABSTENTION
    # -------------------------------------------------------------

    if output["abstained"]:

        print()
        print(
            "ABSTENTION"
        )

        print(
            "No supported parameter "
            "was detected."
        )

        return

    # -------------------------------------------------------------
    # RETRIEVAL STATISTICS
    # -------------------------------------------------------------

    print()
    print("CANDIDATE FILTERING")
    print("-" * 90)

    print(
        f"Structured candidates : "
        f"{len(structured_results)}"
    )

    print(
        f"BM25 candidates       : "
        f"{len(bm25_results)}"
    )

    print(
        f"Combined candidates   : "
        f"{len(candidate_ids)}"
    )

    print(
        f"Compatible candidates : "
        f"{len(compatible_ids)}"
    )

    print(
        f"Rejected candidates   : "
        f"{len(rejected_ids)}"
    )

    # -------------------------------------------------------------
    # FINAL
    # -------------------------------------------------------------

    print()
    print("FINAL RESULTS")
    print("-" * 90)

    if not results:

        print(
            "No compatible records found."
        )

        return

    for result in results:

        record = result[
            "record"
        ]

        print()

        print(
            f"Rank            : "
            f"{result['hybrid_rank']}"
        )

        print(
            f"Record ID       : "
            f"{result['record_id']}"
        )

        print(
            f"RRF Score       : "
            f"{result['hybrid_score']:.6f}"
        )

        print(
            f"Structured Rank : "
            f"{result['structured_rank']}"
        )

        print(
            f"BM25 Rank       : "
            f"{result['bm25_rank']}"
        )

        print(
            f"Page            : "
            f"{record.get('page')}"
        )

        print(
            f"Section         : "
            f"{record.get('section')}"
        )

        print(
            f"Parameter       : "
            f"{record.get('parameter')}"
        )

        print(
            f"Sub-parameter   : "
            f"{record.get('sub_parameter')}"
        )

        print(
            f"Side            : "
            f"{record.get('side')}"
        )

        print(
            f"Values          : "
            f"{record.get('values')}"
        )

        print("-" * 90)


# ---------------------------------------------------------------------
# TEST QUERIES
# ---------------------------------------------------------------------

TEST_QUERIES = [

    (
        'What is the inlet line size of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),

    (
        'What is the inlet steam flow of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),

    (
        'What is the inlet body material of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),

    (
        'What is the material code of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),

    (
        'What is the outlet piping class of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),

    (
        'What is the inlet steam flow of '
        'DSH 6"300RF-INTEG TCV 1"600RF-HART?'
    ),

    (
        'What is the material code of '
        'DSH 12"300RF-INTEG TCV 1"300RF-HART?'
    ),

    (
        'What is the stem plug material of '
        'the inlet steam of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),

    (
        'What is the turbine shaft diameter of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    )
]


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Field-Aware Hybrid "
            "Retrieval V2"
        )
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

    args = parser.parse_args()

    print("=" * 90)
    print("FIELD-AWARE HYBRID RETRIEVAL V2")
    print("=" * 90)

    print()
    print(
        f"Dataset        : {DATA_FILE}"
    )

    records = load_hybrid_records()

    print(
        f"Records        : {len(records)}"
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

    print()
    print(
        "Building Hybrid V2..."
    )

    retriever = HybridRetrieverV2(
        records,
        rrf_k=args.rrf_k
    )

    print(
        "Hybrid V2 ready."
    )

    for query in TEST_QUERIES:

        output = retriever.search(

            query,

            top_k=args.top_k,

            structured_k=args.structured_k,

            bm25_k=args.bm25_k

        )

        print_results(
            output
        )


if __name__ == "__main__":
    main()