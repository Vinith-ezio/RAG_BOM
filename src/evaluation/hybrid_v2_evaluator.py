"""
Hybrid Retrieval V2 Evaluator
=============================

Evaluates:

- Field-aware Hybrid Retrieval V2
- Structured + BM25 + RRF
- Single-intent retrieval
- Negative query abstention

Primary metrics:

1. Precision@1
2. Hit@1
3. Hit@3
4. Hit@5
5. MRR
6. Recall@5

Secondary:

7. Exact Match
8. Negative Abstention

Multi-intent queries are intentionally not used
as proof of native multi-intent retrieval.
"""

import json
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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "hybrid_v2_evaluation.json"
)

sys.path.insert(
    0,
    str(RETRIEVAL_DIR)
)


# ---------------------------------------------------------------------
# IMPORT V2
# ---------------------------------------------------------------------

from hybrid_retriever_v2 import (
    HybridRetrieverV2,
    load_hybrid_records,
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

TOP_K = 5

STRUCTURED_K = 10

BM25_K = 10

RRF_K = 60


# ---------------------------------------------------------------------
# GOLDEN DATASET
# ---------------------------------------------------------------------

GOLDEN_QUERIES = [

    # -------------------------------------------------------------
    # BASIC
    # -------------------------------------------------------------

    {
        "id": "Q001",
        "category": "basic",
        "query": (
            'What is the inlet line size of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": [
            "p01_r8_inlet"
        ]
    },

    {
        "id": "Q002",
        "category": "basic",
        "query": (
            'What is the inlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": [
            "p01_r12_inlet"
        ]
    },

    {
        "id": "Q003",
        "category": "basic",
        "query": (
            'What is the inlet body material of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": [
            "p01_r17_inlet"
        ]
    },

    {
        "id": "Q004",
        "category": "basic",
        "query": (
            'What is the material code of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": [
            "p01_r58"
        ]
    },

    # -------------------------------------------------------------
    # SIDE
    # -------------------------------------------------------------

    {
        "id": "Q005",
        "category": "side",
        "query": (
            'What is the outlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": [
            "p01_r12_outlet"
        ]
    },

    {
        "id": "Q006",
        "category": "side",
        "query": (
            'What is the outlet line size of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": [
            "p01_r8_outlet"
        ]
    },

    {
        "id": "Q007",
        "category": "sub_parameter",
        "query": (
            'What is the outlet piping class of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": [
            "p01_r8_outlet"
        ]
    },

    # -------------------------------------------------------------
    # CROSS PAGE
    # -------------------------------------------------------------

    {
        "id": "Q008",
        "category": "cross_page",
        "query": (
            'What is the inlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"600RF-HART?'
        ),
        "expected": [
            "p02_r12_inlet"
        ]
    },

    {
        "id": "Q009",
        "category": "cross_page",
        "query": (
            'What is the outlet piping class of '
            'DSH 6"300RF-INTEG TCV 1"600RF-HART?'
        ),
        "expected": [
            "p02_r8_outlet"
        ]
    },

    {
        "id": "Q010",
        "category": "cross_page",
        "query": (
            'What is the inlet line size of '
            'DSH 12"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": [
            "p05_r8_inlet"
        ]
    },

    {
        "id": "Q011",
        "category": "cross_page",
        "query": (
            'What is the material code of '
            'DSH 12"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": [
            "p05_r58"
        ]
    },

    # -------------------------------------------------------------
    # VALUE FIELD
    # -------------------------------------------------------------

    {
        "id": "Q012",
        "category": "value_field",
        "query": (
            'What is the normal inlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": [
            "p01_r12_inlet"
        ]
    },

    {
        "id": "Q013",
        "category": "value_field",
        "query": (
            'What is the maximum inlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": [
            "p01_r12_inlet"
        ]
    },

    {
        "id": "Q014",
        "category": "value_field",
        "query": (
            'What is the minimum inlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": [
            "p01_r12_inlet"
        ]
    },

    # -------------------------------------------------------------
    # SUB PARAMETER
    # -------------------------------------------------------------

    {
        "id": "Q015",
        "category": "sub_parameter",
        "query": (
            'What is the inlet piping class of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": [
            "p01_r8_inlet"
        ]
    },

    {
        "id": "Q016",
        "category": "sub_parameter",
        "query": (
            'What is the stem plug material of '
            'the inlet steam of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": [
            "p01_r17_inlet"
        ]
    },

    # -------------------------------------------------------------
    # NEGATIVE
    # -------------------------------------------------------------

    {
        "id": "Q020",
        "category": "negative",
        "query": (
            'What is the turbine shaft diameter of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": []
    }
]


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

def reciprocal_rank(
    retrieved,
    expected
):

    if not expected:
        return 0.0

    expected_set = set(
        expected
    )

    for rank, record_id in enumerate(
        retrieved,
        start=1
    ):

        if record_id in expected_set:

            return 1.0 / rank

    return 0.0


def hit_at_k(
    retrieved,
    expected,
    k
):

    if not expected:
        return len(retrieved) == 0

    expected_set = set(
        expected
    )

    return any(
        record_id in expected_set
        for record_id in retrieved[:k]
    )


def recall_at_k(
    retrieved,
    expected,
    k
):

    if not expected:
        return 1.0 if not retrieved else 0.0

    expected_set = set(
        expected
    )

    retrieved_set = set(
        retrieved[:k]
    )

    if not expected_set:
        return 0.0

    return (
        len(
            expected_set
            & retrieved_set
        )
        /
        len(expected_set)
    )


def precision_at_1(
    retrieved,
    expected
):

    if not expected:
        return 1.0 if not retrieved else 0.0

    if not retrieved:
        return 0.0

    return (
        1.0
        if retrieved[0] in set(expected)
        else 0.0
    )


def exact_match(
    retrieved,
    expected
):

    return (
        set(retrieved)
        ==
        set(expected)
    )


# ---------------------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------------------

def main():

    print("=" * 90)
    print("HYBRID RETRIEVAL V2 EVALUATION")
    print("=" * 90)

    print()
    print(
        f"Golden tests : "
        f"{len(GOLDEN_QUERIES)}"
    )

    print(
        f"Top-K        : "
        f"{TOP_K}"
    )

    print(
        f"Structured K : "
        f"{STRUCTURED_K}"
    )

    print(
        f"BM25 K       : "
        f"{BM25_K}"
    )

    print(
        f"RRF K        : "
        f"{RRF_K}"
    )

    # -------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------

    print()
    print(
        "Loading retrieval dataset..."
    )

    records = load_hybrid_records()

    print(
        f"Loaded records : "
        f"{len(records)}"
    )

    # -------------------------------------------------------------
    # BUILD
    # -------------------------------------------------------------

    print()
    print(
        "Building Hybrid V2..."
    )

    retriever = HybridRetrieverV2(
        records,
        rrf_k=RRF_K
    )

    print(
        "Hybrid V2 ready."
    )

    # -------------------------------------------------------------
    # METRIC STORAGE
    # -------------------------------------------------------------

    evaluation_results = []

    precision1_values = []

    hit1_values = []

    hit3_values = []

    hit5_values = []

    mrr_values = []

    recall5_values = []

    exact_values = []

    negative_total = 0

    negative_abstained = 0

    # -------------------------------------------------------------
    # RUN TESTS
    # -------------------------------------------------------------

    for test in GOLDEN_QUERIES:

        query_id = test["id"]

        category = test["category"]

        query = test["query"]

        expected = test["expected"]

        output = retriever.search(

            query,

            top_k=TOP_K,

            structured_k=STRUCTURED_K,

            bm25_k=BM25_K

        )

        retrieved = [

            result["record_id"]

            for result in output["results"]

        ]

        # ---------------------------------------------------------
        # METRICS
        # ---------------------------------------------------------

        p1 = precision_at_1(
            retrieved,
            expected
        )

        h1 = hit_at_k(
            retrieved,
            expected,
            1
        )

        h3 = hit_at_k(
            retrieved,
            expected,
            3
        )

        h5 = hit_at_k(
            retrieved,
            expected,
            5
        )

        mrr = reciprocal_rank(
            retrieved,
            expected
        )

        recall5 = recall_at_k(
            retrieved,
            expected,
            5
        )

        exact = exact_match(
            retrieved,
            expected
        )

        # ---------------------------------------------------------
        # NEGATIVE
        # ---------------------------------------------------------

        if category == "negative":

            negative_total += 1

            if output["abstained"]:

                negative_abstained += 1

        # ---------------------------------------------------------
        # STORE
        # ---------------------------------------------------------

        precision1_values.append(
            p1
        )

        hit1_values.append(
            float(h1)
        )

        hit3_values.append(
            float(h3)
        )

        hit5_values.append(
            float(h5)
        )

        mrr_values.append(
            mrr
        )

        recall5_values.append(
            recall5
        )

        exact_values.append(
            float(exact)
        )

        result = {

            "id":
                query_id,

            "category":
                category,

            "query":
                query,

            "expected":
                expected,

            "retrieved":
                retrieved,

            "precision_at_1":
                p1,

            "hit_at_1":
                h1,

            "hit_at_3":
                h3,

            "hit_at_5":
                h5,

            "mrr":
                mrr,

            "recall_at_5":
                recall5,

            "exact_match":
                exact,

            "abstained":
                output["abstained"],

            "candidate_count":
                len(
                    output[
                        "candidate_ids"
                    ]
                ),

            "compatible_count":
                len(
                    output[
                        "compatible_ids"
                    ]
                ),

            "rejected_count":
                len(
                    output[
                        "rejected_ids"
                    ]
                )

        }

        evaluation_results.append(
            result
        )

        # ---------------------------------------------------------
        # STATUS
        # ---------------------------------------------------------

        if category == "negative":

            passed = (
                output["abstained"]
                and not retrieved
            )

        else:

            passed = (
                h1
                and recall5 == 1.0
            )

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print()
        print("-" * 90)

        print(
            f"{query_id} | "
            f"{category} | "
            f"{status}"
        )

        print(
            f"Query     : "
            f"{query}"
        )

        print(
            f"Expected  : "
            f"{expected}"
        )

        print(
            f"Retrieved : "
            f"{retrieved}"
        )

        print(
            f"P@1       : "
            f"{p1:.2%}"
        )

        print(
            f"Hit@1     : "
            f"{h1}"
        )

        print(
            f"Hit@3     : "
            f"{h3}"
        )

        print(
            f"Hit@5     : "
            f"{h5}"
        )

        print(
            f"MRR       : "
            f"{mrr:.4f}"
        )

        print(
            f"Recall@5  : "
            f"{recall5:.2%}"
        )

        print(
            f"Exact     : "
            f"{exact}"
        )

        print(
            f"Candidates: "
            f"{len(output['candidate_ids'])}"
        )

        print(
            f"Compatible: "
            f"{len(output['compatible_ids'])}"
        )

        print(
            f"Rejected  : "
            f"{len(output['rejected_ids'])}"
        )

    # -------------------------------------------------------------
    # AGGREGATE
    # -------------------------------------------------------------

    def average(values):

        if not values:
            return 0.0

        return (
            sum(values)
            /
            len(values)
        )

    precision1 = average(
        precision1_values
    )

    hit1 = average(
        hit1_values
    )

    hit3 = average(
        hit3_values
    )

    hit5 = average(
        hit5_values
    )

    mrr = average(
        mrr_values
    )

    recall5 = average(
        recall5_values
    )

    exact = average(
        exact_values
    )

    abstention_rate = (

        negative_abstained
        /
        negative_total

        if negative_total
        else 0.0

    )

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------

    summary = {

        "golden_tests":
            len(GOLDEN_QUERIES),

        "top_k":
            TOP_K,

        "structured_k":
            STRUCTURED_K,

        "bm25_k":
            BM25_K,

        "rrf_k":
            RRF_K,

        "precision_at_1":
            precision1,

        "hit_at_1":
            hit1,

        "hit_at_3":
            hit3,

        "hit_at_5":
            hit5,

        "mrr":
            mrr,

        "recall_at_5":
            recall5,

        "exact_match":
            exact,

        "negative_queries":
            negative_total,

        "negative_abstention":
            abstention_rate

    }

    # -------------------------------------------------------------
    # PRINT SUMMARY
    # -------------------------------------------------------------

    print()
    print("=" * 90)
    print("HYBRID V2 SUMMARY")
    print("=" * 90)

    print(
        f"Precision@1 : "
        f"{precision1:.2%}"
    )

    print(
        f"Hit@1       : "
        f"{hit1:.2%}"
    )

    print(
        f"Hit@3       : "
        f"{hit3:.2%}"
    )

    print(
        f"Hit@5       : "
        f"{hit5:.2%}"
    )

    print(
        f"MRR         : "
        f"{mrr:.4f}"
    )

    print(
        f"Recall@5    : "
        f"{recall5:.2%}"
    )

    print(
        f"Exact Match : "
        f"{exact:.2%}"
    )

    print()
    print("NEGATIVE QUERY")
    print(
        f"Negative queries : "
        f"{negative_total}"
    )

    print(
        f"Abstention rate  : "
        f"{abstention_rate:.2%}"
    )

    # -------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------

    evaluation_output = {

        "engine":
            "Hybrid Retrieval V2",

        "description":
            (
                "Field-aware hybrid retrieval "
                "using Structured Retrieval + "
                "BM25 + RRF."
            ),

        "summary":
            summary,

        "results":
            evaluation_results

    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            evaluation_output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 90)
    print("EVALUATION COMPLETED")
    print("=" * 90)

    print()
    print(
        f"Output : {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()