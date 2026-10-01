"""
Embedding Retrieval Evaluator

Evaluates the standalone semantic embedding retriever
against the same Q001-Q020 golden dataset used by
Hybrid V3.

This evaluator does NOT:
- use Structured Retrieval
- use BM25
- use Hybrid V3
- modify/reorder embedding results
- apply heuristic boosts
- apply a similarity threshold

It measures the raw embedding baseline.

Metrics:
- Precision@1
- Hit@1
- Hit@3
- Hit@5
- Recall@5
- MRR
- Exact Match
- Negative-query behavior
"""

import json
from pathlib import Path
import sys


# ================================================================
# PATH SETUP
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETRIEVAL_DIR = (
    PROJECT_ROOT
    / "src"
    / "retrieval"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "embedding_evaluation.json"
)

sys.path.insert(
    0,
    str(RETRIEVAL_DIR)
)


# ================================================================
# IMPORT
# ================================================================

from embedding_retriever import (
    EmbeddingRetriever,
    load_records
)


# ================================================================
# CONFIGURATION
# ================================================================

TOP_K = 5


# ================================================================
# GOLDEN DATASET
# ================================================================

GOLDEN_QUERIES = [

    # ------------------------------------------------------------
    # SINGLE INTENT
    # ------------------------------------------------------------

    {
        "id": "Q001",
        "type": "single",
        "query":
            'What is the inlet line size of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?',
        "expected": [
            "p01_r8_inlet"
        ]
    },

    {
        "id": "Q002",
        "type": "single",
        "query":
            'What is the inlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?',
        "expected": [
            "p01_r12_inlet"
        ]
    },

    {
        "id": "Q003",
        "type": "single",
        "query":
            'What is the inlet body material of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?',
        "expected": [
            "p01_r17_inlet"
        ]
    },

    {
        "id": "Q004",
        "type": "single",
        "query":
            'What is the material code of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?',
        "expected": [
            "p01_r58"
        ]
    },

    {
        "id": "Q005",
        "type": "single",
        "query":
            'What is the outlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?',
        "expected": [
            "p01_r12_outlet"
        ]
    },

    {
        "id": "Q006",
        "type": "single",
        "query":
            'What is the outlet steam line size of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?',
        "expected": [
            "p01_r8_outlet"
        ]
    },

    {
        "id": "Q007",
        "type": "single",
        "query":
            'What is the piping class of the outlet steam of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?',
        "expected": [
            "p01_r8_outlet"
        ]
    },

    {
        "id": "Q008",
        "type": "single",
        "query":
            'What is the inlet steam flow of '
            'DSH 6"600RF-INTEG TCV 1"600RF-HART?',
        "expected": [
            "p02_r12_inlet"
        ]
    },

    {
        "id": "Q009",
        "type": "single",
        "query":
            'What is the piping class of the outlet steam of '
            'DSH 6"600RF-INTEG TCV 1"600RF-HART?',
        "expected": [
            "p02_r8_outlet"
        ]
    },

    {
        "id": "Q010",
        "type": "single",
        "query":
            'What is the inlet line size of '
            'DSH 3"300RF-INTEG TCV 1"300RF-HART?',
        "expected": [
            "p05_r8_inlet"
        ]
    },

    {
        "id": "Q011",
        "type": "single",
        "query":
            'What is the material code of '
            'DSH 3"300RF-INTEG TCV 1"300RF-HART?',
        "expected": [
            "p05_r58"
        ]
    },

    {
        "id": "Q012",
        "type": "single",
        "query":
            'What is the normal inlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?',
        "expected": [
            "p01_r12_inlet"
        ]
    },

    {
        "id": "Q013",
        "type": "single",
        "query":
            'What is the maximum inlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?',
        "expected": [
            "p01_r12_inlet"
        ]
    },

    {
        "id": "Q014",
        "type": "single",
        "query":
            'What is the minimum inlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?',
        "expected": [
            "p01_r12_inlet"
        ]
    },

    {
        "id": "Q015",
        "type": "single",
        "query":
            'What is the piping class of the inlet steam of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?',
        "expected": [
            "p01_r8_inlet"
        ]
    },

    {
        "id": "Q016",
        "type": "single",
        "query":
            'What is the stem plug material of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?',
        "expected": [
            "p01_r17_inlet"
        ]
    },

    # ------------------------------------------------------------
    # MULTI-INTENT
    #
    # Standalone embedding retrieval is NOT a native
    # multi-intent retriever.
    #
    # Therefore these are evaluated as retrieval queries,
    # not as intent-coverage tests.
    # ------------------------------------------------------------

    {
        "id": "Q017",
        "type": "multi",
        "query":
            'What is the line size of the inlet steam and '
            'the piping class of the outlet steam of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?',
        "expected": [
            "p01_r8_inlet",
            "p01_r8_outlet"
        ]
    },

    {
        "id": "Q018",
        "type": "multi",
        "query":
            'What is the inlet steam flow, outlet steam flow, '
            'outlet piping class, and material code of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?',
        "expected": [
            "p01_r12_inlet",
            "p01_r12_outlet",
            "p01_r8_outlet",
            "p01_r58"
        ]
    },

    {
        "id": "Q019",
        "type": "multi",
        "query":
            'What is the inlet line size and material code of '
            'DSH 3"300RF-INTEG TCV 1"300RF-HART?',
        "expected": [
            "p05_r8_inlet",
            "p05_r58"
        ]
    },

    # ------------------------------------------------------------
    # NEGATIVE
    # ------------------------------------------------------------

    {
        "id": "Q020",
        "type": "negative",
        "query":
            'What is the turbine shaft diameter of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?',
        "expected": []
    }

]


# ================================================================
# METRIC FUNCTIONS
# ================================================================

def get_record_ids(results):

    return [
        result["record_id"]
        for result in results
    ]


def reciprocal_rank(
    retrieved_ids,
    expected_ids
):

    expected_set = set(
        expected_ids
    )

    for rank, record_id in enumerate(
        retrieved_ids,
        start=1
    ):

        if record_id in expected_set:
            return 1.0 / rank

    return 0.0


def hit_at_k(
    retrieved_ids,
    expected_ids,
    k
):

    expected_set = set(
        expected_ids
    )

    return any(
        record_id in expected_set
        for record_id in retrieved_ids[:k]
    )


def precision_at_k(
    retrieved_ids,
    expected_ids,
    k
):

    expected_set = set(
        expected_ids
    )

    top_k = retrieved_ids[:k]

    if not top_k:
        return 0.0

    relevant = sum(
        1
        for record_id in top_k
        if record_id in expected_set
    )

    return relevant / len(top_k)


def recall_at_k(
    retrieved_ids,
    expected_ids,
    k
):

    expected_set = set(
        expected_ids
    )

    if not expected_set:
        return 0.0

    relevant = sum(
        1
        for record_id in retrieved_ids[:k]
        if record_id in expected_set
    )

    return relevant / len(expected_set)


def exact_match(
    retrieved_ids,
    expected_ids
):

    return (
        set(retrieved_ids)
        == set(expected_ids)
    )


# ================================================================
# SINGLE QUERY EVALUATION
# ================================================================

def evaluate_single(
    retriever,
    test
):

    results = retriever.search(
        test["query"],
        top_k=TOP_K
    )

    retrieved_ids = get_record_ids(
        results
    )

    expected_ids = test[
        "expected"
    ]

    return {

        "id":
            test["id"],

        "type":
            test["type"],

        "query":
            test["query"],

        "expected":
            expected_ids,

        "retrieved":
            retrieved_ids,

        "scores": [
            result["score"]
            for result in results
        ],

        "hit_at_1":
            hit_at_k(
                retrieved_ids,
                expected_ids,
                1
            ),

        "hit_at_3":
            hit_at_k(
                retrieved_ids,
                expected_ids,
                3
            ),

        "hit_at_5":
            hit_at_k(
                retrieved_ids,
                expected_ids,
                5
            ),

        "precision_at_1":
            precision_at_k(
                retrieved_ids,
                expected_ids,
                1
            ),

        "recall_at_5":
            recall_at_k(
                retrieved_ids,
                expected_ids,
                5
            ),

        "mrr":
            reciprocal_rank(
                retrieved_ids,
                expected_ids
            ),

        "exact_match":
            exact_match(
                retrieved_ids,
                expected_ids
            )

    }


# ================================================================
# MULTI QUERY EVALUATION
# ================================================================

def evaluate_multi(
    retriever,
    test
):

    results = retriever.search(
        test["query"],
        top_k=TOP_K
    )

    retrieved_ids = get_record_ids(
        results
    )

    expected_ids = set(
        test["expected"]
    )

    retrieved_set = set(
        retrieved_ids
    )

    found = (
        expected_ids
        & retrieved_set
    )

    missing = (
        expected_ids
        - retrieved_set
    )

    unexpected = (
        retrieved_set
        - expected_ids
    )

    return {

        "id":
            test["id"],

        "type":
            test["type"],

        "query":
            test["query"],

        "expected":
            sorted(expected_ids),

        "retrieved":
            retrieved_ids,

        "found":
            sorted(found),

        "missing":
            sorted(missing),

        "unexpected":
            sorted(unexpected),

        "intent_coverage":
            (
                len(found)
                / len(expected_ids)
                if expected_ids
                else 0.0
            ),

        "full_query_success":
            (
                found
                == expected_ids
            ),

        "exact_match":
            (
                retrieved_set
                == expected_ids
            )

    }


# ================================================================
# NEGATIVE EVALUATION
# ================================================================

def evaluate_negative(
    retriever,
    test
):

    results = retriever.search(
        test["query"],
        top_k=TOP_K
    )

    retrieved_ids = get_record_ids(
        results
    )

    # Raw embedding retrieval has no abstention
    # mechanism at this stage.
    no_results = (
        len(retrieved_ids) == 0
    )

    return {

        "id":
            test["id"],

        "type":
            test["type"],

        "query":
            test["query"],

        "retrieved":
            retrieved_ids,

        "no_results":
            no_results,

        "abstained":
            no_results,

        "passed":
            no_results

    }


# ================================================================
# MAIN
# ================================================================

def main():

    print()
    print("=" * 90)
    print("EMBEDDING RETRIEVAL EVALUATION")
    print("=" * 90)

    print(
        f"Golden tests : "
        f"{len(GOLDEN_QUERIES)}"
    )

    print(
        f"Top-K       : "
        f"{TOP_K}"
    )

    print("=" * 90)

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------

    records = load_records()

    print(
        f"Records loaded: "
        f"{len(records)}"
    )

    # ------------------------------------------------------------
    # Create retriever
    # ------------------------------------------------------------

    retriever = EmbeddingRetriever(
        records
    )

    # ------------------------------------------------------------
    # Load existing FAISS index
    # ------------------------------------------------------------

    retriever.load_index()

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    single_results = []
    multi_results = []
    negative_results = []

    # ------------------------------------------------------------
    # Run tests
    # ------------------------------------------------------------

    for test in GOLDEN_QUERIES:

        if test["type"] == "single":

            result = evaluate_single(
                retriever,
                test
            )

            single_results.append(
                result
            )

            status = (
                "PASS"
                if result["hit_at_1"]
                else "FAIL"
            )

            print(
                f"{test['id']} | "
                f"SINGLE | "
                f"{status} | "
                f"Expected={result['expected']} | "
                f"Retrieved={result['retrieved']}"
            )

        elif test["type"] == "multi":

            result = evaluate_multi(
                retriever,
                test
            )

            multi_results.append(
                result
            )

            status = (
                "PASS"
                if result["full_query_success"]
                else "FAIL"
            )

            print(
                f"{test['id']} | "
                f"MULTI | "
                f"{status} | "
                f"Coverage="
                f"{result['intent_coverage']:.2%} | "
                f"Retrieved={result['retrieved']}"
            )

        elif test["type"] == "negative":

            result = evaluate_negative(
                retriever,
                test
            )

            negative_results.append(
                result
            )

            status = (
                "PASS"
                if result["passed"]
                else "FAIL"
            )

            print(
                f"{test['id']} | "
                f"NEGATIVE | "
                f"{status} | "
                f"Retrieved={result['retrieved']}"
            )

    # ============================================================
    # SINGLE METRICS
    # ============================================================

    if single_results:

        count = len(
            single_results
        )

        precision1 = sum(
            r["precision_at_1"]
            for r in single_results
        ) / count

        hit1 = sum(
            r["hit_at_1"]
            for r in single_results
        ) / count

        hit3 = sum(
            r["hit_at_3"]
            for r in single_results
        ) / count

        hit5 = sum(
            r["hit_at_5"]
            for r in single_results
        ) / count

        recall5 = sum(
            r["recall_at_5"]
            for r in single_results
        ) / count

        mrr = sum(
            r["mrr"]
            for r in single_results
        ) / count

        exact = sum(
            r["exact_match"]
            for r in single_results
        ) / count

        single_metrics = {

            "count":
                count,

            "precision_at_1":
                precision1,

            "hit_at_1":
                hit1,

            "hit_at_3":
                hit3,

            "hit_at_5":
                hit5,

            "recall_at_5":
                recall5,

            "mrr":
                mrr,

            "exact_match":
                exact

        }

    else:

        single_metrics = {}


    # ============================================================
    # MULTI METRICS
    # ============================================================

    if multi_results:

        count = len(
            multi_results
        )

        coverage = sum(
            r["intent_coverage"]
            for r in multi_results
        ) / count

        full_success = sum(
            r["full_query_success"]
            for r in multi_results
        ) / count

        exact = sum(
            r["exact_match"]
            for r in multi_results
        ) / count

        multi_metrics = {

            "count":
                count,

            "intent_coverage":
                coverage,

            "full_query_success":
                full_success,

            "exact_match":
                exact

        }

    else:

        multi_metrics = {}


    # ============================================================
    # NEGATIVE METRICS
    # ============================================================

    if negative_results:

        count = len(
            negative_results
        )

        passed = sum(
            r["passed"]
            for r in negative_results
        ) / count

        abstention = sum(
            r["abstained"]
            for r in negative_results
        ) / count

        negative_metrics = {

            "count":
                count,

            "pass_rate":
                passed,

            "abstention_rate":
                abstention

        }

    else:

        negative_metrics = {}


    # ============================================================
    # SAVE
    # ============================================================

    evaluation = {

        "evaluator":
            "embedding_evaluator",

        "version":
            "1.0",

        "model":
            "sentence-transformers/all-MiniLM-L6-v2",

        "index":
            "FAISS IndexFlatIP",

        "configuration": {

            "top_k":
                TOP_K

        },

        "dataset": {

            "records":
                len(records),

            "golden_queries":
                len(GOLDEN_QUERIES)

        },

        "single_intent":
            single_metrics,

        "multi_intent":
            multi_metrics,

        "negative":
            negative_metrics,

        "details": {

            "single":
                single_results,

            "multi":
                multi_results,

            "negative":
                negative_results

        }

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
            evaluation,
            file,
            indent=2,
            ensure_ascii=False
        )


    # ============================================================
    # SUMMARY
    # ============================================================

    print()
    print("=" * 90)
    print("EMBEDDING EVALUATION SUMMARY")
    print("=" * 90)

    if single_metrics:

        print(
            f"Precision@1 : "
            f"{single_metrics['precision_at_1']:.2%}"
        )

        print(
            f"Hit@1       : "
            f"{single_metrics['hit_at_1']:.2%}"
        )

        print(
            f"Hit@3       : "
            f"{single_metrics['hit_at_3']:.2%}"
        )

        print(
            f"Hit@5       : "
            f"{single_metrics['hit_at_5']:.2%}"
        )

        print(
            f"Recall@5    : "
            f"{single_metrics['recall_at_5']:.2%}"
        )

        print(
            f"MRR         : "
            f"{single_metrics['mrr']:.4f}"
        )

        print(
            f"Exact Match : "
            f"{single_metrics['exact_match']:.2%}"
        )

    if multi_metrics:

        print()

        print(
            f"Multi Coverage     : "
            f"{multi_metrics['intent_coverage']:.2%}"
        )

        print(
            f"Multi Full Success : "
            f"{multi_metrics['full_query_success']:.2%}"
        )

        print(
            f"Multi Exact Match  : "
            f"{multi_metrics['exact_match']:.2%}"
        )

    if negative_metrics:

        print()

        print(
            f"Negative Pass Rate  : "
            f"{negative_metrics['pass_rate']:.2%}"
        )

        print(
            f"Negative Abstention : "
            f"{negative_metrics['abstention_rate']:.2%}"
        )

    print()
    print(
        f"Saved evaluation:"
    )
    print(
        OUTPUT_FILE
    )

    print("=" * 90)


if __name__ == "__main__":
    main()