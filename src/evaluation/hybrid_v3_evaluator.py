"""
Hybrid Retrieval V3 Evaluator

Evaluates:

- Q001-Q016 : Single-intent retrieval
- Q017-Q019 : Native multi-intent retrieval
- Q020       : Negative / unsupported query

Metrics:

- Precision@1
- Hit@1
- Hit@3
- Hit@5
- MRR
- Recall@5
- Exact Match
- Multi-intent Full Query Success
- Negative-query Abstention

Important:
The evaluator does NOT split multi-intent queries itself.

Q017-Q019 are passed directly to the V3 retriever.
"""

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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "hybrid_v3_evaluation.json"
)

sys.path.insert(
    0,
    str(RETRIEVAL_DIR)
)


# =====================================================================
# IMPORT V3
# =====================================================================

from hybrid_retriever_v3 import (
    NativeMultiIntentHybridRetriever,
    load_v3_records
)


# =====================================================================
# CONFIGURATION
# =====================================================================

TOP_K = 5
STRUCTURED_K = 10
BM25_K = 10
RRF_K = 60


# =====================================================================
# GOLDEN DATASET
# =====================================================================

GOLDEN_QUERIES = [

    # ---------------------------------------------------------------
    # SINGLE INTENT
    # ---------------------------------------------------------------

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
            'DSH 6"300RF-INTEG TCV 1"600RF-HART?',
        "expected": [
            "p02_r12_inlet"
        ]
    },

    {
        "id": "Q009",
        "type": "single",
        "query":
            'What is the piping class of the outlet steam of '
            'DSH 6"300RF-INTEG TCV 1"600RF-HART?',
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


    # ---------------------------------------------------------------
    # NATIVE MULTI-INTENT
    # ---------------------------------------------------------------

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


    # ---------------------------------------------------------------
    # NEGATIVE
    # ---------------------------------------------------------------

    {
        "id": "Q020",
        "type": "negative",
        "query":
            'What is the turbine shaft diameter of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?',
        "expected": []
    }

]


# =====================================================================
# HELPERS
# =====================================================================

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

    return relevant / len(
        expected_set
    )


def exact_match(
    retrieved_ids,
    expected_ids
):

    return (
        set(retrieved_ids)
        == set(expected_ids)
    )


# =====================================================================
# SINGLE-INTENT EVALUATION
# =====================================================================

def evaluate_single(
    retriever,
    test
):

    output = retriever.search(
        test["query"],
        top_k=TOP_K,
        structured_k=STRUCTURED_K,
        bm25_k=BM25_K
    )

    retrieved_ids = get_record_ids(
        output["flattened_results"]
    )

    expected_ids = test[
        "expected"
    ]

    rr = reciprocal_rank(
        retrieved_ids,
        expected_ids
    )

    result = {

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

        "top1":
            retrieved_ids[:1],

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
            rr,

        "exact_match":
            exact_match(
                retrieved_ids,
                expected_ids
            ),

        "abstained":
            output["abstained_intents"] > 0

    }

    return result


# =====================================================================
# MULTI-INTENT EVALUATION
# =====================================================================

def evaluate_multi(
    retriever,
    test
):

    output = retriever.search(
        test["query"],
        top_k=TOP_K,
        structured_k=STRUCTURED_K,
        bm25_k=BM25_K
    )

    expected_ids = set(
        test["expected"]
    )

    retrieved_ids = get_record_ids(
        output["flattened_results"]
    )

    retrieved_set = set(
        retrieved_ids
    )

    found_ids = (
        expected_ids
        & retrieved_set
    )

    missing_ids = (
        expected_ids
        - retrieved_set
    )

    unexpected_ids = (
        retrieved_set
        - expected_ids
    )

    intent_count = (
        output["intent_count"]
    )

    successful_intents = (
        output["successful_intents"]
    )

    # ---------------------------------------------------------------
    # Intent coverage
    # ---------------------------------------------------------------

    intent_coverage = (

        successful_intents
        == intent_count
        == len(expected_ids)

    )

    # ---------------------------------------------------------------
    # Full query success
    # ---------------------------------------------------------------

    full_query_success = (

        found_ids == expected_ids

    )

    # ---------------------------------------------------------------
    # Exact set match
    # ---------------------------------------------------------------

    exact = (
        retrieved_set
        == expected_ids
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
            sorted(found_ids),

        "missing":
            sorted(missing_ids),

        "unexpected":
            sorted(unexpected_ids),

        "intent_count":
            intent_count,

        "successful_intents":
            successful_intents,

        "abstained_intents":
            output[
                "abstained_intents"
            ],

        "intent_coverage":
            intent_coverage,

        "full_query_success":
            full_query_success,

        "exact_match":
            exact

    }


# =====================================================================
# NEGATIVE EVALUATION
# =====================================================================

def evaluate_negative(
    retriever,
    test
):

    output = retriever.search(
        test["query"],
        top_k=TOP_K,
        structured_k=STRUCTURED_K,
        bm25_k=BM25_K
    )

    retrieved_ids = get_record_ids(
        output["flattened_results"]
    )

    abstained = (
        output["abstained_intents"]
        > 0
    )

    no_results = (
        len(retrieved_ids)
        == 0
    )

    passed = (
        abstained
        and no_results
    )

    return {

        "id":
            test["id"],

        "type":
            test["type"],

        "query":
            test["query"],

        "expected":
            [],

        "retrieved":
            retrieved_ids,

        "abstained":
            abstained,

        "no_results":
            no_results,

        "passed":
            passed

    }


# =====================================================================
# MAIN EVALUATION
# =====================================================================

def main():

    print()
    print(
        "=" * 90
    )
    print(
        "HYBRID RETRIEVAL V3 EVALUATION"
    )
    print(
        "=" * 90
    )

    print(
        f"Dataset       : {DATA_FILE}"
    )

    print(
        f"Golden tests  : {len(GOLDEN_QUERIES)}"
    )

    print(
        f"Top-K         : {TOP_K}"
    )

    print(
        f"Structured-K  : {STRUCTURED_K}"
    )

    print(
        f"BM25-K        : {BM25_K}"
    )

    print(
        f"RRF-K         : {RRF_K}"
    )

    print(
        "=" * 90
    )


    # ================================================================
    # LOAD
    # ================================================================

    records = load_v3_records()

    print(
        f"Records loaded: {len(records)}"
    )


    # ================================================================
    # RETRIEVER
    # ================================================================

    retriever = (
        NativeMultiIntentHybridRetriever(
            records,
            rrf_k=RRF_K
        )
    )


    # ================================================================
    # RESULT COLLECTION
    # ================================================================

    single_results = []
    multi_results = []
    negative_results = []


    # ================================================================
    # RUN GOLDEN DATASET
    # ================================================================

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
                if (
                    result["hit_at_1"]
                    and result["exact_match"]
                )
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
                if result[
                    "full_query_success"
                ]
                else "FAIL"
            )

            print(
                f"{test['id']} | "
                f"MULTI | "
                f"{status} | "
                f"Expected={result['expected']} | "
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


    # ================================================================
    # SINGLE METRICS
    # ================================================================

    if single_results:

        single_count = len(
            single_results
        )

        hit1 = sum(
            r["hit_at_1"]
            for r in single_results
        )

        hit3 = sum(
            r["hit_at_3"]
            for r in single_results
        )

        hit5 = sum(
            r["hit_at_5"]
            for r in single_results
        )

        precision1 = sum(
            r["precision_at_1"]
            for r in single_results
        )

        recall5 = sum(
            r["recall_at_5"]
            for r in single_results
        )

        mrr = sum(
            r["mrr"]
            for r in single_results
        )

        exact = sum(
            r["exact_match"]
            for r in single_results
        )

        single_metrics = {

            "count":
                single_count,

            "precision_at_1":
                precision1
                / single_count,

            "hit_at_1":
                hit1
                / single_count,

            "hit_at_3":
                hit3
                / single_count,

            "hit_at_5":
                hit5
                / single_count,

            "recall_at_5":
                recall5
                / single_count,

            "mrr":
                mrr
                / single_count,

            "exact_match":
                exact
                / single_count

        }

    else:

        single_metrics = {}


    # ================================================================
    # MULTI METRICS
    # ================================================================

    if multi_results:

        multi_count = len(
            multi_results
        )

        intent_coverage = sum(
            r["intent_coverage"]
            for r in multi_results
        )

        full_success = sum(
            r["full_query_success"]
            for r in multi_results
        )

        multi_exact = sum(
            r["exact_match"]
            for r in multi_results
        )

        multi_metrics = {

            "count":
                multi_count,

            "intent_coverage":
                intent_coverage
                / multi_count,

            "full_query_success":
                full_success
                / multi_count,

            "exact_match":
                multi_exact
                / multi_count

        }

    else:

        multi_metrics = {}


    # ================================================================
    # NEGATIVE METRICS
    # ================================================================

    if negative_results:

        negative_count = len(
            negative_results
        )

        negative_pass = sum(
            r["passed"]
            for r in negative_results
        )

        negative_abstention = sum(
            r["abstained"]
            for r in negative_results
        )

        negative_metrics = {

            "count":
                negative_count,

            "pass_rate":
                negative_pass
                / negative_count,

            "abstention_rate":
                negative_abstention
                / negative_count

        }

    else:

        negative_metrics = {}


    # ================================================================
    # OVERALL
    # ================================================================

    all_positive = (
        single_results
        + multi_results
    )

    total_positive = len(
        all_positive
    )

    overall_full_success = (

        sum(
            1
            for r in single_results
            if (
                r["hit_at_1"]
                and r["exact_match"]
            )
        )

        +

        sum(
            1
            for r in multi_results
            if r[
                "full_query_success"
            ]
        )

    )

    overall = {

        "positive_queries":
            total_positive,

        "negative_queries":
            len(
                negative_results
            ),

        "positive_full_success":
            (
                overall_full_success
                / total_positive
                if total_positive
                else 0.0
            ),

        "negative_abstention_rate":
            (
                negative_metrics.get(
                    "abstention_rate",
                    0.0
                )
            )

    }


    # ================================================================
    # FINAL SUMMARY
    # ================================================================

    evaluation = {

        "evaluator":
            "hybrid_v3_evaluator",

        "version":
            "3.0",

        "configuration": {

            "top_k":
                TOP_K,

            "structured_k":
                STRUCTURED_K,

            "bm25_k":
                BM25_K,

            "rrf_k":
                RRF_K

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

        "overall":
            overall,

        "details": {

            "single":
                single_results,

            "multi":
                multi_results,

            "negative":
                negative_results

        }

    }


    # ================================================================
    # SAVE
    # ================================================================

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


    # ================================================================
    # PRINT SUMMARY
    # ================================================================

    print()
    print(
        "=" * 90
    )

    print(
        "V3 EVALUATION SUMMARY"
    )

    print(
        "=" * 90
    )


    if single_metrics:

        print(
            f"Single Precision@1 : "
            f"{single_metrics['precision_at_1']:.2%}"
        )

        print(
            f"Single Hit@1       : "
            f"{single_metrics['hit_at_1']:.2%}"
        )

        print(
            f"Single Hit@3       : "
            f"{single_metrics['hit_at_3']:.2%}"
        )

        print(
            f"Single Hit@5       : "
            f"{single_metrics['hit_at_5']:.2%}"
        )

        print(
            f"Single Recall@5    : "
            f"{single_metrics['recall_at_5']:.2%}"
        )

        print(
            f"Single MRR         : "
            f"{single_metrics['mrr']:.4f}"
        )

        print(
            f"Single Exact Match : "
            f"{single_metrics['exact_match']:.2%}"
        )


    if multi_metrics:

        print()

        print(
            f"Multi Intent Coverage : "
            f"{multi_metrics['intent_coverage']:.2%}"
        )

        print(
            f"Multi Full Success    : "
            f"{multi_metrics['full_query_success']:.2%}"
        )

        print(
            f"Multi Exact Match     : "
            f"{multi_metrics['exact_match']:.2%}"
        )


    if negative_metrics:

        print()

        print(
            f"Negative Pass Rate    : "
            f"{negative_metrics['pass_rate']:.2%}"
        )

        print(
            f"Negative Abstention   : "
            f"{negative_metrics['abstention_rate']:.2%}"
        )


    print()

    print(
        f"Overall Positive Success : "
        f"{overall['positive_full_success']:.2%}"
    )

    print(
        f"Overall Negative Abstention : "
        f"{overall['negative_abstention_rate']:.2%}"
    )

    print()

    print(
        f"Saved evaluation:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "=" * 90
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()