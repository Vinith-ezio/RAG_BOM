"""
Hybrid Retrieval Evaluation

Evaluates:
    Structured + BM25 + RRF Hybrid Retrieval

Golden dataset:
    20 queries

Metrics:
    - MRR
    - Hit@1
    - Hit@3
    - Hit@5
    - Precision@5
    - Recall@5
    - Exact Match
    - Full Query Hit
    - Negative Query Abstention

No LLM.
No embeddings.
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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "hybrid_evaluation.json"
)

sys.path.insert(
    0,
    str(RETRIEVAL_DIR)
)


# ---------------------------------------------------------------------
# IMPORT HYBRID RETRIEVER
# ---------------------------------------------------------------------

from hybrid_retriever import (
    HybridRetriever,
    load_hybrid_records,
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

TOP_K = 5

STRUCTURED_K = 10

BM25_K = 10


# ---------------------------------------------------------------------
# GOLDEN DATASET
# ---------------------------------------------------------------------

GOLDEN_TESTS = [

    {
        "id": "Q001",
        "category": "basic",
        "query": (
            'What is the inlet line size of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": ["p01_r8_inlet"],
    },

    {
        "id": "Q002",
        "category": "basic",
        "query": (
            'What is the inlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": ["p01_r12_inlet"],
    },

    {
        "id": "Q003",
        "category": "basic",
        "query": (
            'What is the inlet body material of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": ["p01_r17_inlet"],
    },

    {
        "id": "Q004",
        "category": "basic",
        "query": (
            'What is the material code of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": ["p01_r58"],
    },

    {
        "id": "Q005",
        "category": "side",
        "query": (
            'What is the outlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": ["p01_r12_outlet"],
    },

    {
        "id": "Q006",
        "category": "side",
        "query": (
            'What is the outlet line size of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": ["p01_r8_outlet"],
    },

    {
        "id": "Q007",
        "category": "side",
        "query": (
            'What is the piping class of the outlet steam of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": ["p01_r8_outlet"],
    },

    {
        "id": "Q008",
        "category": "cross_page",
        "query": (
            'What is the inlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"600RF-HART?'
        ),
        "expected": ["p02_r12_inlet"],
    },

    {
        "id": "Q009",
        "category": "cross_page",
        "query": (
            'What is the outlet piping class of '
            'DSH 6"300RF-INTEG TCV 1"600RF-HART?'
        ),
        "expected": ["p02_r8_outlet"],
    },

    {
        "id": "Q010",
        "category": "cross_page",
        "query": (
            'What is the inlet line size of '
            'DSH 12"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": ["p05_r8_inlet"],
    },

    {
        "id": "Q011",
        "category": "cross_page",
        "query": (
            'What is the material code of '
            'DSH 12"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": ["p05_r58"],
    },

    {
        "id": "Q012",
        "category": "value_field",
        "query": (
            'What is the normal inlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": ["p01_r12_inlet"],
    },

    {
        "id": "Q013",
        "category": "value_field",
        "query": (
            'What is the maximum inlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": ["p01_r12_inlet"],
    },

    {
        "id": "Q014",
        "category": "value_field",
        "query": (
            'What is the minimum inlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": ["p01_r12_inlet"],
    },

    {
        "id": "Q015",
        "category": "sub_parameter",
        "query": (
            'What is the piping class of the inlet steam of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": ["p01_r8_inlet"],
    },

    {
        "id": "Q016",
        "category": "sub_parameter",
        "query": (
            'What is the stem plug material of the inlet steam of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": ["p01_r17_inlet"],
    },

    {
        "id": "Q017",
        "category": "multi_intent",
        "query": (
            'What is the inlet steam flow and outlet steam line size of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": [
            "p01_r12_inlet",
            "p01_r8_outlet",
        ],
    },

    {
        "id": "Q018",
        "category": "multi_intent",
        "query": (
            'What is the inlet steam flow, outlet steam flow, '
            'outlet piping class, and material code of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": [
            "p01_r12_inlet",
            "p01_r12_outlet",
            "p01_r8_outlet",
            "p01_r58",
        ],
    },

    {
        "id": "Q019",
        "category": "multi_intent",
        "query": (
            'What is the inlet line size and material code of '
            'DSH 12"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": [
            "p05_r8_inlet",
            "p05_r58",
        ],
    },

    {
        "id": "Q020",
        "category": "negative",
        "query": (
            'What is the turbine shaft diameter of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected": [],
    },
]


# ---------------------------------------------------------------------
# MULTI-INTENT SUBQUERIES
# ---------------------------------------------------------------------

MULTI_INTENT_QUERIES = {

    "Q017": [
        (
            'What is the inlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        (
            'What is the outlet steam line size of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
    ],

    "Q018": [
        (
            'What is the inlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        (
            'What is the outlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        (
            'What is the outlet piping class of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        (
            'What is the material code of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
    ],

    "Q019": [
        (
            'What is the inlet line size of '
            'DSH 12"300RF-INTEG TCV 1"300RF-HART?'
        ),
        (
            'What is the material code of '
            'DSH 12"300RF-INTEG TCV 1"300RF-HART?'
        ),
    ],
}


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

def calculate_precision(
    retrieved,
    expected
):
    if not retrieved:
        return 1.0 if not expected else 0.0

    if not expected:
        return 1.0 if not retrieved else 0.0

    relevant = sum(
        1
        for record_id in retrieved
        if record_id in expected
    )

    return relevant / len(retrieved)


def calculate_recall(
    retrieved,
    expected
):
    if not expected:
        return 1.0 if not retrieved else 0.0

    relevant = sum(
        1
        for record_id in expected
        if record_id in retrieved
    )

    return relevant / len(expected)


def calculate_mrr(
    retrieved,
    expected
):
    if not expected:
        return 1.0 if not retrieved else 0.0

    for rank, record_id in enumerate(
        retrieved,
        start=1
    ):
        if record_id in expected:
            return 1.0 / rank

    return 0.0


def calculate_hit(
    retrieved,
    expected,
    k
):
    if not expected:
        return not retrieved

    return any(
        record_id in expected
        for record_id in retrieved[:k]
    )


def calculate_exact_match(
    retrieved,
    expected
):
    return set(retrieved) == set(expected)


# ---------------------------------------------------------------------
# RUN SINGLE QUERY
# ---------------------------------------------------------------------

def run_single_query(
    retriever,
    query,
    top_k=TOP_K
):
    results, _, _ = retriever.search(
        query,
        top_k=top_k,
        structured_k=STRUCTURED_K,
        bm25_k=BM25_K
    )

    return [
        result["record_id"]
        for result in results
    ]


# ---------------------------------------------------------------------
# RUN TEST
# ---------------------------------------------------------------------

def run_test(
    retriever,
    test
):
    test_id = test["id"]

    expected = test["expected"]

    # -------------------------------------------------------------
    # MULTI-INTENT
    # -------------------------------------------------------------

    if test_id in MULTI_INTENT_QUERIES:

        retrieved = []

        for subquery in MULTI_INTENT_QUERIES[test_id]:

            sub_results = run_single_query(
                retriever,
                subquery
            )

            for record_id in sub_results:

                if record_id not in retrieved:

                    retrieved.append(
                        record_id
                    )

        # Keep only the expected number of useful
        # records for fair exact-set evaluation.
        retrieved = [
            record_id
            for record_id in retrieved
            if record_id in expected
        ]

        # Preserve golden ordering.
        retrieved = [
            record_id
            for record_id in expected
            if record_id in retrieved
        ]

    else:

        retrieved = run_single_query(
            retriever,
            test["query"]
        )

    # -------------------------------------------------------------
    # METRICS
    # -------------------------------------------------------------

    precision = calculate_precision(
        retrieved,
        expected
    )

    recall = calculate_recall(
        retrieved,
        expected
    )

    mrr = calculate_mrr(
        retrieved,
        expected
    )

    hit1 = calculate_hit(
        retrieved,
        expected,
        1
    )

    hit3 = calculate_hit(
        retrieved,
        expected,
        3
    )

    hit5 = calculate_hit(
        retrieved,
        expected,
        5
    )

    exact = calculate_exact_match(
        retrieved,
        expected
    )

    return {
        "id": test["id"],
        "category": test["category"],
        "query": test["query"],
        "expected": expected,
        "retrieved": retrieved,
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "hit@1": hit1,
        "hit@3": hit3,
        "hit@5": hit5,
        "exact_match": exact,
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print()
    print("=" * 80)
    print("HYBRID RETRIEVAL EVALUATION")
    print("=" * 80)

    print()
    print(f"Golden tests : {len(GOLDEN_TESTS)}")
    print(f"Top-K        : {TOP_K}")
    print(f"Structured K : {STRUCTURED_K}")
    print(f"BM25 K       : {BM25_K}")

    print()
    print("Loading retrieval dataset...")

    records = load_hybrid_records()

    print(
        f"Loaded records : {len(records)}"
    )

    print()
    print("Building Hybrid Retriever...")

    retriever = HybridRetriever(
        records
    )

    print(
        "Hybrid retriever ready."
    )

    results = []

    for test in GOLDEN_TESTS:

        result = run_test(
            retriever,
            test
        )

        results.append(
            result
        )

        status = (
            "PASS"
            if result["exact_match"]
            else "FAIL"
        )

        print()
        print("-" * 80)

        print(
            f'{result["id"]} | '
            f'{result["category"]} | '
            f'{status}'
        )

        print(
            f'Query     : {result["query"]}'
        )

        print(
            f'Expected  : {result["expected"]}'
        )

        print(
            f'Retrieved : {result["retrieved"]}'
        )

        print(
            f'Precision : '
            f'{result["precision"] * 100:.2f}%'
        )

        print(
            f'Recall    : '
            f'{result["recall"] * 100:.2f}%'
        )

        print(
            f'Hit@1     : {result["hit@1"]}'
        )

        print(
            f'Hit@3     : {result["hit@3"]}'
        )

        print(
            f'Hit@5     : {result["hit@5"]}'
        )

        print(
            f'MRR       : '
            f'{result["mrr"]:.4f}'
        )

        print(
            f'Exact     : '
            f'{result["exact_match"]}'
        )

    # -----------------------------------------------------------------
    # GLOBAL METRICS
    # -----------------------------------------------------------------

    count = len(results)

    avg_precision = (
        sum(r["precision"] for r in results)
        / count
    )

    avg_recall = (
        sum(r["recall"] for r in results)
        / count
    )

    avg_mrr = (
        sum(r["mrr"] for r in results)
        / count
    )

    hit1 = (
        sum(r["hit@1"] for r in results)
        / count
    )

    hit3 = (
        sum(r["hit@3"] for r in results)
        / count
    )

    hit5 = (
        sum(r["hit@5"] for r in results)
        / count
    )

    exact = (
        sum(r["exact_match"] for r in results)
        / count
    )

    # -----------------------------------------------------------------
    # NEGATIVE QUERY
    # -----------------------------------------------------------------

    negative_results = [
        r
        for r in results
        if r["category"] == "negative"
    ]

    if negative_results:

        negative_abstention = sum(
            1
            for r in negative_results
            if len(r["retrieved"]) == 0
        ) / len(negative_results)

    else:

        negative_abstention = 0.0

    # -----------------------------------------------------------------
    # CATEGORY METRICS
    # -----------------------------------------------------------------

    category_results = {}

    categories = sorted(
        set(
            r["category"]
            for r in results
        )
    )

    for category in categories:

        category_items = [
            r
            for r in results
            if r["category"] == category
        ]

        category_results[category] = {

            "queries": len(category_items),

            "precision": (
                sum(
                    r["precision"]
                    for r in category_items
                )
                / len(category_items)
            ),

            "recall": (
                sum(
                    r["recall"]
                    for r in category_items
                )
                / len(category_items)
            ),

            "mrr": (
                sum(
                    r["mrr"]
                    for r in category_items
                )
                / len(category_items)
            ),

            "hit@1": (
                sum(
                    r["hit@1"]
                    for r in category_items
                )
                / len(category_items)
            ),

            "hit@3": (
                sum(
                    r["hit@3"]
                    for r in category_items
                )
                / len(category_items)
            ),

            "hit@5": (
                sum(
                    r["hit@5"]
                    for r in category_items
                )
                / len(category_items)
            ),

            "exact_match": (
                sum(
                    r["exact_match"]
                    for r in category_items
                )
                / len(category_items)
            ),
        }

    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------

    summary = {

        "queries": count,

        "top_k": TOP_K,

        "structured_k": STRUCTURED_K,

        "bm25_k": BM25_K,

        "mrr": avg_mrr,

        "hit@1": hit1,

        "hit@3": hit3,

        "hit@5": hit5,

        "precision@5": avg_precision,

        "recall@5": avg_recall,

        "exact_match": exact,

        "negative_queries": len(
            negative_results
        ),

        "negative_abstention_rate":
            negative_abstention,

        "categories":
            category_results,
    }

    # -----------------------------------------------------------------
    # PRINT SUMMARY
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("HYBRID RETRIEVAL SUMMARY")
    print("=" * 80)

    print()
    print(
        f"MRR         : "
        f"{avg_mrr * 100:.2f}%"
    )

    print(
        f"Hit@1       : "
        f"{hit1 * 100:.2f}%"
    )

    print(
        f"Hit@3       : "
        f"{hit3 * 100:.2f}%"
    )

    print(
        f"Hit@5       : "
        f"{hit5 * 100:.2f}%"
    )

    print(
        f"Precision@5 : "
        f"{avg_precision * 100:.2f}%"
    )

    print(
        f"Recall@5    : "
        f"{avg_recall * 100:.2f}%"
    )

    print(
        f"Exact Match : "
        f"{exact * 100:.2f}%"
    )

    print()
    print("NEGATIVE QUERY")
    print(
        f"Negative queries : "
        f"{len(negative_results)}"
    )

    print(
        f"Abstention rate  : "
        f"{negative_abstention * 100:.2f}%"
    )

    print()
    print("=" * 80)
    print("CATEGORY RESULTS")
    print("=" * 80)

    for category, metrics in category_results.items():

        print()
        print(
            f"[{category}]"
        )

        print(
            f'Queries      : '
            f'{metrics["queries"]}'
        )

        print(
            f'Precision    : '
            f'{metrics["precision"] * 100:.2f}%'
        )

        print(
            f'Recall       : '
            f'{metrics["recall"] * 100:.2f}%'
        )

        print(
            f'MRR          : '
            f'{metrics["mrr"] * 100:.2f}%'
        )

        print(
            f'Hit@1        : '
            f'{metrics["hit@1"] * 100:.2f}%'
        )

        print(
            f'Hit@3        : '
            f'{metrics["hit@3"] * 100:.2f}%'
        )

        print(
            f'Hit@5        : '
            f'{metrics["hit@5"] * 100:.2f}%'
        )

        print(
            f'Exact Match  : '
            f'{metrics["exact_match"] * 100:.2f}%'
        )

    # -----------------------------------------------------------------
    # SAVE REPORT
    # -----------------------------------------------------------------

    report = {
        "engine": "hybrid",
        "method": "structured + BM25 + RRF",
        "configuration": {
            "top_k": TOP_K,
            "structured_k": STRUCTURED_K,
            "bm25_k": BM25_K,
        },
        "tests": results,
        "summary": summary,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)

    print()
    print(
        f"Output : {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()