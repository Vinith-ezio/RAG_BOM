import argparse
import json
from pathlib import Path

from multi_intent_retriever import (
    load_records,
    multi_intent_retrieve,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_INPUT = (
    r"E:\PDF Ingestion\data\output\retrieval_enriched.json"
)


# ============================================================
# GOLDEN TEST DATASET
# ============================================================

GOLDEN_TESTS = [

    # --------------------------------------------------------
    # BASIC PARAMETER RETRIEVAL
    # --------------------------------------------------------

    {
        "id": "Q001",
        "category": "basic",
        "query": (
            'What is the inlet line size of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected_records": [
            "p01_r8_inlet"
        ],
    },

    {
        "id": "Q002",
        "category": "basic",
        "query": (
            'What is the inlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected_records": [
            "p01_r12_inlet"
        ],
    },

    {
        "id": "Q003",
        "category": "basic",
        "query": (
            'What is the inlet body material of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected_records": [
            "p01_r17_inlet"
        ],
    },

    {
        "id": "Q004",
        "category": "basic",
        "query": (
            'What is the material code of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected_records": [
            "p01_r58"
        ],
    },


    # --------------------------------------------------------
    # INLET / OUTLET
    # --------------------------------------------------------

    {
        "id": "Q005",
        "category": "side",
        "query": (
            'What is the outlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected_records": [
            "p01_r12_outlet"
        ],
    },

    {
        "id": "Q006",
        "category": "side",
        "query": (
            'What is the outlet line size of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected_records": [
            "p01_r8_outlet"
        ],
    },

    {
        "id": "Q007",
        "category": "side",
        "query": (
            'What is the piping class of the outlet steam of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected_records": [
            "p01_r8_outlet"
        ],
    },


    # --------------------------------------------------------
    # CROSS-PAGE RETRIEVAL
    # --------------------------------------------------------

    {
        "id": "Q008",
        "category": "cross_page",
        "query": (
            'What is the inlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"600RF-HART?'
        ),
        "expected_records": [
            "p02_r12_inlet"
        ],
    },

    {
        "id": "Q009",
        "category": "cross_page",
        "query": (
            'What is the outlet piping class of '
            'DSH 6"300RF-INTEG TCV 1"600RF-HART?'
        ),
        "expected_records": [
            "p02_r8_outlet"
        ],
    },

    {
        "id": "Q010",
        "category": "cross_page",
        "query": (
            'What is the inlet line size of '
            'DSH 12"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected_records": [
            "p05_r8_inlet"
        ],
    },

    {
        "id": "Q011",
        "category": "cross_page",
        "query": (
            'What is the material code of '
            'DSH 12"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected_records": [
            "p05_r58"
        ],
    },


    # --------------------------------------------------------
    # VALUE-SPECIFIC RETRIEVAL
    # --------------------------------------------------------

    {
        "id": "Q012",
        "category": "value_field",
        "query": (
            'What is the normal inlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected_records": [
            "p01_r12_inlet"
        ],
    },

    {
        "id": "Q013",
        "category": "value_field",
        "query": (
            'What is the maximum inlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected_records": [
            "p01_r12_inlet"
        ],
    },

    {
        "id": "Q014",
        "category": "value_field",
        "query": (
            'What is the minimum inlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected_records": [
            "p01_r12_inlet"
        ],
    },


    # --------------------------------------------------------
    # SUB-PARAMETER
    # --------------------------------------------------------

    {
        "id": "Q015",
        "category": "sub_parameter",
        "query": (
            'What is the piping class of the inlet steam of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected_records": [
            "p01_r8_inlet"
        ],
    },

    {
        "id": "Q016",
        "category": "sub_parameter",
        "query": (
            'What is the stem plug material of the inlet '
            'steam of DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected_records": [
            "p01_r17_inlet"
        ],
    },


    # --------------------------------------------------------
    # MULTI-INTENT
    # --------------------------------------------------------

    {
        "id": "Q017",
        "category": "multi_intent",
        "query": (
            'What is the inlet steam flow and '
            'outlet steam line size of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected_records": [
            "p01_r12_inlet",
            "p01_r8_outlet",
        ],
    },

    {
        "id": "Q018",
        "category": "multi_intent",
        "query": (
            'What is the inlet steam flow, '
            'outlet steam flow, '
            'outlet piping class, '
            'and material code of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected_records": [
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
            'What is the inlet line size and '
            'material code of '
            'DSH 12"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected_records": [
            "p05_r8_inlet",
            "p05_r58",
        ],
    },


    # --------------------------------------------------------
    # NEGATIVE QUERY
    # --------------------------------------------------------

    {
        "id": "Q020",
        "category": "negative",
        "query": (
            'What is the turbine shaft diameter of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),
        "expected_records": [],
    },
]


# ============================================================
# RECORD COLLECTION
# ============================================================

def collect_retrieved_record_ids(results):
    """
    Collect unique record IDs from all retrieved intents.
    """

    record_ids = []

    for result in results:

        for record in result["matches"]:

            record_id = record.get("record_id")

            if (
                record_id
                and record_id not in record_ids
            ):
                record_ids.append(record_id)

    return record_ids


# ============================================================
# METRICS
# ============================================================

def calculate_precision(
    expected,
    retrieved,
):
    """
    Precision:

        relevant retrieved records
        --------------------------
        all retrieved records
    """

    if not retrieved:
        return 1.0 if not expected else 0.0

    relevant = len(
        set(expected) & set(retrieved)
    )

    return relevant / len(retrieved)


def calculate_recall(
    expected,
    retrieved,
):
    """
    Recall:

        relevant retrieved records
        --------------------------
        all expected records
    """

    if not expected:
        return 1.0 if not retrieved else 0.0

    relevant = len(
        set(expected) & set(retrieved)
    )

    return relevant / len(expected)


def calculate_hit_at_k(
    expected,
    retrieved,
    k,
):
    """
    Hit@K:

    At least one expected record appears
    within the first K retrieved records.
    """

    top_k = retrieved[:k]

    return bool(
        set(expected) & set(top_k)
    )


def calculate_exact_match(
    expected,
    retrieved,
):
    """
    Exact record-set match.

    Useful for our deterministic structured baseline.
    """

    return (
        set(expected)
        == set(retrieved)
    )


# ============================================================
# SINGLE TEST EVALUATION
# ============================================================

def evaluate_test(
    test,
    records,
):
    """
    Execute one golden query and calculate metrics.
    """

    results = multi_intent_retrieve(
        test["query"],
        records,
    )

    retrieved_ids = collect_retrieved_record_ids(
        results
    )

    expected_ids = test["expected_records"]

    precision = calculate_precision(
        expected_ids,
        retrieved_ids,
    )

    recall = calculate_recall(
        expected_ids,
        retrieved_ids,
    )

    hit_at_1 = calculate_hit_at_k(
        expected_ids,
        retrieved_ids,
        1,
    )

    hit_at_3 = calculate_hit_at_k(
        expected_ids,
        retrieved_ids,
        3,
    )

    hit_at_5 = calculate_hit_at_k(
        expected_ids,
        retrieved_ids,
        5,
    )

    exact_match = calculate_exact_match(
        expected_ids,
        retrieved_ids,
    )

    return {
        "id": test["id"],
        "category": test["category"],
        "query": test["query"],
        "expected": expected_ids,
        "retrieved": retrieved_ids,
        "precision": precision,
        "recall": recall,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "hit_at_5": hit_at_5,
        "exact_match": exact_match,
    }


# ============================================================
# CATEGORY METRICS
# ============================================================

def calculate_category_metrics(results):
    """
    Calculate aggregate metrics per category.
    """

    categories = {}

    for result in results:

        category = result["category"]

        categories.setdefault(
            category,
            []
        )

        categories[category].append(
            result
        )

    summary = {}

    for category, items in categories.items():

        count = len(items)

        summary[category] = {
            "queries": count,

            "precision": (
                sum(
                    x["precision"]
                    for x in items
                )
                / count
            ),

            "recall": (
                sum(
                    x["recall"]
                    for x in items
                )
                / count
            ),

            "hit_at_1": (
                sum(
                    x["hit_at_1"]
                    for x in items
                )
                / count
            ),

            "hit_at_3": (
                sum(
                    x["hit_at_3"]
                    for x in items
                )
                / count
            ),

            "hit_at_5": (
                sum(
                    x["hit_at_5"]
                    for x in items
                )
                / count
            ),

            "exact_match": (
                sum(
                    x["exact_match"]
                    for x in items
                )
                / count
            ),
        }

    return summary


# ============================================================
# OVERALL METRICS
# ============================================================

def calculate_overall_metrics(results):
    """
    Calculate overall evaluation metrics.
    """

    total = len(results)

    if total == 0:
        return {}

    return {
        "queries": total,

        "precision": (
            sum(
                x["precision"]
                for x in results
            )
            / total
        ),

        "recall": (
            sum(
                x["recall"]
                for x in results
            )
            / total
        ),

        "hit_at_1": (
            sum(
                x["hit_at_1"]
                for x in results
            )
            / total
        ),

        "hit_at_3": (
            sum(
                x["hit_at_3"]
                for x in results
            )
            / total
        ),

        "hit_at_5": (
            sum(
                x["hit_at_5"]
                for x in results
            )
            / total
        ),

        "exact_match": (
            sum(
                x["exact_match"]
                for x in results
            )
            / total
        ),
    }


# ============================================================
# DISPLAY
# ============================================================

def percentage(value):
    return f"{value * 100:.2f}%"


def print_result(result):

    status = (
        "PASS"
        if result["exact_match"]
        else "FAIL"
    )

    print()
    print("-" * 72)
    print(
        f"{result['id']} | "
        f"{result['category']} | "
        f"{status}"
    )
    print("-" * 72)

    print(
        f"Query     : {result['query']}"
    )

    print(
        f"Expected  : {result['expected']}"
    )

    print(
        f"Retrieved : {result['retrieved']}"
    )

    print(
        f"Precision : "
        f"{percentage(result['precision'])}"
    )

    print(
        f"Recall    : "
        f"{percentage(result['recall'])}"
    )

    print(
        f"Hit@1     : "
        f"{result['hit_at_1']}"
    )

    print(
        f"Hit@3     : "
        f"{result['hit_at_3']}"
    )

    print(
        f"Hit@5     : "
        f"{result['hit_at_5']}"
    )

    print(
        f"Exact     : "
        f"{result['exact_match']}"
    )


def print_summary(
    overall,
    category_summary,
):
    """

    Print final evaluation report.
    """

    print()
    print("=" * 72)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 72)

    print(
        f"Queries       : "
        f"{overall['queries']}"
    )

    print(
        f"Precision     : "
        f"{percentage(overall['precision'])}"
    )

    print(
        f"Recall        : "
        f"{percentage(overall['recall'])}"
    )

    print(
        f"Hit@1         : "
        f"{percentage(overall['hit_at_1'])}"
    )

    print(
        f"Hit@3         : "
        f"{percentage(overall['hit_at_3'])}"
    )

    print(
        f"Hit@5         : "
        f"{percentage(overall['hit_at_5'])}"
    )

    print(
        f"Exact Match   : "
        f"{percentage(overall['exact_match'])}"
    )

    print()
    print("=" * 72)
    print("CATEGORY RESULTS")
    print("=" * 72)

    for category, metrics in category_summary.items():

        print()
        print(
            f"[{category}]"
        )

        print(
            f"Queries      : "
            f"{metrics['queries']}"
        )

        print(
            f"Precision    : "
            f"{percentage(metrics['precision'])}"
        )

        print(
            f"Recall       : "
            f"{percentage(metrics['recall'])}"
        )

        print(
            f"Hit@1        : "
            f"{percentage(metrics['hit_at_1'])}"
        )

        print(
            f"Hit@3        : "
            f"{percentage(metrics['hit_at_3'])}"
        )

        print(
            f"Exact Match  : "
            f"{percentage(metrics['exact_match'])}"
        )

    print()
    print("=" * 72)


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    results,
    overall,
    category_summary,
    output_path,
):
    """
    Save evaluation results as JSON.
    """

    report = {
        "evaluation": {
            "name": "structured-retrieval-baseline",
            "version": "1.0",
        },

        "overall": overall,

        "categories": category_summary,

        "queries": results,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Report saved : {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Step 4.6 Retrieval Evaluation"
        )
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=(
            "Path to retrieval_enriched.json"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            r"E:\PDF Ingestion\data\output"
            r"\retrieval_evaluation.json"
        ),
        help=(
            "Evaluation report output path"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 4.6 — RETRIEVAL EVALUATION")
    print("=" * 72)

    print(
        f"Input dataset : {args.input}"
    )

    records = load_records(
        Path(args.input)
    )

    print(
        f"Records       : {len(records)}"
    )

    # --------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------

    results = []

    for test in GOLDEN_TESTS:

        result = evaluate_test(
            test,
            records,
        )

        results.append(result)

        print_result(result)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    overall = calculate_overall_metrics(
        results
    )

    category_summary = calculate_category_metrics(
        results
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print_summary(
        overall,
        category_summary,
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    save_report(
        results,
        overall,
        category_summary,
        Path(args.output),
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()