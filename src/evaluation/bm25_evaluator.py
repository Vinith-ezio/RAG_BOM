"""
BM25 Retrieval Evaluation

Evaluates BM25 retrieval against the same golden test dataset
used by the structured retrieval evaluator.

Metrics:
    - MRR
    - Hit@1
    - Hit@3
    - Hit@5
    - Precision@5
    - Recall@5
    - Full-query Hit@K for multi-intent queries
    - Negative-query abstention

This is a BASELINE evaluator.
No score boosting or heuristic ranking is applied.
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean


# ---------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"
RETRIEVAL_DIR = SRC_DIR / "retrieval"

OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(RETRIEVAL_DIR))


# ---------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------

from bm25_retriever import BM25Retriever

from retrieval_evaluator import GOLDEN_TESTS

from multi_intent_retriever import (
    split_multi_intent_query,
    detect_item,
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

DEFAULT_TOP_K = 5

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "retrieval_enriched.json"
)

OUTPUT_FILE = OUTPUT_DIR / "bm25_evaluation.json"


# ---------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------

def reciprocal_rank(expected_ids, retrieved_ids):
    """
    Calculate Reciprocal Rank.

    Example:
        expected = ["abc"]
        retrieved = ["x", "y", "abc"]

        RR = 1 / 3
    """

    expected_set = set(expected_ids)

    for rank, record_id in enumerate(retrieved_ids, start=1):

        if record_id in expected_set:
            return 1.0 / rank

    return 0.0


def hit_at_k(expected_ids, retrieved_ids, k):
    """
    Returns 1 if at least one expected record
    appears in top-k results.
    """

    expected_set = set(expected_ids)

    top_k = retrieved_ids[:k]

    return int(
        any(record_id in expected_set for record_id in top_k)
    )


def precision_at_k(expected_ids, retrieved_ids, k):
    """
    Precision@K.

    Number of relevant retrieved records / K.
    """

    expected_set = set(expected_ids)

    top_k = retrieved_ids[:k]

    if not top_k:
        return 0.0

    relevant = sum(
        1
        for record_id in top_k
        if record_id in expected_set
    )

    return relevant / len(top_k)


def recall_at_k(expected_ids, retrieved_ids, k):
    """
    Recall@K.

    Number of relevant records retrieved / total relevant records.
    """

    expected_set = set(expected_ids)

    if not expected_set:
        return None

    top_k = set(retrieved_ids[:k])

    relevant_found = len(
        expected_set.intersection(top_k)
    )

    return relevant_found / len(expected_set)


def full_query_hit(expected_per_intent, retrieved_per_intent, k):
    """
    For multi-intent queries.

    Returns 1 only if EVERY intent finds its expected
    record within top-k.

    Example:

        Intent 1 -> expected record found
        Intent 2 -> expected record found

        => full-query hit = 1
    """

    if not expected_per_intent:
        return 0

    for expected_ids, retrieved_ids in zip(
        expected_per_intent,
        retrieved_per_intent
    ):

        if hit_at_k(expected_ids, retrieved_ids, k) == 0:
            return 0

    return 1


# ---------------------------------------------------------------------
# QUERY PREPARATION
# ---------------------------------------------------------------------

def prepare_intents(query, records):
    """
    Split a query into individual intents.

    For multi-intent queries, the ITEM is shared across
    intents
    when it is available.
    """

    intents = split_multi_intent_query(query)

    # detect_item() requires the retrieval records
    item = detect_item(query, records)

    prepared = []

    for intent in intents:

        intent = intent.strip()

        if item and item.lower() not in intent.lower():
            intent = f"{intent} {item}"

        prepared.append(intent)

    return prepared


# ---------------------------------------------------------------------
# EXPECTED RECORD MAPPING
# ---------------------------------------------------------------------

def map_expected_records(intents, expected_records):
    """
    Map expected records to intents.

    For single intent:
        1 intent -> 1 expected record

    For multi intent:
        N intents -> N expected records

    If the counts do not match, the mapping falls back
    to assigning the complete expected set to each intent.
    """

    if not expected_records:

        return [[] for _ in intents]

    if len(intents) == len(expected_records):

        return [
            [record_id]
            for record_id in expected_records
        ]

    return [
        expected_records
        for _ in intents
    ]


# ---------------------------------------------------------------------
# DATASET LOADER
# ---------------------------------------------------------------------

def load_records():
    """
    Load enriched retrieval records used by BM25.
    """

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"BM25 input dataset not found:\n{DATA_FILE}"
        )

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    # Handle normal list format
    if isinstance(data, list):
        records = data

    # Handle dictionary containing records
    elif isinstance(data, dict):

        if "records" in data:
            records = data["records"]

        else:
            raise ValueError(
                "JSON file does not contain a 'records' field."
            )

    else:

        raise ValueError(
            "Unsupported retrieval dataset format."
        )

    if not records:

        raise ValueError(
            "Retrieval dataset contains zero records."
        )

    return records

# ---------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------

def evaluate(top_k):
    """
    Run BM25 evaluation against the golden dataset.
    """

    print("=" * 80)
    print("BM25 RETRIEVAL EVALUATION")
    print("=" * 80)

    print(f"Golden tests : {len(GOLDEN_TESTS)}")
    print(f"Top-K        : {top_k}")
    print()

    print("Loading BM25 dataset...")

    records = load_records()

    print(f"Loaded records  : {len(records)}")

    print("Building BM25 index...")

    retriever = BM25Retriever(records)

    print(f"Indexed records  : {len(retriever.records)}")
    print()

    results = []

    positive_intent_metrics = []

    negative_results = []

    for test in GOLDEN_TESTS:

        test_id = test["id"]
        query = test["query"]
        expected_records = test["expected_records"]

        intents = prepare_intents(
            query,
            records
        )

        expected_per_intent = map_expected_records(
            intents,
            expected_records
        )

        intent_results = []

        for intent_index, intent_query in enumerate(intents):

            expected_ids = expected_per_intent[
                intent_index
            ]

            retrieved = retriever.search(
                intent_query,
                top_k=top_k
            )

            retrieved_ids = [
                result["record_id"]
                for result in retrieved
            ]

            # ---------------------------------------------------------
            # NEGATIVE QUERY
            # ---------------------------------------------------------

            if not expected_ids:

                intent_result = {
                    "intent_index": intent_index + 1,
                    "query": intent_query,
                    "expected_records": [],
                    "retrieved_records": retrieved_ids,
                    "hit_at_1": None,
                    "hit_at_3": None,
                    "hit_at_5": None,
                    "mrr": None,
                    "precision_at_5": None,
                    "recall_at_5": None,
                }

                negative_results.append(
                    {
                        "test_id": test_id,
                        "query": query,
                        "retrieved_records": retrieved_ids,
                        "abstained": False,
                    }
                )

            # ---------------------------------------------------------
            # POSITIVE QUERY
            # ---------------------------------------------------------

            else:

                rr = reciprocal_rank(
                    expected_ids,
                    retrieved_ids
                )

                p5 = precision_at_k(
                    expected_ids,
                    retrieved_ids,
                    5
                )

                r5 = recall_at_k(
                    expected_ids,
                    retrieved_ids,
                    5
                )

                intent_result = {
                    "intent_index": intent_index + 1,
                    "query": intent_query,
                    "expected_records": expected_ids,
                    "retrieved_records": retrieved_ids,
                    "hit_at_1": hit_at_k(
                        expected_ids,
                        retrieved_ids,
                        1
                    ),
                    "hit_at_3": hit_at_k(
                        expected_ids,
                        retrieved_ids,
                        3
                    ),
                    "hit_at_5": hit_at_k(
                        expected_ids,
                        retrieved_ids,
                        5
                    ),
                    "mrr": rr,
                    "precision_at_5": p5,
                    "recall_at_5": r5,
                }

                positive_intent_metrics.append(
                    intent_result
                )

            intent_results.append(
                intent_result
            )

        # -------------------------------------------------------------
        # FULL QUERY METRICS
        # -------------------------------------------------------------

        retrieved_per_intent = [
            item["retrieved_records"]
            for item in intent_results
        ]

        query_result = {
            "id": test_id,
            "query": query,
            "expected_records": expected_records,
            "intents": intent_results,
            "full_query_hit_at_1": None,
            "full_query_hit_at_3": None,
            "full_query_hit_at_5": None,
        }

        if expected_records:

            query_result["full_query_hit_at_1"] = full_query_hit(
                expected_per_intent,
                retrieved_per_intent,
                1
            )

            query_result["full_query_hit_at_3"] = full_query_hit(
                expected_per_intent,
                retrieved_per_intent,
                3
            )

            query_result["full_query_hit_at_5"] = full_query_hit(
                expected_per_intent,
                retrieved_per_intent,
                5
            )

        results.append(query_result)

        # -------------------------------------------------------------
        # CONSOLE OUTPUT
        # -------------------------------------------------------------

        print("-" * 80)
        print(f"{test_id}: {query}")

        for index, intent_result in enumerate(
            intent_results,
            start=1
        ):

            print()
            print(f"  Intent {index}:")
            print(
                f"    Query      : "
                f"{intent_result['query']}"
            )

            print(
                f"    Expected   : "
                f"{intent_result['expected_records']}"
            )

            print(
                f"    Retrieved  : "
                f"{intent_result['retrieved_records']}"
            )

            if intent_result["expected_records"]:

                print(
                    f"    Hit@1      : "
                    f"{intent_result['hit_at_1']}"
                )

                print(
                    f"    Hit@3      : "
                    f"{intent_result['hit_at_3']}"
                )

                print(
                    f"    Hit@5      : "
                    f"{intent_result['hit_at_5']}"
                )

                print(
                    f"    MRR        : "
                    f"{intent_result['mrr']:.4f}"
                )

    # -----------------------------------------------------------------
    # AGGREGATE METRICS
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("BM25 SUMMARY")
    print("=" * 80)

    if positive_intent_metrics:

        mrr = mean(
            x["mrr"]
            for x in positive_intent_metrics
        )

        hit1 = mean(
            x["hit_at_1"]
            for x in positive_intent_metrics
        )

        hit3 = mean(
            x["hit_at_3"]
            for x in positive_intent_metrics
        )

        hit5 = mean(
            x["hit_at_5"]
            for x in positive_intent_metrics
        )

        precision5 = mean(
            x["precision_at_5"]
            for x in positive_intent_metrics
        )

        recall5 = mean(
            x["recall_at_5"]
            for x in positive_intent_metrics
        )

    else:

        mrr = 0.0
        hit1 = 0.0
        hit3 = 0.0
        hit5 = 0.0
        precision5 = 0.0
        recall5 = 0.0

    # -----------------------------------------------------------------
    # FULL QUERY METRICS
    # -----------------------------------------------------------------

    positive_query_results = [
        result
        for result in results
        if result["expected_records"]
    ]

    full_hit1 = mean(
        result["full_query_hit_at_1"]
        for result in positive_query_results
    )

    full_hit3 = mean(
        result["full_query_hit_at_3"]
        for result in positive_query_results
    )

    full_hit5 = mean(
        result["full_query_hit_at_5"]
        for result in positive_query_results
    )

    # -----------------------------------------------------------------
    # NEGATIVE QUERY
    # -----------------------------------------------------------------

    negative_count = len(negative_results)

    negative_abstentions = sum(
        1
        for result in negative_results
        if result["abstained"]
    )

    if negative_count:

        negative_abstention_rate = (
            negative_abstentions /
            negative_count
        )

    else:

        negative_abstention_rate = None

    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------

    summary = {
        "total_golden_queries": len(GOLDEN_TESTS),

        "positive_intent_count": len(
            positive_intent_metrics
        ),

        "negative_query_count": negative_count,

        "metrics": {
            "MRR": mrr,
            "Hit@1": hit1,
            "Hit@3": hit3,
            "Hit@5": hit5,
            "Precision@5": precision5,
            "Recall@5": recall5,
        },

        "full_query_metrics": {
            "FullQueryHit@1": full_hit1,
            "FullQueryHit@3": full_hit3,
            "FullQueryHit@5": full_hit5,
        },

        "negative_query_metrics": {
            "abstention_rate": negative_abstention_rate
        }
    }

    print()
    print(
        f"MRR         : {mrr:.4f}"
    )

    print(
        f"Hit@1       : {hit1:.2%}"
    )

    print(
        f"Hit@3       : {hit3:.2%}"
    )

    print(
        f"Hit@5       : {hit5:.2%}"
    )

    print(
        f"Precision@5 : {precision5:.2%}"
    )

    print(
        f"Recall@5    : {recall5:.2%}"
    )

    print()
    print("FULL QUERY / MULTI-INTENT")
    print(
        f"FullQueryHit@1 : {full_hit1:.2%}"
    )

    print(
        f"FullQueryHit@3 : {full_hit3:.2%}"
    )

    print(
        f"FullQueryHit@5 : {full_hit5:.2%}"
    )

    print()
    print("NEGATIVE QUERY")
    print(
        f"Negative queries : {negative_count}"
    )

    if negative_abstention_rate is not None:

        print(
            f"Abstention rate  : "
            f"{negative_abstention_rate:.2%}"
        )

    # -----------------------------------------------------------------
    # SAVE RESULT
    # -----------------------------------------------------------------

    output = {
        "engine": "BM25",
        "evaluation_type": "baseline",
        "top_k": top_k,
        "summary": summary,
        "negative_queries": negative_results,
        "results": results,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)

    print(
        f"Output : {OUTPUT_FILE}"
    )

    return output


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Evaluate BM25 retrieval"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of BM25 results to evaluate"
    )

    args = parser.parse_args()

    evaluate(
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()