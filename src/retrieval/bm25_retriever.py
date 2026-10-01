import argparse
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from multi_intent_retriever import (
    load_records,
    split_multi_intent_query,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_INPUT = (
    r"E:\PDF Ingestion\data\output\retrieval_enriched.json"
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text before tokenization.
    """

    if text is None:
        return ""

    text = str(text)

    text = text.lower()

    # Normalize common quote variants
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("’", "'")

    # Keep alphanumeric content and engineering symbols
    text = re.sub(
        r"[^a-z0-9\"'./+\-]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):
    """
    Tokenize searchable text.

    BM25 is lexical, so preserving meaningful
    engineering tokens is important.
    """

    normalized = normalize_text(text)

    if not normalized:
        return []

    return normalized.split()


# ============================================================
# RECORD -> SEARCH TEXT
# ============================================================

def build_search_text(record):
    """
    Build BM25 searchable text from structured record.

    We deliberately include structured metadata because
    engineering queries often use exact terminology.
    """

    parts = []

    fields = [
        ("item", record.get("item")),
        ("section", record.get("section")),
        ("subsection", record.get("subsection")),
        ("parameter", record.get("parameter")),
        ("sub_parameter", record.get("sub_parameter")),
        ("side", record.get("side")),
    ]

    for field_name, value in fields:

        if value is not None:
            parts.append(
                f"{field_name} {value}"
            )

    values = record.get("values", {})

    if isinstance(values, dict):

        for key, value in values.items():

            if value is None:
                continue

            parts.append(
                f"{key} {value}"
            )

    retrieval_text = record.get(
        "retrieval_text"
    )

    if retrieval_text:
        parts.append(
            retrieval_text
        )

    return " ".join(parts)


# ============================================================
# BM25 INDEX
# ============================================================

class BM25Retriever:
    """
    BM25 retrieval engine for structured records.
    """

    def __init__(self, records):

        self.records = records

        self.documents = [
            build_search_text(record)
            for record in records
        ]

        self.tokenized_documents = [
            tokenize(document)
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

    # --------------------------------------------------------
    # Single query
    # --------------------------------------------------------

    def search(
        self,
        query,
        top_k=5,
    ):
        """
        Return top-k BM25 results.
        """

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indices[:top_k]:

            record = dict(
                self.records[index]
            )

            record["_bm25_score"] = float(
                scores[index]
            )

            results.append(record)

        return results

    # --------------------------------------------------------
    # Multi-intent query
    # --------------------------------------------------------

    def search_multi_intent(
        self,
        query,
        top_k=5,
    ):
        """
        Run BM25 separately for each intent.
        """

        intent_queries = split_multi_intent_query(
            query
        )

        results = []

        for intent_query in intent_queries:

            matches = self.search(
                intent_query,
                top_k=top_k,
            )

            results.append(
                {
                    "query": intent_query,
                    "matches": matches,
                }
            )

        return results


# ============================================================
# DISPLAY
# ============================================================

def print_record(record):

    print(
        f"Record ID      : "
        f"{record.get('record_id')}"
    )

    print(
        f"Page           : "
        f"{record.get('page')}"
    )

    print(
        f"Section        : "
        f"{record.get('section')}"
    )

    print(
        f"Subsection     : "
        f"{record.get('subsection')}"
    )

    print(
        f"Parameter      : "
        f"{record.get('parameter')}"
    )

    print(
        f"Sub-parameter  : "
        f"{record.get('sub_parameter')}"
    )

    print(
        f"Side           : "
        f"{record.get('side')}"
    )

    print(
        f"BM25 Score     : "
        f"{record.get('_bm25_score', 0):.4f}"
    )

    print(
        f"Values         : "
        f"{record.get('values')}"
    )


# ============================================================
# BASIC TEST QUERIES
# ============================================================

TEST_QUERIES = [

    (
        "Inlet Steam Flow",
        'What is the inlet steam flow of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),

    (
        "Outlet Piping Class",
        'What is the piping class of the outlet steam '
        'of DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),

    (
        "Material Code",
        'What is the material code of '
        'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
    ),

    (
        "Page 2 Flow",
        'What is the inlet steam flow of '
        'DSH 6"300RF-INTEG TCV 1"600RF-HART?'
    ),

    (
        "Page 5 Material Code",
        'What is the material code of '
        'DSH 12"300RF-INTEG TCV 1"300RF-HART?'
    ),
]


# ============================================================
# TEST RUNNER
# ============================================================

def run_tests(retriever):

    print()
    print("=" * 72)
    print("BM25 RETRIEVAL TEST")
    print("=" * 72)

    for test_name, query in TEST_QUERIES:

        print()
        print("-" * 72)
        print(
            f"TEST : {test_name}"
        )
        print("-" * 72)

        print(
            f"Query : {query}"
        )

        results = retriever.search(
            query,
            top_k=5,
        )

        print(
            f"Results : {len(results)}"
        )

        for rank, record in enumerate(
            results,
            start=1,
        ):

            print()
            print(
                f"RANK {rank}"
            )

            print_record(record)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "BM25 retrieval engine for "
            "complex structured table records."
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
        "--query",
        default=None,
        help=(
            "Run one custom BM25 query."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help=(
            "Number of BM25 results."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("BM25 RETRIEVAL ENGINE")
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
    # Build index
    # --------------------------------------------------------

    print()
    print(
        "Building BM25 index..."
    )

    retriever = BM25Retriever(
        records
    )

    print(
        f"Indexed records : "
        f"{len(retriever.documents)}"
    )

    # --------------------------------------------------------
    # Custom query
    # --------------------------------------------------------

    if args.query:

        print()
        print("=" * 72)
        print("CUSTOM BM25 QUERY")
        print("=" * 72)

        print(
            f"Query : {args.query}"
        )

        results = retriever.search(
            args.query,
            top_k=args.top_k,
        )

        for rank, record in enumerate(
            results,
            start=1,
        ):

            print()
            print(
                f"RANK {rank}"
            )

            print_record(record)

        return

    # --------------------------------------------------------
    # Tests
    # --------------------------------------------------------

    run_tests(
        retriever
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()