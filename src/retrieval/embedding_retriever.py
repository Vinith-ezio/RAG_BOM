"""
Embedding Retriever

Standalone semantic retrieval baseline.

Input:
    data/output/retrieval_enriched.json

Embedding:
    sentence-transformers/all-MiniLM-L6-v2

Vector index:
    FAISS IndexFlatIP

No:
    - BM25
    - Structured filtering
    - LLM
    - heuristic boosting
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "retrieval_enriched.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
)

INDEX_FILE = (
    OUTPUT_DIR
    / "embedding_index.faiss"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "embedding_metadata.json"
)


# ================================================================
# CONFIGURATION
# ================================================================

MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

DEFAULT_TOP_K = 5


# ================================================================
# RECORD LOADER
# ================================================================

def load_records():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    if isinstance(data, dict):

        if "records" in data:
            records = data["records"]

        else:
            raise ValueError(
                "JSON object does not contain 'records'."
            )

    elif isinstance(data, list):

        records = data

    else:

        raise ValueError(
            "Unsupported dataset format."
        )

    return records


# ================================================================
# EMBEDDING TEXT
# ================================================================

def build_embedding_text(record):

    """
    Use the retrieval-oriented representation already
    created by the pipeline.

    We deliberately do not add custom boosts or
    query-specific rules here.
    """

    retrieval_text = (
        record.get("retrieval_text")
        or ""
    ).strip()

    if retrieval_text:
        return retrieval_text

    # Fallback if retrieval_text is missing.
    parts = []

    fields = [
        "item",
        "section",
        "subsection",
        "parameter",
        "sub_parameter",
        "sub_sub_parameter",
        "side"
    ]

    for field in fields:

        value = record.get(field)

        if value:
            parts.append(
                f"{field}: {value}"
            )

    values = record.get(
        "values",
        {}
    )

    if values:

        for key, value in values.items():

            parts.append(
                f"{key}: {value}"
            )

    return ". ".join(parts)


# ================================================================
# EMBEDDING MODEL
# ================================================================

class EmbeddingRetriever:

    def __init__(
        self,
        records,
        model_name=MODEL_NAME
    ):

        self.records = records

        print(
            f"Loading embedding model: "
            f"{model_name}"
        )

        self.model = SentenceTransformer(
            model_name
        )

        self.texts = [
            build_embedding_text(record)
            for record in records
        ]

        self.index = None


    # ============================================================
    # BUILD INDEX
    # ============================================================

    def build_index(self):

        print(
            f"Creating embeddings for "
            f"{len(self.texts)} records..."
        )

        embeddings = self.model.encode(
            self.texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        dimension = (
            embeddings.shape[1]
        )

        print(
            f"Embedding dimension: "
            f"{dimension}"
        )

        # Normalized embeddings + inner product
        # = cosine similarity.
        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            embeddings
        )

        print(
            f"FAISS vectors indexed: "
            f"{self.index.ntotal}"
        )

        return embeddings


    # ============================================================
    # SAVE INDEX
    # ============================================================

    def save_index(self):

        if self.index is None:
            raise RuntimeError(
                "Index has not been built."
            )

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            str(INDEX_FILE)
        )

        metadata = []

        for index, record in enumerate(
            self.records
        ):

            metadata.append({

                "vector_id":
                    index,

                "record_id":
                    record.get(
                        "record_id"
                    ),

                "page":
                    record.get(
                        "page"
                    ),

                "table_id":
                    record.get(
                        "table_id"
                    ),

                "item":
                    record.get(
                        "item"
                    ),

                "section":
                    record.get(
                        "section"
                    ),

                "subsection":
                    record.get(
                        "subsection"
                    ),

                "parameter":
                    record.get(
                        "parameter"
                    ),

                "sub_parameter":
                    record.get(
                        "sub_parameter"
                    ),

                "side":
                    record.get(
                        "side"
                    ),

                "values":
                    record.get(
                        "values",
                        {}
                    )

            })

        with open(
            METADATA_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"FAISS index saved: "
            f"{INDEX_FILE}"
        )

        print(
            f"Metadata saved: "
            f"{METADATA_FILE}"
        )


    # ============================================================
    # LOAD INDEX
    # ============================================================

    def load_index(self):

        if not INDEX_FILE.exists():
            raise FileNotFoundError(
                f"FAISS index not found: "
                f"{INDEX_FILE}"
            )

        self.index = faiss.read_index(
            str(INDEX_FILE)
        )

        print(
            f"Loaded FAISS index: "
            f"{self.index.ntotal} vectors"
        )


    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query,
        top_k=DEFAULT_TOP_K
    ):

        if self.index is None:
            raise RuntimeError(
                "Index is not loaded/built."
            )

        query_embedding = (
            self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = (
            self.index.search(
                query_embedding,
                top_k
            )
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index < 0:
                continue

            record = self.records[
                int(index)
            ]

            results.append({

                "rank":
                    len(results) + 1,

                "score":
                    float(score),

                "record_id":
                    record.get(
                        "record_id"
                    ),

                "page":
                    record.get(
                        "page"
                    ),

                "item":
                    record.get(
                        "item"
                    ),

                "section":
                    record.get(
                        "section"
                    ),

                "subsection":
                    record.get(
                        "subsection"
                    ),

                "parameter":
                    record.get(
                        "parameter"
                    ),

                "sub_parameter":
                    record.get(
                        "sub_parameter"
                    ),

                "side":
                    record.get(
                        "side"
                    ),

                "values":
                    record.get(
                        "values",
                        {}
                    )

            })

        return results


# ================================================================
# BUILD + TEST
# ================================================================

def main():

    print()
    print("=" * 80)
    print("EMBEDDING RETRIEVER")
    print("=" * 80)

    records = load_records()

    print(
        f"Records loaded: {len(records)}"
    )

    retriever = EmbeddingRetriever(
        records
    )

    retriever.build_index()

    retriever.save_index()

    print()
    print("=" * 80)
    print("SANITY TESTS")
    print("=" * 80)

    test_queries = [

        (
            'What is the inlet steam flow of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),

        (
            'What is the piping class of the '
            'outlet steam of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),

        (
            'What is the material code of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        ),

        (
            'What is the inlet line size of '
            'DSH 3"300RF-INTEG TCV 1"300RF-HART?'
        ),

        (
            'What is the turbine shaft diameter of '
            'DSH 6"300RF-INTEG TCV 1"300RF-HART?'
        )

    ]

    for query in test_queries:

        print()
        print(
            f"Query: {query}"
        )

        results = retriever.search(
            query,
            top_k=5
        )

        for result in results:

            print(
                f"  {result['rank']}. "
                f"{result['record_id']} "
                f"| score="
                f"{result['score']:.4f}"
            )

    print()
    print("=" * 80)
    print("EMBEDDING RETRIEVER READY")
    print("=" * 80)


if __name__ == "__main__":
    main()