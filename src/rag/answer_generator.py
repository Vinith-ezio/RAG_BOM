"""
Answer Generator
----------------
Connects:

Intent Retrieval Adapter
        ↓
Context Builder
        ↓
Prompt Builder
        ↓
Qwen
        ↓
Grounded Answer
"""

from pathlib import Path
import sys
from typing import Any, Dict
import time

# ---------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

QUERY_DIR = ROOT / "src" / "query"
RETRIEVAL_DIR = ROOT / "src" / "retrieval"
CONTEXT_DIR = ROOT / "src" / "context"
RAG_DIR = ROOT / "src" / "rag"

for directory in (
    QUERY_DIR,
    RETRIEVAL_DIR,
    CONTEXT_DIR,
    RAG_DIR,
):
    directory_str = str(directory)

    if directory_str not in sys.path:
        sys.path.insert(0, directory_str)


# ---------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------

from intent_retrieval_adapter import IntentRetrievalAdapter
from context_builder import ContextBuilder

from prompt_builder import build_prompt
from qwen_client import QwenClient


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

RETRIEVAL_RECORDS_PATH = (
    ROOT
    / "data"
    / "output"
    / "retrieval_enriched.json"
)

SCHEMA_CATALOG_PATH = (
    ROOT
    / "data"
    / "output"
    / "schema_catalog.json"
)

ENTITY_INDEX_PATH = (
    ROOT
    / "data"
    / "output"
    / "entity_index.json"
)


# ---------------------------------------------------------------------
# LOAD JSON
# ---------------------------------------------------------------------

import json


def load_json(path: Path) -> Dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def load_retrieval_records(path: Path):

    data = load_json(path)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        for key in (
            "records",
            "retrieval_records",
            "data",
        ):

            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(
        f"Could not find retrieval records in: {path}"
    )


# ---------------------------------------------------------------------
# APPLICATION
# ---------------------------------------------------------------------

class RAGAnswerGenerator:

    def __init__(
        self,
        qwen_model: str = "qwen2.5:7b",
    ):

        print("Loading retrieval data...")

        retrieval_records = load_retrieval_records(
            RETRIEVAL_RECORDS_PATH
        )

        schema_catalog = load_json(
            SCHEMA_CATALOG_PATH
        )

        entity_index = load_json(
            ENTITY_INDEX_PATH
        )

        print(
            f"Retrieval records loaded: "
            f"{len(retrieval_records)}"
        )

        self.adapter = IntentRetrievalAdapter(
            schema_catalog=schema_catalog,
            entity_index=entity_index,
            retrieval_records=retrieval_records,
        )

        self.context_builder = ContextBuilder()

        self.qwen = QwenClient(
            model=qwen_model
        )

    def answer(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:

        # -------------------------------------------------------------
        # STEP 1 — RETRIEVAL
        # -------------------------------------------------------------

        retrieval_result = self.adapter.search(
            query,
            top_k=top_k,
        )

        records = retrieval_result.get(
            "records",
            [],
        )

        abstained = retrieval_result.get(
            "abstained",
            False,
        )

        # -------------------------------------------------------------
        # STEP 2 — ABSTENTION
        # -------------------------------------------------------------

        if abstained or not records:

            return {
                "query": query,
                "retrieval": retrieval_result,
                "context": None,
                "answer": (
                    "The requested information was not "
                    "found in the retrieved document context."
                ),
                "llm_used": False,
            }

        # -------------------------------------------------------------
        # STEP 3 — CONTEXT BUILDER
        # -------------------------------------------------------------

        context_result = self.context_builder.build(
            retrieval_result
        )

        context_records = context_result.get(
            "records",
            [],
        )

        if not context_records:

            return {
                "query": query,
                "retrieval": retrieval_result,
                "context": context_result,
                "answer": (
                    "The retrieved records did not contain "
                    "sufficient information to answer the question."
                ),
                "llm_used": False,
            }

        # -------------------------------------------------------------
        # STEP 4 — PROMPT
        # -------------------------------------------------------------

        prompt = build_prompt(
            query=query,
            context_result=context_result,
        )

        # -------------------------------------------------------------
        # STEP 5 — QWEN
        # -------------------------------------------------------------

        llm_start = time.perf_counter()

        print()
        print("=" * 80)
        print("PROMPT SENT TO QWEN")
        print("=" * 80)
        print(prompt["system"])
        print()
        print(prompt["user"])
        print("=" * 80)

        answer = self.qwen.generate(
            system_prompt=prompt["system"],
            user_prompt=prompt["user"],
            temperature=0.0,
        )

        llm_time = time.perf_counter() - llm_start

        print(
            f"\nQwen inference time: {llm_time:.2f} seconds"
        )

        # -------------------------------------------------------------
        # STEP 6 — FINAL RESULT
        # -------------------------------------------------------------

        return {
            "query": query,

            "retrieval": retrieval_result,

            "context": context_result,

            "prompt": prompt,

            "answer": answer,

            "llm_used": True,
        }


# ---------------------------------------------------------------------
# MANUAL TEST
# ---------------------------------------------------------------------

def main():

    print("=" * 80)
    print("PDF RAG — QWEN ANSWER GENERATOR")
    print("=" * 80)

    generator = RAGAnswerGenerator()

    print()

    if generator.qwen.health_check():

        print("Ollama: CONNECTED")

    else:

        print("Ollama: NOT AVAILABLE")

        print(
            "\nStart Ollama before running this script."
        )

        return

    print()

    while True:

        query = input(
            "Enter your question "
            "(or 'exit'): "
        ).strip()

        if query.lower() in {
            "exit",
            "quit",
            "q",
        }:

            break

        if not query:

            continue

        print()
        print("-" * 80)
        print("PROCESSING")
        print("-" * 80)

        try:

            result = generator.answer(query)

            retrieval = result["retrieval"]

            print(
                f"\nRetrieval Mode : "
                f"{retrieval.get('retrieval_mode')}"
            )

            print(
                f"Retrieved      : "
                f"{len(retrieval.get('records', []))}"
            )

            print(
                f"Abstained      : "
                f"{retrieval.get('abstained')}"
            )

            print()
            print("-" * 80)
            print("FINAL ANSWER")
            print("-" * 80)

            print(result["answer"])

            print()
            print(
                f"LLM Used       : "
                f"{result['llm_used']}"
            )

        except Exception as exc:

            print()
            print("ERROR:")
            print(exc)

        print()
        print("=" * 80)


if __name__ == "__main__":
    main()