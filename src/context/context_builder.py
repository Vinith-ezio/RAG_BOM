"""
Context Builder
===============

Phase 4/5 of the PDF RAG pipeline.

Input:
    Retrieval contract produced by IntentRetrievalAdapter / Hybrid retrieval.
    A plain list of records is also supported for backward compatibility.

Output:
    Deterministic structured RAG context.

Responsibilities:
    1. Extract the correct record collection from the retrieval contract.
    2. Preserve full collection results for collection lookups.
    3. Validate retrieved records.
    4. Preserve document/table/provenance metadata.
    5. Preserve section hierarchy and parameter/value relationships.
    6. Deduplicate records.
    7. Build deterministic context blocks and context text.

The builder does NOT:
    - retrieve
    - use an LLM
    - use embeddings
    - infer missing values
    - rename document fields
    - contain document-specific parameter/value mappings
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple


class ContextBuilder:
    """Convert a retrieval result into deterministic RAG context."""

    REQUIRED_FIELDS = (
        "record_id",
        "page",
        "parameter",
        "values",
    )

    def __init__(self, document_name: Optional[str] = None):
        # No document-specific default. Prefer metadata supplied by retrieval.
        self.document_name = document_name

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return []

    @staticmethod
    def _record_id(record: Dict[str, Any]) -> Optional[str]:
        value = record.get("record_id")
        return str(value) if value is not None else None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_record(self, record: Any) -> Dict[str, Any]:
        if not isinstance(record, dict):
            return {
                "valid": False,
                "missing_fields": list(self.REQUIRED_FIELDS),
                "error": "record is not an object",
            }

        missing = [
            field
            for field in self.REQUIRED_FIELDS
            if field not in record
        ]

        return {
            "valid": not missing,
            "missing_fields": missing,
        }

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def deduplicate_records(
        self,
        records: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        unique: List[Dict[str, Any]] = []
        seen = set()

        for record in records:
            record_id = self._record_id(record)
            if not record_id or record_id in seen:
                continue

            seen.add(record_id)
            unique.append(record)

        return unique

    # ------------------------------------------------------------------
    # Value formatting
    # ------------------------------------------------------------------

    def format_values(self, values: Any) -> List[str]:
        if values is None:
            return []

        if not isinstance(values, dict):
            return [f"Value: {values}"]

        lines: List[str] = []
        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue

            label = str(key).replace("_", " ").strip().title()
            lines.append(f"{label}: {value}")

        return lines

    # ------------------------------------------------------------------
    # Record -> deterministic text
    # ------------------------------------------------------------------

    def record_to_text(self, record: Dict[str, Any]) -> str:
        lines: List[str] = []

        document_name = (
            self.document_name
            or record.get("document")
            or record.get("document_name")
        )
        if document_name:
            lines.append(f"Document: {document_name}")

        ordered_fields: Tuple[Tuple[str, str], ...] = (
            ("page", "Page"),
            ("table_id", "Table"),
            ("row_number", "Row"),
            ("item", "Item"),
            ("section", "Section"),
            ("subsection", "Subsection"),
            ("parameter", "Parameter"),
            ("sub_parameter", "Sub-parameter"),
            ("sub_sub_parameter", "Sub-sub-parameter"),
            ("side", "Side"),
        )

        for field, label in ordered_fields:
            value = record.get(field)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            lines.append(f"{label}: {value}")

        value_lines = self.format_values(record.get("values"))
        lines.extend(value_lines)

        # Keep an already-generated retrieval_text available as provenance,
        # but do not replace structured fields with it.
        if record.get("retrieval_text"):
            lines.append(f"Retrieval Text: {record['retrieval_text']}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Retrieval-contract extraction
    # ------------------------------------------------------------------

    def _extract_from_intent(
        self,
        intent_result: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Extract records from one intent-level retrieval result."""

        mode = intent_result.get("retrieval_mode")

        # Collection lookups must consume the full deterministic collection,
        # not the preview returned in `results[:top_k]`.
        if mode in {
            "section_lookup",
            "entity_scoped_lookup",
            "identifier_to_item",
        }:
            candidates = intent_result.get("structured_records")
            if isinstance(candidates, list):
                return candidates, {
                    "source": "structured_records",
                    "retrieval_mode": mode,
                    "collection": True,
                }

        # Generic explicit collection marker for future adapters.
        if intent_result.get("collection_lookup") is True:
            candidates = intent_result.get("structured_records")
            if isinstance(candidates, list):
                return candidates, {
                    "source": "structured_records",
                    "retrieval_mode": mode,
                    "collection": True,
                }

        candidates = intent_result.get("results")
        if isinstance(candidates, list):
            return candidates, {
                "source": "results",
                "retrieval_mode": mode,
                "collection": False,
            }

        candidates = intent_result.get("structured_records")
        if isinstance(candidates, list):
            return candidates, {
                "source": "structured_records",
                "retrieval_mode": mode,
                "collection": False,
            }

        return [], {
            "source": "none",
            "retrieval_mode": mode,
            "collection": False,
        }

    def extract_records(
        self,
        retrieval_result: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Convert a retrieval contract into the records that should become
        context.

        Supported inputs:
            - list[record]                       (legacy)
            - single retrieval contract dict
            - multi-intent retrieval contract
        """

        if isinstance(retrieval_result, list):
            return retrieval_result, {
                "source": "list",
                "retrieval_mode": "legacy_list",
                "collection": False,
            }

        if not isinstance(retrieval_result, dict):
            raise TypeError(
                "retrieval_result must be a retrieval contract dict "
                "or a list of records."
            )

        # Native multi-intent contract.
        intents = retrieval_result.get("intents")
        if isinstance(intents, list):
            combined: List[Dict[str, Any]] = []
            sources = []

            for intent_result in intents:
                if not isinstance(intent_result, dict):
                    continue
                records, metadata = self._extract_from_intent(intent_result)
                combined.extend(records)
                sources.append(metadata)

            # Some adapters expose a flattened list in addition to intent
            # results. Use it only when no intent-level records were found.
            if not combined:
                flattened = retrieval_result.get("flattened_results")
                if isinstance(flattened, list):
                    combined = flattened
                    sources.append({
                        "source": "flattened_results",
                        "retrieval_mode": retrieval_result.get("retrieval_mode"),
                        "collection": False,
                    })

            return combined, {
                "source": "intents",
                "retrieval_mode": retrieval_result.get("retrieval_mode", "multi"),
                "collection": any(s.get("collection") for s in sources),
                "intent_sources": sources,
            }

        # Single retrieval contract.
        return self._extract_from_intent(retrieval_result)

    # ------------------------------------------------------------------
    # Metadata/provenance
    # ------------------------------------------------------------------

    def _build_retrieval_metadata(
        self,
        retrieval_result: Any,
        extraction_metadata: Dict[str, Any],
        record_count: int,
    ) -> Dict[str, Any]:
        if not isinstance(retrieval_result, dict):
            return {
                "retrieval_mode": extraction_metadata.get("retrieval_mode"),
                "source": extraction_metadata.get("source"),
                "collection": extraction_metadata.get("collection", False),
                "record_count": record_count,
            }

        intent = retrieval_result.get("intent")

        return {
            "retrieval_mode": retrieval_result.get(
                "retrieval_mode",
                extraction_metadata.get("retrieval_mode"),
            ),
            "abstained": bool(retrieval_result.get("abstained", False)),
            "query": retrieval_result.get("query"),
            "intent": intent if isinstance(intent, dict) else None,
            "identifier": retrieval_result.get("identifier"),
            "result_count": retrieval_result.get("result_count"),
            "source": extraction_metadata.get("source"),
            "collection": extraction_metadata.get("collection", False),
            "context_record_count": record_count,
        }

    # ------------------------------------------------------------------
    # Build context
    # ------------------------------------------------------------------

    def build(self, retrieval_result: Any) -> Dict[str, Any]:
        retrieved_records, extraction_metadata = self.extract_records(
            retrieval_result
        )

        valid_records: List[Dict[str, Any]] = []
        invalid_records: List[Dict[str, Any]] = []

        for record in retrieved_records:
            validation = self.validate_record(record)

            if validation["valid"]:
                valid_records.append(record)
            else:
                invalid_records.append({
                    "record_id": (
                        record.get("record_id")
                        if isinstance(record, dict)
                        else None
                    ),
                    "missing_fields": validation["missing_fields"],
                    "error": validation.get("error"),
                })

        unique_records = self.deduplicate_records(valid_records)

        context_blocks = []
        for record in unique_records:
            # Preserve structured provenance alongside deterministic text.
            # This keeps the context useful for both auditability and later
            # LLM/RAG stages without changing the source retrieval record.
            block = {
                "record_id": record.get("record_id"),
                "page": record.get("page"),
                "table_id": record.get("table_id"),
                "row_number": record.get("row_number"),
                "item": record.get("item"),
                "section": record.get("section"),
                "subsection": record.get("subsection"),
                "parameter": record.get("parameter"),
                "sub_parameter": record.get("sub_parameter"),
                "sub_sub_parameter": record.get("sub_sub_parameter"),
                "side": record.get("side"),
                "values": record.get("values"),
                "text": self.record_to_text(record),
            }
            context_blocks.append(block)

        context_text = "\n\n".join(
            block["text"] for block in context_blocks
        )

        retrieval_metadata = self._build_retrieval_metadata(
            retrieval_result,
            extraction_metadata,
            len(unique_records),
        )

        # Resolve document name without hardcoding a specific document.
        document = self.document_name
        if document is None and isinstance(retrieval_result, dict):
            document = retrieval_result.get("document")
        if document is None:
            for record in unique_records:
                document = record.get("document") or record.get("document_name")
                if document:
                    break

        return {
            "schema": "rag-context",
            "version": "1.0",
            "document": document,
            "retrieval": retrieval_metadata,
            "retrieved_count": len(retrieved_records),
            "valid_count": len(valid_records),
            "unique_count": len(unique_records),
            "invalid_count": len(invalid_records),
            "invalid_records": invalid_records,
            "records": unique_records,
            "context_blocks": context_blocks,
            "context_text": context_text,
        }


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------

def build_context(
    retrieval_result: Any,
    document_name: Optional[str] = None,
) -> Dict[str, Any]:
    builder = ContextBuilder(document_name=document_name)
    return builder.build(retrieval_result)


# ----------------------------------------------------------------------
# Minimal local test
# ----------------------------------------------------------------------

def main() -> None:
    records = [
        {
            "record_id": "demo_1",
            "page": 1,
            "table_id": "table_1",
            "row_number": 12,
            "section": "SECTION A",
            "parameter": "Flow",
            "sub_parameter": "Min / Max",
            "side": "inlet",
            "values": {"min": "10", "max": "20", "unit": "KG/HR"},
        },
        {
            "record_id": "demo_2",
            "page": 2,
            "table_id": "table_2",
            "row_number": 8,
            "section": "SECTION B",
            "parameter": "Line Size",
            "values": {"value": '6"'},
        },
    ]

    # Simulates the adapter's collection lookup contract. The important
    # property is that `structured_records` is full while `results` is a
    # top-k preview.
    retrieval_contract = {
        "query": "What are all records in section A?",
        "retrieval_mode": "section_lookup",
        "result_count": 1,
        "structured_records": records[:1],
        "results": records[:1],
        "abstained": False,
    }

    result = ContextBuilder().build(retrieval_contract)

    print("=" * 72)
    print("CONTEXT BUILDER TEST")
    print("=" * 72)
    print(f"Retrieved : {result['retrieved_count']}")
    print(f"Valid     : {result['valid_count']}")
    print(f"Unique    : {result['unique_count']}")
    print(f"Invalid   : {result['invalid_count']}")
    print("-" * 72)
    print(result["context_text"])
    print("=" * 72)


if __name__ == "__main__":
    main()
