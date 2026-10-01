"""
Phase 3.2 — Intent -> Retrieval Adapter

Updated integration adapter.

Design rules:
- Query understanding remains document-agnostic.
- Hybrid V3 remains frozen; this adapter may use deterministic structured
  candidates as an authoritative fallback/constraint.
- Identified entities never fall back to global retrieval.
- Unknown identifiers abstain instead of leaking unrelated records.
- Section lookups return the complete collection, not only the top-k preview.
- Identifier + multi-intent queries are split and resolved inside one entity scope.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# PROJECT PATHS
# ----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
QUERY_DIR = ROOT / "src" / "query"
RETRIEVAL_DIR = ROOT / "src" / "retrieval"

for directory in (QUERY_DIR, RETRIEVAL_DIR):
    value = str(directory)
    if value not in sys.path:
        sys.path.insert(0, value)


from query_understanding import (  # noqa: E402
    build_schema_index,
    discover_identifier_fields,
    build_intent,
)
from hybrid_retriever_v3 import (  # noqa: E402
    NativeMultiIntentHybridRetriever,
)
from multi_intent_retriever import (  # noqa: E402
    split_multi_intent_query,
)


# ----------------------------------------------------------------------
# NORMALIZATION
# ----------------------------------------------------------------------


def normalize(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().upper().split())


def normalize_query(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


# ----------------------------------------------------------------------
# RECORD INDEX
# ----------------------------------------------------------------------


def build_record_index(
    records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    return {
        str(record["record_id"]): record
        for record in records
        if isinstance(record, dict) and record.get("record_id")
    }


# ----------------------------------------------------------------------
# STRUCTURED FILTERS
# ----------------------------------------------------------------------


def record_matches_parameter(
    record: Dict[str, Any], parameter: Optional[str]
) -> bool:
    return not parameter or normalize(record.get("parameter")) == normalize(parameter)


def record_matches_sub_parameter(
    record: Dict[str, Any], sub_parameter: Optional[str]
) -> bool:
    return (
        not sub_parameter
        or normalize(record.get("sub_parameter")) == normalize(sub_parameter)
    )


def record_matches_section(
    record: Dict[str, Any], section: Optional[str]
) -> bool:
    return not section or normalize(record.get("section")) == normalize(section)


def record_matches_side(
    record: Dict[str, Any], side: Optional[str]
) -> bool:
    if not side:
        return True

    requested = normalize(side)
    actual = normalize(record.get("side"))

    if not requested:
        return True
    if not actual:
        return False
    if requested == actual:
        return True

    # Query understanding may retain a structural qualifier such as
    # ``INLET STEAM`` while the canonical record side is simply ``INLET``.
    requested_tokens = requested.split()
    actual_tokens = actual.split()
    return bool(actual_tokens) and actual_tokens[0] in requested_tokens


def filter_records_by_intent(
    records: List[Dict[str, Any]],
    intent: Dict[str, Any],
) -> List[Dict[str, Any]]:
    parameter = intent.get("parameter")
    sub_parameter = intent.get("sub_parameter")
    section = intent.get("section")
    side = intent.get("side")

    result = []
    for record in records:
        if not record_matches_parameter(record, parameter):
            continue
        if not record_matches_sub_parameter(record, sub_parameter):
            continue
        if not record_matches_section(record, section):
            continue
        if not record_matches_side(record, side):
            continue
        result.append(record)
    return result


# ----------------------------------------------------------------------
# ENTITY SCOPE
# ----------------------------------------------------------------------


def resolve_entity_scope(
    identifier: Optional[Dict[str, Any]],
    record_index: Dict[str, Dict[str, Any]],
    all_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Resolve an identifier to the complete entity/item scope."""

    if not identifier:
        return all_records

    identifier_type = identifier.get("type")

    if identifier_type == "item":
        target = normalize(identifier.get("value"))
        if not target:
            return []
        return [
            record
            for record in all_records
            if normalize(record.get("item")) == target
        ]

    if identifier_type == "document_value":
        items = set()

        for item in identifier.get("items", []) or []:
            if item:
                items.add(normalize(item))

        for record_id in identifier.get("record_ids", []) or []:
            record = record_index.get(str(record_id))
            if record and record.get("item"):
                items.add(normalize(record["item"]))

        if not items:
            return []

        return [
            record
            for record in all_records
            if normalize(record.get("item")) in items
        ]

    return []


# ----------------------------------------------------------------------
# DOCUMENT-DERIVED UNKNOWN IDENTIFIER DETECTION
# ----------------------------------------------------------------------


def _identifier_field_pattern(field: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", str(field))
    if not words:
        return ""
    return r"\s+".join(re.escape(word) for word in words)


def looks_like_unknown_identifier(
    query: str,
    identifier_fields: List[str],
) -> Optional[Dict[str, str]]:
    """
    Detect an identifier reference whose value was not resolved.

    This is intentionally derived from identifier fields discovered from the
    entity index. It does not know names such as 'Material Code' in advance.
    """

    text = str(query or "")

    # Document-derived identifier fields.
    for field in identifier_fields:
        pattern = _identifier_field_pattern(field)
        if not pattern:
            continue

        match = re.search(
            rf"\b{pattern}\b\s*(?:is|=|:)?\s*([A-Za-z][A-Za-z0-9._/-]{{5,}})",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            candidate = match.group(1).strip(".,;:?!")
            if candidate:
                return {"field": field, "value": candidate}

    # Generic ITEM -> value form. ITEM is a schema concept, not a document
    # specific identifier name, so only use it when a value follows it.
    match = re.search(
        r"\bitem\b\s+(?:of|for)\s+([A-Za-z][A-Za-z0-9._/-]{3,})",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return {"field": "ITEM", "value": match.group(1).strip(".,;:?!")}

    return None


# ----------------------------------------------------------------------
# SCHEMA REPAIR / RESOLUTION FALLBACK
# ----------------------------------------------------------------------


def _contains_term(query: str, term: str) -> bool:
    # Compare lexical phrases after punctuation is converted to whitespace.
    # This makes schema terms such as ``Stem / Plug Material`` match natural
    # queries such as ``stem plug material`` without introducing aliases.
    normalized_query = re.sub(r"[^a-z0-9]+", " ", normalize_query(query)).strip()
    normalized_term = re.sub(r"[^a-z0-9]+", " ", normalize_query(term)).strip()
    if not normalized_term:
        return False
    return re.search(
        rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])",
        normalized_query,
    ) is not None


def _schema_entries(schema_index: Dict[str, Any], key: str) -> List[str]:
    """Return display terms from a generated schema catalog/index."""
    raw = schema_index.get(key) or {}
    if isinstance(raw, dict):
        return [str(v) for v in raw.values() if v]
    if isinstance(raw, list):
        return [str(v) for v in raw if v]
    return []


def _schema_term_candidates(
    query: str,
    schema_index: Dict[str, Any],
) -> List[Tuple[int, str]]:
    """Find exact canonical parameter phrases occurring in the query.

    This is intentionally derived from schema_catalog.json; no document
    parameter names are hardcoded here.
    """
    normalized = normalize_query(query)
    candidates: List[Tuple[int, str]] = []

    for display in _schema_entries(schema_index, "parameters"):
        term = normalize_query(display)
        if not term:
            continue
        if _contains_term(normalized, term):
            candidates.append((len(term), display))

    return candidates


def _schema_lexical_candidates(
    query: str,
    schema_index: Dict[str, Any],
) -> List[Tuple[int, str]]:
    """Find schema parameters through conservative lexical prefix matching.

    This is only a fallback when exact Phase 3 resolution produced no
    parameter. It handles common schema abbreviations such as ``Temp.`` and
    ``Press.`` without introducing document-specific aliases.
    """
    query_tokens = re.findall(r"[a-z0-9]+", normalize_query(query))
    candidates: List[Tuple[int, str]] = []

    for display in _schema_entries(schema_index, "parameters"):
        term_tokens = re.findall(r"[a-z0-9]+", normalize_query(display))
        if not term_tokens:
            continue

        matched = 0
        for term_token in term_tokens:
            if any(
                qt == term_token
                or (len(term_token) >= 4 and len(qt) >= 4 and
                    (qt.startswith(term_token) or term_token.startswith(qt)))
                for qt in query_tokens
            ):
                matched += 1

        if matched == len(term_tokens):
            # Prefer shorter canonical parameters when the query only names
            # the generic concept (e.g. Temp. over Temp. (after ...)).
            candidates.append((matched * 1000 - len(term_tokens) * 10 - len(normalize_query(display)), display))

    return candidates


def _schema_hierarchy_matches(
    query: str,
    schema_index: Dict[str, Any],
) -> List[Tuple[int, str, str]]:
    hierarchy = (
        schema_index.get("original_hierarchy")
        or schema_index.get("_original_hierarchy")
        or {}
    )
    child_to_parent = schema_index.get("child_to_parent") or {}
    normalized = normalize_query(query)
    matches: List[Tuple[int, str, str]] = []

    if isinstance(child_to_parent, dict):
        for child, parent in child_to_parent.items():
            if child and _contains_term(normalized, str(child)):
                matches.append((len(normalize_query(str(child))), str(child), str(parent)))

    if matches:
        return matches

    if not isinstance(hierarchy, dict):
        return matches

    for parent, children in hierarchy.items():
        if not isinstance(children, list):
            continue
        for child in children:
            if child and _contains_term(normalized, str(child)):
                matches.append((len(normalize_query(str(child))), str(child), str(parent)))
    return matches


def _repair_schema_intent(
    query: str,
    intent: Dict[str, Any],
    schema_index: Dict[str, Any],
) -> Dict[str, Any]:
    """Repair ambiguous schema resolution using only generated metadata.

    The Phase 3 parser remains the primary source. Exact canonical schema
    phrases in the query take precedence over a shorter/ambiguous parser
    match. This is what prevents a phrase such as ``actuator type`` from
    being interpreted as the generic ``Type`` child of ``Instrument Type``.
    """
    repaired = dict(intent)
    if not repaired.get("type") and repaired.get("intent_type"):
        repaired["type"] = repaired.get("intent_type")

    # 1. Exact canonical parameter phrase in the generated schema.
    direct = _schema_term_candidates(query, schema_index)
    if direct:
        _, canonical_parameter = max(direct, key=lambda x: x[0])
        repaired["parameter"] = canonical_parameter

        # A direct parent parameter phrase should not inherit a child from a
        # competing parser interpretation. Keep a child only when the query
        # explicitly contains that child as well.
        hierarchy_matches = _schema_hierarchy_matches(query, schema_index)
        if hierarchy_matches:
            _, child, parent = max(hierarchy_matches, key=lambda x: x[0])
            if normalize(parent) == normalize(canonical_parameter):
                repaired["sub_parameter"] = child
            elif normalize(repaired.get("sub_parameter")) != normalize(child):
                repaired["sub_parameter"] = None
        elif repaired.get("sub_parameter") and normalize(repaired.get("parameter")) != normalize(canonical_parameter):
            repaired["sub_parameter"] = None

    # 1b. Conservative lexical fallback for generic schema abbreviations
    # such as ``temperature`` -> ``Temp.`` and ``pressure`` -> ``Press.``.
    if not repaired.get("parameter"):
        lexical = _schema_lexical_candidates(query, schema_index)
        if lexical:
            _, canonical_parameter = max(lexical, key=lambda x: x[0])
            repaired["parameter"] = canonical_parameter

    # 2. Hierarchical child-only query, e.g. "piping class".
    hierarchy_matches = _schema_hierarchy_matches(query, schema_index)
    if hierarchy_matches and not direct:
        _, child, parent = max(hierarchy_matches, key=lambda x: x[0])
        repaired["parameter"] = parent
        repaired["sub_parameter"] = child

    # 3. Fill missing section/side from generated catalog.
    normalized = normalize_query(query)
    if not repaired.get("section"):
        sections = [
            display
            for display in _schema_entries(schema_index, "sections")
            if _contains_term(normalized, display)
        ]
        if sections:
            repaired["section"] = max(sections, key=lambda x: len(normalize_query(x)))

    if not repaired.get("side"):
        sides = [
            display
            for display in _schema_entries(schema_index, "sides")
            if _contains_term(normalized, display)
        ]
        if sides:
            repaired["side"] = max(sides, key=lambda x: len(normalize_query(x)))

    # Keep the existing identifier classification semantics.
    if repaired.get("identifier"):
        if (
            normalize(repaired.get("parameter")) == "ITEM"
            and repaired["identifier"].get("type") == "document_value"
        ):
            repaired["type"] = "identifier_to_item"
        else:
            repaired["type"] = "entity_scoped_lookup"
    elif repaired.get("section") and not repaired.get("parameter") and not repaired.get("sub_parameter"):
        repaired["type"] = "section_lookup"
    elif repaired.get("parameter") or repaired.get("sub_parameter"):
        repaired["type"] = "schema_lookup"

    return repaired


def _query_has_unknown_qualifiers(
    query: str,
    intent: Dict[str, Any],
    schema_index: Dict[str, Any],
    identifier_fields: List[str],
) -> bool:
    """Detect meaningful query qualifiers absent from the generated schema.

    This prevents frozen V3 from turning unsupported questions such as
    ``motor bearing temperature`` into a generic temperature lookup.
    Structural words are derived from the schema and common query grammar;
    document values/identifiers are removed before the check.
    """
    normalized = normalize_query(query)
    tokens = set(re.findall(r"[a-z0-9]+", normalized))

    # Remove generic question grammar.
    stopwords = {
        "WHAT", "IS", "ARE", "THE", "A", "AN", "OF", "FOR", "TO",
        "FROM", "UNDER", "IN", "ON", "WITH", "AND", "OR", "DO", "DOES",
        "GIVE", "ME", "SHOW", "LIST", "INFORMATION", "VALUE", "VALUES",
        "PLEASE", "CAN", "YOU", "TELL", "ABOUT", "BE", "ALL",
    }
    tokens -= stopwords

    # Remove explicit side words and their generic structural companions.
    tokens -= {"INLET", "OUTLET", "STEAM", "WATER", "CONDITIONS"}

    # Remove canonical schema parameter/sub-parameter/section terms.
    schema_terms: List[str] = []
    schema_terms.extend(_schema_entries(schema_index, "parameters"))
    schema_terms.extend(_schema_entries(schema_index, "sections"))
    schema_terms.extend(_schema_entries(schema_index, "sides"))
    schema_terms.extend(_schema_entries(schema_index, "subsections"))

    hierarchy = (
        schema_index.get("original_hierarchy")
        or schema_index.get("_original_hierarchy")
        or {}
    )
    if isinstance(hierarchy, dict):
        for parent, children in hierarchy.items():
            schema_terms.append(str(parent))
            if isinstance(children, list):
                schema_terms.extend(str(x) for x in children if x)

    for term in schema_terms:
        tokens -= set(re.findall(r"[a-z0-9]+", normalize_query(term)))

    # Remove identifier field labels and resolved identifier values.
    for field in identifier_fields or []:
        tokens -= set(re.findall(r"[a-z0-9]+", normalize_query(field)))

    identifier = intent.get("identifier") or {}
    for value in [identifier.get("value"), *(identifier.get("items") or [])]:
        if value:
            tokens -= set(re.findall(r"[a-z0-9]+", normalize_query(value)))

    # Also remove document values that occur literally in the entity index.
    # Only small lexical values are considered; this remains metadata-driven.
    for field in identifier_fields or []:
        if _contains_term(normalized, field):
            continue

    return bool(tokens)


# ----------------------------------------------------------------------
# CONTRACT
# ----------------------------------------------------------------------


def build_retrieval_contract(
    query: str,
    intent: Dict[str, Any],
    entity_scope: List[Dict[str, Any]],
    structured_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "query": query,
        "intent": {
            "type": intent.get("type"),
            "parameter": intent.get("parameter"),
            "sub_parameter": intent.get("sub_parameter"),
            "section": intent.get("section"),
            "side": intent.get("side"),
        },
        "identifier": intent.get("identifier"),
        "entity_scoped": bool(intent.get("identifier")),
        "entity_scope_record_ids": [
            r.get("record_id") for r in entity_scope if r.get("record_id")
        ],
        "structured_record_ids": [
            r.get("record_id") for r in structured_records if r.get("record_id")
        ],
        "structured_records": structured_records,
        # Full collection for downstream context construction.
        "records": structured_records,
        "results": structured_records,
    }


# ----------------------------------------------------------------------
# MAIN ADAPTER
# ----------------------------------------------------------------------


class IntentRetrievalAdapter:
    def __init__(
        self,
        schema_catalog: Dict[str, Any],
        entity_index: Dict[str, Any],
        retrieval_records: List[Dict[str, Any]],
    ):
        self.schema_index = build_schema_index(schema_catalog)
        # Preserve the generated hierarchy for adapter-level child resolution.
        self.schema_index["original_hierarchy"] = schema_catalog.get("parameter_hierarchy", {})
        self.entity_index = entity_index
        try:
            self.identifier_fields = discover_identifier_fields(schema_catalog, entity_index)
        except TypeError as exc:
            if "takes 1 positional argument" not in str(exc):
                raise
            self.identifier_fields = discover_identifier_fields(schema_catalog)
        self.records = retrieval_records
        self.record_index = build_record_index(retrieval_records)
        self.retriever = NativeMultiIntentHybridRetriever(retrieval_records)

    def understand(self, query: str) -> Dict[str, Any]:
        intent = build_intent(
            query=query,
            schema_index=self.schema_index,
            entity_index=self.entity_index,
            identifier_fields=self.identifier_fields,
        )
        return _repair_schema_intent(query, intent, self.schema_index)

    def _search_single_entity_intent(
        self,
        query: str,
        intent: Dict[str, Any],
        entity_scope: List[Dict[str, Any]],
        top_k: int,
    ) -> Dict[str, Any]:
        scoped_records = filter_records_by_intent(entity_scope, intent)
        return {
            "query": query,
            "intent": intent,
            "results": scoped_records,
            "preview_results": scoped_records[:top_k],
            "structured_records": scoped_records,
            "result_count": len(scoped_records),
            "abstained": len(scoped_records) == 0,
            "retrieval_mode": "entity_scoped_lookup",
        }

    def _search_entity_multi_intent(
        self,
        query: str,
        base_intent: Dict[str, Any],
        entity_scope: List[Dict[str, Any]],
        top_k: int,
    ) -> Optional[Dict[str, Any]]:
        raw_intents = split_multi_intent_query(query)
        if len(raw_intents) <= 1:
            return None

        identifier = base_intent.get("identifier") or {}
        identifier_value = identifier.get("value")
        if not identifier_value:
            return None

        intent_results = []
        merged: List[Dict[str, Any]] = []
        seen = set()

        for index, raw_intent in enumerate(raw_intents, start=1):
            fragment = raw_intent.strip()
            if not fragment:
                continue

            search_query = fragment
            if normalize_query(identifier_value) not in normalize_query(fragment):
                search_query = f"{fragment} {identifier_value}"

            intent = self.understand(search_query)
            intent["identifier"] = identifier
            intent["type"] = "entity_scoped_lookup"

            scoped_records = filter_records_by_intent(entity_scope, intent)

            ids = [r.get("record_id") for r in scoped_records if r.get("record_id")]
            for record in scoped_records:
                rid = record.get("record_id")
                if rid and rid not in seen:
                    seen.add(rid)
                    merged.append(record)

            intent_results.append({
                "intent_index": index,
                "query": fragment,
                "search_query": search_query,
                "intent": intent,
                "record_ids": ids,
                "results": scoped_records[:top_k],
                "result_count": len(scoped_records),
                "abstained": len(scoped_records) == 0,
            })

        return {
            "query": query,
            "intent": base_intent,
            "intents": intent_results,
            "entity_scope": entity_scope,
            "entity_scope_record_ids": [
                r.get("record_id") for r in entity_scope if r.get("record_id")
            ],
            "structured_records": merged,
            "results": merged,
            "preview_results": merged[:top_k] if top_k > 0 else merged,
            "result_count": len(merged),
            "abstained": len(merged) == 0,
            "retrieval_mode": "entity_scoped_lookup",
            "multi_intent": True,
            "successful_intents": sum(not x["abstained"] for x in intent_results),
            "intent_count": len(intent_results),
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        intent = self.understand(query)

        # Unknown identifier must never fall through to global retrieval.
        if not intent.get("identifier"):
            unknown = looks_like_unknown_identifier(query, self.identifier_fields)
            if unknown:
                return {
                    "query": query,
                    "intent": intent,
                    "entity_scope": [],
                    "structured_records": [],
                    "records": [],
                    "results": [],
                    "result_count": 0,
                    "abstained": True,
                    "retrieval_mode": "abstained",
                    "reason": (
                        f"Identifier value '{unknown['value']}' was not resolved "
                        f"for field '{unknown['field']}'."
                    ),
                }

        if not intent.get("parameter") and not intent.get("section"):
            return {
                "query": query,
                "intent": intent,
                "entity_scope": [],
                "structured_records": [],
                "results": [],
                "result_count": 0,
                "abstained": True,
                "retrieval_mode": "abstained",
                "reason": "Intent could not be resolved.",
            }

        identifier = intent.get("identifier")
        entity_scope = resolve_entity_scope(
            identifier=identifier,
            record_index=self.record_index,
            all_records=self.records,
        )

        # --------------------------------------------------------------
        # SECTION LOOKUP — return the COMPLETE collection.
        # --------------------------------------------------------------
        if intent.get("type") == "section_lookup":
            section_records = filter_records_by_intent(self.records, intent)
            contract = build_retrieval_contract(
                query=query,
                intent=intent,
                entity_scope=self.records,
                structured_records=section_records,
            )
            return {
                **contract,
                "results": section_records,
                "records": section_records,
                "preview_results": section_records[:top_k],
                "result_count": len(section_records),
                "abstained": len(section_records) == 0,
                "retrieval_mode": "section_lookup",
            }

        # --------------------------------------------------------------
        # ENTITY-SCOPED LOOKUP
        # --------------------------------------------------------------
        if identifier:
            if not entity_scope:
                return {
                    "query": query,
                    "intent": intent,
                    "entity_scope": [],
                    "structured_records": [],
                    "records": [],
                    "results": [],
                    "result_count": 0,
                    "abstained": True,
                    "retrieval_mode": "entity_scoped_lookup",
                    "reason": "Identifier resolved, but entity scope is empty.",
                }

            multi_result = self._search_entity_multi_intent(
                query=query,
                base_intent=intent,
                entity_scope=entity_scope,
                top_k=top_k,
            )
            if multi_result is not None:
                return multi_result

            scoped = filter_records_by_intent(entity_scope, intent)

            # Header-only records can share a generic child term (for example
            # ``Type``). They are not answer-bearing records when their values
            # object is empty. Keep the filter schema-driven rather than using
            # document-specific record IDs.
            if intent.get("parameter") or intent.get("sub_parameter"):
                answerable = [
                    r for r in scoped
                    if isinstance(r.get("values"), dict) and bool(r.get("values"))
                ]
                if answerable:
                    scoped = answerable

            contract = build_retrieval_contract(
                query=query,
                intent=intent,
                entity_scope=entity_scope,
                structured_records=scoped,
            )
            return {
                **contract,
                "records": scoped,
                "results": scoped,
                "preview_results": scoped[:top_k],
                "result_count": len(scoped),
                "abstained": len(scoped) == 0,
                "retrieval_mode": "entity_scoped_lookup",
            }

        # --------------------------------------------------------------
        # GLOBAL LOOKUP
        # --------------------------------------------------------------
        structured_records = filter_records_by_intent(self.records, intent)

        v3_result = self.retriever.search(query, top_k=top_k)
        v3_records: List[Dict[str, Any]] = []
        for result in v3_result.get("flattened_results", []) or []:
            record_id = result if isinstance(result, str) else result.get("record_id")
            if record_id and record_id in self.record_index:
                v3_records.append(self.record_index[record_id])

        # Deterministic structured candidates are preferred whenever the
        # parser resolved a schema field. V3 then fills remaining slots.
        results: List[Dict[str, Any]] = []
        seen = set()
        for record in structured_records + v3_records:
            rid = record.get("record_id")
            if not rid or rid in seen:
                continue
            seen.add(rid)
            results.append(record)
            if len(results) >= top_k:
                break

        contract = build_retrieval_contract(
            query=query,
            intent=intent,
            entity_scope=self.records,
            structured_records=structured_records,
        )

        return {
            **contract,
            "records": results,
            "results": results,
            "preview_results": results[:top_k],
            "result_count": len(results),
            "abstained": len(results) == 0,
            "retrieval_mode": "global_hybrid_v3",
            "v3": v3_result,
            "structured_fallback_used": bool(structured_records),
        }
