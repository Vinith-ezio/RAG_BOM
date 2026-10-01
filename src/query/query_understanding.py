from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCHEMA_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "schema_catalog.json"
)

ENTITY_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "entity_index.json"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value: Any) -> str:
    """
    Normalize text for document-agnostic matching.

    Rules:
        - Convert to string
        - Strip whitespace
        - Uppercase
        - Normalize smart quotes
        - Collapse repeated whitespace
    """

    if value is None:
        return ""

    text = str(value).strip().upper()

    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("’", "'")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# BUILD SCHEMA INDEX
# ============================================================

def build_schema_index(
    schema_catalog: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build fast lookup structures from schema_catalog.json.

    Everything is derived from the generated schema catalog.
    No document-specific schema values are hardcoded.
    """

    parameters = schema_catalog.get(
        "parameters",
        [],
    )

    sections = schema_catalog.get(
        "sections",
        [],
    )

    subsections = schema_catalog.get(
        "subsections",
        [],
    )

    sides = schema_catalog.get(
        "sides",
        [],
    )

    hierarchy = schema_catalog.get(
        "parameter_hierarchy",
        {},
    )

    # --------------------------------------------------------
    # Parameter lookup
    # --------------------------------------------------------

    parameter_lookup = {
        normalize_text(parameter): parameter
        for parameter in parameters
        if parameter
    }

    # --------------------------------------------------------
    # Section lookup
    # --------------------------------------------------------

    section_lookup = {
        normalize_text(section): section
        for section in sections
        if section
    }

    # --------------------------------------------------------
    # Subsection lookup
    # --------------------------------------------------------

    subsection_lookup = {
        normalize_text(subsection): subsection
        for subsection in subsections
        if subsection
    }

    # --------------------------------------------------------
    # Side lookup
    # --------------------------------------------------------

    side_lookup = {
        normalize_text(side): side
        for side in sides
        if side
    }

    # --------------------------------------------------------
    # Reverse hierarchy
    #
    # Child -> Parent
    #
    # Example:
    #
    # Piping Class
    #      ↓
    # Line Size
    # --------------------------------------------------------

    child_to_parent = {}

    for parent, children in hierarchy.items():

        if not isinstance(children, list):
            continue

        for child in children:

            if not child:
                continue

            child_to_parent[
                normalize_text(child)
            ] = parent

    return {
        "parameters": parameter_lookup,
        "sections": section_lookup,
        "subsections": subsection_lookup,
        "sides": side_lookup,
        "child_to_parent": child_to_parent,
        "original_hierarchy": hierarchy,
    }


# ============================================================
# DISCOVER IDENTIFIER FIELDS
# ============================================================

def discover_identifier_fields(
    entity_index: Dict[str, Any],
) -> List[str]:
    """
    Discover fields whose values look like document identifiers.

    This is document-derived.

    No document-specific values are hardcoded.
    """

    identifier_fields = set()

    values_lookup = (
        entity_index
        .get("lookup", {})
        .get("values", {})
    )

    for _, entries in values_lookup.items():

        if isinstance(entries, dict):
            entries = [entries]

        if not isinstance(entries, list):
            continue

        for entry in entries:

            if not isinstance(entry, dict):
                continue

            parameter = entry.get(
                "parameter"
            )

            value = entry.get(
                "value"
            )

            if not parameter:
                continue

            if not isinstance(value, str):
                continue

            value = value.strip()

            if not value:
                continue

            # ------------------------------------------------
            # Candidate identifier
            #
            # reasonably long
            # contains alphabetic characters
            # contains numeric characters
            # ------------------------------------------------

            if (
                len(value) >= 6
                and re.search(
                    r"[A-Z]",
                    value,
                    re.IGNORECASE,
                )
                and re.search(
                    r"\d",
                    value,
                )
            ):
                identifier_fields.add(
                    parameter
                )

    return sorted(
        identifier_fields
    )


# ============================================================
# BUILD DOCUMENT VALUE INDEX
# ============================================================

def build_document_value_index(
    entity_index: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build normalized document-value lookup.
    """

    values = (
        entity_index
        .get("lookup", {})
        .get("values", {})
    )

    index = {}

    for normalized_value, entries in values.items():

        if isinstance(entries, dict):
            entries = [entries]

        if not isinstance(entries, list):
            continue

        clean_entries = [
            entry
            for entry in entries
            if isinstance(entry, dict)
        ]

        if not clean_entries:
            continue

        index[
            normalize_text(normalized_value)
        ] = clean_entries

    return index


# ============================================================
# IDENTIFIER DETECTION
# ============================================================

def detect_identifier(
    query: str,
    entity_index: Dict[str, Any],
    identifier_fields: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Resolve document-derived identifiers.

    Priority:

        1. Full ITEM match
        2. Identifier-like document value
        3. Longer match wins

    Short numeric values are deliberately ignored.
    """

    normalized_query = normalize_text(
        query
    )

    # ========================================================
    # 1. ITEM MATCH
    # ========================================================

    item_lookup = (
        entity_index
        .get("lookup", {})
        .get("items", {})
    )

    item_candidates = []

    for normalized_item, entity in item_lookup.items():

        normalized_item = normalize_text(
            normalized_item
        )

        if not normalized_item:
            continue

        if normalized_item in normalized_query:

            if isinstance(entity, dict):

                item_candidates.append(
                    (
                        len(normalized_item),
                        entity,
                    )
                )

            elif isinstance(entity, list):

                for item in entity:

                    if isinstance(item, dict):

                        item_candidates.append(
                            (
                                len(normalized_item),
                                item,
                            )
                        )

    if item_candidates:

        item_candidates.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        _, entity = item_candidates[0]

        return {
            "type": "item",
            "value": entity.get(
                "value"
            ),
            "parameter": "ITEM",
            "record_ids": entity.get(
                "record_ids",
                [],
            ),
            "pages": entity.get(
                "pages",
                [],
            ),
            "items": entity.get(
                "items",
                [],
            ),
        }

    # ========================================================
    # 2. DOCUMENT VALUE MATCH
    # ========================================================

    value_index = build_document_value_index(
        entity_index
    )

    candidates = []

    for normalized_value, entries in value_index.items():

        # ----------------------------------------------------
        # Prevent numeric fragments such as:
        #
        # 1
        # 10
        # 105
        # 300
        #
        # from becoming identifiers.
        # ----------------------------------------------------

        if len(normalized_value) < 6:
            continue

        if normalized_value not in normalized_query:
            continue

        for entry in entries:

            parameter = entry.get(
                "parameter"
            )

            identifier_priority = (
                parameter in identifier_fields
            )

            candidates.append(
                (
                    identifier_priority,
                    len(normalized_value),
                    entry,
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            x[0],
            x[1],
        ),
        reverse=True,
    )

    _, _, entry = candidates[0]

    return {
        "type": "document_value",
        "value": entry.get(
            "value"
        ),
        "parameter": entry.get(
            "parameter"
        ),
        "field": entry.get(
            "field"
        ),
        "record_ids": entry.get(
            "record_ids",
            [],
        ),
        "pages": entry.get(
            "pages",
            [],
        ),
        "items": entry.get(
            "items",
            [],
        ),
    }


# ============================================================
# PHRASE TOKEN NORMALIZATION
# ============================================================

def normalize_phrase_tokens(
    value: Any,
) -> List[str]:
    """
    Normalize text specifically for natural-language
    schema phrase matching.

    This is intentionally separate from normalize_text()
    because document values may contain meaningful
    punctuation such as:

        "
        /
        -
        .
        etc.

    For query phrase matching, punctuation at token
    boundaries such as '?', ',', '.', '!' is ignored.
    """

    text = normalize_text(value)

    if not text:
        return []

    tokens = text.split()

    cleaned = []

    for token in tokens:

        # Remove punctuation that commonly appears
        # around natural-language query terms.

        token = token.strip(
            " \t\r\n"
            ".,!?;:"
            "()[]{}"
        )

        if token:
            cleaned.append(token)

    return cleaned


# ============================================================
# COMPLETE PHRASE MATCH
# ============================================================

def phrase_exists(
    query: str,
    phrase: str,
) -> bool:
    """
    Check whether a complete schema phrase occurs
    in a natural-language query.

    Examples:

        "What is the actuator type?"
            -> ACTUATOR TYPE      True

        "What is the material code?"
            -> MATERIAL CODE      True

        "What are the water conditions?"
            -> WATER CONDITIONS   True

    Matching is token-based so partial terms are avoided.
    """

    query_tokens = normalize_phrase_tokens(
        query
    )

    phrase_tokens = normalize_phrase_tokens(
        phrase
    )

    if not query_tokens:
        return False

    if not phrase_tokens:
        return False

    phrase_length = len(
        phrase_tokens
    )

    if phrase_length > len(query_tokens):
        return False

    for index in range(
        len(query_tokens) - phrase_length + 1
    ):

        window = query_tokens[
            index:index + phrase_length
        ]

        if window == phrase_tokens:
            return True

    return False


# ============================================================
# SCHEMA TERM MATCHING
# ============================================================

def find_schema_term(
    query: str,
    terms,
) -> Optional[str]:
    """
    Resolve a query against generated schema terms.

    Matching priority:

        1. Exact phrase
        2. Longest complete phrase
        3. No match

    Schema terms are generated from schema_catalog.json.

    No document-specific aliases are hardcoded.
    """

    if not query or not terms:
        return None

    # --------------------------------------------------------
    # Convert schema structure into canonical terms.
    # --------------------------------------------------------

    if isinstance(terms, dict):

        candidates = list(
            terms.values()
        )

    elif isinstance(terms, list):

        candidates = terms

    else:

        return None

    normalized_candidates = []

    seen = set()

    for term in candidates:

        if not term:
            continue

        canonical = str(term).strip()

        normalized = normalize_text(
            canonical
        )

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)

        normalized_candidates.append(
            {
                "canonical": canonical,
                "normalized": normalized,
                "token_count": len(
                    normalize_phrase_tokens(
                        canonical
                    )
                ),
                "char_count": len(
                    normalized
                ),
            }
        )

    if not normalized_candidates:
        return None

    # --------------------------------------------------------
    # Most specific / longest phrase first.
    # --------------------------------------------------------

    normalized_candidates.sort(
        key=lambda item: (
            item["token_count"],
            item["char_count"],
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # Exact phrase match.
    #
    # This is punctuation-aware.
    # --------------------------------------------------------

    query_tokens = normalize_phrase_tokens(
        query
    )

    for candidate in normalized_candidates:

        candidate_tokens = (
            normalize_phrase_tokens(
                candidate["canonical"]
            )
        )

        if (
            query_tokens
            == candidate_tokens
        ):
            return candidate[
                "canonical"
            ]

    # --------------------------------------------------------
    # Complete phrase contained in query.
    #
    # Longest candidate is tested first.
    # --------------------------------------------------------

    for candidate in normalized_candidates:

        if phrase_exists(
            query,
            candidate["canonical"],
        ):
            return candidate[
                "canonical"
            ]

    return None


# ============================================================
# SCHEMA RESOLUTION
# ============================================================

def resolve_schema(
    query: str,
    schema_index: Dict[str, Any],
) -> Dict[str, Optional[str]]:
    """
    Resolve:

        section
        parameter
        sub_parameter
        side

    Resolution order:

        1. Section
        2. Side
        3. Direct parameter
        4. Child under the direct parameter
        5. Child -> parent fallback

    Important:

    A direct parameter always has priority over a
    globally discovered child parameter.

    This prevents:

        actuator type

    from being interpreted as:

        Instrument Type -> Type
    """

    # ========================================================
    # SECTION
    # ========================================================

    section = find_schema_term(
        query,
        schema_index.get(
            "sections",
            {},
        ),
    )

    # ========================================================
    # SIDE
    # ========================================================

    side = find_schema_term(
        query,
        schema_index.get(
            "sides",
            {},
        ),
    )

    # ========================================================
    # DIRECT PARAMETER
    # ========================================================

    parameter = find_schema_term(
        query,
        schema_index.get(
            "parameters",
            {},
        ),
    )

    sub_parameter = None

    # ========================================================
    # DIRECT PARAMETER FOUND
    #
    # Search ONLY children belonging to this parameter.
    # ========================================================

    if parameter:

        _, child = find_sub_parameter(
            query,
            schema_index,
            parent_parameter=parameter,
        )

        if child:

            sub_parameter = child

    # ========================================================
    # NO DIRECT PARAMETER
    #
    # Search child -> parent.
    #
    # Example:
    #
    # Piping Class
    #
    # generated hierarchy:
    #
    # Line Size -> Piping Class
    # ========================================================

    else:

        parent_from_child, child = (
            find_sub_parameter(
                query,
                schema_index,
                parent_parameter=None,
            )
        )

        if parent_from_child:

            parameter = (
                parent_from_child
            )

            sub_parameter = child

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "section": section,
        "parameter": parameter,
        "sub_parameter": sub_parameter,
        "side": side,
    }


# ============================================================
# SUB-PARAMETER RESOLUTION
# ============================================================

def find_sub_parameter(
    query: str,
    schema_index: Dict[str, Any],
    parent_parameter: Optional[str] = None,
) -> Tuple[
    Optional[str],
    Optional[str],
]:
    """
    Resolve a child parameter.

    If parent_parameter is supplied:

        Only children belonging to that parent
        are considered.

    This is critical for ambiguous child names.

    Example:

        Actuator Type
            └── Spring Range

        Instrument Type
            └── Type

    Query:

        "actuator type"

    resolves to:

        Actuator Type

    and does NOT incorrectly select:

        Instrument Type -> Type
    """

    hierarchy = schema_index.get(
        "original_hierarchy",
        {},
    )

    candidates = []

    # ========================================================
    # MODE 1
    #
    # Parent is already known.
    #
    # Search ONLY its children.
    # ========================================================

    if parent_parameter:

        parent_canonical = None

        normalized_parent = normalize_text(
            parent_parameter
        )

        for parent, children in hierarchy.items():

            if (
                normalize_text(parent)
                == normalized_parent
            ):
                parent_canonical = parent
                break

        if parent_canonical is None:
            return None, None

        children = hierarchy.get(
            parent_canonical,
            [],
        )

        if not isinstance(
            children,
            list,
        ):
            return None, None

        for child in children:

            if not child:
                continue

            if phrase_exists(
                query,
                child,
            ):

                candidates.append(
                    (
                        len(
                            normalize_text(
                                child
                            ).split()
                        ),
                        len(
                            normalize_text(
                                child
                            )
                        ),
                        parent_canonical,
                        child,
                    )
                )

    # ========================================================
    # MODE 2
    #
    # Parent unknown.
    #
    # Search all child parameters.
    #
    # This is used only when no direct parameter
    # was identified.
    # ========================================================

    else:

        for parent, children in hierarchy.items():

            if not isinstance(
                children,
                list,
            ):
                continue

            for child in children:

                if not child:
                    continue

                if phrase_exists(
                    query,
                    child,
                ):

                    normalized_child = (
                        normalize_text(child)
                    )

                    candidates.append(
                        (
                            len(
                                normalized_child.split()
                            ),
                            len(
                                normalized_child
                            ),
                            parent,
                            child,
                        )
                    )

    if not candidates:
        return None, None

    # --------------------------------------------------------
    # Longest child phrase first.
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: (
            x[0],
            x[1],
        ),
        reverse=True,
    )

    _, _, parent, child = candidates[0]

    return parent, child


# ============================================================
# SCHEMA RESOLUTION
# ============================================================

def resolve_schema(
    query: str,
    schema_index: Dict[str, Any],
) -> Dict[str, Optional[str]]:
    """
    Resolve:

        section
        parameter
        sub_parameter
        side

    Important resolution order:

        1. Section
        2. Side
        3. Direct parameter
        4. Child parameter under direct parameter
        5. Child -> parent fallback

    The direct parameter MUST be resolved before globally
    searching child parameters.

    This prevents:

        "actuator type"

    from becoming:

        Instrument Type -> Type
    """

    # ========================================================
    # SECTION
    # ========================================================

    section = find_schema_term(
        query,
        schema_index["sections"],
    )

    # ========================================================
    # SIDE
    # ========================================================

    side = find_schema_term(
        query,
        schema_index["sides"],
    )

    # ========================================================
    # DIRECT PARAMETER FIRST
    # ========================================================

    parameter = find_schema_term(
        query,
        schema_index["parameters"],
    )

    sub_parameter = None

    # ========================================================
    # IF DIRECT PARAMETER FOUND
    #
    # Only search children belonging to this parameter.
    # ========================================================

    if parameter:

        _, child = find_sub_parameter(
            query,
            schema_index,
            parent_parameter=parameter,
        )

        if child:
            sub_parameter = child

    # ========================================================
    # IF NO DIRECT PARAMETER
    #
    # Search child -> parent.
    #
    # Example:
    #
    # "piping class"
    #
    # no direct parameter "Piping Class"
    #
    # hierarchy:
    #
    # Line Size -> Piping Class
    #
    # result:
    #
    # parameter = Line Size
    # sub_parameter = Piping Class
    # ========================================================

    else:

        parent_from_child, child = (
            find_sub_parameter(
                query,
                schema_index,
                parent_parameter=None,
            )
        )

        if parent_from_child:

            parameter = (
                parent_from_child
            )

            sub_parameter = child

    # ========================================================
    # ALWAYS RETURN DICTIONARY
    # ========================================================

    return {
        "section": section,
        "parameter": parameter,
        "sub_parameter": sub_parameter,
        "side": side,
    }


# ============================================================
# QUERY INTENT
# ============================================================

def build_intent(
    query: str,
    schema_index: Dict[str, Any],
    entity_index: Dict[str, Any],
    identifier_fields: List[str],
) -> Dict[str, Any]:
    """
    Build a document-agnostic query intent.
    """

    identifier = detect_identifier(
        query,
        entity_index,
        identifier_fields,
    )

    schema = resolve_schema(
        query,
        schema_index,
    )

    # ========================================================
    # RECORD SCOPE
    # ========================================================

    record_scope = []

    if identifier:

        record_scope = identifier.get(
            "record_ids",
            [],
        )

    # ========================================================
    # SCHEMA VALUES
    # ========================================================

    parameter = schema.get(
        "parameter"
    )

    section = schema.get(
        "section"
    )

    sub_parameter = schema.get(
        "sub_parameter"
    )

    side = schema.get(
        "side"
    )

    # ========================================================
    # IDENTIFIER -> ITEM
    # ========================================================

    if (
        parameter == "ITEM"
        and identifier
        and identifier.get("type")
        == "document_value"
    ):

        intent_type = (
            "identifier_to_item"
        )

    # ========================================================
    # ENTITY SCOPED LOOKUP
    # ========================================================

    elif identifier:

        intent_type = (
            "entity_scoped_lookup"
        )

    # ========================================================
    # SECTION LOOKUP
    # ========================================================

    elif section:

        intent_type = (
            "section_lookup"
        )

    # ========================================================
    # SCHEMA LOOKUP
    # ========================================================

    elif (
        parameter
        or sub_parameter
    ):

        intent_type = (
            "schema_lookup"
        )

    # ========================================================
    # NOTHING UNDERSTOOD
    # ========================================================

    else:

        intent_type = (
            "unresolved"
        )

    return {
        "query": query,
        "intent_type": intent_type,
        "identifier": identifier,
        "section": section,
        "parameter": parameter,
        "sub_parameter": sub_parameter,
        "side": side,
        "record_scope": record_scope,
    }


# ============================================================
# TEST QUERIES
# ============================================================

TEST_QUERIES = [

    'What is the actuator type of '
    'DSH 6"300RF-INTEG TCV 1"600RF-HART?',

    "What is the item of TC9765132581?",

    "What is the actuator type for "
    "material code TC9765132581?",

    "What is the inlet line size for "
    "TC9765132581?",

    "What is the piping class of "
    "the outlet steam?",

    "What are the water conditions?",

    "What is the material code?",
]


# ============================================================
# SCHEMA PRIORITY TESTS
# ============================================================

SCHEMA_PRIORITY_TESTS = [

    "What is the actuator type?",

    "What is the instrument type?",

    "What is the type?",

    "What is the line size?",

    "What is the size?",

    "What is the piping class?",

]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "PHASE 3 — DOCUMENT-AGNOSTIC QUERY UNDERSTANDING"
    )
    print("=" * 72)

    # ========================================================
    # LOAD
    # ========================================================

    schema_catalog = load_json(
        SCHEMA_FILE
    )

    entity_index = load_json(
        ENTITY_FILE
    )

    # ========================================================
    # BUILD SCHEMA INDEX
    # ========================================================

    schema_index = build_schema_index(
        schema_catalog
    )

    # ========================================================
    # DISCOVER IDENTIFIER FIELDS
    # ========================================================

    identifier_fields = (
        discover_identifier_fields(
            entity_index
        )
    )

    print()
    print(
        "Discovered identifier fields:"
    )

    for field in identifier_fields:

        print(
            f"  - {field}"
        )

    # ========================================================
    # SCHEMA PRIORITY TESTS
    # ========================================================

    print()
    print("-" * 72)
    print("SCHEMA TERM PRIORITY TESTS")
    print("-" * 72)

    for query in SCHEMA_PRIORITY_TESTS:

        result = resolve_schema(
            query,
            schema_index,
        )

        print()
        print(
            f"Query  : {query}"
        )

        print(
            f"Parameter     : "
            f"{result['parameter']}"
        )

        print(
            f"Sub-parameter : "
            f"{result['sub_parameter']}"
        )

    # ========================================================
    # QUERY TESTS
    # ========================================================

    print()
    print("-" * 72)
    print("QUERY TESTS")
    print("-" * 72)

    successful = 0

    for number, query in enumerate(
        TEST_QUERIES,
        start=1,
    ):

        result = build_intent(
            query,
            schema_index,
            entity_index,
            identifier_fields,
        )

        print()
        print(
            f"[{number}] {query}"
        )

        print(
            f"  Intent type   : "
            f"{result['intent_type']}"
        )

        print(
            f"  Identifier    : "
            f"{result['identifier']}"
        )

        print(
            f"  Section       : "
            f"{result['section']}"
        )

        print(
            f"  Parameter     : "
            f"{result['parameter']}"
        )

        print(
            f"  Sub-parameter : "
            f"{result['sub_parameter']}"
        )

        print(
            f"  Side          : "
            f"{result['side']}"
        )

        print(
            f"  Record scope  : "
            f"{len(result['record_scope'])}"
        )

        if result["intent_type"] != "unresolved":

            successful += 1

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("-" * 72)
    print("PHASE 3 SUMMARY")
    print("-" * 72)

    print(
        f"Queries tested : "
        f"{len(TEST_QUERIES)}"
    )

    print(
        f"Resolved       : "
        f"{successful}"
    )

    print(
        f"Unresolved     : "
        f"{len(TEST_QUERIES) - successful}"
    )

    print()
    print("=" * 72)
    print("PHASE 3 TEST COMPLETE")
    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()