"""
entity_index.py

PHASE 2
-------
Build a document-derived entity index from retrieval_enriched.json.

Purpose:
    Discover identifiable entities from the extracted document
    without hardcoding document-specific values.

Input:
    data/output/retrieval_enriched.json

Schema reference:
    data/output/schema_catalog.json

Output:
    data/output/entity_index.json

IMPORTANT:
    This script contains NO document-specific:
        - item names
        - material codes
        - parameter names
        - section names
        - aliases
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETRIEVAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "retrieval_enriched.json"
)

SCHEMA_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "schema_catalog.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "entity_index.json"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value: str) -> str:
    """
    Generic text normalization.

    This does NOT change the stored original value.
    It is only used for lookup keys.
    """

    if value is None:
        return ""

    value = str(value).strip()

    # Normalize repeated whitespace
    value = re.sub(r"\s+", " ", value)

    # Case-insensitive lookup
    return value.casefold()


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


# ============================================================
# GENERIC ENTITY VALUE DETECTION
# ============================================================

def is_identifier_like(value: str) -> bool:
    """
    Generic heuristic for identifier-like values.

    This is intentionally generic.

    It does NOT know about:
        TC9765132581
        DSH...
        or any specific document value.

    It looks for values that contain a meaningful combination
    of letters and/or digits and are reasonably compact.
    """

    if not isinstance(value, str):
        return False

    value = value.strip()

    if not value:
        return False

    # Ignore very long natural-language text.
    if len(value) > 150:
        return False

    # Must contain at least one alphanumeric character.
    if not re.search(r"[A-Za-z0-9]", value):
        return False

    return True


# ============================================================
# ENTITY STORE
# ============================================================

def create_entity_entry(
    entity_type: str,
    value: str
):
    """
    Create the basic entity representation.
    """

    return {
        "entity_type": entity_type,
        "value": value,
        "normalized": normalize_text(value),
        "occurrences": [],
        "record_ids": [],
        "pages": [],
        "sections": [],
        "parameters": []
    }


def add_unique(items: list, value):
    if value is None:
        return

    if value not in items:
        items.append(value)


# ============================================================
# BUILD ITEM INDEX
# ============================================================

def build_item_index(records):
    """
    Discover unique ITEM values directly from records.
    """

    entities = {}

    for record in records:

        if not isinstance(record, dict):
            continue

        item = record.get("item")

        if not is_identifier_like(item):
            continue

        normalized = normalize_text(item)

        if normalized not in entities:
            entities[normalized] = create_entity_entry(
                "item",
                item
            )

        entity = entities[normalized]

        record_id = record.get("record_id")
        page = record.get("page")
        section = record.get("section")
        parameter = record.get("parameter")

        if record_id:
            add_unique(
                entity["record_ids"],
                record_id
            )

        if page is not None:
            add_unique(
                entity["pages"],
                page
            )

        if section:
            add_unique(
                entity["sections"],
                section
            )

        if parameter:
            add_unique(
                entity["parameters"],
                parameter
            )

        entity["occurrences"].append({
            "record_id": record_id,
            "page": page,
            "section": section,
            "parameter": parameter
        })

    return entities


# ============================================================
# BUILD VALUE-BASED ENTITY INDEX
# ============================================================

def build_value_entity_index(
    records,
    schema_catalog
):
    """
    Discover candidate identifier-like values from record values.

    The important point:
        We inspect the actual JSON.

    We do not hardcode:
        Material Code
        TC...
        etc.

    Parameter names are obtained from schema_catalog.
    """

    entities = {}

    # --------------------------------------------------------
    # Parameters discovered from Phase 1
    # --------------------------------------------------------

    parameters = set(
        schema_catalog.get(
            "parameters",
            []
        )
    )

    # --------------------------------------------------------
    # Process records
    # --------------------------------------------------------

    for record in records:

        if not isinstance(record, dict):
            continue

        parameter = record.get(
            "parameter"
        )

        values = record.get(
            "values"
        )

        if not isinstance(values, dict):
            continue

        # ----------------------------------------------------
        # We are interested in values belonging to fields
        # discovered in the document.
        # ----------------------------------------------------

        for field_name, value in values.items():

            if not is_identifier_like(value):
                continue

            # ------------------------------------------------
            # Create a generic value entity.
            #
            # Example:
            #   parameter = Material Code
            #   field     = value
            #   value     = TC...
            #
            # But none of those are hardcoded here.
            # ------------------------------------------------

            normalized_value = normalize_text(
                value
            )

            entity_key = (
                parameter or "UNKNOWN",
                field_name,
                normalized_value
            )

            if entity_key not in entities:

                entities[entity_key] = {
                    "entity_type": "document_value",
                    "parameter": parameter,
                    "field": field_name,
                    "value": value,
                    "normalized": normalized_value,
                    "occurrences": [],
                    "record_ids": [],
                    "pages": [],
                    "sections": [],
                    "items": []
                }

            entity = entities[entity_key]

            record_id = record.get(
                "record_id"
            )

            page = record.get(
                "page"
            )

            section = record.get(
                "section"
            )

            item = record.get(
                "item"
            )

            entity["occurrences"].append({
                "record_id": record_id,
                "page": page,
                "section": section,
                "item": item
            })

            if record_id:
                add_unique(
                    entity["record_ids"],
                    record_id
                )

            if page is not None:
                add_unique(
                    entity["pages"],
                    page
                )

            if section:
                add_unique(
                    entity["sections"],
                    section
                )

            if item:
                add_unique(
                    entity["items"],
                    item
                )

    return entities


# ============================================================
# BUILD MAIN INDEX
# ============================================================

def build_entity_index(
    source_data,
    schema_catalog,
    records
):

    # --------------------------------------------------------
    # ITEM INDEX
    # --------------------------------------------------------

    item_entities = build_item_index(
        records
    )

    # --------------------------------------------------------
    # VALUE ENTITY INDEX
    # --------------------------------------------------------

    value_entities = build_value_entity_index(
        records,
        schema_catalog
    )

    # --------------------------------------------------------
    # Convert dictionaries into deterministic lists
    # --------------------------------------------------------

    items = sorted(
        item_entities.values(),
        key=lambda x: x["normalized"]
    )

    document_values = sorted(
        value_entities.values(),
        key=lambda x: (
            x["normalized"],
            x.get("parameter") or "",
            x.get("field") or ""
        )
    )

    # --------------------------------------------------------
    # Build lookup maps
    # --------------------------------------------------------

    item_lookup = {}

    for entity in items:

        item_lookup[
            entity["normalized"]
        ] = {
            "value": entity["value"],
            "record_ids": entity["record_ids"],
            "pages": entity["pages"]
        }

    value_lookup = {}

    for entity in document_values:

        key = entity["normalized"]

        if key not in value_lookup:
            value_lookup[key] = []

        value_lookup[key].append({
            "parameter": entity["parameter"],
            "field": entity["field"],
            "value": entity["value"],
            "record_ids": entity["record_ids"],
            "pages": entity["pages"],
            "sections": entity["sections"],
            "items": entity["items"]
        })

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    unique_value_strings = {
        entity["normalized"]
        for entity in document_values
    }

    catalog = {

        "schema": {
            "name": "document-derived-entity-index",
            "version": "1.0"
        },

        "source": {
            "retrieval_schema": source_data.get(
                "schema"
            ),
            "document": source_data.get(
                "source"
            )
        },

        "statistics": {
            "records_analyzed": len(records),
            "unique_items": len(items),
            "unique_document_values": len(
                unique_value_strings
            ),
            "document_value_entries": len(
                document_values
            )
        },

        "entities": {

            "items": items,

            "document_values": document_values
        },

        "lookup": {

            "items": item_lookup,

            "values": value_lookup
        }
    }

    return catalog


# ============================================================
# VALIDATION
# ============================================================

def validate_entity_index(
    entity_index
):

    required_sections = [
        "schema",
        "source",
        "statistics",
        "entities",
        "lookup"
    ]

    for key in required_sections:

        if key not in entity_index:

            raise ValueError(
                f"Missing required section: {key}"
            )

    if not entity_index["entities"]["items"]:

        raise ValueError(
            "No ITEM entities discovered."
        )

    return True


# ============================================================
# SAVE
# ============================================================

def save_json(
    data,
    path: Path
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("PHASE 2 — DOCUMENT ENTITY INDEX GENERATOR")
    print("=" * 72)

    print("\nRetrieval file:")
    print(RETRIEVAL_FILE)

    print("\nSchema catalog:")
    print(SCHEMA_FILE)

    print("\nOutput:")
    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # Load retrieval data
    # --------------------------------------------------------

    source_data = load_json(
        RETRIEVAL_FILE
    )

    records = source_data.get(
        "records"
    )

    if not isinstance(records, list):

        raise ValueError(
            "'records' must be a list."
        )

    print(
        f"\nRecords loaded: {len(records)}"
    )

    # --------------------------------------------------------
    # Load schema catalog
    # --------------------------------------------------------

    schema_catalog = load_json(
        SCHEMA_FILE
    )

    print(
        "Schema catalog loaded."
    )

    # --------------------------------------------------------
    # Build entity index
    # --------------------------------------------------------

    entity_index = build_entity_index(
        source_data,
        schema_catalog,
        records
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_entity_index(
        entity_index
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_json(
        entity_index,
        OUTPUT_FILE
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    stats = entity_index[
        "statistics"
    ]

    print("\n" + "-" * 72)
    print("ENTITY DISCOVERY SUMMARY")
    print("-" * 72)

    print(
        f"Records analyzed          : "
        f"{stats['records_analyzed']}"
    )

    print(
        f"Unique items              : "
        f"{stats['unique_items']}"
    )

    print(
        f"Unique document values    : "
        f"{stats['unique_document_values']}"
    )

    print(
        f"Document value entries    : "
        f"{stats['document_value_entries']}"
    )

    # --------------------------------------------------------
    # Items
    # --------------------------------------------------------

    print("\nDiscovered ITEMS:")

    for entity in (
        entity_index["entities"]["items"]
    ):

        print(
            f"  - {entity['value']}"
        )

        print(
            f"      Pages: {entity['pages']}"
        )

    # --------------------------------------------------------
    # Show sample value entities
    # --------------------------------------------------------

    print("\nSample document values:")

    sample_values = (
        entity_index[
            "entities"
        ]["document_values"][:20]
    )

    for entity in sample_values:

        print(
            f"  - [{entity['parameter']}] "
            f"{entity['field']} = "
            f"{entity['value']}"
        )

    print("\n" + "-" * 72)
    print("VALIDATION PASSED")
    print("-" * 72)

    print(
        "\nEntity index written to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()