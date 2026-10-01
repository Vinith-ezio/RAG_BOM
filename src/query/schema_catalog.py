"""
schema_catalog.py

PHASE 1
-------
Build a document-derived schema catalog from retrieval_enriched.json.

IMPORTANT:
- No document-specific parameter names are hardcoded.
- No document-specific item names are hardcoded.
- No document-specific material codes are hardcoded.
- The catalog is generated entirely from the input JSON.

Input:
    data/output/retrieval_enriched.json

Output:
    data/output/schema_catalog.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "retrieval_enriched.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "schema_catalog.json"
)


# ============================================================
# HELPERS
# ============================================================

def clean_value(value):
    """
    Normalize simple values for catalog generation.

    We do NOT alter the source record.
    """
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        return value if value else None

    return value


def add_unique(target_list, value):
    """
    Append value only if it is non-empty and not already present.
    """
    value = clean_value(value)

    if value is None:
        return

    if value not in target_list:
        target_list.append(value)


def sorted_unique(values):
    """
    Return unique non-empty values in deterministic order.
    """
    cleaned = {
        value.strip()
        for value in values
        if isinstance(value, str) and value.strip()
    }

    return sorted(cleaned, key=str.casefold)


# ============================================================
# LOAD DATASET
# ============================================================

def load_records(path: Path):
    """
    Load retrieval_enriched.json.

    Expected structure:

    {
        "schema": {...},
        "source": {...},
        "statistics": {...},
        "records": [...]
    }
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{path}"
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            "Expected top-level JSON object."
        )

    records = data.get("records")

    if not isinstance(records, list):
        raise ValueError(
            "Expected 'records' to be a list."
        )

    return data, records


# ============================================================
# SCHEMA DISCOVERY
# ============================================================

def build_schema_catalog(source_data, records):
    """
    Discover schema information dynamically from records.

    Nothing here assumes the document contains:
        Actuator
        Flow
        Material Code
        Water Conditions
        etc.

    Those are discovered from the JSON.
    """

    sections = set()
    subsections = set()
    parameters = set()
    sub_parameters = set()
    sub_sub_parameters = set()
    sides = set()

    parameter_to_subparameters = defaultdict(set)
    section_to_parameters = defaultdict(set)
    subsection_to_parameters = defaultdict(set)

    parameter_to_sections = defaultdict(set)
    parameter_to_sides = defaultdict(set)

    value_fields = set()

    items = set()

    # --------------------------------------------------------
    # Process every record
    # --------------------------------------------------------

    for record in records:

        if not isinstance(record, dict):
            continue

        section = clean_value(record.get("section"))
        subsection = clean_value(record.get("subsection"))
        parameter = clean_value(record.get("parameter"))
        sub_parameter = clean_value(record.get("sub_parameter"))
        sub_sub_parameter = clean_value(
            record.get("sub_sub_parameter")
        )
        side = clean_value(record.get("side"))
        item = clean_value(record.get("item"))

        # ----------------------------------------------------
        # Basic fields
        # ----------------------------------------------------

        if section:
            sections.add(section)

        if subsection:
            subsections.add(subsection)

        if parameter:
            parameters.add(parameter)

        if sub_parameter:
            sub_parameters.add(sub_parameter)

        if sub_sub_parameter:
            sub_sub_parameters.add(sub_sub_parameter)

        if side:
            sides.add(side)

        if item:
            items.add(item)

        # ----------------------------------------------------
        # Parameter -> Sub-parameter hierarchy
        # ----------------------------------------------------

        if parameter and sub_parameter:
            parameter_to_subparameters[
                parameter
            ].add(sub_parameter)

        # ----------------------------------------------------
        # Section -> Parameter hierarchy
        # ----------------------------------------------------

        if section and parameter:
            section_to_parameters[
                section
            ].add(parameter)

        # ----------------------------------------------------
        # Subsection -> Parameter hierarchy
        # ----------------------------------------------------

        if subsection and parameter:
            subsection_to_parameters[
                subsection
            ].add(parameter)

        # ----------------------------------------------------
        # Parameter -> Sections
        # ----------------------------------------------------

        if parameter and section:
            parameter_to_sections[
                parameter
            ].add(section)

        # ----------------------------------------------------
        # Parameter -> Sides
        # ----------------------------------------------------

        if parameter and side:
            parameter_to_sides[
                parameter
            ].add(side)

        # ----------------------------------------------------
        # Discover value keys dynamically
        # ----------------------------------------------------

        values = record.get("values")

        if isinstance(values, dict):
            for key in values.keys():
                if isinstance(key, str) and key.strip():
                    value_fields.add(key.strip())

    # ========================================================
    # Build catalog
    # ========================================================

    catalog = {
        "schema": {
            "name": "document-derived-schema-catalog",
            "version": "1.0"
        },

        "source": {
            "input_schema": source_data.get("schema"),
            "source": source_data.get("source"),
        },

        "statistics": {
            "records_analyzed": len(records),
            "unique_sections": len(sections),
            "unique_subsections": len(subsections),
            "unique_parameters": len(parameters),
            "unique_sub_parameters": len(sub_parameters),
            "unique_sub_sub_parameters": len(
                sub_sub_parameters
            ),
            "unique_sides": len(sides),
            "unique_items": len(items),
            "unique_value_fields": len(value_fields)
        },

        "sections": sorted(
            sections,
            key=str.casefold
        ),

        "subsections": sorted(
            subsections,
            key=str.casefold
        ),

        "parameters": sorted(
            parameters,
            key=str.casefold
        ),

        "sub_parameters": sorted(
            sub_parameters,
            key=str.casefold
        ),

        "sub_sub_parameters": sorted(
            sub_sub_parameters,
            key=str.casefold
        ),

        "sides": sorted(
            sides,
            key=str.casefold
        ),

        "value_fields": sorted(
            value_fields,
            key=str.casefold
        ),

        "parameter_hierarchy": {
            parameter: sorted(
                sub_parameters,
                key=str.casefold
            )
            for parameter, sub_parameters
            in sorted(
                parameter_to_subparameters.items(),
                key=lambda x: x[0].casefold()
            )
        },

        "section_hierarchy": {
            section: sorted(
                params,
                key=str.casefold
            )
            for section, params
            in sorted(
                section_to_parameters.items(),
                key=lambda x: x[0].casefold()
            )
        },

        "subsection_hierarchy": {
            subsection: sorted(
                params,
                key=str.casefold
            )
            for subsection, params
            in sorted(
                subsection_to_parameters.items(),
                key=lambda x: x[0].casefold()
            )
        },

        "parameter_sections": {
            parameter: sorted(
                sections,
                key=str.casefold
            )
            for parameter, sections
            in sorted(
                parameter_to_sections.items(),
                key=lambda x: x[0].casefold()
            )
        },

        "parameter_sides": {
            parameter: sorted(
                sides,
                key=str.casefold
            )
            for parameter, sides
            in sorted(
                parameter_to_sides.items(),
                key=lambda x: x[0].casefold()
            )
        }
    }

    return catalog


# ============================================================
# VALIDATION
# ============================================================

def validate_catalog(catalog):
    """
    Basic consistency checks.
    """

    required_keys = [
        "schema",
        "source",
        "statistics",
        "sections",
        "parameters",
        "parameter_hierarchy",
        "section_hierarchy",
        "value_fields"
    ]

    missing = [
        key
        for key in required_keys
        if key not in catalog
    ]

    if missing:
        raise ValueError(
            f"Catalog missing required keys: {missing}"
        )

    stats = catalog["statistics"]

    if stats["records_analyzed"] <= 0:
        raise ValueError(
            "No records were analyzed."
        )

    if stats["unique_parameters"] <= 0:
        raise ValueError(
            "No parameters discovered."
        )

    return True


# ============================================================
# SAVE
# ============================================================

def save_catalog(catalog, path: Path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            catalog,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("PHASE 1 — DOCUMENT SCHEMA CATALOG GENERATOR")
    print("=" * 72)

    print(f"\nInput:")
    print(INPUT_FILE)

    print(f"\nOutput:")
    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    source_data, records = load_records(
        INPUT_FILE
    )

    print(
        f"\nRecords loaded: {len(records)}"
    )

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    catalog = build_schema_catalog(
        source_data,
        records
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_catalog(catalog)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_catalog(
        catalog,
        OUTPUT_FILE
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    stats = catalog["statistics"]

    print("\n" + "-" * 72)
    print("SCHEMA DISCOVERY SUMMARY")
    print("-" * 72)

    print(
        f"Records analyzed      : "
        f"{stats['records_analyzed']}"
    )

    print(
        f"Unique sections       : "
        f"{stats['unique_sections']}"
    )

    print(
        f"Unique subsections    : "
        f"{stats['unique_subsections']}"
    )

    print(
        f"Unique parameters     : "
        f"{stats['unique_parameters']}"
    )

    print(
        f"Unique sub-parameters : "
        f"{stats['unique_sub_parameters']}"
    )

    print(
        f"Unique sides          : "
        f"{stats['unique_sides']}"
    )

    print(
        f"Unique items          : "
        f"{stats['unique_items']}"
    )

    print(
        f"Value fields          : "
        f"{stats['unique_value_fields']}"
    )

    print("\nDiscovered parameters:")

    for parameter in catalog["parameters"]:
        print(f"  - {parameter}")

    print("\nDiscovered sections:")

    for section in catalog["sections"]:
        print(f"  - {section}")

    print("\nParameter hierarchy:")

    for parameter, children in (
        catalog["parameter_hierarchy"].items()
    ):
        if children:
            print(
                f"  {parameter}"
            )

            for child in children:
                print(
                    f"      └── {child}"
                )

    print("\n" + "-" * 72)
    print("VALIDATION PASSED")
    print("-" * 72)

    print(
        f"\nSchema catalog written to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()