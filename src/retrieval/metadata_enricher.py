import json
import argparse
import re
from pathlib import Path


def normalize_text(value):
    """
    Normalize text for exact/keyword matching.

    Original values are NOT modified.
    This is only used for searchable metadata.
    """

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    # Normalize whitespace
    value = re.sub(r"\s+", " ", value)

    # Case-insensitive search representation
    return value.upper()


def find_page_items(records):
    """
    Find ITEM values for every page.

    Example source record:

    parameter = ITEM
    values = {
        "value": "DSH 6..."
    }

    Returns:

    {
        1: "DSH 6...",
        2: "DSH 6...",
        ...
    }
    """

    page_items = {}

    for record in records:

        parameter = record.get("parameter")

        if not parameter:
            continue

        if normalize_text(parameter) != "ITEM":
            continue

        page = record.get("page")

        values = record.get("values", {})

        if not isinstance(values, dict):
            continue

        item_value = values.get("value")

        if item_value is None:
            continue

        item_value = str(item_value).strip()

        if not item_value:
            continue

        page_items[page] = item_value

    return page_items


def enrich_record(record, page_items):
    """
    Add searchable metadata to one retrieval record.
    """

    enriched = dict(record)

    page = record.get("page")

    # ---------------------------------------------------------
    # ITEM / ENTITY
    # ---------------------------------------------------------

    item = page_items.get(page)

    enriched["item"] = item

    enriched["item_normalized"] = normalize_text(item)

    # ---------------------------------------------------------
    # Existing hierarchy
    # ---------------------------------------------------------

    enriched["section_normalized"] = normalize_text(
        record.get("section")
    )

    enriched["subsection_normalized"] = normalize_text(
        record.get("subsection")
    )

    enriched["parameter_normalized"] = normalize_text(
        record.get("parameter")
    )

    enriched["sub_parameter_normalized"] = normalize_text(
        record.get("sub_parameter")
    )

    enriched["sub_sub_parameter_normalized"] = normalize_text(
        record.get("sub_sub_parameter")
    )

    enriched["side_normalized"] = normalize_text(
        record.get("side")
    )

    # ---------------------------------------------------------
    # Searchable identity
    # ---------------------------------------------------------

    enriched["search_metadata"] = {
        "item": enriched["item_normalized"],
        "page": page,
        "table_id": record.get("table_id"),
        "section": enriched["section_normalized"],
        "subsection": enriched["subsection_normalized"],
        "parameter": enriched["parameter_normalized"],
        "sub_parameter": enriched["sub_parameter_normalized"],
        "side": enriched["side_normalized"]
    }

    return enriched


def validate_records(records):
    """
    Validate metadata enrichment.
    """

    errors = []

    for record in records:

        record_id = record.get("record_id")

        # ITEM should exist for all records that belong
        # to a page containing an ITEM definition.
        if not record.get("item"):
            errors.append(
                f"{record_id}: missing item"
            )

        # Parameter should exist for retrievable records.
        if not record.get("parameter"):
            errors.append(
                f"{record_id}: missing parameter"
            )

        # Retrieval text should remain intact.
        if "retrieval_text" not in record:
            errors.append(
                f"{record_id}: missing retrieval_text"
            )

    return errors


def build_enriched_dataset(input_path, output_path):

    input_path = Path(input_path)
    output_path = Path(output_path)

    print("=" * 72)
    print("RETRIEVAL METADATA ENRICHER")
    print("STEP 4.1")
    print("=" * 72)

    print(f"Input  : {input_path}")
    print(f"Output : {output_path}")
    print("-" * 72)

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as f:

        dataset = json.load(f)

    records = dataset.get("records", [])

    if not records:
        raise ValueError(
            "No records found in input dataset."
        )

    # ---------------------------------------------------------
    # Find ITEM per page
    # ---------------------------------------------------------

    page_items = find_page_items(records)

    print(
        f"Pages with ITEM identified : "
        f"{len(page_items)}"
    )

    for page, item in sorted(page_items.items()):

        print(
            f"  Page {page}: {item}"
        )

    print("-" * 72)

    # ---------------------------------------------------------
    # Enrich records
    # ---------------------------------------------------------

    enriched_records = []

    for record in records:

        enriched = enrich_record(
            record,
            page_items
        )

        enriched_records.append(enriched)

    # ---------------------------------------------------------
    # Validate
    # ---------------------------------------------------------

    errors = validate_records(
        enriched_records
    )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    pages = sorted(
        set(
            record.get("page")
            for record in enriched_records
            if record.get("page") is not None
        )
    )

    records_with_item = sum(
        1
        for record in enriched_records
        if record.get("item")
    )

    records_with_side = sum(
        1
        for record in enriched_records
        if record.get("side")
    )

    records_with_section = sum(
        1
        for record in enriched_records
        if record.get("section")
    )

    statistics = {
        "input_records": len(records),
        "output_records": len(enriched_records),
        "pages": len(pages),
        "pages_with_item": len(page_items),
        "records_with_item": records_with_item,
        "records_with_side": records_with_side,
        "records_with_section": records_with_section,
        "validation_errors": len(errors)
    }

    # ---------------------------------------------------------
    # Build output
    # ---------------------------------------------------------

    output_dataset = {
        "schema": {
            "name": "complex-table-retrieval-enriched",
            "version": "1.0"
        },

        "source": dataset.get(
            "source",
            {}
        ),

        "statistics": statistics,

        "records": enriched_records
    }

    # ---------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output_dataset,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ---------------------------------------------------------
    # Console result
    # ---------------------------------------------------------

    print(f"Records processed       : {len(records)}")
    print(f"Records created         : {len(enriched_records)}")
    print(f"Records with ITEM       : {records_with_item}")
    print(f"Records with side       : {records_with_side}")
    print(f"Records with section    : {records_with_section}")
    print(f"Validation errors       : {len(errors)}")

    print("-" * 72)

    if errors:
        print("VALIDATION : FAILED")

        print("\nFirst validation errors:")

        for error in errors[:20]:
            print(f"  - {error}")

        raise ValueError(
            f"Metadata validation failed with "
            f"{len(errors)} errors."
        )

    print("VALIDATION : PASSED")
    print("=" * 72)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Enrich complex-table retrieval records "
            "with page-level ITEM and searchable metadata."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to retrieval_text.json"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to retrieval_enriched.json"
    )

    args = parser.parse_args()

    build_enriched_dataset(
        args.input,
        args.output
    )


if __name__ == "__main__":
    main()