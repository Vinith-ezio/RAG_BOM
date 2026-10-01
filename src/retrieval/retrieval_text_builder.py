import json
import argparse
from pathlib import Path


def format_value(key, value):
    """
    Convert a structured value into readable retrieval text.
    """

    if value is None:
        return None

    if isinstance(value, list):
        value = ", ".join(str(v) for v in value)

    if isinstance(value, dict):
        parts = []

        for nested_key, nested_value in value.items():
            if nested_value is None:
                continue

            parts.append(
                f"{nested_key.replace('_', ' ').title()}: {nested_value}"
            )

        return "; ".join(parts)

    return str(value)


def build_retrieval_text(record):
    """
    Convert one structured retrieval record into
    embedding-friendly text while preserving context.
    """

    parts = []

    # ---------------------------------------------------------
    # Hierarchical context
    # ---------------------------------------------------------

    if record.get("section"):
        parts.append(f"Section: {record['section']}.")

    if record.get("subsection"):
        parts.append(f"Subsection: {record['subsection']}.")

    # ---------------------------------------------------------
    # Parameter information
    # ---------------------------------------------------------

    if record.get("parameter"):
        parts.append(f"Parameter: {record['parameter']}.")

    if record.get("sub_parameter"):
        parts.append(
            f"Sub-parameter: {record['sub_parameter']}."
        )

    if record.get("sub_sub_parameter"):
        parts.append(
            f"Sub-sub-parameter: {record['sub_sub_parameter']}."
        )

    # ---------------------------------------------------------
    # Side information
    # ---------------------------------------------------------

    if record.get("side"):
        parts.append(
            f"Side: {record['side'].capitalize()}."
        )

    # ---------------------------------------------------------
    # Structured values
    # ---------------------------------------------------------

    values = record.get("values", {})

    if isinstance(values, dict):

        for key, value in values.items():

            formatted = format_value(key, value)

            if formatted is None:
                continue

            # Ignore completely empty strings
            if isinstance(formatted, str) and not formatted.strip():
                continue

            label = key.replace("_", " ").title()

            parts.append(
                f"{label}: {formatted}."
            )

    # ---------------------------------------------------------
    # Final text
    # ---------------------------------------------------------

    return " ".join(parts)


def build_retrieval_dataset(input_path, output_path):

    input_path = Path(input_path)
    output_path = Path(output_path)

    print("=" * 72)
    print("RETRIEVAL TEXT BUILDER")
    print("=" * 72)

    print(f"Input  : {input_path}")
    print(f"Output : {output_path}")
    print("-" * 72)

    # ---------------------------------------------------------
    # Load input
    # ---------------------------------------------------------

    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    records = dataset.get("records", [])

    if not records:
        raise ValueError(
            "No retrieval records found in input dataset."
        )

    # ---------------------------------------------------------
    # Build retrieval text
    # ---------------------------------------------------------

    output_records = []

    empty_text_count = 0

    total_characters = 0

    for record in records:

        new_record = dict(record)

        retrieval_text = build_retrieval_text(record)

        new_record["retrieval_text"] = retrieval_text

        # Useful for later embedding validation
        new_record["retrieval_text_length"] = len(
            retrieval_text
        )

        if not retrieval_text.strip():
            empty_text_count += 1

        total_characters += len(retrieval_text)

        output_records.append(new_record)

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    statistics = {
        "input_records": len(records),
        "output_records": len(output_records),
        "empty_retrieval_text": empty_text_count,
        "total_characters": total_characters,
        "average_characters_per_record": (
            round(total_characters / len(output_records), 2)
            if output_records
            else 0
        )
    }

    # ---------------------------------------------------------
    # Output dataset
    # ---------------------------------------------------------

    output_dataset = {
        "schema": {
            "name": "complex-table-retrieval-text",
            "version": "1.0"
        },

        "source": {
            "name": dataset.get("source_dataset", {}).get("name"),
            "version": dataset.get("source_dataset", {}).get("version"),
            "document": dataset.get("source_dataset", {}).get("document")
        },

        "statistics": statistics,

        "records": output_records
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
    # Console validation
    # ---------------------------------------------------------

    print(f"Records processed : {len(records)}")
    print(f"Records created   : {len(output_records)}")
    print(f"Empty text        : {empty_text_count}")
    print(
        f"Average text size : "
        f"{statistics['average_characters_per_record']} chars"
    )

    print("-" * 72)

    if empty_text_count == 0:
        print("VALIDATION : PASSED")
    else:
        print(
            f"VALIDATION : WARNING "
            f"({empty_text_count} empty records)"
        )

    print("=" * 72)


def main():

    parser = argparse.ArgumentParser(
        description="Build retrieval text from structured RAG records."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to retrieval_records.json"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to retrieval_text.json"
    )

    args = parser.parse_args()

    build_retrieval_dataset(
        args.input,
        args.output
    )


if __name__ == "__main__":
    main()