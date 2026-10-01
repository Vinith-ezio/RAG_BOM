"""
Prompt Builder
--------------
Builds a grounded prompt for Qwen using only the retrieved RAG context.
"""

from typing import Any, Dict, List


SYSTEM_PROMPT = """
You answer questions using the provided document context.

The document context is the source of truth.
See the context parameter, sub parameter, value and additional values mostly the answers are in the context. 
If the requested value exists in the context, return that value.

Do not invent information.

If the requested value genuinely does not exist in the context,
say that it was not found.

Return the answer directly and concisely.
""".strip()


def _format_values(values: Any) -> str:
    """Format a record's values dictionary safely."""

    if values is None:
        return "N/A"

    if not isinstance(values, dict):
        return str(values)

    parts = []

    for key, value in values.items():
        if value is None:
            continue

        parts.append(f"{key}: {value}")

    return ", ".join(parts) if parts else "N/A"


def format_record(record: Dict[str, Any], index: int) -> str:
    """Convert one structured retrieval record into prompt text."""

    lines = [
        f"Record {index}:",
        f"Record ID: {record.get('record_id', 'N/A')}",
        f"Page: {record.get('page', 'N/A')}",
        f"Item: {record.get('item', 'N/A')}",
        f"Section: {record.get('section', 'N/A')}",
        f"Subsection: {record.get('subsection', 'N/A')}",
        f"Parameter: {record.get('parameter', 'N/A')}",
        f"Sub-parameter: {record.get('sub_parameter', 'N/A')}",
        f"Sub-sub-parameter: {record.get('sub_sub_parameter', 'N/A')}",
        f"Side: {record.get('side', 'N/A')}",
        f"Values: {_format_values(record.get('values'))}",
    ]

    return "\n".join(lines)


def build_context_text(records: List[Dict[str, Any]]) -> str:
    """Build structured context from retrieved records."""

    if not records:
        return "NO RETRIEVED RECORDS."

    blocks = []

    for index, record in enumerate(records, start=1):
        blocks.append(format_record(record, index))

    return "\n\n".join(blocks)


def build_prompt(
    query: str,
    context_result: Dict[str, Any],
) -> Dict[str, str]:

    retrieval = context_result.get(
        "retrieval",
        {},
    )

    records = context_result.get(
        "records",
        [],
    )

    context_text = context_result.get(
        "context_text",
        "",
    )

    if not context_text and records:
        context_text = build_context_text(records)

    if not context_text:
        context_text = "NO RETRIEVED RECORDS."

    user_prompt = f"""
Context:

{context_text}

Question:

{query}

Answer using only the context above.
""".strip()

    return {
        "system": SYSTEM_PROMPT,
        "user": user_prompt,
    }