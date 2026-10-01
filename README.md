# Structured JSON → RAG Pipeline

> **Project:** PDF Ingestion / Structured Retrieval / RAG  
> **Current focus:** Learning and building a document-grounded RAG system starting from structured JSON  
> **LLM:** Qwen 2.5 7B via Ollama  
> **Embeddings:** Not used in the primary retrieval pipeline  
> **Primary retrieval:** Structured Retrieval + BM25 + Hybrid V3

---

## 1. Project Overview

This project converts a complex, nested engineering-table JSON dataset into a **retrieval-ready and RAG-ready knowledge pipeline**.

The system is designed around one important principle:

> **Do not hardcode document-specific knowledge into the retrieval code. Discover the document structure from the input data and use that discovered structure to understand and retrieve user queries.**

The pipeline is intentionally divided into independent phases so that each responsibility can be tested and debugged separately.

### High-Level Flow

```text
Structured JSON
      │
      ▼
┌───────────────────────┐
│ 1. Dataset Loader     │
│ Validation            │
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│ 2. Record Builder     │
│ Flat retrieval units  │
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│ 3. Retrieval Text     │
│ Searchable text       │
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│ 4. Metadata Enricher  │
│ Normalized metadata   │
└──────────┬────────────┘
           ▼
┌──────────────────────────────────────┐
│ 5. Schema Catalog                    │
│ 6. Entity Index                      │
└──────────┬───────────────────────────┘
           ▼
┌───────────────────────┐
│ 7. Query Understanding│
│ Natural language →    │
│ structured intent     │
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│ 8. Intent Retrieval   │
│ Adapter               │
└──────────┬────────────┘
           ▼
┌──────────────────────────────────────┐
│ 9. Retrieval                         │
│                                      │
│ Structured Retrieval + BM25          │
│              ↓                       │
│          Hybrid V3                   │
└──────────┬───────────────────────────┘
           ▼
┌───────────────────────┐
│ 10. Context Builder   │
│ Grounded context      │
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│ 11. Prompt Builder    │
│ Grounded prompt       │
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│ 12. Qwen / Ollama     │
│ Answer generation     │
└──────────┬────────────┘
           ▼
      Final Answer
```

---

# 2. What Problem Does This Solve?

The source data is not a simple flat table.

A single engineering specification can contain:

- sections
- subsections
- parameters
- sub-parameters
- inlet/outlet values
- minimum / normal / maximum values
- units
- material codes
- equipment/item identifiers
- nested relationships

For example:

```json
{
  "parameter": "Line Size",
  "sub_parameter": "Piping Class",
  "inlet": {
    "value": "6\"",
    "class": "A2A"
  },
  "outlet": {
    "value": "6\"",
    "class": "A2A"
  }
}
```

A user may ask:

> What is the inlet line size for material code TC9765132581?

The system must understand:

```text
Identifier       → TC9765132581
Identifier type  → Material Code
Parameter        → Line Size
Side             → Inlet
```

Then retrieve the correct record and pass only the relevant evidence to the answer-generation stage.

---

# 3. Dataset Used for Development

The current controlled dataset is:

```text
data/input/desuperheater_complex_table_rag_dataset.json
```

### Dataset characteristics

| Property | Value |
|---|---:|
| Pages | 7 |
| Tables | 7 |
| Source rows | 406 |
| Retrieval records | 434 |
| Document | Instrument Specification - Desuperheater |
| Dataset schema | `complex-table-rag-dataset` |
| Schema version | `1.0` |
| Source type | Uploaded screenshot |
| Extraction method | Manual visual transcription |
| LLM used for source transcription | No |

The source JSON declares:

```json
"dataset": {
  "page_count": 7,
  "total_rows": 406
}
```

The **434 retrieval records** are produced later by the Record Builder. They are not the original row count.

---

# 4. Project Structure

```text
E:\PDF Ingestion\
│
├── .venv\
│
├── data\
│   ├── input\
│   │   └── desuperheater_complex_table_rag_dataset.json
│   │
│   └── output\
│       ├── retrieval_records.json
│       ├── retrieval_text.json
│       ├── retrieval_enriched.json
│       ├── schema_catalog.json
│       └── entity_index.json
│
├── src\
│   │
│   ├── retrieval\
│   │   ├── dataset_loader.py
│   │   ├── structured_record_builder.py
│   │   ├── retrieval_text_builder.py
│   │   ├── metadata_enricher.py
│   │   ├── structured_retriever.py
│   │   ├── bm25_retriever.py
│   │   ├── hybrid_retriever_v2.py
│   │   ├── hybrid_retriever_v3.py
│   │   └── retrieval_evaluator.py
│   │
│   ├── query\
│   │   ├── schema_catalog.py
│   │   ├── entity_index.py
│   │   ├── query_understanding.py
│   │   └── intent_retrieval_adapter.py
│   │
│   ├── context\
│   │   ├── context_builder.py
│   │   └── manual_retrieval_tester.py
│   │
│   └── rag\
│       ├── prompt_builder.py
│       ├── qwen_client.py
│       └── answer_generator.py
│
└── tests\
```

> Filenames can evolve as the project develops. The responsibilities described below are the important architectural boundaries.

---

# 5. Phase-by-Phase Architecture

## Phase 1 — Dataset Loader & Validation

### File

```text
src/retrieval/dataset_loader.py
```

### Responsibility

Load the original JSON and verify that it follows the expected data contract.

### Input

```text
data/input/desuperheater_complex_table_rag_dataset.json
```

### Output

The loader returns the validated JSON as a Python dictionary.

It does **not** create a new JSON output file.

### Validation rules

The loader checks:

1. Required root keys exist.
2. `schema` is an object.
3. Schema name is `complex-table-rag-dataset`.
4. `pages` is a list.
5. At least one page exists.
6. Required page fields exist.
7. `rows` is a list.
8. `row_count` matches the actual number of rows.
9. Every row is an object.
10. Every row contains `no` and `parameter`.
11. Declared `page_count` matches actual pages when present.
12. Declared `total_rows` matches actual rows when present.

### Run

```powershell
python src\retrieval\dataset_loader.py --input "E:\PDF Ingestion\data\input\desuperheater_complex_table_rag_dataset.json"
```

### Expected result

```text
RAG DATASET LOADER
Pages   : 7
Tables  : 7
Rows    : 406
VALIDATION : PASSED
```

---

# 6. Phase 2 — Structured Record Builder

### File

```text
src/retrieval/structured_record_builder.py
```

### Responsibility

Convert nested document rows into smaller, retrieval-friendly records.

The original JSON is optimized for **document representation**.

Retrieval requires smaller units that can be:

- filtered
- matched
- ranked
- traced back to their source

### Example transformation

Source:

```text
Line Size
 ├── Inlet
 │    ├── value = 6"
 │    └── class = A2A
 └── Outlet
      ├── value = 6"
      └── class = A2A
```

Becomes:

```text
p01_r8_inlet
p01_r8_outlet
```

### Important result

```text
406 source rows
        ↓
434 retrieval records
```

The increase happens because some rows contain multiple logical retrieval units, such as inlet and outlet values.

### Output

```text
data/output/retrieval_records.json
```

### Run

```powershell
python src\retrieval\structured_record_builder.py
```

---

# 7. Phase 3 — Retrieval Text Builder

### File

```text
src/retrieval/retrieval_text_builder.py
```

### Responsibility

Create a text representation of each structured record.

This representation is used by lexical retrieval systems such as BM25.

### Example

Structured data:

```text
Parameter: Line Size
Sub-parameter: Piping Class
Side: inlet
Value: 6"
Class: A2A
```

Retrieval text becomes a searchable representation containing concepts such as:

```text
Line Size
Piping Class
inlet
6"
A2A
```

### Why keep retrieval text separate?

The structured record is optimized for:

```text
exact filtering
provenance
field matching
```

The retrieval text is optimized for:

```text
lexical search
BM25
keyword matching
```

### Output

```text
data/output/retrieval_text.json
```

### Run

```powershell
python src\retrieval\retrieval_text_builder.py
```

---

# 8. Phase 4 — Metadata Enrichment

### File

```text
src/retrieval/metadata_enricher.py
```

### Responsibility

Add retrieval-oriented metadata to the records.

Examples include:

- ITEM
- normalized parameter
- normalized sub-parameter
- normalized values
- search metadata
- provenance information

### Purpose

This phase prepares records for downstream:

```text
Schema Discovery
Entity Resolution
Query Understanding
Retrieval
```

It does **not** answer user questions.

### Output

```text
data/output/retrieval_enriched.json
```

### Run

```powershell
python src\retrieval\metadata_enricher.py `
  --input "E:\PDF Ingestion\data\output\retrieval_text.json" `
  --output "E:\PDF Ingestion\data\output\retrieval_enriched.json"
```

---

# 9. Phase 5 — Schema Catalog

### File

```text
src/query/schema_catalog.py
```

### Responsibility

Discover the structure of the dataset automatically.

The system learns from the JSON instead of embedding document-specific field names in retrieval logic.

### Discovered information

Examples:

```text
Sections
Parameters
Sub-parameters
Sides
Items
Value fields
Parent/child relationships
```

Current dataset discovery:

```text
Records analyzed : 434
Unique sections   : 2
Unique parameters : 42
Unique subparams  : 26
Unique sides      : 2
Unique items      : 7
Value fields      : 16
```

### Example hierarchy

```text
Line Size
   └── Piping Class

Body Material
   └── Stem / Plug Material

Manufacturer
   └── Model No.
```

### Why this matters

Instead of hardcoding:

```python
if query contains "piping class":
    parameter = "Line Size"
```

the system can use the generated hierarchy:

```text
Piping Class → parent → Line Size
```

This makes the architecture more reusable across documents with different structures.

### Output

```text
data/output/schema_catalog.json
```

### Run

```powershell
python src\query\schema_catalog.py
```

---

# 10. Phase 6 — Entity Index

### File

```text
src/query/entity_index.py
```

### Responsibility

Build fast lookup structures for document entities and values.

Examples:

```text
Material Code
ITEM
Line Material
Fluid Connection
Press.
Pneumatic Supply
```

### Current dataset

```text
Records analyzed       : 434
Unique items            : 7
Unique document values  : 112
Document value entries  : 158
```

### Example lookup concept

```text
TC9765132581
      ↓
Material Code
      ↓
Page 2
      ↓
ITEM = DSH 6"300RF-INTEG TCV 1"600RF-HART
      ↓
Relevant record IDs
```

### Why this exists

Without an entity index, the system would repeatedly scan all records to resolve identifiers.

With an index:

```text
Query identifier
      ↓
Entity Index
      ↓
Relevant record scope
```

### Output

```text
data/output/entity_index.json
```

### Run

```powershell
python src\query\entity_index.py
```

---

# 11. Phase 7 — Query Understanding

### File

```text
src/query/query_understanding.py
```

### Responsibility

Convert natural language into a structured intent.

Example query:

> What is the inlet line size for material code TC9765132581?

Becomes conceptually:

```text
Identifier      : TC9765132581
Identifier type : Material Code
Parameter       : Line Size
Side            : Inlet
```

### Query Understanding Flow

```text
Natural-language query
          ↓
Identifier detection
          ↓
Schema term detection
          ↓
Hierarchy resolution
          ↓
Side detection
          ↓
Intent construction
```

### Important design rule

The query-understanding layer should not contain document-specific values such as:

```text
TC9765132581
DSH 6"300RF-INTEG...
```

Those values come from the generated entity index.

The logic is generic; the data is discovered.

---

# 12. Phase 8 — Intent Retrieval Adapter

### File

```text
src/query/intent_retrieval_adapter.py
```

### Responsibility

Act as the bridge between:

```text
Query Understanding
        ↓
Actual Retrieval
```

It decides how the structured intent should be retrieved.

### Retrieval modes

The adapter supports concepts such as:

```text
Entity-scoped lookup
Identifier → ITEM lookup
Parameter/schema lookup
Section lookup
Multi-intent retrieval
Abstention
```

### Example

Query:

```text
What is the actuator type for material code TC9765132581?
```

Flow:

```text
Query
 ↓
Identifier = TC9765132581
 ↓
Entity Index
 ↓
Page 2 / relevant entity scope
 ↓
Parameter = Actuator Type
 ↓
Relevant record
```

This prevents the system from searching the entire dataset unnecessarily.

---

# 13. Phase 9 — Structured Retrieval

### File

```text
src/retrieval/structured_retriever.py
```

### Responsibility

Perform exact structured matching.

Think of it as a database-style filter.

For:

```text
What is the inlet line size for material code TC9765132581?
```

the structured retrieval target is conceptually:

```text
ITEM / entity scope
AND parameter = Line Size
AND side = inlet
```

Expected record:

```text
p02_r8_inlet
```

### Strength

Structured retrieval is strong when the query can be translated into exact fields.

It provides:

- deterministic matching
- precise filtering
- strong provenance
- predictable behavior

---

# 14. Phase 10 — BM25 Retrieval

### File

```text
src/retrieval/bm25_retriever.py
```

### Responsibility

Perform lexical text retrieval over the generated retrieval text.

BM25 evaluates how strongly query terms match candidate documents.

Example query:

```text
inlet line size material code TC9765132581
```

BM25 searches the text representations for matching terms.

### Conceptual flow

```text
Query
 ↓
Tokenization / lexical matching
 ↓
BM25 scoring
 ↓
Ranked candidates
```

### Important distinction

Structured Retrieval asks:

> Does this record match the requested structured conditions?

BM25 asks:

> Which records contain terms that are lexically relevant to this query?

They solve different retrieval problems.

---

# 15. Phase 11 — Hybrid Retrieval

## Hybrid V2

### File

```text
src/retrieval/hybrid_retriever_v2.py
```

Hybrid V2 combines:

```text
Structured Retrieval
        +
BM25
        ↓
Candidate Union
        ↓
Compatibility Filtering
        ↓
RRF Ranking
```

### RRF

Reciprocal Rank Fusion combines rankings from different retrieval strategies.

The goal is not to replace either retriever, but to combine their evidence.

---

## Hybrid V3

### File

```text
src/retrieval/hybrid_retriever_v3.py
```

Class:

```python
NativeMultiIntentHybridRetriever
```

V3 adds native multi-intent handling.

Example:

```text
What is the inlet line size and outlet piping class
for material code TC9765132581?
```

The query can be decomposed into multiple retrieval intents and processed separately.

### V3 architecture

```text
Natural-language query
          ↓
Multi-intent split
          ↓
Intent 1 ──→ Hybrid V2
Intent 2 ──→ Hybrid V2
Intent 3 ──→ Hybrid V2
          ↓
Group + Deduplicate
          ↓
Final retrieval set
```

### Current design

The primary V3 pipeline intentionally uses:

```text
Structured retrieval
BM25
RRF
Multi-intent handling
```

and does not require:

```text
Embeddings
LLM-based retrieval
Heuristic score boosting
```

---

# 16. Phase 12 — Context Builder

### File

```text
src/context/context_builder.py
```

### Responsibility

Convert retrieval results into clean, deterministic context for answer generation.

It preserves:

- record ID
- page
- table
- row
- ITEM
- section
- subsection
- parameter
- sub-parameter
- side
- values
- provenance

### Flow

```text
Retrieval result
      ↓
Validate records
      ↓
Remove duplicates
      ↓
Preserve metadata
      ↓
Build context blocks
      ↓
Build context text
```

### Important rule

The Context Builder does not:

```text
retrieve
infer
guess
generate an answer
```

It only prepares the evidence.

---

# 17. Phase 13 — Prompt Builder

### File

```text
src/rag/prompt_builder.py
```

### Responsibility

Convert the retrieved context into a grounded prompt for Qwen.

The prompt instructs Qwen to:

- use only retrieved context
- avoid outside knowledge
- avoid inventing values
- preserve technical values
- preserve units
- distinguish multiple records
- report missing information when it is not in context

### Flow

```text
User Query
    +
Retrieved Context
    ↓
Grounding Instructions
    ↓
Qwen Prompt
```

This is the boundary between deterministic retrieval and generative answer generation.

---

# 18. Phase 14 — Qwen Client

### File

```text
src/rag/qwen_client.py
```

### Responsibility

Communicate with the local Ollama server.

Default model:

```text
qwen2.5:7b
```

Default Ollama endpoint:

```text
http://localhost:11434
```

### Architecture

```text
Python
  ↓
QwenClient
  ↓
Ollama API
  ↓
Qwen 2.5 7B
```

The client handles:

- health check
- model request
- prompt submission
- response retrieval

---

# 19. Phase 15 — Answer Generator

### File

```text
src/rag/answer_generator.py
```

### Responsibility

Connect the complete RAG flow.

```text
User Query
     ↓
Intent Retrieval Adapter
     ↓
Retrieved Records
     ↓
Context Builder
     ↓
Prompt Builder
     ↓
Qwen Client
     ↓
Grounded Answer
```

### Important behavior

If retrieval abstains or produces no records:

```text
Do not call the LLM.
```

The system can return a controlled response indicating that the requested information was not found in the retrieved document context.

This keeps retrieval failure separate from LLM generation.

---

# 20. RAG Folder

The `src/rag` directory contains the final generation layer.

```text
src/rag/
│
├── prompt_builder.py
│     └── Builds grounded prompts
│
├── qwen_client.py
│     └── Talks to Ollama / Qwen
│
└── answer_generator.py
      └── Runs the complete RAG answer flow
```

### RAG responsibility boundary

```text
Retrieval
   ↓
Context
   ↓
Prompt
   ↓
LLM
   ↓
Answer
```

The RAG layer should not become responsible for document parsing or retrieval logic.

---

# 21. Manual Retrieval Testing

The context/manual tester is useful for checking the pipeline **before involving Qwen**.

### File

```text
src/context/manual_retrieval_tester.py
```

### Run

```powershell
python src\context\manual_retrieval_tester.py
```

### Example queries

```text
What is the actuator type for material code TC9765132581?

What is the inlet line size for material code TC9765132581?

What is the outlet piping class for material code TC9765132581?

What is the body material of DSH 6"300RF-INTEG TCV 1"600RF-HART?

What are the water conditions?

What is the turbine shaft diameter?
```

### What this tester verifies

```text
Query Understanding
        ↓
Intent Retrieval
        ↓
Retrieved Records
        ↓
Context
        ↓
Provenance
```

No Qwen inference is required.

---

# 22. Qwen / RAG Answer Testing

Before starting the answer generator, make sure Ollama is running and the Qwen model is available.

### Start Ollama server

If using the custom model directory:

```powershell
$env:OLLAMA_MODELS="E:\OllamaModels"
ollama serve
```

Open another PowerShell terminal:

```powershell
ollama list
```

Verify:

```text
qwen2.5:7b
```

### Run the complete RAG application

From:

```text
E:\PDF Ingestion
```

run:

```powershell
python src\rag\answer_generator.py
```

The application provides an interactive prompt:

```text
Enter your question (or 'exit'):
```

Example:

```text
What is the actuator type for material code TC9765132581?
```

---

# 23. Complete Pipeline Execution

Run the pipeline in this order when rebuilding the artifacts from the original JSON.

## Step 1 — Validate input

```powershell
python src\retrieval\dataset_loader.py `
  --input "E:\PDF Ingestion\data\input\desuperheater_complex_table_rag_dataset.json"
```

## Step 2 — Build retrieval records

```powershell
python src\retrieval\structured_record_builder.py
```

## Step 3 — Build retrieval text

```powershell
python src\retrieval\retrieval_text_builder.py
```

## Step 4 — Enrich metadata

```powershell
python src\retrieval\metadata_enricher.py `
  --input "E:\PDF Ingestion\data\output\retrieval_text.json" `
  --output "E:\PDF Ingestion\data\output\retrieval_enriched.json"
```

## Step 5 — Build schema catalog

```powershell
python src\query\schema_catalog.py
```

## Step 6 — Build entity index

```powershell
python src\query\entity_index.py
```

## Step 7 — Test retrieval/context

```powershell
python src\context\manual_retrieval_tester.py
```

## Step 8 — Start Ollama

```powershell
$env:OLLAMA_MODELS="E:\OllamaModels"
ollama serve
```

## Step 9 — Run Qwen RAG

In another terminal:

```powershell
python src\rag\answer_generator.py
```

---

# 24. Output Artifact Responsibilities

| Artifact | Created by | Purpose |
|---|---|---|
| `retrieval_records.json` | Record Builder | Flat retrieval units |
| `retrieval_text.json` | Retrieval Text Builder | Lexical search text |
| `retrieval_enriched.json` | Metadata Enricher | Retrieval metadata |
| `schema_catalog.json` | Schema Catalog | Discovered document schema |
| `entity_index.json` | Entity Index | Fast entity/value lookup |

### Simple mental model

```text
retrieval_records.json
    = What is the structured record?

retrieval_text.json
    = What text should lexical retrieval search?

retrieval_enriched.json
    = What metadata helps retrieval and query understanding?

schema_catalog.json
    = What structure does this document contain?

entity_index.json
    = Where are the important document entities?
```

---

# 25. Example End-to-End Query

## User Query

```text
What is the inlet line size for material code TC9765132581?
```

## Step 1 — Query Understanding

```text
Identifier      = TC9765132581
Identifier Type = Material Code
Parameter       = Line Size
Side            = Inlet
```

## Step 2 — Entity Resolution

```text
TC9765132581
      ↓
Relevant document entity
      ↓
Page 2
```

## Step 3 — Retrieval

```text
Structured Retrieval
        +
BM25
        ↓
Hybrid V3
        ↓
p02_r8_inlet
```

## Step 4 — Context

```text
Record ID : p02_r8_inlet
Page      : 2
Parameter : Line Size
Side      : inlet
Value     : 6"
```

## Step 5 — Prompt

The context is passed to Qwen with grounding instructions.

## Step 6 — Answer

Qwen generates an answer based only on the retrieved evidence.

---

# 26. Negative Query / Abstention

A production RAG system must handle questions that are not supported by the document.

Example:

```text
What is the turbine shaft diameter?
```

If the query cannot be resolved to a supported schema/entity scope, the retrieval layer should abstain rather than return unrelated records.

### Desired flow

```text
Unsupported query
       ↓
Query Understanding
       ↓
No valid retrieval intent
       ↓
ABSTAIN
       ↓
No context
       ↓
No unnecessary LLM call
```

This is important because retrieving a technically unrelated record is worse than returning no result.

---

# 27. Design Principles

## 27.1 No document-specific hardcoding

Avoid code such as:

```python
if material_code == "TC9765132581":
    return page_2
```

Instead:

```text
Input JSON
   ↓
Entity Index
   ↓
Generic lookup
```

---

## 27.2 Data-driven schema

Do not hardcode the complete parameter list.

The Schema Catalog discovers:

```text
parameters
sub-parameters
sections
sides
hierarchies
```

from the input data.

---

## 27.3 Deterministic retrieval before generation

The system should first establish:

```text
What records are relevant?
```

Only then should Qwen answer:

```text
What should I say about those records?
```

---

## 27.4 Preserve provenance

Every answer should be traceable back to:

```text
Record ID
Page
Table
Row
ITEM
Parameter
Values
```

---

## 27.5 Retrieval and generation are separate

```text
Retrieval
    ≠
LLM generation
```

This separation makes debugging much easier.

If the answer is wrong, determine first whether:

```text
Retrieval was wrong
```

or:

```text
Qwen interpreted correct context incorrectly
```

---

# 28. Debugging Strategy

When an answer is incorrect, debug from left to right.

```text
1. Source JSON
      ↓
2. Retrieval records
      ↓
3. Metadata
      ↓
4. Schema Catalog
      ↓
5. Entity Index
      ↓
6. Query Understanding
      ↓
7. Retrieval Adapter
      ↓
8. Structured/BM25/Hybrid
      ↓
9. Context
      ↓
10. Prompt
      ↓
11. Qwen
```

### Example

If Qwen gives the wrong answer:

Do not immediately modify the prompt.

First verify:

```text
Was the correct record retrieved?
```

Then:

```text
Was the correct record included in context?
```

Then:

```text
Was the context correctly inserted into the prompt?
```

Only after those checks should the LLM layer be investigated.

---

# 29. Evaluation

The project uses evaluation datasets and retrieval tests to measure system behavior.

Important retrieval metrics include:

```text
Precision
Recall
Hit@1
Hit@3
Hit@5
MRR
Exact Match
Negative Abstention
Multi-intent Coverage
```

### Why multiple metrics?

A retrieval system can have high recall but poor first-rank precision.

Therefore, one metric alone does not describe the complete retrieval behavior.

### Current development status

The current Hybrid V3 pipeline has been evaluated against the controlled query set and frozen as a development baseline.

The embedding experiment was also tested separately and was not selected as the primary retrieval mechanism for this controlled dataset.

---

# 30. Dependencies

Core Python components include:

```text
Python 3.11
```

Retrieval-related packages include:

```text
rank-bm25
```

LLM serving:

```text
Ollama
Qwen 2.5 7B
```

The project can run its structured retrieval/context stages without an LLM.

---

# 31. What Does Not Require Qwen?

The following stages are deterministic:

```text
Dataset Loading
Record Building
Retrieval Text Building
Metadata Enrichment
Schema Catalog
Entity Index
Query Understanding
Structured Retrieval
BM25
Hybrid Retrieval
Context Building
```

Qwen is required only for:

```text
Final natural-language answer generation
```

This is useful for testing because the retrieval system can be validated independently.

---

# 32. Final Architecture

```text
                         ┌─────────────────────┐
                         │   Structured JSON    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Dataset Validation  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Structured Records  │
                         └──────────┬──────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
          ┌──────────────────┐          ┌──────────────────┐
          │ Retrieval Text   │          │ Metadata         │
          └────────┬─────────┘          └────────┬─────────┘
                   │                             │
                   └──────────────┬──────────────┘
                                  ▼
                    ┌──────────────────────────┐
                    │ Schema Catalog           │
                    │ Entity Index              │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Query Understanding      │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Intent Retrieval Adapter │
                    └────────────┬─────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  ▼                             ▼
       ┌────────────────────┐       ┌────────────────────┐
       │ Structured         │       │ BM25               │
       │ Retrieval          │       │ Lexical Retrieval  │
       └──────────┬─────────┘       └──────────┬─────────┘
                  │                            │
                  └─────────────┬──────────────┘
                                ▼
                     ┌────────────────────────┐
                     │ Hybrid V3              │
                     │ RRF + Multi-intent     │
                     └───────────┬────────────┘
                                 │
                                 ▼
                     ┌────────────────────────┐
                     │ Context Builder         │
                     └───────────┬────────────┘
                                 │
                                 ▼
                     ┌────────────────────────┐
                     │ Prompt Builder          │
                     └───────────┬────────────┘
                                 │
                                 ▼
                     ┌────────────────────────┐
                     │ Qwen 2.5 7B / Ollama   │
                     └───────────┬────────────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │ Final Answer  │
                         └───────────────┘
```

---

# 33. One-Line Explanation for Each Phase

| Phase | One-line explanation |
|---|---|
| 1. Dataset Loader | Validates the input JSON against predefined data-contract rules. |
| 2. Record Builder | Converts nested document rows into smaller retrieval-friendly records. |
| 3. Retrieval Text | Creates searchable text representations for lexical retrieval. |
| 4. Metadata Enricher | Adds normalized and retrieval-oriented metadata to records. |
| 5. Schema Catalog | Automatically discovers the document's schema and hierarchy. |
| 6. Entity Index | Builds fast lookup structures for document entities and values. |
| 7. Query Understanding | Converts natural language into structured retrieval intent. |
| 8. Retrieval Adapter | Maps the structured intent to the appropriate retrieval strategy. |
| 9. Structured Retrieval | Performs exact field-based retrieval. |
| 10. BM25 | Performs lexical relevance retrieval over searchable record text. |
| 11. Hybrid V3 | Combines structured and lexical retrieval and supports multi-intent queries. |
| 12. Context Builder | Converts retrieved records into clean, traceable evidence. |
| 13. Prompt Builder | Packages the evidence into a grounded LLM prompt. |
| 14. Qwen Client | Sends the grounded prompt to Qwen through Ollama. |
| 15. Answer Generator | Orchestrates retrieval, context construction, prompting, and answer generation. |

---

# 34. Current Learning Goal

The project is intentionally being learned **phase by phase**.

The recommended learning order is:

```text
JSON Structure
      ↓
Dataset Loader
      ↓
Record Builder
      ↓
Retrieval Text
      ↓
Metadata
      ↓
Schema Discovery
      ↓
Entity Resolution
      ↓
Query Understanding
      ↓
Retrieval
      ↓
Context
      ↓
Prompt
      ↓
Qwen
      ↓
Complete RAG
```

The key objective is to understand **why every layer exists**, not just how to execute the scripts.

---

# 35. Quick Start

From the project root:

```powershell
cd "E:\PDF Ingestion"
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Validate the dataset:

```powershell
python src\retrieval\dataset_loader.py `
  --input "E:\PDF Ingestion\data\input\desuperheater_complex_table_rag_dataset.json"
```

Build the retrieval artifacts:

```powershell
python src\retrieval\structured_record_builder.py

python src\retrieval\retrieval_text_builder.py

python src\retrieval\metadata_enricher.py `
  --input "E:\PDF Ingestion\data\output\retrieval_text.json" `
  --output "E:\PDF Ingestion\data\output\retrieval_enriched.json"

python src\query\schema_catalog.py

python src\query\entity_index.py
```

Test retrieval without Qwen:

```powershell
python src\context\manual_retrieval_tester.py
```

Start Qwen:

```powershell
$env:OLLAMA_MODELS="E:\OllamaModels"
ollama serve
```

Run complete RAG:

```powershell
python src\rag\answer_generator.py
```

---

## End State

The completed system is:

```text
Structured Engineering JSON
          ↓
Data Validation
          ↓
Retrieval Representation
          ↓
Data-Driven Schema + Entity Discovery
          ↓
Natural Language Query Understanding
          ↓
Hybrid Retrieval
          ↓
Grounded Context
          ↓
Qwen
          ↓
Document-Grounded Answer
```

The architecture is intentionally modular so that **document ingestion, retrieval, context construction, and LLM generation can be tested independently**.
