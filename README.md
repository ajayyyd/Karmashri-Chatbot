# Karmashri: Kerala Government Orders Intelligence Platform

Karmashri is an authoritative, deterministic legal Retrieval-Augmented Generation (RAG) platform and side-by-side audit workbench engineered specifically for Kerala Government Orders (GOs).

Standard off-the-shelf RAG pipelines frequently fail on bureaucratic and legislative documents due to arbitrary character chunking, missing clause hierarchies, hallucinated financial allowances, and an inability to reconcile superseding amendments over time. Karmashri addresses these challenges through deterministic hierarchical parsing, zero-loss text coverage validation, a dual-database storage paradigm (PostgreSQL + Qdrant), and grounded executive synthesis with explicit temporal precedence tracking.

---

## Key Architectural Principles

1. **Zero-Loss Hierarchy Invariant (`assert_complete_text_coverage`)**
   - Documents are parsed into a deterministic tree (`Part > Section > Para > Sub-para`) rather than arbitrary token blocks.
   - Text extraction undergoes strict Unicode composition (`NFC`), newline standardization, and non-breaking space replacement (`SafeTextNormalizer`).
   - Every substantive character from the source PDF is accounted for across parsed nodes, ensuring zero dropped clauses.

2. **Dual-Engine Hybrid Retrieval**
   - **Dense Semantic Retrieval:** Powered by Qdrant vector search utilizing `BAAI/bge-small-en-v1.5` embeddings.
   - **Sparse Exact Identifier Search:** Powered by PostgreSQL Full-Text Search (`tsvector` / `tsquery`) to reliably pinpoint administrative file numbers, exact dates, and GO identifiers.
   - **Reciprocal Rank Fusion (RRF):** Merges dense and sparse ranks with temporal date weighting to ensure recent amendments receive appropriate retrieval priority.

3. **Grounded Synthesis & Lineage Tracking**
   - Synthesizes findings using local LLMs (`qwen2.5:3b` via Ollama) with temperature set to `0.0`.
   - Context is organized strictly by temporal precedence (newest to oldest), instructing the model to distinguish between active primary governing orders and historical originating amendments.
   - Constrained JSON grammar schemas ensure structured, deterministic metadata extraction during pipeline fallbacks.

4. **Split-Screen Audit Workbench**
   - Built with FastAPI and a modern split-pane UI.
   - Every synthesized ruling provides linked citations showing the exact GO number, date, department, and parent path.
   - Clicking any citation immediately loads the original, un-redacted source PDF in the adjacent viewport for direct verification.

---

## Directory Structure

```text
.
├── data/
│   ├── pdf/               # Source Kerala Government Order PDFs (.gitkeep)
│   ├── processed/         # Extracted clause trees (*_nodes.json) & clean text
│   └── dead_letter/       # Unparseable/corrupt files flagged for manual review
├── docs/                  # Technical specifications, architecture logs & plans
├── src/
│   ├── api.py             # FastAPI backend serving verification API & split UI
│   ├── ask_karmashri.py   # Interactive CLI query utility
│   ├── batch_ingest.py    # Extraction orchestrator (PDF -> OCR/Text -> Node tree)
│   ├── generation.py      # Grounded LLM synthesis & temporal precedence engine
│   ├── hybrid_ingestion.py# Safe text normalization & regex parsing state machine
│   ├── init_schema.sql    # PostgreSQL schema for relational nodes and audit log
│   ├── retrieval_qdrant.py# Hybrid retriever (Qdrant vectors + PostgreSQL FTS)
│   └── storage_quadrant.py# Embedding generator and dual-database sync worker
├── static/
│   └── index.html         # Interactive side-by-side audit workbench interface
├── .gitignore             # Strict exclusion of data binaries, models, & caches
├── requirements.txt       # Production Python dependencies (CUDA 12.4+ ready)
└── README.md
```

---

## System Requirements

- **Operating System:** Linux (Ubuntu 22.04+ / Debian 12 recommended) or macOS
- **Python:** 3.11 or higher
- **PostgreSQL:** 15+ (with Full-Text Search enabled)
- **Qdrant:** 1.8+
- **Ollama:** Installed locally
- **System Packages:**
  - `poppler-utils` (for `pdftotext` / `pdftoppm`)
  - `tesseract-ocr` (with English and Malayalam language data if processing regional scans)

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ajayyyd/Karmashri-Chatbot.git
cd Karmashri-Chatbot
```

### 2. Set Up Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

#### For GPU Systems (NVIDIA RTX 50-Series / Blackwell & CUDA 12.4+)
Install CUDA-enabled PyTorch first before installing other dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

#### For Standard CPU Systems
```bash
pip install -r requirements.txt
```

Verify your PyTorch environment:
```bash
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

### 3. Launch Infrastructure Services

#### A. Qdrant Vector Engine
Launch Qdrant via Docker with persistent disk storage:

```bash
docker run -d \
  -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  --name karmashri-qdrant \
  qdrant/qdrant
```

#### B. PostgreSQL Relational & FTS Database
Ensure PostgreSQL is active and initialize the statutory database schema:

```bash
# Create database
createdb -U postgres -h localhost karmashri

# Apply schema
psql -U postgres -h localhost -d karmashri -f src/init_schema.sql
```

#### C. Local LLM (Ollama)
Ensure Ollama is running and pull the default synthesis model:

```bash
ollama serve
ollama pull qwen2.5:3b
```

---

## Operating the Pipeline

### Step 1: Document Ingestion & Hierarchical Parsing
Place Kerala Government Order PDFs into `data/pdf/`:

```bash
# Example: copy orders into the ingestion directory
cp /path/to/orders/*.pdf data/pdf/
```

Execute the batch ingestion pipeline:
```bash
python3 src/batch_ingest.py
```
*What this does:*
- Inspects PDF text layers, automatically routing image-only scans through OCR.
- Cleans and standardizes text via `SafeTextNormalizer`.
- Deterministically parses GO numbers, dates, preambles, operative sections, and clause hierarchies.
- Validates text coverage against source input to guarantee zero dropped content.
- Emits structured `*_nodes.json` files to `data/processed/`.

### Step 2: Indexing to Qdrant & PostgreSQL
Sync parsed nodes into both database backends:

```bash
python3 src/storage_quadrant.py
```
*What this does:*
- Generates 384-dimensional dense embeddings using `BAAI/bge-small-en-v1.5`.
- Indexes dense vector payloads into Qdrant collection `kerala_go_nodes`.
- Writes full metadata, parent paths, and `tsvector` representations to PostgreSQL.

### Step 3: Launch the Audit Workbench
Start the FastAPI application:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Access the interface in your browser:
- **Audit Workbench:** [http://localhost:8000](http://localhost:8000)
- **Interactive OpenAPI Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Testing & Verification

You can query the system programmatically or via CLI:

```bash
python3 src/generation.py
```

### Representative Verification Queries

1. **Allowance Rates & Prerequisites:**
   > *"What are the rates of Children Education Allowance and condition for claiming Hostel Subsidy?"*
   - Verifies numerical ceiling extraction and condition checking against statutory tables.

2. **Legal Definitions & Supersessions:**
   > *"What is the definition of a Private Hospital under the All India Services Medical Facilities Scheme?"*
   - Verifies temporal precedence handling across initial orders, amendments, and comprehensive codifications.

3. **Reimbursement Ceilings:**
   > *"What are the ceiling rates for room rent and ICU charges under the AIS Medical Facilities Scheme?"*
   - Verifies explicit negative attribution (confirming when caps are not defined in the source order).

---

## License

This project is developed for administrative modernization and legal intelligence processing under the Government of Kerala framework. Internal use and open collaboration guidelines apply.
