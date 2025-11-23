# Enhanced Citation Network (ECN) - Project Documentation

## 1. Overview

The **Enhanced Citation Network (ECN)** is a system designed to transform scientific citation graphs from simple bibliographic maps into interpretable knowledge systems. By enriching paper nodes with structured semantic data (Problem, Method, Results) and characterizing edges with explicit reasoning (Support, Contrast, Extend), the ECN enables "Structure-Aware" reasoning for LLMs.

This repository contains a functional prototype of the ECN pipeline, covering data extraction, relation discovery, graph storage, and reasoning.

## 2. System Architecture

The system is built in four logical phases:

1.  **Node Extraction (Phase 1):** Transforms raw LaTeX/PDFs into structured JSON objects using LLM prompting.
2.  **Relation Discovery (Phase 2):** Identifies semantic relationships between papers using three views:
    *   **View T (Textual):** Embedding similarity of specific sections.
    *   **View L (LLM):** Pairwise debate/reasoning.
    *   **View G (Graph):** Link prediction using Graph Neural Networks (GNNs).
3.  **Storage Layer (Phase 3):** A Neo4j Graph Database storing papers as nodes and semantic relations as edges, indexed with vector embeddings.
4.  **Reasoning Engine (Phase 4):** A GraphRAG (Retrieval-Augmented Generation) system that retrieves relevant subgraphs to answer complex scientific queries.

## 3. Repository Structure

```text
.
├── main.py                     # Master orchestration script (Runs the full pipeline)
├── requirements.txt            # Python dependencies
├── prompts.md                  # System prompts for LLM extraction and reasoning
├── schema_design.md            # Neo4j graph schema definition
│
├── extract_node.py             # Phase 1: Parses LaTeX and generates extraction prompts
├── extract_relation.py         # Phase 2 (View L): Generates pairwise reasoning prompts
├── compute_similarity.py       # Phase 2 (View T): Computes embedding similarity (Problem/Method)
├── link_prediction.py          # Phase 2 (View G): GNN model for predicting missing citations
│
├── graph_loader.py             # Phase 3: Loads nodes and edges into Neo4j
├── graph_rag.py                # Phase 4: Performs vector search and graph expansion for QA
│
├── arxiv_scrape_latex.py       # Utility: Downloads LaTeX source from arXiv
├── arxiv_to_text.py            # Utility: Converts PDF to text
└── arxiv_source/               # Data directory containing raw LaTeX files
```

## 4. Component Details

### Phase 1: Node Extraction
*   **File:** `extract_node.py`
*   **Function:** Recursively resolves LaTeX `\input` commands to assemble full text. It heuristically extracts "Introduction" (Problem) and "Methodology" sections.
*   **Prompts:** `prompts.md` contains the specific instructions for an LLM to convert these raw text sections into structured JSON (e.g., extracting "Core Contribution" or "Key Findings").

### Phase 2: Relation Discovery
*   **View T (`compute_similarity.py`):** Uses `sentence-transformers` (default: `all-MiniLM-L6-v2`) to encode "Problem" and "Method" sections separately. It applies heuristics to classify relations (e.g., High Problem Sim + Low Method Sim = "Alternative Approach").
*   **View L (`extract_relation.py`):** Prepares prompts for an LLM to act as a scientific editor and classify the relationship between two papers (Extend, Support, Contrast, etc.).
*   **View G (`link_prediction.py`):** Implements a Graph Convolutional Network (GCN) using `PyTorch Geometric`. It trains on the citation graph to predict latent connections between papers.

### Phase 3: Graph Storage
*   **File:** `graph_loader.py`
*   **Database:** Neo4j.
*   **Schema:**
    *   **Nodes:** `Paper` (Properties: `id`, `title`, `problem_statement`, `core_method`, `embedding_problem`, etc.)
    *   **Edges:** `SEMANTIC_RELATION` (Properties: `relation_type`, `confidence`, `reasoning`).
*   **Indexing:** Automatically creates Vector Indexes on embedding properties for fast retrieval.

### Phase 4: Reasoning Engine (GraphRAG)
*   **File:** `graph_rag.py`
*   **Workflow:**
    1.  **Anchor Retrieval:** Converts user query to vector -> Searches Neo4j Vector Index for top-k papers.
    2.  **Graph Expansion:** Retrieves the 1-hop semantic neighborhood of anchor papers.
    3.  **Prompt Construction:** Assembles a context window containing both the paper content and the *reasoning* on the edges connecting them.
    4.  **Generation:** (Simulated) Sends the structured context to an LLM to answer the query.

## 5. Setup & Usage

### Prerequisites
*   Python 3.9+
*   Neo4j Database (Community Edition or AuraDB) running locally or in the cloud.

### Installation
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Set up environment variables (create a `.env` file or export):
    ```bash
    export NEO4J_URI="bolt://localhost:7687"
    export NEO4J_USER="neo4j"
    export NEO4J_PASSWORD="your_password"
    ```

### Running the Pipeline
To run the end-to-end simulation (Extraction -> Similarity -> Loading -> Reasoning):

```bash
python3 main.py
```

To run individual components:

*   **Link Prediction Training:** `python3 link_prediction.py`
*   **Similarity Analysis:** `python3 compute_similarity.py`
*   **Graph Loading:** `python3 graph_loader.py`

## 6. Current Status & Next Steps

**Current Status:**
*   The architecture is fully implemented.
*   The pipeline runs end-to-end with local data (`arxiv_source`).
*   LLM calls are currently simulated (prompts are generated but not sent to an API) to avoid costs during development.
*   Neo4j integration is live and functional.

**Next Steps:**
1.  **LLM Integration:** Replace the simulated steps in `main.py` and `graph_rag.py` with actual calls to OpenAI (GPT-4o) or Anthropic APIs using the prompts in `prompts.md`.
2.  **Data Scale-up:** Use `arxiv_scrape_latex.py` to download a larger corpus (e.g., 100 papers on a specific topic).
3.  **View G Integration:** Connect the trained GNN model from `link_prediction.py` into the `main.py` pipeline to suggest missing edges during ingestion.
