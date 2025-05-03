# Product Requirements Document (PRD)

## 1. Introduction

### Purpose
Define the requirements for an LLM Comparison Tool to benchmark and compare multiple LLM backends under a unified RAG and evaluation framework.

## 2. Objectives & Goals

- Enable fair comparison of LLMs on custom datasets
- Support RAG-enhanced queries over web-crawled content, PDFs, and code
- Provide automated evaluation and visualization of metrics (e.g., accuracy, latency, cost)
- Offer a user-friendly UI for non-developers

## 3. Scope

### In Scope
- Data ingestion (web/PDF/code)
- RAG pipeline orchestration
- Embedding integration
- Local and remote LLM backends
- Evaluation logging
- Interactive UI

### Out of Scope
- Hosted SaaS deployment
- Production-grade security (MVP)

## 4. Functional Requirements

### 4.1 Model Comparison
- Connect to multiple LLM backends (e.g., OpenAI, Ollama locally)
- Standardize prompts and capture raw outputs and metadata

### 4.2 RAG Pipeline & Embeddings
- Use llama_index for ingestion and retrieval from a vector store
- Generate embeddings via Nomic Atlas, with fallback to alternative providers

### 4.3 Tool Integrations
- Comet ML Opik: Log and visualize evaluation traces and annotations
- Crawl4AI: Automate website crawling for ingestion

### 4.4 UI Requirements
- Streamlit dashboard to select data sources, models, and evaluation suites
- Display side-by-side comparisons of outputs and metrics

## 5. Non-Functional Requirements

- **Modular codebase**: Easy to extend with new components
- **Performance**: Support batch ingestion and parallel evaluation
- **Configurability**: All parameters in a YAML/JSON config file
- **Portability**: Dockerized for one-click local launch

## 6. Technical Architecture

```
User Interface (Streamlit)
       ↓
Workflow Orchestrator (llama_index)
       ↓
Vector Store (Postgres/SQLite with pgvector)
       ↓
Embedding API (Nomic Atlas)
       ↓
LLM Backends (Ollama, OpenAI, etc.)
       ↓
Evaluation & Logging (Comet ML Opik)
       ↑
Web Crawler (Crawl4AI) → Ingestion
```

## 7. Technology Stack

- **Orchestration**: llama_index
- **Embeddings**: Nomic Atlas
- **Local Serving**: Ollama
- **Evaluation**: Comet ML Opik
- **Crawling**: Crawl4AI
- **UI**: Streamlit
- **Data Store**: PostgreSQL + pgvector

## 8. Milestones & Timeline

1. **Week 1–2**: Scaffold project, Docker setup, basic llama_index ingestion
2. **Week 3–4**: Integrate embeddings and vector store
3. **Week 5–6**: Add Ollama backend and Comet Opik logging
4. **Week 7–8**: Implement Crawl4AI ingestion and RAG queries
5. **Week 9–10**: Build Streamlit UI and comparison dashboard
6. **Week 11–12**: Testing, documentation, and MVP release

## 9. Risks & Mitigations

- **Model compatibility issues**: Provide adapter interfaces for each LLM
- **Embedding API rate limits**: Cache embeddings; allow local fallback
- **Resource constraints**: Enable CPU-only builds; optimize batch sizes

## 10. Open Questions

- Which LLMs beyond Ollama/OpenAI should we support first?
- What evaluation metrics are most critical (e.g., ROUGE, BLEU, custom)?
- Should we include a cloud-hosted demo environment in Phase 2?

## 11. Prompt for AI Coding Agent

You are an expert AI developer assistant. 
Your task is to generate a scaffold for an LLM comparison tool with the following requirements:

1. **Workflow orchestration**  
   - Use llama_index to ingest data (web pages, PDFs, code) into a vector store and execute retrieval-augmented queries against multiple LLMs.
   
2. **Embedding layer**  
   - Integrate Nomic Atlas embeddings for text and code, with flexibility to swap embedding providers.
   
3. **Local model serving**  
   - Serve models via Ollama on the user's machine, with Docker-compose config and GPU detection.
   
4. **Evaluation framework**  
   - Instrument Comet ML Opik to log prompts, responses, and evaluation metrics; enable automated benchmarking and annotation.
   
5. **Web crawling**  
   - Incorporate Crawl4AI to crawl and preprocess websites into the vector store for RAG.
   
6. **UI**  
   - Build an interactive Streamlit interface for selecting models, running tests, visualizing metrics, and comparing outputs.
   
7. **Extensibility**  
   - Structure code modularly to add new LLM backends, embedding providers, and evaluation metrics without rewriting core logic.
   
8. **Deliverables**  
   - Provide a directory scaffold, Docker-compose files, Python modules with docstrings, unit tests, and a README with setup and usage.

*Last Updated: [Current Date]* 