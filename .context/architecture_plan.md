# Architecture Plan

## System Overview
The LLM Comparison Tool is designed as a modular system that orchestrates multiple components to provide a comprehensive LLM evaluation framework. The system enables users to ingest various data sources, process them through a RAG pipeline, query multiple LLM backends, and evaluate their performance through an intuitive UI.

## Architecture Diagram
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

## Component Details

### User Interface (Streamlit)
- Provides interactive dashboard for model selection and comparison
- Visualizes evaluation metrics and outputs
- Offers configuration management for all system components

### Workflow Orchestration (llama_index)
- Manages data ingestion from multiple sources
- Orchestrates the RAG pipeline
- Handles query processing and routing to appropriate LLM backends

### Vector Store
- Uses PostgreSQL with pgvector extension (or SQLite for local development)
- Stores document embeddings for efficient retrieval
- Supports similarity search for RAG queries

### Embedding Layer
- Primary provider: Nomic Atlas API
- Generates embeddings for text and code
- Includes adapter interfaces for alternative embedding providers

### LLM Backends
- Local serving via Ollama with GPU detection
- Support for remote APIs (OpenAI, etc.)
- Standardized interface for adding new LLM providers

### Evaluation Framework (Comet ML Opik)
- Logs prompts, responses, and metadata
- Calculates evaluation metrics
- Enables annotation and benchmarking

### Web Crawler (Crawl4AI)
- Automates website crawling
- Preprocesses web content for ingestion
- Feeds data into the RAG pipeline

## Data Flow

1. User selects data sources, models, and evaluation metrics in the UI
2. Data is ingested via direct upload or web crawling
3. Content is processed and embedded using Nomic Atlas
4. Embeddings are stored in the vector database
5. User queries are processed through the RAG pipeline
6. Multiple LLM backends generate responses in parallel
7. Responses and metrics are logged via Comet ML Opik
8. Results are displayed in the UI for comparison

## Technical Interfaces

### API Interfaces
- Nomic Atlas API for embeddings
- LLM provider APIs (OpenAI, etc.)
- Comet ML Opik API for logging

### Internal Interfaces
- Vector store client interface
- LLM backend adapter interface
- Embedding provider interface
- Evaluation metric interface

## Deployment Architecture

### Development Environment
- Local Docker setup with docker-compose
- SQLite vector store for simplicity
- Local model serving via Ollama

### Production Environment (Future)
- Kubernetes deployment (optional for advanced users)
- PostgreSQL with pgvector
- Optional cloud-hosted LLM APIs

## Security Considerations
- API key management for third-party services
- Local deployment for sensitive data
- No persistent storage of user queries by default

## Performance Considerations
- Batch processing for large datasets
- Caching of embeddings and responses
- GPU acceleration for local models when available

*Last Updated: [Current Date]* 