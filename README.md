# LLM Comparison Tool

A comprehensive platform for benchmarking and comparing multiple LLM backends under a unified RAG and evaluation framework.

## Overview

The LLM Comparison Tool enables fair comparison of LLMs on custom datasets, supporting RAG-enhanced queries over web-crawled content, PDFs, and code. It provides automated evaluation and visualization of metrics through an intuitive user interface.

## Features

- **Workflow Orchestration**: Uses llama_index to ingest data and execute RAG queries against multiple LLMs
- **Embedding Layer**: Supports both external API-based (Nomic Atlas, OpenAI) and local embeddings via Sentence Transformers
- **Local Model Serving**: Serves models via Ollama on the user's machine, with Docker-compose config and GPU detection
- **Evaluation Framework**: Instruments Comet ML Opik to log prompts, responses, and metrics
- **Web Crawling**: Incorporates Crawl4AI to crawl and preprocess websites for RAG
- **Interactive UI**: Built with Streamlit for model selection, testing, and comparison
- **Advanced Visualization**: Side-by-side comparisons, radar charts, and performance leaderboards
- **Code Repository Analysis**: Ingest and query code repositories for technical documentation or code understanding

## Architecture

```
User Interface (Streamlit)
       ↓
Workflow Orchestrator (llama_index)
       ↓
Vector Store (Postgres/SQLite with pgvector)
       ↓
Embedding Layer (Nomic Atlas, OpenAI, or Local Sentence Transformers)
       ↓
LLM Backends (Ollama, OpenAI, etc.)
       ↓
Evaluation & Logging (Comet ML Opik)
       ↑
Web Crawler (Crawl4AI) → Ingestion
```

## Installation

### Prerequisites

- Python 3.9+
- Docker and docker-compose
- (Optional) NVIDIA GPU with CUDA support for GPU acceleration
- Ollama installed locally for local model serving

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/llm-comparison.git
   cd llm-comparison
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

4. Start the services using Docker:
   ```bash
   docker-compose up -d
   ```

5. Run the Streamlit UI:
   ```bash
   streamlit run src/features/base_ui/app.py
   ```

## Detailed Usage Guide

### 1. Data Ingestion

The Data Ingestion page offers three methods to load data:

#### File Upload
- Supports PDF, TXT, MD, and HTML files
- Documents are processed and stored in a vector index
- Customize index name for better organization

#### Web Crawler
- Enter URLs to crawl websites automatically
- Configure crawl depth and maximum pages
- Results are stored in a vector index for querying

#### Code Repository
- Clone and analyze GitHub repositories
- Filter files by pattern (e.g., *.py, *.md)
- Set maximum files to process for large repositories
- Creates a searchable index of code files

### 2. Model Selection

The Model Selection page allows you to:

- Select local models from Ollama
- Configure remote models from providers like OpenAI and Anthropic
- Customize model parameters (temperature, max tokens, top_p, etc.)
- Save global prompt templates for all queries

**Setting up Ollama models:**
1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull models using the command line: `ollama pull llama3` (or other models)
3. Select the pulled models in the UI

### 3. Embedding Configuration

The Settings page allows you to configure your embedding provider:

#### External API Providers
- **Nomic Atlas**: High-quality embeddings optimized for text and code (requires API key)
- **OpenAI**: Industry-standard embeddings with multiple model options (requires API key)

#### Local Embeddings
- **Sentence Transformers**: Run embeddings entirely locally without API calls
- Available models include:
  - `all-MiniLM-L6-v2`: Fast, lightweight model (384 dimensions)
  - `all-mpnet-base-v2`: Higher quality but slower (768 dimensions)
  - `paraphrase-multilingual-MiniLM-L12-v2`: Multilingual support

#### Nomic Local Embeddings
- **Nomic Embed**: State-of-the-art local embeddings from Nomic AI
- No API key required, runs completely on your machine
- Available models:
  - `nomic-ai/nomic-embed-text-v1`: Powerful 768-dimensional embeddings
  - `nomic-ai/nomic-embed-text-v1.5`: Newer model with improved performance
- Supports task-specific prefixes:
  - `search_document`: For embedding documents in a retrieval system
  - `search_query`: For embedding queries to search against documents
  - `clustering`: For grouping similar texts together
  - `classification`: For classifying texts into categories

Models are automatically downloaded on first use and cached for future runs.

To configure:
1. Navigate to the Settings page in the UI
2. Select your preferred embedding provider
3. Choose a specific model
4. Provide API keys if using external providers
5. Click "Save Embedding Settings"

### 4. RAG Query

The RAG Query page enables you to:

- Select a previously created index
- Enter a natural language query
- Choose which models to compare for this query
- Configure RAG parameters (similarity top-k, threshold)
- View retrieved context with sources and relevance scores
- Compare model responses side-by-side or individually
- Save results to evaluation experiments

### 5. Evaluation

The Evaluation page helps you:

- Create or load evaluation experiments
- Select evaluation metrics (ROUGE, semantic similarity, etc.)
- Run batch evaluations against multiple models
- Automate testing with multiple queries
- Add optional ground truth answers for accuracy measurement

### 6. Results Visualization

The Results page provides comprehensive visualization:

- Metrics Overview: Charts for each evaluation metric
- Model Comparison: Side-by-side comparisons and radar charts
- Response Analysis: Searchable history of all model responses
- Export: Download results in JSON, CSV, or HTML report formats
- Model Leaderboard: Overall performance ranking

## Troubleshooting

### Ollama Connection Issues
- Ensure Ollama is running locally (`ollama serve`)
- Check that the Ollama API is accessible at http://localhost:11434
- Try running `ollama list` to see available models

### Model-Specific Issues

#### Qwen Models
- If you encounter "Extra data" JSON parsing errors with Qwen models (particularly Qwen3:30b-a3b), the application includes a specialized parser to handle this
- These errors typically appear as: "Extra data: line 2 column 1 (char 113)"
- If problems persist, try:
  - Rerunning the query (sometimes the model returns valid JSON on subsequent attempts)
  - Reducing the input prompt length
  - Using a different model for critical applications

### Vector Database Connection
- Verify PostgreSQL with pgvector is running (`docker ps`)
- Check connection settings in your .env file
- Database tables are created automatically on first run

### API Key Problems
- Ensure all required API keys are in your .env file
- Check API usage limits for remote services

### Embedding Issues
- For API-based embeddings, verify your API keys in .env file
- For local embeddings:
  - Ensure sentence-transformers is properly installed
  - Check for disk space if models fail to download
  - Monitor memory usage as models are loaded into RAM
  - First-time model downloads may take several minutes depending on your connection

### Performance Issues
- For large documents, increase chunk size in config.yaml
- If using GPU, ensure proper CUDA configuration
- Reduce batch sizes for embedding operations
- When experiencing response parsing errors with specialized models, check log files for detailed diagnostic information

## Development

### Project Structure

```
llm-comparison/
├── database/
│   └── supabase/
│       └── supabase_setup.sql  # SQL setup for Supabase
├── docker/                  # Docker configuration
├── src/
│   ├── features/
│   │   ├── base_ui/         # Streamlit UI
│   │   ├── llm_compare/     # Core comparison logic
│   ├── shared/              # Shared utilities
├── tests/                   # Test suite
├── .env.example             # Example environment variables
├── docker-compose.yml       # Docker compose configuration
├── docker-compose.dev.yml   # Development docker-compose with hot reloading
├── dev.sh                   # Script to launch dev environment in Docker
├── run_dev.py               # Script to run app locally with hot reloading
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

### Development Mode

There are two ways to run the application in development mode with hot reloading:

#### 1. Using Docker (recommended)

This approach runs everything in containers but still enables hot reloading when you change the code:

```bash
# Make sure dev.sh is executable
chmod +x dev.sh

# Run the development environment
./dev.sh
```

#### 2. Running Locally

If you prefer to run the Streamlit app directly on your host machine:

```bash
# Activate your virtual environment
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Run the development script
python run_dev.py
```

Both methods enable hot reloading, which automatically refreshes the app when you modify the code.

### Running Tests

```bash
pytest tests/
```

### Adding New Models

To add support for a new LLM provider:
1. Create a new provider class in `src/features/llm_compare/llm.py`
2. Implement the required interface methods
3. Register the provider in the `get_llm_provider` factory function
4. Update the UI to display the new provider option

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [llama_index](https://docs.llamaindex.ai/)
- [Nomic Atlas](https://docs.nomic.ai/atlas/)
- [Ollama](https://ollama.ai/)
- [Comet ML Opik](https://www.comet.com/site/products/opik/)
- [Crawl4AI](https://github.com/unclecode/crawl4ai)
- [Streamlit](https://streamlit.io/)

## Vector Database Options

### PostgreSQL with pgvector (Default)

The default configuration uses a local PostgreSQL database with the pgvector extension for vector storage and similarity search. This option works out of the box with the provided docker-compose configuration.

### Supabase Integration

You can also use Supabase as your vector database. To set this up:

1. Create a Supabase project at https://supabase.com
2. Enable the pgvector extension in your Supabase project
3. Create the necessary tables and functions by running the SQL in `database/supabase/supabase_setup.sql`
4. Update your `.env`