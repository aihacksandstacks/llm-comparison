"""
Copyright (c) 2025 AI Hacks and Stacks
All rights reserved.

This file is part of the LLM Comparison Tool.
"""

import streamlit as st
import os
import sys
import time
import json
import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
import yaml

# Add the project root to the path to enable imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import our modules
from src.features.llm_compare.rag import get_rag_processor
from src.features.llm_compare.crawler import get_web_crawler, crawl_website, simple_http_crawl_sync
from src.features.llm_compare.evaluation import get_evaluation_manager
from src.features.llm_compare.llm import get_llm_provider
from src.features.llm_compare.embeddings import get_embedding_provider
from src.shared.config import DATA_DIR, get_config

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.indices = {}
    st.session_state.models = []
    st.session_state.selected_index = None
    st.session_state.current_experiment = None
    st.session_state.crawled_sites = []
    st.session_state.processed_files = []
    
    # Load embedding provider at startup
    try:
        # Initialize the embedding provider only once
        st.session_state.embedding_provider = get_embedding_provider()
        st.session_state.embedding_config = get_config("embeddings")
    except Exception as e:
        st.session_state.embedding_provider = None
        st.session_state.embedding_error = str(e)

# Function to refresh embedding provider only when necessary
def get_cached_embedding_provider():
    """Get embedding provider from cache, or create a new one if not available or if config changed"""
    current_config = get_config("embeddings")
    
    # Only create a new provider if we don't have one yet or if the config has changed
    if (not hasattr(st.session_state, 'embedding_provider') or 
        st.session_state.embedding_provider is None or
        not hasattr(st.session_state, 'embedding_config') or
        current_config != st.session_state.embedding_config):
        
        try:
            st.session_state.embedding_provider = get_embedding_provider()
            st.session_state.embedding_config = current_config
            if 'embedding_error' in st.session_state:
                del st.session_state.embedding_error
        except Exception as e:
            st.session_state.embedding_provider = None
            st.session_state.embedding_error = str(e)
    
    return st.session_state.embedding_provider

# Helpers for creating directory structure
def ensure_directories():
    """Ensure all required directories exist."""
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(DATA_DIR, "uploaded").mkdir(parents=True, exist_ok=True)
    Path(DATA_DIR, "crawled").mkdir(parents=True, exist_ok=True)
    Path(DATA_DIR, "indices").mkdir(parents=True, exist_ok=True)
    Path(DATA_DIR, "evaluations").mkdir(parents=True, exist_ok=True)

ensure_directories()

st.set_page_config(
    page_title="LLM Comparison Tool",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title and description
st.title("LLM Comparison Tool")
st.markdown("""
    A comprehensive platform for benchmarking and comparing multiple LLM backends 
    under a unified RAG and evaluation framework.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Home", "Data Ingestion", "Model Selection", "RAG Query", "Evaluation", "Results", "Settings"],
)

# Home page
if page == "Home":
    st.header("Welcome to the LLM Comparison Tool")
    
    st.markdown("""
        ### Features
        
        - **Workflow Orchestration**: Uses llama_index to ingest data and execute RAG queries against multiple LLMs
        - **Embedding Layer**: Supports both external API-based (Nomic Atlas, OpenAI) and local embeddings via Sentence Transformers
        - **Local Model Serving**: Serves models via Ollama on your machine
        - **Evaluation Framework**: Instruments Comet ML Opik to log prompts, responses, and metrics
        - **Web Crawling**: Incorporates Crawl4AI to crawl websites for RAG
        
        ### Getting Started
        
        1. Navigate to **Data Ingestion** to add documents or crawl websites
        2. Go to **Model Selection** to choose which LLMs to compare
        3. Try out RAG queries in the **RAG Query** page
        4. Configure settings in **Evaluation** to define metrics and run evaluations
        5. View the results in **Results** page
    """)
    
    # System status
    st.subheader("System Status")
    
    # Check for Ollama availability
    ollama_available = False
    try:
        ollama_provider = get_llm_provider("ollama")
        models = ollama_provider.get_available_models()
        ollama_available = len(models) > 0
    except Exception as e:
        st.error(f"Ollama error: {e}")
    
    # Check for database availability (simplified check)
    db_available = True  # Placeholder
    
    # Show embedding model
    try:
        # Use cached embedding provider, only refreshes if config changes
        embedding_provider = get_cached_embedding_provider()
        if embedding_provider:
            provider_type = embedding_provider.__class__.__name__
            
            # Get a more descriptive name for display
            if provider_type == "NomicEmbeddingProvider":
                embedding_model = f"Nomic Atlas ({embedding_provider.model})"
            elif provider_type == "OpenAIEmbeddingProvider":
                embedding_model = f"OpenAI ({embedding_provider.model})"
            elif provider_type == "LocalEmbeddingProvider":
                embedding_model = f"Local ({embedding_provider.model_name})"
            elif provider_type == "NomicLocalEmbeddingProvider":
                model_short_name = embedding_provider.model_name.split('/')[-1]
                embedding_model = f"Nomic Local ({model_short_name})"
            else:
                embedding_model = provider_type
        else:
            embedding_model = "Not Available"
    except Exception as e:
        if 'embedding_error' in st.session_state:
            st.warning(f"Embedding provider error: {st.session_state.embedding_error}")
        else:
            st.warning(f"Embedding provider error: {str(e)}")
        embedding_model = "Not Available"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Database", value="Connected" if db_available else "Disconnected")
    with col2:
        st.metric(label="Ollama Service", value="Available" if ollama_available else "Unavailable")
    with col3:
        st.metric(label="Embedding Model", value=embedding_model)

# Data Ingestion page
elif page == "Data Ingestion":
    st.header("Data Ingestion")
    
    tab1, tab2, tab3 = st.tabs(["File Upload", "Web Crawler", "Code Repository"])
    
    with tab1:
        st.subheader("Upload Files")
        uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=["pdf", "txt", "md", "html"])
        
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} files. Ready for processing.")
            index_name = st.text_input("Index Name", value=f"uploaded-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            
            if st.button("Process Files"):
                with st.spinner("Processing files..."):
                    # Save uploaded files to disk
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(DATA_DIR, "uploaded", uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        file_paths.append(file_path)
                    
                    # Process files with RAG
                    rag_processor = get_rag_processor()
                    documents = rag_processor.load_documents_from_files(file_paths)
                    
                    if documents:
                        index = rag_processor.create_index(documents, index_name)
                        st.session_state.indices[index_name] = index
                        st.session_state.processed_files.extend(file_paths)
                        st.success(f"Successfully created index '{index_name}' with {len(documents)} documents.")
                    else:
                        st.error("No documents were loaded. Please try again with different files.")
    
    with tab2:
        st.subheader("Web Crawler")
        url = st.text_input("URL to crawl", placeholder="https://hacksandstacks.ai")
        depth = st.slider("Crawl Depth", min_value=1, max_value=5, value=2)
        max_pages = st.slider("Maximum Pages", min_value=10, max_value=200, value=50)
        output_name = st.text_input("Output Name (optional)", placeholder="Custom name for the crawled dataset")
        use_simple_mode = st.checkbox("Use simple HTTP mode (for troubleshooting)", help="Uses simple HTTP requests instead of browser-based crawling")
        debug_mode = st.checkbox("Debug mode", help="Show detailed debugging information")
        
        if url:
            if st.button("Start Crawling"):
                with st.spinner(f"Crawling {url} with depth {depth}..."):
                    try:
                        # Use the module-level synchronous function instead of the instance method
                        from src.features.llm_compare.crawler import get_web_crawler, crawl_website, simple_http_crawl_sync
                        
                        if debug_mode:
                            # Enable debug logging
                            import logging
                            from src.features.llm_compare.crawler import logger
                            logger.setLevel(logging.DEBUG)
                            st.info("Debug mode enabled")
                        
                        # Start crawling
                        if use_simple_mode:
                            st.info("Using simple HTTP mode for crawling")
                            # Use the simple HTTP crawl method directly
                            simple_results = simple_http_crawl_sync(url)
                            
                            # Format results to match the expected structure
                            result = {
                                "results": simple_results,
                                "metadata": {
                                    "url": url,
                                    "depth": 1,
                                    "max_pages": 1,
                                    "pages_crawled": len(simple_results),
                                    "crawl_time": 0.0,
                                    "output_name": output_name or url.replace("https://", "").replace("http://", "").split("/")[0].replace(".", "_"),
                                    "method": "simple_http"
                                }
                            }
                        else:
                            if debug_mode:
                                st.info(f"Starting deep crawl with depth={depth}, max_pages={max_pages}")
                            
                            result = crawl_website(
                                url=url,
                                depth=depth,
                                max_pages=max_pages,
                                output_name=output_name if output_name else None
                            )
                            
                            if debug_mode:
                                st.code(f"Crawler returned: {type(result)}", language="python")
                                st.code(f"Keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}", language="python")
                        
                        if not isinstance(result, dict):
                            st.error(f"Unexpected result type: {type(result)}. Expected dictionary.")
                            st.code(str(result), language="python")
                            # Skip the rest of the processing
                        
                        elif "error" in result:
                            error_message = result['error']
                            # Show a more detailed error display with expandable traceback if available
                            st.error(f"Crawl error: {error_message}")
                            if "traceback" in result:
                                with st.expander("Show detailed error information"):
                                    st.code(result["traceback"], language="python")
                            # Show any additional details that might be helpful
                            if "metadata" in result:
                                with st.expander("Crawl attempt details"):
                                    st.json(result["metadata"])
                        elif "metadata" in result and "results" in result:
                            # Get metadata
                            metadata = result["metadata"]
                            st.success(f"Crawled {metadata['pages_crawled']} pages in {metadata['crawl_time']:.2f} seconds.")
                            
                            # Add to session state
                            st.session_state.crawled_sites.append(metadata)
                            
                            # Create index from crawled content
                            with st.spinner("Creating index from crawled content..."):
                                # Convert to documents
                                rag_processor = get_rag_processor()
                                documents = []
                                
                                from llama_index.core import Document
                                for page in result["results"]:
                                    doc = Document(
                                        text=page["content"],
                                        metadata={
                                            "url": page["url"],
                                            "title": page["title"],
                                            "source": "crawl",
                                            **page["metadata"]
                                        }
                                    )
                                    documents.append(doc)
                                
                                # Create index
                                index_name = f"crawled-{metadata['output_name']}"
                                if documents:
                                    index = rag_processor.create_index(documents, index_name)
                                    st.session_state.indices[index_name] = index
                                    st.success(f"Created index '{index_name}' with {len(documents)} pages.")
                                else:
                                    st.warning("No content found in the crawled pages.")
                        else:
                            st.error("Unexpected result format from crawler. Missing 'results' or 'metadata'.")
                            if debug_mode:
                                st.code(str(result), language="python")
                    
                    except Exception as e:
                        st.error(f"Error during crawl: {str(e)}")
                        import traceback
                        with st.expander("Show detailed error traceback"):
                            st.code(traceback.format_exc(), language="python")
                
                # Show history of crawled sites
                if st.session_state.crawled_sites:
                    st.subheader("Crawling History")
                    for i, metadata in enumerate(st.session_state.crawled_sites):
                        with st.expander(f"{metadata['url']} ({metadata['pages_crawled']} pages)"):
                            st.write(f"Depth: {metadata['depth']}")
                            st.write(f"Pages: {metadata['pages_crawled']}")
                            st.write(f"Time: {metadata['crawl_time']:.2f} seconds")
                            st.write(f"Output: {metadata['output_name']}")
    
    with tab3:
        st.subheader("Code Repository")
        repo_url = st.text_input("Enter GitHub repository URL", placeholder="https://github.com/user/repo")
        branch = st.text_input("Branch (optional)", value="main")
        max_files = st.slider("Maximum Files to Process", min_value=10, max_value=500, value=100)
        include_patterns = st.text_input("Include File Patterns (comma-separated)", 
                                       value="*.py,*.md,*.txt,*.js,*.html,*.css")
        exclude_patterns = st.text_input("Exclude File Patterns (comma-separated)", 
                                       value="*.pyc,__pycache__/*,node_modules/*,venv/*")
        
        if repo_url:
            if st.button("Clone Repository"):
                with st.spinner(f"Cloning {repo_url}..."):
                    try:
                        # Temporary directory for cloning
                        repo_dir = os.path.join(DATA_DIR, "repositories")
                        Path(repo_dir).mkdir(parents=True, exist_ok=True)
                        
                        # Generate a unique name for this repo
                        repo_name = repo_url.split("/")[-1].replace(".git", "")
                        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                        clone_dir = os.path.join(repo_dir, f"{repo_name}-{timestamp}")
                        
                        # Import Git
                        import git
                        
                        # Clone the repository
                        st.info(f"Cloning {repo_url} (branch: {branch}) to {clone_dir}...")
                        repo = git.Repo.clone_from(repo_url, clone_dir, branch=branch)
                        
                        # Process repository files
                        st.info("Processing repository files...")
                        
                        # Parse patterns
                        include_list = [p.strip() for p in include_patterns.split(",") if p.strip()]
                        exclude_list = [p.strip() for p in exclude_patterns.split(",") if p.strip()]
                        
                        from pathlib import Path
                        import fnmatch
                        
                        # Find files that match the patterns
                        all_files = []
                        for root, dirs, files in os.walk(clone_dir):
                            # Skip excluded directories
                            dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, pattern) for pattern in exclude_list)]
                            
                            for file in files:
                                file_path = os.path.join(root, file)
                                rel_path = os.path.relpath(file_path, clone_dir)
                                
                                # Check if file matches include patterns and not exclude patterns
                                if any(fnmatch.fnmatch(rel_path, pattern) for pattern in include_list) and \
                                   not any(fnmatch.fnmatch(rel_path, pattern) for pattern in exclude_list):
                                    all_files.append(file_path)
                        
                        # Limit to max_files
                        if len(all_files) > max_files:
                            st.warning(f"Repository has {len(all_files)} matching files. Processing only {max_files} files.")
                            all_files = all_files[:max_files]
                        
                        # Create documents from files
                        rag_processor = get_rag_processor()
                        documents = rag_processor.load_documents_from_files(all_files)
                        
                        # Create index
                        index_name = f"code-{repo_name}-{timestamp}"
                        if documents:
                            index = rag_processor.create_index(documents, index_name)
                            st.session_state.indices[index_name] = index
                            st.success(f"Successfully created index '{index_name}' with {len(documents)} files from {repo_name}.")
                        else:
                            st.error("No documents were loaded. Please try again with different file patterns.")
                            
                    except Exception as e:
                        st.error(f"Error cloning repository: {str(e)}")

# Model Selection page
elif page == "Model Selection":
    st.header("Model Selection")
    
    # Add tabs for different model providers
    model_tab1, model_tab2, model_tab3 = st.tabs(["Local Models (Ollama)", "Remote Models", "Custom Configuration"])
    
    with model_tab1:
        st.subheader("Local Models (Ollama)")
        try:
            # Get available local models
            ollama_provider = get_llm_provider("ollama")
            local_models = ollama_provider.get_available_models()
            
            if not local_models:
                st.warning("No local models found. Make sure Ollama is running and has models pulled.")
                local_models = ["llama3", "mistral", "mixtral", "phi3"]  # Fallback examples
            
            # Add a refresh button
            if st.button("Refresh Models", key="refresh_local"):
                st.experimental_rerun()
            
            # Display pulling instructions if needed
            st.info("If you don't see your desired model, you can pull it using the Ollama CLI: `ollama pull model_name`")
            
            # Allow multi-select of models
            selected_local_models = st.multiselect(
                "Select local models",
                local_models,
                default=local_models[:1] if local_models else []
            )
            
            # Show model cards for selected models
            if selected_local_models:
                for model in selected_local_models:
                    with st.expander(f"{model} details", expanded=False):
                        # Try to get model info
                        try:
                            model_info = ollama_provider.get_model_info(model)
                            if model_info:
                                st.write(f"**Model:** {model}")
                                st.write(f"**Description:** {model_info.get('description', 'No description available')}")
                                st.write(f"**Parameters:** {model_info.get('parameter_count', 'Unknown')} parameters")
                                st.write(f"**Context Length:** {model_info.get('context_length', 'Unknown')} tokens")
                                
                                # Display tags
                                if model_info.get('tags'):
                                    st.write(f"**Tags:** {', '.join(model_info.get('tags', []))}")
                            else:
                                st.write("Model information not available")
                        except Exception as e:
                            st.write(f"Could not fetch model info: {str(e)}")
        except Exception as e:
            st.error(f"Error connecting to Ollama: {e}")
            selected_local_models = []
        
    with model_tab2:
        st.subheader("Remote Models")
        try:
            # Check if OpenAI API key is available
            from src.shared.config import OPENAI_API_KEY
            
            # Remote model providers
            remote_provider = st.selectbox(
                "Select Provider", 
                ["OpenAI", "Anthropic", "Mistral AI", "Other"],
                index=0
            )
            
            # Show different models based on provider
            if remote_provider == "OpenAI":
                if OPENAI_API_KEY:
                    remote_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"]
                else:
                    remote_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"]
                    st.warning("OpenAI API key not found. Add it to your .env file to use these models.")
            elif remote_provider == "Anthropic":
                remote_models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
                st.info("Anthropic API key required to use these models.")
            elif remote_provider == "Mistral AI":
                remote_models = ["mistral-small", "mistral-medium", "mistral-large"]
                st.info("Mistral AI API key required to use these models.")
            else:
                remote_models = []
                custom_model = st.text_input("Enter model name")
                if custom_model:
                    remote_models = [custom_model]
            
            selected_remote_models = st.multiselect(
                f"Select {remote_provider} models",
                remote_models,
                default=[]
            )
            
            # Information about models
            if selected_remote_models and remote_provider == "OpenAI":
                for model in selected_remote_models:
                    with st.expander(f"{model} details", expanded=False):
                        if model == "gpt-3.5-turbo":
                            st.write("**GPT-3.5 Turbo**: Fast and cost-effective model, good for everyday tasks.")
                            st.write("**Context Length:** 16K tokens")
                            st.write("**Use for:** General purpose tasks, RAG, chatbots")
                        elif "gpt-4" in model:
                            st.write("**GPT-4**: Advanced capabilities for complex reasoning and understanding.")
                            if "turbo" in model:
                                st.write("**Context Length:** 128K tokens")
                            elif "o" in model:
                                st.write("**Context Length:** 128K tokens")
                                st.write("**Note:** Latest model with improved capabilities")
                            else:
                                st.write("**Context Length:** 8K tokens")
                            st.write("**Use for:** Complex analysis, advanced reasoning, creative tasks")
            
            elif selected_remote_models and remote_provider == "Anthropic":
                for model in selected_remote_models:
                    with st.expander(f"{model} details", expanded=False):
                        if "opus" in model:
                            st.write("**Claude 3 Opus**: Anthropic's most powerful model for highly complex tasks.")
                            st.write("**Context Length:** 200K tokens")
                        elif "sonnet" in model:
                            st.write("**Claude 3 Sonnet**: Balanced performance and efficiency.")
                            st.write("**Context Length:** 200K tokens")
                        elif "haiku" in model:
                            st.write("**Claude 3 Haiku**: Fastest and most lightweight model.")
                            st.write("**Context Length:** 200K tokens")
        except Exception as e:
            st.error(f"Error with remote models: {e}")
            selected_remote_models = []
    
    with model_tab3:
        st.subheader("Model Configuration")
        
        # Get combined selected models
        all_selected_models = [
            {"name": model, "provider": "ollama"} for model in selected_local_models
        ] + [
            {"name": model, "provider": remote_provider.lower() if 'remote_provider' in locals() else "openai"} for model in selected_remote_models
        ]
        
        if all_selected_models:
            st.write(f"Configure parameters for {len(all_selected_models)} selected models:")
            
            # Global parameters (apply to all models)
            st.write("### Global Parameters")
            
            use_global_params = st.checkbox("Use same parameters for all models", value=True)
            
            if use_global_params:
                col1, col2 = st.columns(2)
                with col1:
                    global_temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
                    global_max_tokens = st.slider("Max Tokens", min_value=16, max_value=4096, value=512, step=16)
                with col2:
                    global_top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
                    global_frequency_penalty = st.slider("Frequency Penalty", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
            
            # Individual model parameters
            if not use_global_params:
                st.write("### Model-Specific Parameters")
                
                # Store parameters for each model
                if "model_parameters" not in st.session_state:
                    st.session_state.model_parameters = {}
                
                for model_info in all_selected_models:
                    model_key = f"{model_info['name']}_{model_info['provider']}"
                    
                    # Initialize model parameters if not already set
                    if model_key not in st.session_state.model_parameters:
                        st.session_state.model_parameters[model_key] = {
                            "temperature": 0.7,
                            "max_tokens": 512,
                            "top_p": 0.9,
                            "frequency_penalty": 0.0
                        }
                    
                    with st.expander(f"{model_info['name']} ({model_info['provider']})", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.session_state.model_parameters[model_key]["temperature"] = st.slider(
                                "Temperature", min_value=0.0, max_value=1.0, 
                                value=st.session_state.model_parameters[model_key]["temperature"], 
                                step=0.1, key=f"temp_{model_key}"
                            )
                            
                            st.session_state.model_parameters[model_key]["max_tokens"] = st.slider(
                                "Max Tokens", min_value=16, max_value=4096, 
                                value=st.session_state.model_parameters[model_key]["max_tokens"], 
                                step=16, key=f"max_{model_key}"
                            )
                        
                        with col2:
                            st.session_state.model_parameters[model_key]["top_p"] = st.slider(
                                "Top P", min_value=0.0, max_value=1.0, 
                                value=st.session_state.model_parameters[model_key]["top_p"], 
                                step=0.05, key=f"top_p_{model_key}"
                            )
                            
                            st.session_state.model_parameters[model_key]["frequency_penalty"] = st.slider(
                                "Frequency Penalty", min_value=0.0, max_value=2.0, 
                                value=st.session_state.model_parameters[model_key]["frequency_penalty"], 
                                step=0.1, key=f"freq_{model_key}"
                            )
        else:
            st.warning("No models selected. Please select models in the Local Models or Remote Models tabs.")
    
    # Combine all selected models and update session state
    all_selected_models = [
        {"name": model, "provider": "ollama"} for model in selected_local_models
    ] + [
        {"name": model, "provider": remote_provider.lower() if 'remote_provider' in locals() else "openai"} for model in selected_remote_models
    ]
    
    # Update session state
    st.session_state.models = all_selected_models
    
    if "model_parameters" not in st.session_state:
        st.session_state.model_parameters = {}
    
    # Apply global parameters to all models if enabled
    if all_selected_models and 'use_global_params' in locals() and use_global_params:
        for model_info in all_selected_models:
            model_key = f"{model_info['name']}_{model_info['provider']}"
            st.session_state.model_parameters[model_key] = {
                "temperature": global_temperature,
                "max_tokens": global_max_tokens,
                "top_p": global_top_p,
                "frequency_penalty": global_frequency_penalty
            }
    
    # Show selected models
    if all_selected_models:
        st.success(f"Selected {len(all_selected_models)} models for comparison.")
        
        # Display model table
        model_table_data = {
            "Model": [],
            "Provider": [],
            "Temperature": [],
            "Max Tokens": []
        }
        
        for model_info in all_selected_models:
            model_key = f"{model_info['name']}_{model_info['provider']}"
            params = st.session_state.model_parameters.get(model_key, {})
            
            model_table_data["Model"].append(model_info['name'])
            model_table_data["Provider"].append(model_info['provider'])
            model_table_data["Temperature"].append(params.get("temperature", 0.7))
            model_table_data["Max Tokens"].append(params.get("max_tokens", 512))
        
        import pandas as pd
        model_df = pd.DataFrame(model_table_data)
        st.table(model_df)
    else:
        st.warning("No models selected. Please select at least one model to compare.")
    
    # System prompt template
    st.subheader("Prompt Settings")
    system_prompt = st.text_area(
        "System Prompt Template", 
        value="You are an assistant tasked with answering questions based on the provided context. Be concise and factual.",
        height=100
    )
    
    prompt_template = st.text_area(
        "User Prompt Template", 
        value="Context: {context}\n\nQuestion: {question}\n\nAnswer the question based on the context provided.",
        height=150
    )
    
    # Save settings
    if st.button("Save Model Settings"):
        st.session_state.system_prompt = system_prompt
        st.session_state.prompt_template = prompt_template
        st.success("Model settings saved successfully!")

# RAG Query page
elif page == "RAG Query":
    st.header("RAG Query Testing")
    
    # Get RAG processor for potential database operations
    rag_processor = get_rag_processor()
    
    # Check if we need to load collections from the database
    if not st.session_state.indices and rag_processor.db_connected:
        with st.spinner("Checking database for available collections..."):
            try:
                # Get collections from the database
                collections = rag_processor.db_provider.get_collections()
                
                if collections:
                    st.info(f"Found {len(collections)} collections in the database")
                    
                    # Let the user select collections to load
                    selected_collections = st.multiselect(
                        "Select collections to load from database",
                        options=collections,
                        default=collections[:1] if collections else []
                    )
                    
                    if selected_collections and st.button("Load Selected Collections"):
                        with st.spinner("Loading collections from database..."):
                            for collection_name in selected_collections:
                                # Create a dummy index for database access
                                # This helps maintain the same UI flow
                                st.session_state.indices[collection_name] = None
                            
                            st.success(f"Loaded {len(selected_collections)} collections from database")
                            st.rerun()  # Refresh the page to show the loaded collections
            except Exception as e:
                st.error(f"Error loading collections from database: {str(e)}")
    
    # Select an index
    if st.session_state.indices:
        index_names = list(st.session_state.indices.keys())
        selected_index_name = st.selectbox("Select Index", index_names)
        
        if selected_index_name:
            st.session_state.selected_index = selected_index_name
            index = st.session_state.indices[selected_index_name]
            
            # Query interface
            st.subheader("Query")
            query = st.text_area("Enter your query", height=100)
            
            # Model selection for this query
            if not st.session_state.models:
                st.warning("No models selected. Please go to the Model Selection page first.")
            else:
                # Create a clean list of models with provider info for display
                available_models = [
                    {"label": f"{model['name']} ({model['provider']})", 
                    "value": f"{model['name']}|{model['provider']}"} 
                    for model in st.session_state.models
                ]
                
                # Let user select models for comparison
                selected_models_for_query = st.multiselect(
                    "Select models for this query",
                    options=[model['value'] for model in available_models],
                    default=[available_models[0]['value']] if available_models else [],
                    format_func=lambda x: next((m['label'] for m in available_models if m['value'] == x), x)
                )
                
                # Display query settings
                st.write("---")
                st.subheader("Query Pipeline Settings")
                
                # Configure RAG settings
                col1, col2 = st.columns(2)
                with col1:
                    similarity_top_k = st.slider("Top K Retrieval", min_value=1, max_value=20, value=5)
                with col2:
                    similarity_threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
                
                # Display and edit prompt templates
                default_system_prompt = "You are a helpful assistant that answers questions based on the provided context."
                default_prompt_template = "Context: {context}\n\nQuestion: {question}\n\nAnswer the question based only on the context provided. If you don't know the answer based on the context, say so."
                
                system_prompt = st.text_area(
                    "System Prompt", 
                    value=st.session_state.get("system_prompt", default_system_prompt),
                    height=80
                )
                
                user_prompt_template = st.text_area(
                    "Prompt Template", 
                    value=st.session_state.get("prompt_template", default_prompt_template),
                    height=150
                )
                
                st.write("---")
                
                if query and selected_models_for_query and st.button("Run Query", type="primary"):
                    # Get RAG processor
                    rag_processor = get_rag_processor()
                    
                    # Process with RAG to get context
                    with st.spinner("Retrieving relevant context..."):
                        try:
                            # Update RAG parameters
                            rag_processor.similarity_top_k = similarity_top_k
                            rag_processor.similarity_threshold = similarity_threshold
                            
                            # Process the query
                            rag_result = rag_processor.query(index, query)
                            context = rag_result["response"]
                            source_nodes = rag_result.get("source_nodes", [])
                            
                            # Display retrieved context
                            st.success("✅ Context retrieved successfully")
                            with st.expander("Retrieved Context", expanded=True):
                                st.markdown(context)
                                
                                # Show source information if available
                                if source_nodes:
                                    st.subheader("Sources")
                                    for i, node in enumerate(source_nodes[:similarity_top_k]):
                                        source_score = node.get("score", 0) if isinstance(node, dict) else getattr(node, "score", 0)
                                        source_text = node.get("text", "") if isinstance(node, dict) else getattr(node, "text", "")
                                        source_doc = node.get("document", {}) if isinstance(node, dict) else getattr(node, "document", {})
                                        
                                        # Get metadata
                                        metadata = source_doc.get("metadata", {}) if isinstance(source_doc, dict) else getattr(source_doc, "metadata", {})
                                        
                                        with st.expander(f"Source {i+1} - Relevance: {source_score:.4f}", expanded=i==0):
                                            # Show source type with icon
                                            if metadata.get("source") == "crawl":
                                                st.markdown(f"🌐 **Web Page**: {metadata.get('url', 'Unknown URL')}")
                                                if metadata.get("title"):
                                                    st.markdown(f"**Title**: {metadata.get('title')}")
                                            elif metadata.get("source") == "file":
                                                file_icon = "📄"
                                                if metadata.get("file_path", "").endswith((".pdf", ".PDF")):
                                                    file_icon = "📊"
                                                elif metadata.get("file_path", "").endswith((".md", ".MD")):
                                                    file_icon = "📝"
                                                
                                                st.markdown(f"{file_icon} **File**: {metadata.get('file_path', 'Unknown file')}")
                                                if metadata.get("page_number"):
                                                    st.markdown(f"**Page**: {metadata.get('page_number')}")
                                            elif metadata.get("source") == "code":
                                                st.markdown(f"💻 **Code**: {metadata.get('file_path', 'Unknown file')}")
                                                if metadata.get("language"):
                                                    st.markdown(f"**Language**: {metadata.get('language')}")
                                            
                                            # Show the text snippet
                                            st.markdown("**Content:**")
                                            st.markdown(source_text)
                            
                            # Create a prompt from template
                            prompt = user_prompt_template.replace("{context}", context).replace("{question}", query)
                            
                            # Set up tracking for results
                            results = {}
                            
                            # Create progress indicator
                            progress_text = "Running queries against selected models..."
                            progress_bar = st.progress(0)
                            model_status = st.empty()
                            
                            # Run against selected models
                            for i, model_str in enumerate(selected_models_for_query):
                                model_name, provider_name = model_str.split('|')
                                model_status.info(f"Querying {model_name} ({provider_name})...")
                                
                                try:
                                    # Get LLM provider
                                    llm_provider = get_llm_provider(provider_name, model_name)
                                    
                                    # Get model-specific parameters
                                    model_key = f"{model_name}_{provider_name}"
                                    model_params = st.session_state.model_parameters.get(model_key, {})
                                    
                                    # Use asyncio to run the async generate function
                                    async def run_async():
                                        start_time = time.time()
                                        try:
                                            response = await llm_provider.generate(
                                                prompt=prompt,
                                                system_prompt=system_prompt,
                                                temperature=model_params.get("temperature", 0.7),
                                                max_tokens=model_params.get("max_tokens", 512),
                                                top_p=model_params.get("top_p", 0.9),
                                                frequency_penalty=model_params.get("frequency_penalty", 0.0),
                                                model=model_name
                                            )
                                            response["metadata"] = response.get("metadata", {})
                                            response["metadata"]["response_time"] = time.time() - start_time
                                            return response
                                        except Exception as e:
                                            error_msg = str(e)
                                            # Special handling for Qwen models
                                            if "Qwen" in model_name and "Extra data" in error_msg:
                                                st.error(f"Error with {model_name}: Qwen response format issue - the model returned invalid JSON")
                                                st.info("Try rerunning the query - this is a known issue with some Qwen models")
                                            # Return error response
                                            return {
                                                "text": f"Error: {error_msg}",
                                                "model": model_name,
                                                "provider": provider_name,
                                                "metadata": {
                                                    "response_time": time.time() - start_time,
                                                    "error": True,
                                                    "error_message": error_msg
                                                }
                                            }
                                    
                                    response = asyncio.run(run_async())
                                    results[model_str] = response
                                    
                                    # Update progress
                                    progress_bar.progress((i + 1) / len(selected_models_for_query))
                                    
                                except Exception as e:
                                    progress_bar.progress((i + 1) / len(selected_models_for_query))
                                    st.error(f"Error with {model_name}: {str(e)}")
                            
                            # Clear status messages
                            model_status.empty()
                            
                            # Display results
                            if results:
                                st.subheader("Model Responses")
                                
                                # Allow toggling between side-by-side and individual views
                                view_type = st.radio(
                                    "View Type",
                                    ["Side by Side", "Individual"],
                                    horizontal=True
                                )
                                
                                if view_type == "Side by Side" and len(results) > 1:
                                    # Create columns for each model
                                    cols = st.columns(min(len(results), 3))
                                    
                                    # If we have more than 3 models, we need multiple rows
                                    models_per_row = min(len(results), 3)
                                    rows_needed = (len(results) + models_per_row - 1) // models_per_row
                                    
                                    for row in range(rows_needed):
                                        start_idx = row * models_per_row
                                        end_idx = min(start_idx + models_per_row, len(results))
                                        
                                        # Create columns for this row
                                        if row > 0:  # Only create new columns if this isn't the first row
                                            cols = st.columns(min(end_idx - start_idx, 3))
                                        
                                        # Fill columns with model responses
                                        for i, (model_str, idx) in enumerate(zip(list(results.keys())[start_idx:end_idx], range(models_per_row))):
                                            result = results[model_str]
                                            model_name, provider_name = model_str.split('|')
                                            
                                            with cols[i]:
                                                st.markdown(f"### {model_name}")
                                                st.markdown(f"*Provider: {provider_name}*")
                                                st.markdown(result["text"])
                                                
                                                # Show metrics
                                                metrics_cols = st.columns(2)
                                                with metrics_cols[0]:
                                                    if "usage" in result:
                                                        st.metric("Tokens", result["usage"].get("total_tokens", "N/A"))
                                                with metrics_cols[1]:
                                                    response_time = result.get("metadata", {}).get("response_time", "N/A")
                                                    if response_time != "N/A":
                                                        st.metric("Response Time", f"{response_time:.2f}s")
                                                    else:
                                                        st.metric("Response Time", "N/A")
                                else:
                                    # Individual view - more detailed information for each model
                                    for model_str, result in results.items():
                                        model_name, provider_name = model_str.split('|')
                                        
                                        with st.expander(f"{model_name} ({provider_name})", expanded=True):
                                            st.markdown(result["text"])
                                            
                                            # Show metrics in a more organized way
                                            st.markdown("---")
                                            st.markdown("**Generation Metrics:**")
                                            
                                            # Create a three-column layout for metrics
                                            metric_cols = st.columns(3)
                                            
                                            # Column 1: Token usage
                                            with metric_cols[0]:
                                                if "usage" in result:
                                                    st.metric("Total Tokens", result["usage"].get("total_tokens", "N/A"))
                                                    if "prompt_tokens" in result["usage"]:
                                                        st.metric("Prompt Tokens", result["usage"].get("prompt_tokens", "N/A"))
                                                    if "completion_tokens" in result["usage"]:
                                                        st.metric("Completion Tokens", result["usage"].get("completion_tokens", "N/A"))
                                            
                                            # Column 2: Time metrics
                                            with metric_cols[1]:
                                                response_time = result.get("metadata", {}).get("response_time", "N/A")
                                                if response_time != "N/A":
                                                    st.metric("Response Time", f"{response_time:.2f}s")
                                                    
                                                    # Calculate tokens per second if available
                                                    if "usage" in result and "completion_tokens" in result["usage"]:
                                                        completion_tokens = result["usage"].get("completion_tokens", 0)
                                                        if completion_tokens > 0 and response_time > 0:
                                                            tokens_per_sec = completion_tokens / response_time
                                                            st.metric("Tokens/Second", f"{tokens_per_sec:.1f}")
                                            
                                            # Column 3: Model parameters
                                            with metric_cols[2]:
                                                model_key = f"{model_name}_{provider_name}"
                                                params = st.session_state.model_parameters.get(model_key, {})
                                                
                                                st.metric("Temperature", params.get("temperature", 0.7))
                                                st.metric("Max Tokens", params.get("max_tokens", 512))
                                
                                # Allow saving to evaluation experiment
                                if "current_experiment" in st.session_state and st.session_state.current_experiment:
                                    st.markdown("---")
                                    if st.button("Save to Current Evaluation Experiment"):
                                        # Get the evaluation manager
                                        eval_manager = get_evaluation_manager()
                                        
                                        # Log the prompt
                                        sample = eval_manager.log_prompt(
                                            experiment=st.session_state.current_experiment,
                                            prompt=prompt,
                                            system_prompt=system_prompt,
                                            context=context
                                        )
                                        
                                        # Log each response
                                        for model_str, result in results.items():
                                            model_name, provider_name = model_str.split('|')
                                            
                                            # Update the response with model and provider info
                                            result["model"] = model_name
                                            result["provider"] = provider_name
                                            
                                            eval_manager.log_response(
                                                experiment=st.session_state.current_experiment,
                                                sample_id=sample["id"],
                                                response=result
                                            )
                                        
                                        st.success("Saved to evaluation experiment!")
                        
                        except Exception as e:
                            st.error(f"Error processing query: {str(e)}")
    else:
        st.warning("No indices available. Please go to the Data Ingestion page to create an index.")

# Evaluation page
elif page == "Evaluation":
    st.header("Evaluation Configuration")
    
    # Experiment management
    st.subheader("Experiment")
    
    # Create new experiment
    exp_tab1, exp_tab2 = st.tabs(["Create New Experiment", "Load Existing Experiment"])
    
    with exp_tab1:
        exp_name = st.text_input("Experiment Name", value=f"Experiment-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        exp_tags = st.text_input("Tags (comma separated)", value="llm-comparison")
        
        if st.button("Create Experiment"):
            try:
                # Get evaluation manager
                eval_manager = get_evaluation_manager()
                
                # Create experiment
                experiment = eval_manager.create_experiment(
                    name=exp_name,
                    tags=exp_tags.split(",") if exp_tags else None
                )
                
                # Store in session state
                st.session_state.current_experiment = experiment
                st.success(f"Created experiment: {exp_name} (ID: {experiment['id']})")
            except Exception as e:
                st.error(f"Error creating experiment: {str(e)}")
    
    with exp_tab2:
        # This is just a placeholder - in a real app, we would fetch actual experiments
        # For now, we'll just use a text input for the ID
        exp_id = st.text_input("Experiment ID")
        
        if exp_id and st.button("Load Experiment"):
            try:
                # Get evaluation manager
                eval_manager = get_evaluation_manager()
                
                # Load experiment
                experiment = eval_manager.load_experiment(exp_id)
                
                if experiment:
                    # Store in session state
                    st.session_state.current_experiment = experiment
                    st.success(f"Loaded experiment: {experiment['name']} (ID: {experiment['id']})")
                else:
                    st.error(f"Experiment with ID {exp_id} not found.")
            except Exception as e:
                st.error(f"Error loading experiment: {str(e)}")
    
    # Show current experiment
    if "current_experiment" in st.session_state and st.session_state.current_experiment:
        st.info(f"Current experiment: {st.session_state.current_experiment['name']} (ID: {st.session_state.current_experiment['id']})")
    
    # Evaluation metrics
    st.subheader("Evaluation Metrics")
    metrics = st.multiselect(
        "Select evaluation metrics",
        ["ROUGE", "Semantic Similarity", "Response Time", "Token Count", "Hallucination Detection"],
        ["ROUGE", "Semantic Similarity", "Response Time"]
    )
    
    # Test queries
    st.subheader("Test Queries")
    test_queries = st.text_area("Enter test queries (one per line)", height=150)
    ground_truth = st.text_area("Ground truth answers (optional, one per line matching queries)", height=150)
    
    # Select index for context
    if st.session_state.indices:
        index_names = list(st.session_state.indices.keys())
        eval_index_name = st.selectbox("Select Index for RAG context", index_names)
    else:
        eval_index_name = None
        st.warning("No indices available. RAG evaluation will not include context.")
    
    # Run evaluation
    if "current_experiment" in st.session_state and st.session_state.current_experiment and st.session_state.models:
        if st.button("Run Evaluation"):
            if not test_queries:
                st.error("Please enter at least one test query.")
            else:
                # Parse queries and ground truth
                queries = [q.strip() for q in test_queries.split("\n") if q.strip()]
                truths = [t.strip() for t in ground_truth.split("\n") if t.strip()] if ground_truth else []
                
                # Pad truths if needed
                if len(truths) < len(queries):
                    truths.extend([""] * (len(queries) - len(truths)))
                
                # Get managers
                eval_manager = get_evaluation_manager()
                rag_processor = get_rag_processor()
                
                # Get index if selected
                index = st.session_state.indices.get(eval_index_name) if eval_index_name else None
                
                # Process each query
                progress_bar = st.progress(0)
                
                for i, (query, truth) in enumerate(zip(queries, truths)):
                    try:
                        # Get context if index available
                        context = None
                        if index:
                            rag_result = rag_processor.query(index, query)
                            context = rag_result["response"]
                        
                        # Create a prompt
                        if "prompt_template" in st.session_state:
                            prompt = st.session_state.prompt_template
                            if context:
                                prompt = prompt.replace("{context}", context)
                            prompt = prompt.replace("{question}", query)
                            system_prompt = st.session_state.system_prompt if "system_prompt" in st.session_state else None
                            temperature = st.session_state.temperature if "temperature" in st.session_state else 0.7
                        else:
                            prompt = f"Question: {query}\n\nAnswer:"
                            if context:
                                prompt = f"Context: {context}\n\n{prompt}"
                            system_prompt = "You are a helpful assistant."
                            temperature = 0.7
                        
                        # Log the prompt
                        sample = eval_manager.log_prompt(
                            experiment=st.session_state.current_experiment,
                            prompt=prompt,
                            system_prompt=system_prompt,
                            context=context
                        )
                        
                        # Process with each model
                        for model_info in st.session_state.models:
                            try:
                                # Get LLM provider
                                llm_provider = get_llm_provider(model_info["provider"], model_info["name"])
                                
                                # Use asyncio to run the async generate function
                                async def run_async():
                                    return await llm_provider.generate(
                                        prompt=prompt,
                                        system_prompt=system_prompt,
                                        temperature=temperature,
                                        model=model_info["name"]
                                    )
                                
                                response = asyncio.run(run_async())
                                
                                # Log the response
                                eval_manager.log_response(
                                    experiment=st.session_state.current_experiment,
                                    sample_id=sample["id"],
                                    response=response,
                                    ground_truth=truth if truth else None
                                )
                                
                            except Exception as e:
                                st.error(f"Error with {model_info['name']}: {str(e)}")
                        
                    except Exception as e:
                        st.error(f"Error processing query '{query}': {str(e)}")
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(queries))
                
                st.success("Evaluation completed!")
    else:
        if not st.session_state.models:
            st.warning("No models selected. Please go to the Model Selection page first.")
        else:
            st.warning("No current experiment. Please create or load an experiment first.")

# Results page
elif page == "Results":
    st.header("Comparison Results")
    
    if "current_experiment" in st.session_state and st.session_state.current_experiment:
        experiment = st.session_state.current_experiment
        
        # Display experiment info
        st.subheader(f"Experiment: {experiment['name']}")
        st.write(f"ID: {experiment['id']}")
        st.write(f"Tags: {', '.join(experiment['tags'])}")
        
        # Load the latest experiment data
        try:
            eval_manager = get_evaluation_manager()
            experiment = eval_manager.load_experiment(experiment['id'])
            
            if experiment and 'samples' in experiment and experiment['samples']:
                # Add tabs for different views
                results_tab1, results_tab2, results_tab3, results_tab4 = st.tabs([
                    "Metrics Overview", "Model Comparison", "Response Analysis", "Export"
                ])
                
                # Get metrics and models from responses
                all_metrics = {}
                all_models = set()
                model_providers = {}
                
                for sample in experiment['samples']:
                    for response in sample.get('responses', []):
                        model = response.get('model', 'unknown')
                        provider = response.get('provider', 'unknown')
                        all_models.add(model)
                        model_providers[model] = provider
                        
                        for metric_name, metric_value in response.get('metrics', {}).items():
                            if metric_name not in all_metrics:
                                all_metrics[metric_name] = {}
                            
                            if model not in all_metrics[metric_name]:
                                all_metrics[metric_name][model] = []
                            
                            all_metrics[metric_name][model].append(metric_value)
                
                # Calculate averages
                avg_metrics = {}
                for metric_name, model_values in all_metrics.items():
                    avg_metrics[metric_name] = {}
                    for model, values in model_values.items():
                        # Filter out any non-numeric values
                        numeric_values = [v for v in values if isinstance(v, (int, float))]
                        if numeric_values:
                            avg_metrics[metric_name][model] = sum(numeric_values) / len(numeric_values)
                
                # Tab 1: Metrics Overview
                with results_tab1:
                    if avg_metrics:
                        st.subheader("Average Metrics by Model")
                        
                        import pandas as pd
                        import numpy as np
                        import altair as alt
                        
                        # Create a combined DataFrame for all metrics
                        metrics_df_data = []
                        for metric_name, model_avgs in avg_metrics.items():
                            for model, avg_value in model_avgs.items():
                                metrics_df_data.append({
                                    'Metric': metric_name,
                                    'Model': model,
                                    'Value': avg_value,
                                    'Provider': model_providers.get(model, 'unknown')
                                })
                        
                        if metrics_df_data:
                            metrics_df = pd.DataFrame(metrics_df_data)
                            
                            # Normalize metrics for fair comparison (0-1 scale)
                            for metric in metrics_df['Metric'].unique():
                                metric_min = metrics_df[metrics_df['Metric'] == metric]['Value'].min()
                                metric_max = metrics_df[metrics_df['Metric'] == metric]['Value'].max()
                                if metric_max > metric_min:  # Avoid division by zero
                                    metrics_df.loc[metrics_df['Metric'] == metric, 'Normalized Value'] = \
                                        (metrics_df.loc[metrics_df['Metric'] == metric, 'Value'] - metric_min) / (metric_max - metric_min)
                                else:
                                    metrics_df.loc[metrics_df['Metric'] == metric, 'Normalized Value'] = 1.0
                            
                            # Create metric-specific charts
                            for metric in metrics_df['Metric'].unique():
                                st.write(f"### {metric}")
                                metric_data = metrics_df[metrics_df['Metric'] == metric]
                                
                                # Determine if higher is better based on metric name
                                higher_better = not ('time' in metric.lower() or 'latency' in metric.lower() or 
                                                   'error' in metric.lower())
                                
                                # Create a horizontal bar chart
                                chart = alt.Chart(metric_data).mark_bar().encode(
                                    x=alt.X('Value:Q', title=metric),
                                    y=alt.Y('Model:N', sort='-x' if higher_better else 'x'),
                                    color=alt.Color('Provider:N', legend=alt.Legend(title="Provider")),
                                    tooltip=['Model', 'Provider', 'Value']
                                ).properties(
                                    height=max(100, len(all_models) * 30)
                                )
                                
                                st.altair_chart(chart, use_container_width=True)
                    else:
                        st.warning("No metrics found in this experiment.")
                
                # Tab 2: Model Comparison
                with results_tab2:
                    if avg_metrics and len(all_models) >= 2:
                        st.subheader("Side by Side Model Comparison")
                        
                        # Allow user to select models to compare
                        selected_models = st.multiselect(
                            "Select models to compare",
                            list(all_models),
                            default=list(all_models)[:min(3, len(all_models))]
                        )
                        
                        if selected_models and len(selected_models) >= 2:
                            # Create comparison table
                            comparison_data = {
                                'Metric': []
                            }
                            
                            for model in selected_models:
                                comparison_data[f"{model} ({model_providers.get(model, 'unknown')})"] = []
                            
                            for metric_name in sorted(avg_metrics.keys()):
                                comparison_data['Metric'].append(metric_name)
                                for model in selected_models:
                                    value = avg_metrics[metric_name].get(model, "N/A")
                                    if isinstance(value, float):
                                        formatted_value = f"{value:.4f}"
                                    else:
                                        formatted_value = str(value)
                                    comparison_data[f"{model} ({model_providers.get(model, 'unknown')})"].append(formatted_value)
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            st.table(comparison_df)
                            
                            # Radar chart for normalized metric comparison
                            st.subheader("Model Performance Radar")
                            
                            # Get normalized values for selected models and metrics
                            radar_metrics = [m for m in avg_metrics.keys() if all(model in avg_metrics[m] for model in selected_models)]
                            
                            if radar_metrics:
                                # Prepare data for radar chart using Plotly
                                try:
                                    import plotly.graph_objects as go
                                    
                                    # Invert time-based metrics (lower is better)
                                    radar_data = []
                                    
                                    for model in selected_models:
                                        model_values = []
                                        for metric in radar_metrics:
                                            value = avg_metrics[metric].get(model, 0)
                                            
                                            # Invert metrics where lower is better
                                            if 'time' in metric.lower() or 'latency' in metric.lower() or 'error' in metric.lower():
                                                # Get the max value for this metric
                                                max_val = max([avg_metrics[metric].get(m, 0) for m in selected_models])
                                                if max_val > 0:  # Avoid division by zero
                                                    value = max_val - value  # Invert
                                            
                                            model_values.append(value)
                                        
                                        radar_data.append(go.Scatterpolar(
                                            r=model_values,
                                            theta=radar_metrics,
                                            fill='toself',
                                            name=f"{model} ({model_providers.get(model, 'unknown')})"
                                        ))
                                    
                                    fig = go.Figure(data=radar_data)
                                    fig.update_layout(
                                        polar=dict(
                                            radialaxis=dict(
                                                visible=True,
                                            )
                                        ),
                                        showlegend=True
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                except ImportError:
                                    st.warning("Plotly is required for radar charts. Install it with `pip install plotly`.")
                            else:
                                st.warning("Insufficient data for radar chart. Not all models have the same metrics.")
                            
                            # Model Leaderboard
                            st.subheader("Model Leaderboard")
                            
                            # Calculate an overall score based on normalized metrics
                            if metrics_df_data:
                                leaderboard_df = pd.DataFrame(metrics_df_data)
                                
                                # Only use selected models
                                leaderboard_df = leaderboard_df[leaderboard_df['Model'].isin(selected_models)]
                                
                                # Calculate scores (higher is better)
                                model_scores = {}
                                for model in selected_models:
                                    model_metrics = leaderboard_df[leaderboard_df['Model'] == model]
                                    
                                    # Calculate score as average of normalized values, adjusting for lower-is-better metrics
                                    scores = []
                                    for _, row in model_metrics.iterrows():
                                        metric = row['Metric']
                                        norm_value = row.get('Normalized Value', 0)
                                        
                                        # Invert normalized value for metrics where lower is better
                                        if 'time' in metric.lower() or 'latency' in metric.lower() or 'error' in metric.lower():
                                            norm_value = 1 - norm_value
                                        
                                        scores.append(norm_value)
                                    
                                    if scores:
                                        model_scores[model] = sum(scores) / len(scores)
                                    else:
                                        model_scores[model] = 0
                                
                                # Create leaderboard DataFrame
                                leaderboard_data = {
                                    'Model': [],
                                    'Provider': [],
                                    'Overall Score': []
                                }
                                
                                for model, score in model_scores.items():
                                    leaderboard_data['Model'].append(model)
                                    leaderboard_data['Provider'].append(model_providers.get(model, 'unknown'))
                                    leaderboard_data['Overall Score'].append(score)
                                
                                leaderboard_df = pd.DataFrame(leaderboard_data)
                                leaderboard_df = leaderboard_df.sort_values('Overall Score', ascending=False)
                                
                                # Show leaderboard with formatting
                                for i, (_, row) in enumerate(leaderboard_df.iterrows()):
                                    col1, col2, col3 = st.columns([1, 3, 2])
                                    with col1:
                                        if i == 0:
                                            st.markdown("🥇")
                                        elif i == 1:
                                            st.markdown("🥈")
                                        elif i == 2:
                                            st.markdown("🥉")
                                        else:
                                            st.markdown(f"{i+1}")
                                    with col2:
                                        st.markdown(f"**{row['Model']}** ({row['Provider']})")
                                    with col3:
                                        st.markdown(f"Score: {row['Overall Score']:.4f}")
                        else:
                            st.warning("Please select at least two models to compare.")
                    else:
                        st.warning("Need at least two models with metrics for comparison.")
                
                # Tab 3: Response Analysis
                with results_tab3:
                    st.subheader("Sample Responses")
                    
                    # Add filter for response search
                    search_term = st.text_input("Search in responses", "")
                    
                    filtered_samples = experiment['samples']
                    if search_term:
                        filtered_samples = [
                            sample for sample in experiment['samples']
                            if search_term.lower() in sample.get('prompt', '').lower() or
                               any(search_term.lower() in response.get('text', '').lower() 
                                   for response in sample.get('responses', []))
                        ]
                    
                    for i, sample in enumerate(filtered_samples):
                        with st.expander(f"Query {i+1}: {sample.get('prompt', '')[:50]}...", expanded=i==0 and len(filtered_samples) <= 5):
                            st.write("**Prompt:**")
                            st.write(sample.get('prompt', 'No prompt available'))
                            
                            if sample.get('system_prompt'):
                                st.write("**System Prompt:**")
                                st.write(sample.get('system_prompt'))
                            
                            if sample.get('context'):
                                with st.expander("Context", expanded=False):
                                    st.write(sample.get('context'))
                            
                            # Responses
                            if sample.get('responses'):
                                # Allow showing responses side by side or one by one
                                view_mode = st.radio(
                                    "View mode", 
                                    ["Side by side", "One by one"],
                                    key=f"view_mode_{i}",
                                    horizontal=True
                                )
                                
                                if view_mode == "Side by side" and len(sample.get('responses', [])) > 1:
                                    # Create columns based on number of responses (max 3 per row)
                                    responses = sample.get('responses', [])
                                    num_rows = (len(responses) + 2) // 3
                                    
                                    for row in range(num_rows):
                                        start_idx = row * 3
                                        end_idx = min(start_idx + 3, len(responses))
                                        row_responses = responses[start_idx:end_idx]
                                        
                                        cols = st.columns(len(row_responses))
                                        
                                        for j, (col, response) in enumerate(zip(cols, row_responses)):
                                            with col:
                                                model = response.get('model', 'unknown')
                                                provider = response.get('provider', 'unknown')
                                                st.markdown(f"**{model} ({provider})**")
                                                st.markdown(response.get('text', 'No response'))
                                                
                                                # Show metrics if available
                                                metrics = response.get('metrics', {})
                                                if metrics:
                                                    st.write("*Metrics:*")
                                                    for metric_name, metric_value in metrics.items():
                                                        st.metric(
                                                            label=metric_name, 
                                                            value=f"{metric_value:.3f}" if isinstance(metric_value, float) else metric_value
                                                        )
                                else:
                                    # Show responses one by one
                                    st.write("**Responses:**")
                                    
                                    for response in sample.get('responses', []):
                                        st.write(f"**{response.get('model', 'unknown')} ({response.get('provider', 'unknown')}):**")
                                        st.write(response.get('text', 'No response'))
                                        
                                        # Display metrics for this response
                                        metrics = response.get('metrics', {})
                                        if metrics:
                                            st.write("*Metrics:*")
                                            metric_cols = st.columns(min(3, len(metrics)))
                                            for j, (metric_name, metric_value) in enumerate(metrics.items()):
                                                metric_cols[j % len(metric_cols)].metric(
                                                    label=metric_name, 
                                                    value=f"{metric_value:.3f}" if isinstance(metric_value, float) else metric_value
                                                )
                                            st.markdown("---")
                            else:
                                st.warning("No responses found for this query.")
                
                # Tab 4: Export
                with results_tab4:
                    st.subheader("Export Results")
                    
                    export_format = st.selectbox(
                        "Export Format",
                        ["JSON", "CSV", "HTML Report"]
                    )
                    
                    if st.button("Export Results"):
                        try:
                            import json
                            import base64
                            
                            if export_format == "JSON":
                                # Create a JSON export
                                export_data = experiment.copy()
                                if "comet_experiment" in export_data:
                                    del export_data["comet_experiment"]
                                
                                # Convert to JSON string
                                json_str = json.dumps(export_data, indent=2)
                                
                                # Create download link
                                b64 = base64.b64encode(json_str.encode()).decode()
                                filename = f"{experiment['name']}_results.json"
                                href = f'<a href="data:application/json;base64,{b64}" download="{filename}">Download JSON Results</a>'
                                st.markdown(href, unsafe_allow_html=True)
                            
                            elif export_format == "CSV":
                                # Create CSV for metrics
                                if metrics_df_data:
                                    metrics_df = pd.DataFrame(metrics_df_data)
                                    csv = metrics_df.to_csv(index=False)
                                    
                                    # Create download link
                                    b64 = base64.b64encode(csv.encode()).decode()
                                    filename = f"{experiment['name']}_metrics.csv"
                                    href = f'<a href="data:text/csv;base64,{b64}" download="{filename}">Download CSV Metrics</a>'
                                    st.markdown(href, unsafe_allow_html=True)
                                else:
                                    st.warning("No metrics data to export.")
                            
                            elif export_format == "HTML Report":
                                # Create an HTML report
                                html_content = f"""
                                <html>
                                <head>
                                    <title>{experiment['name']} - LLM Comparison Report</title>
                                    <style>
                                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                                        h1, h2, h3 {{ color: #2c3e50; }}
                                        .model {{ margin-bottom: 10px; padding: 10px; border: 1px solid #ddd; }}
                                        .metrics {{ display: flex; flex-wrap: wrap; }}
                                        .metric {{ margin-right: 20px; }}
                                        .response {{ margin-top: 10px; padding: 10px; background-color: #f9f9f9; }}
                                    </style>
                                </head>
                                <body>
                                    <h1>{experiment['name']} - LLM Comparison Report</h1>
                                    <p>Experiment ID: {experiment['id']}</p>
                                    <p>Tags: {', '.join(experiment['tags'])}</p>
                                    
                                    <h2>Metrics Summary</h2>
                                """
                                
                                # Add metrics table
                                if metrics_df_data:
                                    metrics_df = pd.DataFrame(metrics_df_data)
                                    pivot_df = metrics_df.pivot_table(
                                        index=['Model', 'Provider'], 
                                        columns='Metric', 
                                        values='Value', 
                                        aggfunc='mean'
                                    ).reset_index()
                                    
                                    html_content += pivot_df.to_html(index=False)
                                
                                # Add samples
                                html_content += "<h2>Sample Queries and Responses</h2>"
                                
                                for i, sample in enumerate(experiment['samples']):
                                    html_content += f"""
                                    <h3>Query {i+1}</h3>
                                    <p><strong>Prompt:</strong> {sample.get('prompt', 'No prompt')}</p>
                                    """
                                    
                                    if sample.get('system_prompt'):
                                        html_content += f"<p><strong>System Prompt:</strong> {sample.get('system_prompt')}</p>"
                                    
                                    if sample.get('context'):
                                        html_content += f"<details><summary>Context</summary><p>{sample.get('context')}</p></details>"
                                    
                                    html_content += "<h4>Responses</h4>"
                                    
                                    for response in sample.get('responses', []):
                                        model = response.get('model', 'unknown')
                                        provider = response.get('provider', 'unknown')
                                        
                                        html_content += f"""
                                        <div class="model">
                                            <strong>{model} ({provider})</strong>
                                            <div class="response">{response.get('text', 'No response')}</div>
                                            
                                            <div class="metrics">
                                        """
                                        
                                        for metric_name, metric_value in response.get('metrics', {}).items():
                                            html_content += f"""
                                            <div class="metric">
                                                <strong>{metric_name}:</strong> {metric_value:.3f if isinstance(metric_value, float) else metric_value}
                                            </div>
                                            """
                                        
                                        html_content += "</div></div>"
                                
                                html_content += """
                                </body>
                                </html>
                                """
                                
                                # Create download link
                                b64 = base64.b64encode(html_content.encode()).decode()
                                filename = f"{experiment['name']}_report.html"
                                href = f'<a href="data:text/html;base64,{b64}" download="{filename}">Download HTML Report</a>'
                                st.markdown(href, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Error exporting results: {str(e)}")
            else:
                st.warning("No samples found in this experiment. Run an evaluation first.")
                
        except Exception as e:
            st.error(f"Error loading experiment results: {str(e)}")
    else:
        st.warning("No experiment selected. Please go to the Evaluation page to create or load an experiment.")
        
        # Show example visualization with mock data
        st.subheader("Sample Visualization")
        
        import pandas as pd
        import numpy as np
        
        # Sample data for demonstration
        models = ["llama3", "gpt-4", "mistral"]
        metrics = ["ROUGE", "Semantic Similarity", "Response Time (s)"]
        
        data = {
            "Model": np.repeat(models, len(metrics)),
            "Metric": metrics * len(models),
            "Value": [0.75, 0.82, 1.2, 0.88, 0.95, 2.5, 0.72, 0.80, 0.9]
        }
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create a pivot table
        pivot_df = df.pivot(index="Model", columns="Metric", values="Value")
        st.table(pivot_df)
        
        # Create a bar chart
        st.bar_chart(pivot_df)

# Settings page
elif page == "Settings":
    st.header("Settings")
    
    # Create tabs for different settings categories
    tab1, tab2 = st.tabs(["Embedding Settings", "System Settings"])
    
    with tab1:
        st.subheader("Embedding Provider Configuration")
        
        # Get current config
        embedding_config = get_config("embeddings")
        current_provider = embedding_config.get("provider", "nomic")
        
        # Provider selection
        provider_options = {
            "nomic_local": "Nomic Local (No API key needed)",  # Move this to first position to highlight it
            "nomic": "Nomic Atlas (External API)",
            "openai": "OpenAI (External API)",
            "local": "Local (Sentence Transformers)"
        }
        
        # Default to nomic_local if no provider is set
        if not current_provider or current_provider not in provider_options:
            current_provider = "nomic_local"
            
        selected_provider = st.selectbox(
            "Select Embedding Provider", 
            options=list(provider_options.keys()),
            format_func=lambda x: provider_options[x],
            index=list(provider_options.keys()).index(current_provider)
        )
        
        # Add info about nomic_local
        if selected_provider == "nomic_local":
            st.info("""
                **Nomic Local** embeddings run entirely on your machine with no API key required.
                These high-quality embeddings are:
                - State-of-the-art open source models that outperform many commercial alternatives
                - Able to run locally without sending data to external services
                - Fast and cost-effective for high-volume embedding tasks
                - Automatically optimized with task-specific prefixes
            """)
        elif selected_provider in ["nomic", "openai"]:
            st.warning("This provider requires an API key and will incur usage costs.")
        
        # Model selection based on provider
        if selected_provider == "nomic":
            model_options = ["nomic-embed-text-v1.5", "nomic-embed-text-v1"]
            default_model = embedding_config.get("model", "nomic-embed-text-v1.5")
            if default_model not in model_options:
                default_model = model_options[0]
                
            selected_model = st.selectbox(
                "Select Nomic Model",
                options=model_options,
                index=model_options.index(default_model)
            )
            
            # API key input
            api_key = st.text_input(
                "Nomic API Key",
                value="",
                type="password",
                help="Leave blank to use the key from environment variables"
            )
            
        elif selected_provider == "openai":
            model_options = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
            default_model = embedding_config.get("model", "text-embedding-3-small")
            if default_model not in model_options:
                default_model = model_options[0]
                
            selected_model = st.selectbox(
                "Select OpenAI Model",
                options=model_options,
                index=model_options.index(default_model)
            )
            
            # API key input
            api_key = st.text_input(
                "OpenAI API Key",
                value="",
                type="password",
                help="Leave blank to use the key from environment variables"
            )
            
        elif selected_provider == "local":
            model_options = [
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "paraphrase-multilingual-MiniLM-L12-v2",
                "multi-qa-MiniLM-L6-cos-v1"
            ]
            default_model = embedding_config.get("model", "all-MiniLM-L6-v2")
            if default_model not in model_options:
                default_model = model_options[0]
                
            selected_model = st.selectbox(
                "Select Local Model",
                options=model_options,
                index=model_options.index(default_model) if default_model in model_options else 0
            )
            
            st.info("""
                Local models run entirely on your machine and don't require an API key.
                Models will be downloaded automatically the first time they're used.
            """)
            
        elif selected_provider == "nomic_local":
            model_options = [
                "nomic-ai/nomic-embed-text-v1",
                "nomic-ai/nomic-embed-text-v1.5"
            ]
            default_model = embedding_config.get("model", "nomic-ai/nomic-embed-text-v1")
            if default_model not in model_options:
                default_model = model_options[0]
                
            selected_model = st.selectbox(
                "Select Nomic Local Model",
                options=model_options,
                index=model_options.index(default_model) if default_model in model_options else 0
            )
            
            task_type_options = ["search_document", "search_query", "clustering", "classification"]
            default_task_type = embedding_config.get("task_type", "search_document")
            
            selected_task_type = st.selectbox(
                "Select Task Type",
                options=task_type_options,
                index=task_type_options.index(default_task_type) if default_task_type in task_type_options else 0,
                help="Nomic Embed requires a task instruction prefix based on your use case"
            )
            
            st.info("""
                Nomic Embed models run entirely on your machine and don't require an API key.
                These are state-of-the-art open source models that outperform OpenAI's ada models.
                Models will be downloaded automatically the first time they're used (~270MB).
                
                Note: Each text being embedded requires a task instruction prefix (automatically added):
                - search_document: For embedding documents in a retrieval system
                - search_query: For embedding queries to search against documents
                - clustering: For grouping similar texts together
                - classification: For classifying texts into categories
            """)
        
        # Cache settings
        use_cache = st.checkbox("Enable Embedding Cache", value=embedding_config.get("cache_enabled", True))
        
        # Save settings button
        if st.button("Save Embedding Settings"):
            # Create a minimal config file update
            try:
                # Read the current config file
                with open(os.path.join(os.path.dirname(DATA_DIR), 'config.yaml'), 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Update embedding settings
                if 'embeddings' not in config_data:
                    config_data['embeddings'] = {}
                    
                config_data['embeddings']['provider'] = selected_provider
                config_data['embeddings']['model'] = selected_model
                config_data['embeddings']['cache_enabled'] = use_cache
                
                # Save task_type for Nomic Local
                if selected_provider == "nomic_local" and 'selected_task_type' in locals():
                    config_data['embeddings']['task_type'] = selected_task_type
                
                # Write the updated config
                with open(os.path.join(os.path.dirname(DATA_DIR), 'config.yaml'), 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False)
                
                # If API key was provided for relevant providers, save it to .env
                if (selected_provider in ["nomic", "openai"]) and 'api_key' in locals() and api_key:
                    env_path = os.path.join(os.path.dirname(DATA_DIR), '.env')
                    
                    # Read existing .env
                    env_contents = {}
                    if os.path.exists(env_path):
                        with open(env_path, 'r') as f:
                            for line in f:
                                if '=' in line and not line.startswith('#'):
                                    key, value = line.strip().split('=', 1)
                                    env_contents[key] = value
                    
                    # Update API key
                    if selected_provider == "nomic":
                        env_contents["NOMIC_API_KEY"] = api_key
                    elif selected_provider == "openai":
                        env_contents["OPENAI_API_KEY"] = api_key
                    
                    # Write back to .env
                    with open(env_path, 'w') as f:
                        for key, value in env_contents.items():
                            f.write(f"{key}={value}\n")
                
                # Force refresh of the embedding provider with new settings
                # This will happen automatically next time get_cached_embedding_provider is called
                if 'embedding_config' in st.session_state:
                    del st.session_state.embedding_config
                
                st.success("Successfully saved embedding settings!")
                
            except Exception as e:
                st.error(f"Error saving settings: {str(e)}")
    
    with tab2:
        st.subheader("System Configuration")
        st.info("Additional system settings will be available here in future updates.")
        
        # Show current config for reference
        if st.checkbox("Show Current Configuration"):
            st.json(get_config())

# Footer
st.sidebar.markdown("---")
st.sidebar.info("LLM Comparison Tool v0.1.0") 