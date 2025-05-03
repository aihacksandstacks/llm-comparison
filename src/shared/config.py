"""
Configuration module for the LLM Comparison Tool.
Loads settings from environment variables and config files.
"""

import os
import yaml
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any, Optional

# Load environment variables from .env file
load_dotenv()

# Base directories
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = os.getenv('DATA_DIR', str(ROOT_DIR / 'data'))
CACHE_DIR = os.getenv('EMBEDDING_CACHE_DIR', str(ROOT_DIR / 'cache' / 'embeddings'))
MODELS_DIR = os.getenv('MODELS_DIR', str(ROOT_DIR / 'models'))

# Create directories if they don't exist
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

# Database configuration
DB_PROVIDER = os.getenv('DB_PROVIDER', 'postgres').lower()
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', '5432')),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
    'database': os.getenv('POSTGRES_DB', 'llm_comparison'),
}

# Supabase configuration
SUPABASE_CONFIG = {
    'url': os.getenv('SUPABASE_URL', ''),
    'key': os.getenv('SUPABASE_KEY', ''),
    'table': os.getenv('SUPABASE_TABLE', 'embeddings'),
}

# Ollama configuration
OLLAMA_CONFIG = {
    'host': os.getenv('OLLAMA_HOST', 'localhost'),
    'port': int(os.getenv('OLLAMA_PORT', '11434')),
}

# Opik configuration
OPIK_CONFIG = {
    'host': os.getenv('OPIK_HOST', 'localhost'),
    'port': int(os.getenv('OPIK_PORT', '8000')),
}

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
NOMIC_API_KEY = os.getenv('NOMIC_API_KEY', '')
COMET_API_KEY = os.getenv('COMET_API_KEY', '')
CRAWL4AI_API_KEY = os.getenv('CRAWL4AI_API_KEY', '')

# Comet ML configuration
COMET_CONFIG = {
    'api_key': COMET_API_KEY,
    'workspace': os.getenv('COMET_WORKSPACE', ''),
    'project_name': os.getenv('COMET_PROJECT_NAME', 'llm_comparison'),
}

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_DIR = os.getenv('LOG_DIR', str(ROOT_DIR / 'logs'))
LOG_CONFIG = {
    'level': LOG_LEVEL,
    'dir': LOG_DIR,
    'file_name_prefix': 'llm_comparison',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# Load additional configuration from YAML file if exists
CONFIG_FILE = ROOT_DIR / 'config.yaml'
_yaml_config = {}

if CONFIG_FILE.exists():
    try:
        with open(CONFIG_FILE, 'r') as f:
            _yaml_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")


def get_config(section: Optional[str] = None) -> Dict[str, Any]:
    """
    Get configuration from the YAML config file.
    
    Args:
        section: Optional section name to retrieve. If None, returns the entire config.
        
    Returns:
        Dict containing the requested configuration.
    """
    if section is None:
        return _yaml_config
    return _yaml_config.get(section, {})
