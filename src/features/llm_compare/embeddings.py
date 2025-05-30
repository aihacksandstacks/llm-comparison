"""
Copyright (c) 2025 AI Hacks and Stacks
All rights reserved.

This file is part of the LLM Comparison Tool.

Embeddings module for the LLM Comparison Tool.
Provides interfaces for different embedding providers.
"""

from abc import ABC, abstractmethod
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import hashlib

from src.shared.config import CACHE_DIR, NOMIC_API_KEY, OPENAI_API_KEY, get_config

class EmbeddingProvider(ABC):
    """Base class for embedding providers."""
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of embedding vectors.
        """
        pass
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed.
            
        Returns:
            Embedding vector.
        """
        pass


class NomicEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using Nomic Atlas API."""
    
    def __init__(self, model: str = "nomic-embed-text-v1.5", use_cache: bool = True):
        """
        Initialize the Nomic Atlas embedding provider.
        
        Args:
            model: Model name to use for embeddings.
            use_cache: Whether to cache embeddings.
        """
        self.model = model
        self.api_key = NOMIC_API_KEY
        self.use_cache = use_cache
        self.cache_dir = Path(CACHE_DIR) / "nomic"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.api_key:
            raise ValueError("Nomic API key not found. Please set the NOMIC_API_KEY environment variable.")
        
        # Import here to avoid dependency issues if not using this provider
        try:
            import nomic
            nomic.login(self.api_key)
        except ImportError:
            raise ImportError("Nomic package not installed. Please install it with 'pip install nomic'.")
        
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Nomic Atlas.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of embedding vectors.
        """
        # Check cache first if enabled
        if self.use_cache:
            cached_embeddings = []
            texts_to_embed = []
            text_indices = []
            
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                cache_path = self.cache_dir / f"{cache_key}.pkl"
                
                if cache_path.exists():
                    with open(cache_path, 'rb') as f:
                        cached_embeddings.append(pickle.load(f))
                else:
                    texts_to_embed.append(text)
                    text_indices.append(i)
            
            # If all embeddings are cached, return them
            if not texts_to_embed:
                return cached_embeddings
            
            # Otherwise, embed the remaining texts
            import nomic
            new_embeddings = nomic.embed(texts_to_embed, model=self.model)
            
            # Cache the new embeddings
            for i, embedding in zip(text_indices, new_embeddings):
                cache_key = self._get_cache_key(texts[i])
                cache_path = self.cache_dir / f"{cache_key}.pkl"
                with open(cache_path, 'wb') as f:
                    pickle.dump(embedding, f)
            
            # Combine cached and new embeddings
            result = [None] * len(texts)
            for i, embedding in enumerate(cached_embeddings):
                result[i] = embedding
            for i, embedding in zip(text_indices, new_embeddings):
                result[i] = embedding
            
            return result
        else:
            # No caching, just embed all texts
            import nomic
            return nomic.embed(texts, model=self.model)
        
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using Nomic Atlas.
        
        Args:
            text: Text string to embed.
            
        Returns:
            Embedding vector.
        """
        return self.get_embeddings([text])[0]
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a text string."""
        return hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using OpenAI API."""
    
    def __init__(self, model: str = "text-embedding-3-small", use_cache: bool = True):
        """
        Initialize the OpenAI embedding provider.
        
        Args:
            model: Model name to use for embeddings.
            use_cache: Whether to cache embeddings.
        """
        self.model = model
        self.api_key = OPENAI_API_KEY
        self.use_cache = use_cache
        self.cache_dir = Path(CACHE_DIR) / "openai"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        
        # Import here to avoid dependency issues if not using this provider
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Please install it with 'pip install openai'.")
        
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of embedding vectors.
        """
        # Check cache first if enabled
        if self.use_cache:
            cached_embeddings = []
            texts_to_embed = []
            text_indices = []
            
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                cache_path = self.cache_dir / f"{cache_key}.pkl"
                
                if cache_path.exists():
                    with open(cache_path, 'rb') as f:
                        cached_embeddings.append(pickle.load(f))
                else:
                    texts_to_embed.append(text)
                    text_indices.append(i)
            
            # If all embeddings are cached, return them
            if not texts_to_embed:
                return cached_embeddings
            
            # Otherwise, embed the remaining texts
            response = self.client.embeddings.create(
                input=texts_to_embed,
                model=self.model
            )
            new_embeddings = [item.embedding for item in response.data]
            
            # Cache the new embeddings
            for i, embedding in zip(text_indices, new_embeddings):
                cache_key = self._get_cache_key(texts[i])
                cache_path = self.cache_dir / f"{cache_key}.pkl"
                with open(cache_path, 'wb') as f:
                    pickle.dump(embedding, f)
            
            # Combine cached and new embeddings
            result = [None] * len(texts)
            for i, embedding in enumerate(cached_embeddings):
                result[i] = embedding
            for i, embedding in zip(text_indices, new_embeddings):
                result[i] = embedding
            
            return result
        else:
            # No caching, just embed all texts
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            return [item.embedding for item in response.data]
        
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using OpenAI.
        
        Args:
            text: Text string to embed.
            
        Returns:
            Embedding vector.
        """
        return self.get_embeddings([text])[0]
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a text string."""
        return hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()


class LocalEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using local Sentence Transformers models."""
    
    def __init__(self, model: str = "all-MiniLM-L6-v2", use_cache: bool = True):
        """
        Initialize the local embedding provider using Sentence Transformers.
        
        Args:
            model: Model name to use for embeddings (from Hugging Face).
            use_cache: Whether to cache embeddings.
        """
        self.model_name = model
        self.use_cache = use_cache
        self.cache_dir = Path(CACHE_DIR) / "local" / model
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Import here to avoid dependency issues if not using this provider
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model)
        except ImportError:
            raise ImportError("Sentence Transformers package not installed. Please install it with 'pip install sentence-transformers'.")
        
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using local models.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of embedding vectors.
        """
        # Check cache first if enabled
        if self.use_cache:
            cached_embeddings = []
            texts_to_embed = []
            text_indices = []
            
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                cache_path = self.cache_dir / f"{cache_key}.pkl"
                
                if cache_path.exists():
                    with open(cache_path, 'rb') as f:
                        cached_embeddings.append(pickle.load(f))
                else:
                    texts_to_embed.append(text)
                    text_indices.append(i)
            
            # If all embeddings are cached, return them
            if not texts_to_embed:
                return cached_embeddings
            
            # Otherwise, embed the remaining texts
            embeddings = self.model.encode(texts_to_embed)
            new_embeddings = [embedding.tolist() for embedding in embeddings]
            
            # Cache the new embeddings
            for i, embedding in zip(text_indices, new_embeddings):
                cache_key = self._get_cache_key(texts[i])
                cache_path = self.cache_dir / f"{cache_key}.pkl"
                with open(cache_path, 'wb') as f:
                    pickle.dump(embedding, f)
            
            # Combine cached and new embeddings
            result = [None] * len(texts)
            for i, embedding in enumerate(cached_embeddings):
                result[i] = embedding
            for i, embedding in zip(text_indices, new_embeddings):
                result[i] = embedding
            
            return result
        else:
            # No caching, just embed all texts
            embeddings = self.model.encode(texts)
            return [embedding.tolist() for embedding in embeddings]
        
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using local model.
        
        Args:
            text: Text string to embed.
            
        Returns:
            Embedding vector.
        """
        return self.get_embeddings([text])[0]
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a text string."""
        return hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()


class NomicLocalEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using local Nomic Embed models without API key."""
    
    def __init__(self, model: str = "nomic-ai/nomic-embed-text-v1", task_type: str = "search_document", use_cache: bool = True):
        """
        Initialize the local Nomic Embed provider.
        
        Args:
            model: Model name to use for embeddings (from Hugging Face).
            task_type: Task type prefix to use ('search_document', 'search_query', 'clustering', 'classification').
            use_cache: Whether to cache embeddings.
        """
        self.model_name = model
        self.task_type = task_type
        self.use_cache = use_cache
        self.cache_dir = Path(CACHE_DIR) / "nomic_local" / model.split('/')[-1]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Valid task types
        self.valid_task_types = ["search_document", "search_query", "clustering", "classification"]
        if task_type not in self.valid_task_types:
            raise ValueError(f"Invalid task_type: {task_type}. Must be one of {self.valid_task_types}")
        
        # Import here to avoid dependency issues if not using this provider
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model, trust_remote_code=True)
        except ImportError:
            raise ImportError("Sentence Transformers package not installed. Please install it with 'pip install sentence-transformers'.")
        
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using local Nomic Embed model.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of embedding vectors.
        """
        # Add task prefix to each text if not already present
        processed_texts = []
        for text in texts:
            if not any(text.startswith(f"{task_type}: ") for task_type in self.valid_task_types):
                processed_texts.append(f"{self.task_type}: {text}")
            else:
                processed_texts.append(text)
        
        # Check cache first if enabled
        if self.use_cache:
            cached_embeddings = []
            texts_to_embed = []
            text_indices = []
            
            for i, text in enumerate(processed_texts):
                cache_key = self._get_cache_key(text)
                cache_path = self.cache_dir / f"{cache_key}.pkl"
                
                if cache_path.exists():
                    with open(cache_path, 'rb') as f:
                        cached_embeddings.append(pickle.load(f))
                else:
                    texts_to_embed.append(text)
                    text_indices.append(i)
            
            # If all embeddings are cached, return them
            if not texts_to_embed:
                return cached_embeddings
            
            # Otherwise, embed the remaining texts
            embeddings = self.model.encode(texts_to_embed)
            new_embeddings = [embedding.tolist() for embedding in embeddings]
            
            # Cache the new embeddings
            for i, embedding in zip(text_indices, new_embeddings):
                cache_key = self._get_cache_key(processed_texts[i])
                cache_path = self.cache_dir / f"{cache_key}.pkl"
                with open(cache_path, 'wb') as f:
                    pickle.dump(embedding, f)
            
            # Combine cached and new embeddings
            result = [None] * len(processed_texts)
            for i, embedding in enumerate(cached_embeddings):
                result[i] = embedding
            for i, embedding in zip(text_indices, new_embeddings):
                result[i] = embedding
            
            return result
        else:
            # No caching, just embed all texts
            embeddings = self.model.encode(processed_texts)
            return [embedding.tolist() for embedding in embeddings]
        
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using local Nomic model.
        
        Args:
            text: Text string to embed.
            
        Returns:
            Embedding vector.
        """
        return self.get_embeddings([text])[0]
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a text string."""
        return hashlib.md5(f"{self.model_name}:{self.task_type}:{text}".encode()).hexdigest()


def get_embedding_provider(provider_name: Optional[str] = None) -> EmbeddingProvider:
    """
    Factory function to get the configured embedding provider.
    
    Args:
        provider_name: Optional name of the provider to use. If None, uses the one from config.
        
    Returns:
        An instance of the appropriate EmbeddingProvider.
    """
    config = get_config("embeddings")
    provider = provider_name or config.get("provider", "nomic_local")
    model = config.get("model")
    use_cache = config.get("cache_enabled", True)
    
    if provider == "nomic":
        return NomicEmbeddingProvider(
            model=model or "nomic-embed-text-v1.5",
            use_cache=use_cache
        )
    elif provider == "openai":
        return OpenAIEmbeddingProvider(
            model=model or "text-embedding-3-small",
            use_cache=use_cache
        )
    elif provider == "local":
        return LocalEmbeddingProvider(
            model=model or "all-MiniLM-L6-v2",
            use_cache=use_cache
        )
    elif provider == "nomic_local":
        task_type = config.get("task_type", "search_document")
        return NomicLocalEmbeddingProvider(
            model=model or "nomic-ai/nomic-embed-text-v1",
            task_type=task_type,
            use_cache=use_cache
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}") 