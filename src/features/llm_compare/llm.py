"""
Copyright (c) 2025 AI Hacks and Stacks
All rights reserved.

This file is part of the LLM Comparison Tool.

LLM module for the LLM Comparison Tool.
Provides interfaces for different LLM providers.
"""

from abc import ABC, abstractmethod
import os
import json
import httpx
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from src.shared.logger import get_logger
from src.shared.config import OPENAI_API_KEY, OLLAMA_CONFIG, get_config

logger = get_logger(__name__)

class LLMProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    async def generate(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 512,
                 **kwargs) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt to send to the LLM.
            system_prompt: Optional system prompt to control the behavior.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Additional model-specific parameters.
            
        Returns:
            Dictionary containing the response and metadata.
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get a list of available models for this provider.
        
        Returns:
            List of model names.
        """
        pass


class OllamaProvider(LLMProvider):
    """LLM provider using Ollama."""
    
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        """
        Initialize the Ollama provider.
        
        Args:
            host: Host address for the Ollama server.
            port: Port for the Ollama server.
        """
        self.host = host or OLLAMA_CONFIG["host"]
        self.port = port or OLLAMA_CONFIG["port"]
        self.base_url = f"http://{self.host}:{self.port}"
        logger.debug(f"Ollama provider initialized with base URL: {self.base_url}")
        
    async def generate(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 512,
                 model: str = "llama3",
                 **kwargs) -> Dict[str, Any]:
        """
        Generate a response from Ollama.
        
        Args:
            prompt: User prompt to send to the LLM.
            system_prompt: Optional system prompt to control the behavior.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum number of tokens to generate.
            model: Model name to use.
            **kwargs: Additional model-specific parameters.
            
        Returns:
            Dictionary containing the response and metadata.
        """
        url = f"{self.base_url}/api/generate"
        logger.info(f"Generating with URL: {url}")

        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            **kwargs
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        start_time = __import__("time").time()
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        
        end_time = __import__("time").time()
        
        return {
            "text": data["response"],
            "model": model,
            "provider": "ollama",
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            },
            "metadata": {
                "response_time": end_time - start_time,
                "raw_response": data
            }
        }
    
    def get_available_models(self) -> List[str]:
        """
        Get a list of available models in Ollama.
        
        Returns:
            List of model names.
        """
        url = f"{self.base_url}/api/tags"
        
        try:
            response = httpx.get(url)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            print(f"Error getting available models from Ollama: {e}")
            return []


class OpenAIProvider(LLMProvider):
    """LLM provider using OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI provider.
        
        Args:
            api_key: OpenAI API key.
        """
        self.api_key = api_key or OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Please install it with 'pip install openai'.")
    
    async def generate(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 512,
                 model: str = "gpt-3.5-turbo",
                 **kwargs) -> Dict[str, Any]:
        """
        Generate a response from OpenAI.
        
        Args:
            prompt: User prompt to send to the LLM.
            system_prompt: Optional system prompt to control the behavior.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum number of tokens to generate.
            model: Model name to use.
            **kwargs: Additional model-specific parameters.
            
        Returns:
            Dictionary containing the response and metadata.
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        start_time = __import__("time").time()
        
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        end_time = __import__("time").time()
        
        return {
            "text": response.choices[0].message.content,
            "model": model,
            "provider": "openai",
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "metadata": {
                "response_time": end_time - start_time,
                "raw_response": response
            }
        }
    
    def get_available_models(self) -> List[str]:
        """
        Get a list of available models in OpenAI.
        
        Returns:
            List of model names.
        """
        try:
            response = self.client.models.list()
            return [model.id for model in response.data]
        except Exception as e:
            print(f"Error getting available models from OpenAI: {e}")
            return []


def get_llm_provider(provider_name: Optional[str] = None, model: Optional[str] = None) -> LLMProvider:
    """
    Factory function to get the configured LLM provider.
    
    Args:
        provider_name: Optional name of the provider to use. If None, uses the one from config.
        model: Optional model name to determine the provider if provider_name is not specified.
        
    Returns:
        An instance of the appropriate LLMProvider.
    """
    if model and not provider_name:
        # Determine provider from model
        if model.startswith("gpt-") or model in ["text-davinci-003", "text-curie-001"]:
            provider_name = "openai"
        elif model in ["llama2", "llama3", "mistral", "mixtral", "phi3"]:
            provider_name = "ollama"
    
    if not provider_name:
        # Default to the first provider in config
        llm_providers = get_config("llm_providers")
        provider_name = next(iter(llm_providers.keys()), "ollama")
    
    if provider_name == "ollama":
        return OllamaProvider()
    elif provider_name == "openai":
        return OpenAIProvider()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_name}") 