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
            
            # Handle potential extra data in JSON response
            try:
                content = response.text
                # Log a sample of the response content (first 100 chars) for debugging
                content_preview = content[:100] + "..." if len(content) > 100 else content
                logger.debug(f"Raw response from {model} (sample): {content_preview}")
                data = self._parse_json_safely(content)
            except Exception as e:
                logger.error(f"Error parsing JSON from Ollama: {e}")
                # Log full response when parsing fails to help diagnose the issue
                logger.debug(f"Failed to parse response from {model}. Full content: {content}")
                # Try a more permissive parsing approach for models like Qwen
                try:
                    import json
                    # Try to extract just the first valid JSON object
                    first_json = content.strip().split('\n')[0]
                    data = json.loads(first_json)
                    logger.warning(f"Used fallback JSON parsing for model {model}")
                except Exception as nested_e:
                    logger.error(f"Fallback parsing also failed: {nested_e}")
                    # Special handling for Qwen3:30b-a3b and similar models with extra data
                    if "Qwen" in model:
                        try:
                            # Use the specialized Qwen parser
                            data = self._parse_qwen_response(content)
                            logger.warning(f"Used specialized Qwen parser for model {model}")
                        except Exception as qwen_e:
                            logger.error(f"Qwen-specific parsing also failed: {qwen_e}")
                            raise ValueError(f"Could not parse Ollama response for model {model}: {e}")
                    else:
                        raise ValueError(f"Could not parse Ollama response for model {model}: {e}")
        
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
    
    def _parse_json_safely(self, content: str) -> Dict[str, Any]:
        """
        Safely parse JSON content, handling various edge cases.
        
        Some models (like Qwen) may return JSON with extra data or multiple objects.
        This method attempts to handle those cases.
        
        Args:
            content: The JSON string to parse
            
        Returns:
            Parsed JSON data
        """
        import json
        
        # First try simple parsing
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"Standard JSON parsing failed: {e}")
            
            # Check if there might be multiple JSON objects
            if '\n' in content:
                # Try to parse just the first line
                first_line = content.split('\n')[0].strip()
                if first_line.endswith('}'):
                    try:
                        return json.loads(first_line)
                    except:
                        pass
            
            # Try finding the closing brace of the first object
            try:
                # Find the position of the first opening brace
                start = content.find('{')
                if start != -1:
                    # Track nested braces to find the matching closing brace
                    depth = 0
                    for i, char in enumerate(content[start:]):
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                # We found the closing brace of the outermost object
                                end = start + i + 1
                                try:
                                    return json.loads(content[start:end])
                                except json.JSONDecodeError:
                                    # If there are escape characters or invalid JSON, try to fix common issues
                                    # This is especially important for models like Qwen that may have formatting quirks
                                    clean_json = content[start:end].replace('\n', ' ').replace('\r', '')
                                    return json.loads(clean_json)
            except:
                pass
            
            # Special handling for Qwen models which are known to have extra data issues
            if "qwen" in content.lower():
                try:
                    # Try a more aggressive approach - find the first valid JSON object
                    brace_count = 0
                    in_quotes = False
                    escape = False
                    start_pos = content.find('{')
                    
                    if start_pos >= 0:
                        for i, char in enumerate(content[start_pos:], start_pos):
                            if escape:
                                escape = False
                                continue
                                
                            if char == '\\':
                                escape = True
                            elif char == '"' and not escape:
                                in_quotes = not in_quotes
                            elif not in_quotes:
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        # We found a complete JSON object
                                        try:
                                            return json.loads(content[start_pos:i+1])
                                        except:
                                            # Last attempt - try to clean the string
                                            clean_text = content[start_pos:i+1].replace('\n', ' ').replace('\r', '')
                                            return json.loads(clean_text)
                except:
                    pass
                
            # If all else fails, re-raise the original error
            raise
    
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
            content = response.text
            
            # Use the safe JSON parser
            data = self._parse_json_safely(content)
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.error(f"Error getting available models from Ollama: {e}")
            return []
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific Ollama model.
        
        Args:
            model_name: Name of the model to get information for.
            
        Returns:
            Dictionary with model information including description, parameters, etc.
        """
        url = f"{self.base_url}/api/show"
        
        try:
            response = httpx.post(url, json={"name": model_name})
            response.raise_for_status()
            content = response.text
            
            # Use the safe JSON parser
            data = self._parse_json_safely(content)
            
            # Extract relevant information in a consistent format
            model_info = {
                "name": model_name,
                "description": data.get("system", "No description available"),
                "parameter_count": self._extract_parameter_count(data.get("system", "")),
                "context_length": data.get("context_length", 4096),
                "model_size": data.get("size", "Unknown"),
                "modified_at": data.get("modified_at", ""),
                "tags": [t.strip() for t in data.get("tags", "").split(",")] if data.get("tags") else []
            }
            
            return model_info
        except Exception as e:
            logger.error(f"Error getting model info from Ollama for {model_name}: {e}")
            return {
                "name": model_name,
                "description": "No description available",
                "parameter_count": "Unknown",
                "context_length": 4096,
                "tags": []
            }
    
    def _extract_parameter_count(self, description: str) -> str:
        """Extract parameter count from model description if available."""
        import re
        # Try to find something like "7B" or "70B" in the description
        match = re.search(r'(\d+)B', description)
        if match:
            return f"{match.group(1)} billion"
        return "Unknown"

    def _parse_qwen_response(self, content: str) -> Dict[str, Any]:
        """
        Special parser for Qwen model responses which often have formatting issues.
        
        The Qwen3:30b-a3b model in particular is known to return valid JSON followed by
        extra data on line 2 (char 113), causing standard JSON parsers to fail.
        
        Args:
            content: Raw response text from Qwen model
            
        Returns:
            Parsed JSON data
        """
        import json
        import re
        
        logger.debug("Using specialized Qwen response parser")
        
        # Method 1: Try to extract just the first valid JSON object by finding matching braces
        try:
            # Find the position of the first opening brace
            start = content.find('{')
            if start != -1:
                # Track nested braces to find the matching closing brace
                depth = 0
                for i, char in enumerate(content[start:]):
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            # Found the closing brace of the first JSON object
                            json_str = content[start:start+i+1]
                            logger.debug(f"Found JSON object with length {len(json_str)}")
                            return json.loads(json_str)
        except Exception as e:
            logger.debug(f"Method 1 failed: {e}")
        
        # Method 2: Try to extract just the part before the error point (char 113)
        try:
            first_part = content[:113].strip()
            # Ensure it ends with a closing brace
            if first_part.endswith('}'):
                logger.debug("Using first 113 characters that end with '}'")
                return json.loads(first_part)
        except Exception as e:
            logger.debug(f"Method 2 failed: {e}")
        
        # Method 3: Try to find the first complete JSON object using regex
        try:
            # Python's re module doesn't support recursion (?R), so use a simpler approach
            # Try to find a pattern that looks like a JSON object with reasonable constraints
            json_pattern = re.compile(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})')
            match = json_pattern.search(content)
            if match:
                logger.debug("Found JSON using regex")
                return json.loads(match.group(0))
        except Exception as e:
            logger.debug(f"Method 3 failed: {e}")
        
        # Method 4: Last resort - try with a simplistic approach
        try:
            # Find the first { and the next } after skipping nested braces
            open_index = content.find('{')
            if open_index >= 0:
                # Try to find the matching closing brace
                nested = 0
                for i in range(open_index + 1, len(content)):
                    if content[i] == '{':
                        nested += 1
                    elif content[i] == '}':
                        if nested == 0:
                            # This is the matching closing brace
                            json_str = content[open_index:i+1]
                            logger.debug(f"Last resort method found JSON with length {len(json_str)}")
                            return json.loads(json_str)
                        nested -= 1
        except Exception as e:
            logger.debug(f"Method 4 failed: {e}")
        
        # If all methods fail, raise an informative error
        raise ValueError("Could not parse Qwen response with any method. The response format may have changed.")


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