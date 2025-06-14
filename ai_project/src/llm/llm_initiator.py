from langchain.llms import OpenAI, HuggingFaceHub
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from typing import List, Dict, Optional, Union, Any
import os
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from .model_locator import ModelLocation, RemoteModelLocation, LocalModelLocation
from langchain_huggingface import HuggingFaceEndpoint

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, model_location: ModelLocation):
        self.model_location = model_location

    @abstractmethod
    def get_llm(self):
        pass

    @abstractmethod
    def get_embeddings(self):
        pass

# Open AI Models
class OpenAIProvider(BaseLLMProvider):
    """OpenAI implementation - Always remote"""

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        super().__init__(RemoteModelLocation())  # OpenAI is always remote
        self.model_name = model_name
        self.temperature = temperature

    def get_llm(self):
        return OpenAI(
            temperature=self.temperature,
            model_name=self.model_name
        )

    def get_embeddings(self):
        return OpenAIEmbeddings()

# Hugging Face Models
class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace implementation with remote/local options"""

    def __init__(self,
                 model_name: str = "google/flan-t5-base",
                 api_token: Optional[str] = None,
                 model_location: Optional[ModelLocation] = None):
        super().__init__(model_location or RemoteModelLocation())
        self.model_name = model_name
        self.api_token=api_token

    def get_llm(self):
        model_config = self.model_location.get_model(self.model_name)

        if model_config["location"] == "remote":
            return HuggingFaceHub(
                repo_id=self.model_name,
                huggingfacehub_api_token=self.api_token,
                model_kwargs={"temperature": 0.7}
            )
        else:
            # For API-only usage without downloading
            return HuggingFaceHub(
                repo_id=self.model_name,
                huggingfacehub_api_token=self.api_token
            )

    def get_embeddings(self):
        model_config = self.model_location.get_model(self.model_name)

        if model_config["location"] == "remote":
            return HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"use_api": True}
            )
        else:
            return HuggingFaceEmbeddings(
                model_name=self.model_name,
                cache_folder=model_config.get("cache_dir")
            )

def initialize_llm(
    model_name: str,
    api_token: Optional[str] = None,
    temperature: float = 0.7,
    cache_dir: Optional[str] = None
):
    """Initializes the appropriate LLM provider based on model name and configuration."""
    if "gpt" in model_name.lower():
        return OpenAIProvider(model_name=model_name, temperature=temperature)
    elif "flan" in model_name.lower() or "qwen" in model_name.lower():
        model_location = LocalModelLocation(cache_dir=cache_dir) if cache_dir else RemoteModelLocation()
        return HuggingFaceProvider(model_name=model_name, api_token=api_token, model_location=model_location)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
