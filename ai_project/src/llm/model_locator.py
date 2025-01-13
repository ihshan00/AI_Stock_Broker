from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any


class ModelLocation(ABC):
    """Strategy for model location handling"""
    @abstractmethod
    def get_model(self, model_name: str):
        pass

class RemoteModelLocation(ModelLocation):
    """Handles remote API-based models"""
    def get_model(self, model_name: str):
        return {"location": "remote", "model_name": model_name}

class LocalModelLocation(ModelLocation):
    """Handles local model loading"""
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = "data/model"

    def get_model(self, model_name: str):
        return {
            "location": "local",
            "model_name": model_name,
            "cache_dir": self.cache_dir
        }