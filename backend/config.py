"""
Configuration management for the PDF RAG application.
Handles model configurations, environment variables, and application settings.
"""

import os
from enum import Enum
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ModelType(Enum):
    """Available LLM models."""
    LLAMA2_13B = "llama2:13b-chat"
    LLAMA2_7B = "llama2:7b-chat"
    MISTRAL_7B = "mistral:7b-instruct"
    CODELLAMA_7B = "codellama:7b-instruct"


class Config:
    """Application configuration class."""
    
    # Default model
    DEFAULT_MODEL = ModelType.LLAMA2_13B
    
    # Model configurations
    MODELS = {
        ModelType.LLAMA2_13B: {
            "name": "Llama2-13B-Chat",
            "description": "High-quality responses, 16GB memory",
            "context_window": 4096,
            "max_tokens": 2048,
            "temperature": 0.7,
            "memory_usage": "16GB",
            "speed": "Medium",
            "quality": "Excellent"
        },
        ModelType.LLAMA2_7B: {
            "name": "Llama2-7B-Chat",
            "description": "Balanced performance, 8GB memory",
            "context_window": 4096,
            "max_tokens": 2048,
            "temperature": 0.7,
            "memory_usage": "8GB",
            "speed": "Fast",
            "quality": "Good"
        },
        ModelType.MISTRAL_7B: {
            "name": "Mistral-7B-Instruct",
            "description": "Fast responses, excellent instruction following",
            "context_window": 8192,
            "max_tokens": 2048,
            "temperature": 0.7,
            "memory_usage": "8GB",
            "speed": "Fast",
            "quality": "Good"
        },
        ModelType.CODELLAMA_7B: {
            "name": "CodeLlama-7B-Instruct",
            "description": "Specialized for code and technical documents",
            "context_window": 4096,
            "max_tokens": 2048,
            "temperature": 0.7,
            "memory_usage": "8GB",
            "speed": "Fast",
            "quality": "Good"
        }
    }
    
    @classmethod
    def get_model(cls) -> str:
        """Get the current model from environment variable."""
        return os.getenv("LLM_MODEL", cls.DEFAULT_MODEL.value)
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        for model_type, config in cls.MODELS.items():
            if model_type.value == model_name:
                return config
        return cls.MODELS[cls.DEFAULT_MODEL]
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get all available models with their configurations."""
        return {model.value: config for model, config in cls.MODELS.items()}
    
    @classmethod
    def get_temperature(cls) -> float:
        """Get temperature setting from environment."""
        return float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    @classmethod
    def get_max_tokens(cls) -> int:
        """Get max tokens setting from environment."""
        return int(os.getenv("LLM_MAX_TOKENS", "2048"))
    
    @classmethod
    def get_chroma_persist_directory(cls) -> str:
        """Get Chroma DB persistence directory."""
        return os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    
    @classmethod
    def get_embedding_model(cls) -> str:
        """Get embedding model name."""
        return os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    @classmethod
    def get_chunk_size(cls) -> int:
        """Get document chunk size."""
        return int(os.getenv("CHUNK_SIZE", "1000"))
    
    @classmethod
    def get_chunk_overlap(cls) -> int:
        """Get document chunk overlap."""
        return int(os.getenv("CHUNK_OVERLAP", "200"))
    
    @classmethod
    def get_max_file_size_mb(cls) -> int:
        """Get maximum file size in MB."""
        return int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    
    @classmethod
    def get_api_host(cls) -> str:
        """Get API host."""
        return os.getenv("API_HOST", "0.0.0.0")
    
    @classmethod
    def get_api_port(cls) -> int:
        """Get API port."""
        return int(os.getenv("API_PORT", "8000"))
    
    @classmethod
    def get_streamlit_port(cls) -> int:
        """Get Streamlit port."""
        return int(os.getenv("STREAMLIT_PORT", "8501"))
    
    @classmethod
    def is_debug(cls) -> bool:
        """Check if debug mode is enabled."""
        return os.getenv("DEBUG", "False").lower() == "true"
    
    @classmethod
    def get_log_level(cls) -> str:
        """Get log level."""
        return os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate_model(cls, model_name: str) -> bool:
        """Validate if a model name is supported."""
        return model_name in [model.value for model in ModelType]
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model."""
        if cls.validate_model(model_name):
            return cls.get_model_config(model_name)
        return None


# Configuration validation
def validate_configuration():
    """Validate the current configuration."""
    current_model = Config.get_model()
    if not Config.validate_model(current_model):
        print(f"Warning: Model '{current_model}' is not in the supported list.")
        print(f"Using default model: {Config.DEFAULT_MODEL.value}")
        return False
    return True


if __name__ == "__main__":
    # Test configuration
    print("Current Configuration:")
    print(f"Model: {Config.get_model()}")
    print(f"Temperature: {Config.get_temperature()}")
    print(f"Max Tokens: {Config.get_max_tokens()}")
    print(f"Chunk Size: {Config.get_chunk_size()}")
    print(f"Debug Mode: {Config.is_debug()}")
    
    print("\nAvailable Models:")
    for model_name, config in Config.get_available_models().items():
        print(f"- {model_name}: {config['name']} ({config['memory_usage']})")
    
    validate_configuration() 