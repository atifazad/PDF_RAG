"""
LLM Client for the PDF RAG application.
Provides a unified interface for different LLM models via Ollama.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import ollama
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate_response(self, prompt: str, context: str = None, **kwargs) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        pass


class OllamaClient(LLMClient):
    """Ollama client implementation."""
    
    def __init__(self, model_name: str = None, host: str = "http://localhost:11434"):
        """
        Initialize the Ollama client.
        
        Args:
            model_name: Name of the model to use (from config if None)
            host: Ollama server host
        """
        self.model_name = model_name or Config.get_model()
        self.host = host
        self.client = ollama.Client(host=host)
        self.temperature = Config.get_temperature()
        self.max_tokens = Config.get_max_tokens()
        
        # Validate model
        if not Config.validate_model(self.model_name):
            logger.warning(f"Model '{self.model_name}' not in supported list. Using default.")
            self.model_name = Config.DEFAULT_MODEL.value
        
        logger.info(f"Initialized Ollama client with model: {self.model_name}")
    
    def is_available(self) -> bool:
        """Check if Ollama service is available."""
        try:
            # Try to list models to check connectivity
            self.client.list()
            return True
        except Exception as e:
            logger.error(f"Ollama service not available: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        try:
            # Get model configuration
            model_config = Config.get_model_config(self.model_name)
            
            # Check if model is available in Ollama
            models = self.client.list()
            model_available = any(model['name'] == self.model_name for model in models['models'])
            
            return {
                "name": self.model_name,
                "available": model_available,
                "config": model_config,
                "host": self.host,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                "name": self.model_name,
                "available": False,
                "error": str(e)
            }
    
    def generate_response(self, prompt: str, context: str = None, **kwargs) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user's question/prompt
            context: Optional context to include in the prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            Generated response text
        """
        try:
            # Build the full prompt
            full_prompt = self._build_prompt(prompt, context)
            
            # Get parameters
            temperature = kwargs.get('temperature', self.temperature)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            
            logger.info(f"Generating response with model: {self.model_name}")
            
            # Generate response
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": full_prompt}],
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            
            response_text = response['message']['content']
            logger.info(f"Generated response (length: {len(response_text)})")
            
            return response_text
            
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            logger.error(error_msg)
            raise LLMError(error_msg) from e
    
    def _build_prompt(self, prompt: str, context: str = None) -> str:
        """
        Build the full prompt with context.
        
        Args:
            prompt: The user's question
            context: Optional context to include
        
        Returns:
            Formatted prompt string
        """
        if context:
            return f"""Context: {context}

Question: {prompt}

Please answer the question based on the provided context. If the context doesn't contain enough information to answer the question, please say so."""
        else:
            return prompt
    
    def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different model.
        
        Args:
            model_name: Name of the new model
        
        Returns:
            True if successful, False otherwise
        """
        if not Config.validate_model(model_name):
            logger.error(f"Invalid model: {model_name}")
            return False
        
        try:
            # Check if the model is available
            models = self.client.list()
            model_available = any(model['name'] == model_name for model in models['models'])
            
            if not model_available:
                logger.warning(f"Model {model_name} not found in Ollama. You may need to pull it first.")
                logger.info(f"Run: ollama pull {model_name}")
                return False
            
            self.model_name = model_name
            logger.info(f"Switched to model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error switching model: {e}")
            return False
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models in Ollama.
        
        Returns:
            List of model information dictionaries
        """
        try:
            models = self.client.list()
            return models['models']
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []


class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass


# Factory function for creating LLM clients
def create_llm_client(model_name: str = None, client_type: str = "ollama") -> LLMClient:
    """
    Factory function to create LLM clients.
    
    Args:
        model_name: Name of the model to use
        client_type: Type of client to create (currently only "ollama")
    
    Returns:
        LLMClient instance
    """
    if client_type.lower() == "ollama":
        return OllamaClient(model_name)
    else:
        raise ValueError(f"Unsupported client type: {client_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test the LLM client
    try:
        client = create_llm_client()
        
        # Check availability
        if client.is_available():
            print("✅ Ollama service is available")
            
            # Get model info
            info = client.get_model_info()
            print(f"📋 Model info: {info}")
            
            # Test response generation
            response = client.generate_response("Hello! Can you help me with document analysis?")
            print(f"🤖 Response: {response}")
            
        else:
            print("❌ Ollama service is not available")
            print("Please make sure Ollama is running and the model is downloaded.")
            
    except Exception as e:
        print(f"❌ Error: {e}") 