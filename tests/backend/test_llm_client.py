"""
Tests for the LLM client.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from llm_client import OllamaClient, LLMError, create_llm_client
from config import Config


class TestOllamaClient(unittest.TestCase):
    """Test cases for the OllamaClient."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear any existing environment variables
        self.env_vars_to_clear = ['LLM_MODEL', 'LLM_TEMPERATURE', 'LLM_MAX_TOKENS']
        for var in self.env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]
    
    @patch('llm_client.ollama.Client')
    def test_client_initialization(self, mock_client):
        """Test client initialization with default model."""
        # Mock the client
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        client = OllamaClient()
        
        self.assertEqual(client.model_name, Config.DEFAULT_MODEL.value)
        self.assertEqual(client.host, "http://localhost:11434")
        self.assertEqual(client.temperature, Config.get_temperature())
        self.assertEqual(client.max_tokens, Config.get_max_tokens())
    
    @patch('llm_client.ollama.Client')
    def test_client_initialization_with_custom_model(self, mock_client):
        """Test client initialization with custom model."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        client = OllamaClient(model_name="mistral:7b-instruct")
        
        self.assertEqual(client.model_name, "mistral:7b-instruct")
    
    @patch('llm_client.ollama.Client')
    def test_client_initialization_with_invalid_model(self, mock_client):
        """Test client initialization with invalid model."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        client = OllamaClient(model_name="invalid-model")
        
        # Should fall back to default model
        self.assertEqual(client.model_name, Config.DEFAULT_MODEL.value)
    
    @patch('llm_client.ollama.Client')
    def test_is_available_success(self, mock_client):
        """Test is_available when Ollama is running."""
        mock_client_instance = Mock()
        mock_client_instance.list.return_value = {"models": []}
        mock_client.return_value = mock_client_instance
        
        client = OllamaClient()
        result = client.is_available()
        
        self.assertTrue(result)
        mock_client_instance.list.assert_called_once()
    
    @patch('llm_client.ollama.Client')
    def test_is_available_failure(self, mock_client):
        """Test is_available when Ollama is not running."""
        mock_client_instance = Mock()
        mock_client_instance.list.side_effect = Exception("Connection failed")
        mock_client.return_value = mock_client_instance
        
        client = OllamaClient()
        result = client.is_available()
        
        self.assertFalse(result)
    
    @patch('llm_client.ollama.Client')
    def test_get_model_info_success(self, mock_client):
        """Test get_model_info when successful."""
        mock_client_instance = Mock()
        mock_client_instance.list.return_value = {
            "models": [{"name": "llama2:13b-chat"}]
        }
        mock_client.return_value = mock_client_instance
        
        client = OllamaClient()
        info = client.get_model_info()
        
        self.assertEqual(info["name"], "llama2:13b-chat")
        self.assertTrue(info["available"])
        self.assertIn("config", info)
        self.assertEqual(info["host"], "http://localhost:11434")
    
    @patch('llm_client.ollama.Client')
    def test_get_model_info_model_not_available(self, mock_client):
        """Test get_model_info when model is not available in Ollama."""
        mock_client_instance = Mock()
        mock_client_instance.list.return_value = {
            "models": [{"name": "other-model"}]
        }
        mock_client.return_value = mock_client_instance
        
        client = OllamaClient()
        info = client.get_model_info()
        
        self.assertEqual(info["name"], "llama2:13b-chat")
        self.assertFalse(info["available"])
    
    @patch('llm_client.ollama.Client')
    def test_generate_response_success(self, mock_client):
        """Test generate_response when successful."""
        mock_client_instance = Mock()
        mock_client_instance.chat.return_value = {
            "message": {"content": "Hello! I can help you with document analysis."}
        }
        mock_client.return_value = mock_client_instance
        
        client = OllamaClient()
        response = client.generate_response("Hello!")
        
        self.assertEqual(response, "Hello! I can help you with document analysis.")
        mock_client_instance.chat.assert_called_once()
    
    @patch('llm_client.ollama.Client')
    def test_generate_response_with_context(self, mock_client):
        """Test generate_response with context."""
        mock_client_instance = Mock()
        mock_client_instance.chat.return_value = {
            "message": {"content": "Based on the context, the answer is..."}
        }
        mock_client.return_value = mock_client_instance
        
        client = OllamaClient()
        response = client.generate_response(
            "What is the main topic?",
            context="This document discusses machine learning algorithms."
        )
        
        self.assertEqual(response, "Based on the context, the answer is...")
        
        # Check that the prompt was built correctly
        call_args = mock_client_instance.chat.call_args
        messages = call_args[1]["messages"]
        self.assertIn("Context: This document discusses machine learning algorithms.", messages[0]["content"])
        self.assertIn("Question: What is the main topic?", messages[0]["content"])
    
    @patch('llm_client.ollama.Client')
    def test_generate_response_failure(self, mock_client):
        """Test generate_response when it fails."""
        mock_client_instance = Mock()
        mock_client_instance.chat.side_effect = Exception("Model not found")
        mock_client.return_value = mock_client_instance
        
        client = OllamaClient()
        
        with self.assertRaises(LLMError):
            client.generate_response("Hello!")
    
    @patch('llm_client.ollama.Client')
    def test_switch_model_success(self, mock_client):
        """Test switch_model when successful."""
        mock_client_instance = Mock()
        mock_client_instance.list.return_value = {
            "models": [
                {"name": "llama2:13b-chat"},
                {"name": "mistral:7b-instruct"}
            ]
        }
        mock_client.return_value = mock_client_instance
        
        client = OllamaClient()
        result = client.switch_model("mistral:7b-instruct")
        
        self.assertTrue(result)
        self.assertEqual(client.model_name, "mistral:7b-instruct")
    
    @patch('llm_client.ollama.Client')
    def test_switch_model_invalid_model(self, mock_client):
        """Test switch_model with invalid model."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        client = OllamaClient()
        result = client.switch_model("invalid-model")
        
        self.assertFalse(result)
        self.assertEqual(client.model_name, "llama2:13b-chat")  # Should not change
    
    @patch('llm_client.ollama.Client')
    def test_switch_model_not_available(self, mock_client):
        """Test switch_model when model is not available in Ollama."""
        mock_client_instance = Mock()
        mock_client_instance.list.return_value = {
            "models": [{"name": "llama2:13b-chat"}]
        }
        mock_client.return_value = mock_client_instance
        
        client = OllamaClient()
        result = client.switch_model("mistral:7b-instruct")
        
        self.assertFalse(result)
        self.assertEqual(client.model_name, "llama2:13b-chat")  # Should not change
    
    @patch('llm_client.ollama.Client')
    def test_list_available_models(self, mock_client):
        """Test list_available_models."""
        mock_client_instance = Mock()
        mock_client_instance.list.return_value = {
            "models": [
                {"name": "llama2:13b-chat"},
                {"name": "mistral:7b-instruct"}
            ]
        }
        mock_client.return_value = mock_client_instance
        
        client = OllamaClient()
        models = client.list_available_models()
        
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["name"], "llama2:13b-chat")
        self.assertEqual(models[1]["name"], "mistral:7b-instruct")
    
    @patch('llm_client.ollama.Client')
    def test_list_available_models_failure(self, mock_client):
        """Test list_available_models when it fails."""
        mock_client_instance = Mock()
        mock_client_instance.list.side_effect = Exception("Connection failed")
        mock_client.return_value = mock_client_instance
        
        client = OllamaClient()
        models = client.list_available_models()
        
        self.assertEqual(models, [])


class TestLLMClientFactory(unittest.TestCase):
    """Test cases for the LLM client factory."""
    
    def test_create_ollama_client(self):
        """Test creating an Ollama client."""
        client = create_llm_client(client_type="ollama")
        self.assertIsInstance(client, OllamaClient)
    
    def test_create_ollama_client_with_model(self):
        """Test creating an Ollama client with specific model."""
        client = create_llm_client("mistral:7b-instruct", "ollama")
        self.assertIsInstance(client, OllamaClient)
        self.assertEqual(client.model_name, "mistral:7b-instruct")
    
    def test_create_unsupported_client(self):
        """Test creating an unsupported client type."""
        with self.assertRaises(ValueError):
            create_llm_client(client_type="unsupported")


if __name__ == '__main__':
    unittest.main() 