"""
Tests for the configuration system.
"""

import os
import sys
import unittest
from unittest.mock import patch

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from config import Config, ModelType, validate_configuration


class TestConfig(unittest.TestCase):
    """Test cases for the configuration system."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear any existing environment variables
        self.env_vars_to_clear = [
            'LLM_MODEL', 'LLM_TEMPERATURE', 'LLM_MAX_TOKENS',
            'CHROMA_PERSIST_DIRECTORY', 'EMBEDDING_MODEL',
            'CHUNK_SIZE', 'CHUNK_OVERLAP', 'MAX_FILE_SIZE_MB',
            'API_HOST', 'API_PORT', 'STREAMLIT_PORT', 'DEBUG', 'LOG_LEVEL'
        ]
        
        for var in self.env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]
    
    def test_default_model(self):
        """Test default model configuration."""
        model = Config.get_model()
        self.assertEqual(model, ModelType.LLAMA2_13B.value)
    
    def test_model_validation(self):
        """Test model validation."""
        # Valid models
        self.assertTrue(Config.validate_model("llama2:13b-chat"))
        self.assertTrue(Config.validate_model("mistral:7b-instruct"))
        
        # Invalid models
        self.assertFalse(Config.validate_model("invalid-model"))
        self.assertFalse(Config.validate_model(""))
    
    def test_model_config(self):
        """Test getting model configuration."""
        config = Config.get_model_config("llama2:13b-chat")
        self.assertEqual(config["name"], "Llama2-13B-Chat")
        self.assertEqual(config["memory_usage"], "16GB")
        self.assertEqual(config["quality"], "Excellent")
    
    def test_available_models(self):
        """Test getting all available models."""
        models = Config.get_available_models()
        self.assertIn("llama2:13b-chat", models)
        self.assertIn("mistral:7b-instruct", models)
        self.assertIn("llama2:7b-chat", models)
        self.assertIn("codellama:7b-instruct", models)
    
    def test_environment_variables(self):
        """Test environment variable configuration."""
        # Set environment variables
        os.environ['LLM_MODEL'] = 'mistral:7b-instruct'
        os.environ['LLM_TEMPERATURE'] = '0.5'
        os.environ['LLM_MAX_TOKENS'] = '1024'
        os.environ['CHUNK_SIZE'] = '500'
        os.environ['DEBUG'] = 'true'
        
        # Test that they are read correctly
        self.assertEqual(Config.get_model(), 'mistral:7b-instruct')
        self.assertEqual(Config.get_temperature(), 0.5)
        self.assertEqual(Config.get_max_tokens(), 1024)
        self.assertEqual(Config.get_chunk_size(), 500)
        self.assertTrue(Config.is_debug())
    
    def test_default_values(self):
        """Test default values when environment variables are not set."""
        self.assertEqual(Config.get_temperature(), 0.7)
        self.assertEqual(Config.get_max_tokens(), 2048)
        self.assertEqual(Config.get_chunk_size(), 1000)
        self.assertEqual(Config.get_chunk_overlap(), 200)
        self.assertEqual(Config.get_max_file_size_mb(), 50)
        self.assertEqual(Config.get_api_host(), "0.0.0.0")
        self.assertEqual(Config.get_api_port(), 8000)
        self.assertEqual(Config.get_streamlit_port(), 8501)
        self.assertFalse(Config.is_debug())
        self.assertEqual(Config.get_log_level(), "INFO")
    
    def test_model_info(self):
        """Test getting model information."""
        info = Config.get_model_info("llama2:13b-chat")
        self.assertIsNotNone(info)
        self.assertEqual(info["name"], "Llama2-13B-Chat")
        
        # Test with invalid model
        info = Config.get_model_info("invalid-model")
        self.assertIsNone(info)
    
    def test_validate_configuration(self):
        """Test configuration validation."""
        # Test with valid configuration
        self.assertTrue(validate_configuration())
        
        # Test with invalid model
        with patch.dict(os.environ, {'LLM_MODEL': 'invalid-model'}):
            self.assertFalse(validate_configuration())


if __name__ == '__main__':
    unittest.main() 