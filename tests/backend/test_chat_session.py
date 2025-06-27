"""
Tests for the chat session functionality.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from chat_session import ChatSession, ChatMessage, PromptTemplate, LLMError


class TestPromptTemplate(unittest.TestCase):
    """Test cases for prompt templates."""
    
    def test_general_chat_template(self):
        """Test general chat prompt template."""
        prompt = PromptTemplate.general_chat("What is AI?")
        self.assertIn("What is AI?", prompt)
        self.assertIn("helpful AI assistant", prompt)
        self.assertIn("conversational manner", prompt)
    
    def test_rag_question_template(self):
        """Test RAG question prompt template."""
        context = "AI is artificial intelligence."
        question = "What is AI?"
        prompt = PromptTemplate.rag_question(question, context)
        
        self.assertIn(context, prompt)
        self.assertIn(question, prompt)
        self.assertIn("Answer based ONLY on the provided context", prompt)
        self.assertIn("If the context doesn't contain enough information", prompt)
    
    def test_document_analysis_template(self):
        """Test document analysis prompt template."""
        context = "This document discusses machine learning."
        question = "What is the main topic?"
        prompt = PromptTemplate.document_analysis(question, context)
        
        self.assertIn(context, prompt)
        self.assertIn(question, prompt)
        self.assertIn("expert document analyst", prompt)
    
    def test_code_analysis_template(self):
        """Test code analysis prompt template."""
        context = "def hello(): print('Hello')"
        question = "What does this code do?"
        prompt = PromptTemplate.code_analysis(question, context)
        
        self.assertIn(context, prompt)
        self.assertIn(question, prompt)
        self.assertIn("expert code analyst", prompt)


class TestChatMessage(unittest.TestCase):
    """Test cases for chat messages."""
    
    def test_chat_message_creation(self):
        """Test creating a chat message."""
        message = ChatMessage("user", "Hello!")
        
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "Hello!")
        self.assertIsInstance(message.timestamp, datetime)
        self.assertEqual(message.metadata, {})
    
    def test_chat_message_with_metadata(self):
        """Test creating a chat message with metadata."""
        metadata = {"prompt_type": "rag", "context": "test"}
        message = ChatMessage("assistant", "Response", metadata=metadata)
        
        self.assertEqual(message.role, "assistant")
        self.assertEqual(message.content, "Response")
        self.assertEqual(message.metadata, metadata)
    
    def test_chat_message_to_dict(self):
        """Test converting message to dictionary."""
        message = ChatMessage("user", "Hello!")
        message_dict = message.to_dict()
        
        self.assertEqual(message_dict["role"], "user")
        self.assertEqual(message_dict["content"], "Hello!")
        self.assertIn("timestamp", message_dict)
        self.assertIn("metadata", message_dict)
    
    def test_chat_message_string_representation(self):
        """Test string representation of message."""
        message = ChatMessage("user", "Hello!")
        string_repr = str(message)
        
        self.assertIn("USER", string_repr)
        self.assertIn("Hello!", string_repr)


class TestChatSession(unittest.TestCase):
    """Test cases for chat session."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear any existing environment variables
        self.env_vars_to_clear = ['LLM_MODEL', 'LLM_TEMPERATURE', 'LLM_MAX_TOKENS']
        for var in self.env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]
    
    @patch('chat_session.create_llm_client')
    def test_chat_session_initialization(self, mock_create_client):
        """Test chat session initialization."""
        # Mock the LLM client
        mock_client = Mock()
        mock_client.model_name = "llama2:13b-chat"
        mock_create_client.return_value = mock_client
        
        chat = ChatSession()
        
        self.assertEqual(chat.model_name, "llama2:13b-chat")
        self.assertEqual(chat.max_history, 50)
        self.assertEqual(len(chat.messages), 0)
        self.assertIsNotNone(chat.session_id)
    
    @patch('chat_session.create_llm_client')
    def test_chat_session_with_custom_model(self, mock_create_client):
        """Test chat session with custom model."""
        mock_client = Mock()
        mock_client.model_name = "mistral:7b-instruct"
        mock_create_client.return_value = mock_client
        
        chat = ChatSession(model_name="mistral:7b-instruct")
        
        self.assertEqual(chat.model_name, "mistral:7b-instruct")
    
    @patch('chat_session.create_llm_client')
    def test_ask_general_question(self, mock_create_client):
        """Test asking a general question."""
        # Mock the LLM client
        mock_client = Mock()
        mock_client.model_name = "llama2:13b-chat"
        mock_client.generate_response.return_value = "Hello! I'm doing well, thank you for asking."
        mock_create_client.return_value = mock_client
        
        chat = ChatSession()
        response = chat.ask("Hello! How are you?")
        
        self.assertEqual(response, "Hello! I'm doing well, thank you for asking.")
        self.assertEqual(len(chat.messages), 2)  # User message + assistant response
        
        # Check user message
        user_message = chat.messages[0]
        self.assertEqual(user_message.role, "user")
        self.assertEqual(user_message.content, "Hello! How are you?")
        
        # Check assistant message
        assistant_message = chat.messages[1]
        self.assertEqual(assistant_message.role, "assistant")
        self.assertEqual(assistant_message.content, "Hello! I'm doing well, thank you for asking.")
    
    @patch('chat_session.create_llm_client')
    def test_ask_rag_question(self, mock_create_client):
        """Test asking a RAG-style question."""
        mock_client = Mock()
        mock_client.model_name = "llama2:13b-chat"
        mock_client.generate_response.return_value = "Based on the context, AI is artificial intelligence."
        mock_create_client.return_value = mock_client
        
        chat = ChatSession()
        context = "AI is artificial intelligence."
        response = chat.ask("What is AI?", context=context, prompt_type="rag")
        
        self.assertEqual(response, "Based on the context, AI is artificial intelligence.")
        
        # Check that context was passed to the client
        mock_client.generate_response.assert_called_once()
        call_args = mock_client.generate_response.call_args
        self.assertEqual(call_args[1]["context"], context)
    
    @patch('chat_session.create_llm_client')
    def test_ask_with_llm_error(self, mock_create_client):
        """Test handling LLM errors."""
        mock_client = Mock()
        mock_client.model_name = "llama2:13b-chat"
        mock_client.generate_response.side_effect = LLMError("Model not found")
        mock_create_client.return_value = mock_client
        
        chat = ChatSession()
        response = chat.ask("Hello!")
        
        self.assertIn("I apologize, but I encountered an error", response)
        self.assertEqual(len(chat.messages), 2)  # User message + error response
        
        # Check error message metadata
        error_message = chat.messages[1]
        self.assertTrue(error_message.metadata.get("error", False))
    
    @patch('chat_session.create_llm_client')
    def test_switch_model(self, mock_create_client):
        """Test switching models."""
        mock_client = Mock()
        mock_client.model_name = "llama2:13b-chat"
        mock_client.switch_model.return_value = True
        mock_create_client.return_value = mock_client
        
        chat = ChatSession()
        success = chat.switch_model("mistral:7b-instruct")
        
        self.assertTrue(success)
        self.assertEqual(chat.model_name, "mistral:7b-instruct")
        mock_client.switch_model.assert_called_once_with("mistral:7b-instruct")
    
    @patch('chat_session.create_llm_client')
    def test_switch_model_failure(self, mock_create_client):
        """Test model switching failure."""
        mock_client = Mock()
        mock_client.model_name = "llama2:13b-chat"
        mock_client.switch_model.return_value = False
        mock_create_client.return_value = mock_client
        
        chat = ChatSession()
        success = chat.switch_model("invalid-model")
        
        self.assertFalse(success)
        self.assertEqual(chat.model_name, "llama2:13b-chat")  # Should not change
    
    @patch('chat_session.create_llm_client')
    def test_get_history(self, mock_create_client):
        """Test getting conversation history."""
        mock_client = Mock()
        mock_client.model_name = "llama2:13b-chat"
        mock_client.generate_response.return_value = "Response"
        mock_create_client.return_value = mock_client
        
        chat = ChatSession()
        chat.ask("Hello!")
        
        history = chat.get_history()
        
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "Hello!")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["content"], "Response")
    
    @patch('chat_session.create_llm_client')
    def test_clear_history(self, mock_create_client):
        """Test clearing conversation history."""
        mock_client = Mock()
        mock_client.model_name = "llama2:13b-chat"
        mock_client.generate_response.return_value = "Response"
        mock_create_client.return_value = mock_client
        
        chat = ChatSession()
        chat.ask("Hello!")
        
        self.assertEqual(len(chat.messages), 2)
        
        chat.clear_history()
        
        self.assertEqual(len(chat.messages), 0)
    
    @patch('chat_session.create_llm_client')
    def test_get_session_info(self, mock_create_client):
        """Test getting session information."""
        mock_client = Mock()
        mock_client.model_name = "llama2:13b-chat"
        mock_create_client.return_value = mock_client
        
        chat = ChatSession()
        info = chat.get_session_info()
        
        self.assertEqual(info["model_name"], "llama2:13b-chat")
        self.assertEqual(info["message_count"], 0)
        self.assertIsNotNone(info["session_id"])
        self.assertIsNone(info["created_at"])  # No messages yet
        self.assertIsNone(info["last_activity"])  # No messages yet
    
    @patch('chat_session.create_llm_client')
    def test_format_response(self, mock_create_client):
        """Test response formatting."""
        mock_client = Mock()
        mock_client.model_name = "llama2:13b-chat"
        mock_client.generate_response.return_value = "Answer: This is the response."
        mock_create_client.return_value = mock_client
        
        chat = ChatSession()
        response = chat.ask("Test question")
        
        # Should remove "Answer:" prefix
        self.assertEqual(response, "This is the response.")
    
    @patch('chat_session.create_llm_client')
    def test_format_rag_response(self, mock_create_client):
        """Test RAG response formatting."""
        mock_client = Mock()
        mock_client.model_name = "llama2:13b-chat"
        mock_client.generate_response.return_value = "This is the answer."
        mock_create_client.return_value = mock_client
        
        chat = ChatSession()
        response = chat.ask("Test question", context="test context", prompt_type="rag")
        
        # Should add "Based on the provided context:" prefix
        self.assertEqual(response, "Based on the provided context: This is the answer.")
    
    @patch('chat_session.create_llm_client')
    def test_compare_models(self, mock_create_client):
        """Test model comparison functionality."""
        # Mock client that can switch models
        mock_client = Mock()
        mock_client.model_name = "llama2:13b-chat"
        mock_client.generate_response.return_value = "Response from model"
        mock_client.switch_model.return_value = True
        mock_create_client.return_value = mock_client
        
        chat = ChatSession()
        models = ["llama2:13b-chat", "mistral:7b-instruct"]
        results = chat.compare_models("Test question", models)
        
        self.assertEqual(len(results), 2)
        self.assertIn("llama2:13b-chat", results)
        self.assertIn("mistral:7b-instruct", results)
        self.assertEqual(results["llama2:13b-chat"], "Response from model")
        self.assertEqual(results["mistral:7b-instruct"], "Response from model")


if __name__ == '__main__':
    unittest.main() 