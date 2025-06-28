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

from chat_session import ChatSession, ChatMessage, UnifiedPromptTemplate, LLMError


class TestUnifiedPromptTemplate(unittest.TestCase):
    """Test cases for unified prompt templates."""
    
    def test_create_prompt_with_context(self):
        """Test unified prompt creation with context."""
        question = "What is the main topic?"
        context = "This document discusses machine learning."
        prompt = UnifiedPromptTemplate.create_prompt(question, context)
        
        self.assertIn(question, prompt)
        self.assertIn(context, prompt)
        self.assertIn("IMPORTANT RULES:", prompt)
        self.assertIn("RESPONSE STYLES", prompt)
        self.assertIn("DECISION RULES:", prompt)
        self.assertIn("source attribution", prompt)
    
    def test_create_prompt_without_context(self):
        """Test unified prompt creation without context."""
        question = "Hello! How are you?"
        prompt = UnifiedPromptTemplate.create_prompt(question)
        
        self.assertIn(question, prompt)
        self.assertIn("helpful AI assistant", prompt)
        self.assertIn("conversational manner", prompt)
        self.assertNotIn("INSTRUCTIONS:", prompt)  # Should not have instructions when no context


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
        metadata = {"context": "test"}
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
    
    @patch('chat_session.create_llm_client')
    def test_chat_session_initialization(self, mock_create_client):
        """Test chat session initialization."""
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
    def test_ask_with_context(self, mock_create_client):
        """Test asking a question with context."""
        mock_client = Mock()
        mock_client.model_name = "llama2:13b-chat"
        mock_client.generate_response.return_value = "Based on the context, AI is artificial intelligence."
        mock_create_client.return_value = mock_client
        
        chat = ChatSession()
        context = "AI is artificial intelligence."
        response = chat.ask("What is AI?", context=context)
        
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
    def test_ask_with_unexpected_error(self, mock_create_client):
        """Test handling unexpected errors."""
        mock_client = Mock()
        mock_client.model_name = "llama2:13b-chat"
        mock_client.generate_response.side_effect = Exception("Unexpected error")
        mock_create_client.return_value = mock_client
        
        chat = ChatSession()
        response = chat.ask("Hello!")
        
        self.assertIn("I apologize, but I encountered an unexpected error", response)
        self.assertEqual(len(chat.messages), 2)  # User message + error response
    
    @patch('chat_session.create_llm_client')
    def test_switch_model_success(self, mock_create_client):
        """Test successful model switching."""
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
        """Test failed model switching."""
        mock_client = Mock()
        mock_client.model_name = "llama2:13b-chat"
        mock_client.switch_model.return_value = False
        mock_create_client.return_value = mock_client
        
        chat = ChatSession()
        success = chat.switch_model("invalid-model")
        
        self.assertFalse(success)
        self.assertEqual(chat.model_name, "llama2:13b-chat")  # Should remain unchanged
    
    @patch('chat_session.create_llm_client')
    def test_get_history(self, mock_create_client):
        """Test getting conversation history."""
        mock_client = Mock()
        mock_client.model_name = "llama2:13b-chat"
        mock_client.generate_response.return_value = "Test response"
        mock_create_client.return_value = mock_client
        
        chat = ChatSession()
        chat.ask("Hello!")
        
        history = chat.get_history()
        
        self.assertEqual(len(history), 2)  # User message + assistant response
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "Hello!")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["content"], "Test response")
    
    @patch('chat_session.create_llm_client')
    def test_clear_history(self, mock_create_client):
        """Test clearing conversation history."""
        mock_client = Mock()
        mock_client.model_name = "llama2:13b-chat"
        mock_client.generate_response.return_value = "Test response"
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
        mock_client.generate_response.return_value = "Test response"
        mock_create_client.return_value = mock_client
        
        chat = ChatSession()
        chat.ask("Hello!")
        
        info = chat.get_session_info()
        
        self.assertIn("session_id", info)
        self.assertIn("model_name", info)
        self.assertIn("message_count", info)
        self.assertIn("created_at", info)
        self.assertIn("last_activity", info)
        
        self.assertEqual(info["model_name"], "llama2:13b-chat")
        self.assertEqual(info["message_count"], 2)
    
    @patch('chat_session.create_llm_client')
    def test_compare_models(self, mock_create_client):
        """Test comparing different models."""
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