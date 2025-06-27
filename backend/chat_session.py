"""
Chat Session for the PDF RAG application.
Provides conversation management, prompt templates, and response formatting.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from llm_client import create_llm_client, LLMError
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptTemplate:
    """Template for different types of prompts."""
    
    @staticmethod
    def general_chat(prompt: str) -> str:
        """General chat prompt template."""
        return f"""You are a helpful AI assistant. Please provide a clear and informative response to the following question:

Question: {prompt}

Please respond in a helpful and conversational manner."""

    @staticmethod
    def rag_question(prompt: str, context: str) -> str:
        """RAG-style prompt template with context."""
        return f"""You are a helpful AI assistant. Please answer the following question based on the provided context.

Context: {context}

Question: {prompt}

Instructions:
- Answer based ONLY on the provided context
- If the context doesn't contain enough information to answer the question, say so
- Be specific and cite relevant parts of the context
- Keep your response concise but informative

Please provide a clear and accurate answer based on the context."""

    @staticmethod
    def document_analysis(prompt: str, context: str) -> str:
        """Document analysis prompt template."""
        return f"""You are an expert document analyst. Please analyze the following document content and answer the question.

Document Content: {context}

Question: {prompt}

Please provide a detailed analysis based on the document content."""

    @staticmethod
    def code_analysis(prompt: str, context: str) -> str:
        """Code analysis prompt template."""
        return f"""You are an expert code analyst. Please analyze the following code and answer the question.

Code: {context}

Question: {prompt}

Please provide a technical analysis of the code."""


class ChatMessage:
    """Represents a single message in the chat."""
    
    def __init__(self, role: str, content: str, timestamp: datetime = None, metadata: Dict[str, Any] = None):
        self.role = role  # 'user' or 'assistant'
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        return f"[{self.role.upper()}] {self.content}"


class ChatSession:
    """Manages a chat session with conversation history and model switching."""
    
    def __init__(self, model_name: str = None, max_history: int = 50):
        """
        Initialize a chat session.
        
        Args:
            model_name: Name of the model to use
            max_history: Maximum number of messages to keep in history
        """
        self.llm_client = create_llm_client(model_name)
        self.model_name = self.llm_client.model_name
        self.max_history = max_history
        self.messages: List[ChatMessage] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Created chat session {self.session_id} with model: {self.model_name}")
    
    def ask(self, question: str, context: str = None, prompt_type: str = "general") -> str:
        """
        Ask a question and get a response.
        
        Args:
            question: The user's question
            context: Optional context for RAG-style questions
            prompt_type: Type of prompt template to use
        
        Returns:
            The AI's response
        """
        try:
            # Add user message to history
            user_message = ChatMessage("user", question, metadata={"context": context, "prompt_type": prompt_type})
            self.messages.append(user_message)
            
            # Build prompt based on type
            if prompt_type == "rag" and context:
                prompt = PromptTemplate.rag_question(question, context)
            elif prompt_type == "document" and context:
                prompt = PromptTemplate.document_analysis(question, context)
            elif prompt_type == "code" and context:
                prompt = PromptTemplate.code_analysis(question, context)
            else:
                prompt = PromptTemplate.general_chat(question)
            
            # Generate response
            logger.info(f"Generating response for prompt type: {prompt_type}")
            response_text = self.llm_client.generate_response(prompt, context=context)
            
            # Format response
            formatted_response = self._format_response(response_text, prompt_type)
            
            # Add assistant message to history
            assistant_message = ChatMessage("assistant", formatted_response, metadata={"prompt_type": prompt_type})
            self.messages.append(assistant_message)
            
            # Trim history if needed
            self._trim_history()
            
            return formatted_response
            
        except LLMError as e:
            error_response = f"I apologize, but I encountered an error: {str(e)}"
            logger.error(f"LLM error in chat session: {e}")
            
            # Add error message to history
            error_message = ChatMessage("assistant", error_response, metadata={"error": True})
            self.messages.append(error_message)
            
            return error_response
            
        except Exception as e:
            error_response = "I apologize, but I encountered an unexpected error. Please try again."
            logger.error(f"Unexpected error in chat session: {e}")
            
            # Add error message to history
            error_message = ChatMessage("assistant", error_response, metadata={"error": True})
            self.messages.append(error_message)
            
            return error_response
    
    def _format_response(self, response: str, prompt_type: str) -> str:
        """
        Format the response based on prompt type.
        
        Args:
            response: Raw response from LLM
            prompt_type: Type of prompt used
        
        Returns:
            Formatted response
        """
        # Clean up response
        response = response.strip()
        
        # Remove common prefixes if they exist
        prefixes_to_remove = ["Answer:", "Response:", "A:", "R:"]
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Add formatting based on prompt type
        if prompt_type == "rag":
            # Add source attribution for RAG responses
            if not response.startswith("Based on the context"):
                response = f"Based on the provided context: {response}"
        
        return response
    
    def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different model.
        
        Args:
            model_name: Name of the new model
        
        Returns:
            True if successful, False otherwise
        """
        success = self.llm_client.switch_model(model_name)
        if success:
            self.model_name = model_name
            logger.info(f"Switched to model: {model_name}")
        return success
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history as a list of dictionaries."""
        return [msg.to_dict() for msg in self.messages]
    
    def get_messages(self) -> List[ChatMessage]:
        """Get conversation history as ChatMessage objects."""
        return self.messages.copy()
    
    def clear_history(self):
        """Clear conversation history."""
        self.messages.clear()
        logger.info("Cleared conversation history")
    
    def _trim_history(self):
        """Trim history to max_history size."""
        if len(self.messages) > self.max_history:
            # Keep the most recent messages
            self.messages = self.messages[-self.max_history:]
            logger.info(f"Trimmed history to {self.max_history} messages")
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session."""
        return {
            "session_id": self.session_id,
            "model_name": self.model_name,
            "message_count": len(self.messages),
            "created_at": self.messages[0].timestamp.isoformat() if self.messages else None,
            "last_activity": self.messages[-1].timestamp.isoformat() if self.messages else None
        }
    
    def compare_models(self, question: str, models: List[str], context: str = None) -> Dict[str, str]:
        """
        Compare responses from different models.
        
        Args:
            question: Question to ask
            models: List of model names to compare
            context: Optional context
        
        Returns:
            Dictionary mapping model names to responses
        """
        results = {}
        original_model = self.model_name
        
        for model_name in models:
            try:
                # Switch to model
                if self.switch_model(model_name):
                    response = self.ask(question, context, prompt_type="rag" if context else "general")
                    results[model_name] = response
                else:
                    results[model_name] = f"Error: Could not switch to model {model_name}"
            except Exception as e:
                results[model_name] = f"Error: {str(e)}"
        
        # Switch back to original model
        self.switch_model(original_model)
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Test the chat session
    try:
        chat = ChatSession()
        
        # Test general chat
        print("🤖 Testing general chat...")
        response = chat.ask("Hello! How are you?")
        print(f"Response: {response}")
        
        # Test RAG-style question
        print("\n📚 Testing RAG-style question...")
        context = "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed."
        response = chat.ask("What is machine learning?", context=context, prompt_type="rag")
        print(f"Response: {response}")
        
        # Test document analysis
        print("\n📄 Testing document analysis...")
        doc_context = "This research paper discusses the application of deep learning in computer vision tasks."
        response = chat.ask("What is the main topic of this document?", context=doc_context, prompt_type="document")
        print(f"Response: {response}")
        
        # Show session info
        print(f"\n📊 Session info: {chat.get_session_info()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("This is expected if Ollama is not running.") 