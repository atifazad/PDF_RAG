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


class UnifiedPromptTemplate:
    """Unified prompt template that handles all question types with few-shot examples."""
    
    @staticmethod
    def create_prompt(question: str, context: str = None) -> str:
        """
        Create a unified prompt that handles all question types with few-shot examples.
        
        Args:
            question: The user's question
            context: Optional context for RAG-style questions
        
        Returns:
            Formatted prompt string
        """
        if context:
            return f"""You are a helpful AI assistant for a PDF RAG application. Your job is to classify the user's question and respond appropriately.

IMPORTANT RULES:
- Only use greetings, jokes, or conversational openers for GENERAL CONVERSATION.
- For RAG, DOCUMENT ANALYSIS, and CODE ANALYSIS, answer directly and factually. Do NOT include greetings, jokes, or small talk.
- If the context does not contain the answer for a RAG question, say: "The context does not provide this information."
- For RAG questions, ALWAYS include source attribution in this format: "This answer came from [document name], [page/section info]."

RESPONSE STYLES (use these as examples of how to respond, NOT as questions to answer):

1. GENERAL CONVERSATION style:
   - Use greetings and friendly tone
   - Example response style: "Hello! I'm doing great, thank you for asking! How can I assist you today?"

2. RAG QUESTION style:
   - Answer based ONLY on the provided context
   - Cite specific parts of the context
   - ALWAYS include source attribution
   - Example response style: "According to the context, [specific fact]. The document states [exact quote]. This answer came from [document name], [page/section info]."

3. DOCUMENT ANALYSIS style:
   - Analyze document structure, purpose, organization
   - Example response style: "The document is structured as follows: [analysis]. The purpose is [analysis]."

4. CODE ANALYSIS style:
   - Provide technical analysis of code/implementation
   - Example response style: "This function [technical explanation]. It uses [technical details]."

NOW ANSWER THIS ACTUAL QUESTION:

CONTEXT: {context}

QUESTION: {question}

DECISION RULES:
- If the question is about greetings, casual conversation, or general knowledge NOT in the context → Use GENERAL CONVERSATION style
- If the question asks for specific facts, numbers, or information that should be found in the provided context → Use RAG QUESTION style (include source attribution)
- If the question asks about document structure, organization, purpose, or format → Use DOCUMENT ANALYSIS style
- If the question asks about code, algorithms, or technical implementation details → Use CODE ANALYSIS style

Please respond to the above question using the appropriate style."""
        else:
            return f"""You are a helpful AI assistant. Please provide a clear and informative response to the following question:

Question: {question}

Please respond in a helpful and conversational manner."""


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
    
    def ask(self, question: str, context: str = None) -> str:
        """
        Ask a question and get a response using unified prompt approach.
        
        Args:
            question: The user's question
            context: Optional context for RAG-style questions
        
        Returns:
            The AI's response
        """
        try:
            # Add user message to history
            user_message = ChatMessage("user", question, metadata={"context": context})
            self.messages.append(user_message)
            
            # Create unified prompt
            prompt = UnifiedPromptTemplate.create_prompt(question, context)
            
            # Generate response
            logger.info("Generating response,,,")
            response_text = self.llm_client.generate_response(prompt, context=context)
            
            # Format response
            formatted_response = self._format_response(response_text)
            
            # Add assistant message to history
            assistant_message = ChatMessage("assistant", formatted_response)
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
    
    def _format_response(self, response: str) -> str:
        """
        Format the response.
        
        Args:
            response: Raw response from LLM
        
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
                    response = self.ask(question, context=context)
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
        chat = ChatSession(model_name="mistral:7b-instruct")
        
        # Test general chat
        print("🤖 Testing general chat...")
        response = chat.ask("Hello! How are you?")
        print(f"Response: {response}")
        
        # Test RAG-style question
        print("\n📚 Testing RAG-style question...")
        context = """Document: 'Introduction to Machine Learning' by John Smith, Page 5. Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed. The field has seen significant growth in recent years.

Document: 'Introduction to Machine Learning' by John Smith, Page 12. Deep learning is a specialized subset of machine learning that uses neural networks with multiple layers to model complex patterns in data.

Document: 'Introduction to Machine Learning' by John Smith, Page 18. Supervised learning involves training a model on labeled data, where the correct answers are provided during training."""
        response = chat.ask("What is supervised learning?", context=context)
        print(f"Response: {response}")
        
        # Test document analysis
        print("\n📄 Testing document analysis...")
        doc_context = """Document: 'Deep Learning in Computer Vision' by Sarah Johnson, Page 1. This research paper discusses the application of deep learning in computer vision tasks. The paper begins with an introduction to computer vision challenges.

Document: 'Deep Learning in Computer Vision' by Sarah Johnson, Page 3. The methodology section describes the use of convolutional neural networks for image classification tasks.

Document: 'Deep Learning in Computer Vision' by Sarah Johnson, Page 7. The results section shows that the proposed model achieved 95% accuracy on the test dataset."""
        response = chat.ask("What is the main topic of this document?", context=doc_context)
        print(f"Response: {response}")
        
        # Test code analysis
        print("\n💻 Testing code analysis...")
        code_context = """Document: 'Python Programming Guide' by Mike Chen, Page 15. Code example: def calculate_fibonacci(n): return n if n <= 1 else calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

Document: 'Python Programming Guide' by Mike Chen, Page 22. Code example: def bubble_sort(arr): for i in range(len(arr)): for j in range(0, len(arr)-i-1): if arr[j] > arr[j+1]: arr[j], arr[j+1] = arr[j+1], arr[j]

Document: 'Python Programming Guide' by Mike Chen, Page 28. Code example: def binary_search(arr, target): left, right = 0, len(arr)-1; while left <= right: mid = (left + right) // 2; if arr[mid] == target: return mid; elif arr[mid] < target: left = mid + 1; else: right = mid - 1; return -1"""
        response = chat.ask("What does bubble_sort function do?", context=code_context)
        print(f"Response: {response}")
        
        # Show session info
        print(f"\n📊 Session info: {chat.get_session_info()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("This is expected if Ollama is not running or the model is not available.") 