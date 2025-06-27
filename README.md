# PDF RAG - Chat Over Your Documents

A document analysis application that allows you to upload PDFs and chat with them using AI.

## Features

- 📄 **Multi-PDF Support**: Upload and process multiple PDF documents
- 🤖 **Configurable AI Models**: Switch between Llama2-13B, Mistral-7B, and more
- 🔍 **Smart Document Processing**: Handles both text-based and scanned PDFs
- 🧠 **Intelligent Q&A**: Get AI-generated answers grounded in your documents
- 📍 **Source Attribution**: See exactly which documents and pages answers come from
- 🚀 **Fast & Local**: Runs entirely on your machine with Ollama

## Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: Streamlit
- **LLM**: Ollama with configurable models (Llama2-13B, Mistral-7B)
- **Vector Database**: Chroma
- **Document Processing**: pdfplumber + EasyOCR
- **Embeddings**: Sentence Transformers
- **Containerization**: Docker

## Quick Start

### Prerequisites

1. **Install Ollama**: [https://ollama.ai/](https://ollama.ai/)
2. **Download Models**:
   ```bash
   ollama pull llama2:13b-chat
   ollama pull mistral:7b-instruct
   ```

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd PDF_RAG
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your preferences
   ```

5. **Run the application**:
   ```bash
   # Development mode
   streamlit run app.py
   
   # Or with Docker
   docker-compose up
   ```

## Configuration

### Model Selection

Set the `LLM_MODEL` environment variable:
```bash
# Use Llama2-13B (default)
export LLM_MODEL=llama2:13b-chat

# Use Mistral-7B
export LLM_MODEL=mistral:7b-instruct
```

### Available Models

| Model | Memory Usage | Speed | Quality | Best For |
|-------|--------------|-------|---------|----------|
| Llama2-13B | 16GB | Medium | Excellent | High-quality responses |
| Mistral-7B | 8GB | Fast | Good | Quick responses |
| Llama2-7B | 8GB | Fast | Good | Balanced approach |

## Usage

1. **Upload PDFs**: Use the file upload interface
2. **Wait for Processing**: Documents are automatically chunked and embedded
3. **Ask Questions**: Chat with your documents using natural language
4. **View Sources**: See which documents and pages contain the answers

## Project Structure

```
PDF_RAG/
├── app.py                 # Streamlit frontend
├── backend/               # FastAPI backend
│   ├── main.py           # API endpoints
│   ├── config.py         # Configuration
│   ├── llm_client.py     # LLM interface
│   ├── document_processor.py  # PDF processing
│   └── vector_store.py   # Vector database
├── frontend/             # Streamlit components
├── tests/                # Test files
├── data/                 # Document storage
├── requirements.txt      # Dependencies
├── Dockerfile           # Container configuration
└── README.md           # This file
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black .
flake8 .
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request 