# Integration Tests

This directory contains integration tests that test the document processor with actual PDF files.

## Test Files

### `test_document_processor_integration.py`
Automated integration test that processes available PDF files in the project.

**Features:**
- Automatically finds PDF files in `data/` directory or current directory
- Tests file validation, metadata extraction, and document processing
- Shows processing summary and page previews
- No user interaction required
- Perfect for CI/CD pipelines and automated testing

**Usage:**
```bash
python tests/integration/backend/test_document_processor_integration.py
```

## Prerequisites

1. **Virtual Environment**: Make sure your virtual environment is activated
   ```bash
   source ~/venv/venv_PDF_RAG/bin/activate
   ```

2. **Dependencies**: All required packages should be installed
   ```bash
   pip install -r requirements.txt
   ```

3. **Test Data**: Place PDF files in the `data/` directory for testing

## What This Test Does

This integration test verifies that the document processor can:

- ✅ Validate PDF files (size, format, existence)
- ✅ Extract metadata (pages, title, author, file size)
- ✅ Process documents using text extraction
- ✅ Process documents using OCR when needed
- ✅ Generate processing summaries
- ✅ Handle different types of PDFs (text-based vs scanned)

## Differences from Unit Tests

- **Unit Tests** (`tests/backend/`): Use mocks and fake data, test individual functions
- **Integration Tests** (`tests/integration/backend/`): Use real PDF files, test the full pipeline

## Adding Test PDFs

To add PDF files for testing:

1. Place them in the `data/` directory
2. The test will automatically find and use them
3. PDF files in `data/` are ignored by git (see `.gitignore`)

## Example Output

```
🧪 Integration Document Processor Test
==================================================
📄 Found 1 PDF file(s):
  1. data/devplan.pdf

🔍 Testing with: data/devplan.pdf

1. Validating file...
✅ File validation passed!

2. Extracting metadata...
✅ Metadata extracted:
   - Filename: devplan.pdf
   - Total pages: 10
   - File size: 0.16 MB
   - Title: ExpenseShare App - Development Plan

3. Processing document...
✅ Document processing successful!
   - Pages processed: 10
   - Total text length: 8036 characters

📊 Processing Summary:
   - Text pages: 10
   - OCR pages: 0
   - Processing time: 0.34 seconds
``` 