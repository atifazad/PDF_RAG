"""
Tests for the document processor functionality.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import tempfile
import shutil

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from document_processor import (
    DocumentProcessor, DocumentMetadata, PageContent
)


class TestDocumentMetadata(unittest.TestCase):
    """Test cases for DocumentMetadata class."""
    
    def test_metadata_creation(self):
        """Test creating document metadata."""
        metadata = DocumentMetadata("test.pdf", "/path/to/test.pdf")
        
        self.assertEqual(metadata.filename, "test.pdf")
        self.assertEqual(metadata.file_path, "/path/to/test.pdf")
        self.assertEqual(metadata.total_pages, 0)
        self.assertEqual(metadata.file_size_mb, 0.0)
        self.assertEqual(metadata.is_scanned, False)
    
    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = DocumentMetadata("test.pdf", "/path/to/test.pdf")
        metadata.total_pages = 10
        metadata.file_size_mb = 2.5
        metadata.title = "Test Document"
        metadata.author = "Test Author"
        metadata.is_scanned = True
        
        metadata_dict = metadata.to_dict()
        
        self.assertEqual(metadata_dict["filename"], "test.pdf")
        self.assertEqual(metadata_dict["total_pages"], 10)
        self.assertEqual(metadata_dict["file_size_mb"], 2.5)
        self.assertEqual(metadata_dict["title"], "Test Document")
        self.assertEqual(metadata_dict["author"], "Test Author")
        self.assertEqual(metadata_dict["is_scanned"], True)


class TestPageContent(unittest.TestCase):
    """Test cases for PageContent class."""
    
    def test_page_content_creation(self):
        """Test creating page content."""
        page = PageContent(1, "Test text content", "text")
        
        self.assertEqual(page.page_number, 1)
        self.assertEqual(page.text, "Test text content")
        self.assertEqual(page.extraction_method, "text")
        self.assertEqual(page.confidence, 1.0)
    
    def test_page_content_with_ocr(self):
        """Test creating page content with OCR data."""
        page = PageContent(1, "OCR text", "ocr")
        page.confidence = 0.85
        page.bbox = [[10, 20, 100, 30]]
        
        self.assertEqual(page.extraction_method, "ocr")
        self.assertEqual(page.confidence, 0.85)
        self.assertEqual(page.bbox, [[10, 20, 100, 30]])
    
    def test_page_content_to_dict(self):
        """Test converting page content to dictionary."""
        page = PageContent(1, "Test text", "text")
        page.images = ["image1.png", "image2.png"]
        
        page_dict = page.to_dict()
        
        self.assertEqual(page_dict["page_number"], 1)
        self.assertEqual(page_dict["text"], "Test text")
        self.assertEqual(page_dict["extraction_method"], "text")
        self.assertEqual(page_dict["images"], ["image1.png", "image2.png"])


class TestDocumentProcessor(unittest.TestCase):
    """Test cases for DocumentProcessor class."""
    
    def setUp(self):
        """Set up test environment."""
        self.processor = DocumentProcessor()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_processor_initialization(self):
        """Test document processor initialization."""
        self.assertIsNotNone(self.processor)
        self.assertIsNone(self.processor._ocr_reader)
        self.assertEqual(self.processor.max_file_size_mb, 50)  # Default from config
    
    @patch('document_processor.Config.get_max_file_size_mb')
    def test_processor_with_custom_config(self, mock_config):
        """Test processor with custom configuration."""
        mock_config.return_value = 100
        
        processor = DocumentProcessor()
        self.assertEqual(processor.max_file_size_mb, 100)
    
    def test_validate_file_nonexistent(self):
        """Test file validation with nonexistent file."""
        is_valid, error_msg = self.processor.validate_file("/nonexistent/file.pdf")
        
        self.assertFalse(is_valid)
        self.assertIn("File does not exist", error_msg)
    
    def test_validate_file_wrong_extension(self):
        """Test file validation with wrong file extension."""
        # Create a temporary file with wrong extension
        temp_file = os.path.join(self.temp_dir, "test.txt")
        with open(temp_file, 'w') as f:
            f.write("test content")
        
        is_valid, error_msg = self.processor.validate_file(temp_file)
        
        self.assertFalse(is_valid)
        self.assertIn("Unsupported file type", error_msg)
    
    def test_validate_file_too_large(self):
        """Test file validation with file too large."""
        # Create a temporary file
        temp_file = os.path.join(self.temp_dir, "test.pdf")
        
        # Mock the file size to be larger than limit
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value = Mock(st_size=100 * 1024 * 1024)  # 100MB
            
            is_valid, error_msg = self.processor.validate_file(temp_file)
            
            self.assertFalse(is_valid)
            self.assertIn("exceeds limit", error_msg)
    
    def test_validate_file_valid(self):
        """Test file validation with valid file."""
        # Create a temporary PDF file
        temp_file = os.path.join(self.temp_dir, "test.pdf")
        with open(temp_file, 'w') as f:
            f.write("test content")
        
        # Mock the file size to be within limit
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value = Mock(st_size=1024)  # 1KB
            
            is_valid, error_msg = self.processor.validate_file(temp_file)
            
            self.assertTrue(is_valid)
            self.assertEqual(error_msg, "")
    
    @patch('document_processor.fitz.open')
    def test_extract_metadata(self, mock_fitz_open):
        """Test metadata extraction."""
        # Mock PyMuPDF document
        mock_doc = Mock()
        mock_doc.metadata = {
            'title': 'Test Document',
            'author': 'Test Author',
            'subject': 'Test Subject',
            'creator': 'Test Creator',
            'producer': 'Test Producer',
            'creationDate': '2023-01-01',
            'modDate': '2023-01-02'
        }
        mock_doc.__len__ = Mock(return_value=10)
        mock_fitz_open.return_value = mock_doc
        
        # Mock file size
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value = Mock(st_size=1024 * 1024)  # 1MB
            
            metadata = self.processor.extract_metadata("/test/file.pdf")
            
            self.assertEqual(metadata.filename, "file.pdf")
            self.assertEqual(metadata.total_pages, 10)
            self.assertEqual(metadata.file_size_mb, 1.0)
            self.assertEqual(metadata.title, "Test Document")
            self.assertEqual(metadata.author, "Test Author")
    
    @patch('document_processor.pdfplumber.open')
    def test_extract_text_with_pdfplumber(self, mock_pdfplumber_open):
        """Test text extraction with pdfplumber."""
        # Mock pdfplumber pages
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page1.images = ["image1.png"]
        
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 content"
        mock_page2.images = []
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf
        
        pages = self.processor.extract_text_with_pdfplumber("/test/file.pdf")
        
        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0].page_number, 1)
        self.assertEqual(pages[0].text, "Page 1 content")
        self.assertEqual(pages[0].extraction_method, "text")
        self.assertEqual(pages[0].images, ["image1.png"])
        self.assertEqual(pages[1].page_number, 2)
        self.assertEqual(pages[1].text, "Page 2 content")
    
    @patch('document_processor.fitz.open')
    @patch('document_processor.easyocr.Reader')
    def test_extract_text_with_ocr(self, mock_easyocr_reader, mock_fitz_open):
        """Test text extraction with OCR."""
        # Mock EasyOCR reader
        mock_reader = Mock()
        mock_reader.readtext.return_value = [
            ([[10, 20], [100, 20], [100, 30], [10, 30]], "OCR text", 0.9)
        ]
        mock_easyocr_reader.return_value = mock_reader
        
        # Mock PyMuPDF document
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)
        
        mock_page = Mock()
        mock_pixmap = Mock()
        mock_pixmap.tobytes.return_value = b"fake_image_data"
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_doc.load_page.return_value = mock_page
        
        mock_fitz_open.return_value = mock_doc
        
        # Mock PIL Image
        with patch('document_processor.Image.open') as mock_pil_open:
            mock_image = Mock()
            mock_pil_open.return_value = mock_image
            
            pages = self.processor.extract_text_with_ocr("/test/file.pdf")
            
            self.assertEqual(len(pages), 1)
            self.assertEqual(pages[0].page_number, 1)
            self.assertEqual(pages[0].text, "OCR text")
            self.assertEqual(pages[0].extraction_method, "ocr")
            self.assertEqual(pages[0].confidence, 0.9)
    
    def test_is_page_scanned_short_text(self):
        """Test scanned page detection with short text."""
        # Short text should be considered scanned
        is_scanned = self.processor.is_page_scanned("Short text", min_text_length=50)
        self.assertTrue(is_scanned)
    
    def test_is_page_scanned_long_text(self):
        """Test scanned page detection with long text."""
        # Long text should not be considered scanned
        long_text = "This is a long text that contains enough content to be considered a proper text-based article. It is just a regular paragraph of text for testing purposes."
        is_scanned = self.processor.is_page_scanned(long_text, min_text_length=50)
        self.assertFalse(is_scanned)
    
    def test_is_page_scanned_with_indicators(self):
        """Test scanned page detection with scanned indicators."""
        # Text with scanned indicators should be considered scanned
        text_with_indicators = "This is an image of a scanned document page that contains pictures and photos."
        is_scanned = self.processor.is_page_scanned(text_with_indicators, min_text_length=50)
        self.assertTrue(is_scanned)
    
    @patch('document_processor.DocumentProcessor.extract_metadata')
    @patch('document_processor.DocumentProcessor.extract_text_with_pdfplumber')
    @patch('document_processor.DocumentProcessor.extract_text_with_ocr')
    @patch('document_processor.DocumentProcessor.validate_file')
    def test_process_document_hybrid(self, mock_validate, mock_ocr, mock_pdfplumber, mock_metadata):
        """Test hybrid document processing."""
        # Mock validation
        mock_validate.return_value = (True, "")
        
        # Mock metadata
        mock_metadata_obj = Mock()
        mock_metadata_obj.filename = "test.pdf"
        mock_metadata_obj.total_pages = 2
        mock_metadata_obj.file_size_mb = 1.0
        mock_metadata.return_value = mock_metadata_obj
        
        # Mock pdfplumber pages (one text-based, one scanned)
        text_page1 = PageContent(1, "Long text content that should be considered text-based", "text")
        text_page2 = PageContent(2, "Short", "text")  # Short text, should trigger OCR
        
        mock_pdfplumber.return_value = [text_page1, text_page2]
        
        # Mock OCR pages
        ocr_page2 = PageContent(2, "OCR extracted text", "ocr")
        ocr_page2.confidence = 0.85
        mock_ocr.return_value = [PageContent(1, "", "ocr"), ocr_page2]
        
        # Mock time
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0.0, 1.5]  # Start and end time
            
            metadata, pages = self.processor.process_document("/test/file.pdf")
            
            # Check results
            self.assertEqual(len(pages), 2)
            self.assertEqual(pages[0].extraction_method, "text")  # First page kept as text
            self.assertEqual(pages[1].extraction_method, "ocr")   # Second page used OCR
            
            # Check metadata updates
            self.assertEqual(metadata.text_pages, 1)
            self.assertEqual(metadata.ocr_pages, 1)
            self.assertEqual(metadata.processing_time, 1.5)
    
    def test_get_processing_summary(self):
        """Test processing summary generation."""
        # Create test metadata
        metadata = DocumentMetadata("test.pdf", "/test/file.pdf")
        metadata.total_pages = 3
        metadata.file_size_mb = 2.5
        metadata.text_pages = 2
        metadata.ocr_pages = 1
        metadata.processing_time = 1.5
        
        # Create test pages
        pages = [
            PageContent(1, "Page 1 content", "text"),
            PageContent(2, "Page 2 content", "text"),
            PageContent(3, "Page 3 content", "ocr")
        ]
        
        summary = self.processor.get_processing_summary(metadata, pages)
        
        self.assertEqual(summary["filename"], "test.pdf")
        self.assertEqual(summary["total_pages"], 3)
        self.assertEqual(summary["file_size_mb"], 2.5)
        # Calculate expected text length: "Page 1 content" (14) + "Page 2 content" (14) + "Page 3 content" (14) = 42
        self.assertEqual(summary["total_text_length"], 42)
        self.assertEqual(summary["avg_text_per_page"], 14)  # 42 / 3
        self.assertEqual(summary["extraction_methods"]["text"], 2)
        self.assertEqual(summary["extraction_methods"]["ocr"], 1)
        self.assertEqual(summary["processing_time"], 1.5)
    
    @patch('document_processor.easyocr.Reader')
    def test_ocr_reader_lazy_loading(self, mock_easyocr_reader):
        """Test that EasyOCR reader is loaded lazily."""
        # Initially no reader
        self.assertIsNone(self.processor._ocr_reader)
        
        # Access the reader property
        reader = self.processor.ocr_reader
        
        # Reader should now be initialized
        self.assertIsNotNone(self.processor._ocr_reader)
        mock_easyocr_reader.assert_called_once_with(['en'])
        
        # Second access should not create another reader
        reader2 = self.processor.ocr_reader
        self.assertEqual(reader, reader2)
        # Should still only be called once
        mock_easyocr_reader.assert_called_once()


if __name__ == '__main__':
    unittest.main() 