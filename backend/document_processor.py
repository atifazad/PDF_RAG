"""
Document Processing Pipeline for the PDF RAG application.
Handles PDF text extraction, OCR for scanned documents, and metadata extraction.
"""

import logging
import os
import io
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import fitz  # PyMuPDF for metadata
import pdfplumber
import easyocr
from PIL import Image
import numpy as np
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentMetadata:
    """Represents metadata extracted from a document."""
    
    def __init__(self, filename: str, file_path: str):
        self.filename = filename
        self.file_path = file_path
        self.total_pages = 0
        self.file_size_mb = 0.0
        self.title = ""
        self.author = ""
        self.subject = ""
        self.creator = ""
        self.producer = ""
        self.creation_date = ""
        self.modification_date = ""
        self.is_scanned = False
        self.text_pages = 0
        self.ocr_pages = 0
        self.processing_time = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "filename": self.filename,
            "file_path": self.file_path,
            "total_pages": self.total_pages,
            "file_size_mb": self.file_size_mb,
            "title": self.title,
            "author": self.author,
            "subject": self.subject,
            "creator": self.creator,
            "producer": self.producer,
            "creation_date": self.creation_date,
            "modification_date": self.modification_date,
            "is_scanned": self.is_scanned,
            "text_pages": self.text_pages,
            "ocr_pages": self.ocr_pages,
            "processing_time": self.processing_time
        }


class PageContent:
    """Represents content extracted from a single page."""
    
    def __init__(self, page_number: int, text: str = "", extraction_method: str = "unknown"):
        self.page_number = page_number
        self.text = text
        self.extraction_method = extraction_method  # "text", "ocr", "hybrid"
        self.confidence = 1.0  # For OCR results
        self.bbox = None  # Bounding box for OCR text
        self.images = []  # List of images found on page
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert page content to dictionary."""
        return {
            "page_number": self.page_number,
            "text": self.text,
            "extraction_method": self.extraction_method,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "images": self.images
        }


class DocumentProcessor:
    """Main document processing class."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.max_file_size_mb = Config.get_max_file_size_mb()
        
        # Initialize EasyOCR reader (lazy loading)
        self._ocr_reader = None
        
        logger.info(f"Initialized DocumentProcessor with max file size: {self.max_file_size_mb}MB")
    
    @property
    def ocr_reader(self):
        """Lazy load EasyOCR reader."""
        if self._ocr_reader is None:
            logger.info("Initializing EasyOCR reader...")
            self._ocr_reader = easyocr.Reader(['en'])
            logger.info("EasyOCR reader initialized")
        return self._ocr_reader
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate if a file can be processed.
        
        Args:
            file_path: Path to the file
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                return False, f"File does not exist: {file_path}"
            
            # Check file size
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                return False, f"File size ({file_size_mb:.2f}MB) exceeds limit ({self.max_file_size_mb}MB)"
            
            # Check file extension
            if path.suffix.lower() != '.pdf':
                return False, f"Unsupported file type: {path.suffix}. Only PDF files are supported."
            
            return True, ""
            
        except Exception as e:
            return False, f"Error validating file: {str(e)}"
    
    def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """
        Extract metadata from a PDF file.
        
        Args:
            file_path: Path to the PDF file
        
        Returns:
            DocumentMetadata object
        """
        metadata = DocumentMetadata(
            filename=Path(file_path).name,
            file_path=file_path
        )
        
        try:
            # Get file size
            file_size = Path(file_path).stat().st_size
            metadata.file_size_mb = file_size / (1024 * 1024)
            
            # Extract PDF metadata using PyMuPDF
            doc = fitz.open(file_path)
            metadata.total_pages = len(doc)
            
            # Get PDF metadata
            pdf_metadata = doc.metadata
            metadata.title = pdf_metadata.get('title', '')
            metadata.author = pdf_metadata.get('author', '')
            metadata.subject = pdf_metadata.get('subject', '')
            metadata.creator = pdf_metadata.get('creator', '')
            metadata.producer = pdf_metadata.get('producer', '')
            metadata.creation_date = pdf_metadata.get('creationDate', '')
            metadata.modification_date = pdf_metadata.get('modDate', '')
            
            doc.close()
            
            logger.info(f"Extracted metadata for {metadata.filename}: {metadata.total_pages} pages")
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {e}")
        
        return metadata
    
    def extract_text_with_pdfplumber(self, file_path: str) -> List[PageContent]:
        """
        Extract text from PDF using pdfplumber.
        
        Args:
            file_path: Path to the PDF file
        
        Returns:
            List of PageContent objects
        """
        pages = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        # Extract text
                        text = page.extract_text()
                        
                        # Extract images
                        images = page.images if hasattr(page, 'images') else []
                        
                        page_content = PageContent(
                            page_number=page_num,
                            text=text or "",
                            extraction_method="text"
                        )
                        page_content.images = images
                        
                        pages.append(page_content)
                        
                        logger.debug(f"Extracted text from page {page_num}: {len(text or '')} characters")
                        
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {e}")
                        # Add empty page content
                        pages.append(PageContent(
                            page_number=page_num,
                            text="",
                            extraction_method="text"
                        ))
        
        except Exception as e:
            logger.error(f"Error opening PDF with pdfplumber: {e}")
        
        return pages
    
    def extract_text_with_ocr(self, file_path: str) -> List[PageContent]:
        """
        Extract text from PDF using OCR.
        
        Args:
            file_path: Path to the PDF file
        
        Returns:
            List of PageContent objects
        """
        pages = []
        
        try:
            # Convert PDF pages to images
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    
                    # Render page to image
                    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Perform OCR
                    results = self.ocr_reader.readtext(np.array(img))
                    
                    # Extract text and confidence
                    text_parts = []
                    total_confidence = 0.0
                    bboxes = []
                    
                    for (bbox, text, confidence) in results:
                        if text.strip():  # Only include non-empty text
                            text_parts.append(text)
                            total_confidence += confidence
                            bboxes.append(bbox)
                    
                    # Combine text parts
                    full_text = " ".join(text_parts)
                    avg_confidence = total_confidence / len(results) if results else 0.0
                    
                    page_content = PageContent(
                        page_number=page_num + 1,
                        text=full_text,
                        extraction_method="ocr"
                    )
                    page_content.confidence = avg_confidence
                    page_content.bbox = bboxes
                    
                    pages.append(page_content)
                    
                    logger.debug(f"OCR extracted from page {page_num + 1}: {len(full_text)} characters, confidence: {avg_confidence:.2f}")
                    
                except Exception as e:
                    logger.warning(f"Error performing OCR on page {page_num + 1}: {e}")
                    pages.append(PageContent(
                        page_number=page_num + 1,
                        text="",
                        extraction_method="ocr"
                    ))
            
            doc.close()
        
        except Exception as e:
            logger.error(f"Error performing OCR on {file_path}: {e}")
        
        return pages
    
    def is_page_scanned(self, text_content: str, min_text_length: int = 50) -> bool:
        """
        Determine if a page is scanned based on text content.
        
        Args:
            text_content: Text extracted from the page
            min_text_length: Minimum text length to consider as text-based
        
        Returns:
            True if page appears to be scanned
        """
        # If very little text was extracted, likely scanned
        if len(text_content.strip()) < min_text_length:
            return True
        
        # Check for common scanned document indicators
        scanned_indicators = [
            "scanned", "scan", "document", "page", "sheet"
        ]
        
        text_lower = text_content.lower()
        indicator_count = sum(1 for indicator in scanned_indicators if indicator in text_lower)
        # If many indicators found, likely scanned
        if indicator_count > 1:
            return True
        
        return False
    
    def process_document(self, file_path: str) -> Tuple[DocumentMetadata, List[PageContent]]:
        """
        Process a PDF document using hybrid extraction strategy.
        
        Args:
            file_path: Path to the PDF file
        
        Returns:
            Tuple of (DocumentMetadata, List[PageContent])
        """
        import time
        start_time = time.time()
        
        # Validate file
        is_valid, error_msg = self.validate_file(file_path)
        if not is_valid:
            raise ValueError(error_msg)
        
        logger.info(f"Processing document: {file_path}")
        
        # Extract metadata
        metadata = self.extract_metadata(file_path)
        
        # Extract text using pdfplumber
        text_pages = self.extract_text_with_pdfplumber(file_path)
        
        # Determine which pages need OCR
        final_pages = []
        text_extracted_pages = 0
        ocr_extracted_pages = 0
        
        for page_content in text_pages:
            if self.is_page_scanned(page_content.text):
                # Page appears to be scanned, use OCR
                logger.info(f"Page {page_content.page_number} appears scanned, using OCR")
                
                # Extract OCR for this specific page
                ocr_pages = self.extract_text_with_ocr(file_path)
                ocr_page = next((p for p in ocr_pages if p.page_number == page_content.page_number), None)
                
                if ocr_page and ocr_page.text.strip():
                    final_pages.append(ocr_page)
                    ocr_extracted_pages += 1
                    logger.info(f"OCR extracted {len(ocr_page.text)} characters from page {page_content.page_number}")
                else:
                    # OCR failed, keep original text
                    final_pages.append(page_content)
                    text_extracted_pages += 1
            else:
                # Page has sufficient text, keep it
                final_pages.append(page_content)
                text_extracted_pages += 1
        
        # Update metadata
        metadata.text_pages = text_extracted_pages
        metadata.ocr_pages = ocr_extracted_pages
        metadata.is_scanned = ocr_extracted_pages > text_extracted_pages
        metadata.processing_time = time.time() - start_time
        
        logger.info(f"Document processing complete: {text_extracted_pages} text pages, {ocr_extracted_pages} OCR pages")
        logger.info(f"Processing time: {metadata.processing_time:.2f} seconds")
        
        return metadata, final_pages
    
    def get_processing_summary(self, metadata: DocumentMetadata, pages: List[PageContent]) -> Dict[str, Any]:
        """
        Generate a processing summary.
        
        Args:
            metadata: Document metadata
            pages: List of page contents
        
        Returns:
            Processing summary dictionary
        """
        total_text_length = sum(len(page.text) for page in pages)
        avg_text_per_page = total_text_length / len(pages) if pages else 0
        
        extraction_methods = {}
        for page in pages:
            method = page.extraction_method
            extraction_methods[method] = extraction_methods.get(method, 0) + 1
        
        return {
            "filename": metadata.filename,
            "total_pages": metadata.total_pages,
            "file_size_mb": metadata.file_size_mb,
            "total_text_length": total_text_length,
            "avg_text_per_page": avg_text_per_page,
            "extraction_methods": extraction_methods,
            "is_scanned": metadata.is_scanned,
            "processing_time": metadata.processing_time,
            "text_pages": metadata.text_pages,
            "ocr_pages": metadata.ocr_pages
        }


# Example usage and testing
if __name__ == "__main__":
    import io
    
    # Test the document processor
    processor = DocumentProcessor()
    
    # Test with a sample PDF (you'll need to provide a real PDF file)
    test_file = "sample.pdf"
    
    if os.path.exists(test_file):
        try:
            metadata, pages = processor.process_document(test_file)
            
            print(f"📄 Document: {metadata.filename}")
            print(f"📊 Pages: {metadata.total_pages}")
            print(f"📏 Size: {metadata.file_size_mb:.2f}MB")
            print(f"⏱️ Processing time: {metadata.processing_time:.2f}s")
            print(f"🔍 Text pages: {metadata.text_pages}, OCR pages: {metadata.ocr_pages}")
            print(f"📝 Is scanned: {metadata.is_scanned}")
            
            summary = processor.get_processing_summary(metadata, pages)
            print(f"\n📋 Summary: {summary}")
            
            # Show first page content
            if pages:
                first_page = pages[0]
                print(f"\n📄 First page ({first_page.extraction_method}):")
                print(f"Text length: {len(first_page.text)} characters")
                print(f"Preview: {first_page.text[:200]}...")
        
        except Exception as e:
            print(f"❌ Error processing document: {e}")
    else:
        print(f"❌ Test file not found: {test_file}")
        print("Please provide a PDF file to test the document processor.") 