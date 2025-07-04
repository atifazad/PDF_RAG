#!/usr/bin/env python3
"""
Integration test script for the document processor.
This script allows you to test the document processor with actual PDF files.
"""

import os
import sys
from pathlib import Path

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'backend'))

from document_processor import DocumentProcessor


def test_document_processor():
    """Test the document processor with a PDF file."""
    
    # Initialize the processor
    processor = DocumentProcessor()
    
    # Check if there are any PDF files in the current directory or data directory
    pdf_files = []
    
    # Look in current directory
    for file in os.listdir('.'):
        if file.lower().endswith('.pdf'):
            pdf_files.append(file)
    
    # Look in data directory if it exists
    if os.path.exists('data'):
        for file in os.listdir('data'):
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join('data', file))
    
    if not pdf_files:
        print("❌ No PDF files found in current directory or data/ directory.")
        print("Please place a PDF file in the current directory or data/ directory to test.")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF file(s):")
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"  {i}. {pdf_file}")
    
    # Test with the first PDF file
    test_file = pdf_files[0]
    print(f"\n🔍 Testing with: {test_file}")
    
    try:
        # Validate the file
        print("\n1. Validating file...")
        is_valid, error_msg = processor.validate_file(test_file)
        
        if not is_valid:
            print(f"❌ File validation failed: {error_msg}")
            return
        
        print("✅ File validation passed!")
        
        # Extract metadata
        print("\n2. Extracting metadata...")
        metadata = processor.extract_metadata(test_file)
        print(f"✅ Metadata extracted:")
        print(f"   - Filename: {metadata.filename}")
        print(f"   - Total pages: {metadata.total_pages}")
        print(f"   - File size: {metadata.file_size_mb:.2f} MB")
        print(f"   - Title: {metadata.title or 'N/A'}")
        print(f"   - Author: {metadata.author or 'N/A'}")
        print(f"   - Is scanned: {metadata.is_scanned}")
        
        # Process the document
        print("\n3. Processing document...")
        metadata, pages = processor.process_document(test_file)
        
        if metadata and pages:
            print("✅ Document processing successful!")
            print(f"   - Pages processed: {len(pages)}")
            print(f"   - Total text length: {sum(len(page.text) for page in pages)} characters")
            
            # Show processing summary
            summary = processor.get_processing_summary(metadata, pages)
            print(f"\n📊 Processing Summary:")
            print(f"   - Text pages: {summary['text_pages']}")
            print(f"   - OCR pages: {summary['ocr_pages']}")
            print(f"   - Total pages: {summary['total_pages']}")
            print(f"   - Processing time: {summary['processing_time']:.2f} seconds")
            
            # Show first few pages
            print(f"\n📄 First 3 pages preview:")
            for i, page in enumerate(pages[:3]):
                print(f"   Page {page.page_number} ({page.extraction_method}):")
                text_preview = page.text[:100] + "..." if len(page.text) > 100 else page.text
                print(f"     {text_preview}")
                if page.confidence < 1.0:
                    print(f"     Confidence: {page.confidence:.2f}")
                print()
                
        else:
            print(f"❌ Document processing failed: No metadata or pages returned")
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()


def test_with_sample_pdf():
    """Create a simple test PDF if no PDF files are available."""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        test_pdf = "test_sample.pdf"
        print(f"📝 Creating sample PDF: {test_pdf}")
        
        # Create a simple PDF
        c = canvas.Canvas(test_pdf, pagesize=letter)
        c.drawString(100, 750, "This is a test PDF document.")
        c.drawString(100, 700, "It contains multiple lines of text.")
        c.drawString(100, 650, "This will be used to test the document processor.")
        c.save()
        
        print(f"✅ Created sample PDF: {test_pdf}")
        return test_pdf
        
    except ImportError:
        print("⚠️  reportlab not available. Install with: pip install reportlab")
        return None


if __name__ == "__main__":
    print("🧪 Integration Document Processor Test")
    print("=" * 50)
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Virtual environment may not be activated.")
        print("   Make sure to activate your virtual environment first.")
        print()
    
    # Test the document processor
    test_document_processor()
    
    print("\n" + "=" * 50)
    print("🏁 Test completed!") 