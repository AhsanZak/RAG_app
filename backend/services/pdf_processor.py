"""
PDF Processing Service
Extracts text from PDF files
"""

import pdfplumber
import PyPDF2
from typing import List, Dict
import io
import os
from typing import Optional

# Optional OCR imports
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


class PDFProcessor:
    """Service for processing PDF files and extracting text"""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def extract_text_from_pdf(self, pdf_bytes: bytes, filename: str = None) -> Dict:
        """
        Extract text from PDF file
        
        Args:
            pdf_bytes: PDF file content as bytes
            filename: Optional filename for error tracking
            
        Returns:
            Dictionary with extracted text chunks and metadata
        """
        try:
            print(f"[PDF] extract_text_from_pdf: filename={filename}, bytes_len={len(pdf_bytes) if pdf_bytes else 0}")
            # First pass: native text extraction
            try:
                result = self._extract_with_pdfplumber(pdf_bytes, filename)
                print(f"[PDF] pdfplumber: chunks={result.get('total_chunks')}, pages={result.get('metadata',{}).get('total_pages')}")
            except Exception as e:
                print(f"[PDF] pdfplumber failed: {e}")
                result = self._extract_with_pypdf2(pdf_bytes, filename)
                print(f"[PDF] pypdf2: chunks={result.get('total_chunks')}, pages={result.get('metadata',{}).get('total_pages')}")

            # If no text found, attempt OCR fallback
            if result['total_chunks'] == 0:
                print(f"[PDF] no native text found, OCR_AVAILABLE={OCR_AVAILABLE}")
                ocr_result = self._extract_with_ocr(pdf_bytes, filename)
                print(f"[PDF] OCR result: chunks={ocr_result.get('total_chunks')}, method={ocr_result.get('metadata',{}).get('extraction_method')}")
                return ocr_result
            return result
        except Exception as e:
            print(f"[PDF] extract_text_from_pdf error: {e}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def _extract_with_pdfplumber(self, pdf_bytes: bytes, filename: str = None) -> Dict:
        """Extract text using pdfplumber"""
        chunks = []
        metadata = {
            'total_pages': 0,
            'extraction_method': 'pdfplumber'
        }
        
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            metadata['total_pages'] = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    chunks.append({
                        'page': page_num,
                        'text': text.strip(),
                        'metadata': {
                            'page_number': page_num,
                            'filename': filename or 'unknown.pdf'
                        }
                    })
        
        return {
            'chunks': chunks,
            'metadata': metadata,
            'total_chunks': len(chunks)
        }
    
    def _extract_with_pypdf2(self, pdf_bytes: bytes, filename: str = None) -> Dict:
        """Extract text using PyPDF2 (fallback)"""
        chunks = []
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        metadata = {
            'total_pages': len(pdf_reader.pages),
            'extraction_method': 'pypdf2'
        }
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            text = page.extract_text()
            if text and text.strip():
                chunks.append({
                    'page': page_num,
                    'text': text.strip(),
                    'metadata': {
                        'page_number': page_num,
                        'filename': filename or 'unknown.pdf'
                    }
                })
        
        return {
            'chunks': chunks,
            'metadata': metadata,
            'total_chunks': len(chunks)
        }

    def _extract_with_ocr(self, pdf_bytes: bytes, filename: Optional[str] = None) -> Dict:
        """Extract text using OCR if PDF is image/scanned. Requires Tesseract installed."""
        chunks: List[Dict] = []
        metadata: Dict = {
            'total_pages': 0,
            'extraction_method': 'ocr',
            'ocr_available': OCR_AVAILABLE
        }
        if not OCR_AVAILABLE:
            print("[PDF][OCR] OCR not available in environment (pytesseract/pdf2image/Pillow missing)")
            # No OCR available in environment
            return {
                'chunks': [],
                'metadata': metadata,
                'total_chunks': 0
            }

        # Convert pages to images
        try:
            images = convert_from_bytes(pdf_bytes)
        except Exception as e:
            print(f"[PDF][OCR] convert_from_bytes failed: {e}")
            return {
                'chunks': [],
                'metadata': metadata,
                'total_chunks': 0
            }
        metadata['total_pages'] = len(images)
        print(f"[PDF][OCR] pages_as_images={len(images)}")
        for idx, img in enumerate(images, 1):
            try:
                text = pytesseract.image_to_string(img)
                if text and text.strip():
                    chunks.append({
                        'page': idx,
                        'text': text.strip(),
                        'metadata': {
                            'page_number': idx,
                            'filename': filename or 'unknown.pdf',
                            'extracted_via': 'ocr'
                        }
                    })
                else:
                    print(f"[PDF][OCR] page={idx} empty text")
            except Exception as e:
                print(f"[PDF][OCR] page={idx} OCR failed: {e}")
                continue

        return {
            'chunks': chunks,
            'metadata': metadata,
            'total_chunks': len(chunks)
        }
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """
        Split text into smaller chunks for embedding
        
        Args:
            text: Text to chunk
            chunk_size: Maximum size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks with metadata
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            chunks.append({
                'text': chunk_text,
                'start': start,
                'end': end
            })
            
            start = end - overlap  # Overlap for context
            
        return chunks

