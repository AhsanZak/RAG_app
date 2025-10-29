"""
PDF Processing Service
Extracts text from PDF files
"""

import pdfplumber
import PyPDF2
from typing import List, Dict
import io
import os


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
            # Try pdfplumber first (better for complex layouts)
            try:
                return self._extract_with_pdfplumber(pdf_bytes, filename)
            except Exception as e:
                # Fallback to PyPDF2
                return self._extract_with_pypdf2(pdf_bytes, filename)
        except Exception as e:
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

