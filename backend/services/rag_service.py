"""
RAG Service
Combines retrieval and generation for RAG-based chat
"""

from typing import Dict, List, Optional
from services.embedding_service import EmbeddingService
from services.chromadb_service import ChromaDBService
from services.llm_service import LLMService
import numpy as np


class RAGService:
    """Service for RAG (Retrieval-Augmented Generation)"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.chromadb_service = ChromaDBService()
        self.llm_service = LLMService()
    
    def query(
        self,
        query_text: str,
        collection_name: str,
        model_config: Dict,
        n_results: int = 5
    ) -> Dict:
        """
        Query using RAG: retrieve relevant context and generate response
        
        Args:
            query_text: User query/question
            collection_name: ChromaDB collection name
            model_config: LLM model configuration
            n_results: Number of documents to retrieve
            
        Returns:
            Dictionary with response and sources
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.generate_embedding(query_text)
            
            # Retrieve relevant documents
            results = self.chromadb_service.query_collection(
                collection_name=collection_name,
                query_embeddings=np.array([query_embedding]),
                n_results=n_results
            )
            
            # Extract documents and metadata
            documents = results.get('documents', [[]])[0] if results.get('documents') else []
            metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
            
            # Build context from retrieved documents
            context = self._build_context(documents, metadatas)
            
            # Generate response using LLM
            response = self.llm_service.generate_response(
                prompt=query_text,
                model_config=model_config,
                context=context
            )
            
            # Extract source filenames
            sources = self._extract_sources(metadatas)
            
            return {
                'response': response,
                'sources': sources,
                'retrieved_documents': documents[:3],  # Return top 3 for reference
                'retrieval_count': len(documents)
            }
        except Exception as e:
            raise Exception(f"RAG query failed: {str(e)}")
    
    def _build_context(self, documents: List[str], metadatas: List[Dict]) -> str:
        """
        Build context string from retrieved documents
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dicts
            
        Returns:
            Formatted context string
        """
        if not documents:
            return ""
        
        context_parts = []
        for i, (doc, metadata) in enumerate(zip(documents, metadatas), 1):
            filename = metadata.get('filename', 'unknown')
            page = metadata.get('page_number', '?')
            context_parts.append(f"[Document {i} - {filename}, Page {page}]\n{doc}\n")
        
        return "\n".join(context_parts)
    
    def _extract_sources(self, metadatas: List[Dict]) -> List[str]:
        """
        Extract unique source filenames from metadata
        
        Args:
            metadatas: List of metadata dicts
            
        Returns:
            List of unique source filenames
        """
        sources = set()
        for metadata in metadatas:
            filename = metadata.get('filename', 'unknown')
            sources.add(filename)
        return list(sources)

