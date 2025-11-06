"""
ChromaDB Service
Manages vector storage and retrieval using ChromaDB
"""

# Lazy import to avoid DLL issues at module load time
# chromadb will be imported when first needed
from typing import List, Dict, Optional, Any
import numpy as np
import os
from pathlib import Path


class ChromaDBService:
    """Service for managing ChromaDB collections and vector operations"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB service
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Lazy initialization - don't import chromadb at init to avoid DLL issues
        self.client = None
        self._chromadb = None
        self._Settings = None
    
    def _get_chromadb(self):
        """Lazy import of chromadb - only when actually needed"""
        if self._chromadb is None:
            try:
                import chromadb
                from chromadb.config import Settings
                self._chromadb = chromadb
                self._Settings = Settings
            except (OSError, ImportError) as e:
                error_msg = str(e)
                if "DLL" in error_msg or "dll" in error_msg.lower() or "onnxruntime" in error_msg.lower():
                    raise ImportError(
                        f"ChromaDB/ONNX DLL loading failed: {error_msg}\n"
                        "Solutions:\n"
                        "1. Install Microsoft Visual C++ Redistributable 2015-2022\n"
                        "2. Restart your computer after installing redistributables"
                    )
                raise
        return self._chromadb, self._Settings
    
    def _ensure_client(self):
        """Ensure ChromaDB client is initialized"""
        if self.client is None:
            chromadb, Settings = self._get_chromadb()
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            print(f"ChromaDB initialized at: {self.persist_directory}")
    
    def create_collection(self, collection_name: str, metadata: Optional[Dict] = None) -> Any:
        """
        Create or get a ChromaDB collection
        
        Args:
            collection_name: Name of the collection
            metadata: Optional metadata for the collection
            
        Returns:
            ChromaDB Collection object
        """
        self._ensure_client()
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=collection_name)
            print(f"Collection '{collection_name}' already exists")
            return collection
        except Exception:
            # Create new collection if it doesn't exist
            # ChromaDB requires non-empty metadata or None
            collection_metadata = metadata if metadata and len(metadata) > 0 else None
            collection = self.client.create_collection(
                name=collection_name,
                metadata=collection_metadata
            )
            print(f"Created new collection '{collection_name}'")
            return collection
    
    def add_documents(
        self,
        collection_name: str,
        texts: List[str],
        embeddings: Optional[np.ndarray] = None,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to a collection
        
        Args:
            collection_name: Name of the collection
            texts: List of text documents
            embeddings: Optional pre-computed embeddings (numpy array)
            metadatas: Optional list of metadata dicts for each document
            ids: Optional list of IDs for each document
            
        Returns:
            List of document IDs
        """
        collection = self.create_collection(collection_name)
        
        if ids is None:
            ids = [f"doc_{i}_{collection_name}" for i in range(len(texts))]
        
        if metadatas is None:
            # Create non-empty metadata for each document (ChromaDB requirement)
            metadatas = [{"index": i, "source": "document"} for i in range(len(texts))]
        else:
            # Ensure all metadata dicts are non-empty
            metadatas = [
                meta if meta and isinstance(meta, dict) and len(meta) > 0 
                else {"index": i, "source": "document"}
                for i, meta in enumerate(metadatas)
            ]
        
        try:
            if embeddings is not None:
                # Use pre-computed embeddings
                collection.add(
                    embeddings=embeddings.tolist(),
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
            else:
                # Let ChromaDB generate embeddings
                collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
            
            print(f"Added {len(texts)} documents to collection '{collection_name}'")
            return ids
        except Exception as e:
            raise Exception(f"Failed to add documents to collection: {str(e)}")
    
    def query_collection(
        self,
        collection_name: str,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[np.ndarray] = None,
        n_results: int = 5,
        where: Optional[Dict] = None,
        include: Optional[List[str]] = None
    ) -> Dict:
        """
        Query a collection for similar documents
        
        Args:
            collection_name: Name of the collection
            query_texts: Query text(s) to search for
            query_embeddings: Optional pre-computed query embeddings
            n_results: Number of results to return
            where: Optional metadata filter
            include: What to include in results (default: ['documents', 'metadatas', 'distances'])
            
        Returns:
            Dictionary with query results
        """
        self._ensure_client()
        try:
            collection = self.client.get_collection(name=collection_name)
            
            if include is None:
                include = ['documents', 'metadatas', 'distances']
            
            if query_embeddings is not None:
                results = collection.query(
                    query_embeddings=query_embeddings.tolist(),
                    n_results=n_results,
                    where=where,
                    include=include
                )
            elif query_texts:
                results = collection.query(
                    query_texts=query_texts,
                    n_results=n_results,
                    where=where,
                    include=include
                )
            else:
                raise ValueError("Either query_texts or query_embeddings must be provided")
            
            return results
        except Exception as e:
            error_msg = str(e)
            if "does not exist" in error_msg or "not found" in error_msg.lower():
                raise Exception(f"Collection '{collection_name}' does not exist. Please process PDF files first for this session.")
            raise Exception(f"Failed to query collection '{collection_name}': {error_msg}")
    
    def delete_collection(self, collection_name: str):
        """Delete a collection"""
        self._ensure_client()
        try:
            self.client.delete_collection(name=collection_name)
            print(f"Deleted collection '{collection_name}'")
        except Exception as e:
            raise Exception(f"Failed to delete collection: {str(e)}")
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        self._ensure_client()
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            raise Exception(f"Failed to list collections: {str(e)}")
    
    def get_collection_info(self, collection_name: str) -> Dict:
        """Get information about a collection"""
        self._ensure_client()
        try:
            collection = self.client.get_collection(name=collection_name)
            count = collection.count()
            return {
                'name': collection_name,
                'count': count
            }
        except Exception as e:
            raise Exception(f"Failed to get collection info: {str(e)}")

