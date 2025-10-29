"""
Embedding Service
Generates embeddings for text using sentence-transformers
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import os

# Try to ensure huggingface_hub is available
try:
    import huggingface_hub
except ImportError:
    print("Warning: huggingface_hub not found. Installing it might be required.")
    huggingface_hub = None


class EmbeddingService:
    """Service for generating text embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding service
        
        Args:
            model_name: Name of the sentence transformer model to use
                - "all-MiniLM-L6-v2": Fast and efficient (default)
                - "all-mpnet-base-v2": Higher quality but slower
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Embedding model loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load embedding model: {str(e)}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10
            )
            return embeddings
        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text string to embed
            
        Returns:
            numpy array of embedding
        """
        return self.generate_embeddings([text])[0]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        if self.model is None:
            return 384  # Default for all-MiniLM-L6-v2
        return self.model.get_sentence_embedding_dimension()

