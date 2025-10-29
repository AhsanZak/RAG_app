"""
Embedding Service
Generates embeddings for text using sentence-transformers
Supports dynamic model loading and switching
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import numpy as np
import os
import threading

# Try to ensure huggingface_hub is available
try:
    import huggingface_hub
    from huggingface_hub import snapshot_download
except ImportError:
    print("Warning: huggingface_hub not found. Installing it might be required.")
    huggingface_hub = None
    snapshot_download = None


class EmbeddingService:
    """Service for generating text embeddings with dynamic model loading"""
    
    def __init__(self, default_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding service
        
        Args:
            default_model_name: Default model name to use
        """
        self.default_model_name = default_model_name
        self.loaded_models: Dict[str, SentenceTransformer] = {}
        self.loading_models: Dict[str, threading.Lock] = {}
        self._load_model(default_model_name)
    
    def _load_model(self, model_name: str):
        """Load a sentence transformer model"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # Use lock to prevent concurrent loading of same model
        if model_name not in self.loading_models:
            self.loading_models[model_name] = threading.Lock()
        
        with self.loading_models[model_name]:
            # Double-check after acquiring lock
            if model_name in self.loaded_models:
                return self.loaded_models[model_name]
            
            try:
                print(f"Loading embedding model: {model_name}")
                model = SentenceTransformer(model_name)
                self.loaded_models[model_name] = model
                print(f"Embedding model '{model_name}' loaded successfully")
                return model
            except Exception as e:
                raise Exception(f"Failed to load embedding model '{model_name}': {str(e)}")
    
    def download_model(self, model_name: str) -> Dict:
        """
        Download a model from HuggingFace Hub
        
        Args:
            model_name: HuggingFace model identifier
            
        Returns:
            Dictionary with download status
        """
        try:
            print(f"Downloading embedding model: {model_name}")
            # Download model using sentence-transformers (handles download automatically)
            # But we can also pre-download using huggingface_hub
            if snapshot_download:
                try:
                    # Try to download model files
                    model_path = snapshot_download(
                        repo_id=model_name,
                        repo_type="model"
                    )
                    return {
                        "success": True,
                        "message": f"Model '{model_name}' downloaded successfully",
                        "model_path": model_path
                    }
                except Exception as e:
                    # If direct download fails, sentence-transformers will download on first use
                    print(f"Direct download failed, will download on first use: {str(e)}")
                    return {
                        "success": True,
                        "message": f"Model '{model_name}' will be downloaded on first use",
                        "model_path": None
                    }
            else:
                return {
                    "success": True,
                    "message": f"Model '{model_name}' will be downloaded on first use",
                    "model_path": None
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to download model: {str(e)}"
            }
    
    def generate_embeddings(self, texts: List[str], model_name: Optional[str] = None) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            model_name: Optional model name to use (defaults to default model)
            
        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        model_name = model_name or self.default_model_name
        model = self._load_model(model_name)
        
        try:
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10
            )
            return embeddings
        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def generate_embedding(self, text: str, model_name: Optional[str] = None) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text string to embed
            model_name: Optional model name to use
            
        Returns:
            numpy array of embedding
        """
        return self.generate_embeddings([text], model_name)[0]
    
    def get_embedding_dimension(self, model_name: Optional[str] = None) -> int:
        """Get the dimension of the embeddings"""
        model_name = model_name or self.default_model_name
        model = self._load_model(model_name)
        
        if model is None:
            return 384  # Default for all-MiniLM-L6-v2
        return model.get_sentence_embedding_dimension()
    
    def unload_model(self, model_name: str):
        """Unload a model from memory"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            print(f"Unloaded model: {model_name}")
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        return list(self.loaded_models.keys())
