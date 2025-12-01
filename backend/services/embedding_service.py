"""
Embedding Service
Generates embeddings for text using sentence-transformers
Supports dynamic model loading and switching
"""

# Lazy import to avoid DLL issues at module load time
SentenceTransformer = None

from typing import List, Dict, Optional, Any
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


class RemoteSentenceTransformer:
    """
    Lightweight wrapper that uses Hugging Face Inference API for embeddings
    when local SentenceTransformer (Torch) cannot be loaded.
    """

    def __init__(self, model_name: str, token: str):
        from huggingface_hub import InferenceClient

        self.model_name = model_name
        self.client = InferenceClient(model=model_name, token=token)
        self._dimension = None

    def encode(self, texts, convert_to_numpy: bool = True, show_progress_bar: bool = False):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            try:
                # feature_extraction returns either a sentence embedding or token embeddings
                result = self.client.feature_extraction(text)
            except Exception as err:
                raise RuntimeError(
                    f"Hugging Face Inference API request failed for model '{self.model_name}': {err}"
                ) from err

            # Debug info to inspect response structures
            if isinstance(result, list):
                preview = f"list(len={len(result)}) type0={type(result[0]) if result else 'n/a'}"
            elif isinstance(result, np.ndarray):
                preview = f"ndarray shape={result.shape}"
            elif isinstance(result, dict):
                preview = f"dict keys={list(result.keys())}"
            else:
                preview = str(type(result))
            print(f"[DEBUG] HF raw response preview: {preview}")

            if isinstance(result, dict):
                if "embeddings" in result:
                    result = result["embeddings"]
                elif "outputs" in result:
                    result = result["outputs"]
                else:
                    raise RuntimeError(f"Unexpected dict response from Hugging Face API: {result.keys()}")

            if isinstance(result, np.ndarray):
                arr = result.astype(np.float32)
            else:
                if not isinstance(result, list) or len(result) == 0:
                    raise RuntimeError(f"Unexpected embedding response format from Hugging Face API (type={type(result)})")
                arr = np.array(result, dtype=np.float32)
            print(f"[DEBUG] HF response converted to ndarray shape={arr.shape}")

            # Normalize different response shapes to a single pooled vector
            if arr.ndim == 3:
                # (batch, tokens, hidden)
                pooled = arr.mean(axis=1)
            elif arr.ndim == 2:
                if arr.shape[0] == 1:
                    pooled = arr
                else:
                    pooled = arr.mean(axis=0, keepdims=True)
            elif arr.ndim == 1:
                pooled = arr[np.newaxis, :]
            else:
                raise RuntimeError(f"Unexpected embedding array shape {arr.shape} from Hugging Face API.")

            print(f"[DEBUG] HF pooled embedding shape={pooled.shape}")
            embeddings.append(pooled[0])

        if self._dimension is None and embeddings:
            self._dimension = embeddings[0].shape[0]

        stacked = np.vstack(embeddings).astype(np.float32)
        print(f"[DEBUG] HF final stacked embeddings shape={stacked.shape}")
        if convert_to_numpy:
            return stacked
        return stacked.tolist()

    def get_sentence_embedding_dimension(self):
        if self._dimension is None:
            probe = self.encode(["probe"], convert_to_numpy=True)
            self._dimension = probe.shape[1]
        return int(self._dimension)


def _import_sentence_transformer():
    """Lazy import for SentenceTransformer to avoid DLL load issues on Windows."""
    global SentenceTransformer
    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as ST
            SentenceTransformer = ST
        except (OSError, ImportError) as e:
            error_msg = str(e)
            if "DLL" in error_msg or "dll" in error_msg.lower() or "1114" in error_msg:
                print(f"[ERROR] PyTorch DLL loading issue: {error_msg}")
                print("[INFO] Suggested fixes:")
                print("  1. Install Microsoft Visual C++ Redistributable 2015-2022")
                print("  2. Reinstall PyTorch: pip uninstall torch && pip install torch")
                print("  3. Use CPU-only torch build: pip install torch --index-url https://download.pytorch.org/whl/cpu")
                print("  4. Restart your machine after installing dependencies")
                raise ImportError("SentenceTransformer not available due to DLL load issue") from e
            raise
    return SentenceTransformer


class EmbeddingService:
    """Service for generating text embeddings with dynamic model loading"""
    
    def __init__(self, default_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding service
        
        Args:
            default_model_name: Default model name to use
        """
        self.default_model_name = default_model_name
        self.loaded_models: Dict[str, Any] = {}
        self.loading_models: Dict[str, threading.Lock] = {}
        # Do not load model during init to avoid DLL issues; load on demand.
    
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
                ST = _import_sentence_transformer()
                model = ST(model_name)
                self.loaded_models[model_name] = model
                print(f"Embedding model '{model_name}' loaded successfully")
                return model
            except Exception as e:
                # Attempt remote fallback via Hugging Face Inference API
                print(f"[WARNING] Local SentenceTransformer load failed: {e}")
                print("[INFO] Attempting fallback via Hugging Face Inference API.")
                fallback_model = self._create_remote_model(model_name)
                if fallback_model is not None:
                    self.loaded_models[model_name] = fallback_model
                    print(f"Using Hugging Face Inference API for model '{model_name}'.")
                    return fallback_model
                raise Exception(
                    f"Failed to load embedding model '{model_name}': {str(e)}. "
                    "No fallback embedding provider available."
                )

    def _create_remote_model(self, model_name: str) -> Optional[RemoteSentenceTransformer]:
        """
        Attempt to create a remote embedding model using Hugging Face Inference API.
        Requires HUGGINGFACE_API_TOKEN or HF_API_TOKEN environment variable.
        """
        token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_API_TOKEN")
        if not token:
            print("[ERROR] Missing Hugging Face API token. Set HUGGINGFACE_API_TOKEN to enable remote embeddings.")
            return None

        try:
            fallback_model = RemoteSentenceTransformer(model_name, token)
            # Probe once to verify connectivity/dimensions
            fallback_model.get_sentence_embedding_dimension()
            return fallback_model
        except ImportError:
            print("[ERROR] huggingface_hub package is required for remote embeddings. Install with 'pip install huggingface-hub'.")
        except Exception as err:
            print(f"[ERROR] Unable to initialize remote embedding model '{model_name}': {err}")
        return None
    
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
        
        # Filter out None or empty texts
        texts = [str(t) if t is not None else "" for t in texts]
        if not texts:
            return np.array([])
        
        model_name = model_name or self.default_model_name
        
        try:
            model = self._load_model(model_name)
        except Exception as load_error:
            error_msg = str(load_error).lower()
            if "dll" in error_msg or "import" in error_msg:
                raise Exception(
                    f"DLL/Import error loading embedding model '{model_name}': {str(load_error)}. "
                    "This is likely a PyTorch/SentenceTransformers issue. "
                    "Consider: 1) Installing Microsoft Visual C++ Redistributable, "
                    "2) Setting HUGGINGFACE_API_TOKEN for remote embeddings, or "
                    "3) Reinstalling PyTorch."
                ) from load_error
            raise Exception(f"Failed to load embedding model '{model_name}': {str(load_error)}") from load_error
        
        try:
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10
            )
            
            # Validate embeddings
            if embeddings is None:
                raise Exception("Model.encode() returned None")
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            if len(embeddings) == 0:
                raise Exception("Model.encode() returned empty array")
            if len(embeddings) != len(texts):
                raise Exception(f"Embedding count mismatch: {len(embeddings)} != {len(texts)}")
            
            return embeddings
        except Exception as e:
            error_msg = str(e).lower()
            if "dll" in error_msg or "import" in error_msg or "load" in error_msg:
                raise Exception(
                    f"DLL/Import error during embedding generation: {str(e)}. "
                    "This might be a PyTorch/SentenceTransformers issue. "
                    "Consider using Hugging Face Inference API as fallback by setting HUGGINGFACE_API_TOKEN."
                ) from e
            raise Exception(f"Failed to generate embeddings: {str(e)}") from e
    
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
