"""
Available Embedding Models Configuration
List of embedding models that can be used for RAG
"""

AVAILABLE_EMBEDDINGS = {
    "all-MiniLM-L6-v2": {
        "provider": "sentence-transformers",
        "languages": ["en"],
        "description": "Fast small English model",
        "dimension": 384,
        "display_name": "MiniLM-L6-v2 (English)",
        "is_default": True
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "provider": "sentence-transformers",
        "languages": ["en"],
        "description": "Fast small English model",
        "dimension": 384,
        "display_name": "MiniLM-L6-v2 (English)",
        "is_default": True
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "provider": "sentence-transformers",
        "languages": ["en"],
        "description": "Higher quality English embedding model",
        "dimension": 768,
        "display_name": "MPNet Base v2 (English)",
        "is_default": False
    },
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "provider": "sentence-transformers",
        "languages": ["multi"],
        "description": "Multilingual embedding model supporting 50+ languages",
        "dimension": 384,
        "display_name": "Multilingual MiniLM (50+ languages)",
        "is_default": False
    },
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
        "provider": "sentence-transformers",
        "languages": ["multi"],
        "description": "High-quality multilingual embedding model",
        "dimension": 768,
        "display_name": "Multilingual MPNet (50+ languages)",
        "is_default": False
    },
    "intfloat/multilingual-e5-large": {
        "provider": "huggingface",
        "languages": ["multi"],
        "description": "High-quality multilingual embeddings",
        "dimension": 1024,
        "display_name": "Multilingual E5 Large",
        "is_default": False
    },
    "intfloat/multilingual-e5-base": {
        "provider": "huggingface",
        "languages": ["multi"],
        "description": "Base multilingual E5 model",
        "dimension": 768,
        "display_name": "Multilingual E5 Base",
        "is_default": False
    },
    "BAAI/bge-large-en": {
        "provider": "huggingface",
        "languages": ["en"],
        "description": "High-accuracy English embeddings",
        "dimension": 1024,
        "display_name": "BGE Large EN",
        "is_default": False
    },
    "BAAI/bge-base-en": {
        "provider": "huggingface",
        "languages": ["en"],
        "description": "Base English BGE model",
        "dimension": 768,
        "display_name": "BGE Base EN",
        "is_default": False
    },
    "BAAI/bge-large-en-v1.5": {
        "provider": "huggingface",
        "languages": ["en"],
        "description": "High-accuracy English embeddings v1.5",
        "dimension": 1024,
        "display_name": "BGE Large EN v1.5",
        "is_default": False
    },
    "BAAI/bge-m3": {
        "provider": "huggingface",
        "languages": ["multi"],
        "description": "Multilingual BGE model",
        "dimension": 1024,
        "display_name": "BGE M3 Multilingual",
        "is_default": False
    },
    "aubmindlab/bert-base-arabertv2": {
        "provider": "huggingface",
        "languages": ["ar"],
        "description": "Arabic-specific BERT model",
        "dimension": 768,
        "display_name": "AraBERT v2 (Arabic)",
        "is_default": False
    },
    "CAMeL-Lab/bert-base-arabic-camelbert-da": {
        "provider": "huggingface",
        "languages": ["ar"],
        "description": "Arabic CAMeLBERT model",
        "dimension": 768,
        "display_name": "CAMeLBERT (Arabic)",
        "is_default": False
    },
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
        "provider": "sentence-transformers",
        "languages": ["multi"],
        "description": "High-quality multilingual embedding model",
        "dimension": 768,
        "display_name": "Multilingual MPNet (50+ languages)",
        "is_default": False
    }
}

def get_default_embedding_model():
    """Get the default embedding model name"""
    for model_name, config in AVAILABLE_EMBEDDINGS.items():
        if config.get("is_default", False):
            return model_name
    return "all-MiniLM-L6-v2"  # Fallback

def get_embedding_model_info(model_name: str):
    """Get information about a specific embedding model"""
    return AVAILABLE_EMBEDDINGS.get(model_name)

def list_embedding_models(language: str = None):
    """List all available embedding models, optionally filtered by language"""
    if language:
        if language == "multi":
            return {k: v for k, v in AVAILABLE_EMBEDDINGS.items() if "multi" in v.get("languages", [])}
        else:
            return {k: v for k, v in AVAILABLE_EMBEDDINGS.items() if language in v.get("languages", [])}
    return AVAILABLE_EMBEDDINGS

def get_models_by_provider(provider: str):
    """Get all models from a specific provider"""
    return {k: v for k, v in AVAILABLE_EMBEDDINGS.items() if v.get("provider") == provider}

