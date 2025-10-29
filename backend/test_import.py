"""Test script to verify embedding models can be imported"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from config.embedding_models import AVAILABLE_EMBEDDINGS
    print(f"SUCCESS: Imported {len(AVAILABLE_EMBEDDINGS)} models")
    print(f"First 3 models: {list(AVAILABLE_EMBEDDINGS.keys())[:3]}")
except Exception as e:
    print(f"FAILED: {str(e)}")
    import traceback
    traceback.print_exc()

