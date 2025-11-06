"""
Run server without reload to avoid Windows DLL issues with PyTorch
"""
import uvicorn
import sys
import os

# Set environment variables to help with DLL loading
os.environ.setdefault('PYTORCH_NO_CUDA_MEMORY_CACHING', '1')

if __name__ == "__main__":
    # Run without reload to avoid multiprocessing DLL issues
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload to avoid DLL issues
        log_level="info"
    )

