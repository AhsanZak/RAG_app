# Fix PyTorch DLL Loading Issues on Windows

If you're experiencing DLL loading errors with PyTorch on Windows, follow these steps to fix it:

## Quick Fix (Recommended)

### Step 1: Install Microsoft Visual C++ Redistributable

Download and install the Visual C++ Redistributable:
- **Download**: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Run the installer and restart your computer

### Step 2: Reinstall PyTorch (CPU-only version)

The CPU-only version is more stable on Windows:

```powershell
# Uninstall existing PyTorch
pip uninstall torch torchvision torchaudio

# Install CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Reinstall sentence-transformers

```powershell
pip install --upgrade --force-reinstall sentence-transformers
```

### Step 4: Restart Your Machine

After installation, restart your computer to ensure all DLLs are properly loaded.

## Alternative: Use GPU Version (if you have NVIDIA GPU)

If you have an NVIDIA GPU and want to use it:

```powershell
# Uninstall existing PyTorch
pip uninstall torch torchvision torchaudio

# Install CUDA version (check https://pytorch.org for latest CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Verify Installation

Test if PyTorch loads correctly:

```python
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "from sentence_transformers import SentenceTransformer; print('SentenceTransformers loaded successfully')"
```

## What Changed

The embedding service now:
- **Prioritizes local embeddings** by default
- **Increases retries** from 3 to 5 attempts for DLL issues
- **Uses longer delays** (1-5 seconds) between retries
- **Focuses error messages** on fixing local installation
- **Only uses remote embeddings** if explicitly enabled via `PREFER_REMOTE_EMBEDDINGS=true`

## Disable Remote Embeddings (Optional)

If you want to ensure only local embeddings are used:

```powershell
# Windows PowerShell
$env:DISABLE_REMOTE_EMBEDDINGS="true"

# Windows CMD
set DISABLE_REMOTE_EMBEDDINGS=true

# Linux/Mac
export DISABLE_REMOTE_EMBEDDINGS=true
```

## Common Issues

**Issue**: Still getting DLL errors after installation
- **Solution**: Make sure you restarted your computer after installing Visual C++ Redistributable

**Issue**: "No module named 'torch'"
- **Solution**: Make sure you're in the correct virtual environment and PyTorch is installed

**Issue**: DLL errors are intermittent (work sometimes, fail other times)
- **Solution**: This is normal on Windows. The retry mechanism (5 attempts) should handle this automatically.

## Need Help?

If you continue to experience issues:
1. Check that Visual C++ Redistributable is installed
2. Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`
3. Try the CPU-only version (more stable on Windows)
4. Restart your machine after any installation

