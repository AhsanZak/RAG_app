# Embedding Service Troubleshooting Guide

## 🚀 Quick Fix (Recommended)

If you're seeing DLL errors, the **easiest solution** is to use remote embeddings:

1. **Get a free Hugging Face API token**: https://huggingface.co/settings/tokens
2. **Set the environment variable** (choose your shell):
   ```powershell
   # Windows PowerShell
   $env:HUGGINGFACE_API_TOKEN="your_token_here"
   ```
   ```cmd
   # Windows CMD
   set HUGGINGFACE_API_TOKEN=your_token_here
   ```
   ```bash
   # Linux/Mac
   export HUGGINGFACE_API_TOKEN="your_token_here"
   ```
3. **Restart your application** - the system will automatically use remote embeddings when DLL errors occur

That's it! The system will automatically detect DLL errors and fall back to remote embeddings without any additional configuration.

---

## Problem: Intermittent PyTorch DLL Loading Errors

On Windows, you may encounter intermittent PyTorch DLL loading errors:
```
[WinError 1114] A dynamic link library (DLL) initialization routine failed.
Error loading "torch\lib\c10.dll" or one of its dependencies.
```

This happens because:
- Windows sometimes has issues loading PyTorch DLLs, especially after system updates or when multiple processes try to load them
- The error is **intermittent** - it may work sometimes and fail other times
- This is a known Windows + PyTorch compatibility issue

## Solution: Two Options Available

The embedding service now has **automatic fallback** and **two configuration options**:

### Option 1: Use Remote Embeddings (Recommended for Reliability)

Use Hugging Face Inference API for embeddings. This avoids DLL issues entirely.

**Setup:**
1. Get a free Hugging Face API token: https://huggingface.co/settings/tokens
2. Set the environment variable:
   ```bash
   # Windows PowerShell
   $env:HUGGINGFACE_API_TOKEN="your_token_here"
   
   # Windows CMD
   set HUGGINGFACE_API_TOKEN=your_token_here
   
   # Linux/Mac
   export HUGGINGFACE_API_TOKEN="your_token_here"
   ```

3. (Optional) Prefer remote embeddings by default:
   ```bash
   $env:PREFER_REMOTE_EMBEDDINGS="true"
   ```

**How it works:**
- When local PyTorch fails, the system **automatically detects** the `HUGGINGFACE_API_TOKEN` and falls back to Hugging Face API
- The fallback happens **immediately** when DLL errors are detected (no need to wait for all retries)
- If `PREFER_REMOTE_EMBEDDINGS=true`, it uses remote API first (no local attempt)
- No DLL issues, but requires internet connection
- **No additional configuration needed** - just set the token and the system handles the rest

### Option 2: Fix Local PyTorch Installation

Fix the PyTorch DLL issue to use local embeddings.

**Steps:**
1. Install Microsoft Visual C++ Redistributable 2015-2022:
   - Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Install and restart your machine

2. Reinstall PyTorch (CPU-only version is more stable on Windows):
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

3. Restart your machine after installation

**How it works:**
- System tries to load local PyTorch model
- If it fails with DLL error and `HUGGINGFACE_API_TOKEN` is set, **immediately** attempts remote API (smart fallback)
- Otherwise, automatically retries up to 3 times with exponential backoff
- If all retries fail and `HUGGINGFACE_API_TOKEN` is set, falls back to remote API
- If no token is set, shows helpful error message with setup instructions

## Automatic Retry Logic

The system now includes:
- **Retry mechanism**: Up to 3 attempts with exponential backoff (0.5s, 1s, 2s delays)
- **Failure caching**: Remembers which models failed to avoid repeated attempts
- **Automatic fallback**: If local fails and `HUGGINGFACE_API_TOKEN` is set, uses remote API automatically
- **Better error messages**: Clear guidance on what to do when errors occur

## Configuration Summary

| Environment Variable | Purpose | Required |
|---------------------|---------|----------|
| `HUGGINGFACE_API_TOKEN` | Enables remote embedding fallback | No (but recommended) |
| `HF_API_TOKEN` | Alternative name for Hugging Face token | No |
| `PREFER_REMOTE_EMBEDDINGS` | Use remote API first (skip local attempts) | No |

## How It Works

1. **First attempt**: Try to load local SentenceTransformer model
2. **On DLL error**: Retry up to 3 times with delays
3. **Automatic fallback**: If `HUGGINGFACE_API_TOKEN` is set, automatically detects DLL errors and uses remote API **without waiting for all retries**
4. **After retries**: If local loading still fails and token is available, uses remote API
5. **If no token**: Show helpful error message with instructions
6. **Future requests**: If a model previously failed and `PREFER_REMOTE_EMBEDDINGS=true`, skip local attempt

**Key Improvement**: The system now automatically detects when `HUGGINGFACE_API_TOKEN` is set and will immediately attempt remote embeddings when DLL errors occur, rather than waiting for all retries to complete.

## Testing

To test if your setup works:

1. **Test local embeddings** (if PyTorch is working):
   ```python
   from services.embedding_service import EmbeddingService
   service = EmbeddingService()
   embeddings = service.generate_embeddings(["test text"])
   print(f"Embeddings shape: {embeddings.shape}")
   ```

2. **Test remote embeddings** (requires token):
   ```bash
   $env:HUGGINGFACE_API_TOKEN="your_token"
   $env:PREFER_REMOTE_EMBEDDINGS="true"
   ```
   Then run the same Python code - it should use remote API.

## Troubleshooting

**Q: Still getting DLL errors even with retries?**
- Set `HUGGINGFACE_API_TOKEN` for automatic fallback
- Or set `PREFER_REMOTE_EMBEDDINGS=true` to skip local attempts entirely

**Q: Remote embeddings not working?**
- Check your internet connection
- Verify your API token is valid at https://huggingface.co/settings/tokens
- Check that `huggingface-hub` package is installed: `pip install huggingface-hub`

**Q: Want to force local embeddings only?**
- Don't set `HUGGINGFACE_API_TOKEN`
- Fix PyTorch DLL issues using Option 2 above

**Q: Why does it work sometimes but not others?**
- Windows DLL loading can be affected by:
  - System load
  - Other processes using PyTorch
  - System updates
  - Memory pressure
- This is why the retry mechanism helps - sometimes the second or third attempt succeeds

