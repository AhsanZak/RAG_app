import os
import sys
import subprocess
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VENVSCRIPTS = REPO_ROOT / 'venv' / ('Scripts' if os.name == 'nt' else 'bin')

REQUIREMENTS = REPO_ROOT / 'requirements.txt'
CHROMA_DIR = REPO_ROOT / 'chroma_db'

def run(cmd, check=True):
    print(f"[SETUP] $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)

def ensure_venv():
    venv_dir = REPO_ROOT / 'venv'
    if not venv_dir.exists():
        print("[SETUP] Creating virtual environment...")
        run([sys.executable, '-m', 'venv', 'venv'])
    else:
        print("[SETUP] venv already exists.")
    return venv_dir

def pip_exe():
    return str(VENVSCRIPTS / ('pip.exe' if os.name == 'nt' else 'pip'))

def python_exe():
    return str(VENVSCRIPTS / ('python.exe' if os.name == 'nt' else 'python'))

def install_requirements():
    if not REQUIREMENTS.exists():
        print(f"[SETUP][WARN] requirements.txt not found at {REPO_ROOT}")
        return
    print("[SETUP] Installing Python dependencies...")
    run([pip_exe(), 'install', '--upgrade', 'pip'])
    run([pip_exe(), 'install', '-r', str(REQUIREMENTS)])

def verify_ocr():
    # Verify Python OCR packages
    missing = []
    for mod in ['pdf2image', 'pytesseract', 'PIL']:
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        print(f"[SETUP][WARN] Missing Python OCR modules: {', '.join(missing)}")
        print("[SETUP][HINT] They should be installed via requirements.txt. Re-run install if needed.")

    # Verify system tools (best-effort)
    def which(cmd):
        return shutil.which(cmd) is not None

    has_tesseract = which('tesseract') or which('tesseract.exe')
    if not has_tesseract:
        print("[SETUP][WARN] Tesseract not found on PATH. OCR for scanned PDFs will not work until installed.")
    else:
        print("[SETUP] Tesseract found.")

    # Poppler is required by pdf2image
    has_pdfinfo = which('pdfinfo') or which('pdfinfo.exe')
    if not has_pdfinfo:
        print("[SETUP][WARN] Poppler (pdfinfo) not found on PATH. Install Poppler and add to PATH for pdf2image.")
    else:
        print("[SETUP] Poppler (pdfinfo) found.")

def ensure_chroma_dir():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[SETUP] Ensured ChromaDB directory at: {CHROMA_DIR}")

    # Ensure chroma_db is ignored
    gitignore = REPO_ROOT / '.gitignore'
    backend_gitignore = REPO_ROOT / '.gitignore'  # optional root
    patterns = [
        'chroma_db/\n', '**/chroma_db/**\n', '.chroma/\n', '**/.chroma/**\n'
    ]
    for gi in [REPO_ROOT / '.gitignore', REPO_ROOT / 'backend' / '.gitignore']:
        try:
            if gi.exists():
                content = gi.read_text(encoding='utf-8')
            else:
                content = ''
            changed = False
            for p in patterns:
                if p not in content:
                    content += ('' if content.endswith('\n') else '\n') + p
                    changed = True
            if changed:
                gi.write_text(content, encoding='utf-8')
                print(f"[SETUP] Updated gitignore: {gi}")
        except Exception as e:
            print(f"[SETUP][WARN] Could not update gitignore {gi}: {e}")

def run_server():
    print("[SETUP] Starting FastAPI server (uvicorn)...")
    run([python_exe(), '-m', 'uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '8000', '--reload'], check=False)

if __name__ == '__main__':
    print("=== RAG Backend Setup ===")
    ensure_venv()
    install_requirements()
    verify_ocr()
    ensure_chroma_dir()
    print("[SETUP] Setup complete. You can now start the server:")
    print(f"  {python_exe()} -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
    # Uncomment to auto-run
    # run_server()
