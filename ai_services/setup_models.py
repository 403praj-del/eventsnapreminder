import easyocr
import os
import sys

MODEL_DIR = os.environ.get('EASYOCR_MODEL_DIR', './easyocr_models')
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Pre-downloading EasyOCR models to {MODEL_DIR}...")
try:
    # Explicitly set gpu=False to avoid crashes on CPU-only build servers
    easyocr.Reader(['ta', 'en'], download_enabled=True, model_storage_directory=MODEL_DIR, gpu=False)
    print("Models downloaded and verified successfully.")
except Exception as e:
    print(f"Error downloading models: {e}")
    # Don't exit with error, let the server try again on startup if possible
    # but print to stderr so it shows in logs
    sys.stderr.write(f"MODEL_DOWNLOAD_FAILED: {str(e)}\n")
    sys.exit(0) # Exit with 0 to allow build to continue, server will retry 
