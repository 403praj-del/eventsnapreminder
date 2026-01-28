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
    sys.stderr.write(f"MODEL_DOWNLOAD_FAILED: {str(e)}\n")
    sys.exit(0) # Allow build to proceed, will retry on startup
