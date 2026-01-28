import easyocr
import os

MODEL_DIR = os.environ.get('EASYOCR_MODEL_DIR', './easyocr_models')
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Pre-downloading EasyOCR models to {MODEL_DIR}...")
easyocr.Reader(['ta', 'en'], download_enabled=True, model_storage_directory=MODEL_DIR)
print("Models downloaded successfully.")
