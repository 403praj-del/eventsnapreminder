import sys
import os
import logging
import json
import traceback

# Force UTF-8 for stdout/stderr to avoid Windows charmap errors
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Context manager to suppress stdout/stderr (EasyOCR can be chatty on init)
class suppress_output:
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]
    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)
    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

def debug(msg):
    sys.stderr.write(f"[EASYOCR_DEBUG] {msg}\n")
    sys.stderr.flush()

if len(sys.argv) < 2:
    print("Usage: python test_tamil_ocr.py <image_path>")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]
debug(f"Input: {IMAGE_PATH}")

try:
    # Import directly to catch errors, but suppress "Device: cpu" if possible
    # Import inside try to catch import errors (e.g. missing dlls)
    # Suppress import warnings
    with suppress_output():
        import easyocr
        import cv2

    debug("Initializing EasyOCR Reader...")
    
    # Force use of a LOCAL directory
    local_storage = os.path.join(os.getcwd(), 'easyocr_models')
    if not os.path.exists(local_storage):
        os.makedirs(local_storage)

    # Suppress initialization logs (model download warnings etc)
    with suppress_output():
        reader = easyocr.Reader(
            ['ta', 'en'], 
            gpu=False, 
            verbose=False, 
            download_enabled=True,
            model_storage_directory=local_storage
        )

    debug("Starting Prediction...")
    
    # --- PREPROCESSING ---
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise ValueError("Could not read image with OpenCV")

    # 1. Smart Upscaling (for WhatsApp images)
    height, width = img.shape[:2]
    if width < 1000:
        scale = 2 if width > 500 else 3
        debug(f"Image too narrow ({width}px). Upscaling by {scale}x...")
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 2. Contrast Enhancement (CLAHE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(gray)
    
    # Pass processed image to EasyOCR
    # readtext returns a list of tuples: (box, text, confidence)
    # PARAMETER TUNING FOR ACCURACY:
    # text_threshold: 0.4 (Lower confidence allowed to catch faint text)
    # low_text: 0.3 (Lower bound score to detect text areas)
    # link_threshold: 0.4 (Link closer words vs separating them)
    # mag_ratio: 1.5 (Internal magnification to see details)
    result = reader.readtext(
        enhanced_img, 
        detail=0, 
        paragraph=True,
        text_threshold=0.4,
        low_text=0.3, 
        link_threshold=0.4,
        mag_ratio=1.5
    )
    
    debug(f"Prediction Complete. Found {len(result)} blocks.")
    
    import re
    
    def clean_noise(text_list):
        cleaned = []
        for line in text_list:
            # 1. Remove excessively short junk lines (Keep 'On', 'At', 'To' -> len 2)
            if len(line.strip()) < 2:
                continue
            
            # 2. Heuristic: If line is 50% non-alphanumeric (symbols), toss it
            # But be careful with Tamil characters! They are technically "symbols" in ASCII views but "alpha" in unicode.
            # So we check if it has ENOUGH alpha (English/Tamil)
            # Regex \w matches Unicode word characters (including Tamil).
            chars = len(line)
            alpha = len(re.findall(r'\w', line))
            
            # If less than 40% are actual letters/numbers, it's likely noise like "---=--^"
            if alpha / chars < 0.4:
                # Exception: Dates 24-10-2025 might be symbol heavy (dashes), but usually >50% numbers.
                continue
                
            # 3. Known garbage start patterns
            if line.startswith('I^') or line.startswith('|'):
                continue

            cleaned.append(line.strip())
        return cleaned

    # Apply Cleaning
    cleaned_result = clean_noise(result)
    debug(f"Cleaning Complete. Kept {len(cleaned_result)} of {len(result)} lines.")
    
    final_text = " ".join(cleaned_result).strip()
    
    if not final_text:
        print("NO_TEXT_FOUND")
        debug("Warning: Empty result.")
    else:
        print(final_text)
        debug("Success.")

except Exception as e:
    err_msg = traceback.format_exc()
    # Print error to stderr for logs
    sys.stderr.write(f"[SCRIPT_ERROR] {err_msg}\n")
    # Return error as RAW TEXT to stdout to avoid double-JSON in API
    print(f"OCR_SCRIPT_ERROR: {str(e)}")
    sys.exit(1)
