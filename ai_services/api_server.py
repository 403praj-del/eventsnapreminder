from fastapi import FastAPI, UploadFile, HTTPException
import uvicorn
import os
import shutil
import easyocr
import cv2
import numpy as np
import re
import traceback
import requests
import json
import gc
import torch # Ensure we can set threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

try:
    from ctransformers import AutoModelForCausalLM
except ImportError:
    AutoModelForCausalLM = None

# --- CONFIGURATION ---
ENV_TYPE = os.getenv("ENV_TYPE", "LOCAL").upper() # LOCAL or CLOUD
PORT = int(os.getenv("PORT", 8000))
MODEL_DIR = os.getenv("EASYOCR_MODEL_DIR", os.path.join(os.getcwd(), 'easyocr_models'))
LOCAL_LLM_DIR = os.getenv("LOCAL_LLM_DIR", os.path.join(os.getcwd(), 'local_llm_models'))

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOCAL_LLM_DIR, exist_ok=True)

# Primary AI (OpenRouter)
# We will load MULTIPLE keys from .env manually to handle the user's format
OPENROUTER_MODEL = "google/gemma-3n-e4b-it:free"

def load_api_keys():
    keys = []
    # 1. Check System Environment Variables (Priority for Cloud/Railway)
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        keys.append(env_key)

    # 2. Check local .env file (for Local testing)
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.strip().split('=', 1)[1].strip()
                    if key and key not in keys:
                        keys.append(key)
    
    # 3. Fallback to hardcoded default
    if not keys:
        keys = ["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"]
    return keys

API_KEYS = load_api_keys()
print(f"Loaded {len(API_KEYS)} API Keys for Rotation.")

# Secondary AI (Local OpenChat)
LOCAL_MODEL_REPO = "TheBloke/openchat_3.5-GGUF"
LOCAL_MODEL_FILE = "openchat_3.5.Q4_K_M.gguf"

app = FastAPI()

# Global Variables
reader = None
local_llm = None

ocr_error = None

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "env": ENV_TYPE,
        "ocr_loaded": reader is not None,
        "ocr_error": ocr_error,
        "llm_loaded": local_llm is not None,
        "timestamp": time.time()
    }

# --- AI PROMPT ---
SYSTEM_PROMPT = """
You are a high-precision Indian invitation data extractor.
Text may contain Tamil + English.
Return ONLY structured JSON.

[GOAL]
Extract: event_name, event_type, date, time, venue, address, names, phone_numbers

[HIERARCHY OF NAMES]
1. COUPLE (High Priority): Names following "Selvan/Mappilai" (Groom) or "Selvi/Penn" (Bride).
2. PARENTS (Skip): Names following "Thiru/Thirumathi", "Son of", "Daughter of", "S/o", "D/o".
3. HOSTS/ELDERS (Skip): Names in "Invited by", "With blessings of", "Anoopunar".

[CLEANING RULES]
- Strip honorifics (Mr, Mrs, Thiru, Selvan, Selvi, Er, Dr, Late) from final names.
- If name is "Thiru valar [Name]", the name is "[Name]".
- Fix typos (e.g. "Vivaham" instead of "Vvaham").
- TIME: "காலை" = AM, "மாலை" = PM. (e.g. "மாலை 6.30" -> "06:30 PM")

[FEW-SHOT EXAMPLES]
# Traditional Tamil
Input: "மாலை 6.30 மணியளவில்"
Output: {"time": "06:30 PM", ...}

Input: "Thiru. Mani & Thirumathi. Jaya invite you to the Subha Muhurtham of their son Selvan. ARUN with Selvi. DIVYA (D/o Thiru. Ravi)"
Output: {"names": {"bride": "DIVYA", "groom": "ARUN"}, "event_name": "Subha Muhurtham", "event_type": "Marriage"}

# Formal English (Host Line)
Input: "Mr. and Mrs. Robert Johnson request the honor of your presence at the marriage of their daughter Sarah Elizabeth to Michael James, son of Mr. and Mrs. Alan Smith"
Output: {"names": {"bride": "Sarah Elizabeth", "groom": "Michael James"}, "event_name": "Marriage", "event_type": "Wedding"}

# Mixed Language / Noisy
Input: "Cordially invite you for the Wedding of Selvan VICKY S/o Mr. Siva and Selvi RADHA D/o Mr. Kumar"
Output: {"names": {"bride": "RADHA", "groom": "VICKY"}, "event_name": "Wedding", "event_type": "Marriage"}

Values:
- date: DD-MM-YYYY
- time: HH:MM AM/PM
- names: { "bride": "CLEAN_NAME", "groom": "CLEAN_NAME" }
"""

# --- STARTUP LOGIC ---
@app.on_event("startup")
def startup_sequence():
    global reader, local_llm, ocr_error
    print(f"--- [SERVER STARTUP: {ENV_TYPE}] ---")
    
    # 1. Load OCR (Skip in Extreme Cloud mode to keep RAM zero at idle)
    if ENV_TYPE == "CLOUD":
        print("[1/3] Extreme Cloud Mode: OCR will be loaded per-request (English First) to save RAM.")
    else:
        print("[1/3] Loading EasyOCR (Tamil+English)...")
        try:
            reader = easyocr.Reader(['ta', 'en'], gpu=False, verbose=False, download_enabled=True, model_storage_directory=MODEL_DIR)
            print("      EasyOCR Loaded.")
        except Exception as e:
            ocr_error = str(e)
            print(f"      FATAL: OCR Failed: {e}")

    # 2. Load Local LLM (Secondary) - OPTIONAL ON CLOUD
    if ENV_TYPE == "CLOUD":
        print("[2/3] Skipping Local LLM (CLOUD MODE) to save RAM.")
    else:
        print("[2/3] Checking Local LLM (OpenChat)...")
        if AutoModelForCausalLM is None:
            print("      WARNING: ctransformers not installed. Skipping local LLM.")
            return

        model_path = os.path.join(LOCAL_LLM_DIR, LOCAL_MODEL_FILE)
        try:
            print("      Loading Local Model into RAM...")
            local_llm = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_REPO,
                model_file=LOCAL_MODEL_FILE,
                model_type="mistral",
                context_length=1024,
                gpu_layers=0,
                batch_size=256,
                threads=8
            )
            print("      Local LLM Loaded Successfully!")
        except Exception as e:
            print(f"      WARNING: Local LLM Load Failed: {e}")
            print("      Secondary fallback will NOT work.")

    print("[3/3] Server Ready.")

# --- HELPER FUNCTIONS ---
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None: raise ValueError("Could not read image")
    
    # --- EXTREME CLOUD OPTIMIZATION ---
    # Downscale and compress to stay under 512MB
    if ENV_TYPE == "CLOUD":
        h, w = img.shape[:2]
        max_dim = 800 # Very small for strict memory
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            print(f"[PREPROCESS] Extreme Downscale to {max_dim}px (RAM SAVE)")
    else:
        # Local logic (higher quality)
        h, w = img.shape[:2]
        if w < 1000:
            scale = 2 if w > 500 else 3
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            print(f"[PREPROCESS] Upscaled x{scale}")
        
    # CLAHE (Mild)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    return clahe.apply(gray)

def clean_noise(text_list):
    cleaned = []
    for line in text_list:
        if len(line.strip()) < 2: continue # Keep 'At', 'On'
        
        chars = len(line)
        alpha = len(re.findall(r'\w', line))
        if alpha / chars < 0.4: continue # Too many symbols
        
        if line.startswith('I^') or line.startswith('|'): continue
        cleaned.append(line.strip())
    return cleaned

import warnings
# Suppress specific PyTorch warning about pin_memory
warnings.filterwarnings("ignore", message=".*pin_memory.*")

def call_primary_ai(ocr_text):
    print("[AI:Primary] Calling OpenRouter...")
    
    for i, api_key in enumerate(API_KEYS):
        print(f"  > Attempt {i+1}/{len(API_KEYS)} using Key: ...{api_key[-6:]}")
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://eventsnap.com", 
                "X-Title": "EventSnap"
            }
            
            # MERGED PROMPT: Gemma 3N does not support 'system' role
            # We combine System + User into a single User message
            full_prompt = f"System Instruction:\n{SYSTEM_PROMPT}\n\nUser Task:\nEXTRACT FROM:\n{ocr_text}"
            
            payload = {
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "user", "content": full_prompt}
                ],
                "temperature": 0.0, # Zero Creativity for Data Extraction
                "max_tokens": 512
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15 # 15s Timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                print("    [Success] Primary AI Response received.")
                return extract_json_block(content)
            elif response.status_code in [429, 401, 402]: # Rate Limit or Auth Error
                print(f"    [Retry] Error {response.status_code}: {response.text}")
                continue # Try next key
            else:
                print(f"    [Failed] Fatal Error {response.status_code}: {response.text}")
                return None # Don't retry for other errors (like 400 Bad Request)
                
        except Exception as e:
            print(f"    [Error] Exception: {e}")
            continue # Try next key just in case

    print("[AI:Primary] All keys failed.")
    return None

# --- TAMIL CONTEXT FOR LOCAL AI ---
TAMIL_LEARNING_CONTEXT = """
[TAMIL DICTIONARY]
- Thirumanam / Vivaham / Subha Muhurtham -> Wedding / Marriage
- Manamagan / Selvan / Mappilai -> Groom (Priority for Name)
- Manamagal / Selvi / Penn -> Bride (Priority for Name)
- Muhurtham / Neram -> Auspicious Time
- Idam / Nilayam -> Venue
- Petror / Anuppunar -> Parents / Senders (IGNORE for Bride/Groom names)
- Thiru / Thirumathi / Magan / Magal -> Mr / Mrs / Son / Daughter (Parent context - IGNORE for Name)
- Naal / Thethi -> Date
- Kaalai -> AM (Morning)
- Maalai -> PM (Evening)
"""

def call_secondary_ai(ocr_text):
    global local_llm
    if local_llm is None:
        print("[AI:Secondary] Not available (Model not loaded).")
        return None
        
    print("[AI:Secondary] Calling Local OpenChat...")
    try:
        # Improved In-Context Learning for Names
        prompt = f"""System: {SYSTEM_PROMPT}

[CULTURAL CONTEXT]
In Indian invitations, the names following "Selvan" or "Mappilai" is the Groom. 
The names following "Selvi" or "Penn" is the Bride.
Names following "Thiru" and "Thirumathi" are usually the PARENTS. Do not confuse them.
Strip honorifics (Mr., Mrs., Thiru, Selvi) from the final JSON name fields.

[TAMIL DICTIONARY] 
{TAMIL_LEARNING_CONTEXT}

User: Extract the COUPLE names (Bride/Groom) and event details:
{ocr_text}

Assistant: Here is the JSON:
```json
"""
        
        # Add prints to debug blockage
        print("[AI:Secondary] Starting generation...")
        start_time = time.time()
        
        response = local_llm(
            prompt, 
            max_new_tokens=512, # Restored full length (User requested no restriction)
            temperature=0.1,
            stop=["```"]
        )
        
        print(f"[AI:Secondary] Generation complete in {time.time() - start_time:.2f}s")
        print(f"[AI:Secondary] Raw response: {response[:100]}...") # Print first 100 chars
        
        # Cleanup ensures we get valid JSON
        return extract_json_block("{" + response) # We forced the prompt to end at ```json, so response starts with {
        
    except Exception as e:
        print(f"[AI:Secondary] Error: {e}")
        traceback.print_exc()
        return None

def extract_json_block(text):
    try:
        # Find JSON between ```json and ``` or just start to end
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[0]
            
        return json.loads(text.strip())
    except:
        # Retry with soft cleaning
        try:
            return json.loads(text.strip())
        except:
            return None

# --- MAIN ENDPOINT ---
@app.post("/ocr")
async def process_invitation(file: UploadFile):
    global reader, ocr_error
    
    # Lazy Load/Retry if first attempt failed
    if reader is None:
        print("[OCR:RETRY] attempting on-demand initialization...")
        try:
            # In cloud, we try English only first to verify RAM limits
            langs = ['en'] if ENV_TYPE == "CLOUD" else ['ta', 'en']
            # download_enabled=False because Docker build already grabbed them
            reader = easyocr.Reader(langs, gpu=False, download_enabled=False, model_storage_directory=MODEL_DIR)
            ocr_error = None
            print(f"[OCR:STATUS] Reader initialized with {langs}")
        except Exception as retry_err:
            ocr_error = str(retry_err)
            trace = traceback.format_exc()
            print(f"[OCR:RETRY_FATAL] Load failed: {ocr_error}\n{trace}")
            return {
                "status": "failed", 
                "reason": "SERVER_INITIALIZING_FAILED", 
                "error": ocr_error,
                "trace": trace
            }

    temp_path = f"temp_{int(time.time())}_{file.filename}"
    try:
        with open(temp_path, "wb") as f: shutil.copyfileobj(file.file, f)
        
        # 1. OCR Step
        print(f"[OCR:START] Extreme processing for {file.filename}...")
        try:
            # Re-init reader ONLY during request if in Cloud to keep RAM free
            local_reader = reader
            if local_reader is None:
                print("[OCR] Initializing engine on-demand...")
                local_reader = easyocr.Reader(['ta', 'en'], gpu=False, model_storage_directory=MODEL_DIR)

            proc_img = preprocess_image(temp_path)
            
            # Use torch.no_grad to save memory
            with torch.no_grad():
                result = local_reader.readtext(proc_img, detail=0, paragraph=True)
            
            # AGGRESSIVE PURGE
            del proc_img
            if ENV_TYPE == "CLOUD":
                print("[OCR] Purging engine to free RAM...")
                del local_reader
            
            gc.collect() 
            
            cleaned_ocr = clean_noise(result)
            ocr_text = " ".join(cleaned_ocr).strip()
            print(f"[OCR:FINISH] Success. Extracted {len(ocr_text)} chars.")
        except Exception as ocr_err:
            error_trace = traceback.format_exc()
            print(f"[OCR:FATAL] Component Failure: {ocr_err}\n{error_trace}")
            return {"status": "failed", "reason": "OCR_PROCESS_ERROR", "error": str(ocr_err), "trace": error_trace}
        
        if not ocr_text:
            print("[OCR:EMPTY] No text detected in image.")
            return {"status": "failed", "reason": "NO_TEXT_FOUND"}
            
        print(f"[AI:START] Sending text to AI models...")

        # 2. Primary AI
        ai_data = call_primary_ai(ocr_text)
        
        # 3. Secondary AI (Fallback)
        used_source = "primary"
        if ai_data is None:
            print("[Fallback] Switching to Secondary AI...")
            ai_data = call_secondary_ai(ocr_text)
            used_source = "secondary"
            
        # 4. Final Response
        if ai_data:
            ai_data['ocr_text_raw'] = ocr_text # Include raw text for debug
            ai_data['ai_source'] = used_source
            return ai_data
        else:
            # Both failed, return raw text so user can manual
            return {
                "status": "partial",
                "ocr_text_raw": ocr_text,
                "ai_source": "failed"
            }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
