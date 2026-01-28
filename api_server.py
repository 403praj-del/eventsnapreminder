from fastapi import FastAPI, UploadFile, HTTPException
import uvicorn
import os
import shutil
import traceback
import requests
import json
import base64
import time

# --- CONFIGURATION ---
PORT = 8000

def load_api_keys():
    keys = []
    # 1. Check System Environment Variables
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        keys.append(env_key)

    # 2. Check local .env file
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
VISION_MODEL = "google/gemma-3-4b-it:free"

app = FastAPI()

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "keys_loaded": len(API_KEYS),
        "mode": "vision_ai"
    }

@app.on_event("startup")
def startup_sequence():
    print("--- [SERVER STARTUP: VISION AI MODE] ---")
    print(f"    Active model: {VISION_MODEL}")
    print(f"    Loaded {len(API_KEYS)} API Keys for rotation.")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_json_block(text):
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0] # Fix potential dual backticks
        return json.loads(text.strip())
    except:
        # Fallback to finding generic {}
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            return json.loads(text[start:end])
        except:
            return None

# Removed OCR Preprocessing and Manual AI calls - using Direct Vision API

# --- MAIN ENDPOINT ---
@app.post("/ocr")
async def analyze_invitation(file: UploadFile):
    temp_path = f"temp_{int(time.time())}_{file.filename}"
    try:
        with open(temp_path, "wb") as f: 
            shutil.copyfileobj(file.file, f)
        
        print(f"[VISION:START] Analyzing image: {file.filename}")
        base64_image = encode_image(temp_path)
        
        # Prompt for Vision Model
        prompt_text = """
        You are a high-precision Indian invitation data extractor. Return ONLY JSON.
        Extract: event_name, event_type, date, time, venue, address, names (bride, groom).
        Rules: Strip honorifics (Mr, Thiru, Selvan). Format date as DD-MM-YYYY.
        Target JSON: {"event_name": "...", "event_type": "Wedding/Reception/...", "date": "...", "time": "...", "venue": "...", "names": {"bride": "...", "groom": "..."}}
        """

        # Rotation Logic for Multiple API Keys
        for i, api_key in enumerate(API_KEYS):
            print(f"  [Attempt {i+1}] Using key: ...{api_key[-6:]}")
            try:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                
                payload = {
                    "model": VISION_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "response_format": {"type": "json_object"}
                }
                
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result_data = response.json()
                    ai_content = result_data['choices'][0]['message']['content']
                    final_json = extract_json_block(ai_content)
                    if final_json:
                        final_json['ai_source'] = f"vision_api (key_{i+1})"
                        print("[VISION:SUCCESS] Data extracted.")
                        return final_json
                elif response.status_code in [429, 401]:
                    print(f"  [Skip] Key failed with status {response.status_code}. Trying next...")
                    continue
                else:
                    print(f"  [Failed] OpenRouter Error: {response.text}")
                    
            except Exception as inner_e:
                print(f"  [Error] Key {i+1} Exception: {inner_e}")
                continue

        return {"status": "failed", "reason": "ALL_API_KEYS_FAILED"}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
