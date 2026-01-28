import easyocr
import easyocr.config as config

print("=== EASYOCR CONFIG DUMP ===")
print(f"MODULE_PATH: {easyocr.__file__}")
if hasattr(config, 'MODULE_PATH'):
    print(f"CONFIG_MODULE_PATH: {config.MODULE_PATH}")

print(f"download_url_base: {getattr(config, 'download_url_base', 'NOT_FOUND')}")
print(f"model_url details for 'tamil':")
if 'tamil' in config.model_url:
    print(config.model_url['tamil'])
else:
    print(" 'tamil' key not found in model_url")

print("model_url details for 'english_g2':")
if 'english_g2' in config.model_url:
    print(config.model_url['english_g2'])
    
print("=== END DUMP ===")
