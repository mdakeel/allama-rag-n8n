import os
import json
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

# ========= CONFIG =========
MODEL_NAME = "Helsinki-NLP/opus-mt-ur-en"
INPUT_DIR = "data/transcripts_ur"
OUTPUT_DIR = "data/transcripts_en"
# ==========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading translation model...")
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)

def translate_text(text):
    if not text.strip():
        return text

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Translating files...")

files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

for file in tqdm(files):
    in_path = os.path.join(INPUT_DIR, file)
    out_path = os.path.join(OUTPUT_DIR, file)

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for seg in data.get("segments", []):
        ur_text = seg.get("text_roman", "")
        seg["text_en"] = translate_text(ur_text)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

print("DONE ✅ All files translated")
