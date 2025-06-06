import pytesseract
import re
from backend.preprocessimage import preprocess_image
from backend.integrate_model import clarify_hybrid_text, clarify_hybrid_text_v2
from backend.model import clarify_with_neo
from backend.testingmodel import  recognize_sanskrit_by_structure,visualize_recognized_characters
from backend.kannada import clarify_kannada_text,clarify_kannada_text_v2
from backend.trans import translate_sanskrit_to_english
from backend.qq import recognize_handwritten_hindi

def extract_text_from_image(image_path):
    img = preprocess_image(image_path)
    text = pytesseract.image_to_string(img, lang='san+hin')
    return text
import json
from fuzzywuzzy import process

# Load the lookup dictionary from a JSON file
def load_lookup_dict(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Use fuzzywuzzy to find best match in the lookup dictionary
def fuzzy_lookup(text, lookup_dict, threshold=90):
    keys = list(lookup_dict.keys())
    best_match, score = process.extractOne(text, keys)
    if score >= threshold:
        return lookup_dict[best_match]
    return text  # No good match found

# Replace the entire custom OCR output with fuzzy matching based on the full input
def replace_with_fuzzy_matching(text, lookup_dict, threshold=90):
    # Here, we're matching the entire text rather than individual tokens
    return fuzzy_lookup(text, lookup_dict, threshold)

def ocr_to_single_line(ocr_text: str) -> str:

    return ' '.join(ocr_text.split())


def process_sanskrit_ocr(image_path):
    # Step 1: Run custom model
    custom_raw = recognize_sanskrit_by_structure(image_path)
    custom_raw = ocr_to_single_line(custom_raw)
    

    annotated_img = visualize_recognized_characters(image_path)
    text = extract_text_from_image(image_path)


    # Step 2: Load lookup
    lookup_file = '1.json'
    lookup_dict = load_lookup_dict(lookup_file)

    # Step 3: Try fuzzy replacement based on full input
    custom_replaced = replace_with_fuzzy_matching(custom_raw, lookup_dict, threshold=90)

    matches_found = custom_replaced != custom_raw

    if matches_found:

        raw = custom_replaced
    else:
        raw = recognize_handwritten_hindi(image_path)
        raw = clarify_hybrid_text_v2(raw)

    print("\n📝 Final Sanskrit OCR Output:\n", raw)
    
    # Step 4: Translation and Post-processing
    english_text = clarify_hybrid_text(raw)
    print("\n🧾 Translated to English:\n", english_text)

    kannada_text = clarify_kannada_text(english_text)

    return text ,english_text, kannada_text, annotated_img




