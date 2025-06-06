import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import re
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

# Load model
model = load_model("model/sanskrit_model.h5")

# Manually define class indices (must match training folder order)
class_indices = {
    "character_01_ka": 0,
    "character_02_kha": 1,
    "character_03_ga": 2,
    "character_04_gha": 3,
    "character_05_kna": 4,
    "character_06_cha": 5,
    "character_07_chha": 6,
    "character_08_ja": 7,
    "character_09_jha": 8,
    "character_10_yna": 9,
    "character_11_taamatar": 10,
    "character_12_tha": 11,
    "character_13_daa": 12,
    "character_14_dhaa": 13,
    "character_15_adna": 14,
    "character_16_tabala": 15,
    "character_17_tha": 16,
    "character_18_da": 17,
    "character_19_dha": 18,
    "character_20_na": 19,
    "character_21_pa": 20,
    "character_22_pha": 21,
    "character_23_ba": 22,
    "character_24_bha": 23,
    "character_25_ma": 24,
    "character_26_yaw": 25,
    "character_27_ra": 26,
    "character_28_la": 27,
    "character_29_waw": 28,
    "character_30_motosaw": 29,
    "character_31_petchiryakha": 30,
    "character_32_patalosaw": 31,
    "character_33_ha": 32,
    "character_34_chhya": 33,
    "character_35_tra": 34,
    "character_36_gya": 35,
    "digit_0": 36,
    "digit_1": 37,
    "digit_2": 38,
    "digit_3": 39,
    "digit_4": 40,
    "digit_5": 41,
    "digit_6": 42,
    "digit_7": 43,
    "digit_8": 44,
    "digit_9": 45
}
labels = {v: k for k, v in class_indices.items()}

# Mapping label names to Devanagari characters
label_to_unicode = {
    "character_01_ka": "क",
    "character_02_kha": "ख",
    "character_03_ga": "ग",
    "character_04_gha": "घ",
    "character_05_kna": "ङ",
    "character_06_cha": "च",
    "character_07_chha": "छ",
    "character_08_ja": "ज",
    "character_09_jha": "झ",
    "character_10_yna": "ञ",
    "character_11_taamatar": "ट",
    "character_12_thaa": "ठ",
    "character_13_daa": "ड",
    "character_14_dhaa": "ढ",
    "character_15_adna": "ण",
    "character_16_tabala": "त",
    "character_17_tha": "थ",
    "character_18_da": "द",
    "character_19_dha": "ध",
    "character_20_na": "न",
    "character_21_pa": "प",
    "character_22_pha": "फ",
    "character_23_ba": "ब",
    "character_24_bha": "भ",
    "character_25_ma": "म",
    "character_26_yaw": "य",
    "character_27_ra": "र",
    "character_28_la": "ल",
    "character_29_waw": "व",
    "character_30_motosaw": "श",
    "character_31_petchiryakha": "ष",
    "character_32_patalosaw": "स",
    "character_33_ha": "ह",
    "character_34_chhya": "क्ष",
    "character_35_tra": "त्र",
    "character_36_gya": "ज्ञ",

    "digit_0": "०",
    "digit_1": "१",
    "digit_2": "२",
    "digit_3": "३",
    "digit_4": "४",
    "digit_5": "५",
    "digit_6": "६",
    "digit_7": "७",
    "digit_8": "८",
    "digit_9": "९",
}


def preprocess_and_segment(image_path):
    """Preprocess the image and return sorted character bounding boxes."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))  # Optional
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours (characters)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 100]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[1], b[0]))  # Sort top-to-bottom, left-to-right

    return image, thresh, bounding_boxes

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


def preprocess_image(image_path):
    """Load and preprocess the image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize if needed (Optional, comment out if resizing is not required)
    image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    
    # Apply Gaussian Blur and Otsu's binarization
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return image, binary

def segment_lines(binary_image):
    """Segment the image into lines using horizontal projection."""
    hist = np.sum(binary_image, axis=1)
    lines = []
    start = None
    for i, val in enumerate(hist):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            end = i
            lines.append((start, end))
            start = None
    if start is not None:
        lines.append((start, len(hist)))
    return lines

def segment_words(line_img):
    """Segment the line into words using vertical projection."""
    hist = np.sum(line_img, axis=0)
    words = []
    start = None
    for i, val in enumerate(hist):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            end = i
            words.append((start, end))
            start = None
    if start is not None:
        words.append((start, len(hist)))
    return words
    
def segment_characters_by_projection(word_img):
    """Segment a word image into characters using vertical projection."""
    hist = np.sum(word_img, axis=0)
    chars = []
    start = None
    for i, val in enumerate(hist):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            end = i
            if end - start > 2:  # Filter out very narrow regions
                chars.append((start, end))
            start = None
    if start is not None:
        chars.append((start, word_img.shape[1]))
    return chars


def predict_character(img_crop):
    """Predict a character from the given cropped image."""
    char_img = cv2.resize(img_crop, (64, 64))  # Resize to model input size
    char_img = char_img.astype("float32") / 255.0
    char_img = img_to_array(char_img)
    char_img = np.expand_dims(char_img, axis=0)

    pred = model.predict(char_img)
    predicted_label = labels[np.argmax(pred)]  # Get the label with highest probability
    return predicted_label

def draw_unicode_labels(image, boxes, labels, font_path):
    """Draw Unicode labels on the image."""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, 28)

    for (x, y, w, h), label in zip(boxes, labels):
        draw.rectangle([x, y, x+w, y+h], outline="green", width=2)
        draw.text((x, y - 30), label, font=font, fill=(255, 0, 0))

    return np.array(pil_img)

def recognize_sanskrit_by_structure(image_path):
    """Recognize text from an image using segmentation and character prediction."""
    image, binary = preprocess_image(image_path)
    lines = segment_lines(binary)
    full_text = ""

    for (line_start, line_end) in lines:
        line_img = binary[line_start:line_end, :]
        word_bounds = segment_words(line_img)
        line_text = ""

        for (word_start, word_end) in word_bounds:
            word_img = line_img[:, word_start:word_end]

            # Find contours and segment the word into characters
            contours, _ = cv2.findContours(word_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            char_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 50]
            char_boxes = sorted(char_boxes, key=lambda b: b[0])  # Sort left to right

            word_text = ""
            for (x, y, w, h) in char_boxes:
                roi = word_img[y:y+h, x:x+w]
                label = predict_character(roi)
                unicode_char = label_to_unicode.get(label, '')
                word_text += unicode_char  # Append recognized character to word text

            line_text += word_text + ' '  # Add space between words

        full_text += line_text.strip() + '\n'  # Add line break between lines

    return full_text.strip()  # Return the final recognized text


def visualize_recognized_characters(image_path):
    image, thresh, boxes = preprocess_and_segment(image_path)
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to color for annotation

    # Sort boxes again (y, x)
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    widths = [w for x, y, w, h in boxes]
    avg_width = np.mean(widths)
    threshold_gap = avg_width * 0.6

    previous_x = None
    previous_y = None
    predicted_text = []

    for i, (x, y, w, h) in enumerate(boxes):
        roi = thresh[y:y+h, x:x+w]
        label = predict_character(roi)
        unicode_char = label_to_unicode.get(label, '')
        predicted_text.append(unicode_char)

    return output_image

def check_model_prediction(image_path):
    """Check the model's prediction for a single image and display relevant details."""
    # Preprocess and segment the image
    image, thresh, boxes = preprocess_and_segment(image_path)
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Find a Devanagari supporting font
    devanagari_font = None
    for font in fm.fontManager.ttflist:
        if "Nirmala UI" in font.name:  # Try Nirmala UI first
            devanagari_font = font.fname
            break
    if not devanagari_font:
        for font in fm.fontManager.ttflist:
            if "Arial Unicode MS" in font.name:  # Fallback to Arial Unicode MS
                devanagari_font = font.fname
                break
    if not devanagari_font:
        for font in fm.fontManager.ttflist:
            if "Mangal" in font.name:  # Fallback to Mangal
                devanagari_font = font.fname
                break

    if not devanagari_font:
        print("Warning: No Devanagari font found. Predictions might not display correctly.")

    # Loop through each bounding box
    for (x, y, w, h) in boxes:
        roi = thresh[y:y+h, x:x+w]

        # Predict the character
        predicted_label = predict_character(roi)
        predicted_unicode = label_to_unicode.get(predicted_label, '')

        # Print prediction details
        print(f"Predicted class index: {class_indices.get(predicted_label)}")
        print(f"Predicted class name: {predicted_label}")
        print(f"Predicted Unicode: {predicted_unicode}")

    return image, output_image

