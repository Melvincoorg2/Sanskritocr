import cv2

import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Step 1: Resize if image is small
    if img.shape[1] < 800:
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Step 2: Apply median blur to reduce noise
    blurred = cv2.medianBlur(img, 3)

    # Step 3: Check contrast and lighting
    mean_brightness = np.mean(blurred)
    if mean_brightness < 100:
        # Dark image → use adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    else:
        # Light image → use Otsu’s binarization
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # Step 4: Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.dilate(thresh, kernel, iterations=1)

    return cleaned


