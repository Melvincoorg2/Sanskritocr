import requests
import base64

def recognize_handwritten_hindi(image_path: str) -> str:
    """
    Perform OCR on a handwritten Hindi image using Google Vision API.

    Args:
        image_path (str): Path to the handwritten image file.

    Returns:
        str: The extracted text from the image.
    """
    API_KEY = "AIzaSyAHziV_PoBXVJgDWOBvtUd4pCxyOu6WNUo"  # Replace with your actual Google Vision API key
    url = f"https://vision.googleapis.com/v1/images:annotate?key={API_KEY}"
    
    # Read and encode image
    with open(image_path, "rb") as img_file:
        image_content = base64.b64encode(img_file.read()).decode("utf-8")
    
    # Construct request
    headers = {"Content-Type": "application/json"}
    payload = {
        "requests": [{
            "image": {"content": image_content},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]
        }]
    }
    
    # Send request
    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()
    
    # Extract and return text
    try:
        return response_data["responses"][0]["fullTextAnnotation"]["text"]
    except KeyError:
        return "❌ No text detected or an error occurred."

# Example usage
if __name__ == "__main__":
    text = recognize_handwritten_hindi("/home/shadoww/Desktop/sanskrit/test/19.jpeg")  # Replace with your file path
    print("🔤 Recognized Text:\n", text)
