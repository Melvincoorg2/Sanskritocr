import base64
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from ocr_extract import process_sanskrit_ocr

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def encode_image_to_base64(img_array):
    _, buffer = cv2.imencode('.jpg', img_array)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

@app.post("/process-image")
async def process_image(image: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, "temp_image.jpg")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        text, result, result_k, annotated_img = process_sanskrit_ocr(file_path)
        print("The result is being processed:",text , result, result_k)

        if result is None:
            return JSONResponse(content={"error": "No meaningful text was extracted from the image."}, status_code=400)

        encoded_img = encode_image_to_base64(annotated_img)

        return JSONResponse(content={
            "text":text,
            "result": result,
            "kannada_result": result_k,
            "annotated_image_base64": encoded_img
        })

    except Exception as e:
        print("🔥 Backend Exception:", str(e))
        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)
