from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load a stronger model (if RAM allows)
model_name = "google/flan-t5-base"  # Use flan-t5-base if low memory
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

def split_text(text, chunk_size=300):
    sentences = text.replace("\n", " ").split('.')
    chunks = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= chunk_size:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks

def clarify_hybrid_text(text):
    chunks = split_text(text)
    results = []
    for chunk in chunks:
        prompt = (
            "This passage containe sanskrit text can u identify it ,because its all togehter and there are no spaced so please help me identify."
            "This passage contains sanskrit distorted or archaic language, possibly due to OCR errors or poetic text. "
            "Reconstruct it into clear, fluent, modern English. Correct jumbled or unclear words using context, "
            "and preserve the original tone and meaning.\n\n"
            f"Text: {chunk}\n\nModern English version:"
        )
        output = pipe(prompt, max_length=512, do_sample=True, temperature=0.8, top_p=0.95)[0]['generated_text']
        results.append(output.strip())
    return "\n\n".join(results)








import google.generativeai as genai
import os
import time

# Step 1: Configure your API key (Load from environment variable for security)
GOOGLE_API_KEY = "AIzaSyCjCfR6a13nE7hjxgY0AT9VUXf9dd5Vi7Q"  # Load API key from environment variable for security
genai.configure(api_key=GOOGLE_API_KEY)

# Step 2: Initialize the Gemini Pro model (make sure the model name is correct)
model = genai.GenerativeModel("gemini-1.5-flash")

# Step 3: Split noisy OCR Sanskrit text into chunks
def split_text(text, chunk_size=400):
    sentences = text.replace("\n", " ").split('.')
    chunks, current = [], ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= chunk_size:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks

# Step 4: Clarify and translate Sanskrit OCR chunks
def clarify_hybrid_text_v2(text):
    chunks = split_text(text)
    results = []

    for chunk in chunks:
        prompt = (
            "This passage containe sanskrit text can u identify it ,because its all togehter and there are no spaced so please help me identify."
            "This passage contains sanskrit distorted or archaic language, possibly due to OCR errors or poetic text. "
            "Reconstruct it into clear, fluent, modern English. Correct jumbled or unclear words using context, "
            "for short sentence just give translation."
            "and preserve the original tone and meaning.\n\n"
            f"Text: {chunk}\n\nModern English version:"
        )

        try:
            # Making API call with model's generate_content method
            response = model.generate_content(prompt)
            results.append(response.text.strip())

        except Exception as e:
            print(f"Error generating response for chunk: {chunk}. Error: {e}")
            results.append("Error generating response.")
            # Optionally, retry after a brief delay
            time.sleep(2)

    return "\n\n".join(results)



