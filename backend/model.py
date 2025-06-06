from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import re

# Load the model
model_name = "MBZUAI/LaMini-Flan-T5-783M"

# ✅ Explicitly set device: CPU
device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32  # Ensure compatibility with CPU
).to(device)

# ✅ Create pipeline (on CPU)
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1  # -1 forces CPU
)

def chunk_text(text, max_chars=1000):
    """Split text into sentence-based chunks within a character limit."""
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) <= max_chars:
            current += sentence + " "
        else:
            chunks.append(current.strip())
            current = sentence + " "
    if current:
        chunks.append(current.strip())

    return chunks

def clarify_with_neo(text):
    """Clarify and modernize distorted/poetic/OCR-mixed text."""
    chunks = chunk_text(text)
    results = []

    for i, chunk in enumerate(chunks):
        prompt = (
            f"Translate to meaningful English:\n\n{chunk.strip()}"
        )

        try:
            output = pipe(
                prompt,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                repetition_penalty=1.2
            )[0]['generated_text']

            results.append(output.strip())
        except Exception as e:
            print(f"❌ Error processing chunk #{i+1}: {e}")
            results.append("[Error processing this part of the input]")

    return "\n\n".join(results)
