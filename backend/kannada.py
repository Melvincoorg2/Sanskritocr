from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from googletrans import Translator
model_name = "facebook/nllb-200-distilled-600M"  # Light version of NLLB
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang="eng_Latn", tgt_lang="kan_Knda")

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

def clarify_kannada_text(text):
    chunks = split_text(text)
    results = []
    for chunk in chunks:
        output = translator(chunk, max_length=512)[0]['translation_text']
        results.append(output.strip())
    return "\n\n".join(results)

def clarify_kannada_text_v2(text):
    translator = Translator()
    result = translator.translate(text, dest='kn')  # No 'src' parameter
    return result.text
