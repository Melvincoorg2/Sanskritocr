from googletrans import Translator

def translate_sanskrit_to_english(text):
    translator = Translator()
    result = translator.translate(text, dest='en')  # No 'src' parameter
    return result.text


