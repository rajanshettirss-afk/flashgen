import re

def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII
    return text.strip()