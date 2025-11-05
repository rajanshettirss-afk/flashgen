import os
import nltk
from utils.clean_text import clean_text
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')
nltk.download('punkt')

def segment_text_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    cleaned = clean_text(raw_text)
    sentences = sent_tokenize(cleaned)

    with open(output_path, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence.strip() + "\n")

def batch_segment(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            segment_text_file(
                os.path.join(input_folder, filename),
                os.path.join(output_folder, filename)
            )

if __name__ == "__main__":
    batch_segment("data/extracted_text", "data/segmented_text")