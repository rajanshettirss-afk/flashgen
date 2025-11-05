import pdfplumber
import os

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

def batch_extract(folder_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(os.path.join(folder_path, filename))
            with open(os.path.join(output_folder, filename.replace(".pdf", ".txt")), "w", encoding="utf-8") as f:
                f.write(text)

if __name__ == "__main__":
    batch_extract("data/raw_pdfs", "data/extracted_text")