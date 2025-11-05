import sentencepiece as spm
import os

def train_sentencepiece(input_file, model_prefix="flashgen_tokenizer", vocab_size=8000):
    spm.SentencePieceTrainer.train(
        input=f"{input_file}",
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="bpe"
    )

if __name__ == "__main__":
    # Combine all input text into one file
    dataset_path = "data/flashcard_dataset/flashcards.json"
    temp_text_file = "tokenizer/raw_text.txt"

    os.makedirs("tokenizer", exist_ok=True)

    import json
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(temp_text_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(item["input"] + "\n")
            f.write(item["target"] + "\n")

    train_sentencepiece(temp_text_file)