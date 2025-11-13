import sentencepiece as spm
import json
import os
import torch

def load_tokenizer(model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp

def encode_text(sp, text, max_len=128):
    ids = sp.encode(text, out_type=int)
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids += [0] * (max_len - len(ids))  # pad with zeros
    return ids

def encode_flashcards(dataset_path, tokenizer_path, output_path, max_len=128):
    sp = load_tokenizer(tokenizer_path)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    encoded_data = []
    for item in data:
        input_ids = encode_text(sp, item["input"], max_len)
        target_ids = encode_text(sp, item["target"], max_len)
        encoded_data.append({
            "input_ids": input_ids,
            "target_ids": target_ids
        })

    torch.save(encoded_data, output_path)
    print(f"Saved encoded dataset to {output_path}")

if __name__ == "__main__":
    encode_flashcards(
        dataset_path="data/flashcard_dataset/flashcards.json",
        tokenizer_path="tokenizer/flashgen_tokenizer.model",
        output_path="data/flashcard_dataset/encoded_flashcards.pt",
        max_len=128
    )