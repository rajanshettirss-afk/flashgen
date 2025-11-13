import torch
import sentencepiece as spm
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.transformer_model import FlashcardTransformer

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/flashgen_tokenizer.model")

# Load model
vocab_size = 8000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FlashcardTransformer(vocab_size=vocab_size).to(device)
model.load_state_dict(torch.load("models/transformer_model.pt", map_location=device))
model.eval()

def generate_flashcard(input_text, max_len=50):
    # Encode input
    input_ids = sp.encode(input_text, out_type=int)
    input_ids = input_ids[:max_len] + [0] * (max_len - len(input_ids))
    src = torch.tensor([input_ids]).to(device)

    # Start with BOS token
    bos_id = sp.piece_to_id("<BOS>") if sp.piece_to_id("<BOS>") >= 0 else 1
    eos_id = sp.piece_to_id("<EOS>") if sp.piece_to_id("<EOS>") >= 0 else 2
    tgt_ids = [bos_id]

    for _ in range(max_len):
        tgt = torch.tensor([tgt_ids], dtype=torch.long).to(device)
        with torch.no_grad():
            output = model(src, tgt)
            next_token = output[:, -1, :].argmax(dim=-1).item()
            if next_token == eos_id or next_token == 0 or next_token >= sp.get_piece_size():
                break
            tgt_ids.append(next_token)

    print("Predicted token IDs:", tgt_ids)

    if len(tgt_ids) == 1:
        return "[No meaningful output — model may need more training]"

    decoded = sp.decode(tgt_ids[1:])  # skip BOS
    return decoded

if __name__ == "__main__":
    print("🔍 Flashcard Generator — Type a concept or question below\n")
    while True:
        user_input = input("Enter a concept or question (or 'exit'): ")
        if user_input.lower() == "exit":
            break
        answer = generate_flashcard(user_input)
        print(f"Flashcard Answer: {answer}\n")