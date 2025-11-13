import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import Dataset, DataLoader
from models.transformer_model import FlashcardTransformer

class FlashcardDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)

    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return torch.tensor(item["input_ids"]), torch.tensor(item["target_ids"])

# Hyperparameters
vocab_size = 8000
batch_size = 16
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
dataset = FlashcardDataset("data/flashcard_dataset/encoded_flashcards.pt")
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = FlashcardTransformer(vocab_size=vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        output = model(src, tgt[:, :-1])
        loss = criterion(output.reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "models/transformer_model.pt")