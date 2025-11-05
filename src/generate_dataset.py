import os
import json

def generate_flashcard_pairs(input_folder, output_path):
    dataset = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(input_folder, filename), "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if len(line.split()) > 8 and " is " in line:
                    term = line.split(" is ")[0]
                    answer = line
                    question = f"What is {term.strip()}?"
                    dataset.append({
                        "input": line,
                        "target": f"Q: {question} A: {answer}"
                    })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    generate_flashcard_pairs("data/segmented_text", "data/flashcard_dataset/flashcards.json")