import os
import pandas as pd

def create_folders():
    os.makedirs("bert_ner_project/data", exist_ok=True)
    os.makedirs("bert_ner_project/models", exist_ok=True)
    print("✅ Created folders: data/, models/")

def generate_synthetic_data():
    data = {
        "sentence": [
            "John lives in New York",
            "Apple was founded by Steve Jobs",
            "Samantha works at Microsoft"
        ],
        "labels": [
            ["B-PER", "O", "O", "B-LOC", "I-LOC"],
            ["B-ORG", "O", "O", "O", "B-PER", "I-PER"],
            ["B-PER", "O", "O", "B-ORG"]
        ]
    }
    df = pd.DataFrame(data)
    df.to_csv("bert_ner_project/data/synthetic_ner_data.csv", index=False)
    print("✅ Synthetic data saved at data/synthetic_ner_data.csv")

if __name__ == "__main__":
    create_folders()
    generate_synthetic_data()
