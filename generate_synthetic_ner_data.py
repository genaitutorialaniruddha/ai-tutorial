import random
import pandas as pd

# Configurable entity pools
people = ["John", "Samantha", "Steve Jobs", "Alice", "Bob"]
locations = ["New York", "Paris", "London", "Bangalore", "Tokyo"]
organizations = ["Google", "Microsoft", "Apple", "Amazon", "OpenAI"]
actions = ["works at", "founded", "joined", "left", "moved to"]

# Entity tag rules
ENTITY_TAGS = {
    "PER": "B-PER",
    "LOC": "B-LOC",
    "ORG": "B-ORG",
    "O": "O"
}

def tokenize(text):
    return text.split()

def tag_entity(tokens, entity, label):
    """
    Apply BIO tagging for an entity in the token list.
    """
    entity_tokens = tokenize(entity)
    tags = [f"B-{label}"] + [f"I-{label}"] * (len(entity_tokens) - 1)
    return entity_tokens, tags

def generate_sentence():
    # Random selection
    person = random.choice(people)
    action = random.choice(actions)
    org = random.choice(organizations)
    loc = random.choice(locations)

    template = random.choice([
        f"{person} {action} {org} in {loc}",
        f"{person} moved to {loc} after leaving {org}",
        f"{org} was founded by {person} in {loc}"
    ])

    tokens = []
    labels = []

    for word in template.split():
        if word in person:
            entity_tokens, tags = tag_entity(word, person, "PER")
        elif word in org:
            entity_tokens, tags = tag_entity(word, org, "ORG")
        elif word in loc:
            entity_tokens, tags = tag_entity(word, loc, "LOC")
        else:
            entity_tokens = [word]
            tags = ["O"]
        tokens.extend(entity_tokens)
        labels.extend(tags)

    return " ".join(tokens), labels

def generate_dataset(num_samples=100):
    data = {"sentence": [], "labels": []}
    for _ in range(num_samples):
        sentence, label_seq = generate_sentence()
        data["sentence"].append(sentence)
        data["labels"].append(label_seq)
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_dataset(num_samples=100)
    df.to_csv("bert_ner_project/data/synthetic_ner_data.csv", index=False)
    print("✅ Synthetic NER data saved to bert_ner_project/data/synthetic_ner_data.csv")
