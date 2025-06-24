import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments

class NERDataset(Dataset):
    def __init__(self, texts, tags, tokenizer, label2id, max_len=64):
        self.encodings = tokenizer(texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=max_len)
        self.labels = []
        for i, label in enumerate(tags):
            word_ids = self.encodings.word_ids(batch_index=i)
            label_ids = [-100 if id is None else label2id[label[id]] for id in word_ids]
            self.labels.append(label_ids)
        self.encodings.pop("offset_mapping")

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Load data
df = pd.read_csv("bert_ner_project/data/synthetic_ner_data.csv")
df['labels'] = df['labels'].apply(eval)
sentences = [s.split() for s in df['sentence']]
labels = list(df['labels'])

# Label mapping
unique_tags = sorted(set(tag for row in labels for tag in row))
label2id = {t: i for i, t in enumerate(unique_tags)}
id2label = {i: t for t, i in label2id.items()}

# Tokenizer and dataset
tokenizer = BertTokenizerFast.from_pretrained("bert-large-uncased")
dataset = NERDataset(sentences, labels, tokenizer, label2id)

# Model
model = BertForTokenClassification.from_pretrained("bert-large-uncased", num_labels=len(label2id), id2label=id2label, label2id=label2id)

# Training
#args = TrainingArguments("bert_ner_project/models", eval_strategy="epoch", save_strategy="epoch", num_train_epochs=3, per_device_train_batch_size=2)

args = TrainingArguments(
    output_dir="bert_ner_project/models",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_dir="bert_ner_project/logs",
    logging_steps=10
)

trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()

# Save model
trainer.save_model("bert_ner_project/models")
tokenizer.save_pretrained("bert_ner_project/models")
