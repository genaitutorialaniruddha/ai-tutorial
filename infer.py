from transformers import BertTokenizerFast, BertForTokenClassification
import torch

# Load model and tokenizer
model = BertForTokenClassification.from_pretrained("bert_ner_project/models")
tokenizer = BertTokenizerFast.from_pretrained("bert_ner_project/models")
model.eval()

id2label = model.config.id2label

def predict_entities(text: str):
    tokens = text.split()
    inputs = tokenizer(tokens, return_tensors="pt", is_split_into_words=True)
    with torch.no_grad():
        outputs = model(**inputs).logits
    predictions = torch.argmax(outputs, dim=2)
    predicted_labels = [id2label[pred.item()] for pred in predictions[0]]
    return list(zip(tokens, predicted_labels))
