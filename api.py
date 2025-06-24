from fastapi import FastAPI
from pydantic import BaseModel
from bert_ner_project.infer import predict_entities

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict/")
def predict(input: TextInput):
    result = predict_entities(input.text)
    return {"entities": result}
