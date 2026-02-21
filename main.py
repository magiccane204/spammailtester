from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

class Email(BaseModel):
    text: str

@app.post("/predict")
def predict(email: Email):
    vector = vectorizer.transform([email.text])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0][prediction]

    return {
        "prediction": "Spam" if prediction == 1 else "Not Spam",
        "confidence": float(probability)
    }
