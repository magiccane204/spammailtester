from fastapi import FastAPI
from pydantic import BaseModel
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = FastAPI()

# Auto-train if model does not exist
if not os.path.exists("model.pkl"):

    data = pd.read_csv("spam.csv")
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['message'])
    y = data['label']

    model = MultinomialNB()
    model.fit(X, y)

    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

else:
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
