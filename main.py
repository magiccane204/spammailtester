from fastapi import FastAPI
from pydantic import BaseModel
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = FastAPI()

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# ===============================
# TRAIN MODEL IF NOT EXISTS
# ===============================
if not os.path.exists(MODEL_PATH):

    data = pd.read_csv("spam.csv")
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.9,
        min_df=2
    )

    X = vectorizer.fit_transform(data['message'])
    y = data['label']

    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    )

    model.fit(X, y)

    pickle.dump(model, open(MODEL_PATH, "wb"))
    pickle.dump(vectorizer, open(VECTORIZER_PATH, "wb"))

else:
    model = pickle.load(open(MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))

# ===============================
# REQUEST SCHEMA
# ===============================
class Email(BaseModel):
    text: str

# ===============================
# PREDICTION ENDPOINT
# ===============================
@app.post("/predict")
def predict(email: Email):

    vector = vectorizer.transform([email.text])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0][prediction]

    return {
        "prediction": "Spam" if prediction == 1 else "Not Spam",
        "confidence": round(float(probability), 4)
    }
