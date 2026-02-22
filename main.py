from fastapi import FastAPI
from pydantic import BaseModel
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

app = FastAPI()

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
DATA_PATH = "spam.csv"

# ===============================
# TRAIN MODEL IF NOT EXISTS
# ===============================
if not os.path.exists(MODEL_PATH):

    print("Training model...")

    data = pd.read_csv(DATA_PATH)
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

    print("Model trained and saved.")

else:
    print("Loading existing model...")
    model = pickle.load(open(MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))

# ===============================
# COMPUTE METRICS ON STARTUP
# ===============================
print("Calculating model metrics...")

data = pd.read_csv(DATA_PATH)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

X_full = vectorizer.transform(data['message'])
y_true = data['label']
y_pred = model.predict(X_full)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
accuracy = accuracy_score(y_true, y_pred)

print("Metrics ready.")

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

# ===============================
# METRICS ENDPOINT
# ===============================
@app.get("/model-metrics")
def model_metrics():

    return {
        "accuracy": round(float(accuracy), 4),
        "quadrants": {
            "True Positive (Spam correctly detected)": int(tp),
            "True Negative (Ham correctly detected)": int(tn),
            "False Positive (Ham marked as Spam)": int(fp),
            "False Negative (Spam missed)": int(fn)
        }
    }
