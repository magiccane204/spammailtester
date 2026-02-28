from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import pickle
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # VERY IMPORTANT FOR RAILWAY
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

app = FastAPI()

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
DATA_PATH = "spam.csv"
IMAGE_PATH = "confusion_matrix.png"

# ===============================
# ROOT ROUTE (Prevents NX)
# ===============================
@app.get("/")
def home():
    return {"status": "Spam Detection API is live 🚀"}

# ===============================
# TRAIN OR LOAD MODEL
# ===============================
if not os.path.exists(MODEL_PATH):

    print("Training model...")

    data = pd.read_csv(DATA_PATH)
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    X_train, X_test, y_train, y_test = train_test_split(
        data['message'],
        data['label'],
        test_size=0.2,
        random_state=42
    )

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.9,
        min_df=2
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    )

    model.fit(X_train_vec, y_train)

    pickle.dump(model, open(MODEL_PATH, "wb"))
    pickle.dump(vectorizer, open(VECTORIZER_PATH, "wb"))

    print("Model trained and saved.")

else:
    print("Loading existing model...")
    model = pickle.load(open(MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))

# ===============================
# CALCULATE METRICS
# ===============================
data = pd.read_csv(DATA_PATH)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

X = data['message']
y = data['label']

X_vec = vectorizer.transform(X)
y_pred = model.predict(X_vec)

cm = confusion_matrix(y, y_pred)
accuracy = accuracy_score(y, y_pred)

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
# CONFUSION MATRIX IMAGE
# ===============================
@app.get("/confusion-matrix")
def get_confusion_matrix():

    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()

    classes = ["Ham", "Spam"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center")

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    plt.savefig(IMAGE_PATH)
    plt.close()

    return FileResponse(IMAGE_PATH, media_type="image/png")

# ===============================
# METRICS ENDPOINT
# ===============================
@app.get("/model-metrics")
def model_metrics():

    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": round(float(accuracy), 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        }
    }
