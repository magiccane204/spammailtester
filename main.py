from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import io

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

app = FastAPI()

DATA_PATH = "spam.csv"

# ===============================
# LOAD DATA
# ===============================
data = pd.read_csv(DATA_PATH)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# ===============================
# TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    data['message'],
    data['label'],
    test_size=0.2,
    random_state=42,
    stratify=data['label']
)

# ===============================
# VECTORIZATION (Stronger Features)
# ===============================
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2),        # unigrams + bigrams
    max_df=0.95,
    min_df=2,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ===============================
# MODEL (High Precision)
# ===============================
model = LinearSVC(
    C=1.5,
    class_weight='balanced'
)

model.fit(X_train_vec, y_train)

# ===============================
# EVALUATION (ONLY TEST SET)
# ===============================
y_pred = model.predict(X_test_vec)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# ===============================
# ROOT ROUTE
# ===============================
@app.get("/")
def home():
    return {"status": "High-Precision Spam API is live 🚀"}

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

    return {
        "prediction": "Spam" if prediction == 1 else "Not Spam"
    }

# ===============================
# CONFUSION MATRIX (IN MEMORY)
# ===============================
@app.get("/confusion-matrix")
def get_confusion_matrix():

    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(cm, interpolation='nearest')
    ax.set_title("Confusion Matrix")

    classes = ["Ham", "Spam"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center")

    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

# ===============================
# METRICS ENDPOINT
# ===============================
@app.get("/model-metrics")
def model_metrics():

    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": round(float(accuracy), 4),
        "precision_spam": round(report["1"]["precision"], 4),
        "recall_spam": round(report["1"]["recall"], 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        }
    }
