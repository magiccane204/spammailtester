# 🛡️ SentinelMail: Real-Time Autonomous Email Classification & Defense Grid

## 💡 Why This Exists (The Justification)
Modern inboxes are absolute warzones flooded with phishing attempts, unwanted promotions, and algorithmic noise. Existing email filters are black boxes that either hide legitimate messages or let obvious spam slip through. 

**SentinelMail** was engineered to take back control of the inbox. By combining a lightweight background IMAP monitoring daemon with a high-performance FastAPI machine learning inference engine, this system offers autonomous, real-time tactical defense against email threats.

---

## ✨ Architectural Highlights

* **Autonomous Background Daemon:** Continuously polls the inbox via secure SSL IMAP, scans unread messages, and parses multi-part MIME payloads to extract text.
* **LinearSVC Intelligence Core:** Powered by a robust Support Vector Classifier backed by TF-IDF n-grams, stop-word elimination, and balanced class weights to aggressively catch sophisticated spam variants.
* **Automated Threat Neutralization:** Instantly isolates confirmed spam by copying items to the designated Gmail Spam directory and purging them from the active threat vector (Inbox).
* **Diagnostic Telemetry API:** A FastAPI backend offering built-in endpoints for real-time prediction, model performance metrics, and dynamically generated confusion matrix visual streams (`/confusion-matrix`).

---

## 🛠️ Tech Stack

* **Backend / API:** FastAPI, Uvicorn, Pydantic
* **Machine Learning:** Scikit-Learn (LinearSVC, MultinomialNB, TfidfVectorizer), Pandas, NumPy
* **Data Visualization:** Matplotlib (Headless Agg backend for real-time confusion matrix plotting)
* **Protocols & Automation:** `imaplib`, `email` (MIME parsing), `requests`

---

## 🚀 Deployment & Quickstart

1. **Configure Environment Variables:**
   ```bash
   export EMAIL_ADDRESS="your-email@gmail.com"
   export EMAIL_PASSWORD="your-app-password"
   export SPAM_API_URL="http://localhost:8000/predict"
