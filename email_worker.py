import imaplib
import email
import requests
import time
import os

# ==============================
# LOAD ENV VARIABLES
# ==============================
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SPAM_API_URL = os.getenv("SPAM_API_URL")

if not EMAIL_ADDRESS or not EMAIL_PASSWORD or not SPAM_API_URL:
    raise Exception("Missing environment variables")

# ==============================
# CONNECT TO GMAIL
# ==============================
def connect():
    imap = imaplib.IMAP4_SSL("imap.gmail.com")
    imap.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    imap.select("inbox")
    return imap

# ==============================
# PROCESS EMAILS
# ==============================
def scan_emails():
    imap = connect()

    status, messages = imap.search(None, "UNSEEN")

    if messages[0] == b'':
        print("No new emails.")
        imap.logout()
        return

    for num in messages[0].split():
        status, msg_data = imap.fetch(num, "(RFC822)")
        msg = email.message_from_bytes(msg_data[0][1])

        subject = msg["Subject"] or ""
        body = ""

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode(errors="ignore")
                    break
        else:
            body = msg.get_payload(decode=True).decode(errors="ignore")

        full_text = subject + " " + body

        try:
            response = requests.post(SPAM_API_URL, json={"text": full_text})
            result = response.json()

            print("Checking:", subject, "| Prediction:", result["prediction"])

            if result["prediction"] == "Spam":
                print("Moving to Spam:", subject)
                imap.store(num, "+X-GM-LABELS", "\\Spam")

        except Exception as e:
            print("API Error:", e)

    imap.logout()

# ==============================
# RUN FOREVER
# ==============================
while True:
    print("Scanning inbox...")
    scan_emails()
    time.sleep(300)  # every 5 minutes
