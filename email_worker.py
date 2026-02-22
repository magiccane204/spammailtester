import imaplib
import email
import requests
import time
import os
from email.header import decode_header

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
    imap.select("INBOX")
    return imap

# ==============================
# MOVE EMAIL TO SPAM
# ==============================
def move_to_spam(imap, msg_id):
    try:
        imap.copy(msg_id, "[Gmail]/Spam")
        imap.store(msg_id, "+FLAGS", "\\Deleted")
        imap.expunge()
        print("Successfully moved to Spam")
    except Exception as e:
        print("Move failed:", e)

# ==============================
# DECODE SUBJECT SAFELY
# ==============================
def decode_subject(subject_raw):
    decoded = decode_header(subject_raw or "")
    subject = ""

    for part, encoding in decoded:
        if isinstance(part, bytes):
            subject += part.decode(encoding if encoding else "utf-8", errors="ignore")
        else:
            subject += part

    return subject

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

        # ✅ SAFE SUBJECT DECODING
        subject = decode_subject(msg["Subject"])
        body = ""

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode(errors="ignore")
                    break
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode(errors="ignore")

        full_text = subject + " " + body

        try:
            response = requests.post(SPAM_API_URL, json={"text": full_text})
            result = response.json()

            print("Checking:", subject, "| Prediction:", result["prediction"])

            if result["prediction"] == "Spam":
                print("Moving to Spam:", subject)
                move_to_spam(imap, num)

        except Exception as e:
            print("API Error:", e)

    imap.logout()

# ==============================
# RUN FOREVER
# ==============================
while True:
    print("Scanning inbox...")
    scan_emails()
    time.sleep(30)
