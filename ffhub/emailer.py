"""Email the weekly report via Gmail SMTP (needs GMAIL_USER + GMAIL_APP_PASSWORD in .env)."""
from __future__ import annotations
import smtplib, ssl
from email.message import EmailMessage
from .config import load_env, OUT

def send(subject: str, text: str, html: str | None = None, to: str | None = None, attach_hub=True) -> str:
    env = load_env()
    user, pw = env.get("GMAIL_USER"), (env.get("GMAIL_APP_PASSWORD") or "").replace(" ", "")
    to = to or env.get("REPORT_TO") or user
    if not (user and pw):
        return "email skipped: set GMAIL_USER and GMAIL_APP_PASSWORD in .env (Google Account → Security → App passwords)"
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = user, to, subject
    msg.set_content(text)
    if html: msg.add_alternative(html, subtype="html")
    if attach_hub and (OUT / "hub.html").exists():
        msg.add_attachment((OUT / "hub.html").read_bytes(), maintype="text", subtype="html", filename="hub.html")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as s:
        s.login(user, pw); s.send_message(msg)
    return f"emailed {to}"
