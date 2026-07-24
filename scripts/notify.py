"""
Optional error alerts: email (SMTP) and/or ntfy.sh push.
Configured via config/notify.yaml (copy from notify.example.yaml).
"""
from __future__ import annotations

import smtplib
import threading
import urllib.error
import urllib.request
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
NOTIFY_CFG = ROOT / "config" / "notify.yaml"
LAST_ERROR = ROOT / "last_error_report.txt"


def _load_cfg() -> Dict[str, Any]:
    if not NOTIFY_CFG.is_file():
        return {}
    try:
        import yaml

        with NOTIFY_CFG.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_last_error_report(where: str, body: str) -> Path:
    text = (
        f"Bruker CT Toolkit error report\n"
        f"When: {datetime.now().isoformat(timespec='seconds')}\n"
        f"Where: {where}\n"
        f"{'=' * 60}\n"
        f"{body}\n"
    )
    try:
        LAST_ERROR.write_text(text, encoding="utf-8", errors="replace")
    except Exception:
        pass
    return LAST_ERROR


def _send_email(cfg: Dict[str, Any], subject: str, body: str) -> str:
    email_cfg = cfg.get("email") or {}
    if not email_cfg.get("enabled"):
        return "email disabled"
    to_addr = (email_cfg.get("to") or "").strip()
    from_addr = (email_cfg.get("from") or to_addr).strip()
    host = (email_cfg.get("smtp_host") or "").strip()
    port = int(email_cfg.get("smtp_port") or 587)
    user = (email_cfg.get("smtp_user") or "").strip()
    password = (email_cfg.get("smtp_password") or "").strip()
    use_tls = bool(email_cfg.get("use_tls", True))
    if not (to_addr and host and user and password):
        return "email incomplete config (need to/smtp_host/smtp_user/smtp_password)"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(body)

    with smtplib.SMTP(host, port, timeout=30) as smtp:
        if use_tls:
            smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)
    return f"email sent to {to_addr}"


def _send_ntfy(cfg: Dict[str, Any], subject: str, body: str) -> str:
    ntfy = cfg.get("ntfy") or {}
    if not ntfy.get("enabled"):
        return "ntfy disabled"
    topic = (ntfy.get("topic") or "").strip()
    if not topic:
        return "ntfy missing topic"
    server = (ntfy.get("server") or "https://ntfy.sh").rstrip("/")
    url = f"{server}/{topic}"
    data = f"{subject}\n\n{body[:3500]}".encode("utf-8", errors="replace")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Title": subject[:120],
            "Priority": "high",
            "Tags": "warning,ct",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        resp.read()
    return f"ntfy posted to {url}"


def notify_error(where: str, body: str) -> None:
    """
    Fire-and-forget alerts. Never raises into the GUI.
    Always writes last_error_report.txt.
    """
    save_last_error_report(where, body)
    cfg = _load_cfg()
    if not cfg:
        return

    subject = f"[Bruker CT Toolkit] ERROR: {where}"

    def _worker() -> None:
        notes = []
        try:
            notes.append(_send_email(cfg, subject, body))
        except Exception as exc:
            notes.append(f"email failed: {exc}")
        try:
            notes.append(_send_ntfy(cfg, subject, body))
        except Exception as exc:
            notes.append(f"ntfy failed: {exc}")
        try:
            from gui_log import log_line

            for n in notes:
                log_line(f"NOTIFY: {n}")
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()


def notify_status() -> str:
    """Short status for the Activity log / startup banner."""
    cfg = _load_cfg()
    if not cfg:
        return (
            f"Error alerts: OFF (no {NOTIFY_CFG.name}). "
            f"Copy config/notify.example.yaml -> config/notify.yaml to email yourself."
        )
    bits = []
    email = cfg.get("email") or {}
    ntfy = cfg.get("ntfy") or {}
    if email.get("enabled"):
        bits.append(f"email -> {email.get('to') or '?'}")
    if ntfy.get("enabled"):
        bits.append(f"ntfy -> {ntfy.get('topic') or '?'}")
    if not bits:
        return "Error alerts: config present but email/ntfy both disabled"
    return "Error alerts: ON (" + ", ".join(bits) + ")"
