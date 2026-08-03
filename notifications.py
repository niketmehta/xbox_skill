"""
Email notification service for trading alerts.

Sends alerts for:
  - Trade executions (open / close)
  - High-confidence BUY / SELL signals
  - Stop-loss / take-profit triggers
  - Risk alerts
  - Trading council digests
"""

from datetime import datetime
from email.message import EmailMessage
from email.utils import parseaddr
import logging
import re
import smtplib
import ssl
from typing import List, Optional

from config import Config

logger = logging.getLogger(__name__)


def _format_signed_dollars(value: float) -> str:
    amount = float(value or 0.0)
    sign = "+" if amount >= 0 else "-"
    return f"{sign}${abs(amount):.2f}"


class NotificationService:
    """Sends trading alerts by email."""

    def __init__(self):
        self.config = Config()
        self._notification_channel = self._normalize_channel(self.config.NOTIFICATION_CHANNEL)
        self._email_enabled = self.config.EMAIL_ENABLED
        self._email_to = self.config.EMAIL_TO
        self._email_allowed_recipients = set(
            self._parse_email_recipients(self.config.EMAIL_ALLOWED_RECIPIENTS)
        )
        self._last_error = ""

    def is_enabled(self) -> bool:
        return self._notification_channel == "email" and self._email_enabled

    def enable(self, enabled: bool = True):
        """Toggle email delivery at runtime."""
        self.enable_email(enabled)

    def _normalize_channel(self, channel: str) -> str:
        channel = str(channel or "").strip().lower().replace("-", "_")
        if channel in {"email", "smtp"}:
            return "email"
        if channel in {"none", "off", "disabled"}:
            return "none"
        if not channel:
            return "email" if self.config.EMAIL_ENABLED else "none"
        return channel

    def set_notification_channel(self, channel: str) -> bool:
        normalized = self._normalize_channel(channel)
        if normalized not in {"email", "none"}:
            return self._set_last_error(f"Unsupported notification channel: {channel}")
        self._notification_channel = normalized
        logger.info("Notification channel updated to %s", normalized)
        return True

    def get_delivery_channel(self) -> str:
        return self._notification_channel

    def get_delivery_target(self, target: Optional[str] = None) -> str:
        if self.get_delivery_channel() == "email":
            return target or self._email_to or ""
        return ""

    def is_email_enabled(self) -> bool:
        return self._email_enabled

    def enable_email(self, enabled: bool = True):
        """Toggle email delivery at runtime."""
        self._email_enabled = bool(enabled)
        if self._email_enabled and self._notification_channel == "none":
            self._notification_channel = "email"

    def set_email_target(self, target: str) -> bool:
        target = str(target or "").strip()
        recipients = self._parse_email_recipients(target)
        if not self._validate_email_recipients(recipients):
            return False
        self._email_to = ", ".join(recipients)
        logger.info("Email notification target updated to %s", self._redact_targets(recipients))
        return True

    def get_email_target(self) -> str:
        return self._email_to or ""

    def get_email_allowed_recipients(self) -> List[str]:
        return sorted(self._redact_target(target) for target in self._email_allowed_recipients)

    def get_last_error(self) -> str:
        return self._last_error

    def _set_last_error(self, message: str) -> bool:
        self._last_error = message
        logger.error(message)
        return False

    def _redact_target(self, target: str) -> str:
        target = str(target or "")
        if len(target) <= 4:
            return "***"
        return f"{target[:3]}...{target[-2:]}"

    def _redact_targets(self, targets: List[str]) -> str:
        return ", ".join(self._redact_target(target) for target in targets)

    def _parse_email_recipients(self, target: Optional[str] = None) -> List[str]:
        raw = target if target is not None else self._email_to
        if isinstance(raw, (list, tuple, set)):
            parts = [str(item or "").strip() for item in raw]
        else:
            parts = [
                part.strip()
                for part in re.split(r"[,;]", str(raw or ""))
                if part.strip()
            ]

        recipients = []
        seen = set()
        for part in parts:
            _, address = parseaddr(part)
            address = address.strip()
            if not address or address.lower() in seen:
                continue
            seen.add(address.lower())
            recipients.append(address)
        return recipients

    def _validate_email_recipients(self, recipients: List[str]) -> bool:
        if not recipients:
            return self._set_last_error("Email recipient is not configured")

        for recipient in recipients:
            if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", recipient):
                return self._set_last_error(f"Invalid email recipient: {recipient}")

        allowed = {item.lower() for item in self._email_allowed_recipients}
        if allowed:
            blocked = [recipient for recipient in recipients if recipient.lower() not in allowed]
            if blocked:
                return self._set_last_error(
                    "Email blocked recipient outside EMAIL_ALLOWED_RECIPIENTS: "
                    f"{self._redact_targets(blocked)}"
                )

        return True

    def _email_subject(self, subject: Optional[str]) -> str:
        subject = str(subject or "Trading Agent notification").strip()
        subject = re.sub(r"[\r\n]+", " ", subject)
        prefix = str(self.config.EMAIL_SUBJECT_PREFIX or "").strip()
        if prefix and not subject.startswith(prefix):
            subject = f"{prefix} {subject}"
        return subject[:160]

    def send_email(
        self,
        body: str,
        subject: Optional[str] = None,
        target: Optional[str] = None,
    ) -> bool:
        """Send an email notification through SMTP."""
        self._last_error = ""
        body = str(body or "").strip()
        if not body:
            return self._set_last_error("Email message body is empty")

        if not self._email_enabled:
            return self._set_last_error("Email delivery disabled")

        if not self.config.EMAIL_SMTP_HOST:
            return self._set_last_error("EMAIL_SMTP_HOST is not configured")

        sender = self.config.EMAIL_FROM or self.config.EMAIL_SMTP_USERNAME
        if not sender:
            return self._set_last_error("EMAIL_FROM or EMAIL_SMTP_USERNAME is not configured")

        recipients = self._parse_email_recipients(target)
        if not self._validate_email_recipients(recipients):
            return False

        message = EmailMessage()
        message["Subject"] = self._email_subject(subject)
        message["From"] = sender
        message["To"] = ", ".join(recipients)
        message.set_content(body)

        logger.info(
            "Email notification prepared chars=%s lines=%s host=%s port=%s target=%s subject=%r",
            len(body),
            len(body.splitlines()),
            self.config.EMAIL_SMTP_HOST,
            self.config.EMAIL_SMTP_PORT,
            self._redact_targets(recipients),
            message["Subject"],
        )

        try:
            if self.config.EMAIL_USE_SSL:
                with smtplib.SMTP_SSL(
                    self.config.EMAIL_SMTP_HOST,
                    self.config.EMAIL_SMTP_PORT,
                    timeout=self.config.EMAIL_TIMEOUT_SECONDS,
                    context=ssl.create_default_context(),
                ) as server:
                    self._smtp_login(server)
                    server.send_message(message)
            else:
                with smtplib.SMTP(
                    self.config.EMAIL_SMTP_HOST,
                    self.config.EMAIL_SMTP_PORT,
                    timeout=self.config.EMAIL_TIMEOUT_SECONDS,
                ) as server:
                    server.ehlo()
                    if self.config.EMAIL_USE_TLS:
                        server.starttls(context=ssl.create_default_context())
                        server.ehlo()
                    self._smtp_login(server)
                    server.send_message(message)
            logger.info("Email notification sent")
            return True
        except Exception as e:
            return self._set_last_error(f"Email send error: {e}")

    def _smtp_login(self, server):
        if self.config.EMAIL_SMTP_USERNAME or self.config.EMAIL_SMTP_PASSWORD:
            server.login(
                self.config.EMAIL_SMTP_USERNAME,
                self.config.EMAIL_SMTP_PASSWORD,
            )

    def send_notification(
        self,
        body: str,
        subject: Optional[str] = None,
        target: Optional[str] = None,
    ) -> bool:
        """Send through the configured email channel."""
        if self.get_delivery_channel() == "email":
            return self.send_email(body, subject=subject, target=target)
        return self._set_last_error("Notifications disabled")

    def send_email_test(self, target: Optional[str] = None) -> bool:
        """Send a test email notification."""
        return self.send_email(
            "Trading Agent test notification via email.",
            subject="Test notification",
            target=target,
        )

    def send_test(self, target: Optional[str] = None) -> bool:
        """Send a test notification."""
        return self.send_notification(
            "Trading Agent test notification.",
            subject="Test notification",
            target=target,
        )

    def notify_trade_opened(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        horizon: str = "WEEK",
    ):
        body = (
            f"TRADE OPENED: {side.upper()} {quantity} x {symbol} @ ${price:.2f}\n"
            f"Horizon: {horizon}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self.send_notification(body, subject=f"Trade opened: {symbol}")

    def notify_trade_closed(
        self,
        symbol: str,
        quantity: int,
        exit_price: float,
        pnl: float,
        reason: str,
    ):
        outcome = "PROFIT" if pnl >= 0 else "LOSS"
        body = (
            (
                f"TRADE CLOSED - {outcome}: {symbol} x{quantity} "
                f"@ ${exit_price:.2f}, P/L {_format_signed_dollars(pnl)}"
            )
            + "\n"
            f"Reason: {reason}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self.send_notification(body, subject=f"Trade closed: {symbol} {outcome}")

    def notify_signal(
        self,
        symbol: str,
        action: str,
        confidence: float,
        price: float,
        horizon: str = "WEEK",
    ):
        body = (
            f"SIGNAL: {action} {symbol}\n"
            f"Confidence: {confidence:.1f}%\n"
            f"Price: ${price:.2f}\n"
            f"Horizon: {horizon}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self.send_notification(body, subject=f"Signal: {action} {symbol}")

    def notify_risk_alert(self, message: str):
        body = f"RISK ALERT: {message}\nTime: {datetime.now().strftime('%H:%M:%S')}"
        self.send_notification(body, subject="Risk alert")

    def notify_stop_loss(self, symbol: str, price: float, pnl: float):
        body = (
            f"STOP LOSS HIT: {symbol} @ ${price:.2f}, P/L {_format_signed_dollars(pnl)}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self.send_notification(body, subject=f"Stop loss hit: {symbol}")

    def notify_take_profit(self, symbol: str, price: float, pnl: float):
        body = (
            f"TAKE PROFIT HIT: {symbol} @ ${price:.2f}, P/L {_format_signed_dollars(pnl)}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self.send_notification(body, subject=f"Take profit hit: {symbol}")
