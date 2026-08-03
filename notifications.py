"""
Notification service for trading alerts.

Sends alerts for:
  - Trade executions (open / close)
  - High-confidence BUY / SELL signals
  - Stop-loss / take-profit triggers
  - Risk alerts
  - Trading council digests
"""

from email.message import EmailMessage
from email.utils import parseaddr
import logging
import os
import re
import shutil
import smtplib
import ssl
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config import Config

logger = logging.getLogger(__name__)


def _format_signed_dollars(value: float) -> str:
    amount = float(value or 0.0)
    sign = "+" if amount >= 0 else "-"
    return f"{sign}${abs(amount):.2f}"


class NotificationService:
    """Routes trading alerts to the configured notification channel."""

    RETRYABLE_NO_DELIVERY_ERRORS = (
        "No active WhatsApp Web listener",
        "Gateway not reachable",
        "ECONNREFUSED",
        "gateway closed (1006",
    )
    AMBIGUOUS_DELIVERY_ERRORS = (
        "GatewayTransportError",
        "gateway timeout",
        "send timed out",
        "timed out",
        "gateway closed",
    )
    BLOCKED_TARGET_TOKENS = {
        "all",
        "everyone",
        "contacts",
        "all contacts",
        "broadcast",
        "broadcasts",
        "groups",
        "group",
    }

    def __init__(self):
        self.config = Config()
        self._notification_channel = self._normalize_channel(self.config.NOTIFICATION_CHANNEL)
        self._email_enabled = self.config.EMAIL_ENABLED
        self._email_to = self.config.EMAIL_TO
        self._email_allowed_recipients = set(
            self._parse_email_recipients(self.config.EMAIL_ALLOWED_RECIPIENTS)
        )
        self._openclaw_enabled = self.config.OPENCLAW_ENABLED
        self._openclaw_target = self.config.OPENCLAW_WHATSAPP_TARGET
        self._allowed_targets = set(self.config.OPENCLAW_ALLOWED_TARGETS)
        self._last_error = ""

    def is_enabled(self) -> bool:
        if self.get_delivery_channel() == "email":
            return self._email_enabled
        if self.get_delivery_channel() == "openclaw_whatsapp":
            return self._openclaw_enabled
        return False

    def enable(self, enabled: bool = True):
        """Toggle the active notification channel at runtime."""
        if self.get_delivery_channel() == "email":
            self.enable_email(enabled)
        elif self.get_delivery_channel() == "openclaw_whatsapp":
            self.enable_openclaw(enabled)

    def _normalize_channel(self, channel: str) -> str:
        channel = str(channel or "").strip().lower().replace("-", "_")
        if channel in {"email", "smtp"}:
            return "email"
        if channel in {"whatsapp", "openclaw", "openclaw_whatsapp"}:
            return "openclaw_whatsapp"
        if channel in {"none", "off", "disabled"}:
            return "none"
        return channel or "none"

    def set_notification_channel(self, channel: str) -> bool:
        normalized = self._normalize_channel(channel)
        if normalized not in {"email", "openclaw_whatsapp", "none"}:
            return self._set_last_error(f"Unsupported notification channel: {channel}")
        self._notification_channel = normalized
        logger.info("Notification channel updated to %s", normalized)
        return True

    def get_delivery_channel(self) -> str:
        return self._notification_channel

    def get_delivery_target(self, target: Optional[str] = None) -> str:
        if self.get_delivery_channel() == "email":
            return target or self._email_to or ""
        if self.get_delivery_channel() == "openclaw_whatsapp":
            return target or self._openclaw_target or ""
        return ""

    def is_email_enabled(self) -> bool:
        return self._email_enabled

    def enable_email(self, enabled: bool = True):
        """Toggle email delivery at runtime."""
        self._email_enabled = bool(enabled)

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

    def is_openclaw_enabled(self) -> bool:
        return self._openclaw_enabled

    def enable_openclaw(self, enabled: bool = True):
        """Toggle OpenClaw WhatsApp delivery at runtime."""
        self._openclaw_enabled = bool(enabled)

    def set_openclaw_target(self, target: str) -> bool:
        """Update the OpenClaw WhatsApp destination at runtime."""
        target = str(target or "").strip()
        if not self._validate_openclaw_target(target):
            return False
        self._openclaw_target = target
        logger.info("OpenClaw WhatsApp target updated to %s", self._redact_target(target))
        return True

    def get_openclaw_target(self) -> str:
        return self._openclaw_target or ""

    def get_openclaw_allowed_targets(self) -> List[str]:
        return sorted(self._redact_target(target) for target in self._allowed_targets)

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
        """Send through the configured notification channel."""
        channel = self.get_delivery_channel()
        if channel == "email":
            return self.send_email(body, subject=subject, target=target)
        if channel == "openclaw_whatsapp":
            return self.send_openclaw_whatsapp(body, target=target)
        return self._set_last_error("Notifications disabled")

    def _validate_openclaw_target(self, target: str) -> bool:
        target = str(target or "").strip()
        if not target:
            return self._set_last_error("OpenClaw WhatsApp target is not configured")

        if self.config.OPENCLAW_CHANNEL != "whatsapp":
            return self._set_last_error("OpenClaw blocked: only the WhatsApp channel is allowed")

        lowered = re.sub(r"\s+", " ", target.lower()).strip()
        if lowered in self.BLOCKED_TARGET_TOKENS:
            return self._set_last_error(
                f"OpenClaw blocked unsafe WhatsApp target: {self._redact_target(target)}"
            )

        if any(separator in target for separator in (",", ";", "\n", "\r")):
            return self._set_last_error("OpenClaw blocked multi-recipient WhatsApp target")

        if self._allowed_targets and target not in self._allowed_targets:
            return self._set_last_error(
                "OpenClaw blocked target outside OPENCLAW_ALLOWED_TARGETS: "
                f"{self._redact_target(target)}"
            )

        return True

    def _resolve_openclaw_cli(self) -> str:
        configured = self.config.OPENCLAW_CLI or "openclaw"
        resolved = shutil.which(configured)
        if resolved:
            return resolved

        if os.name == "nt":
            if not configured.lower().endswith((".cmd", ".exe", ".bat")):
                resolved = shutil.which(f"{configured}.cmd")
                if resolved:
                    return resolved

            appdata = os.environ.get("APPDATA")
            if appdata:
                npm_shim = Path(appdata) / "npm" / "openclaw.cmd"
                if npm_shim.exists():
                    return str(npm_shim)

        return configured

    def _resolve_openclaw_command_prefix(self) -> List[str]:
        configured = self.config.OPENCLAW_CLI or "openclaw"

        if os.name == "nt" and Path(configured).name.lower() in {
            "openclaw",
            "openclaw.cmd",
            "openclaw.ps1",
        }:
            appdata = os.environ.get("APPDATA")
            if appdata:
                npm_dir = Path(appdata) / "npm"
                entrypoint = npm_dir / "node_modules" / "openclaw" / "openclaw.mjs"
                node = shutil.which("node")
                if not node:
                    default_node = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "nodejs" / "node.exe"
                    if default_node.exists():
                        node = str(default_node)
                if node and entrypoint.exists():
                    return [node, str(entrypoint)]

        return [self._resolve_openclaw_cli()]

    def _is_recoverable_gateway_error(self, output: str) -> bool:
        return self._is_retryable_no_delivery_error(output) or (
            self.config.OPENCLAW_RETRY_AMBIGUOUS_SENDS
            and self._is_ambiguous_delivery_error(output)
        )

    def _is_retryable_no_delivery_error(self, output: str) -> bool:
        return any(marker in output for marker in self.RETRYABLE_NO_DELIVERY_ERRORS)

    def _is_ambiguous_delivery_error(self, output: str) -> bool:
        return any(marker in output for marker in self.AMBIGUOUS_DELIVERY_ERRORS)

    def _openclaw_process_timeout(self, timeout: int) -> int:
        handshake_ms = max(10000, int(self.config.OPENCLAW_HANDSHAKE_TIMEOUT_MS or 0))
        handshake_seconds = (handshake_ms + 999) // 1000
        return max(int(timeout or 0), handshake_seconds + 15)

    def _run_openclaw_command(self, cmd, timeout: int):
        env = os.environ.copy()
        env["OPENCLAW_HANDSHAKE_TIMEOUT_MS"] = str(self.config.OPENCLAW_HANDSHAKE_TIMEOUT_MS)
        kwargs = {
            "capture_output": True,
            "text": True,
            "timeout": self._openclaw_process_timeout(timeout),
            "check": False,
            "env": env,
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            kwargs["startupinfo"] = startupinfo
        return subprocess.run(cmd, **kwargs)

    def _restart_openclaw_gateway(self, openclaw_cmd: List[str]) -> bool:
        logger.warning("Restarting OpenClaw gateway before retrying WhatsApp delivery")
        try:
            result = self._run_openclaw_command(
                openclaw_cmd + ["gateway", "restart"],
                timeout=90,
            )
            output = (result.stderr or result.stdout or "").strip()
            restart_was_started = "Restarted Windows login item" in output
            if result.returncode != 0 and not restart_was_started:
                logger.error(
                    "OpenClaw gateway restart failed (%s): %s",
                    result.returncode,
                    output,
                )
                return False
            if result.returncode != 0:
                logger.warning(
                    "OpenClaw gateway restart reported non-zero exit but appears started: %s",
                    output,
                )
            time.sleep(15)
            return True
        except Exception as e:
            logger.error("OpenClaw gateway restart error: %s", e)
            return False

    def _send_openclaw_with_retries(self, cmd: List[str], attempts: int) -> tuple:
        last_returncode = 1
        last_output = ""
        attempts = max(1, int(attempts or 1))

        for attempt in range(1, attempts + 1):
            try:
                result = self._run_openclaw_command(
                    cmd,
                    timeout=self.config.OPENCLAW_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired:
                last_returncode = "timeout"
                last_output = "OpenClaw WhatsApp send timed out"
            else:
                if result.returncode == 0:
                    return True, result.returncode, ""
                last_returncode = result.returncode
                last_output = (result.stderr or result.stdout or "").strip()

            if (
                attempt < attempts
                and self._is_ambiguous_delivery_error(last_output)
                and not self._is_retryable_no_delivery_error(last_output)
                and not self.config.OPENCLAW_RETRY_AMBIGUOUS_SENDS
            ):
                logger.warning(
                    "OpenClaw WhatsApp attempt %s/%s returned ambiguous delivery status; not retrying to avoid duplicate WhatsApp messages: %s",
                    attempt,
                    attempts,
                    last_output,
                )
                break

            if attempt < attempts and self._is_recoverable_gateway_error(last_output):
                sleep_seconds = min(8 * attempt, 24)
                logger.warning(
                    "OpenClaw WhatsApp attempt %s/%s failed; retrying in %ss: %s",
                    attempt,
                    attempts,
                    sleep_seconds,
                    last_output,
                )
                time.sleep(sleep_seconds)
                continue
            break

        return False, last_returncode, last_output

    def send_openclaw_whatsapp(self, body: str, target: Optional[str] = None) -> bool:
        """Send a WhatsApp message through OpenClaw's CLI gateway."""
        self._last_error = ""
        body = str(body or "").strip()
        if not body:
            return self._set_last_error("OpenClaw WhatsApp message body is empty")

        if not self._openclaw_enabled:
            return self._set_last_error("OpenClaw WhatsApp delivery disabled")

        destination = target or self._openclaw_target
        if not self._validate_openclaw_target(destination):
            return False

        openclaw_cmd = self._resolve_openclaw_command_prefix()
        first_line = body.splitlines()[0] if body else ""
        logger.info(
            "OpenClaw WhatsApp message prepared chars=%s lines=%s cli=%s target=%s first_line=%r",
            len(body),
            len(body.splitlines()),
            Path(openclaw_cmd[0]).name,
            self._redact_target(destination),
            first_line[:120],
        )

        cmd = openclaw_cmd + ["message", "send"]
        if self.config.OPENCLAW_ACCOUNT:
            cmd.extend(["--account", self.config.OPENCLAW_ACCOUNT])
        cmd.extend([
            "--channel",
            self.config.OPENCLAW_CHANNEL,
            "--target",
            destination,
            "--message",
            body,
            "--json",
        ])

        try:
            sent, returncode, output = self._send_openclaw_with_retries(
                cmd,
                self.config.OPENCLAW_SEND_ATTEMPTS,
            )
            if sent:
                logger.info("OpenClaw WhatsApp sent")
                return True
            if (
                self.config.OPENCLAW_AUTO_RESTART
                and self._is_recoverable_gateway_error(output)
                and self._restart_openclaw_gateway(openclaw_cmd)
            ):
                sent, returncode, output = self._send_openclaw_with_retries(
                    cmd,
                    self.config.OPENCLAW_SEND_ATTEMPTS,
                )
                if sent:
                    logger.info("OpenClaw WhatsApp sent after gateway restart")
                    return True
            return self._set_last_error(
                "OpenClaw WhatsApp failed "
                f"({returncode}): {output}"
            )
        except FileNotFoundError:
            return self._set_last_error(f"OpenClaw CLI not found: {' '.join(openclaw_cmd)}")
        except Exception as e:
            return self._set_last_error(f"OpenClaw WhatsApp send error: {e}")

    def send_openclaw_test(self, target: Optional[str] = None) -> bool:
        """Send a test WhatsApp message through OpenClaw."""
        return self.send_openclaw_whatsapp(
            "Trading Agent test notification via OpenClaw WhatsApp.",
            target=target,
        )

    def send_email_test(self, target: Optional[str] = None) -> bool:
        """Send a test email notification."""
        return self.send_email(
            "Trading Agent test notification via email.",
            subject="Test notification",
            target=target,
        )

    def send_test(self, target: Optional[str] = None) -> bool:
        """Send a test through the active notification channel."""
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
