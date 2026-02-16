"""
SMS notification service using Twilio.

Sends alerts for:
  - Trade executions (open / close)
  - High-confidence BUY / SELL signals
  - Stop-loss / take-profit triggers
  - Risk alerts (daily loss limit, drawdown)
"""

import logging
from datetime import datetime
from typing import Optional
from config import Config

logger = logging.getLogger(__name__)


class NotificationService:
    """Thin wrapper around Twilio SMS."""

    def __init__(self):
        self.config = Config()
        self._client = None
        self._enabled = self.config.NOTIFICATIONS_ENABLED
        self._phone = self.config.NOTIFICATION_PHONE_NUMBER
        self._from = self.config.TWILIO_FROM_NUMBER
        self._init_twilio()

    # ── Setup ───────────────────────────────────────────────────────

    def _init_twilio(self):
        """Lazily initialise Twilio client. Silently disable if creds missing."""
        if not (self.config.TWILIO_ACCOUNT_SID and self.config.TWILIO_AUTH_TOKEN):
            logger.warning("Twilio credentials not configured – SMS notifications disabled")
            self._enabled = False
            return
        try:
            from twilio.rest import Client
            self._client = Client(
                self.config.TWILIO_ACCOUNT_SID,
                self.config.TWILIO_AUTH_TOKEN
            )
            logger.info("Twilio SMS client initialised")
        except ImportError:
            logger.warning("twilio package not installed – SMS notifications disabled. "
                           "Install with: pip install twilio")
            self._enabled = False
        except Exception as e:
            logger.error(f"Twilio init error: {e}")
            self._enabled = False

    # ── Core send ───────────────────────────────────────────────────

    def _send_sms(self, body: str) -> bool:
        """Send an SMS message. Returns True on success."""
        if not self._enabled or not self._client:
            return False
        if not self._phone or not self._from:
            logger.warning("Phone numbers not configured for SMS")
            return False
        try:
            message = self._client.messages.create(
                body=body,
                from_=self._from,
                to=self._phone
            )
            logger.info(f"SMS sent (SID {message.sid}): {body[:60]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return False

    # ── Public helpers ──────────────────────────────────────────────

    def is_enabled(self) -> bool:
        return self._enabled and self._client is not None

    def enable(self, enabled: bool = True):
        """Toggle notifications on/off at runtime."""
        if enabled and not self._client:
            self._init_twilio()
        self._enabled = enabled

    def set_phone_number(self, phone: str):
        """Update the destination phone number at runtime."""
        self._phone = phone
        logger.info(f"Notification phone updated to {phone}")

    def get_phone_number(self) -> str:
        return self._phone or ''

    # ── High-level notification methods ─────────────────────────────

    def notify_trade_opened(self, symbol: str, side: str, quantity: int,
                            price: float, horizon: str = 'DAY'):
        body = (
            f"📈 TRADE OPENED\n"
            f"{side.upper()} {quantity} x {symbol} @ ${price:.2f}\n"
            f"Horizon: {horizon}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self._send_sms(body)

    def notify_trade_closed(self, symbol: str, quantity: int,
                            exit_price: float, pnl: float, reason: str):
        emoji = "✅" if pnl >= 0 else "❌"
        body = (
            f"{emoji} TRADE CLOSED\n"
            f"{symbol} x{quantity} @ ${exit_price:.2f}\n"
            f"P&L: ${pnl:+.2f}\n"
            f"Reason: {reason}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self._send_sms(body)

    def notify_signal(self, symbol: str, action: str, confidence: float,
                      price: float, horizon: str = 'DAY'):
        body = (
            f"🔔 SIGNAL: {action} {symbol}\n"
            f"Confidence: {confidence:.1f}%\n"
            f"Price: ${price:.2f}\n"
            f"Horizon: {horizon}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self._send_sms(body)

    def notify_risk_alert(self, message: str):
        body = f"⚠️ RISK ALERT\n{message}\nTime: {datetime.now().strftime('%H:%M:%S')}"
        self._send_sms(body)

    def notify_stop_loss(self, symbol: str, price: float, pnl: float):
        body = (
            f"🛑 STOP LOSS HIT\n"
            f"{symbol} @ ${price:.2f}\n"
            f"P&L: ${pnl:+.2f}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self._send_sms(body)

    def notify_take_profit(self, symbol: str, price: float, pnl: float):
        body = (
            f"🎯 TAKE PROFIT HIT\n"
            f"{symbol} @ ${price:.2f}\n"
            f"P&L: ${pnl:+.2f}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self._send_sms(body)

    def send_test(self) -> bool:
        """Send a test SMS to verify configuration."""
        return self._send_sms(
            "🤖 Trading Agent test notification – SMS is working!"
        )
