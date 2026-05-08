"""
Notification service using OpenClaw WhatsApp.

Sends alerts for:
  - Trade executions (open / close)
  - High-confidence BUY / SELL signals
  - Stop-loss / take-profit triggers
  - Risk alerts
  - Trading council digests
"""

import logging
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import Config

logger = logging.getLogger(__name__)


class NotificationService:
    """Thin wrapper around OpenClaw WhatsApp delivery."""

    def __init__(self):
        self.config = Config()
        self._openclaw_enabled = self.config.OPENCLAW_ENABLED
        self._openclaw_target = self.config.OPENCLAW_WHATSAPP_TARGET
        self._last_error = ""

    def is_enabled(self) -> bool:
        return self._openclaw_enabled

    def enable(self, enabled: bool = True):
        """Compatibility alias for toggling notifications at runtime."""
        self.enable_openclaw(enabled)

    def is_openclaw_enabled(self) -> bool:
        return self._openclaw_enabled

    def enable_openclaw(self, enabled: bool = True):
        """Toggle OpenClaw WhatsApp delivery at runtime."""
        self._openclaw_enabled = bool(enabled)

    def set_openclaw_target(self, target: str):
        """Update the OpenClaw WhatsApp destination at runtime."""
        self._openclaw_target = target
        logger.info("OpenClaw WhatsApp target updated")

    def get_openclaw_target(self) -> str:
        return self._openclaw_target or ""

    def get_last_error(self) -> str:
        return self._last_error

    def _set_last_error(self, message: str) -> bool:
        self._last_error = message
        logger.error(message)
        return False

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

    def send_openclaw_whatsapp(self, body: str, target: Optional[str] = None) -> bool:
        """Send a WhatsApp message through OpenClaw's CLI gateway."""
        self._last_error = ""
        if not self._openclaw_enabled:
            return self._set_last_error("OpenClaw WhatsApp delivery disabled")

        destination = target or self._openclaw_target
        if not destination:
            return self._set_last_error("OpenClaw WhatsApp target is not configured")

        openclaw_cli = self._resolve_openclaw_cli()

        cmd = [
            openclaw_cli,
            "message",
            "send",
            "--channel",
            self.config.OPENCLAW_CHANNEL,
            "--target",
            destination,
            "--message",
            body,
            "--json",
        ]
        if self.config.OPENCLAW_ACCOUNT:
            cmd[5:5] = ["--account", self.config.OPENCLAW_ACCOUNT]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.OPENCLAW_TIMEOUT_SECONDS,
                check=False,
            )
            if result.returncode == 0:
                logger.info("OpenClaw WhatsApp sent")
                return True
            return self._set_last_error(
                "OpenClaw WhatsApp failed "
                f"({result.returncode}): {(result.stderr or result.stdout or '').strip()}"
            )
        except FileNotFoundError:
            return self._set_last_error(f"OpenClaw CLI not found: {openclaw_cli}")
        except subprocess.TimeoutExpired:
            return self._set_last_error("OpenClaw WhatsApp send timed out")
        except Exception as e:
            return self._set_last_error(f"OpenClaw WhatsApp send error: {e}")

    def send_openclaw_test(self, target: Optional[str] = None) -> bool:
        """Send a test WhatsApp message through OpenClaw."""
        return self.send_openclaw_whatsapp(
            "Trading Agent test notification via OpenClaw WhatsApp.",
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
            "TRADE OPENED\n"
            f"{side.upper()} {quantity} x {symbol} @ ${price:.2f}\n"
            f"Horizon: {horizon}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self.send_openclaw_whatsapp(body)

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
            f"TRADE CLOSED - {outcome}\n"
            f"{symbol} x{quantity} @ ${exit_price:.2f}\n"
            f"P&L: ${pnl:+.2f}\n"
            f"Reason: {reason}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self.send_openclaw_whatsapp(body)

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
        self.send_openclaw_whatsapp(body)

    def notify_risk_alert(self, message: str):
        body = f"RISK ALERT\n{message}\nTime: {datetime.now().strftime('%H:%M:%S')}"
        self.send_openclaw_whatsapp(body)

    def notify_stop_loss(self, symbol: str, price: float, pnl: float):
        body = (
            "STOP LOSS HIT\n"
            f"{symbol} @ ${price:.2f}\n"
            f"P&L: ${pnl:+.2f}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self.send_openclaw_whatsapp(body)

    def notify_take_profit(self, symbol: str, price: float, pnl: float):
        body = (
            "TAKE PROFIT HIT\n"
            f"{symbol} @ ${price:.2f}\n"
            f"P&L: ${pnl:+.2f}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self.send_openclaw_whatsapp(body)
