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
    """Thin wrapper around OpenClaw WhatsApp delivery."""

    RECOVERABLE_GATEWAY_ERRORS = (
        "No active WhatsApp Web listener",
        "GatewayTransportError",
        "gateway timeout",
        "send timed out",
        "timed out",
        "Gateway not reachable",
        "ECONNREFUSED",
    )

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
        return any(marker in output for marker in self.RECOVERABLE_GATEWAY_ERRORS)

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
        if not destination:
            return self._set_last_error("OpenClaw WhatsApp target is not configured")

        openclaw_cmd = self._resolve_openclaw_command_prefix()
        first_line = body.splitlines()[0] if body else ""
        logger.info(
            "OpenClaw WhatsApp message prepared chars=%s lines=%s cli=%s first_line=%r",
            len(body),
            len(body.splitlines()),
            Path(openclaw_cmd[0]).name,
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
            (
                f"TRADE CLOSED - {outcome}: {symbol} x{quantity} "
                f"@ ${exit_price:.2f}, P/L {_format_signed_dollars(pnl)}"
            )
            + "\n"
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
        body = f"RISK ALERT: {message}\nTime: {datetime.now().strftime('%H:%M:%S')}"
        self.send_openclaw_whatsapp(body)

    def notify_stop_loss(self, symbol: str, price: float, pnl: float):
        body = (
            f"STOP LOSS HIT: {symbol} @ ${price:.2f}, P/L {_format_signed_dollars(pnl)}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self.send_openclaw_whatsapp(body)

    def notify_take_profit(self, symbol: str, price: float, pnl: float):
        body = (
            f"TAKE PROFIT HIT: {symbol} @ ${price:.2f}, P/L {_format_signed_dollars(pnl)}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self.send_openclaw_whatsapp(body)
