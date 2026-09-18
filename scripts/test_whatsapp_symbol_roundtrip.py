"""Simulate an inbound ticker and deliver both sides of the WhatsApp round trip."""

import argparse
import re
import subprocess
from pathlib import Path

from notifications import NotificationService


SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test inbound WhatsApp council routing.")
    parser.add_argument("symbol", help="Ticker symbol, for example NVDA")
    parser.add_argument("--horizon", choices=("WEEK", "MONTH"), default="WEEK")
    args = parser.parse_args()

    symbol = args.symbol.strip().upper().lstrip("$")
    if not SYMBOL_RE.fullmatch(symbol):
        raise SystemExit("Invalid ticker symbol")

    message = symbol if args.horizon == "WEEK" else f"{symbol} month"
    simulator = (
        Path(__file__).resolve().parents[1]
        / "openclaw"
        / "plugins"
        / "stock-council-router"
        / "simulate.mjs"
    )
    simulated = subprocess.run(
        ["node", str(simulator), message],
        capture_output=True,
        text=True,
        timeout=210,
        check=False,
    )
    reply = simulated.stdout.strip()
    if simulated.returncode or not reply:
        print(f"simulation_error={simulated.stderr.strip() or 'empty response'}")
        return 1

    notifications = NotificationService()
    request_sent = notifications.send_openclaw_whatsapp(message)
    print(f"request_sent={request_sent}")
    if not request_sent:
        print(f"error={notifications.get_last_error()}")
        return 1

    response_sent = notifications.send_openclaw_whatsapp(reply)
    print(f"response_sent={response_sent}")
    if not response_sent:
        print(f"error={notifications.get_last_error()}")
        return 1
    print(f"response_first_line={reply.splitlines()[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
