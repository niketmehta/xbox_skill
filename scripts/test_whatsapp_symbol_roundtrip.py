"""Send a ticker into the configured WhatsApp self-chat for round-trip testing."""

import argparse
import re

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
    notifications = NotificationService()
    sent = notifications.send_openclaw_whatsapp(message)
    print(f"sent={sent}")
    if not sent:
        print(f"error={notifications.get_last_error()}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
