"""Fetch one local council recommendation for an OpenClaw WhatsApp reply."""

import argparse
import json
import os
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import urlopen


SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the trading council for one ticker.")
    parser.add_argument("symbol", help="Ticker symbol, for example NVDA")
    parser.add_argument("--horizon", choices=("WEEK", "MONTH"), default="WEEK")
    args = parser.parse_args()

    symbol = args.symbol.strip().upper().lstrip("$")
    if not SYMBOL_RE.fullmatch(symbol):
        print("Please send one valid ticker symbol, for example NVDA or BRK.B.")
        return 2

    base_url = os.getenv("TRADING_DASHBOARD_URL", "http://127.0.0.1:5001").rstrip("/")
    url = (
        f"{base_url}/api/recommendations/symbol/{quote(symbol, safe='')}?"
        f"{urlencode({'horizon': args.horizon})}"
    )
    try:
        with urlopen(url, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("error")
        except Exception:
            detail = None
        print(f"Council could not analyze {symbol}: {detail or exc.reason}.")
        return 1
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Council service is unavailable for {symbol}: {exc}.")
        return 1

    message = str(result.get("message_preview") or "").strip()
    if not message:
        print(f"Council could not produce a recommendation for {symbol}.")
        return 1
    print(message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
