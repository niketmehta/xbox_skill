import json
import logging
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from notifications import NotificationService
from recommendation_engine import format_top_recommendations_message


DB_PATH = Path("trading_data.db")
LOG_PATH = Path("logs") / "manual_whatsapp_resend.log"


def main():
    LOG_PATH.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT generated_at, payload_json
            FROM recommendation_runs
            ORDER BY generated_at DESC
            LIMIT 1
            """
        ).fetchone()

    if not row:
        print("sent=False")
        print("error=No recommendation run found")
        return 1

    result = json.loads(row["payload_json"])
    symbols = [pick.get("symbol") for pick in result.get("recommendations", [])]
    body = format_top_recommendations_message(result)

    logging.info(
        "Resending latest recommendation generated=%s symbols=%s chars=%s lines=%s",
        row["generated_at"],
        symbols,
        len(body),
        len(body.splitlines()),
    )

    notifier = NotificationService()
    sent = notifier.send_openclaw_whatsapp(body)
    error = notifier.get_last_error()
    logging.info("Manual resend complete sent=%s error=%s", sent, error)

    print(f"generated={row['generated_at']}")
    print(f"symbols={symbols}")
    print(f"chars={len(body)}")
    print(f"lines={len(body.splitlines())}")
    print(f"sent={sent}")
    if error:
        print(f"error={error}")
    return 0 if sent else 2


if __name__ == "__main__":
    raise SystemExit(main())
