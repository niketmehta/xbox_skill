import argparse
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path("trading_data.db")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from council_memory import CouncilLearningMemory


def trade_dates():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT trade_date
            FROM simulated_recommendation_trades
            WHERE pnl_pct IS NOT NULL
            ORDER BY trade_date
            """
        ).fetchall()
    return [row[0] for row in rows]


def main():
    parser = argparse.ArgumentParser(description="Backfill council RAG learning memory.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of latest trade dates to rebuild.",
    )
    args = parser.parse_args()

    dates = trade_dates()
    if args.limit > 0:
        dates = dates[-args.limit:]

    memory = CouncilLearningMemory(DB_PATH)
    total_lessons = 0
    for trade_date in dates:
        result = memory.learn_from_trade_date(trade_date)
        count = result.get("lessons_written", 0)
        total_lessons += count
        print(f"{trade_date}: {count} lessons")

    print(f"Rebuilt council RAG memory for {len(dates)} dates; {total_lessons} lessons saved.")


if __name__ == "__main__":
    main()
