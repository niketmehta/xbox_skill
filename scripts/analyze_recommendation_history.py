"""
Audit recent council recommendations against simulated open-entry outcomes.

This script is intentionally read-only. It summarizes where the recommendation
engine has been losing money and checks whether named symbols ever entered the
recommendation universe.
"""

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


DB_PATH = Path("trading_data.db")
MISSED_SYMBOLS = ("SNDK", "WDC", "STX")


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def parse_args():
    parser = argparse.ArgumentParser(description="Audit recommendation history.")
    parser.add_argument("--limit", type=int, default=30, help="Recent simulated trades to show.")
    parser.add_argument(
        "--symbols",
        default=",".join(MISSED_SYMBOLS),
        help="Comma-separated symbols to check in saved recommendation payloads.",
    )
    return parser.parse_args()


def table_exists(conn, name):
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = ? AND name = ?",
        ("table", name),
    ).fetchone()
    return row is not None


def load_runs(conn):
    if not table_exists(conn, "recommendation_runs"):
        return []
    rows = conn.execute(
        """
        SELECT run_id, generated_at, horizon, requested_limit, candidate_count,
               actionable_count, payload_json
        FROM recommendation_runs
        ORDER BY generated_at ASC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def load_trades(conn):
    if not table_exists(conn, "simulated_recommendation_trades"):
        return []
    rows = conn.execute(
        """
        SELECT trade_date, run_id, rank, symbol, horizon, entry_price, mark_price,
               pnl, pnl_pct, outcome, confidence, council_score, exit_target,
               stop_loss, open_source, mark_source
        FROM simulated_recommendation_trades
        ORDER BY trade_date ASC, rank ASC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def summarize_runs(runs, watch_symbols):
    print("Recommendation runs")
    print(f"  Count: {len(runs)}")
    if not runs:
        return

    first = runs[0]["generated_at"]
    last = runs[-1]["generated_at"]
    print(f"  Range: {first} -> {last}")

    candidate_counts = [row["candidate_count"] or 0 for row in runs]
    actionable_counts = [row["actionable_count"] or 0 for row in runs]
    avg_candidates = sum(candidate_counts) / len(candidate_counts)
    avg_actionable = sum(actionable_counts) / len(actionable_counts)
    print(f"  Avg candidates/actionable: {avg_candidates:.1f}/{avg_actionable:.1f}")

    symbol_seen = {symbol: {"universe": 0, "pick": 0, "last_seen": None} for symbol in watch_symbols}
    top_pick_counts = Counter()

    for row in runs:
        payload = json.loads(row["payload_json"])
        generated_at = row["generated_at"]
        picks = payload.get("recommendations", []) or []
        for pick in picks:
            symbol = str(pick.get("symbol", "")).upper()
            if symbol:
                top_pick_counts[symbol] += 1
            if symbol in symbol_seen:
                symbol_seen[symbol]["pick"] += 1
                symbol_seen[symbol]["last_seen"] = generated_at

        payload_text = row["payload_json"].upper()
        for symbol in watch_symbols:
            if symbol in payload_text:
                symbol_seen[symbol]["universe"] += 1
                symbol_seen[symbol]["last_seen"] = generated_at

    print("  Most frequent picks:")
    for symbol, count in top_pick_counts.most_common(10):
        print(f"    {symbol}: {count}")

    print("  Watch-symbol coverage:")
    for symbol, stats in symbol_seen.items():
        print(
            f"    {symbol}: payload mentions={stats['universe']}, "
            f"top-picks={stats['pick']}, last_seen={stats['last_seen'] or 'never'}"
        )


def summarize_trades(trades, limit):
    print()
    print("Simulated open-entry outcomes")
    print(f"  Count: {len(trades)}")
    marked = [row for row in trades if row.get("pnl") is not None]
    if not marked:
        print("  No marked simulated trades found.")
        return

    total_pnl = sum(float(row["pnl"] or 0) for row in marked)
    avg_pnl_pct = sum(float(row["pnl_pct"] or 0) for row in marked) / len(marked)
    winners = [row for row in marked if float(row["pnl"] or 0) > 0]
    print(f"  Total P/L: ${total_pnl:.2f}")
    print(f"  Avg P/L %: {avg_pnl_pct:+.2f}%")
    print(f"  Win rate: {len(winners)}/{len(marked)} ({len(winners) / len(marked) * 100:.1f}%)")

    by_date = defaultdict(list)
    by_symbol = defaultdict(list)
    by_rank = defaultdict(list)
    by_score_bucket = defaultdict(list)
    by_conf_bucket = defaultdict(list)

    for row in marked:
        by_date[row["trade_date"]].append(row)
        by_symbol[row["symbol"]].append(row)
        by_rank[row["rank"]].append(row)
        score_bucket = int(float(row["council_score"] or 0) // 10) * 10
        conf_bucket = int(float(row["confidence"] or 0) // 10) * 10
        by_score_bucket[score_bucket].append(row)
        by_conf_bucket[conf_bucket].append(row)

    print("  Daily P/L:")
    for trade_date in sorted(by_date):
        rows = by_date[trade_date]
        pnl = sum(float(row["pnl"] or 0) for row in rows)
        win_count = len([row for row in rows if float(row["pnl"] or 0) > 0])
        print(f"    {trade_date}: ${pnl:+.2f} ({win_count}/{len(rows)} wins)")

    print("  Symbol P/L:")
    for symbol, rows in sorted(
        by_symbol.items(),
        key=lambda item: sum(float(row["pnl"] or 0) for row in item[1]),
    ):
        pnl = sum(float(row["pnl"] or 0) for row in rows)
        avg = sum(float(row["pnl_pct"] or 0) for row in rows) / len(rows)
        print(f"    {symbol}: ${pnl:+.2f}, avg {avg:+.2f}%, n={len(rows)}")

    print("  Rank buckets:")
    for rank in sorted(by_rank):
        rows = by_rank[rank]
        pnl = sum(float(row["pnl"] or 0) for row in rows)
        avg = sum(float(row["pnl_pct"] or 0) for row in rows) / len(rows)
        print(f"    rank {rank}: ${pnl:+.2f}, avg {avg:+.2f}%, n={len(rows)}")

    print("  Council-score buckets:")
    for bucket in sorted(by_score_bucket):
        rows = by_score_bucket[bucket]
        pnl = sum(float(row["pnl"] or 0) for row in rows)
        avg = sum(float(row["pnl_pct"] or 0) for row in rows) / len(rows)
        print(f"    {bucket:02d}-{bucket + 9:02d}: ${pnl:+.2f}, avg {avg:+.2f}%, n={len(rows)}")

    print("  Confidence buckets:")
    for bucket in sorted(by_conf_bucket):
        rows = by_conf_bucket[bucket]
        pnl = sum(float(row["pnl"] or 0) for row in rows)
        avg = sum(float(row["pnl_pct"] or 0) for row in rows) / len(rows)
        print(f"    {bucket:02d}-{bucket + 9:02d}: ${pnl:+.2f}, avg {avg:+.2f}%, n={len(rows)}")

    print(f"  Recent {limit} marked trades:")
    for row in marked[-limit:]:
        print(
            f"    {row['trade_date']} #{row['rank']} {row['symbol']}: "
            f"${float(row['pnl'] or 0):+.2f} ({float(row['pnl_pct'] or 0):+.2f}%), "
            f"score={float(row['council_score'] or 0):.1f}, "
            f"conf={float(row['confidence'] or 0):.1f}, outcome={row['outcome']}"
        )


def main():
    args = parse_args()
    watch_symbols = [
        symbol.strip().upper()
        for symbol in args.symbols.split(",")
        if symbol.strip()
    ]

    started = datetime.now()
    print(f"Recommendation history audit generated {started.isoformat(timespec='seconds')}")
    print(f"Database: {DB_PATH.resolve()}")
    with connect() as conn:
        runs = load_runs(conn)
        trades = load_trades(conn)
    summarize_runs(runs, watch_symbols)
    summarize_trades(trades, args.limit)


if __name__ == "__main__":
    main()
