import json
import logging
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from market_calendar import MarketCalendar
from trade_simulator import TradeSimulationEngine, format_period_summary_message


def pick(symbol, rank=1):
    return {
        "symbol": symbol,
        "rank": rank,
        "horizon": "WEEK",
        "action": "BUY",
        "actionable": True,
        "current_price": 100,
        "buy_zone": 100,
        "exit_price": 110,
        "stop_loss": 95,
        "confidence": 70,
        "council_score": 75,
    }


class ExtendedAlertTests(unittest.TestCase):
    def setUp(self):
        # SQLite file handles can linger briefly on Windows after a connection
        # context commits, so test cleanup should not turn that into a failure.
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "test.db"
        self.engine = TradeSimulationEngine.__new__(TradeSimulationEngine)
        self.engine.db_path = self.db_path
        self.engine.logger = logging.getLogger("test")
        self.engine.config = SimpleNamespace(
            ENTRY_ALERT_RECOMMENDATION_LOOKBACK_DAYS=45,
            ENTRY_ALERT_INCLUDE_OVERFLOW_CANDIDATES=True,
            ENTRY_ALERT_OVERFLOW_LIMIT=20,
            PROFIT_TARGET_WEEKLY=500,
            PROFIT_TARGET_MONTHLY=2000,
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE recommendation_runs (
                    run_id TEXT PRIMARY KEY, generated_at TEXT, horizon TEXT,
                    payload_json TEXT
                )"""
            )
            conn.execute(
                """CREATE TABLE simulated_recommendation_trades (
                    trade_date TEXT, symbol TEXT, entry_price REAL, quantity REAL,
                    notional REAL, mark_price REAL, pnl REAL, pnl_pct REAL,
                    outcome TEXT
                )"""
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _save_run(self, run_id, generated_at, recommendations, candidates=None):
        payload = {
            "horizon": "WEEK",
            "recommendations": recommendations,
            "candidate_snapshot": candidates or [],
        }
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO recommendation_runs VALUES (?, ?, ?, ?)",
                (run_id, generated_at, "WEEK", json.dumps(payload)),
            )

    def test_alert_pool_includes_prior_recommendations_and_today_overflow(self):
        self._save_run("old", "2026-09-10T08:00:00", [pick("OLD")])
        self._save_run(
            "today",
            "2026-09-17T08:00:00",
            [pick("TOP")],
            [pick("TOP"), pick("EXTRA", rank=6)],
        )

        candidates = self.engine._entry_alert_candidates(date(2026, 9, 17), top_n=5)
        by_symbol = {item[2]["symbol"]: item[3] for item in candidates}

        self.assertEqual(by_symbol["TOP"], "today_top")
        self.assertEqual(by_symbol["OLD"], "historical_recommendation")
        self.assertEqual(by_symbol["EXTRA"], "today_overflow")
        self.assertEqual(len(by_symbol), 3)

    def test_period_summary_aggregates_each_simulated_entry_once(self):
        rows = [
            ("2026-09-14", "AAA", 100, 10, 1000, 102, 20, 2, "GAIN"),
            ("2026-09-15", "BBB", 50, 20, 1000, 49, -20, -2, "LOSS"),
            ("2026-09-17", "CCC", 25, 40, 1000, 26, 40, 4, "GAIN"),
            ("2026-09-10", "OLD", 10, 100, 1000, 11, 100, 10, "GAIN"),
        ]
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT INTO simulated_recommendation_trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

        summary = self.engine.build_period_summary("WEEK", as_of=date(2026, 9, 17))

        self.assertEqual(summary["start_date"], "2026-09-14")
        self.assertEqual(summary["trade_count"], 3)
        self.assertEqual(summary["trading_days"], 3)
        self.assertEqual(summary["total_pnl"], 40)
        self.assertAlmostEqual(summary["total_pnl_pct"], 40 / 3000 * 100)
        self.assertIn("End-of-week", format_period_summary_message(summary))


class MarketCalendarBoundaryTests(unittest.TestCase):
    def test_thursday_before_closed_friday_is_week_end(self):
        calendar = MarketCalendar()
        open_days = {date(2026, 7, 2), date(2026, 7, 6)}
        calendar.is_trading_day = lambda day=None: day in open_days
        self.assertTrue(calendar.is_last_trading_day_of_week(date(2026, 7, 2)))

    def test_last_session_before_next_month_is_month_end(self):
        calendar = MarketCalendar()
        open_days = {date(2026, 9, 30), date(2026, 10, 1)}
        calendar.is_trading_day = lambda day=None: day in open_days
        self.assertTrue(calendar.is_last_trading_day_of_month(date(2026, 9, 30)))


if __name__ == "__main__":
    unittest.main()
