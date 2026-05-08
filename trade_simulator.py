import json
import logging
import sqlite3
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytz

from config import Config
from data_provider import MarketDataProvider
from notifications import NotificationService


class TradeSimulationEngine:
    """Simulates buying recommendation picks near market open and marks them later."""

    def __init__(
        self,
        data_provider: Optional[MarketDataProvider] = None,
        notifications: Optional[NotificationService] = None,
    ):
        self.config = Config()
        self.data_provider = data_provider or MarketDataProvider()
        self.notifications = notifications or NotificationService()
        self.db_path = Path("trading_data.db")
        self.logger = logging.getLogger(__name__)
        self.eastern_tz = pytz.timezone("US/Eastern")
        self._init_database()

    def capture_open_trades(
        self,
        trade_date: Optional[date] = None,
        top_n: Optional[int] = None,
    ) -> Dict:
        """Create simulated entries for the latest recommendation run."""
        trade_date = trade_date or self._today_eastern()
        top_n = max(1, min(int(top_n or self.config.SIMULATION_TOP_N), 10))
        run = self._latest_recommendation_run(trade_date)
        if not run:
            return {
                "trade_date": trade_date.isoformat(),
                "captured": 0,
                "error": "No recommendation run found for today",
                "trades": [],
            }

        result = json.loads(run["payload_json"])
        picks = result.get("recommendations", [])[:top_n]
        captured = []
        errors = []

        for pick in picks:
            symbol = str(pick.get("symbol", "")).upper()
            if not symbol:
                continue

            try:
                entry_price, entry_time, source = self._market_open_entry(symbol, trade_date)
                if entry_price <= 0:
                    errors.append({"symbol": symbol, "error": "No open-entry price available"})
                    continue

                notional = float(self.config.SIMULATION_NOTIONAL_PER_PICK)
                quantity = round(notional / entry_price, 6)
                row = {
                    "trade_date": trade_date.isoformat(),
                    "run_id": run["run_id"],
                    "rank": int(pick.get("rank") or 0),
                    "symbol": symbol,
                    "horizon": pick.get("horizon", result.get("horizon", "WEEK")),
                    "recommended_price": self._safe_float(pick.get("current_price")),
                    "entry_price": entry_price,
                    "entry_time": entry_time,
                    "quantity": quantity,
                    "notional": notional,
                    "exit_target": self._safe_float(pick.get("exit_price")),
                    "stop_loss": self._safe_float(pick.get("stop_loss")),
                    "confidence": self._safe_float(pick.get("confidence")),
                    "council_score": self._safe_float(pick.get("council_score")),
                    "open_source": source,
                }
                self._insert_simulated_trade(row)
                captured.append(row)
            except Exception as exc:
                self.logger.error("Could not simulate open trade for %s: %s", symbol, exc)
                errors.append({"symbol": symbol, "error": str(exc)})

        return {
            "trade_date": trade_date.isoformat(),
            "run_id": run["run_id"],
            "captured": len(captured),
            "requested": len(picks),
            "trades": captured,
            "errors": errors,
        }

    def build_eod_summary(self, trade_date: Optional[date] = None) -> Dict:
        """Mark simulated trades to the latest available EOD/intraday price."""
        trade_date = trade_date or self._today_eastern()
        trades = self._load_simulated_trades(trade_date)
        rows = []
        total_entry_value = 0.0
        total_mark_value = 0.0

        for trade in trades:
            symbol = trade["symbol"]
            mark_price, mark_time, high, low, source = self._mark_price(symbol, trade_date)
            if mark_price <= 0:
                mark_price = float(trade["entry_price"])
                mark_time = datetime.now().isoformat()
                high = mark_price
                low = mark_price
                source = "entry_fallback"

            quantity = float(trade["quantity"])
            entry_price = float(trade["entry_price"])
            entry_value = entry_price * quantity
            mark_value = mark_price * quantity
            pnl = mark_value - entry_value
            pnl_pct = ((mark_price / entry_price) - 1) * 100 if entry_price else 0.0
            exit_target = float(trade["exit_target"] or 0)
            stop_loss = float(trade["stop_loss"] or 0)
            outcome = self._outcome(entry_price, mark_price, high, low, exit_target, stop_loss)
            target_progress = self._target_progress(entry_price, mark_price, exit_target)

            row = {
                **trade,
                "mark_price": mark_price,
                "mark_time": mark_time,
                "mark_source": source,
                "day_high": high,
                "day_low": low,
                "entry_value": entry_value,
                "mark_value": mark_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "target_progress_pct": target_progress,
                "outcome": outcome,
            }
            rows.append(row)
            total_entry_value += entry_value
            total_mark_value += mark_value
            self._update_simulated_trade(row)

        total_pnl = total_mark_value - total_entry_value
        total_pnl_pct = (total_pnl / total_entry_value * 100) if total_entry_value else 0.0
        winners = len([row for row in rows if row["pnl"] > 0])

        return {
            "trade_date": trade_date.isoformat(),
            "trade_count": len(rows),
            "winners": winners,
            "losers": len(rows) - winners,
            "total_entry_value": total_entry_value,
            "total_mark_value": total_mark_value,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "trades": sorted(rows, key=lambda row: row.get("rank") or 999),
            "generated_at": datetime.now().isoformat(),
        }

    def send_eod_summary_whatsapp(
        self,
        trade_date: Optional[date] = None,
        label: str = "EOD",
    ) -> Dict:
        summary = self.build_eod_summary(trade_date=trade_date)
        body = format_simulation_summary_message(summary, label=label)
        sent = self.notifications.send_openclaw_whatsapp(body)
        summary["delivery"] = {
            "channel": "openclaw_whatsapp",
            "sent": sent,
            "target": self.notifications.get_openclaw_target(),
            "message": body,
            "error": self.notifications.get_last_error(),
        }
        return summary

    def send_open_capture_whatsapp(self, capture: Dict) -> Dict:
        body = format_open_capture_message(capture)
        sent = self.notifications.send_openclaw_whatsapp(body)
        capture["delivery"] = {
            "channel": "openclaw_whatsapp",
            "sent": sent,
            "target": self.notifications.get_openclaw_target(),
            "message": body,
            "error": self.notifications.get_last_error(),
        }
        return capture

    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS simulated_recommendation_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    horizon TEXT,
                    recommended_price REAL,
                    entry_price REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    notional REAL NOT NULL,
                    exit_target REAL,
                    stop_loss REAL,
                    confidence REAL,
                    council_score REAL,
                    open_source TEXT,
                    mark_price REAL,
                    mark_time TEXT,
                    mark_source TEXT,
                    pnl REAL,
                    pnl_pct REAL,
                    outcome TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(trade_date, run_id, symbol)
                )
                """
            )

    def _latest_recommendation_run(self, trade_date: date) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT run_id, generated_at, payload_json
                FROM recommendation_runs
                WHERE substr(generated_at, 1, 10) = ?
                ORDER BY generated_at DESC
                LIMIT 1
                """,
                (trade_date.isoformat(),),
            ).fetchone()
            return dict(row) if row else None

    def _insert_simulated_trade(self, row: Dict):
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO simulated_recommendation_trades
                (trade_date, run_id, rank, symbol, horizon, recommended_price,
                 entry_price, entry_time, quantity, notional, exit_target,
                 stop_loss, confidence, council_score, open_source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["trade_date"],
                    row["run_id"],
                    row["rank"],
                    row["symbol"],
                    row["horizon"],
                    row["recommended_price"],
                    row["entry_price"],
                    row["entry_time"],
                    row["quantity"],
                    row["notional"],
                    row["exit_target"],
                    row["stop_loss"],
                    row["confidence"],
                    row["council_score"],
                    row["open_source"],
                    now,
                    now,
                ),
            )

    def _load_simulated_trades(self, trade_date: date) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT *
                FROM simulated_recommendation_trades
                WHERE trade_date = ?
                  AND run_id = (
                    SELECT t.run_id
                    FROM simulated_recommendation_trades t
                    JOIN recommendation_runs r ON r.run_id = t.run_id
                    WHERE t.trade_date = ?
                      AND substr(r.generated_at, 1, 10) = ?
                    ORDER BY t.created_at DESC
                    LIMIT 1
                  )
                ORDER BY rank ASC
                """,
                (trade_date.isoformat(), trade_date.isoformat(), trade_date.isoformat()),
            ).fetchall()
            return [dict(row) for row in rows]

    def _update_simulated_trade(self, row: Dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE simulated_recommendation_trades
                SET mark_price = ?, mark_time = ?, mark_source = ?,
                    pnl = ?, pnl_pct = ?, outcome = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    row["mark_price"],
                    row["mark_time"],
                    row["mark_source"],
                    row["pnl"],
                    row["pnl_pct"],
                    row["outcome"],
                    datetime.now().isoformat(),
                    row["id"],
                ),
            )

    def _market_open_entry(self, symbol: str, trade_date: date) -> Tuple[float, str, str]:
        intraday = self.data_provider.get_intraday_data(symbol, period="5d", interval="1m")
        day = self._intraday_for_date(intraday, trade_date)
        if not day.empty:
            market_open = dt_time(9, 30)
            open_rows = day[day["_eastern_time"] >= market_open]
            if not open_rows.empty:
                first = open_rows.iloc[0]
                return (
                    self._safe_float(first["Open"]),
                    first["_eastern_dt"].isoformat(),
                    "intraday_open",
                )

        daily = self.data_provider.get_daily_data(symbol, period="10d")
        day_bar = self._daily_bar_for_date(daily, trade_date)
        if day_bar is not None:
            return (
                self._safe_float(day_bar["Open"]),
                datetime.combine(trade_date, dt_time(9, 30)).isoformat(),
                "daily_open",
            )

        quote = self.data_provider.get_real_time_quote(symbol)
        return (
            self._safe_float(quote.get("current_price")),
            datetime.now().isoformat(),
            quote.get("source", "quote_fallback"),
        )

    def _mark_price(self, symbol: str, trade_date: date) -> Tuple[float, str, float, float, str]:
        intraday = self.data_provider.get_intraday_data(symbol, period="5d", interval="1m")
        day = self._intraday_for_date(intraday, trade_date)
        if not day.empty:
            last = day.iloc[-1]
            return (
                self._safe_float(last["Close"]),
                last["_eastern_dt"].isoformat(),
                self._safe_float(day["High"].max()),
                self._safe_float(day["Low"].min()),
                "intraday_latest",
            )

        daily = self.data_provider.get_daily_data(symbol, period="10d")
        day_bar = self._daily_bar_for_date(daily, trade_date)
        if day_bar is not None:
            return (
                self._safe_float(day_bar["Close"]),
                datetime.combine(trade_date, dt_time(16, 0)).isoformat(),
                self._safe_float(day_bar["High"]),
                self._safe_float(day_bar["Low"]),
                "daily_close",
            )

        quote = self.data_provider.get_real_time_quote(symbol)
        price = self._safe_float(quote.get("current_price"))
        return price, datetime.now().isoformat(), price, price, quote.get("source", "quote_fallback")

    def _intraday_for_date(self, data: pd.DataFrame, trade_date: date) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame()

        frame = data.copy()
        idx = pd.DatetimeIndex(frame.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        eastern = idx.tz_convert(self.eastern_tz)
        frame["_eastern_dt"] = eastern
        frame["_eastern_time"] = [stamp.time() for stamp in eastern]
        mask = [stamp.date() == trade_date for stamp in eastern]
        return frame.loc[mask].sort_values("_eastern_dt")

    def _daily_bar_for_date(self, data: pd.DataFrame, trade_date: date):
        if data.empty:
            return None
        for idx, row in data.sort_index(ascending=False).iterrows():
            if pd.Timestamp(idx).date() == trade_date:
                return row
        return None

    def _outcome(
        self,
        entry_price: float,
        mark_price: float,
        high: float,
        low: float,
        exit_target: float,
        stop_loss: float,
    ) -> str:
        hit_target = exit_target > 0 and high >= exit_target
        hit_stop = stop_loss > 0 and low <= stop_loss
        if hit_target and hit_stop:
            return "TARGET_AND_STOP_TOUCHED"
        if hit_target:
            return "TARGET_TOUCHED"
        if hit_stop:
            return "STOP_TOUCHED"
        if mark_price > entry_price:
            return "UP"
        if mark_price < entry_price:
            return "DOWN"
        return "FLAT"

    def _target_progress(self, entry_price: float, mark_price: float, exit_target: float) -> float:
        if entry_price <= 0 or exit_target <= entry_price:
            return 0.0
        return ((mark_price - entry_price) / (exit_target - entry_price)) * 100

    def _today_eastern(self) -> date:
        return datetime.now(self.eastern_tz).date()

    def _safe_float(self, value) -> float:
        try:
            if value is None:
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0


def format_simulation_summary_message(summary: Dict, label: str = "EOD") -> str:
    trades = summary.get("trades", [])
    trade_date = summary.get("trade_date", "")
    total_pnl = summary.get("total_pnl", 0.0)
    total_pnl_pct = summary.get("total_pnl_pct", 0.0)
    sign = "+" if total_pnl >= 0 else ""
    label = (label or "EOD").upper()
    lines = [
        f"Simulated {label} P/L snapshot - {trade_date}",
        (
            f"Total: {sign}${total_pnl:.2f} ({total_pnl_pct:+.2f}%) "
            f"on ${summary.get('total_entry_value', 0):.2f}"
        ),
        f"Win/loss: {summary.get('winners', 0)}/{summary.get('losers', 0)}",
        "Mode: simulated, no real orders placed.",
        "",
    ]

    if not trades:
        lines.append("No simulated trades were captured for this date.")
        return "\n".join(lines).strip()

    for trade in trades:
        pnl = trade.get("pnl", 0.0)
        pnl_sign = "+" if pnl >= 0 else ""
        lines.extend(
            [
                (
                    f"{trade.get('rank')}. {trade.get('symbol')} "
                    f"{pnl_sign}${pnl:.2f} ({trade.get('pnl_pct', 0):+.2f}%)"
                ),
                (
                    f"Bought: ${trade.get('entry_value', 0):.2f} "
                    f"at ${trade.get('entry_price', 0):.2f} "
                    f"x {trade.get('quantity', 0):.4f}"
                ),
                (
                    f"Marked value: ${trade.get('mark_value', 0):.2f} "
                    f"at ${trade.get('mark_price', 0):.2f}"
                ),
                f"P/L: {pnl_sign}${pnl:.2f}",
                (
                    f"Target: ${trade.get('exit_target', 0):.2f} | "
                    f"Stop: ${trade.get('stop_loss', 0):.2f} | "
                    f"Progress: {trade.get('target_progress_pct', 0):.1f}%"
                ),
                f"Status: {trade.get('outcome', 'UNKNOWN')}",
                "",
            ]
        )

    lines.append("This is an open-entry simulation snapshot for research only.")
    return "\n".join(lines).strip()


def format_open_capture_message(capture: Dict) -> str:
    trades = capture.get("trades", [])
    trade_date = capture.get("trade_date", "")
    lines = [
        f"Simulated open entries captured - {trade_date}",
        (
            f"Captured {capture.get('captured', 0)}/{capture.get('requested', 0)} "
            f"recommended picks."
        ),
        "Mode: simulated, no real orders placed.",
        "",
    ]

    if not trades:
        lines.append(capture.get("error") or "No simulated entries were captured.")
        return "\n".join(lines).strip()

    for trade in sorted(trades, key=lambda row: row.get("rank") or 999):
        entry_value = float(trade.get("entry_price", 0) or 0) * float(trade.get("quantity", 0) or 0)
        lines.extend(
            [
                (
                    f"{trade.get('rank')}. {trade.get('symbol')} "
                    f"entry ${trade.get('entry_price', 0):.2f}"
                ),
                (
                    f"Simulated buy: ${entry_value:.2f} "
                    f"at ${trade.get('entry_price', 0):.2f} "
                    f"x {trade.get('quantity', 0):.4f}"
                ),
                (
                    f"Target: ${trade.get('exit_target', 0):.2f} | "
                    f"Stop: ${trade.get('stop_loss', 0):.2f}"
                ),
                "",
            ]
        )

    lines.append("EOD simulated P/L summary is scheduled for later today.")
    return "\n".join(lines).strip()
