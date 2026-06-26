import json
import logging
import sqlite3
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytz

from config import Config
from council_memory import CouncilLearningMemory
from data_provider import MarketDataProvider
from notifications import NotificationService


class TradeSimulationEngine:
    """Simulates buying recommendation picks near market open and marks them later."""

    NON_STOCK_SYMBOLS = {
        "SPY", "QQQ", "IWM", "DIA", "EEM", "XLF", "XLK", "XLE", "XLY", "XLP",
        "XLV", "XLI", "XLB", "XLU", "XLRE", "GLD", "SLV", "USO", "UNG", "TLT",
        "HYG", "LQD", "VXX", "UVXY", "SQQQ", "TQQQ", "SPXL", "SPXS", "SOXL",
        "SOXS", "TECL", "TECS", "FNGU", "FNGD", "ARKK", "ARKW", "ARKG",
    }

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
        self.learning_memory = CouncilLearningMemory(self.db_path, self.config)
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

    def build_eod_summary(
        self,
        trade_date: Optional[date] = None,
        backfill_missing: bool = False,
        top_n: Optional[int] = None,
    ) -> Dict:
        """Mark simulated trades to the latest available EOD/intraday price."""
        trade_date = trade_date or self._today_eastern()
        trades = self._load_simulated_trades(trade_date)
        backfill_result = None
        if backfill_missing and not trades:
            self.logger.warning(
                "No simulated trades found for %s; attempting open-entry backfill",
                trade_date.isoformat(),
            )
            backfill_result = self.capture_open_trades(trade_date=trade_date, top_n=top_n)
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

        summary = {
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
        if backfill_result is not None:
            summary["backfill"] = backfill_result
        return summary

    def send_eod_summary_whatsapp(
        self,
        trade_date: Optional[date] = None,
        label: str = "EOD",
    ) -> Dict:
        summary = self.build_eod_summary(
            trade_date=trade_date,
            backfill_missing=self.config.SIMULATION_BACKFILL_ON_SUMMARY,
            top_n=self.config.SIMULATION_TOP_N,
        )
        if label.upper() == "EOD" and self.config.COUNCIL_RAG_ENABLED:
            summary["learning"] = self.learning_memory.learn_from_summary(summary)
            summary["missed_mover_learning"] = self._learn_from_missed_movers(summary)
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

    def _learn_from_missed_movers(self, summary: Dict) -> Dict:
        trade_date = summary.get("trade_date") or self._today_eastern().isoformat()
        recommended = {
            str(row.get("symbol") or "").strip().upper()
            for row in summary.get("trades", []) or []
            if row.get("symbol")
        }

        run_id = None
        for row in summary.get("trades", []) or []:
            if row.get("run_id"):
                run_id = row["run_id"]
                break
        if not run_id:
            latest = self._latest_recommendation_run(datetime.fromisoformat(trade_date).date())
            run_id = latest.get("run_id") if latest else None

        if run_id:
            try:
                payload = self._recommendation_payload(run_id)
                for pick in payload.get("recommendations", []) or []:
                    symbol = str(pick.get("symbol") or "").strip().upper()
                    if symbol:
                        recommended.add(symbol)
            except Exception:
                pass

        threshold = float(getattr(self.config, "COUNCIL_RAG_MISSED_MOVER_MIN_CHANGE_PCT", 3.0))
        movers = []
        try:
            for mover in self.data_provider.get_market_movers(120):
                symbol = str(mover.get("symbol") or "").strip().upper()
                change_pct = self._safe_float(mover.get("change_percent"))
                if (
                    not symbol
                    or symbol in recommended
                    or not self._is_stock_candidate(symbol)
                    or change_pct < threshold
                ):
                    continue
                movers.append(
                    {
                        "symbol": symbol,
                        "change_percent": change_pct,
                        "current_price": self._safe_float(mover.get("current_price")),
                        "source": mover.get("source", "market_movers"),
                    }
                )
        except Exception as exc:
            self.logger.warning("Could not learn missed movers: %s", exc)
            return {"enabled": True, "lesson_date": trade_date, "lessons_written": 0, "error": str(exc)}

        return self.learning_memory.learn_from_missed_movers(trade_date, movers[:20])

    def _recommendation_payload(self, run_id: str) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT payload_json FROM recommendation_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            return json.loads(row["payload_json"]) if row else {}

    def _is_stock_candidate(self, symbol: str) -> bool:
        symbol = str(symbol or "").strip().upper()
        if not symbol or symbol in self.NON_STOCK_SYMBOLS:
            return False
        if symbol.endswith((".WS", ".W", ".U", ".R")):
            return False
        return not any(fragment in symbol for fragment in ("2X", "3X", "ULTRA", "BEAR", "BULL"))

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
                SELECT t.*
                FROM simulated_recommendation_trades t
                LEFT JOIN recommendation_runs r ON r.run_id = t.run_id
                WHERE t.trade_date = ?
                ORDER BY COALESCE(r.generated_at, t.created_at) ASC, t.rank ASC
                """,
                (trade_date.isoformat(),),
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


def _format_signed_dollars(value: float) -> str:
    amount = float(value or 0.0)
    sign = "+" if amount >= 0 else "-"
    return f"{sign}${abs(amount):.2f}"


def format_simulation_summary_message(summary: Dict, label: str = "EOD") -> str:
    trades = summary.get("trades", [])
    trade_date = summary.get("trade_date", "")
    total_pnl = summary.get("total_pnl", 0.0)
    total_pnl_pct = summary.get("total_pnl_pct", 0.0)
    label = (label or "EOD").upper()
    stock_pnl_summary = " | ".join(
        (
            f"{trade.get('symbol')} "
            f"{_format_signed_dollars(trade.get('pnl', 0.0))} "
            f"({trade.get('pnl_pct', 0):+.2f}%)"
        )
        for trade in trades
    )
    headline = f"Simulated {label} P/L - {trade_date}"
    if trades:
        headline = (
            f"{headline}: Total {_format_signed_dollars(total_pnl)} "
            f"({total_pnl_pct:+.2f}%)"
        )
        if stock_pnl_summary:
            headline = f"{headline} | {stock_pnl_summary}"
    else:
        headline = f"{headline}: no captured trades"

    lines = [
        headline,
        (
            f"Total: {_format_signed_dollars(total_pnl)} ({total_pnl_pct:+.2f}%) "
            f"on ${summary.get('total_entry_value', 0):.2f}"
            if trades
            else "No captured trades"
        ),
        f"Win/loss: {summary.get('winners', 0)}/{summary.get('losers', 0)}",
        "Mode: simulated, no real orders placed.",
        "",
    ]

    learning = summary.get("learning") or {}
    if learning.get("enabled") and learning.get("lessons_written", 0) > 0:
        lines.append(
            f"Learning memory: saved {learning.get('lessons_written', 0)} RAG lessons."
        )

    missed_learning = summary.get("missed_mover_learning") or {}
    if missed_learning.get("enabled") and missed_learning.get("lessons_written", 0) > 0:
        lines.append(
            f"Missed-mover memory: saved {missed_learning.get('lessons_written', 0)} opportunity lessons."
        )

    if not trades:
        backfill = summary.get("backfill") or {}
        if backfill.get("error"):
            lines.append(f"Open-entry backfill failed: {backfill.get('error')}.")
        elif backfill:
            captured = backfill.get("captured", 0)
            requested = backfill.get("requested", 0)
            errors = backfill.get("errors") or []
            lines.append(f"Open-entry backfill captured {captured}/{requested} picks.")
            if errors:
                lines.append("Backfill errors:")
                for item in errors[:5]:
                    symbol = str(item.get("symbol", "unknown"))
                    error = str(item.get("error", "unknown error"))
                    lines.append(f"{symbol}: {error}")
        lines.append("No simulated trades were captured for this date.")
        return "\n".join(lines).strip()

    lines.append("Per-stock P/L:")
    for trade in trades:
        pnl = trade.get("pnl", 0.0)
        lines.append(
            (
                f"{trade.get('rank')}. {trade.get('symbol')}: "
                f"{_format_signed_dollars(pnl)} ({trade.get('pnl_pct', 0):+.2f}%) | "
                f"entry ${trade.get('entry_price', 0):.2f} -> "
                f"mark ${trade.get('mark_price', 0):.2f} | "
                f"target ${trade.get('exit_target', 0):.2f} "
                f"stop ${trade.get('stop_loss', 0):.2f} | "
                f"{trade.get('outcome', 'UNKNOWN')}"
            )
        )

    lines.append("This is an open-entry simulation snapshot for research only.")
    return "\n".join(lines).strip()


def format_open_capture_message(capture: Dict) -> str:
    trades = capture.get("trades", [])
    trade_date = capture.get("trade_date", "")
    entry_summary = " | ".join(
        f"{trade.get('symbol')} ${trade.get('entry_price', 0):.2f}"
        for trade in sorted(trades, key=lambda row: row.get("rank") or 999)
    )
    lines = [
        (
            f"Simulated open entries captured - {trade_date}: {entry_summary}"
            if entry_summary
            else f"Simulated open entries captured - {trade_date}: none"
        ),
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
        lines.append(
            (
                f"{trade.get('rank')}. {trade.get('symbol')}: "
                f"entry ${trade.get('entry_price', 0):.2f} | "
                f"sim ${entry_value:.2f} x {trade.get('quantity', 0):.4f} | "
                f"target ${trade.get('exit_target', 0):.2f} "
                f"stop ${trade.get('stop_loss', 0):.2f}"
            )
        )

    lines.append("MIDDAY and EOD simulated P/L summaries are scheduled for later today.")
    return "\n".join(lines).strip()
