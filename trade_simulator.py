import json
import logging
import sqlite3
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytz

from config import Config
from council_memory import CouncilLearningMemory
from data_provider import MarketDataProvider
from notifications import NotificationService


class TradeSimulationEngine:
    """Simulates recommendation entries and marks them later for feedback."""

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

    def capture_entry_alerts(
        self,
        trade_date: Optional[date] = None,
        top_n: Optional[int] = None,
        max_alerts: Optional[int] = None,
        historical: bool = False,
    ) -> Dict:
        """Capture dip entries from current, historical, and overflow council picks."""
        trade_date = trade_date or self._today_eastern()
        top_n = max(1, min(int(top_n or self.config.SIMULATION_TOP_N), 10))
        max_alerts = max(
            1,
            int(max_alerts or self.config.ENTRY_ALERT_MAX_ALERTS_PER_SCAN),
        )
        candidates = self._entry_alert_candidates(trade_date, top_n)
        if not candidates:
            return {
                "trade_date": trade_date.isoformat(),
                "mode": "intraday_dip_alert",
                "captured": 0,
                "error": "No current or historical recommendation candidates found",
                "alerts": [],
                "evaluated": [],
            }

        existing_symbols = self._existing_simulated_symbols(trade_date)
        evaluated = []
        skipped_existing = []
        qualified = []
        captured = []
        errors = []

        for run, result, pick, source in candidates:
            symbol = str(pick.get("symbol", "")).upper()
            if not symbol:
                continue

            if symbol in existing_symbols:
                skipped_existing.append(symbol)
                continue

            try:
                alert = self._best_entry_alert_candidate(
                    pick,
                    trade_date=trade_date,
                    historical=historical,
                )
                alert["rank"] = int(pick.get("rank") or 0)
                alert["candidate_source"] = source
                alert["recommended_at"] = run.get("generated_at")
                evaluated.append(alert)
                if alert.get("qualified"):
                    qualified.append(
                        (self._safe_float(alert.get("entry_score")), run, result, pick, alert)
                    )
            except Exception as exc:
                self.logger.error("Could not evaluate entry alert for %s: %s", symbol, exc)
                errors.append({"symbol": symbol, "error": str(exc)})

        qualified.sort(key=lambda item: item[0], reverse=True)
        for _, run, result, pick, alert in qualified[:max_alerts]:
            try:
                row = self._entry_alert_trade_row(
                    run=run,
                    result=result,
                    pick=pick,
                    alert=alert,
                    trade_date=trade_date,
                )
                self._insert_simulated_trade(row)
                captured.append(row)
            except Exception as exc:
                symbol = str(pick.get("symbol", "")).upper()
                self.logger.error("Could not capture entry alert for %s: %s", symbol, exc)
                errors.append({"symbol": symbol, "error": str(exc)})

        start_dt, end_dt = self._entry_alert_bounds(trade_date)
        return {
            "trade_date": trade_date.isoformat(),
            "mode": "intraday_dip_alert",
            "historical": historical,
            "captured": len(captured),
            "requested": len(candidates),
            "alerts": captured,
            "evaluated": sorted(
                evaluated,
                key=lambda item: self._safe_float(item.get("entry_score")),
                reverse=True,
            ),
            "skipped_existing": skipped_existing,
            "errors": errors,
            "window": {
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
                "skip_open_minutes": self.config.ENTRY_ALERT_SKIP_OPEN_MINUTES,
                "skip_close_minutes": self.config.ENTRY_ALERT_SKIP_CLOSE_MINUTES,
            },
            "target_weekly": self.config.PROFIT_TARGET_WEEKLY,
            "target_monthly": self.config.PROFIT_TARGET_MONTHLY,
            "candidate_sources": self._candidate_source_counts(candidates),
        }

    def _entry_alert_candidates(self, trade_date: date, top_n: int) -> List[Tuple[Dict, Dict, Dict, str]]:
        """Return one best/latest thesis per symbol, with today's top picks first."""
        lookback_days = max(
            1,
            int(getattr(self.config, "ENTRY_ALERT_RECOMMENDATION_LOOKBACK_DAYS", 45)),
        )
        earliest = trade_date - timedelta(days=lookback_days)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT run_id, generated_at, horizon, payload_json
                FROM recommendation_runs
                WHERE substr(generated_at, 1, 10) BETWEEN ? AND ?
                ORDER BY generated_at DESC
                """,
                (earliest.isoformat(), trade_date.isoformat()),
            ).fetchall()

        candidates = []
        seen = set()
        today = trade_date.isoformat()

        def add(run: Dict, result: Dict, pick: Dict, source: str):
            symbol = str(pick.get("symbol") or "").strip().upper()
            if not symbol or symbol in seen or not self._is_stock_candidate(symbol):
                return
            if not self._safe_float(pick.get("exit_price")) or not self._safe_float(pick.get("stop_loss")):
                return
            normalized = dict(pick)
            normalized["symbol"] = symbol
            normalized.setdefault("horizon", result.get("horizon", run.get("horizon", "WEEK")))
            seen.add(symbol)
            candidates.append((run, result, normalized, source))

        decoded = []
        for row in rows:
            run = dict(row)
            try:
                decoded.append((run, json.loads(run["payload_json"])))
            except (TypeError, ValueError) as exc:
                self.logger.warning("Could not decode recommendation run %s: %s", run.get("run_id"), exc)

        # Today's published recommendations retain first priority.
        for run, result in decoded:
            if str(run.get("generated_at", ""))[:10] != today:
                continue
            for pick in (result.get("recommendations") or [])[:top_n]:
                add(run, result, pick, "today_top")

        # Any still-active prior recommendation can trigger on a later dip.
        for run, result in decoded:
            source = "today_recommendation" if str(run.get("generated_at", ""))[:10] == today else "historical_recommendation"
            for pick in result.get("recommendations") or []:
                add(run, result, pick, source)

        # Actionable BUY candidates below the displayed top-N are eligible too.
        if getattr(self.config, "ENTRY_ALERT_INCLUDE_OVERFLOW_CANDIDATES", True):
            overflow_limit = max(0, int(getattr(self.config, "ENTRY_ALERT_OVERFLOW_LIMIT", 20)))
            added = 0
            for run, result in decoded:
                if str(run.get("generated_at", ""))[:10] != today:
                    continue
                for pick in result.get("candidate_snapshot") or []:
                    if added >= overflow_limit:
                        break
                    if not pick.get("actionable") or str(pick.get("action", "")).upper() != "BUY":
                        continue
                    before = len(candidates)
                    add(run, result, pick, "today_overflow")
                    added += len(candidates) - before

        return candidates

    def _candidate_source_counts(self, candidates: List[Tuple[Dict, Dict, Dict, str]]) -> Dict[str, int]:
        counts = {}
        for _, _, _, source in candidates:
            counts[source] = counts.get(source, 0) + 1
        return counts

    def send_entry_alerts_whatsapp(self, scan: Dict) -> Dict:
        body = format_entry_alert_message(scan)
        sent = self.notifications.send_openclaw_whatsapp(body)
        scan["delivery"] = {
            "channel": "openclaw_whatsapp",
            "sent": sent,
            "target": self.notifications.get_openclaw_target(),
            "message": body,
            "error": self.notifications.get_last_error(),
        }
        return scan

    def entry_alert_window_status(self, now: Optional[datetime] = None) -> Dict:
        now = now or datetime.now(self.eastern_tz)
        if now.tzinfo is None:
            now = self.eastern_tz.localize(now)
        else:
            now = now.astimezone(self.eastern_tz)

        start_dt, end_dt = self._entry_alert_bounds(now.date())
        is_open = start_dt <= now <= end_dt
        return {
            "is_open": is_open,
            "now": now.isoformat(),
            "window": {
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
                "skip_open_minutes": self.config.ENTRY_ALERT_SKIP_OPEN_MINUTES,
                "skip_close_minutes": self.config.ENTRY_ALERT_SKIP_CLOSE_MINUTES,
            },
            "reason": (
                "inside intraday dip-entry window"
                if is_open
                else "outside intraday dip-entry window"
            ),
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
                "No simulated trades found for %s; attempting entry-alert backfill",
                trade_date.isoformat(),
            )
            backfill_result = self.capture_entry_alerts(
                trade_date=trade_date,
                top_n=top_n,
                historical=True,
            )
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

    def build_period_summary(
        self,
        period: str,
        as_of: Optional[date] = None,
    ) -> Dict:
        """Aggregate completed daily recommendation simulations for a calendar period."""
        as_of = as_of or self._today_eastern()
        period = str(period or "WEEK").upper()
        if period == "WEEK":
            start_date = as_of - timedelta(days=as_of.weekday())
        elif period == "MONTH":
            start_date = as_of.replace(day=1)
        else:
            raise ValueError("period must be WEEK or MONTH")

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT trade_date, symbol, entry_price, quantity, notional,
                       mark_price, pnl, pnl_pct, outcome
                FROM simulated_recommendation_trades
                WHERE trade_date BETWEEN ? AND ? AND pnl IS NOT NULL
                ORDER BY trade_date ASC, symbol ASC
                """,
                (start_date.isoformat(), as_of.isoformat()),
            ).fetchall()

        trades = [dict(row) for row in rows]
        total_entry_value = sum(
            self._safe_float(row.get("entry_price")) * self._safe_float(row.get("quantity"))
            for row in trades
        )
        total_pnl = sum(self._safe_float(row.get("pnl")) for row in trades)
        total_pnl_pct = (total_pnl / total_entry_value * 100) if total_entry_value else 0.0
        winners = sum(1 for row in trades if self._safe_float(row.get("pnl")) > 0)
        daily_pnl = {}
        for row in trades:
            day = row.get("trade_date")
            daily_pnl[day] = daily_pnl.get(day, 0.0) + self._safe_float(row.get("pnl"))

        return {
            "period": period,
            "start_date": start_date.isoformat(),
            "end_date": as_of.isoformat(),
            "trade_count": len(trades),
            "trading_days": len(daily_pnl),
            "winners": winners,
            "losers": len(trades) - winners,
            "total_entry_value": total_entry_value,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "target": (
                self.config.PROFIT_TARGET_WEEKLY
                if period == "WEEK"
                else self.config.PROFIT_TARGET_MONTHLY
            ),
            "daily_pnl": [
                {"date": day, "pnl": pnl}
                for day, pnl in sorted(daily_pnl.items())
            ],
            "trades": trades,
            "generated_at": datetime.now().isoformat(),
        }

    def send_period_summary_whatsapp(
        self,
        period: str,
        as_of: Optional[date] = None,
    ) -> Dict:
        summary = self.build_period_summary(period, as_of=as_of)
        body = format_period_summary_message(summary)
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

    def _entry_alert_bounds(self, trade_date: date) -> Tuple[datetime, datetime]:
        market_open = datetime.strptime(self.config.MARKET_OPEN, "%H:%M").time()
        market_close = datetime.strptime(self.config.MARKET_CLOSE, "%H:%M").time()
        start_naive = datetime.combine(trade_date, market_open) + timedelta(
            minutes=max(0, self.config.ENTRY_ALERT_SKIP_OPEN_MINUTES)
        )
        end_naive = datetime.combine(trade_date, market_close) - timedelta(
            minutes=max(0, self.config.ENTRY_ALERT_SKIP_CLOSE_MINUTES)
        )
        return self.eastern_tz.localize(start_naive), self.eastern_tz.localize(end_naive)

    def _regular_session_frame(self, day: pd.DataFrame) -> pd.DataFrame:
        if day.empty:
            return pd.DataFrame()
        market_open = datetime.strptime(self.config.MARKET_OPEN, "%H:%M").time()
        market_close = datetime.strptime(self.config.MARKET_CLOSE, "%H:%M").time()
        return day[
            (day["_eastern_time"] >= market_open)
            & (day["_eastern_time"] <= market_close)
        ].copy()

    def _entry_window_positions(self, frame: pd.DataFrame, trade_date: date) -> List[int]:
        if frame.empty:
            return []
        start_dt, end_dt = self._entry_alert_bounds(trade_date)
        start_time = start_dt.time()
        end_time = end_dt.time()
        return [
            idx
            for idx, stamp_time in enumerate(frame["_eastern_time"])
            if start_time <= stamp_time <= end_time
        ]

    def _best_entry_alert_candidate(
        self,
        pick: Dict,
        trade_date: date,
        historical: bool = False,
    ) -> Dict:
        symbol = str(pick.get("symbol", "")).upper()
        intraday = self.data_provider.get_intraday_data(symbol, period="5d", interval="1m")
        day = self._intraday_for_date(intraday, trade_date)
        regular = self._regular_session_frame(day)
        if regular.empty:
            return self._empty_entry_alert(symbol, "No regular-session intraday data")

        positions = self._entry_window_positions(regular, trade_date)
        if not positions:
            return self._empty_entry_alert(symbol, "No candles inside the entry-alert window")

        if not historical:
            positions = [positions[-1]]

        best = None
        for position in positions:
            candidate = self._score_entry_alert_candidate(pick, regular, position)
            if best is None or candidate.get("entry_score", 0) > best.get("entry_score", 0):
                best = candidate
        return best or self._empty_entry_alert(symbol, "No entry candidate could be scored")

    def _empty_entry_alert(self, symbol: str, reason: str) -> Dict:
        return {
            "symbol": symbol,
            "qualified": False,
            "entry_score": 0.0,
            "reasons": [],
            "fail_reasons": [reason],
        }

    def _score_entry_alert_candidate(self, pick: Dict, frame: pd.DataFrame, position: int) -> Dict:
        symbol = str(pick.get("symbol", "")).upper()
        history = frame.iloc[: position + 1].copy()
        row = history.iloc[-1]

        price = self._safe_float(row.get("Close")) or self._safe_float(row.get("Open"))
        if price <= 0:
            return self._empty_entry_alert(symbol, "No usable intraday price")

        reference_price = (
            self._safe_float(pick.get("buy_zone"))
            or self._safe_float(pick.get("current_price"))
            or price
        )
        exit_target = self._safe_float(pick.get("exit_price"))
        stop_loss = self._safe_float(pick.get("stop_loss"))
        confidence = self._safe_float(pick.get("confidence"))
        council_score = self._safe_float(pick.get("council_score"))

        highs = pd.to_numeric(history["High"], errors="coerce").dropna()
        lows = pd.to_numeric(history["Low"], errors="coerce").dropna()
        closes = pd.to_numeric(history["Close"], errors="coerce").dropna()
        session_high = self._safe_float(highs.max()) if not highs.empty else price
        session_low = self._safe_float(lows.min()) if not lows.empty else price

        drop_from_reference_pct = (
            max(0.0, (reference_price - price) / reference_price * 100)
            if reference_price > 0
            else 0.0
        )
        chase_pct = (
            max(0.0, (price - reference_price) / reference_price * 100)
            if reference_price > 0
            else 0.0
        )
        drop_from_high_pct = (
            max(0.0, (session_high - price) / session_high * 100)
            if session_high > 0
            else 0.0
        )
        bounce_from_low_pct = (
            max(0.0, (price - session_low) / session_low * 100)
            if session_low > 0
            else 0.0
        )
        risk_reward = self._risk_reward_for_entry(price, stop_loss, exit_target)
        target_upside_pct = ((exit_target - price) / price * 100) if exit_target > price else 0.0
        stop_risk_pct = ((price - stop_loss) / price * 100) if 0 < stop_loss < price else 0.0

        ema_9 = self._safe_float(
            closes.ewm(span=min(9, max(len(closes), 1)), adjust=False).mean().iloc[-1]
        ) if not closes.empty else price
        last3_change_pct = 0.0
        if len(closes) >= 4 and closes.iloc[-4] > 0:
            last3_change_pct = ((closes.iloc[-1] / closes.iloc[-4]) - 1) * 100
        vwap = self._intraday_vwap(history) or price
        volume_ratio = self._intraday_volume_ratio(history)

        dip_pct = max(drop_from_reference_pct, drop_from_high_pct)
        dip_enough = dip_pct >= self.config.ENTRY_ALERT_MIN_DIP_PCT
        bounce_enough = bounce_from_low_pct >= self.config.ENTRY_ALERT_MIN_BOUNCE_PCT
        near_buy_zone = chase_pct <= self.config.ENTRY_ALERT_MAX_CHASE_PCT
        stop_buffer_ok = (
            stop_loss > 0
            and price > stop_loss * (1 + self.config.ENTRY_ALERT_STOP_BUFFER_PCT / 100)
        )
        risk_reward_ok = risk_reward >= self.config.ENTRY_ALERT_MIN_RISK_REWARD
        upside_ok = target_upside_pct >= self.config.ENTRY_ALERT_MIN_TARGET_UPSIDE_PCT
        confidence_ok = confidence >= self.config.TOP_RECOMMENDATIONS_MIN_CONFIDENCE
        trend_turn = price >= ema_9 or last3_change_pct >= 0
        reclaiming_vwap = price >= vwap * 0.9965 or last3_change_pct >= 0.05

        fail_reasons = []
        if not dip_enough:
            fail_reasons.append(f"dip {dip_pct:.2f}% < {self.config.ENTRY_ALERT_MIN_DIP_PCT:.2f}%")
        if not bounce_enough:
            fail_reasons.append(
                f"bounce {bounce_from_low_pct:.2f}% < {self.config.ENTRY_ALERT_MIN_BOUNCE_PCT:.2f}%"
            )
        if not near_buy_zone:
            fail_reasons.append(
                f"price is chasing buy zone by {chase_pct:.2f}%"
            )
        if not stop_buffer_ok:
            fail_reasons.append("too close to stop loss")
        if not risk_reward_ok:
            fail_reasons.append(
                f"risk/reward {risk_reward:.2f}x < {self.config.ENTRY_ALERT_MIN_RISK_REWARD:.2f}x"
            )
        if not upside_ok:
            fail_reasons.append(
                f"target upside {target_upside_pct:.2f}% < {self.config.ENTRY_ALERT_MIN_TARGET_UPSIDE_PCT:.2f}%"
            )
        if not confidence_ok:
            fail_reasons.append("confidence below configured floor")
        if not trend_turn:
            fail_reasons.append("no short-term turn up yet")
        if not reclaiming_vwap:
            fail_reasons.append("has not reclaimed VWAP/short momentum")

        qualified = not fail_reasons
        entry_score = (
            council_score * 0.35
            + confidence * 0.15
            + min(dip_pct, 4.0) * 10
            + min(bounce_from_low_pct, 3.0) * 6
            + min(risk_reward, 4.0) * 8
            + min(target_upside_pct, 12.0) * 1.2
            - min(stop_risk_pct, 10.0) * 0.6
        )
        if price <= vwap:
            entry_score += 4
        if trend_turn:
            entry_score += 4
        if not qualified:
            entry_score -= 25

        reasons = []
        if qualified:
            if drop_from_reference_pct >= self.config.ENTRY_ALERT_MIN_DIP_PCT:
                reasons.append("discount to council buy zone")
            elif drop_from_high_pct >= self.config.ENTRY_ALERT_MIN_DIP_PCT:
                reasons.append("intraday pullback from high")
            reasons.append("bounce from session low confirmed")
            reasons.append(f"risk/reward {risk_reward:.2f}x")

        entry_dt = row.get("_eastern_dt")
        return {
            "symbol": symbol,
            "qualified": qualified,
            "entry_score": round(max(entry_score, 0.0), 2),
            "entry_price": round(price, 4),
            "entry_time": entry_dt.isoformat() if hasattr(entry_dt, "isoformat") else datetime.now().isoformat(),
            "reference_price": round(reference_price, 4),
            "exit_target": exit_target,
            "stop_loss": stop_loss,
            "risk_reward": round(risk_reward, 2),
            "target_upside_pct": round(target_upside_pct, 2),
            "stop_risk_pct": round(stop_risk_pct, 2),
            "drop_from_reference_pct": round(drop_from_reference_pct, 2),
            "drop_from_high_pct": round(drop_from_high_pct, 2),
            "bounce_from_low_pct": round(bounce_from_low_pct, 2),
            "chase_pct": round(chase_pct, 2),
            "session_high": round(session_high, 4),
            "session_low": round(session_low, 4),
            "vwap": round(vwap, 4),
            "ema_9": round(ema_9, 4),
            "last3_change_pct": round(last3_change_pct, 2),
            "volume_ratio": round(volume_ratio, 2),
            "confidence": confidence,
            "council_score": council_score,
            "reasons": reasons,
            "fail_reasons": fail_reasons[:4],
        }

    def _entry_alert_trade_row(
        self,
        run: Dict,
        result: Dict,
        pick: Dict,
        alert: Dict,
        trade_date: date,
    ) -> Dict:
        entry_price = self._safe_float(alert.get("entry_price"))
        notional = float(self.config.SIMULATION_NOTIONAL_PER_PICK)
        quantity = round(notional / entry_price, 6) if entry_price > 0 else 0
        return {
            "trade_date": trade_date.isoformat(),
            "run_id": run["run_id"],
            "rank": int(pick.get("rank") or alert.get("rank") or 0),
            "symbol": str(pick.get("symbol", "")).upper(),
            "horizon": pick.get("horizon", result.get("horizon", "WEEK")),
            "recommended_price": self._safe_float(pick.get("current_price")),
            "entry_price": entry_price,
            "entry_time": alert.get("entry_time") or datetime.now().isoformat(),
            "quantity": quantity,
            "notional": notional,
            "exit_target": self._safe_float(pick.get("exit_price")),
            "stop_loss": self._safe_float(pick.get("stop_loss")),
            "confidence": self._safe_float(pick.get("confidence")),
            "council_score": self._safe_float(pick.get("council_score")),
            "open_source": f"intraday_dip_alert:{alert.get('candidate_source', 'recommendation')}",
            "entry_score": alert.get("entry_score"),
            "entry_reason": "; ".join(alert.get("reasons", [])),
            "entry_metrics": alert,
        }

    def _existing_simulated_symbols(self, trade_date: date, run_id: Optional[str] = None) -> set:
        with sqlite3.connect(self.db_path) as conn:
            if run_id:
                rows = conn.execute(
                    """
                    SELECT symbol FROM simulated_recommendation_trades
                    WHERE trade_date = ? AND run_id = ?
                    """,
                    (trade_date.isoformat(), run_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT symbol FROM simulated_recommendation_trades
                    WHERE trade_date = ?
                    """,
                    (trade_date.isoformat(),),
                ).fetchall()
            return {str(row[0] or "").upper() for row in rows}

    def _risk_reward_for_entry(self, entry_price: float, stop_loss: float, exit_target: float) -> float:
        if entry_price <= 0 or stop_loss <= 0 or exit_target <= entry_price:
            return 0.0
        risk = entry_price - stop_loss
        reward = exit_target - entry_price
        return reward / risk if risk > 0 else 0.0

    def _intraday_vwap(self, history: pd.DataFrame) -> float:
        if history.empty:
            return 0.0
        typical = (
            pd.to_numeric(history["High"], errors="coerce")
            + pd.to_numeric(history["Low"], errors="coerce")
            + pd.to_numeric(history["Close"], errors="coerce")
        ) / 3
        if "Volume" in history.columns:
            volume = pd.to_numeric(history["Volume"], errors="coerce").fillna(0)
        else:
            volume = pd.Series([0] * len(history), index=history.index)
        if volume.sum() > 0:
            return self._safe_float((typical * volume).sum() / volume.sum())
        return self._safe_float(typical.mean())

    def _intraday_volume_ratio(self, history: pd.DataFrame) -> float:
        if history.empty or "Volume" not in history.columns:
            return 0.0
        volume = pd.to_numeric(history["Volume"], errors="coerce").fillna(0)
        if len(volume) < 6:
            return 0.0
        current = self._safe_float(volume.iloc[-1])
        baseline = self._safe_float(volume.iloc[:-1].tail(20).median())
        if baseline <= 0:
            return 0.0
        return current / baseline

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
            lines.append(f"Entry-alert backfill failed: {backfill.get('error')}.")
        elif backfill:
            captured = backfill.get("captured", 0)
            requested = backfill.get("requested", 0)
            errors = backfill.get("errors") or []
            lines.append(f"Entry-alert backfill captured {captured}/{requested} picks.")
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

    lines.append("This is an entry-alert simulation snapshot for research only.")
    return "\n".join(lines).strip()


def format_period_summary_message(summary: Dict) -> str:
    period = str(summary.get("period") or "WEEK").upper()
    title = "End-of-week" if period == "WEEK" else "End-of-month"
    total_pnl = float(summary.get("total_pnl") or 0)
    target = float(summary.get("target") or 0)
    target_gap = total_pnl - target
    lines = [
        (
            f"Simulated {title} P/L - {summary.get('start_date')} to "
            f"{summary.get('end_date')}: {_format_signed_dollars(total_pnl)} "
            f"({float(summary.get('total_pnl_pct') or 0):+.2f}%)"
        ),
        (
            f"Trades: {summary.get('trade_count', 0)} across "
            f"{summary.get('trading_days', 0)} day(s) | "
            f"Win/loss: {summary.get('winners', 0)}/{summary.get('losers', 0)}"
        ),
        (
            f"Target: ${target:.2f} | "
            f"{'above' if target_gap >= 0 else 'below'} by ${abs(target_gap):.2f}"
        ),
        "Mode: simulated recommendation entries; no real-order P/L included.",
    ]
    daily = summary.get("daily_pnl") or []
    if daily:
        lines.append("")
        lines.append("Daily totals:")
        for row in daily:
            lines.append(f"{row.get('date')}: {_format_signed_dollars(row.get('pnl', 0))}")
    else:
        lines.append("No completed simulated entries were recorded for this period.")
    return "\n".join(lines).strip()


def format_entry_alert_message(scan: Dict) -> str:
    alerts = scan.get("alerts", [])
    trade_date = scan.get("trade_date", "")
    target_weekly = float(scan.get("target_weekly") or 0)
    target_monthly = float(scan.get("target_monthly") or 0)
    alert_summary = " | ".join(
        f"{alert.get('symbol')} ${alert.get('entry_price', 0):.2f}"
        for alert in sorted(alerts, key=lambda row: row.get("rank") or 999)
    )
    lines = [
        (
            f"BUY DIP ALERT - {trade_date}: {alert_summary}"
            if alert_summary
            else f"BUY DIP ALERT - {trade_date}: no entry trigger"
        ),
        (
            f"Captured {scan.get('captured', 0)}/{scan.get('requested', 0)} "
            "eligible candidates that met the intraday dip rules."
        ),
        "Mode: alert + simulated entry only; no real order placed.",
        (
            f"Target context: ${target_weekly:.0f}/week, ${target_monthly:.0f}/month; "
            "not guaranteed."
        ),
        "",
    ]

    if not alerts:
        evaluated = scan.get("evaluated", []) or []
        if scan.get("error"):
            lines.append(scan["error"])
        elif evaluated:
            lines.append("Closest candidates:")
            for item in evaluated[:3]:
                fail = "; ".join(item.get("fail_reasons", [])[:2])
                lines.append(
                    (
                        f"{item.get('symbol')}: score {item.get('entry_score', 0):.1f}, "
                        f"price ${item.get('entry_price', 0):.2f}"
                        f"{' - ' + fail if fail else ''}"
                    )
                )
        else:
            lines.append("No top picks had usable intraday data yet.")
        return "\n".join(lines).strip()

    for alert in sorted(alerts, key=lambda row: row.get("rank") or 999):
        metrics = alert.get("entry_metrics") or {}
        entry_value = float(alert.get("entry_price", 0) or 0) * float(alert.get("quantity", 0) or 0)
        reasons = "; ".join(metrics.get("reasons", [])[:3] or [alert.get("entry_reason", "")])
        source = str(metrics.get("candidate_source") or "recommendation").replace("_", " ")
        rank = int(alert.get("rank") or 0)
        label = f"{rank}." if rank > 0 else "Watch:"
        lines.append(
            (
                f"{label} {alert.get('symbol')}: "
                f"buy near ${alert.get('entry_price', 0):.2f} | "
                f"sim ${entry_value:.2f} x {alert.get('quantity', 0):.4f}"
            )
        )
        lines.append(
            (
                f"Target ${alert.get('exit_target', 0):.2f} | "
                f"stop ${alert.get('stop_loss', 0):.2f} | "
                f"R/R {metrics.get('risk_reward', 0):.2f}x"
            )
        )
        lines.append(
            (
                f"Dip {metrics.get('drop_from_reference_pct', 0):.2f}% vs pick, "
                f"{metrics.get('drop_from_high_pct', 0):.2f}% from high; "
                f"bounce {metrics.get('bounce_from_low_pct', 0):.2f}%"
            )
        )
        if reasons:
            lines.append(f"Why: {reasons}")
        lines.append(f"Source: {source}")

    lines.append("")
    lines.append("Confirm in the app before placing live trades.")
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
