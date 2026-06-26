import json
import logging
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import Config


class CouncilLearningMemory:
    """Local retrieval memory for council recommendations and daily outcomes."""

    UNKNOWN_SECTOR = "Unknown"
    MARKET_SCOPE = "MARKET"
    SECTOR_SCOPE = "SECTOR"
    SYMBOL_SCOPE = "SYMBOL"
    MISSED_SECTOR_SCOPE = "MISSED_SECTOR"
    MISSED_SYMBOL_SCOPE = "MISSED_SYMBOL"

    SECTOR_FALLBACKS = {
        "AAPL": "Mega-cap Technology",
        "MSFT": "Mega-cap Technology",
        "GOOGL": "Mega-cap Technology",
        "GOOG": "Mega-cap Technology",
        "AMZN": "Consumer Internet",
        "META": "Consumer Internet",
        "NFLX": "Consumer Internet",
        "ORCL": "Enterprise Software",
        "CRM": "Enterprise Software",
        "NOW": "Enterprise Software",
        "ADBE": "Enterprise Software",
        "NVDA": "Semiconductors",
        "AMD": "Semiconductors",
        "AVGO": "Semiconductors",
        "QCOM": "Semiconductors",
        "ADI": "Semiconductors",
        "TXN": "Semiconductors",
        "INTC": "Semiconductors",
        "MRVL": "Semiconductors",
        "ARM": "Semiconductors",
        "LRCX": "Semiconductor Equipment",
        "KLAC": "Semiconductor Equipment",
        "AMAT": "Semiconductor Equipment",
        "ASML": "Semiconductor Equipment",
        "MU": "Memory",
        "SNDK": "Storage",
        "WDC": "Storage",
        "STX": "Storage",
        "PSTG": "Storage",
        "NTAP": "Storage",
        "SMCI": "AI Infrastructure",
        "DELL": "AI Infrastructure",
        "HPE": "AI Infrastructure",
        "UNH": "Health Care",
        "LLY": "Health Care",
        "JNJ": "Health Care",
        "ABBV": "Health Care",
        "JPM": "Financials",
        "BAC": "Financials",
        "V": "Payments",
        "MA": "Payments",
        "XOM": "Energy",
        "CVX": "Energy",
        "COP": "Energy",
        "KO": "Consumer Staples",
        "PG": "Consumer Staples",
        "WMT": "Consumer Staples",
        "COST": "Consumer Staples",
        "CAT": "Industrials",
        "UPS": "Industrials",
        "CSX": "Industrials",
        "ROKU": "High Beta Consumer Tech",
        "PTON": "High Beta Consumer Tech",
        "DOCU": "High Beta Software",
    }

    def __init__(self, db_path: Optional[Path] = None, config: Optional[Config] = None):
        self.config = config or Config()
        self.db_path = Path(db_path or "trading_data.db")
        self.logger = logging.getLogger(__name__)
        self._init_database()

    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS council_learning_lessons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lesson_date TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    scope_type TEXT NOT NULL,
                    scope_key TEXT NOT NULL,
                    sector TEXT,
                    symbol TEXT,
                    trade_count INTEGER NOT NULL,
                    winners INTEGER NOT NULL,
                    losers INTEGER NOT NULL,
                    total_pnl REAL NOT NULL,
                    avg_pnl_pct REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    score_adjustment REAL NOT NULL,
                    lesson_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    UNIQUE(lesson_date, scope_type, scope_key)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS council_symbol_sectors (
                    symbol TEXT PRIMARY KEY,
                    sector TEXT NOT NULL,
                    source TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def learn_from_summary(self, summary: Dict) -> Dict:
        trade_date = summary.get("trade_date") or date.today().isoformat()
        trades = summary.get("trades", []) or []
        return self.learn_from_trades(trade_date, trades)

    def learn_from_trade_date(self, trade_date: str) -> Dict:
        trades = self._load_trade_rows(trade_date)
        return self.learn_from_trades(trade_date, trades)

    def learn_from_trades(self, trade_date: str, trades: List[Dict]) -> Dict:
        if not getattr(self.config, "COUNCIL_RAG_ENABLED", True):
            return {"enabled": False, "lesson_date": trade_date, "lessons_written": 0, "lessons": []}

        marked = [row for row in trades if row.get("pnl_pct") is not None]
        if not marked:
            return {"enabled": True, "lesson_date": trade_date, "lessons_written": 0, "lessons": []}

        run_payloads = self._load_run_payloads(
            [str(row.get("run_id") or "") for row in marked if row.get("run_id")]
        )
        enriched = []
        for row in marked:
            symbol = str(row.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            sector = self._sector_for_trade(row, run_payloads)
            enriched_row = {**row, "symbol": symbol, "sector": sector}
            enriched.append(enriched_row)
            self.remember_symbol_sector(symbol, sector, source="daily_lesson")

        groups: Dict[Tuple[str, str], List[Dict]] = {}
        groups[(self.MARKET_SCOPE, "ALL")] = list(enriched)
        for row in enriched:
            groups.setdefault((self.SYMBOL_SCOPE, row["symbol"]), []).append(row)
            if self._usable_sector(row.get("sector")):
                groups.setdefault((self.SECTOR_SCOPE, row["sector"]), []).append(row)

        lessons = []
        for (scope_type, scope_key), rows in groups.items():
            stats = self._summarize_rows(rows)
            adjustment = self._lesson_adjustment(stats)
            lesson_text = self._lesson_text(trade_date, scope_type, scope_key, stats, adjustment)
            lesson = {
                "lesson_date": trade_date,
                "generated_at": datetime.now().isoformat(),
                "scope_type": scope_type,
                "scope_key": scope_key,
                "sector": scope_key if scope_type == self.SECTOR_SCOPE else None,
                "symbol": scope_key if scope_type == self.SYMBOL_SCOPE else None,
                "score_adjustment": adjustment,
                "lesson_text": lesson_text,
                **stats,
                "metadata": {
                    "symbols": sorted({row["symbol"] for row in rows}),
                    "outcomes": [row.get("outcome", "UNKNOWN") for row in rows],
                },
            }
            self._upsert_lesson(lesson)
            lessons.append(lesson)

        lessons.sort(key=lambda item: abs(item["score_adjustment"]), reverse=True)
        return {
            "enabled": True,
            "lesson_date": trade_date,
            "lessons_written": len(lessons),
            "lessons": [
                {
                    "scope_type": item["scope_type"],
                    "scope_key": item["scope_key"],
                    "score_adjustment": item["score_adjustment"],
                    "lesson_text": item["lesson_text"],
                }
                for item in lessons[:6]
            ],
        }

    def learn_from_missed_movers(self, trade_date: str, movers: List[Dict]) -> Dict:
        """Persist opportunity-gap lessons for screened movers that were not picked."""
        if not getattr(self.config, "COUNCIL_RAG_ENABLED", True):
            return {"enabled": False, "lesson_date": trade_date, "lessons_written": 0, "lessons": []}

        clean = []
        for mover in movers or []:
            symbol = str(mover.get("symbol") or "").strip().upper()
            change_pct = self._safe_float(mover.get("change_percent"))
            if not symbol or change_pct <= 0:
                continue
            sector = mover.get("sector") if self._usable_sector(mover.get("sector")) else self.sector_for_symbol(symbol)
            clean.append({**mover, "symbol": symbol, "sector": sector, "change_percent": change_pct})
            self.remember_symbol_sector(symbol, sector, source="missed_mover")

        if not clean:
            return {"enabled": True, "lesson_date": trade_date, "lessons_written": 0, "lessons": []}

        groups: Dict[Tuple[str, str], List[Dict]] = {}
        for row in clean:
            groups.setdefault((self.MISSED_SYMBOL_SCOPE, row["symbol"]), []).append(row)
            if self._usable_sector(row.get("sector")):
                groups.setdefault((self.MISSED_SECTOR_SCOPE, row["sector"]), []).append(row)

        lessons = []
        for (scope_type, scope_key), rows in groups.items():
            avg_change = sum(self._safe_float(row.get("change_percent")) for row in rows) / len(rows)
            adjustment = round(
                min(
                    float(getattr(self.config, "COUNCIL_RAG_MAX_SCORE_ADJUSTMENT", 14.0)) * 0.6,
                    max(2.0, avg_change * 1.2),
                ),
                2,
            )
            lesson_text = (
                f"missed mover {scope_key} on {trade_date}: "
                f"{len(rows)} screened names averaged +{avg_change:.2f}% but were not picked; "
                "raise priority when current signals confirm."
            )
            lesson = {
                "lesson_date": trade_date,
                "generated_at": datetime.now().isoformat(),
                "scope_type": scope_type,
                "scope_key": scope_key,
                "sector": scope_key if scope_type == self.MISSED_SECTOR_SCOPE else rows[0].get("sector"),
                "symbol": scope_key if scope_type == self.MISSED_SYMBOL_SCOPE else None,
                "trade_count": len(rows),
                "winners": len(rows),
                "losers": 0,
                "total_pnl": 0.0,
                "avg_pnl_pct": round(avg_change, 2),
                "win_rate": 1.0,
                "score_adjustment": adjustment,
                "lesson_text": lesson_text,
                "metadata": {
                    "source": "missed_mover",
                    "symbols": sorted({row["symbol"] for row in rows}),
                    "changes": {
                        row["symbol"]: self._safe_float(row.get("change_percent"))
                        for row in rows
                    },
                },
            }
            self._upsert_lesson(lesson)
            lessons.append(lesson)

        lessons.sort(key=lambda item: item["score_adjustment"], reverse=True)
        return {
            "enabled": True,
            "lesson_date": trade_date,
            "lessons_written": len(lessons),
            "lessons": [
                {
                    "scope_type": item["scope_type"],
                    "scope_key": item["scope_key"],
                    "score_adjustment": item["score_adjustment"],
                    "lesson_text": item["lesson_text"],
                }
                for item in lessons[:6]
            ],
        }

    def retrieve_feedback(
        self,
        symbol: str,
        sector: Optional[str] = None,
        analysis: Optional[Dict] = None,
        limit: Optional[int] = None,
    ) -> Dict:
        if not getattr(self.config, "COUNCIL_RAG_ENABLED", True):
            return self._empty_feedback()

        symbol = str(symbol or "").strip().upper()
        sector = sector if self._usable_sector(sector) else self.sector_for_symbol(symbol)
        lookback_days = max(1, int(getattr(self.config, "COUNCIL_RAG_LOOKBACK_DAYS", 45)))
        max_lessons = max(1, int(limit or getattr(self.config, "COUNCIL_RAG_MAX_LESSONS", 6)))
        since = (datetime.now() - timedelta(days=lookback_days)).date().isoformat()

        clauses = ["(scope_type = ? AND scope_key = ?)"]
        params: List[str] = [self.MARKET_SCOPE, "ALL"]
        if symbol:
            clauses.append("(scope_type = ? AND scope_key = ?)")
            params.extend([self.SYMBOL_SCOPE, symbol])
            clauses.append("(scope_type = ? AND scope_key = ?)")
            params.extend([self.MISSED_SYMBOL_SCOPE, symbol])
        if self._usable_sector(sector):
            clauses.append("(scope_type = ? AND scope_key = ?)")
            params.extend([self.SECTOR_SCOPE, sector])
            clauses.append("(scope_type = ? AND scope_key = ?)")
            params.extend([self.MISSED_SECTOR_SCOPE, sector])
        params.append(since)

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    f"""
                    SELECT *
                    FROM council_learning_lessons
                    WHERE ({' OR '.join(clauses)})
                      AND lesson_date >= ?
                    ORDER BY lesson_date DESC, ABS(score_adjustment) DESC
                    LIMIT 50
                    """,
                    params,
                ).fetchall()
        except Exception as exc:
            self.logger.debug("Could not retrieve council learning memory: %s", exc)
            return self._empty_feedback()

        lessons = []
        symbol_adjustment = 0.0
        sector_adjustment = 0.0
        market_adjustment = 0.0
        symbol_count = 0
        sector_count = 0
        market_count = 0
        cooldown = False

        for row in rows:
            item = dict(row)
            scope_type = item["scope_type"]
            weight = self._scope_weight(scope_type)
            recency = self._recency_weight(item.get("lesson_date"), lookback_days)
            adjustment = self._safe_float(item.get("score_adjustment")) * weight * recency
            relevance = abs(adjustment)
            lesson = {
                "lesson_date": item.get("lesson_date"),
                "scope_type": scope_type,
                "scope_key": item.get("scope_key"),
                "score_adjustment": round(adjustment, 2),
                "relevance": round(relevance, 2),
                "lesson_text": item.get("lesson_text", ""),
                "avg_pnl_pct": self._safe_float(item.get("avg_pnl_pct")),
                "win_rate": self._safe_float(item.get("win_rate")),
                "trade_count": int(item.get("trade_count") or 0),
            }
            lessons.append(lesson)
            if scope_type in (self.SYMBOL_SCOPE, self.MISSED_SYMBOL_SCOPE):
                symbol_count += 1
                symbol_adjustment += adjustment
                if (
                    scope_type == self.SYMBOL_SCOPE
                    and
                    lesson["trade_count"] >= max(2, int(getattr(self.config, "TOP_RECOMMENDATIONS_HISTORY_MIN_TRADES", 2)))
                    and lesson["avg_pnl_pct"] <= -2.0
                    and lesson["win_rate"] <= 0.25
                ):
                    cooldown = True
            elif scope_type in (self.SECTOR_SCOPE, self.MISSED_SECTOR_SCOPE):
                sector_count += 1
                sector_adjustment += adjustment
            else:
                market_count += 1
                market_adjustment += adjustment

        lessons.sort(key=lambda item: item["relevance"], reverse=True)
        total_adjustment = self._cap_adjustment(
            symbol_adjustment + sector_adjustment + market_adjustment
        )
        selected_lessons = lessons[:max_lessons]
        note = self._retrieval_note(selected_lessons)

        return {
            "lesson_count": len(lessons),
            "symbol_lesson_count": symbol_count,
            "sector_lesson_count": sector_count,
            "market_lesson_count": market_count,
            "symbol_adjustment": round(symbol_adjustment, 2),
            "sector_adjustment": round(sector_adjustment, 2),
            "market_adjustment": round(market_adjustment, 2),
            "score_adjustment": round(total_adjustment, 2),
            "cooldown": cooldown,
            "sector": sector,
            "lessons": selected_lessons,
            "note": note,
        }

    def remember_symbol_sector(self, symbol: str, sector: str, source: str = "system"):
        symbol = str(symbol or "").strip().upper()
        if not symbol or not self._usable_sector(sector):
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO council_symbol_sectors (symbol, sector, source, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    sector = excluded.sector,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (symbol, sector, source, datetime.now().isoformat()),
            )

    def sector_for_symbol(self, symbol: str) -> str:
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return self.UNKNOWN_SECTOR
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT sector FROM council_symbol_sectors WHERE symbol = ?",
                    (symbol,),
                ).fetchone()
                if row and self._usable_sector(row[0]):
                    return row[0]
        except Exception:
            pass
        return self.SECTOR_FALLBACKS.get(symbol, self.UNKNOWN_SECTOR)

    def _load_trade_rows(self, trade_date: str) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT *
                FROM simulated_recommendation_trades
                WHERE trade_date = ?
                  AND pnl_pct IS NOT NULL
                ORDER BY run_id, rank
                """,
                (trade_date,),
            ).fetchall()
            return [dict(row) for row in rows]

    def _load_run_payloads(self, run_ids: List[str]) -> Dict[str, Dict]:
        run_ids = sorted({run_id for run_id in run_ids if run_id})
        if not run_ids:
            return {}
        placeholders = ",".join("?" for _ in run_ids)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    f"""
                    SELECT run_id, payload_json
                    FROM recommendation_runs
                    WHERE run_id IN ({placeholders})
                    """,
                    run_ids,
                ).fetchall()
        except Exception:
            return {}

        payloads = {}
        for row in rows:
            try:
                payloads[row["run_id"]] = json.loads(row["payload_json"])
            except Exception:
                payloads[row["run_id"]] = {}
        return payloads

    def _sector_for_trade(self, trade: Dict, run_payloads: Dict[str, Dict]) -> str:
        symbol = str(trade.get("symbol") or "").strip().upper()
        payload = run_payloads.get(str(trade.get("run_id") or ""), {})
        sector = self._sector_from_payload(payload, symbol)
        if self._usable_sector(sector):
            return sector
        return self.sector_for_symbol(symbol)

    def _sector_from_payload(self, payload: Dict, symbol: str) -> str:
        if not payload:
            return self.UNKNOWN_SECTOR
        symbol = str(symbol or "").strip().upper()
        for key in ("recommendations", "candidate_snapshot", "held_momentum", "breakout_watch"):
            for item in payload.get(key, []) or []:
                if str(item.get("symbol") or "").strip().upper() != symbol:
                    continue
                sector = item.get("sector")
                if self._usable_sector(sector):
                    return sector
                feedback = item.get("learning_feedback") or item.get("performance_feedback") or {}
                sector = feedback.get("sector")
                if self._usable_sector(sector):
                    return sector
        return self.UNKNOWN_SECTOR

    def _summarize_rows(self, rows: List[Dict]) -> Dict:
        trade_count = len(rows)
        winners = len([row for row in rows if self._safe_float(row.get("pnl")) > 0])
        losers = trade_count - winners
        total_pnl = sum(self._safe_float(row.get("pnl")) for row in rows)
        avg_pnl_pct = (
            sum(self._safe_float(row.get("pnl_pct")) for row in rows) / trade_count
            if trade_count
            else 0.0
        )
        return {
            "trade_count": trade_count,
            "winners": winners,
            "losers": losers,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl_pct": round(avg_pnl_pct, 2),
            "win_rate": round(winners / trade_count, 3) if trade_count else 0.0,
        }

    def _lesson_adjustment(self, stats: Dict) -> float:
        trade_count = int(stats.get("trade_count") or 0)
        avg = self._safe_float(stats.get("avg_pnl_pct"))
        win_rate = self._safe_float(stats.get("win_rate"))
        adjustment = 0.0

        if trade_count >= max(1, int(getattr(self.config, "TOP_RECOMMENDATIONS_HISTORY_MIN_TRADES", 2))):
            if avg <= -1.0:
                adjustment -= min(10.0, abs(avg) * 2.5)
            if win_rate < 0.35:
                adjustment -= min(6.0, (0.35 - win_rate) * 18.0)
            if avg >= 1.0 and win_rate >= 0.5:
                adjustment += min(8.0, avg * 2.0)
            if win_rate >= 0.75 and avg > 0:
                adjustment += min(4.0, win_rate * 3.0)
        elif avg <= -3.0:
            adjustment -= 3.0
        elif avg >= 3.0:
            adjustment += 3.0

        return round(self._cap_adjustment(adjustment), 2)

    def _lesson_text(self, trade_date: str, scope_type: str, scope_key: str, stats: Dict, adjustment: float) -> str:
        if adjustment > 1:
            action = "raise priority when current signals confirm"
        elif adjustment < -1:
            action = "lower priority or require stronger confirmation"
        else:
            action = "treat as neutral context"
        return (
            f"{scope_type.lower()} {scope_key} on {trade_date}: "
            f"{stats['trade_count']} simulated trades, avg {stats['avg_pnl_pct']:+.2f}%, "
            f"win rate {stats['win_rate']:.0%}, total ${stats['total_pnl']:+.2f}; {action}."
        )

    def _upsert_lesson(self, lesson: Dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO council_learning_lessons
                (lesson_date, generated_at, scope_type, scope_key, sector, symbol,
                 trade_count, winners, losers, total_pnl, avg_pnl_pct, win_rate,
                 score_adjustment, lesson_text, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(lesson_date, scope_type, scope_key) DO UPDATE SET
                    generated_at = excluded.generated_at,
                    sector = excluded.sector,
                    symbol = excluded.symbol,
                    trade_count = excluded.trade_count,
                    winners = excluded.winners,
                    losers = excluded.losers,
                    total_pnl = excluded.total_pnl,
                    avg_pnl_pct = excluded.avg_pnl_pct,
                    win_rate = excluded.win_rate,
                    score_adjustment = excluded.score_adjustment,
                    lesson_text = excluded.lesson_text,
                    metadata_json = excluded.metadata_json
                """,
                (
                    lesson["lesson_date"],
                    lesson["generated_at"],
                    lesson["scope_type"],
                    lesson["scope_key"],
                    lesson.get("sector"),
                    lesson.get("symbol"),
                    lesson["trade_count"],
                    lesson["winners"],
                    lesson["losers"],
                    lesson["total_pnl"],
                    lesson["avg_pnl_pct"],
                    lesson["win_rate"],
                    lesson["score_adjustment"],
                    lesson["lesson_text"],
                    json.dumps(lesson.get("metadata", {}), default=str),
                ),
            )

    def _scope_weight(self, scope_type: str) -> float:
        if scope_type == self.SYMBOL_SCOPE:
            return float(getattr(self.config, "COUNCIL_RAG_SYMBOL_WEIGHT", 1.0))
        if scope_type == self.MISSED_SYMBOL_SCOPE:
            return float(getattr(self.config, "COUNCIL_RAG_MISSED_MOVER_WEIGHT", 0.60))
        if scope_type == self.SECTOR_SCOPE:
            return float(getattr(self.config, "COUNCIL_RAG_SECTOR_WEIGHT", 0.45))
        if scope_type == self.MISSED_SECTOR_SCOPE:
            return (
                float(getattr(self.config, "COUNCIL_RAG_SECTOR_WEIGHT", 0.45))
                * float(getattr(self.config, "COUNCIL_RAG_MISSED_MOVER_WEIGHT", 0.60))
            )
        return float(getattr(self.config, "COUNCIL_RAG_MARKET_WEIGHT", 0.20))

    def _recency_weight(self, lesson_date: Optional[str], lookback_days: int) -> float:
        try:
            parsed = datetime.fromisoformat(str(lesson_date)).date()
            age_days = max((date.today() - parsed).days, 0)
        except Exception:
            return 0.5
        if lookback_days <= 0:
            return 1.0
        return max(0.25, 1.0 - (age_days / max(lookback_days, 1)))

    def _retrieval_note(self, lessons: List[Dict]) -> str:
        if not lessons:
            return ""
        snippets = []
        for lesson in lessons[:3]:
            snippets.append(
                f"{lesson['scope_type']} {lesson['scope_key']} "
                f"{lesson['score_adjustment']:+.1f}: {lesson['avg_pnl_pct']:+.2f}% avg"
            )
        return "RAG retrieved lessons - " + "; ".join(snippets)

    def _cap_adjustment(self, value: float) -> float:
        cap = abs(float(getattr(self.config, "COUNCIL_RAG_MAX_SCORE_ADJUSTMENT", 14.0)))
        return max(-cap, min(cap, self._safe_float(value)))

    def _empty_feedback(self) -> Dict:
        return {
            "lesson_count": 0,
            "symbol_lesson_count": 0,
            "sector_lesson_count": 0,
            "market_lesson_count": 0,
            "symbol_adjustment": 0.0,
            "sector_adjustment": 0.0,
            "market_adjustment": 0.0,
            "score_adjustment": 0.0,
            "cooldown": False,
            "sector": self.UNKNOWN_SECTOR,
            "lessons": [],
            "note": "",
        }

    def _usable_sector(self, sector: Optional[str]) -> bool:
        text = str(sector or "").strip()
        return bool(text and text.lower() not in ("unknown", "none", "nan"))

    def _safe_float(self, value) -> float:
        try:
            if value is None:
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0
