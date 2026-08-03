"""
Top-5 recommendation engine with a lightweight trading-agent council.

The council is intentionally deterministic: each agent reads the existing
strategy output, votes from its own lens, then a skeptic challenges weak picks
before the final arbiter ranks actionable BUY candidates.
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import Config
from council_memory import CouncilLearningMemory


class TopRecommendationEngine:
    """Builds auditable top-pick recommendations from the existing strategy stack."""

    NON_STOCK_SYMBOLS = {
        "SPY", "QQQ", "IWM", "DIA", "EEM", "XLF", "XLK", "XLE", "XLY", "XLP",
        "XLV", "XLI", "XLB", "XLU", "XLRE", "GLD", "SLV", "USO", "UNG", "TLT",
        "HYG", "LQD", "VXX", "UVXY", "SQQQ", "TQQQ", "SPXL", "SPXS", "SOXL",
        "SOXS", "TECL", "TECS", "FNGU", "FNGD", "ARKK", "ARKW", "ARKG",
    }

    COUNCIL_AGENTS = [
        "MomentumAgent",
        "BreakoutAgent",
        "MeanReversionAgent",
        "VolumeAgent",
        "FundamentalAgent",
        "RelativeStrengthAgent",
        "MacroRiskAgent",
        "SkepticAgent",
        "ArbiterAgent",
    ]

    def __init__(self, trading_strategy, portfolio_manager=None):
        self.trading_strategy = trading_strategy
        self.portfolio_manager = portfolio_manager
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.db_path = Path("trading_data.db")
        self.learning_memory = CouncilLearningMemory(self.db_path, self.config)
        self._performance_cache: Optional[Dict[str, Dict]] = None
        self._performance_cache_time = datetime.min
        self._init_database()

    def get_top_recommendations(
        self,
        symbols: List[str],
        horizon: str = "WEEK",
        limit: int = 5,
        universe_size: int = 50,
    ) -> Dict:
        """Analyze a universe and return the strongest challenged BUY candidates."""
        horizon = self._normalize_horizon(horizon)
        limit = max(1, min(int(limit or 5), 10))
        universe_size = max(limit, min(int(universe_size or 50), 120))
        universe = self._prepare_universe(symbols, universe_size)

        run_id = str(uuid.uuid4())
        generated_at = datetime.now()
        candidates = []
        analyses = {}
        errors = []

        for symbol in universe:
            try:
                analysis = self.trading_strategy.analyze_stock(symbol, horizon=horizon)
                if analysis:
                    analyses[symbol] = analysis
                candidate = self._build_candidate(analysis)
                if candidate:
                    candidates.append(candidate)
            except Exception as exc:
                self.logger.error("Council analysis failed for %s: %s", symbol, exc)
                errors.append({"symbol": symbol, "error": str(exc)})

        actionable = [c for c in candidates if c["actionable"]]
        ranked = sorted(actionable, key=lambda c: c["council_score"], reverse=True)
        top = self._select_sector_balanced(ranked, limit)
        held_momentum = self._build_held_momentum_alerts(candidates)
        breakout_watch = self._build_intraday_breakout_watch(universe, analyses, candidates)

        result = {
            "run_id": run_id,
            "generated_at": generated_at.isoformat(),
            "horizon": horizon,
            "limit": limit,
            "universe_size": len(universe),
            "universe": universe,
            "candidate_count": len(candidates),
            "actionable_count": len(actionable),
            "council_agents": self.COUNCIL_AGENTS,
            "recommendations": [
                {**pick, "rank": rank}
                for rank, pick in enumerate(top, start=1)
            ],
            "candidate_snapshot": self._candidate_snapshot(candidates),
            "held_momentum": held_momentum,
            "breakout_watch": breakout_watch,
            "errors": errors[:10],
            "disclaimer": (
                "Research signal only. Use paper trading or human approval before "
                "placing live orders."
            ),
        }
        result["message_preview"] = format_top_recommendations_message(result)
        self._save_run(result)
        return result

    def _normalize_horizon(self, horizon: str) -> str:
        horizon = (horizon or "WEEK").upper()
        return horizon if horizon in ("WEEK", "MONTH") else "WEEK"

    def _prepare_universe(self, symbols: List[str], universe_size: int) -> List[str]:
        seen = set()
        universe = []
        for raw in symbols or []:
            symbol = str(raw or "").strip().upper()
            if not symbol or symbol in seen or not self._is_stock_candidate(symbol):
                continue
            seen.add(symbol)
            universe.append(symbol)
            if len(universe) >= universe_size:
                break
        return universe

    def _is_stock_candidate(self, symbol: str) -> bool:
        symbol = str(symbol or "").strip().upper()
        if not symbol or symbol in self.NON_STOCK_SYMBOLS:
            return False
        if symbol.endswith((".WS", ".W", ".U", ".R")):
            return False
        leveraged_fragments = ("2X", "3X", "ULTRA", "BEAR", "BULL")
        return not any(fragment in symbol for fragment in leveraged_fragments)

    def _build_candidate(self, analysis: Dict) -> Optional[Dict]:
        if not analysis:
            return None

        symbol = analysis.get("symbol", "")
        horizon = analysis.get("horizon", "WEEK")
        rec = analysis.get("recommendation", {}) or {}
        risk = analysis.get("risk_score", {}) or {}
        sector = self._sector_from_analysis(analysis)
        current_price = self._safe_float(analysis.get("current_price", 0))
        exit_price = self._safe_float(rec.get("take_profit", 0))
        stop_loss = self._safe_float(rec.get("stop_loss", 0))
        confidence = self._safe_float(rec.get("confidence", 0))
        position_size = self._safe_float(rec.get("position_size", 0))
        action = str(rec.get("action", "HOLD")).upper()
        risk_reward = self._risk_reward(current_price, stop_loss, exit_price)
        council = self._run_council(analysis, risk_reward)
        recent_history = self._performance_feedback(symbol)
        retrieved = self.learning_memory.retrieve_feedback(
            symbol=symbol,
            sector=sector,
            analysis=analysis,
        )
        history = self._combine_feedback(recent_history, retrieved)
        objections = list(council.get("objections", []))
        thesis = list(council.get("thesis", []))
        if history.get("note"):
            if history.get("score_adjustment", 0) < 0:
                objections.append(history["note"])
            elif history.get("score_adjustment", 0) > 0:
                thesis.append(history["note"])

        already_held = (
            self.portfolio_manager is not None
            and symbol in getattr(self.portfolio_manager, "positions", {})
        )
        actionable = (
            action == "BUY"
            and current_price > 0
            and stop_loss > 0
            and exit_price > current_price
            and stop_loss < current_price
            and risk_reward >= 1.15
            and confidence >= self.config.TOP_RECOMMENDATIONS_MIN_CONFIDENCE
            and not already_held
            and not council.get("hard_reject", False)
            and not history.get("cooldown", False)
        )
        council_score = self._score_candidate(
            analysis=analysis,
            council=council,
            confidence=confidence,
            risk_reward=risk_reward,
            position_size=position_size,
            actionable=actionable,
            history_adjustment=self._safe_float(history.get("score_adjustment")),
        )

        return {
            "symbol": symbol,
            "sector": sector,
            "horizon": horizon,
            "action": action,
            "actionable": actionable,
            "current_price": current_price,
            "buy_zone": current_price,
            "exit_price": exit_price,
            "stop_loss": stop_loss,
            "confidence": confidence,
            "council_score": council_score,
            "risk_reward": risk_reward,
            "position_size": position_size,
            "risk_level": risk.get("level", "MEDIUM"),
            "risk_factors": risk.get("factors", []),
            "already_held": already_held,
            "agent_votes": council.get("votes", {}),
            "approval_count": council.get("approval_count", 0),
            "challenge_count": council.get("challenge_count", 0),
            "objections": self._unique_text(objections)[:6],
            "thesis": self._unique_text(thesis)[:6],
            "performance_feedback": history,
            "learning_feedback": retrieved,
            "timestamp": datetime.now().isoformat(),
        }

    def _run_council(self, analysis: Dict, risk_reward: float) -> Dict:
        signals = analysis.get("signals", {}) or {}
        risk = analysis.get("risk_score", {}) or {}
        rec = analysis.get("recommendation", {}) or {}
        strategy_agents = [
            ("MomentumAgent", "momentum"),
            ("BreakoutAgent", "breakout"),
            ("MeanReversionAgent", "reversal"),
            ("VolumeAgent", "volume"),
            ("FundamentalAgent", "fundamentals"),
            ("RelativeStrengthAgent", "relative_strength"),
        ]

        votes = {}
        thesis = []
        objections = []
        approval_count = 0
        challenge_count = 0

        for agent_name, signal_key in strategy_agents:
            signal = (signals.get(signal_key, {}) or {})
            vote, note = self._vote_from_signal(signal)
            votes[agent_name] = {
                "vote": vote,
                "signal": signal.get("signal", "HOLD"),
                "strength": self._safe_float(signal.get("strength", 0)),
                "note": note,
            }
            if vote == "APPROVE":
                approval_count += 1
                thesis.append(note)
            elif vote == "CHALLENGE":
                challenge_count += 1
                objections.append(note)

        macro_vote, macro_note = self._macro_risk_vote(analysis)
        votes["MacroRiskAgent"] = {"vote": macro_vote, "note": macro_note}
        if macro_vote == "APPROVE":
            approval_count += 1
            thesis.append(macro_note)
        elif macro_vote == "CHALLENGE":
            challenge_count += 1
            objections.append(macro_note)

        skeptic_vote, skeptic_notes, hard_reject = self._skeptic_vote(
            analysis, risk_reward
        )
        votes["SkepticAgent"] = {
            "vote": skeptic_vote,
            "note": "; ".join(skeptic_notes) if skeptic_notes else "No major objection",
        }
        if skeptic_vote == "CHALLENGE":
            challenge_count += 1
            objections.extend(skeptic_notes)
        else:
            approval_count += 1
            thesis.append("Skeptic found no blocking objection")

        arbiter_note = self._arbiter_note(rec, risk, risk_reward, approval_count, challenge_count)
        votes["ArbiterAgent"] = {
            "vote": "APPROVE" if not hard_reject and approval_count > challenge_count else "CHALLENGE",
            "note": arbiter_note,
        }

        return {
            "votes": votes,
            "approval_count": approval_count,
            "challenge_count": challenge_count,
            "hard_reject": hard_reject,
            "objections": self._unique_text(objections)[:5],
            "thesis": self._unique_text(thesis)[:5],
        }

    def _vote_from_signal(self, signal: Dict) -> Tuple[str, str]:
        action = str(signal.get("signal", "HOLD")).upper()
        strength = self._safe_float(signal.get("strength", 0))
        reasons = signal.get("reasons", []) or []
        note = reasons[0] if reasons else f"{action} signal at {strength:.1f}% strength"
        if action == "BUY":
            return "APPROVE", note
        if action == "SELL":
            return "CHALLENGE", note
        return "NEUTRAL", note

    def _macro_risk_vote(self, analysis: Dict) -> Tuple[str, str]:
        risk = analysis.get("risk_score", {}) or {}
        regime = analysis.get("market_regime", {}) or {}
        risk_level = risk.get("level", "MEDIUM")
        regime_name = regime.get("regime", "UNKNOWN")
        if risk_level == "HIGH" or regime_name == "EXTREME_FEAR":
            return "CHALLENGE", f"Macro/risk check is elevated ({risk_level}, {regime_name})"
        if regime.get("position_multiplier", 1.0) >= 0.75 and risk_level == "LOW":
            return "APPROVE", f"Risk backdrop is acceptable ({risk_level}, {regime_name})"
        return "NEUTRAL", f"Risk backdrop is mixed ({risk_level}, {regime_name})"

    def _skeptic_vote(self, analysis: Dict, risk_reward: float) -> Tuple[str, List[str], bool]:
        rec = analysis.get("recommendation", {}) or {}
        risk = analysis.get("risk_score", {}) or {}
        current_price = self._safe_float(analysis.get("current_price", 0))
        exit_price = self._safe_float(rec.get("take_profit", 0))
        stop_loss = self._safe_float(rec.get("stop_loss", 0))
        confidence = self._safe_float(rec.get("confidence", 0))
        objections = []
        hard_reject = False

        if current_price <= 0:
            objections.append("No usable current price")
            hard_reject = True
        if exit_price <= current_price:
            objections.append("Exit target is not above current price")
            hard_reject = True
        if stop_loss <= 0 or stop_loss >= current_price:
            objections.append("Stop loss is not below current price")
            hard_reject = True
        if risk_reward < 1.15:
            objections.append(f"Risk/reward is thin ({risk_reward:.2f}x)")
        if confidence < self.config.TOP_RECOMMENDATIONS_MIN_CONFIDENCE:
            objections.append(
                f"Confidence below floor ({confidence:.1f}% < "
                f"{self.config.TOP_RECOMMENDATIONS_MIN_CONFIDENCE:.1f}%)"
            )
        if risk.get("level") == "HIGH" and confidence < 35:
            objections.append("High risk with weak confidence")
        if rec.get("action") != "BUY":
            objections.append(f"Primary strategy says {rec.get('action', 'HOLD')}")

        return ("CHALLENGE" if objections else "APPROVE"), objections, hard_reject

    def _arbiter_note(
        self,
        rec: Dict,
        risk: Dict,
        risk_reward: float,
        approval_count: int,
        challenge_count: int,
    ) -> str:
        return (
            f"{rec.get('action', 'HOLD')} with {approval_count} approvals, "
            f"{challenge_count} challenges, {risk.get('level', 'MEDIUM')} risk, "
            f"{risk_reward:.2f}x risk/reward"
        )

    def _score_candidate(
        self,
        analysis: Dict,
        council: Dict,
        confidence: float,
        risk_reward: float,
        position_size: float,
        actionable: bool,
        history_adjustment: float = 0.0,
    ) -> float:
        risk = analysis.get("risk_score", {}) or {}
        rec = analysis.get("recommendation", {}) or {}
        max_position = max(self.config.MAX_POSITION_SIZE, 1)
        base = confidence * 0.45
        base += min(risk_reward, 3.5) * 12
        base += council.get("approval_count", 0) * 5
        base -= council.get("challenge_count", 0) * 8
        base -= self._safe_float(risk.get("score", 0)) * 3
        base += min(position_size / max_position, 1.0) * 10
        base += history_adjustment
        if rec.get("action") != "BUY":
            base -= 30
        if not actionable:
            base -= 15
        return round(max(base, 0), 2)

    def _risk_reward(self, current_price: float, stop_loss: float, exit_price: float) -> float:
        if current_price <= 0 or stop_loss <= 0 or exit_price <= current_price:
            return 0.0
        risk = current_price - stop_loss
        reward = exit_price - current_price
        if risk <= 0:
            return 0.0
        return round(reward / risk, 2)

    def _select_sector_balanced(self, ranked: List[Dict], limit: int) -> List[Dict]:
        max_per_sector = max(1, int(getattr(self.config, "TOP_RECOMMENDATIONS_MAX_PER_SECTOR", 2)))
        selected = []
        deferred = []
        sector_counts: Dict[str, int] = {}

        for candidate in ranked:
            sector = str(candidate.get("sector") or "Unknown")
            if sector != "Unknown" and sector_counts.get(sector, 0) >= max_per_sector:
                deferred.append(candidate)
                continue
            selected.append(candidate)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if len(selected) >= limit:
                return selected

        for candidate in deferred:
            selected.append(candidate)
            if len(selected) >= limit:
                break
        return selected

    def _sector_from_analysis(self, analysis: Dict) -> str:
        symbol = str(analysis.get("symbol", "") or "").upper()
        signals = analysis.get("signals", {}) or {}
        fundamentals = signals.get("fundamentals", {}) or {}
        sector = str(fundamentals.get("sector") or "").strip()
        if sector and sector.lower() not in ("unknown", "none", "nan"):
            self.learning_memory.remember_symbol_sector(symbol, sector, source="analysis")
            return sector
        return self.learning_memory.sector_for_symbol(symbol)

    def _combine_feedback(self, recent_history: Dict, retrieved: Dict) -> Dict:
        recent_history = recent_history or {}
        retrieved = retrieved or {}

        if retrieved.get("symbol_lesson_count", 0) > 0:
            base_adjustment = self._safe_float(retrieved.get("score_adjustment"))
        else:
            base_adjustment = (
                self._safe_float(recent_history.get("score_adjustment"))
                + self._safe_float(retrieved.get("score_adjustment"))
            )

        cap = abs(float(getattr(self.config, "COUNCIL_RAG_MAX_SCORE_ADJUSTMENT", 14.0)))
        score_adjustment = max(-cap, min(cap, base_adjustment))
        notes = []
        if recent_history.get("note") and retrieved.get("symbol_lesson_count", 0) == 0:
            notes.append(recent_history["note"])
        if retrieved.get("note"):
            notes.append(retrieved["note"])

        return {
            **recent_history,
            "trades": recent_history.get("trades", 0),
            "score_adjustment": round(score_adjustment, 2),
            "cooldown": bool(recent_history.get("cooldown") or retrieved.get("cooldown")),
            "note": " | ".join(notes),
            "sector": retrieved.get("sector") or recent_history.get("sector") or "Unknown",
            "retrieved_lesson_count": retrieved.get("lesson_count", 0),
            "retrieved_lessons": retrieved.get("lessons", []),
        }

    def _candidate_snapshot(self, candidates: List[Dict]) -> List[Dict]:
        snapshot = []
        for candidate in sorted(candidates, key=lambda c: c.get("council_score", 0), reverse=True):
            snapshot.append(
                {
                    "symbol": candidate.get("symbol"),
                    "sector": candidate.get("sector"),
                    "action": candidate.get("action"),
                    "actionable": candidate.get("actionable"),
                    "council_score": candidate.get("council_score"),
                    "confidence": candidate.get("confidence"),
                    "risk_reward": candidate.get("risk_reward"),
                    "risk_level": candidate.get("risk_level"),
                    "history_adjustment": candidate.get("performance_feedback", {}).get("score_adjustment", 0),
                    "retrieved_lesson_count": candidate.get("performance_feedback", {}).get("retrieved_lesson_count", 0),
                    "history_note": candidate.get("performance_feedback", {}).get("note", ""),
                    "objections": candidate.get("objections", [])[:3],
                }
            )
        return snapshot

    def _performance_feedback(self, symbol: str) -> Dict:
        stats = self._load_recent_performance().get(str(symbol or "").upper())
        if not stats:
            return {
                "trades": 0,
                "avg_pnl_pct": 0.0,
                "win_rate": 0.0,
                "score_adjustment": 0.0,
                "cooldown": False,
                "note": "",
            }
        return stats

    def _load_recent_performance(self) -> Dict[str, Dict]:
        now = datetime.now()
        if (
            self._performance_cache is not None
            and (now - self._performance_cache_time).total_seconds() < 300
        ):
            return self._performance_cache

        lookback_days = max(1, int(self.config.TOP_RECOMMENDATIONS_HISTORY_LOOKBACK_DAYS))
        min_trades = max(1, int(self.config.TOP_RECOMMENDATIONS_HISTORY_MIN_TRADES))
        since = (now - timedelta(days=lookback_days)).date().isoformat()
        by_symbol: Dict[str, List[Dict]] = {}

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                table = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = ? AND name = ?",
                    ("table", "simulated_recommendation_trades"),
                ).fetchone()
                if not table:
                    self._performance_cache = {}
                    self._performance_cache_time = now
                    return {}

                rows = conn.execute(
                    """
                    SELECT symbol, trade_date, pnl, pnl_pct, outcome
                    FROM simulated_recommendation_trades
                    WHERE pnl_pct IS NOT NULL
                      AND trade_date >= ?
                    ORDER BY trade_date ASC, updated_at ASC
                    """,
                    (since,),
                ).fetchall()
        except Exception as exc:
            self.logger.debug("Could not load recommendation performance feedback: %s", exc)
            self._performance_cache = {}
            self._performance_cache_time = now
            return {}

        for row in rows:
            symbol = str(row["symbol"] or "").upper()
            if symbol:
                by_symbol.setdefault(symbol, []).append(dict(row))

        feedback = {}
        for sym, rows in by_symbol.items():
            pnl_values = [self._safe_float(row.get("pnl_pct")) for row in rows]
            trade_count = len(pnl_values)
            if not trade_count:
                continue

            avg_pnl_pct = sum(pnl_values) / trade_count
            wins = len([value for value in pnl_values if value > 0])
            win_rate = wins / trade_count
            recent_values = pnl_values[-3:]
            recent_loss_streak = 0
            for value in reversed(pnl_values):
                if value < 0:
                    recent_loss_streak += 1
                else:
                    break

            adjustment = 0.0
            cooldown = False
            if trade_count >= min_trades:
                if avg_pnl_pct <= -1.0:
                    adjustment -= min(12.0, abs(avg_pnl_pct) * 3.0)
                if win_rate < 0.35:
                    adjustment -= min(8.0, (0.35 - win_rate) * 20.0)
                if recent_loss_streak >= 2:
                    adjustment -= min(8.0, recent_loss_streak * 2.5)
                if avg_pnl_pct >= 1.0 and win_rate >= 0.5:
                    adjustment += min(8.0, avg_pnl_pct * 2.0)
                cooldown = trade_count >= 3 and avg_pnl_pct <= -2.0 and win_rate <= 0.25
            elif pnl_values[-1] <= -3.0:
                adjustment -= 4.0

            note = ""
            if adjustment:
                note = (
                    f"Recent simulation history: {trade_count} trades, "
                    f"avg {avg_pnl_pct:+.2f}%, win rate {win_rate:.0%}, "
                    f"last3 avg {sum(recent_values) / len(recent_values):+.2f}%"
                )

            feedback[sym] = {
                "trades": trade_count,
                "avg_pnl_pct": round(avg_pnl_pct, 2),
                "win_rate": round(win_rate, 3),
                "score_adjustment": round(adjustment, 2),
                "cooldown": cooldown,
                "recent_loss_streak": recent_loss_streak,
                "note": note,
            }

        self._performance_cache = feedback
        self._performance_cache_time = now
        return feedback

    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendation_runs (
                    run_id TEXT PRIMARY KEY,
                    generated_at TEXT NOT NULL,
                    horizon TEXT NOT NULL,
                    requested_limit INTEGER NOT NULL,
                    candidate_count INTEGER NOT NULL,
                    actionable_count INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendation_picks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    horizon TEXT NOT NULL,
                    current_price REAL,
                    exit_price REAL,
                    stop_loss REAL,
                    confidence REAL,
                    council_score REAL,
                    risk_reward REAL,
                    action TEXT,
                    actionable INTEGER,
                    objections TEXT
                )
                """
            )

    def _save_run(self, result: Dict):
        try:
            payload = json.dumps(result, default=str)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO recommendation_runs
                    (run_id, generated_at, horizon, requested_limit, candidate_count,
                     actionable_count, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result["run_id"],
                        result["generated_at"],
                        result["horizon"],
                        result["limit"],
                        result["candidate_count"],
                        result["actionable_count"],
                        payload,
                    ),
                )
                for pick in result.get("recommendations", []):
                    conn.execute(
                        """
                        INSERT INTO recommendation_picks
                        (run_id, rank, symbol, horizon, current_price, exit_price,
                         stop_loss, confidence, council_score, risk_reward, action,
                         actionable, objections)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            result["run_id"],
                            pick.get("rank"),
                            pick.get("symbol"),
                            pick.get("horizon"),
                            pick.get("current_price"),
                            pick.get("exit_price"),
                            pick.get("stop_loss"),
                            pick.get("confidence"),
                            pick.get("council_score"),
                            pick.get("risk_reward"),
                            pick.get("action"),
                            1 if pick.get("actionable") else 0,
                            json.dumps(pick.get("objections", []), default=str),
                        ),
                    )
        except Exception as exc:
            self.logger.error("Could not save recommendation run: %s", exc)

    def _build_held_momentum_alerts(self, candidates: List[Dict]) -> List[Dict]:
        alerts = [
            c for c in candidates
            if c.get("already_held")
            and c.get("action") == "BUY"
            and self._safe_float(c.get("confidence")) >= 15
        ]
        alerts.sort(key=lambda c: c.get("council_score", 0), reverse=True)
        return [
            {
                "symbol": c.get("symbol"),
                "current_price": c.get("current_price"),
                "confidence": c.get("confidence"),
                "council_score": c.get("council_score"),
                "stop_loss": c.get("stop_loss"),
                "exit_price": c.get("exit_price"),
                "risk_reward": c.get("risk_reward"),
                "thesis": c.get("thesis", [])[:2],
                "objections": c.get("objections", [])[:2],
                "note": "Already held; consider add/hold/trail-stop review.",
            }
            for c in alerts[: max(self.config.HELD_MOMENTUM_ALERT_LIMIT, 0)]
        ]

    def _build_intraday_breakout_watch(
        self,
        universe: List[str],
        analyses: Dict[str, Dict],
        candidates: List[Dict],
    ) -> List[Dict]:
        data_provider = getattr(self.trading_strategy, "data_provider", None)
        if data_provider is None or self.config.INTRADAY_BREAKOUT_ALERT_LIMIT <= 0:
            return []

        candidate_by_symbol = {c.get("symbol"): c for c in candidates}
        held_symbols = set(getattr(self.portfolio_manager, "positions", {}) or {})
        alerts = []

        for symbol in universe:
            analysis = analyses.get(symbol)
            if not analysis:
                continue

            quote = data_provider.get_real_time_quote(symbol) or {}
            change_pct = self._safe_float(quote.get("change_percent"))
            if change_pct < self.config.INTRADAY_BREAKOUT_MIN_CHANGE_PCT:
                continue

            signals = analysis.get("signals", {}) or {}
            breakout = signals.get("breakout", {}) or {}
            momentum = signals.get("momentum", {}) or {}
            volume = signals.get("volume", {}) or {}
            current_price = self._safe_float(analysis.get("current_price"))
            resistance = self._safe_float(breakout.get("resistance"))
            pct_from_high = self._safe_float(breakout.get("pct_from_52w_high"))
            volume_ratio = self._safe_float(volume.get("volume_ratio"))

            broke_resistance = bool(resistance and current_price > resistance * 1.002)
            near_high = pct_from_high > -3
            strong_momentum = (
                momentum.get("signal") == "BUY"
                and self._safe_float(momentum.get("score")) >= 4
            )
            strong_price_surge = change_pct >= self.config.INTRADAY_BREAKOUT_MIN_CHANGE_PCT * 1.5
            enough_volume = (
                volume_ratio == 0
                or volume_ratio >= self.config.INTRADAY_BREAKOUT_MIN_VOLUME_RATIO
                or bool(breakout.get("volume_confirmation"))
            )

            if not (broke_resistance or near_high or strong_momentum):
                continue
            if not enough_volume and not (broke_resistance or strong_price_surge):
                continue

            candidate = candidate_by_symbol.get(symbol, {})
            score = change_pct * 4
            score += min(max(volume_ratio, 0), 4) * 6
            score += 18 if broke_resistance else 8 if near_high else 0
            score += 8 if strong_momentum else 0
            score += 6 if strong_price_surge else 0
            score += 5 if symbol in held_symbols else 0

            reasons = []
            if broke_resistance:
                reasons.append("cleared 20/30-day resistance")
            elif near_high:
                reasons.append(f"near 52-week high ({pct_from_high:.1f}%)")
            if strong_momentum:
                reasons.append("strong short-term momentum")
            elif strong_price_surge:
                reasons.append("strong price surge")
            if volume_ratio:
                reasons.append(f"volume {volume_ratio:.1f}x avg")
            if symbol in held_symbols:
                reasons.append("already held")

            alerts.append(
                {
                    "symbol": symbol,
                    "current_price": current_price,
                    "change_percent": round(change_pct, 2),
                    "volume_ratio": round(volume_ratio, 2) if volume_ratio else 0,
                    "score": round(score, 2),
                    "already_held": symbol in held_symbols,
                    "model_action": candidate.get(
                        "action",
                        analysis.get("recommendation", {}).get("action"),
                    ),
                    "model_confidence": candidate.get(
                        "confidence",
                        analysis.get("recommendation", {}).get("confidence"),
                    ),
                    "reasons": reasons[:3],
                }
            )

        alerts.sort(key=lambda item: item.get("score", 0), reverse=True)
        return alerts[: self.config.INTRADAY_BREAKOUT_ALERT_LIMIT]

    def _safe_float(self, value) -> float:
        try:
            if value is None:
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _unique_text(self, items: List[str]) -> List[str]:
        seen = set()
        unique = []
        for item in items:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            unique.append(text)
        return unique


def format_top_recommendations_message(result: Dict) -> str:
    """Format a compact WhatsApp-friendly digest."""
    generated_at = result.get("generated_at", "")
    horizon = result.get("horizon", "WEEK")
    picks = result.get("recommendations", [])
    held_momentum = result.get("held_momentum", [])
    breakout_watch = result.get("breakout_watch", [])
    requested_limit = result.get("limit", 5)
    pick_symbols = ", ".join(
        str(pick.get("symbol") or "").strip()
        for pick in picks
        if pick.get("symbol")
    )
    first_line_details = "; ".join(
        (
            f"{pick.get('symbol')} "
            f"buy ${pick.get('buy_zone', 0):.2f} "
            f"exit ${pick.get('exit_price', 0):.2f} "
            f"stop ${pick.get('stop_loss', 0):.2f}"
        )
        for pick in picks
    )
    pick_summary_lines = [
        (
            f"{pick.get('symbol')}: "
            f"now ${pick.get('current_price', 0):.2f} | "
            f"amt ${pick.get('position_size', 0):.0f} | "
            f"buy ${pick.get('buy_zone', 0):.2f} "
            f"| exit ${pick.get('exit_price', 0):.2f} "
            f"stop ${pick.get('stop_loss', 0):.2f}"
        )
        for pick in picks
    ]
    header = f"Trading council picks - {horizon} ({len(picks)}/{requested_limit}):"
    if first_line_details:
        header = f"{header} {first_line_details}"
    elif pick_symbols:
        header = f"{header} {pick_symbols}"
    lines = [
        header,
    ]
    if pick_summary_lines:
        lines.extend(pick_summary_lines)
    else:
        lines.append("none passed filters")
    lines.extend(
        [
            f"Generated: {generated_at}",
            "Mode: research signal; confirm before live orders.",
            "",
        ]
    )

    if not picks:
        lines.extend(
            [
                "No actionable BUY candidates passed the council today.",
                f"Analyzed {result.get('candidate_count', 0)} candidates.",
            ]
        )
        lines.append("")

    if held_momentum:
        lines.append("Held momentum review:")
        for item in held_momentum:
            lines.extend(
                [
                    f"{item.get('symbol')}",
                    (
                        f"Price: ${item.get('current_price', 0):.2f} | "
                        f"Conf: {item.get('confidence', 0):.1f}%"
                    ),
                    (
                        f"Exit: ${item.get('exit_price', 0):.2f} | "
                        f"Trail/stop: ${item.get('stop_loss', 0):.2f}"
                    ),
                ]
            )
        lines.append("")

    if breakout_watch:
        lines.append("Intraday breakout watch:")
        for item in breakout_watch:
            reasons = ", ".join(item.get("reasons", [])[:2])
            model_action = item.get("model_action", "HOLD")
            model_confidence = item.get("model_confidence", 0) or 0
            lines.extend(
                [
                    (
                        f"{item.get('symbol')}: "
                        f"{item.get('change_percent', 0):+.2f}% "
                        f"@ ${item.get('current_price', 0):.2f} | "
                        f"{model_action} {model_confidence:.1f}%"
                        f"{' - ' + reasons if reasons else ''}"
                    ),
                ]
            )
        lines.append("")

    lines.append("Reply in the app before placing live trades.")
    return "\n".join(lines).strip()
