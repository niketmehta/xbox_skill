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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import Config


class TopRecommendationEngine:
    """Builds auditable top-pick recommendations from the existing strategy stack."""

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
        errors = []

        for symbol in universe:
            try:
                analysis = self.trading_strategy.analyze_stock(symbol, horizon=horizon)
                candidate = self._build_candidate(analysis)
                if candidate:
                    candidates.append(candidate)
            except Exception as exc:
                self.logger.error("Council analysis failed for %s: %s", symbol, exc)
                errors.append({"symbol": symbol, "error": str(exc)})

        actionable = [c for c in candidates if c["actionable"]]
        ranked = sorted(actionable, key=lambda c: c["council_score"], reverse=True)
        top = ranked[:limit]

        result = {
            "run_id": run_id,
            "generated_at": generated_at.isoformat(),
            "horizon": horizon,
            "limit": limit,
            "universe_size": len(universe),
            "candidate_count": len(candidates),
            "actionable_count": len(actionable),
            "council_agents": self.COUNCIL_AGENTS,
            "recommendations": [
                {**pick, "rank": rank}
                for rank, pick in enumerate(top, start=1)
            ],
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
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            universe.append(symbol)
            if len(universe) >= universe_size:
                break
        return universe

    def _build_candidate(self, analysis: Dict) -> Optional[Dict]:
        if not analysis:
            return None

        symbol = analysis.get("symbol", "")
        horizon = analysis.get("horizon", "WEEK")
        rec = analysis.get("recommendation", {}) or {}
        risk = analysis.get("risk_score", {}) or {}
        current_price = self._safe_float(analysis.get("current_price", 0))
        exit_price = self._safe_float(rec.get("take_profit", 0))
        stop_loss = self._safe_float(rec.get("stop_loss", 0))
        confidence = self._safe_float(rec.get("confidence", 0))
        position_size = self._safe_float(rec.get("position_size", 0))
        action = str(rec.get("action", "HOLD")).upper()
        risk_reward = self._risk_reward(current_price, stop_loss, exit_price)
        council = self._run_council(analysis, risk_reward)
        objections = council.get("objections", [])

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
            and not already_held
            and not council.get("hard_reject", False)
        )
        council_score = self._score_candidate(
            analysis=analysis,
            council=council,
            confidence=confidence,
            risk_reward=risk_reward,
            position_size=position_size,
            actionable=actionable,
        )

        return {
            "symbol": symbol,
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
            "objections": objections,
            "thesis": council.get("thesis", []),
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
    requested_limit = result.get("limit", 5)
    pick_summary = "; ".join(
        (
            f"{pick.get('rank')}. {pick.get('symbol')} "
            f"buy ${pick.get('buy_zone', 0):.2f} "
            f"exit ${pick.get('exit_price', 0):.2f} "
            f"stop ${pick.get('stop_loss', 0):.2f}"
        )
        for pick in picks
    )
    lines = [
        (
            f"Trading council picks - {horizon} ({len(picks)}/{requested_limit}): "
            f"{pick_summary if pick_summary else 'none passed filters'}"
        ),
        f"Generated: {generated_at}",
        "Mode: research signal; confirm before live orders.",
        "",
    ]

    if not picks:
        lines.extend(
            [
                "No actionable BUY candidates passed the council today.",
                f"Analyzed {result.get('candidate_count', 0)} candidates.",
            ]
        )
        return "\n".join(lines)

    for pick in picks:
        objections = pick.get("objections", [])
        thesis = pick.get("thesis", [])
        lines.extend(
            [
                (
                    f"{pick.get('rank')}. {pick.get('symbol')} "
                    f"score {pick.get('council_score', 0):.1f}"
                ),
                f"Current: ${pick.get('current_price', 0):.2f}",
                f"Buy zone: ${pick.get('buy_zone', 0):.2f}",
                f"Exit target: ${pick.get('exit_price', 0):.2f}",
                f"Stop: ${pick.get('stop_loss', 0):.2f}",
                (
                    f"Confidence: {pick.get('confidence', 0):.1f}% | "
                    f"R/R: {pick.get('risk_reward', 0):.2f}x | "
                    f"Risk: {pick.get('risk_level', 'MEDIUM')}"
                ),
                f"Why: {thesis[0] if thesis else 'Council approval outweighed challenges'}",
            ]
        )
        if objections:
            lines.append(f"Challenge: {objections[0]}")
        lines.append("")

    lines.append("Reply in the app before placing live trades.")
    return "\n".join(lines).strip()
