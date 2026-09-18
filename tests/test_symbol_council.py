import unittest
from types import SimpleNamespace

from recommendation_engine import (
    TopRecommendationEngine,
    format_symbol_recommendation_message,
)


class SymbolCouncilTests(unittest.TestCase):
    def setUp(self):
        self.engine = TopRecommendationEngine.__new__(TopRecommendationEngine)
        self.engine.config = SimpleNamespace(TOP_RECOMMENDATIONS_MIN_CONFIDENCE=20)

    def test_directional_reward_risk_supports_buy_and_sell(self):
        self.assertEqual(self.engine._risk_reward(100, 95, 110), 2.0)
        self.assertEqual(self.engine._risk_reward(100, 105, 90), 2.0)
        self.assertEqual(self.engine._risk_reward(100, 95, 90), 0.0)

    def test_sell_signals_approve_a_sell_recommendation(self):
        sell = {"signal": "SELL", "strength": 70, "reasons": ["momentum turned down"]}
        buy = {"signal": "BUY", "strength": 60, "reasons": ["oversold bounce"]}

        self.assertEqual(self.engine._vote_from_signal(sell, "SELL")[0], "APPROVE")
        self.assertEqual(self.engine._vote_from_signal(buy, "SELL")[0], "CHALLENGE")

    def test_sell_skeptic_accepts_directional_target_and_stop(self):
        analysis = {
            "current_price": 100,
            "recommendation": {
                "action": "SELL",
                "confidence": 65,
                "take_profit": 90,
                "stop_loss": 105,
            },
            "risk_score": {"level": "MEDIUM"},
        }

        vote, objections, hard_reject = self.engine._skeptic_vote(analysis, 2.0)

        self.assertEqual(vote, "APPROVE")
        self.assertEqual(objections, [])
        self.assertFalse(hard_reject)

    def test_whatsapp_message_is_concise_and_marks_analysis_only(self):
        message = format_symbol_recommendation_message(
            {
                "symbol": "NVDA",
                "horizon": "WEEK",
                "council_action": "BUY",
                "action": "BUY",
                "current_price": 100,
                "exit_price": 110,
                "stop_loss": 95,
                "confidence": 72,
                "risk_reward": 2,
                "approval_count": 6,
                "challenge_count": 2,
                "thesis": ["positive momentum"],
                "objections": ["earnings risk"],
            }
        )

        self.assertIn("COUNCIL BUY: NVDA (WEEK)", message)
        self.assertIn("reward/risk 2.00x", message)
        self.assertIn("no trade was placed", message)


if __name__ == "__main__":
    unittest.main()
