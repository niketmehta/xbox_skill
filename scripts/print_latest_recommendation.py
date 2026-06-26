import json
import sqlite3
from pathlib import Path


DB_PATH = Path("trading_data.db")


def main():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT payload_json
            FROM recommendation_runs
            ORDER BY generated_at DESC
            LIMIT 1
            """
        ).fetchone()

    if not row:
        print("No recommendation run found.")
        return

    result = json.loads(row["payload_json"])
    print(f"Generated: {result.get('generated_at')}")
    print(
        "Universe: {universe_size} | Candidates: {candidate_count} | Actionable: {actionable_count}".format(
            universe_size=result.get("universe_size", 0),
            candidate_count=result.get("candidate_count", 0),
            actionable_count=result.get("actionable_count", 0),
        )
    )
    print()
    print("Top picks")
    for pick in result.get("recommendations", []):
        feedback = pick.get("performance_feedback") or {}
        thesis = " | ".join((pick.get("thesis") or [])[:2])
        challenge = ((pick.get("objections") or [""])[0])
        print(
            "#{rank} {symbol} ({sector}) score={score:.2f} conf={confidence:.1f}% "
            "R/R={risk_reward:.2f} price=${price:.2f} target=${target:.2f} "
            "stop=${stop:.2f} RAG={rag:+.2f} risk={risk}".format(
                rank=pick.get("rank"),
                symbol=pick.get("symbol"),
                sector=pick.get("sector", "Unknown"),
                score=pick.get("council_score", 0),
                confidence=pick.get("confidence", 0),
                risk_reward=pick.get("risk_reward", 0),
                price=pick.get("current_price", 0),
                target=pick.get("exit_price", 0),
                stop=pick.get("stop_loss", 0),
                rag=feedback.get("score_adjustment", 0),
                risk=pick.get("risk_level", "UNKNOWN"),
            )
        )
        if thesis:
            print(f"  Why: {thesis}")
        if challenge:
            print(f"  Challenge: {challenge}")

    breakout_watch = result.get("breakout_watch") or []
    if breakout_watch:
        print()
        print("Breakout watch")
        for item in breakout_watch:
            print(
                "{symbol} move={move:+.2f}% @ ${price:.2f} model={model} "
                "conf={confidence:.1f}% reasons={reasons}".format(
                    symbol=item.get("symbol"),
                    move=item.get("change_percent", 0),
                    price=item.get("current_price", 0),
                    model=item.get("model_action", "HOLD"),
                    confidence=float(item.get("model_confidence") or 0),
                    reasons=", ".join(item.get("reasons", [])),
                )
            )


if __name__ == "__main__":
    main()
