import argparse
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import schedule

from config import Config
from market_calendar import MarketCalendar
from trading_agent import TradingAgent
from trade_simulator import (
    TradeSimulationEngine,
    format_open_capture_message,
    format_simulation_summary_message,
)


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "daily_digest.log"


def configure_logging():
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            RotatingFileHandler(
                LOG_FILE,
                maxBytes=1_000_000,
                backupCount=3,
                encoding="utf-8",
            ),
            logging.StreamHandler(),
        ],
    )


def should_run_market_job(job_name: str) -> bool:
    session = MarketCalendar().get_session()
    if session.get("is_trading_day"):
        return True

    logging.getLogger("daily_digest_runner").info(
        "Skipping %s: market is closed on %s (%s via %s)",
        job_name,
        session.get("date", "today"),
        session.get("reason", "no market session"),
        session.get("source", "unknown"),
    )
    return False


def schedule_weekdays(run_time: str, job_func, *args, **kwargs):
    for weekday in (
        schedule.every().monday,
        schedule.every().tuesday,
        schedule.every().wednesday,
        schedule.every().thursday,
        schedule.every().friday,
    ):
        weekday.at(run_time).do(job_func, *args, **kwargs)


def send_digest_once() -> bool:
    config = Config()
    logger = logging.getLogger("daily_digest_runner")

    if not config.TOP_RECOMMENDATIONS_ENABLED:
        logger.warning("Top recommendations digest is disabled")
        return False
    if not should_run_market_job("top recommendations digest"):
        return True

    logger.info(
        "Generating scheduled top recommendations digest [%s]",
        config.TOP_RECOMMENDATIONS_HORIZON,
    )
    agent = TradingAgent()
    result = agent.send_top_recommendations_whatsapp(
        horizon=config.TOP_RECOMMENDATIONS_HORIZON,
        limit=5,
        universe_size=config.TOP_RECOMMENDATIONS_UNIVERSE_SIZE,
    )
    sent = bool(result.get("delivery", {}).get("sent"))
    symbols = [pick.get("symbol") for pick in result.get("recommendations", [])]
    logger.info("Digest sent=%s symbols=%s", sent, symbols)
    if not sent:
        logger.error("Digest delivery failed: %s", result.get("delivery", {}).get("error"))
    return sent


def capture_open_simulation_once() -> bool:
    config = Config()
    logger = logging.getLogger("daily_digest_runner")

    if not config.SIMULATION_ENABLED:
        logger.warning("Recommendation simulation is disabled")
        return False
    if not should_run_market_job("open simulation capture"):
        return True

    simulator = TradeSimulationEngine()
    result = simulator.capture_open_trades(top_n=config.SIMULATION_TOP_N)
    if result.get("error") == "No recommendation run found for today":
        logger.warning("No recommendation run found for today; generating digest before capture")
        if send_digest_once():
            simulator = TradeSimulationEngine()
            result = simulator.capture_open_trades(top_n=config.SIMULATION_TOP_N)

    logger.info(
        "Open simulation captured %s/%s trades",
        result.get("captured", 0),
        result.get("requested", 0),
    )
    if result.get("errors"):
        logger.warning("Open simulation errors: %s", result.get("errors"))

    if config.SIMULATION_OPEN_WHATSAPP_ENABLED and result.get("captured", 0) > 0:
        result = simulator.send_open_capture_whatsapp(result)
        sent = bool(result.get("delivery", {}).get("sent"))
        logger.info("Open simulation WhatsApp sent=%s", sent)
        if not sent:
            logger.error(
                "Open simulation WhatsApp failed: %s",
                result.get("delivery", {}).get("error"),
            )
        return sent

    return result.get("captured", 0) > 0


def send_eod_summary_once(dry_run: bool = False, label: str = "EOD") -> bool:
    config = Config()
    logger = logging.getLogger("daily_digest_runner")

    if not config.SIMULATION_ENABLED:
        logger.warning("Recommendation simulation is disabled")
        return False
    if not dry_run and not should_run_market_job(f"{label} simulation summary"):
        return True

    simulator = TradeSimulationEngine()
    if dry_run:
        summary = simulator.build_eod_summary(
            backfill_missing=config.SIMULATION_BACKFILL_ON_SUMMARY,
            top_n=config.SIMULATION_TOP_N,
        )
        print(format_simulation_summary_message(summary, label=label))
        return summary.get("trade_count", 0) > 0

    summary = simulator.send_eod_summary_whatsapp(label=label)
    backfill = summary.get("backfill") or {}
    if backfill:
        logger.info(
            "%s simulation backfill captured %s/%s trades",
            label,
            backfill.get("captured", 0),
            backfill.get("requested", 0),
        )
        if backfill.get("errors"):
            logger.warning("%s simulation backfill errors: %s", label, backfill.get("errors"))
    sent = bool(summary.get("delivery", {}).get("sent"))
    logger.info(
        "%s simulation summary sent=%s trades=%s total_pnl=%.2f",
        label,
        sent,
        summary.get("trade_count", 0),
        summary.get("total_pnl", 0),
    )
    if not sent:
        logger.error(
            "%s simulation delivery failed: %s",
            label,
            summary.get("delivery", {}).get("error"),
        )
    return sent


def run_daemon():
    config = Config()
    logger = logging.getLogger("daily_digest_runner")
    schedule_weekdays(config.TOP_RECOMMENDATIONS_TIME, send_digest_once)
    if config.SIMULATION_ENABLED:
        schedule_weekdays(config.SIMULATION_OPEN_TIME, capture_open_simulation_once)
        schedule_weekdays(
            config.SIMULATION_MIDDAY_TIME,
            send_eod_summary_once,
            label="MIDDAY",
        )
        schedule_weekdays(
            config.SIMULATION_EOD_TIME,
            send_eod_summary_once,
            label="EOD",
        )
    logger.info(
        "Daily digest scheduler running at %s on weekdays; market holidays are skipped",
        config.TOP_RECOMMENDATIONS_TIME,
    )
    if config.SIMULATION_ENABLED:
        logger.info(
            "Simulation scheduler running at open=%s, midday=%s, and eod=%s on weekdays; market holidays are skipped",
            config.SIMULATION_OPEN_TIME,
            config.SIMULATION_MIDDAY_TIME,
            config.SIMULATION_EOD_TIME,
        )

    while True:
        schedule.run_pending()
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description="Run the trading council WhatsApp digest.")
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Send one digest immediately and exit.",
    )
    parser.add_argument(
        "--capture-open",
        action="store_true",
        help="Capture simulated entries for the latest recommendations and exit.",
    )
    parser.add_argument(
        "--send-eod-summary",
        action="store_true",
        help="Send the simulated end-of-day P&L WhatsApp summary and exit.",
    )
    parser.add_argument(
        "--send-midday-summary",
        action="store_true",
        help="Send the simulated midday P&L WhatsApp summary and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the end-of-day summary instead of sending WhatsApp.",
    )
    parser.add_argument(
        "--print-open-message",
        action="store_true",
        help="Print the latest open capture message shape and exit.",
    )
    args = parser.parse_args()

    configure_logging()
    if args.run_once:
        raise SystemExit(0 if send_digest_once() else 1)
    if args.print_open_message:
        simulator = TradeSimulationEngine()
        capture = simulator.capture_open_trades(top_n=Config.SIMULATION_TOP_N)
        print(format_open_capture_message(capture))
        raise SystemExit(0 if capture.get("captured", 0) > 0 else 1)
    if args.capture_open:
        raise SystemExit(0 if capture_open_simulation_once() else 1)
    if args.send_midday_summary:
        raise SystemExit(0 if send_eod_summary_once(dry_run=args.dry_run, label="MIDDAY") else 1)
    if args.send_eod_summary:
        raise SystemExit(0 if send_eod_summary_once(dry_run=args.dry_run, label="EOD") else 1)
    run_daemon()


if __name__ == "__main__":
    main()
