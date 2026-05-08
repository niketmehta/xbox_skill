import argparse
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import schedule

from config import Config
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


def send_digest_once() -> bool:
    config = Config()
    logger = logging.getLogger("daily_digest_runner")

    if not config.TOP_RECOMMENDATIONS_ENABLED:
        logger.warning("Top recommendations digest is disabled")
        return False

    logger.info(
        "Generating scheduled top recommendations digest [%s]",
        config.TOP_RECOMMENDATIONS_HORIZON,
    )
    agent = TradingAgent()
    result = agent.send_top_recommendations_whatsapp(
        horizon=config.TOP_RECOMMENDATIONS_HORIZON,
        limit=5,
        universe_size=50,
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

    simulator = TradeSimulationEngine()
    if dry_run:
        summary = simulator.build_eod_summary()
        print(format_simulation_summary_message(summary, label=label))
        return summary.get("trade_count", 0) > 0

    summary = simulator.send_eod_summary_whatsapp(label=label)
    sent = bool(summary.get("delivery", {}).get("sent"))
    logger.info(
        "%s simulation summary sent=%s trades=%s total_pnl=%.2f",
        label,
        sent,
        summary.get("trade_count", 0),
        summary.get("total_pnl", 0),
    )
    if not sent:
        logger.error("EOD simulation delivery failed: %s", summary.get("delivery", {}).get("error"))
    return sent


def run_daemon():
    config = Config()
    logger = logging.getLogger("daily_digest_runner")
    schedule.every().day.at(config.TOP_RECOMMENDATIONS_TIME).do(send_digest_once)
    if config.SIMULATION_ENABLED:
        schedule.every().day.at(config.SIMULATION_OPEN_TIME).do(capture_open_simulation_once)
        schedule.every().day.at(config.SIMULATION_MIDDAY_TIME).do(
            send_eod_summary_once,
            label="MIDDAY",
        )
        schedule.every().day.at(config.SIMULATION_EOD_TIME).do(
            send_eod_summary_once,
            label="EOD",
        )
    logger.info(
        "Daily digest scheduler running at %s every day",
        config.TOP_RECOMMENDATIONS_TIME,
    )
    if config.SIMULATION_ENABLED:
        logger.info(
            "Simulation scheduler running at open=%s, midday=%s, and eod=%s every day",
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
