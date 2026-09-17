import argparse
import logging
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import schedule

from config import Config
from market_calendar import MarketCalendar
from trading_agent import TradingAgent
from trade_simulator import (
    TradeSimulationEngine,
    format_entry_alert_message,
    format_open_capture_message,
    format_period_summary_message,
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


def monitor_entry_alerts_once(historical: bool = False) -> bool:
    config = Config()
    logger = logging.getLogger("daily_digest_runner")

    if not config.ENTRY_ALERTS_ENABLED:
        logger.info("Entry-alert monitor is disabled")
        return True
    if not should_run_market_job("intraday entry-alert monitor"):
        return True

    simulator = TradeSimulationEngine()
    window = simulator.entry_alert_window_status()
    if not historical and not window.get("is_open"):
        logger.info("Entry-alert monitor idle: %s", window.get("reason"))
        return True

    result = simulator.capture_entry_alerts(
        top_n=config.SIMULATION_TOP_N,
        max_alerts=config.ENTRY_ALERT_MAX_ALERTS_PER_SCAN,
        historical=historical,
    )
    if result.get("error") in {
        "No recommendation run found for today",
        "No current or historical recommendation candidates found",
    }:
        logger.warning("No eligible recommendation candidates found; generating digest before entry scan")
        if send_digest_once():
            simulator = TradeSimulationEngine()
            result = simulator.capture_entry_alerts(
                top_n=config.SIMULATION_TOP_N,
                max_alerts=config.ENTRY_ALERT_MAX_ALERTS_PER_SCAN,
                historical=historical,
            )

    logger.info(
        "Entry-alert scan captured %s/%s alerts; skipped_existing=%s",
        result.get("captured", 0),
        result.get("requested", 0),
        result.get("skipped_existing", []),
    )
    if result.get("errors"):
        logger.warning("Entry-alert scan errors: %s", result.get("errors"))

    if result.get("captured", 0) <= 0:
        return True

    if config.ENTRY_ALERT_WHATSAPP_ENABLED:
        result = simulator.send_entry_alerts_whatsapp(result)
        sent = bool(result.get("delivery", {}).get("sent"))
        logger.info("Entry-alert WhatsApp sent=%s", sent)
        if not sent:
            logger.error(
                "Entry-alert WhatsApp failed: %s",
                result.get("delivery", {}).get("error"),
            )
        return sent

    return True


def run_entry_alert_window() -> bool:
    config = Config()
    logger = logging.getLogger("daily_digest_runner")
    interval_seconds = max(1, int(config.ENTRY_ALERT_SCAN_INTERVAL_MINUTES)) * 60
    end_at = _local_datetime_for_time(config.ENTRY_ALERT_END_TIME)

    if datetime.now() > end_at:
        logger.info("Entry-alert window already ended at %s", config.ENTRY_ALERT_END_TIME)
        return True

    logger.info(
        "Entry-alert window monitor running until %s every %s minutes",
        config.ENTRY_ALERT_END_TIME,
        config.ENTRY_ALERT_SCAN_INTERVAL_MINUTES,
    )
    ok = True
    while datetime.now() <= end_at:
        ok = monitor_entry_alerts_once() and ok
        remaining_seconds = max(0, (end_at - datetime.now()).total_seconds())
        if remaining_seconds <= 0:
            break
        time.sleep(min(interval_seconds, remaining_seconds))
    return ok


def _local_datetime_for_time(value: str) -> datetime:
    parsed_time = datetime.strptime(value, "%H:%M").time()
    return datetime.combine(datetime.now().date(), parsed_time)


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
    period_ok = True
    if label.upper() == "EOD" and config.PERIOD_SUMMARIES_ENABLED:
        calendar = MarketCalendar()
        today = calendar.today_eastern()
        periods = []
        if calendar.is_last_trading_day_of_week(today):
            periods.append("WEEK")
        if calendar.is_last_trading_day_of_month(today):
            periods.append("MONTH")
        for period in periods:
            period_ok = send_period_summary_once(period) and period_ok
    return sent and period_ok


def send_period_summary_once(period: str, dry_run: bool = False) -> bool:
    config = Config()
    logger = logging.getLogger("daily_digest_runner")
    if not config.SIMULATION_ENABLED or not config.PERIOD_SUMMARIES_ENABLED:
        logger.info("Period summaries are disabled")
        return True

    simulator = TradeSimulationEngine()
    if dry_run:
        summary = simulator.build_period_summary(period)
        print(format_period_summary_message(summary))
        return True

    summary = simulator.send_period_summary_whatsapp(period)
    sent = bool(summary.get("delivery", {}).get("sent"))
    logger.info(
        "%s simulation summary sent=%s trades=%s total_pnl=%.2f",
        period.upper(),
        sent,
        summary.get("trade_count", 0),
        summary.get("total_pnl", 0),
    )
    if not sent:
        logger.error(
            "%s simulation summary delivery failed: %s",
            period.upper(),
            summary.get("delivery", {}).get("error"),
        )
    return sent


def run_daemon():
    config = Config()
    logger = logging.getLogger("daily_digest_runner")
    schedule_weekdays(config.TOP_RECOMMENDATIONS_TIME, send_digest_once)
    if config.SIMULATION_ENABLED:
        if config.ENTRY_ALERTS_ENABLED:
            schedule.every(max(1, config.ENTRY_ALERT_SCAN_INTERVAL_MINUTES)).minutes.do(
                monitor_entry_alerts_once
            )
        else:
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
        if config.ENTRY_ALERTS_ENABLED:
            logger.info(
                "Entry-alert monitor scanning every %s minutes after open+%s min until close-%s min; summaries at midday=%s and eod=%s",
                config.ENTRY_ALERT_SCAN_INTERVAL_MINUTES,
                config.ENTRY_ALERT_SKIP_OPEN_MINUTES,
                config.ENTRY_ALERT_SKIP_CLOSE_MINUTES,
                config.SIMULATION_MIDDAY_TIME,
                config.SIMULATION_EOD_TIME,
            )
        else:
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
        "--monitor-entry-alerts",
        action="store_true",
        help="Scan top recommendations for intraday dip-entry alerts and exit.",
    )
    parser.add_argument(
        "--run-window",
        action="store_true",
        help="Keep scanning entry alerts until ENTRY_ALERT_END_TIME.",
    )
    parser.add_argument(
        "--historical-entry-backfill",
        action="store_true",
        help="Score the full entry window instead of only the latest candle.",
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
        "--send-weekly-summary",
        action="store_true",
        help="Send the current calendar-week simulated P/L summary and exit.",
    )
    parser.add_argument(
        "--send-monthly-summary",
        action="store_true",
        help="Send the current calendar-month simulated P/L summary and exit.",
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
    parser.add_argument(
        "--print-entry-alert-message",
        action="store_true",
        help="Print the latest entry-alert message shape and exit.",
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
    if args.print_entry_alert_message:
        simulator = TradeSimulationEngine()
        capture = simulator.capture_entry_alerts(
            top_n=Config.SIMULATION_TOP_N,
            historical=args.historical_entry_backfill,
        )
        print(format_entry_alert_message(capture))
        raise SystemExit(0 if capture.get("captured", 0) > 0 else 1)
    if args.capture_open:
        raise SystemExit(0 if capture_open_simulation_once() else 1)
    if args.monitor_entry_alerts:
        if args.run_window:
            raise SystemExit(0 if run_entry_alert_window() else 1)
        raise SystemExit(
            0 if monitor_entry_alerts_once(historical=args.historical_entry_backfill) else 1
        )
    if args.send_midday_summary:
        raise SystemExit(0 if send_eod_summary_once(dry_run=args.dry_run, label="MIDDAY") else 1)
    if args.send_weekly_summary:
        raise SystemExit(0 if send_period_summary_once("WEEK", dry_run=args.dry_run) else 1)
    if args.send_monthly_summary:
        raise SystemExit(0 if send_period_summary_once("MONTH", dry_run=args.dry_run) else 1)
    if args.send_eod_summary:
        raise SystemExit(0 if send_eod_summary_once(dry_run=args.dry_run, label="EOD") else 1)
    run_daemon()


if __name__ == "__main__":
    main()
