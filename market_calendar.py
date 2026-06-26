import logging
from datetime import date, datetime, timedelta
from typing import Dict, Optional

import pytz

from config import Config

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    tradeapi = None


class MarketCalendar:
    """US equities market-session guard for scheduled jobs."""

    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.eastern_tz = pytz.timezone("US/Eastern")
        self._alpaca_api = None

    def today_eastern(self) -> date:
        return datetime.now(self.eastern_tz).date()

    def is_trading_day(self, check_date: Optional[date] = None) -> bool:
        check_date = check_date or self.today_eastern()
        session = self.get_session(check_date)
        return bool(session.get("is_trading_day"))

    def get_session(self, check_date: Optional[date] = None) -> Dict:
        check_date = check_date or self.today_eastern()
        iso_date = check_date.isoformat()

        alpaca_session = self._alpaca_session(check_date)
        if alpaca_session is not None:
            return alpaca_session

        fallback_open = self._fallback_is_trading_day(check_date)
        return {
            "date": iso_date,
            "is_trading_day": fallback_open,
            "source": "fallback_us_equities_calendar",
            "reason": self._fallback_reason(check_date, fallback_open),
        }

    def _alpaca_session(self, check_date: date) -> Optional[Dict]:
        api = self._get_alpaca_api()
        if api is None:
            return None

        iso_date = check_date.isoformat()
        try:
            sessions = api.get_calendar(start=iso_date, end=iso_date)
            if not sessions:
                return {
                    "date": iso_date,
                    "is_trading_day": False,
                    "source": "alpaca_calendar",
                    "reason": "No US equities market session",
                }

            session = sessions[0]
            return {
                "date": iso_date,
                "is_trading_day": True,
                "open": str(getattr(session, "open", "")),
                "close": str(getattr(session, "close", "")),
                "source": "alpaca_calendar",
                "reason": "US equities market session",
            }
        except Exception as exc:
            self.logger.warning("Alpaca calendar unavailable for %s: %s", iso_date, exc)
            return None

    def _get_alpaca_api(self):
        if self._alpaca_api is not None:
            return self._alpaca_api
        if tradeapi is None:
            return None
        if not self.config.ALPACA_API_KEY or not self.config.ALPACA_SECRET_KEY:
            return None

        self._alpaca_api = tradeapi.REST(
            key_id=self.config.ALPACA_API_KEY,
            secret_key=self.config.ALPACA_SECRET_KEY,
            base_url=self.config.ALPACA_BASE_URL,
            api_version="v2",
        )
        return self._alpaca_api

    def _fallback_is_trading_day(self, check_date: date) -> bool:
        if check_date.weekday() >= 5:
            return False
        return check_date not in self._fallback_market_holidays(check_date.year)

    def _fallback_reason(self, check_date: date, is_trading_day: bool) -> str:
        if is_trading_day:
            return "No weekend or common US equities holiday found in fallback calendar"
        if check_date.weekday() >= 5:
            return "Weekend"
        return "Common US equities market holiday"

    def _fallback_market_holidays(self, year: int) -> set:
        holidays = {
            self._observed(date(year, 1, 1)),   # New Year's Day
            self._nth_weekday(year, 1, 0, 3),   # Martin Luther King Jr. Day
            self._nth_weekday(year, 2, 0, 3),   # Presidents Day
            self._good_friday(year),
            self._last_weekday(year, 5, 0),     # Memorial Day
            self._observed(date(year, 6, 19)),  # Juneteenth
            self._observed(date(year, 7, 4)),   # Independence Day
            self._nth_weekday(year, 9, 0, 1),   # Labor Day
            self._nth_weekday(year, 11, 3, 4),  # Thanksgiving
            self._observed(date(year, 12, 25)), # Christmas
        }
        return {holiday for holiday in holidays if holiday.year == year}

    def _observed(self, holiday: date) -> date:
        if holiday.weekday() == 5:
            return holiday - timedelta(days=1)
        if holiday.weekday() == 6:
            return holiday + timedelta(days=1)
        return holiday

    def _nth_weekday(self, year: int, month: int, weekday: int, nth: int) -> date:
        current = date(year, month, 1)
        while current.weekday() != weekday:
            current += timedelta(days=1)
        return current + timedelta(days=7 * (nth - 1))

    def _last_weekday(self, year: int, month: int, weekday: int) -> date:
        if month == 12:
            current = date(year, 12, 31)
        else:
            current = date(year, month + 1, 1) - timedelta(days=1)
        while current.weekday() != weekday:
            current -= timedelta(days=1)
        return current

    def _good_friday(self, year: int) -> date:
        return self._easter_sunday(year) - timedelta(days=2)

    def _easter_sunday(self, year: int) -> date:
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        return date(year, month, day)
