"""
backtest/calendar.py
NEPSE-specific trading calendar.

CRITICAL: NEPSE trades Sunday–Thursday (NOT Monday–Friday).
Saturday and Friday are weekends in Nepal.
This is the most common bug for non-Nepali developers.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

# ---------------------------------------------------------------------------
# Market hours (Nepal Standard Time = UTC+5:45)
# ---------------------------------------------------------------------------

NEPSE_MARKET_HOURS = {
    "timezone": "Asia/Kathmandu",
    "open_time": "11:00",
    "close_time": "15:00",
    "trading_days": ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"],
    # NOTE: weekday() in Python: 0=Mon,1=Tue,2=Wed,3=Thu,4=Fri,5=Sat,6=Sun
    # NEPSE trading days in Python weekday(): 6=Sun, 0=Mon, 1=Tue, 2=Wed, 3=Thu
    "_python_trading_weekdays": {6, 0, 1, 2, 3},  # Sun=6, Mon-Thu=0-3
}

# ---------------------------------------------------------------------------
# NEPSE Public Holidays
# Sources: Official NEPSE circulars + NRB calendar
# ---------------------------------------------------------------------------

NEPSE_HOLIDAYS_2022: set[str] = {
    "2022-01-01",  # New Year (Western)
    "2022-01-17",  # Prithvi Jayanti
    "2022-02-19",  # Democracy Day (Prajatantra Diwas)
    "2022-03-01",  # Maha Shivaratri
    "2022-03-18",  # Ghode Jatra
    "2022-04-14",  # New Year (Nepal Sambat / Navabarsha)
    "2022-05-03",  # Labour Day
    "2022-05-16",  # Buddha Jayanti
    "2022-05-29",  # Republic Day
    "2022-09-09",  # Indra Jatra
    "2022-10-04",  # Fulpati (Dashain)
    "2022-10-05",  # Maha Asthami (Dashain)
    "2022-10-06",  # Maha Nawami (Dashain)
    "2022-10-07",  # Bijaya Dashami (Dashain — main)
    "2022-10-08",  # Ekadashi
    "2022-10-09",  # Dashain holiday
    "2022-10-10",  # Dashain holiday
    "2022-10-24",  # Laxmi Puja (Tihar)
    "2022-10-26",  # Govardhan Puja
    "2022-11-08",  # Chhath Puja
    "2022-12-29",  # Tamu Lhosar (approx)
}

NEPSE_HOLIDAYS_2023: set[str] = {
    "2023-01-01",  # New Year
    "2023-01-15",  # Maghe Sankranti
    "2023-01-16",  # Prithvi Jayanti
    "2023-02-18",  # Democracy Day
    "2023-02-18",  # Maha Shivaratri
    "2023-03-08",  # Maha Shivaratri (approx)
    "2023-04-14",  # Nava Barsha
    "2023-05-05",  # Buddha Jayanti
    "2023-05-29",  # Republic Day
    "2023-09-20",  # Indra Jatra (approx)
    "2023-10-20",  # Fulpati (Dashain)
    "2023-10-23",  # Vijaya Dashami
    "2023-10-24",  # Dashain holiday
    "2023-11-12",  # Laxmi Puja (Tihar)
    "2023-11-13",  # Gobardhan Puja
    "2023-11-14",  # Bhai Tika
    "2023-11-28",  # Chhath Puja (approx)
}

NEPSE_HOLIDAYS_2024: set[str] = {
    "2024-01-01",  # New Year
    "2024-01-15",  # Maghe Sankranti
    "2024-02-19",  # Democracy Day
    "2024-03-08",  # Maha Shivaratri (approx)
    "2024-04-14",  # Nava Barsha
    "2024-04-23",  # Buddha Jayanti (approx)
    "2024-05-29",  # Republic Day
    "2024-10-12",  # Fulpati (Dashain, approx)
    "2024-10-13",  # Asthami
    "2024-10-14",  # Nawami
    "2024-10-15",  # Vijaya Dashami
    "2024-11-01",  # Laxmi Puja (Tihar, approx)
    "2024-11-02",  # Gobardhan Puja
    "2024-11-03",  # Bhai Tika
    "2024-11-17",  # Chhath Puja (approx)
}

NEPSE_HOLIDAYS_2021: set[str] = {
    "2021-01-01",  # New Year
    "2021-01-16",  # Prithvi Jayanti
    "2021-02-19",  # Democracy Day
    "2021-03-11",  # Maha Shivaratri
    "2021-03-29",  # Ghode Jatra
    "2021-04-14",  # Nava Barsha
    "2021-05-01",  # Labour Day
    "2021-05-26",  # Buddha Jayanti
    "2021-05-29",  # Republic Day
    "2021-10-14",  # Fulpati
    "2021-10-15",  # Asthami
    "2021-10-16",  # Nawami
    "2021-10-17",  # Vijaya Dashami
    "2021-11-04",  # Laxmi Puja
    "2021-11-05",  # Gobardhan Puja
    "2021-11-06",  # Bhai Tika
}

# Combined holidays lookup
_ALL_HOLIDAYS: set[str] = (
    NEPSE_HOLIDAYS_2021
    | NEPSE_HOLIDAYS_2022
    | NEPSE_HOLIDAYS_2023
    | NEPSE_HOLIDAYS_2024
)

# Python weekday numbers that are NEPSE trading days
# 0=Mon, 1=Tue, 2=Wed, 3=Thu, 6=Sun
_TRADING_WEEKDAYS = {0, 1, 2, 3, 6}


# ---------------------------------------------------------------------------
# Core calendar functions
# ---------------------------------------------------------------------------

def is_trading_day(d: str | date) -> bool:
    """
    Returns True if NEPSE was open on this date.
    NEPSE trades Sunday through Thursday.
    Returns False for Friday (4), Saturday (5), and public holidays.
    """
    if isinstance(d, str):
        d = date.fromisoformat(d)
    if d.weekday() not in _TRADING_WEEKDAYS:
        return False
    return d.isoformat() not in _ALL_HOLIDAYS


def get_next_trading_day(d: str | date) -> date:
    """
    Returns the next date NEPSE was/will be open.
    Used for: entry price (signal on day T → execute at open on T+1).
    """
    if isinstance(d, str):
        d = date.fromisoformat(d)
    candidate = d + timedelta(days=1)
    for _ in range(14):  # max 14-day holiday streak unlikely
        if is_trading_day(candidate):
            return candidate
        candidate += timedelta(days=1)
    raise ValueError(f"Could not find next trading day within 14 days of {d}")


def get_prev_trading_day(d: str | date) -> date:
    """Returns the most recent trading day before d."""
    if isinstance(d, str):
        d = date.fromisoformat(d)
    candidate = d - timedelta(days=1)
    for _ in range(14):
        if is_trading_day(candidate):
            return candidate
        candidate -= timedelta(days=1)
    raise ValueError(f"Could not find prev trading day within 14 days of {d}")


def get_trading_days(start: str | date, end: str | date) -> list[date]:
    """
    Returns all NEPSE trading days between start and end (inclusive).
    This is the primary iterator used by the backtest engine.
    """
    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(end, str):
        end = date.fromisoformat(end)

    days: list[date] = []
    current = start
    while current <= end:
        if is_trading_day(current):
            days.append(current)
        current += timedelta(days=1)
    return days


def get_nepse_year_trading_days(year: int) -> int:
    """Returns the number of trading days in a NEPSE year (~240)."""
    return len(get_trading_days(f"{year}-01-01", f"{year}-12-31"))


def add_trading_days(d: str | date, n: int) -> date:
    """
    Advance by n trading days (useful for T+3 settlement checks, hold period).
    n can be negative to go backwards.
    """
    if isinstance(d, str):
        d = date.fromisoformat(d)
    if n == 0:
        return d
    step = 1 if n > 0 else -1
    remaining = abs(n)
    current = d
    while remaining > 0:
        current += timedelta(days=step)
        if is_trading_day(current):
            remaining -= 1
    return current


def trading_days_between(start: str | date, end: str | date) -> int:
    """Count trading days between two dates (exclusive of start, inclusive of end)."""
    return len(get_trading_days(start, end)) - (1 if is_trading_day(start) else 0)


def get_week_for_signal(signal_date: str | date) -> tuple[date, date]:
    """
    Given a signal date, return (monday_of_execution_week, friday_equivalent).
    Since NEPSE weeks run Sun-Thu, returns (Sunday, Thursday) of that week.
    If signal_date is Friday or Saturday (weekend), advance to next Sunday.
    """
    if isinstance(signal_date, str):
        signal_date = date.fromisoformat(signal_date)

    entry = get_next_trading_day(signal_date)
    # Find Thursday of same NEPSE week
    wd = entry.weekday()
    # 0=Mon,1=Tue,2=Wed,3=Thu,6=Sun → days to Thursday (3)
    if wd == 6:  # Sunday
        days_to_thu = 4
    elif wd <= 3:  # Mon-Thu
        days_to_thu = 3 - wd
    else:
        days_to_thu = 0
    week_end = entry + timedelta(days=days_to_thu)
    return entry, week_end


# ---------------------------------------------------------------------------
# Utility: check if a date is within a caution period around book close etc.
# ---------------------------------------------------------------------------

def is_within_n_trading_days(
    target: str | date,
    reference: str | date,
    n: int = 3,
) -> bool:
    """True if |trading days between target and reference| <= n."""
    return abs(trading_days_between(min(target, reference),
                                     max(target, reference))) <= n


# ---------------------------------------------------------------------------
# CLI: print calendar stats
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for yr in [2021, 2022, 2023, 2024]:
        n = get_nepse_year_trading_days(yr)
        print(f"NEPSE {yr}: {n} trading days")

    # Verify Sunday–Thursday logic
    test_date = date(2024, 1, 7)  # Sunday
    print(f"\n{test_date} ({test_date.strftime('%A')}) is_trading_day: {is_trading_day(test_date)}")
    test_date2 = date(2024, 1, 5)  # Friday
    print(f"{test_date2} ({test_date2.strftime('%A')}) is_trading_day: {is_trading_day(test_date2)}")

    days_2024 = get_trading_days("2024-01-01", "2024-01-31")
    print(f"\nJanuary 2024 trading days ({len(days_2024)}):")
    for d in days_2024:
        print(f"  {d} ({d.strftime('%A')})")
