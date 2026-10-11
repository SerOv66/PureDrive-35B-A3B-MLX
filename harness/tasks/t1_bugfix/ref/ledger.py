"""Small ledger: entries in cents, balance, interest, statement."""
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Entry:
    day: date
    cents: int          # positive = credit, negative = debit
    memo: str = ""


def balance(entries):
    """Total balance in cents."""
    return sum(e.cents for e in entries)


def running_balance(entries, opening=0):
    """List of running balances after each entry, in date order (entries on the same date keep insertion order)."""
    out, bal = [], opening
    for e in sorted(entries, key=lambda e: e.day):
        bal += e.cents
        out.append(bal)
    return out


def monthly_interest(entries, year, month, annual_rate_pct):
    """Interest of the month in cents: average daily balance * rate/12, rounded to the nearest cent
    (0.5 rounds up). Each day's balance includes that day's entries."""
    import calendar
    ndays = calendar.monthrange(year, month)[1]
    total = 0
    for d in range(1, ndays + 1):
        day = date(year, month, d)
        total += sum(e.cents for e in entries if e.day <= day)
    avg = total / ndays
    return int(avg * annual_rate_pct / 100 / 12 + 0.5)


def statement(entries, year, month):
    """Statement of the month: lines 'YYYY-MM-DD  +12.34  memo' and a final total."""
    rows = [e for e in sorted(entries, key=lambda e: e.day) if (e.day.year, e.day.month) == (year, month)]
    lines = [f"{e.day.isoformat()}  {e.cents / 100:+.2f}  {e.memo}".rstrip() for e in rows]
    lines.append(f"TOTAL  {balance(rows) / 100:+.2f}")
    return "\n".join(lines)
