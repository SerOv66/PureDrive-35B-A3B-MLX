import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from datetime import date
from ledger import Entry, balance, running_balance, monthly_interest, statement

E = [Entry(date(2024, 3, 1), 100000, "salary"), Entry(date(2024, 3, 10), -25050, "rent"),
     Entry(date(2024, 3, 10), -1, "fee"), Entry(date(2024, 2, 28), 5000, "refund")]


def test_balance():
    assert balance(E) == 79949


def test_running_balance_sorted_and_stable():
    assert running_balance(E) == [5000, 105000, 79950, 79949]


def test_running_balance_opening():
    assert running_balance(E, opening=1000) == [6000, 106000, 80950, 80949]


def test_interest_full_month():
    # constant balance of 100000 for all 31 days of March, 12% per year -> 1% per month = 1000 cents
    e = [Entry(date(2024, 2, 1), 100000)]
    assert monthly_interest(e, 2024, 3, 12) == 1000


def test_interest_rounding_half_up():
    # constant balance of 50 cents, 12% per year -> 0.5 cents -> rounds to 1
    assert monthly_interest([Entry(date(2024, 1, 1), 50)], 2024, 4, 12) == 1


def test_interest_movement_on_last_day_counts():
    e = [Entry(date(2024, 4, 30), 30000)]
    # only one day out of 30 with balance 30000 -> average 1000 -> 12% per year -> 10 cents
    assert monthly_interest(e, 2024, 4, 12) == 10


def test_statement():
    out = statement(E, 2024, 3)
    assert out.splitlines() == ["2024-03-01  +1000.00  salary", "2024-03-10  -250.50  rent",
                                "2024-03-10  -0.01  fee", "TOTAL  +749.49"]
