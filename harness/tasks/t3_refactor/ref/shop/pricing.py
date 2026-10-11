"""Prices and discounts. Prices are in cents (int)."""
VAT_RATE = 0.20


def apply_discount(cents, pct):
    return round(cents * (100 - pct) / 100)
