"""Prices and discounts. Prices are in cents (int)."""
VAT_RATE = 0.20


def apply_discount(cents, pct):
    return round(cents * (100 - pct) / 100)


def calc_total(items, discount_pct=0, vat=True):
    """Total of a list of (unit_price_cents, quantity): percentage discount, then VAT."""
    subtotal = sum(price * qty for price, qty in items)
    discounted = apply_discount(subtotal, discount_pct)
    return round(discounted * (1 + VAT_RATE)) if vat else discounted
