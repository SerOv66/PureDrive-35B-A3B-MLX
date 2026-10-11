from shop.pricing import VAT_RATE, apply_discount


def compute_total(items, discount_pct=0, vat=True):
    """Total of a list of (unit_price_cents, quantity): percentage discount, then VAT."""
    subtotal = sum(price * qty for price, qty in items)
    discounted = apply_discount(subtotal, discount_pct)
    return round(discounted * (1 + VAT_RATE)) if vat else discounted
