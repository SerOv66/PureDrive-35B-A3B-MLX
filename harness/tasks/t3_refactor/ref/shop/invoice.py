from shop import totals


def render(cart, number):
    lines = [f"INVOICE {number:05d}"]
    for price, qty in cart.items:
        lines.append(f"{qty} x {price / 100:.2f}")
    net = totals.compute_total(cart.items, cart.discount_pct, vat=False)
    gross = totals.compute_total(cart.items, cart.discount_pct)
    lines.append(f"Net {net / 100:.2f}")
    lines.append(f"Total {gross / 100:.2f}")
    return "\n".join(lines)
