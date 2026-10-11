from shop.totals import compute_total


class Cart:
    def __init__(self, discount_pct=0):
        self.items = []
        self.discount_pct = discount_pct

    def add(self, price_cents, qty=1):
        self.items.append((price_cents, qty))

    def total(self):
        return compute_total(self.items, self.discount_pct)

    def total_net(self):
        return compute_total(self.items, self.discount_pct, vat=False)
