from shop.pricing import calc_total


class Cart:
    def __init__(self, discount_pct=0):
        self.items = []
        self.discount_pct = discount_pct

    def add(self, price_cents, qty=1):
        self.items.append((price_cents, qty))

    def total(self):
        return calc_total(self.items, self.discount_pct)

    def total_net(self):
        return calc_total(self.items, self.discount_pct, vat=False)
