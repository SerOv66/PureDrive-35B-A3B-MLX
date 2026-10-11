import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shop.totals import compute_total
from shop.cart import Cart
from shop import invoice, cli


def test_compute_total():
    assert compute_total([(1000, 2), (550, 1)]) == round(2550 * 1.20)
    assert compute_total([(1000, 2)], 10, vat=False) == 1800


def test_cart():
    c = Cart(discount_pct=10); c.add(1000, 2); c.add(500)
    assert c.total_net() == 2250 and c.total() == round(2250 * 1.20)


def test_invoice():
    c = Cart(); c.add(1000, 2)
    assert invoice.render(c, 7).splitlines() == ["INVOICE 00007", "2 x 10.00", "Net 20.00", "Total 24.00"]


def test_cli_preview():
    assert cli.preview(["1000:2", "550:1"]) == round(2550 * 1.20)


def test_pricing_keeps_discount():
    from shop.pricing import apply_discount
    assert apply_discount(1000, 25) == 750
