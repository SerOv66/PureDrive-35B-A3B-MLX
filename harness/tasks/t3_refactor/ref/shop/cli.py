import sys
from shop.cart import Cart
from shop.totals import compute_total  # used for the quick preview


def preview(args):
    """args: price_cents:qty pairs, e.g. 1000:2 550:1"""
    items = [tuple(int(x) for x in a.split(":")) for a in args]
    return compute_total(items)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    print(f"{preview(argv) / 100:.2f}")


if __name__ == "__main__":
    main()
