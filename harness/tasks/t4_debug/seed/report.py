"""Sales report by region and month.

Usage: python report.py data/sales.csv

CSV format: date,region,product,amount. Rules for dirty data:
  - the date may be ISO (2024-02-17) or DD/MM/YYYY (17/02/2024);
  - the amount may use a decimal comma ("12,50"); rows with a missing amount must be skipped (and counted);
  - the region must be normalized (strip + capitalize): " south " == "South".
Output: for each region (alphabetical order) one line per month with the total and the number of sales, then the grand total
and the number of skipped rows.
"""
import csv
import sys
from collections import defaultdict
from datetime import datetime


def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def load(path):
    totals = defaultdict(lambda: [0.0, 0])
    skipped = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            amount = float(row["amount"])
            d = parse_date(row["date"])
            key = (row["region"], d.strftime("%Y-%m"))
            totals[key][0] += amount
            totals[key][1] += 1
    return totals, skipped


def render(totals, skipped):
    lines = []
    for region in sorted({k[0] for k in totals}):
        for (r, month), (amt, n) in sorted(totals.items()):
            if r == region:
                lines.append(f"{region:<8} {month}  {amt:>10.2f}  {n:>3}")
    lines.append(f"TOTAL    {sum(v[0] for v in totals.values()):>18.2f}")
    lines.append(f"SKIPPED  {skipped}")
    return "\n".join(lines)


if __name__ == "__main__":
    t, s = load(sys.argv[1])
    print(render(t, s))
