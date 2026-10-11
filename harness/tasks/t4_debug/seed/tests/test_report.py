import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from report import load, render


def test_report_matches_expected():
    root = pathlib.Path(__file__).resolve().parents[1]
    t, s = load(str(root / "data" / "sales.csv"))
    assert s == 1
    out = render(t, s)
    assert "South" in out and " south " not in out
    assert out.splitlines()[-1] == "SKIPPED  1"
