import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
from cache import LRUCache


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


def test_basic_get_put():
    c = LRUCache(2)
    c.put("a", 1); c.put("b", 2)
    assert c.get("a") == 1 and c.get("b") == 2
    assert c.get("zzz") is None
    assert c.get("zzz", 42) == 42
    assert len(c) == 2 and "a" in c and "q" not in c


def test_eviction_lru_order():
    c = LRUCache(2)
    c.put("a", 1); c.put("b", 2)
    c.get("a")              # a becomes the most recent
    c.put("c", 3)           # evicts b
    assert "b" not in c and "a" in c and "c" in c
    assert list(c.keys()) == ["a", "c"]   # from least to most recently used


def test_update_refreshes_and_does_not_grow():
    c = LRUCache(2)
    c.put("a", 1); c.put("b", 2); c.put("a", 10)
    c.put("c", 3)
    assert c.get("a") == 10 and "b" not in c and len(c) == 2


def test_ttl_expiry():
    clk = Clock(); c = LRUCache(3, ttl=10, clock=clk)
    c.put("a", 1)
    clk.t = 9.9
    assert c.get("a") == 1
    clk.t = 10.0            # expires exactly at ttl
    assert c.get("a") is None and "a" not in c and len(c) == 0


def test_get_does_not_extend_ttl_but_put_does():
    clk = Clock(); c = LRUCache(3, ttl=10, clock=clk)
    c.put("a", 1)
    clk.t = 6; assert c.get("a") == 1
    clk.t = 11; assert c.get("a") is None
    c.put("b", 2); clk.t = 15; c.put("b", 3); clk.t = 24
    assert c.get("b") == 3


def test_expired_entries_do_not_count_for_eviction():
    clk = Clock(); c = LRUCache(2, ttl=5, clock=clk)
    c.put("a", 1); c.put("b", 2)
    clk.t = 6
    c.put("c", 3); c.put("d", 4)
    assert set(c.keys()) == {"c", "d"}


def test_delete_and_clear_and_stats():
    c = LRUCache(2)
    c.put("a", 1)
    assert c.delete("a") is True and c.delete("a") is False
    c.put("a", 1); c.get("a"); c.get("x")
    assert c.stats() == {"hits": 1, "misses": 1, "size": 1}
    c.clear()
    assert len(c) == 0


def test_invalid_capacity():
    with pytest.raises(ValueError):
        LRUCache(0)
