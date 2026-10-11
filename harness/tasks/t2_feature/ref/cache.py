import time
from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity, ttl=None, clock=None):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity, self.ttl, self._clock = capacity, ttl, clock or time.monotonic
        self._d = OrderedDict()   # key -> (value, expires_at|None)
        self._hits = self._misses = 0

    def _expired(self, exp):
        return exp is not None and self._clock() >= exp

    def _purge(self):
        for k in [k for k, (_, e) in self._d.items() if self._expired(e)]:
            del self._d[k]

    def get(self, key, default=None):
        item = self._d.get(key)
        if item is None or self._expired(item[1]):
            self._d.pop(key, None)
            self._misses += 1
            return default
        self._d.move_to_end(key)
        self._hits += 1
        return item[0]

    def put(self, key, value):
        self._purge()
        exp = None if self.ttl is None else self._clock() + self.ttl
        self._d[key] = (value, exp)
        self._d.move_to_end(key)
        while len(self._d) > self.capacity:
            self._d.popitem(last=False)

    def delete(self, key):
        return self._d.pop(key, None) is not None

    def clear(self):
        self._d.clear()

    def keys(self):
        self._purge()
        return list(self._d)

    def __contains__(self, key):
        item = self._d.get(key)
        return item is not None and not self._expired(item[1])

    def __len__(self):
        self._purge()
        return len(self._d)

    def stats(self):
        return {"hits": self._hits, "misses": self._misses, "size": len(self)}
