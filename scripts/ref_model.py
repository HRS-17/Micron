"""Software model of the cache (no prefetching).

Two uses:
  1. Golden check: with 64 sets x 4 ways and tree-PLRU it must produce exactly the
     same hit count as RTL config 0 on every trace.
  2. Level 2 design-space sweep: size, associativity and replacement policy.
"""
import random


def load_trace(path):
    sids, lines = [], []
    with open(path) as f:
        for row in f:
            row = row.strip()
            if row:
                v = int(row, 16)
                sids.append((v >> 32) & 0x7)
                lines.append((v & 0xFFFF_FFFF) >> 6)
    return sids, lines


class Cache:
    """Set-associative cache. policy: 'plru' (tree, power-of-two ways), 'lru', 'random'."""

    def __init__(self, sets, ways, policy="plru", seed=1):
        self.sets, self.ways, self.policy = sets, ways, policy
        self.tags = [[None] * ways for _ in range(sets)]
        self.stamp = [[0] * ways for _ in range(sets)]
        self.tree = [[0] * max(ways - 1, 1) for _ in range(sets)]
        self.t = 0
        self.rng = random.Random(seed)

    # tree-PLRU: node bit 0 -> victim on the left, touching a way points nodes away from it
    def _plru_victim(self, s):
        n, lo, hi = 0, 0, self.ways
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.tree[s][n] == 0:
                n, hi = 2 * n + 1, mid
            else:
                n, lo = 2 * n + 2, mid
        return lo

    def _plru_touch(self, s, w):
        n, lo, hi = 0, 0, self.ways
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if w < mid:
                self.tree[s][n] = 1
                n, hi = 2 * n + 1, mid
            else:
                self.tree[s][n] = 0
                n, lo = 2 * n + 2, mid

    def access(self, line):
        s, tag = line % self.sets, line // self.sets
        self.t += 1
        row = self.tags[s]
        if tag in row:
            w = row.index(tag)
            self._touch(s, w)
            return True
        if None in row:
            w = row.index(None)
        elif self.policy == "plru":
            w = self._plru_victim(s)
        elif self.policy == "lru":
            w = min(range(self.ways), key=lambda i: self.stamp[s][i])
        else:
            w = self.rng.randrange(self.ways)
        row[w] = tag
        self._touch(s, w)
        return False

    def _touch(self, s, w):
        self.stamp[s][w] = self.t
        if self.policy == "plru" and self.ways > 1:
            self._plru_touch(s, w)


def hits(lines, sets=64, ways=4, policy="plru"):
    c = Cache(sets, ways, policy)
    return sum(c.access(l) for l in lines)
