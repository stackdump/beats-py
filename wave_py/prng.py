"""Bit-exact ports of every PRNG on the wave engine's path.

All arithmetic is done on Python ints with explicit uint32 masking, then
reinterpreted as int32 where the JS does `| 0`. No `random` module.

- mulberry32       — lib/generator/core.js::createRng (compose-time choices)
- deterministic_rand — sequencer-worker.js / wave-engine/net.js (conflicts)
- str_hash         — the same files (place-label salt), over UTF-16 code units
- noise_seed / xorshift32 — wave-engine/synth.js + runner.js (lane noise)
"""

M32 = 0xFFFFFFFF


def u32(x):
    return x & M32


def i32(x):
    x &= M32
    return x - 0x100000000 if x & 0x80000000 else x


def to_int32(v):
    """JS ToInt32 (`v | 0`) for a Python number."""
    if isinstance(v, float):
        if v != v or v in (float('inf'), float('-inf')):
            return 0
        v = int(v)  # truncates toward zero, as ToInt32 does
    return i32(v)


def imul(a, b):
    """Math.imul: low 32 bits of the product, as int32."""
    return i32((a & M32) * (b & M32))


def _mulberry_step(s):
    """One mulberry32 output from the already-advanced int32 state s → uint32."""
    t = imul(s ^ (u32(s) >> 15), 1 | s)
    t = i32(i32(t + imul(t ^ (u32(t) >> 7), 61 | t)) ^ t)
    return u32(t ^ (u32(t) >> 14))


class mulberry32:
    """createRng(seed) — next() → float in [0,1), nextInt(max)."""

    def __init__(self, seed):
        self.s = to_int32(seed)

    def next_u32(self):
        self.s = i32(self.s + 0x6D2B79F5)
        return _mulberry_step(self.s)

    def next(self):
        return self.next_u32() / 4294967296

    def next_int(self, mx):
        import math
        return math.floor(self.next() * mx)


def deterministic_rand(tick, salt):
    s = i32(tick + salt)
    s = i32(s + 0x6D2B79F5)
    return _mulberry_step(s) / 4294967296


def str_hash(s):
    h = 0
    data = s.encode('utf-16-le')
    for i in range(0, len(data), 2):
        cu = data[i] | (data[i + 1] << 8)
        h = i32(imul(31, h) + cu)
    return h


def noise_seed(i):
    s = i32(i32(0x9E3779B9) ^ imul(i + 1, i32(0x85EBCA6B)))
    return 1 if s == 0 else s


def xorshift32(x):
    """One step of the runner's lane noise. x, result: int32."""
    x = u32(x)
    x ^= (x << 13) & M32
    x ^= x >> 17
    x ^= (x << 5) & M32
    return i32(x)
