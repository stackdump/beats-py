"""16-bit PCM WAV via the stdlib `wave` module, with offline.js::encodeWav's
quantiser (clamp, then round(x·32768) below zero and round(x·32767) above)."""

import math
import struct
import wave


def _q(x):
    x = -1.0 if x < -1 else 1.0 if x > 1 else x
    v = x * 32768 if x < 0 else x * 32767
    r = math.floor(v)
    return r + 1 if v - r >= 0.5 else r   # Math.round


def write_wav(path, samples, sr, channels=2):
    frames = bytearray()
    pack = struct.Struct('<' + 'h' * channels).pack
    for x in samples:
        q = _q(x)
        frames += pack(*([q] * channels))
    with wave.open(path, 'wb') as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))
