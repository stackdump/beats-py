"""Port of public/wave-engine/synth.js — voice specs, pulse shapes, the
wavetables, the RBJ biquad, and the closed forms the tests check against:
ring_of, envelope, ring_spectrum.

Wavetables are stored as float32 (array('f')), exactly as the JS
Float32Array tables are, so table lookups agree to the bit.
"""

import math
import re
from array import array

PPQ = 4
TABLE_SIZE = 2048


def tick_seconds(bpm):
    return 60 / (bpm * PPQ)


def midi_to_hz(note):
    return 440 * math.pow(2, (note - 69) / 12)


def pulse(attack, decay):
    if not (attack > 0) or attack >= decay:
        return {'alpha': [1], 'tau': [decay]}
    return {'alpha': [1, -1], 'tau': [decay, attack]}


# --- Voices ------------------------------------------------------------------

KITS = {
    'drums':           {'kickDecay': 0.30, 'kickOctaves': 6, 'snareDecay': 0.15, 'hihatDecay': 0.05},
    'drums-breakbeat': {'kickDecay': 0.20, 'kickOctaves': 4, 'snareDecay': 0.20, 'hihatDecay': 0.08},
    'drums-cr78':      {'kickDecay': 0.25, 'kickOctaves': 3, 'snareDecay': 0.12, 'hihatDecay': 0.04},
    'drums-v8':        {'kickDecay': 0.35, 'kickOctaves': 8, 'snareDecay': 0.18, 'hihatDecay': 0.06},
    'drums-808':       {'kickDecay': 0.60, 'kickOctaves': 8, 'snareDecay': 0.20, 'hihatDecay': 0.04},
    'drums-lofi':      {'kickDecay': 0.25, 'kickOctaves': 4, 'snareDecay': 0.15, 'hihatDecay': 0.05},
}


def is_drum_channel(ch):
    return 10 <= ch <= 15


def drum_kind(note):
    if note in (36, 35):
        return 'kick'
    if note in (38, 40, 37):
        return 'snare'
    if note == 39:
        return 'clap'
    if 42 <= note <= 46:
        return 'hat'
    if note in (49, 57):
        return 'openhat'
    return 'kick'


def drum_voice(kind, instrument):
    k = KITS.get(instrument) or KITS['drums']
    if kind == 'kick':
        return {'kind': kind, 'carrier': 'sweep', 'f1': 48, 'f0': 48 * k['kickOctaves'], 'sweepTau': 0.03,
                'pulse': pulse(0.001, k['kickDecay'] / 2.5), 'level': 0.9}
    if kind == 'snare':
        return {'kind': kind, 'carrier': 'noise+tone', 'filter': 'bandpass', 'fc': 3000, 'q': 0.9,
                'tone': 180, 'toneMix': 0.35, 'pulse': pulse(0.001, k['snareDecay'] / 2.5), 'level': 0.6}
    if kind == 'clap':
        return {'kind': kind, 'carrier': 'noise', 'filter': 'bandpass', 'fc': 1400, 'q': 1.2,
                'pulse': pulse(0.002, 0.06), 'level': 0.6}
    if kind == 'openhat':
        return {'kind': kind, 'carrier': 'noise', 'filter': 'highpass', 'fc': 7000, 'q': 0.7,
                'pulse': pulse(0.001, k['hihatDecay'] * 2.5), 'level': 0.28}
    return {'kind': 'hat', 'carrier': 'noise', 'filter': 'highpass', 'fc': 7000, 'q': 0.7,
            'pulse': pulse(0.001, k['hihatDecay'] / 2), 'level': 0.32}


def tonal_voice(instrument, group):
    name = instrument or ''
    wave, attack, decay_scale, level, max_decay = 'saw', 0.003, 0.5, 0.22, 1.5
    if re.search(r'pad|strings|choir|organ', name) or group == 'harmony' or group == 'pad':
        wave, attack, decay_scale, level, max_decay = 'soft', 0.06, 0.8, 0.12, 3
    elif re.search(r'sub|808', name):
        wave, level = 'sine', 0.42
    elif re.search(r'bass|reese|acid', name) or group == 'bass':
        wave, level = 'saw', 0.26
    elif re.search(r'bell|vibes|marimba|kalimba|music-box|steel|glass', name):
        wave, attack, decay_scale, level = 'bell', 0.001, 1.2, 0.2
    elif re.search(r'square|chiptune|pwm|clav', name):
        wave, level = 'square', 0.16
    elif re.search(r'pluck|stab|piano|guitar|harpsi', name):
        wave, attack, decay_scale, level = 'saw', 0.002, 0.35, 0.2
    return {'kind': 'tonal', 'carrier': 'table', 'wave': wave, 'attack': attack,
            'decayScale': decay_scale, 'maxDecay': max_decay, 'level': level}


# --- Wavetables ----------------------------------------------------------------

PARTIALS = {
    'sine':   [1],
    'saw':    [1, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8],
    'square': [1, 0, 1 / 3, 0, 1 / 5, 0, 1 / 7],
    'soft':   [1, 0.5, 0.25, 0.12],
    'bell':   [1, 0, 0.45, 0, 0, 0.25, 0, 0, 0.12],
}


def build_table(wave):
    amps = PARTIALS.get(wave) or PARTIALS['sine']
    t = array('f', [0.0]) * (TABLE_SIZE + 1)
    peak = 0.0
    for i in range(TABLE_SIZE):
        s = 0
        for k, a in enumerate(amps):
            if a:
                s += a * math.sin(2 * math.pi * (k + 1) * i / TABLE_SIZE)
        t[i] = s                      # rounds to float32, as Float32Array does
        if abs(s) > peak:
            peak = abs(s)
    for i in range(TABLE_SIZE):
        t[i] = t[i] / peak
    t[TABLE_SIZE] = t[0]
    return t


# --- RBJ biquad -----------------------------------------------------------------

def biquad(kind, fc, q, sr):
    w = 2 * math.pi * min(fc, sr * 0.45) / sr
    cw, sw = math.cos(w), math.sin(w)
    alpha = sw / (2 * q)
    if kind == 'highpass':
        b0, b1, b2 = (1 + cw) / 2, -(1 + cw), (1 + cw) / 2
    elif kind == 'lowpass':
        b0, b1, b2 = (1 - cw) / 2, 1 - cw, (1 - cw) / 2
    else:
        b0, b1, b2 = alpha, 0, -alpha
    a0 = 1 + alpha
    return [b0 / a0, b1 / a0, b2 / a0, (-2 * cw) / a0, (1 - alpha) / a0]


# --- Rings: closed forms -----------------------------------------------------------

def ring_of(nb):
    """(n, places, transitions, weights) for a single-token cycle, else None."""
    place_ids = list(nb.places.keys())
    n = len(place_ids)
    if not n or len(nb.transitions) != n:
        return None
    if sum(nb.state.get(p, 0) or 0 for p in place_ids) != 1:
        return None
    consumer, producer = {}, {}
    for a in nb.arcs:
        if a['inhibit']:
            return None
        if a['source'] in nb.places:
            if a['source'] in consumer:
                return None
            consumer[a['source']] = a['target']
        else:
            if a['source'] in producer:
                return None
            producer[a['source']] = a['target']
    start = next(p for p in place_ids if (nb.state.get(p, 0) or 0) > 0)
    places, trans, p = [], [], start
    for _ in range(n):
        t = consumer.get(p)
        if not t or t not in producer:
            return None
        places.append(p)
        trans.append(t)
        p = producer[t]
    if p != start or len(set(places)) != n:
        return None
    weights = []
    for t in trans:
        b = nb.bindings.get(t)
        weights.append(b['velocity'] / 127 if b else 0.0)
    return {'n': n, 'places': places, 'transitions': trans, 'weights': weights}


def envelope(ring, j, phase, p, tick):
    """Stationary closed-form gate with the token on p_j, `phase` of a tick elapsed."""
    n = ring['n']
    g = 0.0
    for alpha, tau in zip(p['alpha'], p['tau']):
        a = math.exp(-tick / tau)
        acc, am = 0.0, 1.0
        for m in range(n):
            acc += ring['weights'][(j - 1 - m) % n] * am
            am *= a
        g += alpha * math.exp(-phase * tick / tau) * acc / (1 - am)
    return g


def ring_spectrum(weights, p, M, sr, K):
    """Predicted Fourier coefficients c_k (k < K) of the sampled gate over one
    ring period N = n·M samples: c_k = P̂(k) · (1/n) Σ_i w_i e^{−2πi k i/n}."""
    n = len(weights)
    N = n * M
    out = []
    for k in range(K):
        wr = wi = 0.0
        for i in range(n):
            ang = -2 * math.pi * k * i / n
            wr += weights[i] * math.cos(ang)
            wi += weights[i] * math.sin(ang)
        wr /= n
        wi /= n
        pr = pi = 0.0
        for alpha, tau in zip(p['alpha'], p['tau']):
            m = math.exp(-1 / (tau * sr))
            ang = -2 * math.pi * k / N
            dr, di = 1 - m * math.cos(ang), -m * math.sin(ang)
            d2 = dr * dr + di * di
            pr += alpha * dr / d2
            pi += alpha * -di / d2
        pr /= M
        pi /= M
        out.append(complex(pr * wr - pi * wi, pr * wi + pi * wr))
    return out
