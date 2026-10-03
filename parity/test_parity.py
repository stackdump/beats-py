#!/usr/bin/env python3
"""Python ↔ JS parity for the wave engine.

  BEATS_PUBLIC=/path/to/beats-bitwrap-io/public python3 parity/test_parity.py [--audio-seconds 3] [--ticks 1200]

Needs node and a beats-bitwrap-io checkout (the JS reference is never vendored).

1. PRNG: mulberry32, deterministicRand, strHash, noiseSeed + xorshift32 vs the
   JS outputs, exact.
2. Trace: the marking trace (wave_py/trace.py vs parity/trace.mjs over the
   unmodified JS executor) for several (genre, seed, structure) plus a net with
   a real conflict — exact equality, every tick. On a mismatch, report the first
   diverging tick, net and place. Never loosened.
3. P-invariants: a basis of {yᵀC = 0} from each net's incidence matrix (exact
   rational arithmetic); y·M must be constant at every tick of every trace.
4. Spectrum: one isolated ring (techno/42 hi-hat, DC carrier, linear master):
   DFT of the rendered gate vs the c_k predicted from the DFT of its place
   weights.
5. Audio: Python offline render vs the JS offline render (renderOffline) of
   the identical project JSON — max abs sample error.

numpy is optional (used for the DFT when present). Results → parity/results.json.
"""

import argparse
import cmath
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))

from wave_py.prng import mulberry32, deterministic_rand, str_hash, noise_seed, xorshift32  # noqa: E402
from wave_py.compose import compose_project  # noqa: E402
from wave_py.trace import iter_trace, _dumps  # noqa: E402
from wave_py.net import compile_project, incidence_matrix, p_invariants  # noqa: E402
from wave_py.project import parse_project  # noqa: E402
from wave_py.synth import pulse, ring_of, ring_spectrum, envelope  # noqa: E402
from wave_py.runner import render_offline  # noqa: E402

TRACE_CASES = [('techno', 42, 'standard'), ('jazz', 7, ''), ('edm', 1234, 'extended'),
               ('dnb', 99, 'standard'), ('ambient', 3, '')]
AUDIO_CASES = [('techno', 42, ''), ('jazz', 7, ''), ('edm', 1234, 'standard')]
AUDIO_TOL = 1e-5

# Two transitions compete for one place every other tick — exercises
# deterministic_rand / str_hash conflict resolution (same net as the JS test).
CONFLICT = {
    'name': 'conflict', 'tempo': 120,
    'nets': {'fork': {
        'track': {'channel': 1},
        'places': {'a': {'initial': [1]}, 'b': {}, 'c': {}},
        'transitions': {'left': {'midi': {'note': 60}}, 'right': {'midi': {'note': 64}}, 'back1': {}, 'back2': {}},
        'arcs': [
            {'source': 'a', 'target': 'left'}, {'source': 'a', 'target': 'right'},
            {'source': 'left', 'target': 'b'}, {'source': 'right', 'target': 'c'},
            {'source': 'b', 'target': 'back1'}, {'source': 'back1', 'target': 'a'},
            {'source': 'c', 'target': 'back2'}, {'source': 'back2', 'target': 'a'},
        ],
    }},
}

failures = []
results = {}


def check(name, ok, detail=''):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{' — ' + detail if detail else ''}", flush=True)
    if not ok:
        failures.append(name)


def node(script, *args):
    r = subprocess.run(['node', os.path.join(HERE, script), *map(str, args)],
                       capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f'{script} failed: {r.stderr[-800:]}')
    return r.stdout


def case_name(genre, seed, structure):
    return f"{genre}/{seed}/{structure or 'loop'}"


# --- 1. PRNG -------------------------------------------------------------------

def test_prng():
    ref = json.loads(node('prng.mjs'))
    ok = True
    for seed, seq in zip(ref['seeds'], ref['mulberry']):
        r = mulberry32(seed)
        ok &= all(r.next() == x for x in seq)
    check('prng: mulberry32 (createRng) bit-exact, 8 seeds × 64 draws', ok)
    check('prng: strHash over UTF-16 code units', [str_hash(s) for s in ref['labels']] == ref['hashes'])
    py_rands = [deterministic_rand(t, h) for h in ref['hashes'] for t in [0, 1, 2, 17, 928, 100000, -3]]
    check('prng: deterministicRand(tick, salt)', py_rands == ref['rands'], f'{len(py_rands)} draws')
    ok = True
    for i, seq in zip([0, 1, 2, 5, 52], ref['noise']):
        x = noise_seed(i)
        out = [x]
        for _ in range(64):
            x = xorshift32(x)
            out.append(x)
        ok &= out == seq
    check('prng: noiseSeed + xorshift32 lane noise', ok)
    results['prng'] = 'exact'


# --- 2 + 3. Trace parity and P-invariants ----------------------------------------------

def first_divergence(py_lines, js_lines):
    hdr_py, hdr_js = json.loads(py_lines[0]), json.loads(js_lines[0])
    if hdr_py != hdr_js:
        return 'header (net/place/transition order) differs'
    for i in range(1, max(len(py_lines), len(js_lines))):
        if i >= len(py_lines) or i >= len(js_lines):
            return f'trace length differs at line {i}'
        a, b = json.loads(py_lines[i]), json.loads(js_lines[i])
        if a == b:
            continue
        for key in ('fired', 'ctl', 'muted', 'mn', 'stop'):
            if a[key] != b[key]:
                return f"tick {a['t']}: '{key}' py={a[key]} js={b[key]}"
        for ni, (ma, mb) in enumerate(zip(a['m'], b['m'])):
            if ma != mb:
                p = next(k for k in range(len(ma)) if ma[k] != mb[k])
                return (f"tick {a['t']}: net {hdr_py['nets'][ni]} place {hdr_py['places'][ni][p]} "
                        f"py={ma[p]} js={mb[p]}")
        return f"tick {a['t']}: records differ"
    return None


def invariant_check(project, records, label):
    g = compile_project(project)
    # Scale each rational basis vector to integers (exact), so y·M is
    # integer arithmetic on the integer marking.
    bases = []
    for n in g.nets:
        ys = []
        for y in p_invariants(incidence_matrix(n)):
            den = 1
            for v in y:
                den = den * v.denominator // math.gcd(den, v.denominator)
            ys.append([int(v * den) for v in y])
        bases.append(ys)
    music = [i for i, n in enumerate(g.nets) if n.bundle.role != 'control']
    ones = 0
    for i in music:
        if any(len({v for v in y if v != 0}) == 1 and all(v != 0 for v in y) for y in bases[i]):
            ones += 1
    init = [[sum(y[p] * n.initial[p] for p in range(n.P) if y[p]) for y in bases[i]]
            for i, n in enumerate(g.nets)]
    breaches, checked = 0, 0
    first = None
    for rec in records:
        for i, m in enumerate(rec['m']):
            for k, y in enumerate(bases[i]):
                checked += 1
                if sum(y[p] * m[p] for p in range(len(m)) if y[p]) != init[i][k]:
                    breaches += 1
                    first = first or (rec['t'], g.nets[i].id)
    none = [n.id for i, n in enumerate(g.nets) if not bases[i]]
    check(f'P-invariants: {label}: y·M constant every tick (exact, rational)', breaches == 0,
          f'{len(records)} ticks × {len(g.nets)} nets, {checked} checks'
          + (f', first breach tick {first[0]} net {first[1]}' if first else ''))
    check(f'P-invariants: {label}: every music ring has the all-ones invariant',
          ones == len(music), f'{ones}/{len(music)} rings; nets with no invariant: {len(none)}')
    return {'checks': checked, 'breaches': breaches, 'rings_all_ones': f'{ones}/{len(music)}',
            'nets_without_invariant': none}


def test_traces(ticks, tmp):
    out = {}
    cases = [(case_name(*c), compose_project(*c)) for c in TRACE_CASES] + [('conflict-net', CONFLICT)]
    for label, project in cases:
        pj = os.path.join(tmp, 'p.json')
        with open(pj, 'w') as f:
            json.dump(project, f)
        js_path = os.path.join(tmp, 'js.jsonl')
        node('trace.mjs', pj, ticks, js_path)
        with open(js_path, encoding='utf-8') as f:
            js_lines = f.read().splitlines()
        recs = list(iter_trace(project, ticks))
        py_lines = [_dumps(r) for r in recs]
        div = first_divergence(py_lines, js_lines)
        byte_eq = py_lines == js_lines
        ticks_with_fire = sum(1 for r in recs[1:] if r['fired'])
        stop_at = next((r['t'] for r in recs[1:] if r['stop']), None)
        check(f'trace: {label}: exact over {ticks} ticks', div is None,
              div or f"{len(recs[0]['nets'])} nets, {ticks_with_fire} ticks with audible fires, "
                     f"{sum(len(r['ctl']) for r in recs[1:])} control fires"
                     + (f', stop-transport at tick {stop_at}' if stop_at else '')
                     + (', byte-identical JSONL' if byte_eq else ''))
        entry = {'ticks': ticks, 'exact': div is None, 'byte_identical': byte_eq,
                 'first_divergence': div, 'nets': len(recs[0]['nets'])}
        if label == 'conflict-net':
            fired = [r['fired'] for r in recs[1:] if r['fired']]
            lefts = sum(1 for f in fired if [0, 0] in f)
            rights = sum(1 for f in fired if [0, 1] in f)
            check('trace: conflict resolution takes both branches', lefts > 0 and rights > 0,
                  f'left {lefts} / right {rights}')
        entry['invariants'] = invariant_check(project, recs[1:], label)
        out[label] = entry
    results['trace'] = out


# --- 4. Spectrum of an isolated ring -------------------------------------------------

def dft_bins(x, K):
    N = len(x)
    try:
        import numpy as np
        X = np.fft.fft(np.asarray(x, dtype=np.float64))[:K] / N
        return [complex(v) for v in X]
    except ImportError:
        tw = [cmath.exp(-2j * math.pi * s / N) for s in range(N)]
        out = []
        for k in range(K):
            acc = 0j
            idx = 0
            for s in range(N):
                acc += x[s] * tw[idx]
                idx += k
                if idx >= N:
                    idx -= N
            out.append(acc / N)
        return out


def test_spectrum():
    SR, BPM, M, WARM = 48000, 120, 6000, 8
    PULSE = pulse(0.001, 0.05)
    base = compose_project('techno', 42, '')
    net = base['nets']['hihat']
    project = {'name': 'ring', 'tempo': BPM, 'nets': {'hihat': net}, 'initialMutes': []}
    ring = ring_of(parse_project(project)['nets']['hihat'])
    n = ring['n']
    N = n * M
    dc = {'kind': 'dc', 'carrier': 'dc', 'pulse': PULSE, 'level': 1}
    t0 = time.time()
    y, _ = render_offline(project, seconds=(WARM + 1) * N / SR, sr=SR,
                          runner_opts={'master': 'linear', 'voices': {'hihat': dc}})
    win = y[WARM * N:(WARM + 1) * N]
    K = 48
    pred = ring_spectrum(ring['weights'], PULSE, M, SR, K)
    got = dft_bins(win, K)
    max_err = max(abs(a - b) for a, b in zip(got, pred))
    max_mag = max(abs(b) for b in pred)
    rel = max_err / max_mag
    check('spectrum: DFT of rendered ring gate = P̂(k)·DFT(place weights)', rel < 1e-5,
          f"n={n}, weights=[{' '.join(f'{w:.2f}' for w in ring['weights'])}], K={K}, "
          f'max rel err {rel:.2e} ({time.time() - t0:.1f}s)')
    # Closed form vs IIR, every sample of one period.
    worst = 0.0
    tick_sec = M / SR
    for u in range(N):
        j = (u // M + 1) % n
        g = envelope(ring, j, (u % M) / M, PULSE, tick_sec)
        worst = max(worst, abs(g - win[u]))
    check('closed form: envelope(ring, marking, phase) = rendered gate', worst < 1e-6,
          f'max abs err {worst:.2e} over {N} samples')
    results['spectrum'] = {'ring': 'techno/42 hihat', 'n': n, 'K': K, 'max_rel_err': rel,
                           'closed_form_max_abs_err': worst}


# --- 5. Audio vs the JS offline render ---------------------------------------------------

def test_audio(seconds, tmp):
    out = {}
    overall = 0.0
    for c in AUDIO_CASES:
        label = case_name(*c)
        project = compose_project(*c)
        pj = os.path.join(tmp, 'a.json')
        with open(pj, 'w') as f:
            json.dump(project, f)
        f32 = os.path.join(tmp, 'js.f32')
        node('render.mjs', pj, seconds, f32, 48000)
        js = array('f')
        with open(f32, 'rb') as f:
            js.frombytes(f.read())
        if sys.byteorder != 'little':
            js.byteswap()
        t0 = time.time()
        py, runner = render_offline(project, seconds=seconds, sr=48000)
        wall = time.time() - t0
        n = min(len(py), len(js))
        diffs = [abs(py[i] - js[i]) for i in range(n)]
        mx = max(diffs) if diffs else 0.0
        at = diffs.index(mx) if diffs else 0
        exact = sum(1 for d in diffs if d == 0)
        peak = max(abs(v) for v in js) if n else 0
        overall = max(overall, mx)
        check(f'audio: {label}: {seconds}s, max abs err ≤ {AUDIO_TOL:g}',
              len(py) == len(js) and mx <= AUDIO_TOL,
              f'max abs err {mx:.3e} at sample {at}, {exact}/{n} samples bit-identical, '
              f'peak {peak:.3f}, {len(runner.lanes)} lanes, py {wall:.1f}s wall')
        out[label] = {'seconds': seconds, 'samples': n, 'max_abs_err': mx, 'at_sample': at,
                      'bit_identical_samples': exact, 'lanes': len(runner.lanes), 'py_wall_s': round(wall, 1)}
    results['audio'] = out
    results['audio_max_abs_err'] = overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticks', type=int, default=1200)
    ap.add_argument('--audio-seconds', type=float, default=3)
    ap.add_argument('--skip-audio', action='store_true')
    ap.add_argument('--only-audio', action='store_true')
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        if not a.only_audio:
            test_prng()
            test_traces(a.ticks, tmp)
            test_spectrum()
        if not a.skip_audio:
            test_audio(a.audio_seconds, tmp)
    results['failures'] = failures
    name = 'results-audio.json' if a.only_audio else 'results.json'
    with open(os.path.join(HERE, name), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{len(failures)} check(s) failed" if failures else '\nall parity checks passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
