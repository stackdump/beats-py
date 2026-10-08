"""Audio side of the describe IR: per-bar rows from a real render, aligned to
the score grid. Optional — needs the `[analysis]` extra (numpy, scipy;
pyloudnorm for LUFS) and ffmpeg to decode webm. Everything degrades: without
the deps describe still prints the score and says why the audio was skipped.

Run it on the *mastered* audio (e.g. beats.bitwrap.io/audio/<cid>.webm),
not on beats-py's own synth, which has no FX, sidechain, swing or macros.

Pipeline (bands as in beats-bitwrap-io scripts/analyze-audio.py):

  decode → mono 44.1k → 5 zero-phase Butterworth bands
         → windowed RMS envelopes (hop 256 ≈ 5.8 ms)
         → onset novelty = positive 2-frame rise of the band's dB envelope
  align  : impulse trains from the score (kick → kick band, hats → high band,
           snare/clap → himid) cross-correlated with the novelty, over a
           global offset and a ±3% tempo scale; the earliest near-best peak wins
  per bar: E (full-band level), 5 band levels (0-9, 4 dB a step below that
           band's loudest bar), detected onsets quantised to 16ths in the
           kick and high bands, score-vs-onset match per drum row with the
           median ms offset, and the low-end pump (kick peak vs the low band
           between kicks; small = bass masking the kick)
"""

import math
import os
import shutil
import subprocess
import tempfile
import urllib.request

SR = 44100
HOP = 256
BANDS = [(20, 80), (80, 250), (250, 2000), (2000, 6000), (6000, 16000)]
BAND_LABELS = ['sub', 'low', 'lomid', 'himid', 'high']
KICK_BAND = (35, 150)
DB_STEP = 4.0          # one level digit = 4 dB
TOL_STEPS = 0.4        # onset/score match window, in 16ths
SCALE_RANGE = 0.03
SCALE_STEP = 0.001

# Which detector hears which drum kind.
KIND_BAND = {'kick': 'kick', 'hat': 'high', 'openhat': 'high', 'snare': 'himid', 'clap': 'himid'}


def missing():
    """None if the analysis deps are importable, else a one-line reason."""
    try:
        import numpy  # noqa: F401
        import scipy.signal  # noqa: F401
    except ImportError as e:
        return f'audio analysis needs the [analysis] extra (pip install "beats-py[analysis]"): {e}'
    return None


def fetch(src, cache_dir=None):
    """Local path for `src`; http(s) URLs are downloaded once into cache_dir."""
    if not src.startswith(('http://', 'https://')):
        return src
    cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), 'beats-py-audio')
    os.makedirs(cache_dir, exist_ok=True)
    name = os.path.basename(src.split('?', 1)[0]) or 'audio'
    path = os.path.join(cache_dir, name)
    if not os.path.exists(path):
        tmp = path + '.part'
        with urllib.request.urlopen(src, timeout=60) as r, open(tmp, 'wb') as f:
            shutil.copyfileobj(r, f)
        os.replace(tmp, path)
    return path


def decode(path, sr=SR):
    import numpy as np
    if shutil.which('ffmpeg'):
        p = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-f', 'f32le', '-ac', '1',
                            '-ar', str(sr), '-'], capture_output=True, check=True)
        return np.frombuffer(p.stdout, dtype=np.float32).copy(), sr
    import wave
    with wave.open(path, 'rb') as w:
        if w.getsampwidth() != 2:
            raise RuntimeError('ffmpeg not found; only 16-bit PCM wav can be read without it')
        ch, rate = w.getnchannels(), w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float32) / 32768
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, rate


def load_features(src):
    x, sr = decode(fetch(src))
    feats = features(x, sr)
    feats['lufs'] = _lufs(x, sr)
    return feats


def _band(x, sr, lo, hi):
    from scipy.signal import butter, sosfiltfilt
    hi = min(hi, sr * 0.45)
    sos = butter(4, [lo, hi], btype='bandpass', fs=sr, output='sos')
    return sosfiltfilt(sos, x)


def _env(x, win):
    """Centred windowed RMS, sampled every HOP samples."""
    import numpy as np
    c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    n = len(x) // HOP
    centres = np.arange(n) * HOP + HOP // 2
    a = np.clip(centres - win // 2, 0, len(x))
    b = np.clip(centres + win // 2, 0, len(x))
    return np.sqrt((c[b] - c[a]) / np.maximum(b - a, 1) + 1e-12)


def _db(v):
    import numpy as np
    return 20 * np.log10(np.maximum(v, 1e-9))


def _novelty(env):
    import numpy as np
    d = _db(env)
    nov = np.zeros_like(d)
    nov[2:] = np.maximum(0, d[2:] - d[:-2])
    floor = d.max() - 45
    nov[d < floor] = 0
    return nov


def _onsets(nov, fps, min_gap_s):
    import numpy as np
    from scipy.signal import find_peaks
    from scipy.ndimage import maximum_filter1d
    w = max(3, int(2 * fps))
    local = maximum_filter1d(nov, size=w)
    thr = np.maximum(3.0, 0.35 * local)
    pk, _ = find_peaks(nov, height=thr, distance=max(1, int(min_gap_s * fps)))
    return pk / fps


def features(x, sr):
    """Band envelopes, novelties and onset times for a mono signal."""
    fps = sr / HOP
    out = {'fps': fps, 'duration': len(x) / sr}
    out['full'] = _env(x, 1024)
    out['bands'] = [_env(_band(x, sr, lo, hi), 1024 if lo < 250 else 512) for lo, hi in BANDS]
    nov = {}
    kick = _band(x, sr, *KICK_BAND)
    nov['kick'] = _novelty(_env(kick, 1024))
    nov['himid'] = _novelty(_env(_band(x, sr, 2000, 6000), 256))
    nov['high'] = _novelty(_env(_band(x, sr, 6000, 16000), 256))
    out['nov'] = nov
    out['low_env'] = _env(_band(x, sr, 20, 250), 1024)
    out['onsets'] = {k: _onsets(v, fps, 0.05) for k, v in nov.items()}
    return out


def score_hits(ir):
    """Score drum hits → {detector band: [(time_s, row, bar, pos)]}, with the
    tonal bass onsets under 'bass'."""
    m = ir['meta']
    step_s = 60 / (m['tempo'] * 4)
    out = {}
    for b in ir['bars']:
        for row, letter in b['rows'].items():
            info = ir['rows'][row]
            grid = ir['patterns'][row][letter]['grid']
            key = KIND_BAND.get(info['kind'])
            if info['kind'] == 'tonal' and (info['part'].startswith('bass') or info['group'] == 'bass'):
                key = 'bass'
            if not key:
                continue
            for pos, c in enumerate(grid):
                if c in 'Xxg':
                    t = ((b['bar'] - 1) * 16 + pos) * step_s
                    out.setdefault(key, []).append((t, row, b['bar'], pos))
    return out


def align(ir, feats):
    """Global (offset_s, scale, quality) maximising score↔novelty correlation."""
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    fps = feats['fps']
    hits = score_hits(ir)
    n = len(feats['full'])
    weights = {'kick': 1.0, 'high': 0.6, 'himid': 0.4}
    novs = {k: gaussian_filter1d(feats['nov'][k], 1.5) for k in weights}
    for k in novs:
        s = novs[k].std()
        if s > 0:
            novs[k] = (novs[k] - novs[k].mean()) / s
    bar_s = 16 * 60 / (ir['meta']['tempo'] * 4)
    min_lag = -int(0.5 * bar_s * fps)
    max_lag = int(min(8.0, feats['duration'] / 4) * fps)
    size = 1 << int(math.ceil(math.log2(2 * n + max_lag + 1)))
    nov_f = {k: np.fft.rfft(v, size) for k, v in novs.items()}
    best = None
    scales = np.arange(1 - SCALE_RANGE, 1 + SCALE_RANGE + 1e-9, SCALE_STEP)
    for sc in scales:
        total = None
        for k, w in weights.items():
            ts = [h[0] for h in hits.get(k, [])]
            if not ts:
                continue
            train = np.zeros(size)
            idx = np.round(np.array(ts) * sc * fps).astype(int)
            idx = idx[idx < n]
            train[idx] = 1
            # corr[lag] = Σ train[i] · nov[i + lag]
            c = np.fft.irfft(np.conj(np.fft.rfft(train)) * nov_f[k], size) * w / max(1, len(idx))
            total = c if total is None else total + c
        if total is None:
            return None
        lags = np.r_[np.arange(min_lag, 0) % size, np.arange(0, max_lag + 1)]
        vals = total[lags]
        i = int(np.argmax(vals))
        if best is None or vals[i] > best[0]:
            best = (vals[i], sc, lags, vals)
    peak, sc, lags, vals = best
    signed = np.where(lags > size // 2, lags - size, lags)
    # Kick-only scores repeat every beat: take the earliest lag within 5% of best.
    near = np.where(vals >= 0.95 * peak)[0]
    i = near[np.argmin(np.abs(signed[near]))]
    rest = vals[np.abs(signed - signed[i]) > 0.1 * fps]
    second = float(rest.max()) if len(rest) else 0.0
    return {'offset_s': round(float(signed[i]) / fps, 4), 'scale': round(float(sc), 4),
            'tempo': round(ir['meta']['tempo'] / float(sc), 2),
            'peak': round(float(vals[i]), 3), 'peak_ratio': round(float(vals[i]) / second, 3) if second > 0 else None}


def _match(score, onsets, tol_s):
    """Nearest onset to each score time within tol → list of offsets (s) or None."""
    import numpy as np
    out = []
    if len(onsets) == 0:
        return [None] * len(score)
    for t in score:
        j = int(np.searchsorted(onsets, t))
        cands = [onsets[k] for k in (j - 1, j) if 0 <= k < len(onsets)]
        d = min(cands, key=lambda o: abs(o - t)) - t
        out.append(d if abs(d) <= tol_s else None)
    return out


def _level_digits(vals_db, ref_db):
    return [max(0, min(9, int(round(9 - (ref_db - v) / DB_STEP)))) for v in vals_db]


def analyze(ir, src, feats=None):
    """Attach per-bar audio rows to `ir` (as ir['audio']) and return them."""
    import numpy as np
    if feats is None:
        feats = load_features(src)
    fps = feats['fps']
    al = align(ir, feats)
    if al is None:
        raise RuntimeError('score has no drum hits to align against')
    off, sc = al['offset_s'], al['scale']
    step_s = 60 / (ir['meta']['tempo'] * 4) * sc
    bar_s = 16 * step_s
    hits = score_hits(ir)
    tol = TOL_STEPS * step_s
    # Per detector: aligned score times, matched offsets.
    matches = {}
    for key in ('kick', 'high', 'himid'):
        hs = hits.get(key, [])
        ts = [off + h[0] * sc for h in hs]
        ms = _match(ts, feats['onsets'][key], tol)
        matches[key] = list(zip(hs, ts, ms))
    nbars = ir['meta']['bars']
    rows = []
    full_db, band_db = [], [[] for _ in BANDS]
    for b in range(nbars):
        a = off + b * bar_s
        i0, i1 = int(a * fps), int((a + bar_s) * fps)
        if i0 < 0 or i1 > len(feats['full']):
            rows.append(None)
            full_db.append(None)
            for lst in band_db:
                lst.append(None)
            continue
        full_db.append(float(_db(np.sqrt(np.mean(feats['full'][i0:i1] ** 2)))))
        for k, env in enumerate(feats['bands']):
            band_db[k].append(float(_db(np.sqrt(np.mean(env[i0:i1] ** 2)))))
        rows.append({'bar': b + 1})
    ref_full = max(v for v in full_db if v is not None) if any(v is not None for v in full_db) else 0
    refs = [max((v for v in lst if v is not None), default=0) for lst in band_db]
    for b, r in enumerate(rows):
        if r is None:
            continue
        a = off + b * bar_s
        r['E'] = _level_digits([full_db[b]], ref_full)[0]
        r['bands'] = ''.join(str(_level_digits([band_db[k][b]], refs[k])[0]) for k in range(len(BANDS)))
        for key, name in (('kick', 'kick_onsets'), ('high', 'high_onsets')):
            g = ['.'] * 16
            for t in feats['onsets'][key]:
                if a - step_s / 2 <= t < a + bar_s - step_s / 2:
                    g[min(15, int(round((t - a) / step_s)))] = 'x'
            r[name] = ''.join(g)
        r['match'] = {}
        for key, lst in matches.items():
            per_row = {}
            for (h, t, d) in lst:
                if h[2] == b + 1:
                    per_row.setdefault(h[1], []).append(d)
            for row, ds in per_row.items():
                got = [d for d in ds if d is not None]
                r['match'][row] = {'hit': len(got), 'of': len(ds),
                                   'ms': round(float(np.median(got)) * 1000) if got else None}
        r['pump_db'] = _pump(feats, [t for (h, t, d) in matches['kick'] if h[2] == b + 1 and d is not None],
                            step_s)
    rows = [r for r in rows if r is not None]
    covered = len(rows)
    summary = _summary(ir, rows, matches, step_s)
    audio = {'src': src, 'duration_s': round(feats['duration'], 2), 'lufs': feats.get('lufs'),
             'align': al, 'bars_covered': covered,
             'audio_bars': int(max(0, feats['duration'] - off) // bar_s),
             'band_labels': BAND_LABELS, 'rows': rows, 'summary': summary}
    ir['audio'] = audio
    return audio


def _pump(feats, kick_times, step_s):
    """Mean dB of kick-band peak just after each kick over the low band in the
    gap before the next beat. Low = the low end never clears between kicks."""
    import numpy as np
    if not kick_times:
        return None
    fps, env = feats['fps'], feats['low_env']
    vals = []
    for t in kick_times:
        i = int(t * fps)
        pk = env[i:i + int(0.06 * fps)]
        gap = env[i + int(0.6 * 4 * step_s * fps) - int(0.04 * fps):i + int(4 * step_s * fps) - int(0.02 * fps)]
        if len(pk) and len(gap):
            vals.append(float(_db(pk.max()) - _db(np.sqrt(np.mean(gap ** 2)))))
    return round(float(np.mean(vals)), 1) if vals else None


def _summary(ir, rows, matches, step_s):
    import numpy as np
    out = {}
    for key, lst in matches.items():
        per_row = {}
        for (h, t, d) in lst:
            per_row.setdefault(h[1], []).append((h[3], d))
        for row, ds in per_row.items():
            got = [d for _, d in ds if d is not None]
            e = {'match': round(len(got) / len(ds), 3), 'n': len(ds)}
            if got:
                e['ms_median'] = round(float(np.median(got)) * 1000, 1)
                q1, q3 = np.percentile(got, [25, 75])
                e['ms_iqr'] = round(float(q3 - q1) * 1000, 1)
                odd = [d for p, d in ds if d is not None and p % 2 == 1]
                even = [d for p, d in ds if d is not None and p % 2 == 0]
                if len(odd) >= 4 and len(even) >= 4:
                    e['swing_ms'] = round(float(np.median(odd) - np.median(even)) * 1000, 1)
            out[row] = e
    pumps = [r['pump_db'] for r in rows if r.get('pump_db') is not None]
    if pumps:
        out['_pump_db'] = round(float(np.median(pumps)), 1)
    return out


def _lufs(x, sr):
    try:
        import pyloudnorm as pyln
        return round(float(pyln.Meter(sr).integrated_loudness(x.astype('float64'))), 1)
    except Exception:
        return None


def render_audio_text(ir):
    a = ir['audio']
    al = a['align']
    L = []
    lufs = f"  {a['lufs']} LUFS" if a.get('lufs') is not None else ''
    L.append(f"AUDIO  {os.path.basename(a['src'])}  {a['duration_s']}s{lufs}  "
             f"offset {al['offset_s']:+.3f}s  tempo {al['tempo']:g} (×{al['scale']:.3f})  "
             f"align peak/2nd {al['peak_ratio']}")
    if a['audio_bars'] > ir['meta']['bars']:
        L.append(f"  audio runs {a['audio_bars']} bars; score has {ir['meta']['bars']} "
                 f"(rows past the score are not shown)")
    s = a['summary']
    parts = []
    for row, e in s.items():
        if row.startswith('_'):
            continue
        t = f"{row} {int(e['match'] * 100)}%"
        if 'ms_median' in e:
            t += f" {e['ms_median']:+.0f}ms±{e['ms_iqr'] / 2:.0f}"
        if 'swing_ms' in e:
            t += f" swing {e['swing_ms']:+.0f}ms"
        parts.append(t)
    L.append('  score→onset match: ' + '; '.join(parts))
    if '_pump_db' in s:
        L.append(f"  low-end pump (kick peak over bass gap) median {s['_pump_db']} dB")
    L.append("  E/bands: 0-9, 4 dB a step below each band's loudest bar; bands = sub low lomid himid high;")
    L.append("  onset grids = detected onsets quantised to 16ths (kick band 35-150 Hz, high 6-16 kHz);")
    L.append("  match = score hits found within ±0.4 step, median ms (+ = audio late)")
    L.append(f"  {'bars':<8} {'sec':<4} E {'bands':<5}  {'kick-band onsets':<19}  {'high-band onsets':<19}  match  pump")

    def key(r):
        return (r['E'], r['bands'], r['kick_onsets'], r['high_onsets'],
                tuple(sorted((k, v['hit'], v['of']) for k, v in r['match'].items())))

    rows = a['rows']
    i = 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and key(rows[j + 1]) == key(rows[i]) and \
                _sec(ir, rows[j + 1]['bar']) == _sec(ir, rows[i]['bar']):
            j += 1
        r = rows[i]
        label = f"{r['bar']}" if i == j else f"{r['bar']}-{rows[j]['bar']}×{j - i + 1}"
        si = _sec(ir, r['bar'])
        sec = f"§{si + 1}" if si is not None else ''
        ms = []
        for row, v in r['match'].items():
            off = [rows[k]['match'][row]['ms'] for k in range(i, j + 1)
                   if row in rows[k]['match'] and rows[k]['match'][row]['ms'] is not None]
            o = f"{sorted(off)[len(off) // 2]:+d}" if off else ''
            ms.append(f"{row} {v['hit']}/{v['of']}{o}")
        pump = f"{r['pump_db']}dB" if r.get('pump_db') is not None else ''
        L.append(f"  {label:<8} {sec:<4} {r['E']} {r['bands']}  {_grid(r['kick_onsets'])}  "
                 f"{_grid(r['high_onsets'])}  {' '.join(ms)}  {pump}".rstrip())
        i = j + 1
    return L


def _sec(ir, bar):
    for i, s in enumerate(ir['sections']):
        if s['bars'][0] <= bar <= s['bars'][1]:
            return i
    return None


def _grid(g):
    return '|'.join(g[i:i + 4] for i in range(0, 16, 4))
