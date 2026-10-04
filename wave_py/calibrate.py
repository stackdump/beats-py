"""Corpus calibration (the Libretto idea): compose N seeds of a genre, describe
each, and keep the distribution of every score metric, so `describe --stats`
can say "hihat 16th 41% (p73)" instead of a bare number an LLM cannot judge.

Score side only, stdlib only. Composition needs node + $BEATS_PUBLIC
(compose.py); `projects=` calibrates from project JSON files instead.

stats.json:
  {"stats": 1, "genre", "structure", "n", "seeds",
   "rows":  {row:  {metric: [sorted values]}},   per-row whole-track metrics
   "kinds": {kind: {metric: [sorted values]}},   fallback when a row name is new
   "track": {metric: [sorted values]}}           bars, kick_bass, section bars
"""

import json

from .compose import compose_project, load_project
from .describe import build_ir

ROW_METRICS = ('density', 'activity', 'odd16', 'offbeat8', 'onbeat', 'accent', 'ghost', 'patterns', 'span')


def _add(d, metric, v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        d.setdefault(metric, []).append(v)


def accumulate(stats, ir):
    for name, t in ir['totals'].items():
        info = ir['rows'][name]
        for bucket in (stats['rows'].setdefault(name, {}), stats['kinds'].setdefault(_kind_key(info), {})):
            for m in ROW_METRICS:
                _add(bucket, m, t.get(m))
    tr = stats['track']
    _add(tr, 'bars', ir['meta']['bars'])
    if ir.get('kick_bass'):
        _add(tr, 'kick_bass', ir['kick_bass']['ratio'])
    for s in ir['sections']:
        _add(tr, f"section:{s['name']}", s['n_bars'])
        _add(tr, 'hits_per_bar', s['hits_per_bar'])


def _kind_key(info):
    return info['kind'] if info['kind'] != 'tonal' else (info.get('group') or 'tonal')


def calibrate(genre='techno', seeds=20, structure='standard', seed_start=1, projects=None, progress=None):
    stats = {'stats': 1, 'genre': genre, 'structure': structure, 'rows': {}, 'kinds': {}, 'track': {},
             'seeds': [], 'source': 'compose'}
    if projects:
        stats['source'] = 'projects'
        for path in projects:
            accumulate(stats, build_ir(load_project(path), {'project': path}))
            stats['seeds'].append(path)
            if progress:
                progress(path)
    else:
        for seed in range(seed_start, seed_start + seeds):
            proj = compose_project(genre, seed, structure)
            accumulate(stats, build_ir(proj, {'genre': genre, 'seed': seed, 'structure': structure}))
            stats['seeds'].append(seed)
            if progress:
                progress(seed)
    stats['n'] = len(stats['seeds'])
    for group in (stats['rows'], stats['kinds']):
        for d in group.values():
            for k in d:
                d[k].sort()
    for k in stats['track']:
        stats['track'][k].sort()
    return stats


def load_stats(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _pct(values, v):
    if not values:
        return None
    lo = sum(1 for x in values if x < v)
    eq = sum(1 for x in values if x == v)
    return int(round(100 * (lo + eq / 2) / len(values)))


def percentile_of(stats, row, metric, value, info=None):
    """Percentile of `value` among the corpus values for row/metric (or None)."""
    if row == '_track':
        return _pct(stats['track'].get(metric), value)
    vals = stats['rows'].get(row, {}).get(metric)
    if not vals and info is not None:
        vals = stats['kinds'].get(_kind_key(info), {}).get(metric)
    if not vals or len(vals) < 3:
        return None
    return _pct(vals, value)


def summary_text(stats):
    """Medians and IQR per row — what 'typical' looks like for this genre."""
    def q(v, f):
        return v[min(len(v) - 1, int(f * len(v)))]
    L = [f"# calibration: {stats['genre']} structure={stats['structure']} n={stats['n']} ({stats['source']})"]
    for name, d in stats['rows'].items():
        bits = []
        for m in ('density', 'activity', 'odd16', 'patterns', 'span'):
            v = d.get(m)
            if v:
                bits.append(f"{m} {q(v, .5):g} [{q(v, .25):g}-{q(v, .75):g}]")
        L.append(f"  {name:<10} n={len(d.get('density', []))}  " + '  '.join(bits))
    for k, v in stats['track'].items():
        L.append(f"  {k:<18} median {q(v, .5):g} [{q(v, .25):g}-{q(v, .75):g}]")
    return '\n'.join(L) + '\n'
