"""Compare two describe IRs: what changed per section and part.

The refine loop: describe → critique → edit generator params → describe →
diff. Pattern letters are per-IR, so comparison is by grid content: a bar
counts as changed when its grid or its notes differ.
"""

from .describe import _fmt_grid


def _bar_content(ir, b, row):
    letter = ir['bars'][b]['rows'].get(row)
    if letter is None:
        return None
    p = ir['patterns'][row][letter]
    return p['grid'], tuple(p['notes'])


def diff_ir(a, b, max_examples=2):
    out = {'meta': {}, 'rows': {'added': [], 'removed': [], 'changed': {}}, 'sections': []}
    for k in ('tempo', 'bars', 'root', 'scale', 'name'):
        if a['meta'].get(k) != b['meta'].get(k):
            out['meta'][k] = [a['meta'].get(k), b['meta'].get(k)]
    ra, rb = set(a['rows']), set(b['rows'])
    out['rows']['added'] = sorted(rb - ra)
    out['rows']['removed'] = sorted(ra - rb)
    for r in sorted(ra & rb):
        ch = {k: [a['rows'][r][k], b['rows'][r][k]] for k in ('instrument', 'channel')
              if a['rows'][r][k] != b['rows'][r][k]}
        if ch:
            out['rows']['changed'][r] = ch
    sa = [(s['name'], s['n_bars']) for s in a['sections']]
    sb = [(s['name'], s['n_bars']) for s in b['sections']]
    if sa != sb:
        out['structure'] = [sa, sb]
    for i in range(max(len(a['sections']), len(b['sections']))):
        A = a['sections'][i] if i < len(a['sections']) else None
        B = b['sections'][i] if i < len(b['sections']) else None
        if A is None or B is None:
            out['sections'].append({'index': i + 1, 'only_in': 'b' if A is None else 'a',
                                    'name': (A or B)['name']})
            continue
        sec = {'index': i + 1, 'name': A['name'] if A['name'] == B['name'] else f"{A['name']}→{B['name']}",
               'parts': {}}
        rows = list(dict.fromkeys(list(A['parts']) + list(B['parts'])))
        n = min(A['n_bars'], B['n_bars'])
        for r in rows:
            pa, pb = A['parts'].get(r), B['parts'].get(r)
            if pa is None or pb is None:
                sec['parts'][r] = {'status': 'enters' if pa is None else 'drops out'}
                continue
            changed, examples = 0, []
            for k in range(n):
                ca = _bar_content(a, A['bars'][0] - 1 + k, r)
                cb = _bar_content(b, B['bars'][0] - 1 + k, r)
                if ca != cb:
                    changed += 1
                    if len(examples) < max_examples:
                        examples.append({'bar': k + 1, 'a': ca, 'b': cb})
            d = {}
            if changed:
                d['bars_changed'] = f'{changed}/{n}'
                d['examples'] = examples
            for m in ('density', 'odd16', 'offbeat8', 'activity'):
                if abs(pa[m] - pb[m]) > 1e-9:
                    d[m] = [pa[m], pb[m]]
            if pa.get('notes') != pb.get('notes'):
                d['notes'] = [pa.get('notes'), pb.get('notes')]
            if d:
                sec['parts'][r] = d
        if A['hits_per_bar'] != B['hits_per_bar']:
            sec['hits_per_bar'] = [A['hits_per_bar'], B['hits_per_bar']]
        if sec['parts'] or 'hits_per_bar' in sec or A['name'] != B['name']:
            out['sections'].append(sec)
    return out


def _fmt_content(c):
    if c is None:
        return '(silent)'
    g, notes = c
    return _fmt_grid(g) + (' ' + ' '.join(notes) if notes else '')


def render_diff_text(d, la='a', lb='b'):
    L = [f'# beats-ir diff  {la} → {lb}']
    for k, (x, y) in d['meta'].items():
        L.append(f'  {k}: {x} → {y}')
    rw = d['rows']
    if rw['added']:
        L.append('  parts added: ' + ' '.join(rw['added']))
    if rw['removed']:
        L.append('  parts removed: ' + ' '.join(rw['removed']))
    for r, ch in rw['changed'].items():
        L.append(f'  {r}: ' + ', '.join(f'{k} {x}→{y}' for k, (x, y) in ch.items()))
    if 'structure' in d:
        fa = ' '.join(f'{n}:{b}' for n, b in d['structure'][0])
        fb = ' '.join(f'{n}:{b}' for n, b in d['structure'][1])
        L.append(f'  structure: {fa}\n          → {fb}')
    if not d['sections']:
        L.append('  sections: no score changes')
    for s in d['sections']:
        if 'only_in' in s:
            L.append(f"§{s['index']} {s['name']}: only in {s['only_in']}")
            continue
        hp = f"  hits/bar {s['hits_per_bar'][0]}→{s['hits_per_bar'][1]}" if 'hits_per_bar' in s else ''
        L.append(f"§{s['index']} {s['name']}{hp}")
        for r, p in s['parts'].items():
            if 'status' in p:
                L.append(f'  {r:<10} {p["status"]}')
                continue
            bits = []
            if 'bars_changed' in p:
                bits.append(f"{p['bars_changed']} bars changed")
            for m in ('density', 'odd16', 'offbeat8', 'activity'):
                if m in p:
                    bits.append(f'{m} {p[m][0]:g}→{p[m][1]:g}')
            if 'notes' in p:
                bits.append(f"notes {' '.join(p['notes'][0] or [])} → {' '.join(p['notes'][1] or [])}")
            L.append(f'  {r:<10} ' + '; '.join(bits))
            for e in p.get('examples', []):
                L.append(f"    bar {e['bar']:<3} {_fmt_content(e['a'])}")
                L.append(f"    {'→':<7} {_fmt_content(e['b'])}")
    return '\n'.join(L) + '\n'
