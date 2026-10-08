"""A legible text IR of a track's score, for an LLM to read, critique and edit.

The score is the marking trace (trace.py): every audible MIDI fire, in tick
order. One tick is one 16th-note step (PPQ = 4 in synth.py; the runner fires
the first tick at sample 0), so step = tick - 1 and a bar is 16 steps.

Fires are mapped to rows — a *part* (the net's riffGroup, else its id, so the
slot variants kick-0/kick-1 read as one "kick") plus, on drum channels, the
drum kind from synth.drum_kind — then binned into bars and grouped into the
project's structure sections. Each row's distinct one-bar patterns get letters
(A, B, …) in order of first appearance, so a section reads as a sequence:
`kick A×8`, `hihat A A B B`.

Grid alphabet (one char per 16th, `|` every beat):

  X accent (vel >= 115)   x hit   g ghost (vel < 80)   - tonal sustain   . rest

Prior art: Libretto (bar/voice/onset-slot grammar, corpus-calibrated stats),
BEAT (uniform step-grid tokens), SAR-LM (interpretable text features).

stdlib only. The audio side is analysis.py; corpus percentiles are calibrate.py.
"""

import json

from .net import compile_project, tick as tick_graph
from .synth import is_drum_channel, drum_kind

IR_VERSION = 1
STEPS_PER_BAR = 16
BEAT = 4
ACCENT, GHOST = 115, 80
DEFAULT_LOOP_BARS = 32
MAX_BARS = 512
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def note_name(n):
    return f'{NOTE_NAMES[n % 12]}{n // 12 - 1}'


def _letters(i):
    s = ''
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _fmt_grid(grid):
    return '|'.join(grid[i:i + BEAT] for i in range(0, len(grid), BEAT))


def _seq_text(seq, max_unit=4):
    """Compress a bar sequence: A A B C B C B → 'A×2 (B C)×2 B'.

    Greedy: at each position take the unit (1..max_unit bars) whose repeats
    cover the most bars; ties go to the shorter unit.
    """
    out, i, n = [], 0, len(seq)
    while i < n:
        best_k, best_r = 1, 1
        for k in range(1, max_unit + 1):
            if i + 2 * k > n:
                break
            unit = seq[i:i + k]
            if k > 1 and len(set(unit)) == 1:
                continue
            r = 1
            while seq[i + r * k:i + (r + 1) * k] == unit:
                r += 1
            if r > 1 and k * r > best_k * best_r:
                best_k, best_r = k, r
        unit = seq[i:i + best_k]
        body = ' '.join(unit)
        if best_r == 1:
            out.append(body)
        else:
            out.append(f'({body})×{best_r}' if best_k > 1 else f'{body}×{best_r}')
        i += best_k * best_r
    return ' '.join(out)


def sections_of(proj, total_steps):
    secs, s = [], 0
    for sec in proj['structure']:
        if sec['steps'] <= 0:
            continue
        secs.append({'name': sec['name'], 'start_step': s, 'steps': sec['steps']})
        s += sec['steps']
    if not secs:
        secs = [{'name': 'loop', 'start_step': 0, 'steps': total_steps}]
    return secs


def _row_key(n, t):
    """(row name, part, drum kind | None)."""
    part = n.bundle.riff_group or n.id
    b = n.bundle.bindings[n.trans_ids[t]]
    ch = b['channel'] or n.bundle.track['channel']
    if is_drum_channel(ch):
        return part, drum_kind(n.note[t])
    return part, None


def collect(project, steps):
    """Run the trace for `steps` ticks → per-row hits, mute/control events."""
    g = compile_project(project)
    hits = {}            # row -> list of (step, vel, note, dur_steps)
    rowinfo = {}         # row -> {part, kind, nets, instrument, channel}
    ctl = []             # (step, action, target)
    mute_track = []      # per step: frozenset of muted net idx
    stopped_at = None
    step_ms = 60000 / (g.tempo * BEAT)
    for step in range(steps):
        tick_graph(g)
        for ni, t in g.fired:
            n = g.nets[ni]
            if n.bundle.role == 'control':
                continue
            part, kind = _row_key(n, t)
            row = (part, kind)
            info = rowinfo.get(row)
            if info is None:
                tr = n.bundle.track
                info = rowinfo[row] = {'part': part, 'kind': kind or 'tonal', 'nets': [],
                                       'instrument': tr.get('instrument') or '',
                                       'channel': n.bundle.bindings[n.trans_ids[t]]['channel'],
                                       'group': tr.get('group', '')}
            if n.id not in info['nets']:
                info['nets'].append(n.id)
            ds = n.dur_steps[t] or (n.dur_ms[t] / step_ms if n.dur_ms[t] else 1)
            hits.setdefault(row, []).append((step, n.vel[t], n.note[t], ds))
        for ni, t in g.controls:
            n = g.nets[ni]
            c = n.bundle.control_bindings.get(n.trans_ids[t])
            if c:
                ctl.append((step, c['action'], c['targetNet']))
        mute_track.append(frozenset(i for i, m in enumerate(g.muted) if m))
        if g.stop_requested and stopped_at is None:
            stopped_at = step
    # Rows: drum parts that only ever play one kind keep the bare part name.
    kinds_per_part = {}
    for (part, kind) in rowinfo:
        kinds_per_part.setdefault(part, set()).add(kind)
    named = {}
    for (part, kind), info in rowinfo.items():
        name = part if len(kinds_per_part[part]) == 1 else f'{part}.{kind}'
        named[name] = (info, hits[(part, kind)])
    # Parts that never fired but exist (muted all song): note them too.
    silent_parts = []
    fired_parts = {p for p, _ in rowinfo}
    for n in g.nets:
        if n.bundle.role == 'control' or not any(n.midi):
            continue
        part = n.bundle.riff_group or n.id
        if part not in fired_parts and part not in silent_parts:
            silent_parts.append(part)
    return g, named, ctl, mute_track, stopped_at, silent_parts


def _bar_pattern(evs, kind):
    """Events within one bar [(pos, vel, note, dur)] → (grid, notes tuple)."""
    grid = ['.'] * STEPS_PER_BAR
    notes = {}
    for pos, vel, note, dur in evs:
        if kind != 'tonal':
            c = 'X' if vel >= ACCENT else 'g' if vel < GHOST else 'x'
            if grid[pos] in '.-' or (c == 'X'):
                grid[pos] = c
            continue
        grid[pos] = 'x'
        notes.setdefault(pos, set()).add(note)
        for k in range(1, max(1, int(round(dur)))):
            if pos + k < STEPS_PER_BAR and grid[pos + k] == '.':
                grid[pos + k] = '-'
    toks = tuple('+'.join(note_name(x) for x in sorted(notes[p])) for p in sorted(notes))
    return ''.join(grid), toks


def _metrics(grids, bars_total):
    """Rhythm metrics over a list of bar grids (None = silent bar)."""
    active = [gr for gr in grids if gr]
    onsets = [i for gr in active for i, c in enumerate(gr) if c in 'Xxg']
    n = len(onsets)
    m = {
        'active_bars': len(active),
        'activity': round(len(active) / bars_total, 3) if bars_total else 0,
        'hits': n,
        'density': round(n / len(active), 2) if active else 0,
        'onbeat': round(sum(1 for i in onsets if i % 4 == 0) / n, 3) if n else 0,
        'offbeat8': round(sum(1 for i in onsets if i % 4 == 2) / n, 3) if n else 0,
        'odd16': round(sum(1 for i in onsets if i % 2 == 1) / n, 3) if n else 0,
        'accent': round(sum(1 for gr in active for c in gr if c == 'X') / n, 3) if n else 0,
        'ghost': round(sum(1 for gr in active for c in gr if c == 'g') / n, 3) if n else 0,
    }
    return m


def build_ir(project, source=None, bars=None):
    """→ IR dict (see README "Describe IR"). `bars` overrides the length."""
    from .project import parse_project
    proj = parse_project(project)
    struct_steps = sum(s['steps'] for s in proj['structure'] if s['steps'] > 0)
    if bars:
        steps = int(bars) * STEPS_PER_BAR
    elif struct_steps:
        steps = struct_steps
    else:
        steps = DEFAULT_LOOP_BARS * STEPS_PER_BAR
    steps = min(steps, MAX_BARS * STEPS_PER_BAR)
    g, rows, ctl, mute_track, stopped_at, silent_parts = collect(project, steps)
    if stopped_at is not None:
        steps = min(steps, stopped_at + 1)
    nbars = -(-steps // STEPS_PER_BAR)
    tempo = g.tempo
    secs = sections_of(proj, steps)
    # Truncate sections to the rendered length.
    out_secs = []
    for s in secs:
        if s['start_step'] >= steps:
            break
        end = min(s['start_step'] + s['steps'], steps)
        out_secs.append({'name': s['name'], 'start_step': s['start_step'], 'steps': end - s['start_step']})
    for s in out_secs:
        s['bar_start'] = s['start_step'] // STEPS_PER_BAR
        s['bar_end'] = -(-(s['start_step'] + s['steps']) // STEPS_PER_BAR)   # exclusive

    # Per row: bar → pattern letter.
    patterns, bar_rows = {}, [dict() for _ in range(nbars)]
    row_meta = {}
    for name in sorted(rows, key=lambda r: _row_order(rows[r][0])):
        info, evs = rows[name]
        per_bar = {}
        for step, vel, note, dur in evs:
            if step >= steps:
                continue
            per_bar.setdefault(step // STEPS_PER_BAR, []).append((step % STEPS_PER_BAR, vel, note, dur))
        pats, index = {}, {}
        for b in range(nbars):
            if b not in per_bar:
                continue
            key = _bar_pattern(per_bar[b], info['kind'])
            if key not in index:
                letter = _letters(len(index))
                index[key] = letter
                pats[letter] = {'grid': key[0], 'notes': list(key[1]),
                                'hits': sum(1 for c in key[0] if c in 'Xxg'), 'first_bar': b + 1}
            bar_rows[b][name] = index[key]
        patterns[name] = pats
        row_meta[name] = {k: info[k] for k in ('part', 'kind', 'instrument', 'channel', 'group', 'nets')}

    def grids_for(name, b0, b1):
        return [patterns[name][bar_rows[b][name]]['grid'] if name in bar_rows[b] else None
                for b in range(b0, b1)]

    def notes_for(name, b0, b1):
        ns = set()
        for b in range(b0, b1):
            if name in bar_rows[b]:
                for tok in patterns[name][bar_rows[b][name]]['notes']:
                    ns.update(tok.split('+'))
        return sorted(ns, key=_note_sort)

    sections = []
    prev_active = set()
    nets_by_part = {}
    for name, m in row_meta.items():
        for nid in m['nets']:
            nets_by_part[nid] = m['part']
    for s in out_secs:
        b0, b1 = s['bar_start'], s['bar_end']
        nb = b1 - b0
        parts = {}
        for name in row_meta:
            seq = [bar_rows[b].get(name, '.') for b in range(b0, b1)]
            if all(x == '.' for x in seq):
                continue
            m = _metrics(grids_for(name, b0, b1), nb)
            entry = {'seq': seq, **m}
            if row_meta[name]['kind'] == 'tonal':
                entry['notes'] = notes_for(name, b0, b1)
            parts[name] = entry
        active = set(parts)
        hits = sum(p['hits'] for p in parts.values())
        events = {}
        explicit = []
        for step, action, target in ctl:
            if s['start_step'] <= step < s['start_step'] + s['steps']:
                key = {'activate-slot': 'slot', 'set-feel': 'feel'}.get(action, action)
                events[key] = events.get(key, 0) + 1
                if action in ('mute-track', 'unmute-track', 'toggle-track', 'stop-transport',
                              'mute-note', 'unmute-note', 'toggle-note'):
                    explicit.append({'bar': step // STEPS_PER_BAR + 1, 'step': step % STEPS_PER_BAR + 1,
                                     'action': action, 'target': target})
        sections.append({
            'name': s['name'], 'bars': [b0 + 1, b1], 'n_bars': nb, 'steps': s['steps'],
            'hits_per_bar': round(hits / nb, 1) if nb else 0,
            'parts': parts,
            'enter': sorted(active - prev_active, key=_row_rank(row_meta)),
            'exit': sorted(prev_active - active, key=_row_rank(row_meta)),
            'controls': events, 'control_events': explicit[:12],
        })
        prev_active = active

    totals = {}
    for name, meta in row_meta.items():
        m = _metrics(grids_for(name, 0, nbars), nbars)
        m['patterns'] = len(patterns[name])
        if meta['kind'] == 'tonal':
            ns = notes_for(name, 0, nbars)
            m['notes'] = ns
            midi = [_note_num(x) for x in ns]
            m['range'] = [note_name(min(midi)), note_name(max(midi))] if midi else []
            m['span'] = (max(midi) - min(midi)) if midi else 0
        totals[name] = m
    # Score-side kick/bass collisions: bass onsets on kick steps.
    kb = _kick_bass(rows, steps)

    meta = {
        'ir': IR_VERSION,
        'name': proj['name'],
        'source': source or {},
        'tempo': tempo,
        'root': note_name(project['rootNote']) if isinstance(project.get('rootNote'), int) else None,
        'scale': project.get('scaleName'),
        'steps_per_bar': STEPS_PER_BAR,
        'bars': nbars,
        'steps': steps,
        'seconds': round(steps * 60 / (tempo * BEAT), 2),
        'stopped': stopped_at is not None,
        'silent_parts': silent_parts,
        'initial_mutes': proj['initialMutes'],
    }
    return {'meta': meta, 'rows': row_meta, 'patterns': patterns, 'sections': sections,
            'bars': [{'bar': i + 1, 'section': _section_at(sections, i + 1), 'rows': r}
                     for i, r in enumerate(bar_rows)],
            'totals': totals, 'kick_bass': kb}


_ORDER = ['kick', 'snare', 'clap', 'hat', 'openhat']
_TONAL_ORDER = ['bass', 'melody', 'arp', 'harmony', 'pad', 'lead']


def _row_order(info):
    if info['kind'] != 'tonal':
        return (0, _ORDER.index(info['kind']) if info['kind'] in _ORDER else 9, info['part'])
    p = info['part']
    for i, k in enumerate(_TONAL_ORDER):
        if p.startswith(k) or info.get('group') == k:
            return (1, i, p)
    return (2, 0, p)


def _row_rank(row_meta):
    order = list(row_meta)
    return lambda n: order.index(n) if n in order else 999


def _note_num(tok):
    i = len(tok) - 1
    while i > 0 and (tok[i - 1].isdigit() or tok[i - 1] == '-'):
        i -= 1
    return NOTE_NAMES.index(tok[:i]) + 12 * (int(tok[i:]) + 1)


def _note_sort(tok):
    try:
        return _note_num(tok)
    except (ValueError, IndexError):
        return 0


def _section_at(sections, bar):
    for i, s in enumerate(sections):
        if s['bars'][0] <= bar <= s['bars'][1]:
            return i
    return None


def _kick_bass(rows, steps):
    kick = set()
    bass = []
    for name, (info, evs) in rows.items():
        if info['kind'] == 'kick':
            kick.update(e[0] for e in evs if e[0] < steps)
        elif info['kind'] == 'tonal' and (info['part'].startswith('bass') or info.get('group') == 'bass'):
            bass.extend(e[0] for e in evs if e[0] < steps)
    if not bass:
        return None
    on = sum(1 for s in bass if s in kick)
    return {'bass_onsets': len(bass), 'on_kick': on, 'ratio': round(on / len(bass), 3)}


# --- Text rendering ---------------------------------------------------------

def _pct(stats, name, metric, value, info=None):
    if not stats:
        return ''
    from .calibrate import percentile_of
    p = percentile_of(stats, name, metric, value, info)
    return f'(p{p})' if p is not None else ''


def _fmt_time(sec):
    return f'{int(sec // 60)}:{int(round(sec % 60)):02d}'


def render_text(ir, stats=None, max_patterns=16):
    m = ir['meta']
    src = m['source']
    head = ' '.join(f'{k}={v}' for k, v in src.items() if v not in ('', None))
    L = []
    key = f"{m['root']} {m['scale']}" if m.get('root') else ''
    L.append(f"# beats-ir/{m['ir']} {head} | {m['tempo']:g} BPM {key} | {m['bars']} bars "
             f"{_fmt_time(m['seconds'])} | 16 steps/bar")
    L.append("# grid: 1 char = 1/16, | = beat; X accent x hit g ghost - sustain . rest; "
             "letters = that row's distinct 1-bar patterns; ×N = repeats")
    if stats:
        L.append(f"# (pNN) = percentile vs {stats.get('genre', '?')} corpus of {stats.get('n', '?')} seeds")
    L.append('')
    L.append('PARTS')
    for name, r in ir['rows'].items():
        t = ir['totals'][name]
        desc = r['kind'] if r['kind'] != 'tonal' else (r['group'] or 'tonal')
        extra = f" range {'-'.join(t['range'])}" if t.get('range') else ''
        L.append(f"  {name:<10} {desc:<8} {r['instrument']:<12} ch{r['channel']:<3} "
                 f"active {t['active_bars']}/{m['bars']} bars{extra} "
                 f"{_pct(stats, name, 'activity', t['activity'], r)}".rstrip())
    if m['silent_parts']:
        L.append(f"  never sounds: {' '.join(m['silent_parts'])}")
    L.append('')
    L.append('PATTERNS')
    for name, pats in ir['patterns'].items():
        items = list(pats.items())
        for letter, p in items[:max_patterns]:
            notes = ' ' + ' '.join(p['notes']) if p['notes'] else ''
            L.append(f"  {name:<10} {letter:<2} {_fmt_grid(p['grid'])}  {p['hits']:>2}{notes}")
        if len(items) > max_patterns:
            L.append(f"  {name:<10} …  +{len(items) - max_patterns} more patterns (--format json for all)")
    L.append('')
    L.append('SECTIONS')
    for i, s in enumerate(ir['sections'], 1):
        b0, b1 = s['bars']
        L.append(f"§{i} {s['name']}  bars {b0}-{b1} ({s['n_bars']})  {s['hits_per_bar']} hits/bar")
        chg = []
        if s['enter']:
            chg.append('in: ' + ' '.join(s['enter']))
        if s['exit']:
            chg.append('out: ' + ' '.join(s['exit']))
        if chg:
            L.append('  ' + '  '.join(chg))
        for name, p in s['parts'].items():
            r = ir['rows'][name]
            d = f"{p['density']:g}/bar{_pct(stats, name, 'density', p['density'], r)}"
            rh = f"on {int(p['onbeat'] * 100)}% off {int(p['offbeat8'] * 100)}% 16th {int(p['odd16'] * 100)}%"
            notes = f"  notes {' '.join(p['notes'][:12])}{'…' if len(p['notes']) > 12 else ''}" if p.get('notes') else ''
            L.append(f"  {name:<10} {_seq_text(p['seq']):<22} {d:<14} {rh}{notes}")
        if s['controls']:
            ev = ' '.join(f'{k}×{v}' for k, v in sorted(s['controls'].items()))
            ex = ' '.join(f"{e['action'].replace('-track', '')} {e['target']}@b{e['bar']}"
                          for e in s['control_events'] if e['action'] != 'stop-transport')
            L.append(f"  ctl        {ev}{'  ' + ex if ex else ''}")
    L.append('')
    L.append('TOTALS')
    for name, t in ir['totals'].items():
        r = ir['rows'][name]
        bits = [f"{t['density']:g}/bar{_pct(stats, name, 'density', t['density'], r)}",
                f"16th {int(t['odd16'] * 100)}%{_pct(stats, name, 'odd16', t['odd16'], r)}",
                f"off {int(t['offbeat8'] * 100)}%",
                f"{t['patterns']} patterns{_pct(stats, name, 'patterns', t['patterns'], r)}"]
        if 'span' in t:
            bits.append(f"{len(t['notes'])} pitches span {t['span']}st{_pct(stats, name, 'span', t['span'], r)}")
        L.append(f"  {name:<10} " + '  '.join(bits))
    kb = ir.get('kick_bass')
    if kb:
        L.append(f"  kick/bass  {kb['on_kick']}/{kb['bass_onsets']} bass onsets land on a kick "
                 f"({int(kb['ratio'] * 100)}%){_pct(stats, '_track', 'kick_bass', kb['ratio'])}")
    if stats:
        L.append(f"  length     {m['bars']} bars{_pct(stats, '_track', 'bars', m['bars'])}")
    if ir.get('audio'):
        from .analysis import render_audio_text
        L.append('')
        L.extend(render_audio_text(ir))
    return '\n'.join(L).rstrip() + '\n'


def render(ir, fmt='text', stats=None):
    if fmt == 'json':
        return json.dumps(ir, indent=1, ensure_ascii=False)
    return render_text(ir, stats)


# --- One entry point for the CLI and the MCP server --------------------------

def resolve_project(project=None, genre='techno', seed=42, structure=''):
    """→ (project json, source meta). A path or a dict wins over genre/seed."""
    from .compose import compose_project, load_project
    if isinstance(project, dict):
        return project, {'project': project.get('name', '(inline)')}
    if project:
        return load_project(project), {'project': project}
    return compose_project(genre, seed, structure or ''), \
        {'genre': genre, 'seed': seed, 'structure': structure or 'loop'}


def describe(project=None, genre='techno', seed=42, structure='', bars=None, audio=None,
             stats=None, fmt='text'):
    """→ (ir, rendered text|json, notes). `stats` is a path or a loaded dict.

    With audio and a loop project (no structure), the score is run for as
    many bars as the audio holds. Audio failures never sink the score: the
    reason lands in `notes` and in ir['audio_error'].
    """
    proj, src = resolve_project(project, genre, seed, structure)
    notes = []
    feats = None
    if audio:
        from . import analysis
        why = analysis.missing()
        if why:
            notes.append(f'audio skipped: {why}')
        else:
            try:
                feats = analysis.load_features(audio)
            except Exception as e:      # decode / download failure
                notes.append(f'audio skipped: {e}')
        if feats and not bars and not parse_structure_steps(proj):
            bars = max(1, int(feats['duration'] // (16 * 60 / (_tempo(proj) * BEAT))))
    ir = build_ir(proj, src, bars=bars)
    if feats:
        from . import analysis
        try:
            analysis.analyze(ir, audio, feats)
        except Exception as e:
            notes.append(f'audio analysis failed: {e}')
    if notes:
        ir['notes'] = notes
    if isinstance(stats, str) and stats:
        from .calibrate import load_stats
        stats = load_stats(stats)
    if fmt == 'json':
        out = render(ir, 'json')
    else:
        out = render_text(ir, stats) + ''.join(f'# {n}\n' for n in notes)
    return ir, out, notes


def parse_structure_steps(project):
    from .project import parse_project
    return sum(s['steps'] for s in parse_project(project)['structure'] if s['steps'] > 0)


def _tempo(project):
    t = project.get('tempo')
    return t if isinstance(t, (int, float)) and t > 0 else 120


def load_ir(spec, bars=None):
    """An IR from an IR json file, a project json file, or 'genre:seed[:structure]'."""
    import os
    if os.path.isfile(spec):
        with open(spec, encoding='utf-8') as f:
            d = json.load(f)
        if 'meta' in d and 'sections' in d:
            return d
        return build_ir(d, {'project': spec}, bars=bars)
    parts = spec.split(':')
    if len(parts) in (2, 3):
        proj, src = resolve_project(None, parts[0], int(parts[1]), parts[2] if len(parts) == 3 else '')
        return build_ir(proj, src, bars=bars)
    raise ValueError(f'{spec}: not a file and not genre:seed[:structure]')
