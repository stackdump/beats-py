"""Port of public/wave-engine/net.js — the executor, the sequencer worker's
`_advanceOneTick` with `deterministicLoop` on:

  nets in project order; enabled transitions in label order; one winner per
  contested input place via deterministic_rand(tick, str_hash(place));
  controls applied immediately, so a mute fired by an earlier net affects
  later nets in the same tick.

Each compiled net also carries its incidence matrix C (P × T, integers for
every composer net), C[p][t] = Σ w(t→p) − Σ w(p→t), inhibitor arcs excluded.
Firing uses the arc lists rather than C·u because the JS clamps a place at 0
after consuming (a no-op for an enabled transition, but it is the reference).

Not ported, as in the JS: macros (`fire-macro`), loop/seek, drift.
"""

from fractions import Fraction

from .prng import deterministic_rand, str_hash
from .project import parse_project

A_NONE, A_MUTE, A_UNMUTE, A_TOGGLE = 0, 1, 2, 3
A_MUTE_NOTE, A_UNMUTE_NOTE, A_TOGGLE_NOTE = 4, 5, 6
A_SLOT, A_STOP, A_OTHER = 7, 8, 9

ACTIONS = {
    'mute-track': A_MUTE, 'unmute-track': A_UNMUTE, 'toggle-track': A_TOGGLE,
    'mute-note': A_MUTE_NOTE, 'unmute-note': A_UNMUTE_NOTE, 'toggle-note': A_TOGGLE_NOTE,
    'activate-slot': A_SLOT, 'stop-transport': A_STOP,
}

FIRED_CAP = 8192      # same fixed capacities as net.js (ints, i.e. pairs × 2)
CONTROLS_CAP = 1024


def _int16(v):
    v &= 0xFFFF
    return v - 0x10000 if v & 0x8000 else v


class Net:
    pass


def _compile_net(nid, nb, index):
    n = Net()
    n.id, n.index, n.bundle = nid, index, nb
    n.place_ids = list(nb.places.keys())
    p_idx = {p: i for i, p in enumerate(n.place_ids)}
    n.trans_ids = nb.transition_labels()
    n.P, n.T = len(n.place_ids), len(n.trans_ids)
    # ins[t] = [(place, weight, inhibit)], outs[t] = [(place, weight)]
    n.ins, n.outs = [], []
    for t in n.trans_ids:
        n.ins.append([(p_idx.get(src, -1), w, inh) for (src, _tgt, w, inh) in nb.input_arcs.get(t, [])])
        n.outs.append([(p_idx.get(tgt, -1), w) for (_src, tgt, w, _inh) in nb.output_arcs.get(t, [])])
    n.initial = [nb.state.get(p, 0) or 0 for p in n.place_ids]
    n.state = list(n.initial)
    n.place_salt = [str_hash(p) for p in n.place_ids]

    n.midi = [0] * n.T
    n.note = [0] * n.T
    n.vel = [0] * n.T
    n.dur_ms = [0.0] * n.T
    n.dur_steps = [0.0] * n.T
    n.action = [0] * n.T
    n.action_note = [0] * n.T
    n.action_target = []
    for i, t in enumerate(n.trans_ids):
        b = nb.bindings.get(t)
        if b:
            n.midi[i] = 1
            n.note[i] = _int16(b['note'])
            n.vel[i] = _int16(b['velocity'])
            n.dur_ms[i] = b['duration']
            n.dur_steps[i] = b['durationSteps'] or 0
        c = nb.control_bindings.get(t)
        n.action_target.append(c['targetNet'] if c else '')
        if c:
            n.action[i] = ACTIONS.get(c['action'], A_OTHER)
            n.action_note[i] = _int16(c['targetNote'] or 0)
    n.action_target_idx = [-1] * n.T
    n.group = nb.riff_group or ''
    n.group_idx = -1
    n.lane = [-1] * n.T
    return n


class Graph:
    pass


def compile_project(json_obj):
    proj = parse_project(json_obj)
    ids = list(proj['nets'].keys())
    g = Graph()
    g.project = proj
    g.nets = [_compile_net(nid, proj['nets'][nid], i) for i, nid in enumerate(ids)]
    g.by_id = {nid: i for i, nid in enumerate(ids)}
    g.groups = []
    for n in g.nets:
        if n.group and n.group not in g.groups:
            g.groups.append(n.group)
    for n in g.nets:
        n.group_idx = g.groups.index(n.group) if n.group else -1
        for t in range(n.T):
            tgt = n.action_target[t]
            if tgt and tgt in g.by_id:
                n.action_target_idx[t] = g.by_id[tgt]
    g.tempo = proj['tempo'] if proj['tempo'] > 0 else 120
    g.tick = 0
    g.muted = [0] * len(g.nets)
    g.muted_notes = [0] * (len(g.nets) * 128)
    g.muted_groups = [0] * max(1, len(g.groups))
    g.stop_requested = False
    g.fired, g.controls, g.overflow = [], [], 0
    reset_graph(g)
    return g


def reset_graph(g):
    g.tick = 0
    g.stop_requested = False
    g.muted = [0] * len(g.nets)
    g.muted_notes = [0] * (len(g.nets) * 128)
    g.muted_groups = [0] * max(1, len(g.groups))
    for n in g.nets:
        n.state = list(n.initial)
    for nid in g.project['initialMutes']:
        i = g.by_id.get(nid)
        if i is not None:
            g.muted[i] = 1


def _is_enabled(n, t):
    ins = n.ins[t]
    if not ins:
        return False
    s = n.state
    for p, w, inh in ins:
        tok = s[p] if p >= 0 else 0
        if inh:
            if tok >= w:
                return False
        elif tok < w:
            return False
    return True


def _fire(n, t):
    s = n.state
    for p, w, inh in n.ins[t]:
        if not inh and p >= 0:
            s[p] -= w
            if s[p] < 0:
                s[p] = 0
    for p, w in n.outs[t]:
        if p >= 0:
            s[p] += w


def _resolve_conflicts(n, enabled, tick_no):
    place_count = [0] * n.P
    contested = False
    for t in enabled:
        for p, _w, inh in n.ins[t]:
            if not inh and p >= 0:
                place_count[p] += 1
                if place_count[p] > 1:
                    contested = True
    if not contested:
        return enabled
    blocked = set()
    import math
    for p in range(n.P):
        if place_count[p] < 2:
            continue
        consumers = [t for t in enabled for (pp, _w, inh) in n.ins[t] if not inh and pp == p]
        c = len(consumers)
        winner = consumers[math.floor(deterministic_rand(tick_no, n.place_salt[p]) * c)]
        for t in consumers:
            if t != winner:
                blocked.add(t)
    return [t for t in enabled if t not in blocked]


def _apply_control(g, n, t):
    tgt = n.action_target_idx[t]
    a = n.action[t]
    if a == A_MUTE:
        if tgt >= 0: g.muted[tgt] = 1
    elif a == A_UNMUTE:
        if tgt >= 0: g.muted[tgt] = 0
    elif a == A_TOGGLE:
        if tgt >= 0: g.muted[tgt] ^= 1
    elif a in (A_MUTE_NOTE, A_UNMUTE_NOTE, A_TOGGLE_NOTE):
        if tgt >= 0:
            k = tgt * 128 + (n.action_note[t] & 127)
            g.muted_notes[k] = 1 if a == A_MUTE_NOTE else 0 if a == A_UNMUTE_NOTE else g.muted_notes[k] ^ 1
    elif a == A_SLOT:
        if tgt >= 0:
            gi = g.nets[tgt].group_idx
            if gi >= 0:
                for o in g.nets:
                    if o.group_idx == gi and o.index != tgt:
                        g.muted[o.index] = 1
                if not g.muted_groups[gi]:
                    g.muted[tgt] = 0
    elif a == A_STOP:
        g.stop_requested = True
    if len(g.controls) < CONTROLS_CAP:
        g.controls.append((n.index, t))
    else:
        g.overflow += 1


def tick(g, on_note=None):
    """Advance every net one tick. on_note(net, t) for each audible MIDI fire."""
    g.tick += 1
    g.fired, g.controls = [], []
    for ni, n in enumerate(g.nets):
        enabled = [t for t in range(n.T) if _is_enabled(n, t)]
        if len(enabled) > 1:
            enabled = _resolve_conflicts(n, enabled, g.tick)
        for t in enabled:
            _fire(n, t)
            if n.action[t]:
                _apply_control(g, n, t)
            if n.midi[t] and not g.muted[ni] and not g.muted_notes[ni * 128 + (n.note[t] & 127)]:
                if len(g.fired) * 2 < FIRED_CAP:
                    g.fired.append((ni, t))
                else:
                    g.overflow += 1
                if on_note is not None:
                    on_note(n, t)


# --- Incidence matrix and P-invariants (exact, rational) ---------------------

def incidence_matrix(n):
    """C[p][t] = Σ w(t→p) − Σ w(p→t), inhibitor arcs excluded."""
    C = [[0] * n.T for _ in range(n.P)]
    for t in range(n.T):
        for p, w, inh in n.ins[t]:
            if not inh and p >= 0:
                C[p][t] -= w
        for p, w in n.outs[t]:
            if p >= 0:
                C[p][t] += w
    return C


def p_invariants(C):
    """Basis of {y : yᵀC = 0} by exact Gaussian elimination on Cᵀ (Fractions)."""
    # Sparse reduced row echelon form: incidence matrices have ~2 non-zeros
    # per column, so rows are dicts {col: Fraction} and fill-in stays small.
    P = len(C)
    T = len(C[0]) if P else 0
    rows = []
    for t in range(T):
        r = {p: Fraction(C[p][t]) for p in range(P) if C[p][t] != 0}
        if r:
            rows.append(r)
    pivot_rows = {}                      # pivot column → normalised row
    for c in range(P):
        best = next((i for i, r in enumerate(rows) if r.get(c)), None)
        if best is None:
            continue
        r = rows.pop(best)
        piv = r[c]
        r = {k: v / piv for k, v in r.items()}
        for lst in (rows, list(pivot_rows.values())):
            for o in lst:
                f = o.get(c)
                if f:
                    for k, v in r.items():
                        nv = o.get(k, 0) - f * v
                        if nv:
                            o[k] = nv
                        else:
                            o.pop(k, None)
        pivot_rows[c] = r
    basis = []
    for fc in (c for c in range(P) if c not in pivot_rows):
        y = [Fraction(0)] * P
        y[fc] = Fraction(1)
        for pc, r in pivot_rows.items():
            v = r.get(fc)
            if v:
                y[pc] = -v
        basis.append(y)
    return basis


def marking_ints(n):
    """The net's marking with integral values as ints (the canonical trace form)."""
    return [int(x) if float(x).is_integer() else x for x in n.state]
