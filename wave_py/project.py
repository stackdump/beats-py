"""Port of the parts of public/lib/pflow.js the executor consumes:
parseProject / parseNetBundle / buildArcIndex / resetState.

Two JS behaviours matter for exact parity and are reproduced on purpose:

- Object key order. JS enumerates integer-like keys ("0", "17") first in
  ascending numeric order, then string keys in insertion order. Net, place
  and transition order drive firing order, so `js_keys` mirrors it.
- Math.round (half toward +inf), not Python's banker's rounding.
"""

import math


def _is_array_index(k):
    # Canonical numeric string for a uint32 < 2^32 - 1 (no leading zeros).
    if not k or not k.isdigit() or not k.isascii():
        return False
    if len(k) > 1 and k[0] == '0':
        return False
    return int(k) < 0xFFFFFFFF


def js_keys(d):
    """Keys of a JSON object in JS own-property enumeration order."""
    keys = list(d.keys())
    idx = sorted((k for k in keys if _is_array_index(k)), key=int)
    rest = [k for k in keys if not _is_array_index(k)]
    return idx + rest


def js_round(x):
    r = math.floor(x)
    return r + 1 if x - r >= 0.5 else r


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def get_string(m, key, default):
    v = m.get(key)
    return v if isinstance(v, str) else default


def get_float(m, key, default):
    v = m.get(key)
    return v if _num(v) else default


def get_int(m, key, default):
    v = m.get(key)
    return js_round(v) if _num(v) else default


def get_float_array(m, key, default):
    v = m.get(key)
    if isinstance(v, list):
        return [x if _num(x) else 0 for x in v]
    return default


def weight_sum(w):
    if not w:
        return 1
    s = 0
    for x in w:
        s += x
    return s


def _is_obj(v):
    return isinstance(v, dict)


class NetBundle:
    __slots__ = ('places', 'transitions', 'arcs', 'track', 'role', 'riff_group',
                 'bindings', 'control_bindings', 'state', 'input_arcs', 'output_arcs')

    def transition_labels(self):
        return list(self.transitions.keys())


def _binding(m, track):
    return {
        'note': get_int(m, 'note', 60),
        'channel': get_int(m, 'channel', track['channel']),
        'velocity': get_int(m, 'velocity', track['defaultVelocity']),
        'duration': get_int(m, 'duration', 100),
        'durationSteps': get_int(m, 'durationSteps', 0),
    }


def _control(c):
    return {
        'action': get_string(c, 'action', 'toggle-track'),
        'targetNet': get_string(c, 'targetNet', ''),
        'targetNote': get_int(c, 'targetNote', 0),
    }


def parse_net_bundle(data):
    nb = NetBundle()
    nb.role = get_string(data, 'role', 'music')
    nb.riff_group = get_string(data, 'riffGroup', '')
    t = data.get('track') if _is_obj(data.get('track')) else {}
    nb.track = {
        'channel': get_int(t, 'channel', 1),
        'defaultVelocity': get_int(t, 'defaultVelocity', 100),
        'instrument': get_string(t, 'instrument', ''),
    }
    if isinstance(t.get('group'), str) and t['group']:
        nb.track['group'] = t['group']

    nb.places, nb.transitions, nb.bindings, nb.control_bindings = {}, {}, {}, {}
    places = data.get('places')
    if _is_obj(places):
        for pid in js_keys(places):
            pd = places[pid] if _is_obj(places[pid]) else {}
            nb.places[pid] = {'initial': get_float_array(pd, 'initial', [0])}
    trans = data.get('transitions')
    if _is_obj(trans):
        for tid in js_keys(trans):
            td = trans[tid] if _is_obj(trans[tid]) else {}
            nb.transitions[tid] = {}
            if _is_obj(td.get('midi')):
                nb.bindings[tid] = _binding(td['midi'], nb.track)
            if _is_obj(td.get('control')):
                nb.control_bindings[tid] = _control(td['control'])
    nb.arcs = []
    if isinstance(data.get('arcs'), list):
        for a in data['arcs']:
            a = a if _is_obj(a) else {}
            inh = a.get('inhibit')
            nb.arcs.append({
                'source': get_string(a, 'source', ''),
                'target': get_string(a, 'target', ''),
                'weight': get_float_array(a, 'weight', [1]),
                'inhibit': inh if isinstance(inh, bool) else False,
            })
    # Re-parse fallback (sibling bindings / controlBindings maps).
    if _is_obj(data.get('bindings')):
        for tid in js_keys(data['bindings']):
            m = data['bindings'][tid]
            if tid not in nb.bindings and _is_obj(m):
                nb.bindings[tid] = _binding(m, nb.track)
    if _is_obj(data.get('controlBindings')):
        for tid in js_keys(data['controlBindings']):
            c = data['controlBindings'][tid]
            if tid not in nb.control_bindings and _is_obj(c):
                nb.control_bindings[tid] = _control(c)

    # buildArcIndex: inputs keyed by arc.target, outputs by arc.source.
    nb.input_arcs, nb.output_arcs = {}, {}
    for a in nb.arcs:
        ca = (a['source'], a['target'], weight_sum(a['weight']), bool(a['inhibit']))
        nb.input_arcs.setdefault(a['target'], []).append(ca)
        nb.output_arcs.setdefault(a['source'], []).append(ca)
    # resetState
    nb.state = {}
    for pid, p in nb.places.items():
        s = 0
        for x in (p['initial'] or [0]):
            s += x
        nb.state[pid] = s
    return nb


def parse_project(data):
    proj = {
        'name': get_string(data, 'name', 'Untitled'),
        'tempo': get_float(data, 'tempo', 120),
        'nets': {},
        'initialMutes': [],
        'structure': [],
    }
    nets = data.get('nets')
    if _is_obj(nets):
        for nid in js_keys(nets):
            if _is_obj(nets[nid]):
                proj['nets'][nid] = parse_net_bundle(nets[nid])
    if isinstance(data.get('structure'), list):
        for s in data['structure']:
            if _is_obj(s):
                proj['structure'].append({'name': get_string(s, 'name', ''),
                                          'steps': js_round(get_float(s, 'steps', 0))})
    if isinstance(data.get('initialMutes'), list):
        proj['initialMutes'] = [v for v in data['initialMutes'] if isinstance(v, str)]
    return proj
