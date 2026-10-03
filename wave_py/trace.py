"""The marking trace — the canonical artifact.

One JSON line per tick, after the tick fires (same shape as
parity/trace.mjs writes from the unmodified JS executor):

  header: {"nets":[ids], "places":[[ids]], "transitions":[[ids]]}
  tick:   {"t":k, "fired":[[net,trans],...], "ctl":[[net,trans],...],
           "muted":[net,...], "mn":[net*128+note,...], "stop":0|1, "m":[[tokens],...]}

`fired` is audible MIDI fires (net un-muted, note un-muted) in firing order;
`ctl` every control transition that fired; `m` the full marking, every net in
project order, places in declaration order. A `stop-transport` sets "stop"
but the trace keeps ticking, so a song's tail is covered too.
"""

import json

from .net import compile_project, tick, marking_ints


def _dumps(o):
    return json.dumps(o, separators=(',', ':'), ensure_ascii=False)


def header(g):
    return {'nets': [n.id for n in g.nets], 'places': [n.place_ids for n in g.nets],
            'transitions': [n.trans_ids for n in g.nets]}


def tick_record(g):
    return {
        't': g.tick,
        'fired': [[ni, t] for ni, t in g.fired],
        'ctl': [[ni, t] for ni, t in g.controls],
        'muted': [i for i, m in enumerate(g.muted) if m],
        'mn': [i for i, m in enumerate(g.muted_notes) if m],
        'stop': 1 if g.stop_requested else 0,
        'm': [marking_ints(n) for n in g.nets],
    }


def iter_trace(project, ticks, graph=None):
    """Yield the header then `ticks` tick records. Pass `graph` to observe it."""
    g = graph or compile_project(project)
    yield header(g)
    for _ in range(ticks):
        tick(g)
        yield tick_record(g)


def write_trace(project, ticks, path):
    with open(path, 'w', encoding='utf-8') as f:
        for rec in iter_trace(project, ticks):
            f.write(_dumps(rec) + '\n')
