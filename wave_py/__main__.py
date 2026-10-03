"""CLI.

  beats-py render --project file.json --seconds N --out file.wav
  beats-py trace  --project file.json --ticks N --out trace.jsonl
  beats-py render --genre G --seed S [--structure X] --seconds N --out file.wav   (needs node)

(`python3 -m wave_py` is the same CLI.) --project needs no node; --genre/--seed/
--structure run the JS composer through node from $BEATS_PUBLIC (see compose.py).
"""

import argparse
import sys
import time

from .compose import compose_project, load_project
from .runner import render_offline
from .trace import write_trace
from .wav import write_wav


def _project(a):
    if a.project:
        return load_project(a.project)
    return compose_project(a.genre, a.seed, a.structure or '')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='beats-py', description='Offline Python port of the beats wave engine.')
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('render', 'trace'):
        p = sub.add_parser(name)
        p.add_argument('--genre', default='techno')
        p.add_argument('--seed', default='42')
        p.add_argument('--structure', default='')
        p.add_argument('--project', default='')
        p.add_argument('--out', required=True)
        if name == 'render':
            p.add_argument('--seconds', type=float, default=10)
            p.add_argument('--rate', type=int, default=48000)
        else:
            p.add_argument('--ticks', type=int, default=1000)
    a = ap.parse_args(argv)
    try:
        project = _project(a)
    except RuntimeError as e:
        print(f'beats-py: {e}', file=sys.stderr)
        return 2
    if a.cmd == 'trace':
        write_trace(project, a.ticks, a.out)
        print(f'{a.out}: {a.ticks} ticks, {len(project["nets"])} nets')
        return 0
    t0 = time.time()
    samples, runner = render_offline(project, seconds=a.seconds, sr=a.rate)
    write_wav(a.out, samples, a.rate)
    print(f'{a.out}: {a.seconds}s @ {a.rate} Hz, {len(runner.lanes)} lanes, '
          f'{runner.graph.tick} ticks, {time.time() - t0:.1f}s wall')
    return 0


if __name__ == '__main__':
    sys.exit(main())
