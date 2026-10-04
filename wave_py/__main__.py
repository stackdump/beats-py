"""CLI.

  beats-py render --project file.json --seconds N --out file.wav
  beats-py trace  --project file.json --ticks N --out trace.jsonl
  beats-py render --genre G --seed S [--structure X] --seconds N --out file.wav   (needs node)
  beats-py describe (--project f.json | --genre G --seed S [--structure X])
                    [--format text|json] [--audio file|URL] [--stats stats.json] [--bars N]
  beats-py calibrate --genre G --seeds N [--structure X] --out stats.json
  beats-py diff A B       (A, B: IR json, project json, or genre:seed[:structure])
  beats-py plot --view raster|wave|mel|bands|residual [--audio f|URL] [--bars A-B] --out f.png

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
    p = sub.add_parser('describe', help='text IR of the score (+ per-bar audio rows with --audio)')
    p.add_argument('--genre', default='techno')
    p.add_argument('--seed', default='42')
    p.add_argument('--structure', default='')
    p.add_argument('--project', default='')
    p.add_argument('--format', choices=('text', 'json'), default='text')
    p.add_argument('--audio', default='', help='wav/webm path or URL of a real render to align')
    p.add_argument('--stats', default='', help='calibration stats.json for percentiles')
    p.add_argument('--bars', type=int, default=0, help='score length (default: structure, or audio)')
    p.add_argument('--out', default='')
    p = sub.add_parser('calibrate', help='corpus percentiles for describe --stats')
    p.add_argument('--genre', default='techno')
    p.add_argument('--seeds', type=int, default=20)
    p.add_argument('--seed-start', type=int, default=1)
    p.add_argument('--structure', default='standard')
    p.add_argument('--projects', nargs='*', default=None, help='calibrate from project files instead')
    p.add_argument('--out', required=True)
    p = sub.add_parser('diff', help='what changed between two IRs')
    p.add_argument('a')
    p.add_argument('b')
    p.add_argument('--format', choices=('text', 'json'), default='text')
    p = sub.add_parser('plot', help='PNG views (needs the [viz] extra)')
    p.add_argument('--view', choices=('raster', 'wave', 'mel', 'bands', 'residual'), default='raster')
    p.add_argument('--genre', default='techno')
    p.add_argument('--seed', default='42')
    p.add_argument('--structure', default='')
    p.add_argument('--project', default='')
    p.add_argument('--audio', default='')
    p.add_argument('--bars', default='', help='A-B to zoom')
    p.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if a.cmd in ('describe', 'calibrate', 'diff', 'plot'):
        try:
            return _ir_cmd(a)
        except (RuntimeError, ValueError, OSError) as e:
            print(f'beats-py: {e}', file=sys.stderr)
            return 2
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


def _emit(text, out):
    if out:
        with open(out, 'w', encoding='utf-8') as f:
            f.write(text)
        print(out)
    else:
        sys.stdout.write(text)


def _ir_cmd(a):
    import json
    if a.cmd == 'describe':
        from .describe import describe
        _, text, notes = describe(a.project or None, a.genre, int(a.seed), a.structure,
                                  bars=a.bars or None, audio=a.audio or None,
                                  stats=a.stats or None, fmt=a.format)
        for n in notes:
            print(f'beats-py: {n}', file=sys.stderr)
        _emit(text if text.endswith('\n') else text + '\n', a.out)
        return 0
    if a.cmd == 'calibrate':
        from .calibrate import calibrate, summary_text
        st = calibrate(a.genre, a.seeds, a.structure, a.seed_start, a.projects,
                       progress=lambda s: print(f'  {s}', file=sys.stderr))
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(st, f, separators=(',', ':'))
        sys.stdout.write(summary_text(st))
        print(a.out)
        return 0
    if a.cmd == 'plot':
        from .viz import plot
        print(plot(a.view, a.out, project=a.project or None, genre=a.genre, seed=int(a.seed),
                   structure=a.structure, audio=a.audio or None, bars=a.bars or None))
        return 0
    from .describe import load_ir
    from .diff import diff_ir, render_diff_text
    d = diff_ir(load_ir(a.a), load_ir(a.b))
    _emit(json.dumps(d, indent=1) + '\n' if a.format == 'json' else render_diff_text(d, a.a, a.b), '')
    return 0


if __name__ == '__main__':
    sys.exit(main())
