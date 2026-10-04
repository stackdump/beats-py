"""MCP server (stdio) over the describe IR: `beats-py-mcp`.

Needs the `[mcp]` extra (the official `mcp` SDK; v2's MCPServer, or v1's
FastMCP). Tools return the text IR by default and every output is bounded —
traces are summarised, never dumped.

  describe   project path | genre+seed+structure, optional audio path/URL, stats
  trace      bounded summary of the marking trace (fires per net, controls, mutes)
  render     beats-py's own synth to a WAV path (≤ 120 s; no FX — see README)
  diff       two IRs (IR json, project json, or "genre:seed[:structure]")
  calibrate  corpus percentiles → stats.json (≤ 50 seeds)
  plot       PNG views of score/audio (needs [viz]); returns the image
"""

import json

try:                                    # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server, Image
except ImportError:                     # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server, Image

MAX_TEXT = 60000
MAX_TRACE_TICKS = 4096
MAX_RENDER_SECONDS = 120
MAX_CAL_SEEDS = 50

INSTRUCTIONS = """beats-py: a legible text IR of generated tracks from the beats.bitwrap.io engine.
Start with describe(genre="techno", seed=N, structure="standard"): a per-part step grid
(16 chars per bar, X accent x hit g ghost - sustain . rest), each part's distinct bar
patterns lettered A, B, … and sections written as letter sequences. Pass audio=<url of
https://beats.bitwrap.io/audio/<cid>.webm> to add per-bar measurements of the real
master aligned to the score, and stats=<stats.json from calibrate> for percentiles.
To refine: describe, critique in techno terms, change genre/seed/structure or edit the
project JSON, describe again, then diff the two."""

mcp = _Server('beats-py', instructions=INSTRUCTIONS)


def _bound(text):
    if len(text) <= MAX_TEXT:
        return text
    return text[:MAX_TEXT] + f'\n… truncated ({len(text) - MAX_TEXT} more chars; narrow with bars= or format)\n'


@mcp.tool()
def describe(project: str = '', genre: str = 'techno', seed: int = 42, structure: str = '',
             audio: str = '', format: str = 'text', stats: str = '', bars: int = 0) -> str:
    """Text IR of a track: parts, lettered 1-bar step patterns, sections as pattern
    sequences, densities/syncopation, and (with audio=path or URL to a wav/webm render)
    per-bar level/band/onset rows aligned to the score. project = path to a project
    JSON; otherwise genre/seed/structure are composed (structure: '' loop, 'standard',
    'extended', …). format 'text' (default) or 'json'. stats = calibrate output path."""
    from .describe import describe as run
    _, out, _ = run(project or None, genre, seed, structure, bars=bars or None,
                    audio=audio or None, stats=stats or None, fmt=format)
    return _bound(out)


@mcp.tool()
def trace(project: str = '', genre: str = 'techno', seed: int = 42, structure: str = '',
          ticks: int = 1024) -> str:
    """Bounded summary of the deterministic marking trace (1 tick = one 16th): fires
    per net, control actions, mute-state changes, stop tick. Never the raw trace."""
    from .describe import resolve_project
    from .net import compile_project, tick
    proj, src = resolve_project(project or None, genre, seed, structure)
    ticks = max(1, min(int(ticks), MAX_TRACE_TICKS))
    g = compile_project(proj)
    fires = [0] * len(g.nets)
    ctl = {}
    changes = []
    prev = list(g.muted)
    stop = None
    for _ in range(ticks):
        tick(g)
        for ni, _t in g.fired:
            fires[ni] += 1
        for ni, t in g.controls:
            n = g.nets[ni]
            c = n.bundle.control_bindings.get(n.trans_ids[t])
            if c:
                ctl[c['action']] = ctl.get(c['action'], 0) + 1
        if g.muted != prev:
            for i, (a, b) in enumerate(zip(prev, g.muted)):
                if a != b and len(changes) < 40:
                    changes.append(f"t{g.tick} {'mute' if b else 'unmute'} {g.nets[i].id}")
            prev = list(g.muted)
        if g.stop_requested and stop is None:
            stop = g.tick
    head = ' '.join(f'{k}={v}' for k, v in src.items())
    L = [f'# trace {head}: {ticks} ticks ({ticks / 16:g} bars) @ {g.tempo:g} BPM, {len(g.nets)} nets',
         f'stop-transport at tick {stop}' if stop else 'no stop-transport in range',
         'fires per net (audible MIDI): ' + ', '.join(
             f'{n.id} {fires[i]}' for i, n in enumerate(g.nets) if fires[i]),
         'silent music nets: ' + (', '.join(n.id for i, n in enumerate(g.nets)
                                            if not fires[i] and any(n.midi)) or '-'),
         'control actions: ' + (', '.join(f'{k}×{v}' for k, v in sorted(ctl.items())) or '-'),
         'mute changes (first 40): ' + ('; '.join(changes) or '-')]
    return _bound('\n'.join(L) + '\n')


@mcp.tool()
def render(out: str, project: str = '', genre: str = 'techno', seed: int = 42,
           structure: str = '', seconds: float = 10, rate: int = 48000) -> str:
    """Render beats-py's forward-model synth (no FX/sidechain/swing/macros) to a WAV
    file at `out`. seconds is capped at 120."""
    from .describe import resolve_project
    from .runner import render_offline
    from .wav import write_wav
    proj, _ = resolve_project(project or None, genre, seed, structure)
    seconds = max(0.1, min(float(seconds), MAX_RENDER_SECONDS))
    samples, runner = render_offline(proj, seconds=seconds, sr=int(rate))
    write_wav(out, samples, int(rate))
    return f'{out}: {seconds:g}s @ {rate} Hz, {len(runner.lanes)} lanes, {runner.graph.tick} ticks'


@mcp.tool()
def diff(a: str, b: str, format: str = 'text') -> str:
    """What changed between two tracks, per section and part. a, b: IR json files
    (describe format=json), project json files, or 'genre:seed[:structure]'."""
    from .describe import load_ir
    from .diff import diff_ir, render_diff_text
    d = diff_ir(load_ir(a), load_ir(b))
    return _bound(json.dumps(d, indent=1) if format == 'json' else render_diff_text(d, a, b))


@mcp.tool()
def calibrate(out: str, genre: str = 'techno', seeds: int = 20, structure: str = 'standard',
              seed_start: int = 1) -> str:
    """Compose `seeds` seeds (≤ 50) of a genre and write metric distributions to `out`
    (use as describe stats=out). Returns the medians/IQR summary."""
    from .calibrate import calibrate as run, summary_text
    st = run(genre, max(1, min(int(seeds), MAX_CAL_SEEDS)), structure, seed_start)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(st, f, separators=(',', ':'))
    return _bound(summary_text(st) + f'written: {out}\n')


@mcp.tool()
def plot(view: str = 'raster', out: str = '', project: str = '', genre: str = 'techno',
         seed: int = 42, structure: str = '', audio: str = '', bars: str = '') -> Image:
    """PNG of one view (needs [viz]): raster (score spikes per part), wave (waveform +
    RMS + score spikes), mel (mel spectrogram), bands (per-bar band heatmap = the IR's
    0-9 numbers), residual (forward-model render vs real master, per-bar band dB
    difference; needs audio). bars='A-B' zooms. Saved to `out` (default: a temp file)."""
    from .viz import plot as run
    path = run(view, out or None, project=project or None, genre=genre, seed=seed,
               structure=structure, audio=audio or None, bars=bars or None)
    return Image(path=path)


def main():
    mcp.run()


if __name__ == '__main__':
    main()
