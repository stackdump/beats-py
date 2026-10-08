"""PNG views of a track, for an LLM (or a person) to look at. `[viz]` extra.

Framing: the waveform is a filtered marked point process,

    x(t) = tanh(g · Σ_k (h_k * μ_k)(t)),

μ_k the event measure of row k (the score spikes the trace emits) and h_k
that instrument's kernel. The views put the two sides of that equation on
one time axis:

  raster    μ — spikes per part over bars (tonal parts as a piano roll),
            section boundaries marked
  wave      x — waveform + RMS envelope with bar/beat grid, μ underneath
  mel       x — mel spectrogram with bar grid
  bands     per-bar band levels (bars × 5 bands), the IR's 0-9 digits
  residual  real master vs beats-py's forward model (render of the same μ
            through h_k, no FX): per-bar band dB difference — what FX,
            sidechain and mastering added or removed

Without --audio, wave/mel/bands use the forward-model render. Images are at
most 1600 px wide, with labelled axes, bar numbers and part names.
"""

import os
import tempfile

DPI = 100
WIDTH_IN = 14.5           # ≤ 1600 px after tight bbox
FONT = 11
MAX_RENDER_S = 180


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': FONT, 'axes.titlesize': FONT + 2, 'axes.labelsize': FONT,
                         'xtick.labelsize': FONT - 1, 'ytick.labelsize': FONT - 1})
    return plt


def parse_bars(spec, nbars):
    if not spec:
        return 1, nbars
    a, _, b = str(spec).partition('-')
    a = max(1, int(a))
    b = min(nbars, int(b)) if b else a
    if b < a:
        raise ValueError(f'bars {spec}: end before start')
    return a, b


def _events(ir, b0, b1):
    """From the IR: [(row, step_abs, char, [midi notes])] for bars b0..b1 (1-based)."""
    from .describe import _note_num
    out = []
    for b in ir['bars'][b0 - 1:b1]:
        for row, letter in b['rows'].items():
            p = ir['patterns'][row][letter]
            notes = list(p['notes'])
            k = 0
            for pos, c in enumerate(p['grid']):
                if c in 'Xxg':
                    ns = [_note_num(t) for t in notes[k].split('+')] if k < len(notes) else []
                    k += 1
                    dur = 1
                    while pos + dur < 16 and p['grid'][pos + dur] == '-':
                        dur += 1
                    out.append((row, (b['bar'] - 1) * 16 + pos, c, ns, dur))
    return out


def _bar_axis(ax, ir, b0, b1, step_s, offset=0.0, label=True):
    """Bar lines (solid), beat lines (dotted), section boundaries (thick) in seconds."""
    for bar in range(b0, b1 + 2):
        t = offset + (bar - 1) * 16 * step_s
        ax.axvline(t, color='0.55', lw=0.8, zorder=0)
        if bar <= b1:
            for beat in (1, 2, 3):
                ax.axvline(t + beat * 4 * step_s, color='0.8', lw=0.5, ls=':', zorder=0)
    for s in ir['sections']:
        if b0 <= s['bars'][0] <= b1 + 1:
            ax.axvline(offset + (s['bars'][0] - 1) * 16 * step_s, color='tab:red', lw=2, zorder=1)
    n = b1 - b0 + 1
    every = 1 if n <= 32 else 2 if n <= 64 else 4
    ticks = [offset + (bar - 1 + 0.5) * 16 * step_s for bar in range(b0, b1 + 1, every)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(bar) for bar in range(b0, b1 + 1, every)] if label else [])
    ax.set_xlim(offset + (b0 - 1) * 16 * step_s, offset + b1 * 16 * step_s)


def _section_labels(ax, ir, b0, b1, step_s, offset=0.0):
    for s in ir['sections']:
        a, e = max(s['bars'][0], b0), min(s['bars'][1], b1)
        if a > e:
            continue
        t = offset + ((a - 1) * 16 + 2) * step_s
        ax.text(t, 1.01, s['name'], transform=ax.get_xaxis_transform(), color='tab:red',
                fontsize=FONT, va='bottom', ha='left')


def _draw_raster(ax, ir, b0, b1, step_s, offset=0.0):
    rows = list(ir['rows'])
    ev = _events(ir, b0, b1)
    import matplotlib
    cmap = matplotlib.colormaps['tab10']
    ylabels = []
    for i, row in enumerate(rows):
        y0 = len(rows) - 1 - i
        ylabels.append((y0 + 0.5, row))
        info = ir['rows'][row]
        mine = [e for e in ev if e[0] == row]
        col = cmap(i % 10)
        if info['kind'] == 'tonal':
            ps = [p for e in mine for p in e[3]] or [60]
            lo, hi = min(ps), max(ps)
            span = max(1, hi - lo)
            for _, st, _c, ns, dur in mine:
                for p in ns:
                    y = y0 + 0.1 + 0.8 * (p - lo) / span
                    ax.plot([offset + st * step_s, offset + (st + dur) * step_s - step_s * 0.15],
                            [y, y], color=col, lw=3, solid_capstyle='butt')
            ylabels[-1] = (y0 + 0.5, f'{row}\n{_nn(lo)}–{_nn(hi)}')
        else:
            for _, st, c, _ns, _d in mine:
                h = 0.85 if c == 'X' else 0.6 if c == 'x' else 0.3
                ax.plot([offset + st * step_s] * 2, [y0 + 0.05, y0 + 0.05 + h], color=col, lw=1.6)
        ax.axhline(y0, color='0.9', lw=0.6)
    ax.set_yticks([y for y, _ in ylabels])
    ax.set_yticklabels([r for _, r in ylabels])
    ax.set_ylim(0, len(rows))


def _nn(n):
    from .describe import note_name
    return note_name(n)


def _source_audio(ir, project, audio, b1, sr=44100):
    """(samples np.float32, sr, offset_s, label). Real audio if given, else the forward model."""
    import numpy as np
    if audio:
        from .analysis import fetch, decode
        x, sr = decode(fetch(audio))
        off = ir.get('audio', {}).get('align', {}).get('offset_s', 0.0)
        return x, sr, off, _short(audio)
    from .runner import render_offline
    step_s = 60 / (ir['meta']['tempo'] * 4)
    secs = min(b1 * 16 * step_s + 0.5, MAX_RENDER_S)
    s, _ = render_offline(project, seconds=secs, sr=sr)
    return np.asarray(s, dtype=np.float32), sr, 0.0, 'beats-py forward model (no FX)'


def _mel(x, sr, n_mels=96, n_fft=2048, hop=512):
    import numpy as np
    from scipy.signal import stft
    f, t, Z = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    P = np.abs(Z) ** 2

    def hz2mel(h):
        return 2595 * np.log10(1 + h / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(30), hz2mel(min(16000, sr / 2)), n_mels + 2))
    fb = np.zeros((n_mels, len(f)))
    for i in range(n_mels):
        l, c, r = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((f - l) / (c - l), (r - f) / (r - c)), 0, None)
    M = fb @ P
    return t, pts[1:-1], 10 * np.log10(M + 1e-10)


def _fig(h_in):
    plt = _plt()
    return plt, plt.figure(figsize=(WIDTH_IN, h_in), dpi=DPI)


def _title(ir, view, b0, b1, extra=''):
    m = ir['meta']
    src = ' '.join(f'{k}={v}' for k, v in m['source'].items())
    return f'{view} · {src} · {m["tempo"]:g} BPM · bars {b0}-{b1}{extra}'


def plot(view='raster', out=None, project=None, genre='techno', seed=42, structure='',
         audio=None, bars=None, ir=None):
    """Render one view to a PNG; returns its path."""
    import numpy as np
    from .describe import describe as run_describe, resolve_project, build_ir
    if view not in ('raster', 'wave', 'mel', 'bands', 'residual'):
        raise ValueError(f'unknown view {view!r}: raster wave mel bands residual')
    if view == 'residual' and not audio:
        raise ValueError('residual needs audio (the real master to compare against)')
    proj, src = resolve_project(project, genre, seed, structure)
    if ir is None:
        if audio:
            ir, _, notes = run_describe(proj, bars=None, audio=audio)
            ir['meta']['source'] = src
            if 'audio' not in ir:
                raise RuntimeError('; '.join(notes) or 'audio analysis failed')
        else:
            ir = build_ir(proj, src)
    b0, b1 = parse_bars(bars, ir['meta']['bars'])
    step_s = 60 / (ir['meta']['tempo'] * 4)
    if out is None:
        fd, out = tempfile.mkstemp(prefix=f'beats-{view}-', suffix='.png')
        os.close(fd)

    if view == 'raster':
        nrows = len(ir['rows'])
        plt, fig = _fig(max(3.5, 0.55 * nrows + 1.6))
        ax = fig.add_subplot(111)
        _draw_raster(ax, ir, b0, b1, step_s)
        _bar_axis(ax, ir, b0, b1, step_s)
        _section_labels(ax, ir, b0, b1, step_s)
        ax.set_xlabel('bar')
        ax.set_title(_title(ir, 'score raster μ', b0, b1), pad=22)

    elif view in ('wave', 'mel'):
        x, sr, off, label = _source_audio(ir, proj, audio, b1)
        sc = ir.get('audio', {}).get('align', {}).get('scale', 1.0) if audio else 1.0
        st = step_s * sc
        t0, t1 = off + (b0 - 1) * 16 * st, off + b1 * 16 * st
        i0, i1 = max(0, int(t0 * sr)), min(len(x), int(t1 * sr))
        seg = x[i0:i1]
        nrows = len(ir['rows'])
        plt, fig = _fig(7.5 if view == 'wave' else 7)
        gs = fig.add_gridspec(2, 1, height_ratios=[2.2, max(1.2, 0.22 * nrows)], hspace=0.08)
        ax = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax)
        if view == 'wave':
            cols = 1600
            n = max(1, len(seg) // cols)
            m = len(seg) // n
            blk = seg[:m * n].reshape(m, n)
            tt = (i0 + np.arange(m) * n + n / 2) / sr
            ax.fill_between(tt, blk.min(axis=1), blk.max(axis=1), color='tab:blue', lw=0, alpha=0.55,
                            label='waveform (min/max)')
            rms = np.sqrt((blk.astype(np.float64) ** 2).mean(axis=1))
            ax.plot(tt, rms, color='tab:orange', lw=1.4, label='RMS')
            ax.plot(tt, -rms, color='tab:orange', lw=1.4)
            ax.set_ylabel('amplitude')
            ax.legend(loc='upper right', fontsize=FONT - 1)
        else:
            t, fc, S = _mel(seg, sr)
            S = np.maximum(S, S.max() - 80)
            mesh = ax.pcolormesh(t + i0 / sr, np.arange(len(fc)), S, shading='auto', cmap='magma')
            yt = [k for k in range(0, len(fc), 12)]
            ax.set_yticks(yt)
            ax.set_yticklabels([f'{fc[k]:.0f}' if fc[k] < 1000 else f'{fc[k] / 1000:.1f}k' for k in yt])
            ax.set_ylabel('mel band (Hz)')
            cb = fig.colorbar(mesh, ax=[ax, ax2], pad=0.01, fraction=0.03)
            cb.set_label('dB')
        _bar_axis(ax, ir, b0, b1, st, off, label=False)
        _section_labels(ax, ir, b0, b1, st, off)
        _draw_raster(ax2, ir, b0, b1, st, off)
        _bar_axis(ax2, ir, b0, b1, st, off)
        ax2.set_xlabel('bar')
        ax.set_title(_title(ir, 'waveform x = tanh(g·Σ h*μ)' if view == 'wave' else 'mel spectrogram',
                            b0, b1, f' · {label}'), pad=22)

    elif view == 'bands':
        if 'audio' not in ir:
            from . import analysis
            x, sr, _, label = _source_audio(ir, proj, None, b1)
            feats = analysis.features(x, sr)
            analysis.analyze(ir, label, feats)
        rows = [r for r in ir['audio']['rows'] if b0 <= r['bar'] <= b1]
        Z = np.array([[int(c) for c in r['bands']] + [r['E']] for r in rows]).T
        plt, fig = _fig(5)
        ax = fig.add_subplot(111)
        bars_x = [r['bar'] for r in rows]
        im = ax.imshow(Z, aspect='auto', cmap='viridis', vmin=0, vmax=9, origin='lower',
                       extent=(bars_x[0] - 0.5, bars_x[-1] + 0.5, -0.5, Z.shape[0] - 0.5))
        labels = ir['audio']['band_labels'] + ['E (full)']
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        if len(rows) <= 40:
            for j, r in enumerate(rows):
                for k in range(Z.shape[0]):
                    ax.text(r['bar'], k, str(Z[k, j]), ha='center', va='center', fontsize=FONT - 2,
                            color='white' if Z[k, j] < 6 else 'black')
        for s in ir['sections']:
            if b0 < s['bars'][0] <= b1:
                ax.axvline(s['bars'][0] - 0.5, color='tab:red', lw=2)
        ax.set_xlabel('bar')
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.set_label('level 0-9 (4 dB/step below band max)')
        ax.set_title(_title(ir, 'per-bar band levels (IR digits)', b0, b1,
                            f" · {_short(ir['audio']['src'])}"))

    else:   # residual
        from . import analysis
        real = ir['audio']
        x, sr, _, _ = _source_audio(ir, proj, None, b1)
        fm = analysis.features(x, sr)
        model = _band_db_rows(ir, fm, 0.0, 1.0, b0, b1)
        al = real['align']
        fr = analysis.features(*analysis.decode(analysis.fetch(audio)))
        master = _band_db_rows(ir, fr, al['offset_s'], al['scale'], b0, b1)
        keep = [k for k in range(len(model)) if model[k] is not None and master[k] is not None]
        if not keep:
            raise RuntimeError('no overlapping bars between render and master')
        A = np.array([master[k] for k in keep]).T
        B = np.array([model[k] for k in keep]).T
        # Remove each signal's overall gain per band so the plot shows shape, not level.
        D = (A - A.mean(axis=1, keepdims=True)) - (B - B.mean(axis=1, keepdims=True))
        bars_x = [b0 + k for k in keep]
        plt, fig = _fig(6.5)
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1.3], hspace=0.35)
        ax = fig.add_subplot(gs[0])
        lim = max(6.0, float(np.abs(D).max()))
        im = ax.imshow(D, aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim, origin='lower',
                       extent=(bars_x[0] - 0.5, bars_x[-1] + 0.5, -0.5, D.shape[0] - 0.5))
        ax.set_yticks(range(len(analysis.BAND_LABELS)))
        ax.set_yticklabels(analysis.BAND_LABELS)
        ax.set_xlabel('bar')
        for s in ir['sections']:
            if b0 < s['bars'][0] <= b1:
                ax.axvline(s['bars'][0] - 0.5, color='k', lw=2)
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.set_label('master − model dB (gain-normalised)')
        ax.set_title(_title(ir, 'residual: real master vs forward model', b0, b1,
                            f" · {_short(real['src'])}"))
        ax3 = fig.add_subplot(gs[1])
        tilt = (A.mean(axis=1) - A.mean()) - (B.mean(axis=1) - B.mean())
        ax3.bar(analysis.BAND_LABELS, tilt, color=['tab:red' if v > 0 else 'tab:blue' for v in tilt])
        ax3.axhline(0, color='k', lw=0.8)
        ax3.set_ylabel('dB')
        ax3.set_title('spectral tilt: master band balance minus model band balance (whole range)')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    width = _png_width(out)
    if width > 1600:                     # long labels widened the tight bbox
        fig.savefig(out, dpi=DPI * 1590 / width, bbox_inches='tight')
    plt.close(fig)
    return out


def _png_width(path):
    with open(path, 'rb') as f:
        return int.from_bytes(f.read(24)[16:20], 'big')


def _short(src, n=28):
    b = os.path.basename(str(src))
    return b if len(b) <= n else b[:n - 9] + '…' + b[-8:]


def _band_db_rows(ir, feats, offset, scale, b0, b1):
    """Per bar (b0..b1): [dB per band] or None where the audio does not cover it."""
    import numpy as np
    from .analysis import _db
    fps = feats['fps']
    bar_s = 16 * 60 / (ir['meta']['tempo'] * 4) * scale
    out = []
    for bar in range(b0, b1 + 1):
        i0 = int((offset + (bar - 1) * bar_s) * fps)
        i1 = int((offset + bar * bar_s) * fps)
        if i0 < 0 or i1 > len(feats['full']):
            out.append(None)
            continue
        out.append([float(_db(np.sqrt(np.mean(env[i0:i1] ** 2)))) for env in feats['bands']])
    return out
