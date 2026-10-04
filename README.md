# beats-py

An offline Python port of the **wave engine** behind
[beats.bitwrap.io](https://beats.bitwrap.io) (the JS lives in
[beats-bitwrap-io](https://github.com/stackdump/beats-bitwrap-io) under
`public/wave-engine/`, served as `?engine=wave`).

A beats track is a set of Petri nets. Each tick, enabled transitions fire,
and firing a MIDI-bound transition strikes a synth voice. This package runs
that executor and that synth in pure Python and writes the results to disk:

- the **marking trace** — the canonical artifact: one JSON line per tick with
  the full integer marking, every audible and control fire, mute state and
  the stop flag;
- **audio** — a projection of the trace, rendered sample by sample to 16-bit
  stereo WAV.

It is **backend-only**: there is no web UI, no real-time playback, and no
MIDI output. Runtime dependencies: none (stdlib only, Python ≥ 3.10).

## Install

```bash
git clone https://github.com/stackdump/beats-py
cd beats-py
python3 -m venv .venv
.venv/bin/pip install -e .            # runtime: stdlib only
.venv/bin/pip install -e '.[test]'    # + pytest and numpy for the tests
.venv/bin/pip install -e '.[analysis,viz,mcp]'   # optional: describe --audio, plot, beats-py-mcp
```

## Usage

```bash
beats-py render --project song.json --seconds 12 --out song.wav
beats-py trace  --project song.json --ticks 1000 --out song.jsonl

# needs node + a beats-bitwrap-io checkout (see below)
beats-py render --genre techno --seed 42 [--structure standard] --seconds 12 --out techno.wav
beats-py trace  --genre jazz   --seed 7                         --ticks 1000 --out jazz.jsonl
```

`python3 -m wave_py ...` is the same CLI. `render` also takes `--rate`
(default 48000). The WAV is written with the stdlib `wave` module using the
JS `encodeWav` quantiser.

From Python:

```python
import json
from wave_py import render_offline
from wave_py.trace import iter_trace

project = json.load(open('song.json'))
samples, runner = render_offline(project, seconds=4, sr=48000)   # array of float32 (mono)
for record in iter_trace(project, ticks=100):                    # header, then one dict per tick
    ...
```

A project file is the JSON that beats.bitwrap.io loads and exports (`nets`
of `places` / `transitions` / `arcs`, plus `tempo`). See `CONFLICT` in
`parity/test_parity.py` for a minimal hand-written one.

## The composer is not ported — node is optional

`(genre, seed, structure) → project JSON` is the composer
(`public/lib/generator/*`, about 5k lines held byte-identical to the Go
generator). It is **not** ported and its JS is **not** vendored here.

- **`--project` needs no node at all.** Importing the package, rendering and
  tracing a project file are pure Python. This is the supported path.
- **`--genre/--seed/--structure` shell out to node**, which imports the
  unmodified composer from a beats-bitwrap-io checkout named by
  `BEATS_PUBLIC`:

  ```bash
  git clone https://github.com/stackdump/beats-bitwrap-io
  export BEATS_PUBLIC=$PWD/beats-bitwrap-io/public
  ```

  Without node or `BEATS_PUBLIC` the CLI exits 2 with a message saying so;
  nothing else is affected.

Node may print a harmless `MODULE_TYPELESS_PACKAGE_JSON` warning when a
`package.json` above the checkout has no `"type"` field.

## Determinism and parity

1. **Trace: exact.** Same project JSON and tick count give the same marking,
   fires, mute state and stop flag at every tick as the JS executor. The JSONL
   is byte-identical to `parity/trace.mjs`. A mismatch is a bug.
2. **Audio: bit-exact in practice, within 1e-5 by contract.** The synthesis
   follows the JS operation order sample by sample (float phase accumulators,
   2048-entry float32 wavetables, IIR gates, RBJ biquads, tanh master). On
   x86-64 Linux, CPython 3.14 vs Node 25, every sample measured came out
   bit-identical (max abs error 0).

   **Caveat:** bit identity depends on the host's C libm (which CPython's
   `math.exp/sin/tanh/pow` call) agreeing with V8's fdlibm port to the last
   ulp. Nothing guarantees that on another platform, libc or Python build; the
   promise there is the 1e-5 tolerance, and the trace stays exact regardless.

Measured on valoper, 2026-10-03, against beats-bitwrap-io `3ebaf60`:

| check | result |
|---|---|
| mulberry32 (8 seeds × 64, incl. negative / > 2³¹ / fractional seeds), strHash (incl. astral-plane UTF-16), deterministicRand, noiseSeed + xorshift32 | bit-exact |
| trace techno/42/standard | exact, byte-identical JSONL, 1200 ticks, 60 nets, stop-transport at tick 897 |
| trace jazz/7/loop | exact, byte-identical, 1200 ticks, 10 nets |
| trace edm/1234/extended | exact, byte-identical, 1200 ticks, 101 nets, 107 control fires |
| trace dnb/99/standard | exact, byte-identical, 1200 ticks, 68 nets, stop at tick 961 |
| trace ambient/3/loop | exact, byte-identical, 1200 ticks, 10 nets |
| trace conflict net (2 transitions, 1 place) | exact; seeded resolution took left 305 / right 295 |
| P-invariants, every tick, every net, all 6 traces | 0 breaches (exact rational). Every music ring has the all-ones invariant (40/40, 10/10, 73/73, 46/46, 10/10, 1/1). The `struct-*` control nets have no invariant (7, 9, 7), matching the JS README |
| FFT of an isolated ring (techno/42 hi-hat, n=8, DC carrier, 48 bins) vs predicted `c_k = P̂(k)·DFT(w)` | max rel err 4.45e-10 |
| closed-form `envelope()` vs rendered gate, one full period | max abs err 2.98e-8 (float32 output) |
| audio techno/42/loop, 10 s, 10 lanes | 480000/480000 samples bit-identical, max abs err 0 |
| audio jazz/7/loop, 10 s, 10 lanes | 480000/480000 bit-identical, max abs err 0 |
| audio edm/1234/standard, 10 s, 53 lanes | 480000/480000 bit-identical, max abs err 0 |
| CLI WAV vs the JS `scripts/wave-render.mjs`, techno/42/standard, 12 s | same sha256 (`92bc55a1…`) |

Pure-Python rendering runs at roughly 2–4 s of wall time per 10 s of audio.

## Describe: a text IR for LLMs

`beats-py describe` turns a track into a compact text intermediate
representation (IR) that an LLM can read, critique in techno terms, and
use to propose generator edits. With `diff` it can then check what those
edits changed. The ideas come from prior work:
Libretto (a bar/voice/onset-slot grammar plus corpus-calibrated stats),
BEAT (uniform step-grid tokens), SAR-LM (interpretable text features
beat opaque embeddings) and LLM2Fx (DSP features in context drive parameter
edits).

```bash
beats-py describe --genre techno --seed 42 --structure standard          # text (default)
beats-py describe --project song.json --format json --out song.ir.json
beats-py describe --genre techno --seed 3976429772 \
    --audio https://beats.bitwrap.io/audio/<cid>.webm                       # + per-bar audio rows
beats-py calibrate --genre techno --seeds 30 --out techno-stats.json
beats-py describe --genre techno --seed 42 --structure standard --stats techno-stats.json
beats-py diff techno:42:standard techno:43:standard                        # or two IR/project files
beats-py plot --view raster|wave|mel|bands|residual [--audio …] [--bars 9-16] --out v.png
```

### Score side (stdlib only)

The score is the marking trace: one tick is one 16th (`PPQ = 4`; the
runner fires tick 1 at sample 0), so step = tick − 1 and a bar is 16
steps. Every audible fire is mapped to a **row**. A row is the net's
`riffGroup`, so the slot variants `kick-0`/`kick-1` read as one `kick`; on
drum channels it also carries the drum kind from `synth.drum_kind`. Fires
are binned into bars, and bars are grouped into the project's `structure`
sections (a loop project has none: it runs `--bars`, default 32, or as many
bars as `--audio` holds).

- **Grid**: one char per 16th, `|` every beat. `X` accent (vel ≥ 115),
  `x` hit, `g` ghost (vel < 80), `-` tonal sustain, `.` rest. Tonal
  patterns list their notes after the grid, and chords are joined with `+`.
- **Patterns**: each row's distinct one-bar patterns are lettered A, B, …
  in order of first appearance. A section is then a sequence of letters,
  compressed so that `A×4` is a run, `(C B D E)×3` is a repeating phrase and
  `.` is a silent bar.
- **Per section**: hits/bar, which parts enter and exit, density, and the
  rhythm split: `on` (on the beat), `off` (8th offbeat), `16th` (odd
  16ths). It also gives the pitch set for tonal parts and the control
  actions, with explicit mutes/unmutes located by bar.
- **Totals**: per-part density, syncopation, pattern count and pitch span,
  and kick/bass collisions (bass onsets landing on a kick step).
- **`--stats`** appends `(pNN)`: the percentile of that value in a corpus
  composed by `calibrate`. The corpus holds whole-track per-part
  distributions, and section values are compared against them too, so read
  a section's `pNN` as indicative.

Real output, `describe --genre techno --seed 42 --structure standard --stats
docs/examples/techno-stats.json` (the full IR is 93 lines; see
`docs/examples/techno-42-standard.ir.txt`):

```
# beats-ir/1 genre=techno seed=42 structure=standard | 128 BPM C3 Major | 56 bars 1:45 | 16 steps/bar
# grid: 1 char = 1/16, | = beat; X accent x hit g ghost - sustain . rest; letters = that row's distinct 1-bar patterns; ×N = repeats
# (pNN) = percentile vs techno corpus of 30 seeds

PARTS
  kick       kick     drums-v8     ch10  active 56/56 bars (p83)
  snare      snare    drums        ch11  active 48/56 bars (p83)
  hihat      hat      drums-v8     ch12  active 56/56 bars (p50)
  bass       bass     reese        ch6   active 49/56 bars range C3-B3 (p17)
  melody     melody   sync-lead    ch4   active 38/56 bars range C4-B4 (p17)
  harmony    harmony  dark-pad     ch7   active 52/56 bars range C3-B3 (p17)
  never sounds: hit1 hit2 hit3 hit4

PATTERNS
  kick       A  X...|X...|X...|X...   4
  snare      A  ....|x...|....|x...   2
  hihat      A  x.xg|.gx.|x.xg|.gx.  10
  bass       A  ....|x-..|x-..|x-..   3 C3 C3 C3
  bass       B  x-..|x-..|x-..|x-..   4 B3 B3 B3 B3
  …

SECTIONS
§1 intro  bars 1-4 (4)  14.0 hits/bar
  in: kick hihat
  kick       A×4                    4/bar(p83)     on 100% off 0% 16th 0%
  hihat      A×4                    10/bar(p0)     on 20% off 40% 16th 40%
  ctl        feel×1
§2 buildup  bars 5-16 (12)  28.2 hits/bar
  in: snare bass melody harmony
  kick       A×12                   4/bar(p83)     on 100% off 0% 16th 0%
  snare      A×12                   2/bar(p50)     on 100% off 0% 16th 0%
  hihat      A×6 B×6                12/bar(p100)   on 25% off 33% 16th 41%
  bass       A (B C)×5 B            3.92/bar(p0)   on 100% off 0% 16th 0%  notes C3 B3
  melody     A (B . . C)×2 B .×2    6.83/bar(p60)  on 56% off 7% 16th 36%  notes C4 D4
  harmony    A (B C)×5 B            2.92/bar(p0)   on 31% off 34% 16th 34%  notes C3 D3 E3 F3 G3 B3
  ctl        feel×1 slot×6 unmute-track×2  unmute snare@b5 unmute harmony@b5
§3 drop  bars 17-28 (12)  32.2 hits/bar
  kick       A×12                   4/bar(p83)     on 100% off 0% 16th 0%
  snare      A×12                   2/bar(p50)     on 100% off 0% 16th 0%
  hihat      C A×5 D B×5            12/bar(p100)   on 25% off 33% 16th 41%
```

### Audio side (`[analysis]` extra)

The rows are measured on a **real master**, e.g. the mastered webm at
`beats.bitwrap.io/audio/<cid>.webm`. beats-py's own synth is not used,
because it has no FX, sidechain, swing or macros. Decoding goes through
ffmpeg into 44.1 kHz mono, then into five zero-phase Butterworth bands (the
same edges as beats-bitwrap-io `scripts/analyze-audio.py`) and windowed RMS
envelopes with a hop of about 5.8 ms. Onsets are positive 2-frame rises of
a band's dB envelope.

- **Alignment**: score impulse trains (kick → 35–150 Hz, hats → 6–16 kHz,
  snare/clap → 2–6 kHz) are cross-correlated with the onset novelty over a
  global offset and a ±3 % tempo scale. The earliest peak within 5 % of the
  best one wins. `peak/2nd` near 1 means the score is periodic, so the
  offset is only known to within a beat or bar. That is harmless for a loop
  and worth checking for a structured track.
- **Per bar**: `E` (full-band level) and five band levels, each a digit
  0–9 that drops one step per 4 dB below that band's loudest bar. It also
  shows the detected onsets quantised to 16ths in the kick and high bands;
  score-vs-onset match per drum part with the median ms offset (`+` = audio
  late); and the low-end **pump**, the kick peak over the low band in the
  gap after it (small = the bass never clears between kicks). Identical
  consecutive bars collapse to `23-24×2`.
- **Summary**: match %, median ± half-IQR offset, a swing estimate (hat
  offset on odd minus even 16ths), median pump, and integrated LUFS
  (pyloudnorm, optional).

Real output on the feed track *techno · Wired Monad* (seed 3976429772,
121 s webm; full text in `docs/examples/wired-monad.audio.ir.txt`):

```
AUDIO  z4EBG9jBr1M4jL6FTrV5TathwNxYUUZH8RTbfRfZuzex2ytV2zp.webm  120.96s  -15.5 LUFS  offset +0.325s  tempo 128 (×1.000)  align peak/2nd 1.001
  score→onset match: kick 94% -4ms±2; hihat 69% +0ms±2 swing +0ms; snare 96% +2ms±1
  low-end pump (kick peak over bass gap) median 8.1 dB
  E/bands: 0-9, 4 dB a step below each band's loudest bar; bands = sub low lomid himid high;
  onset grids = detected onsets quantised to 16ths (kick band 35-150 Hz, high 6-16 kHz);
  match = score hits found within ±0.4 step, median ms (+ = audio late)
  bars     sec  E bands  kick-band onsets     high-band onsets     match  pump
  1        §1   8 88888  .x.x|x.x.|x.x.|x.x.  xxxx|.xx.|x.xx|xxx.  kick 3/4+3 hihat 10/12+1 snare 2/2+3  7.8dB
  …
  35       §1   8 88700  xxx.|x.x.|x.x.|x.x.  ....|....|....|....  kick 4/4-5 hihat 0/12 snare 0/2  7.4dB
  36       §1   8 89740  x.x.|x.x.|x.x.|x.x.  ....|....|....|....  kick 4/4-3 hihat 0/12 snare 1/2+5  8.7dB
  37       §1   8 88788  x.x.|x.x.|x.x.|x.x.  x.xx|xxx.|x.xx|xxx.  kick 4/4-3 hihat 10/12+0 snare 2/2+3  8.0dB
  38       §1   8 88788  xxx.|x.x.|x.x.|xxx.  x.xx|xxx.|x.x.|x.x.  kick 4/4-6 hihat 8/12+1 snare 2/2+0  7.5dB
  …
```

The rows show what the master did to the score. The kick drops out in bars
46 and 63. The hats vanish in bars 35–36: the mel view
(`docs/examples/mel-wired-monad-33-40.png`) shows a low-pass sweep there.
Kick-band onsets also land on 8th offbeats where the score has no kick.
None of this is in the trace. It comes from the live
autoDJ/macro/FX layer.

Without numpy/scipy, `--audio` is skipped with a one-line reason and the
score IR is still printed. A failed download or decode degrades the same way.

### Plots (`[viz]` extra)

`beats-py plot` writes a PNG at most 1600 px wide, with labelled axes, bar
numbers and part names, for an LLM to look at. The framing is a filtered
marked point process, x = tanh(g · Σ_k h_k * μ_k): μ_k are the score spikes
and h_k the instrument kernels.

| view | shows |
|---|---|
| `raster` | μ: spikes per part (height = accent/hit/ghost), with tonal parts drawn as a piano roll and section boundaries in red |
| `wave` | waveform (min/max) + RMS with a bar/beat grid, and μ on the same time axis below |
| `mel` | mel spectrogram (scipy STFT) with a bar grid, and μ below |
| `bands` | bars × (5 bands + E) heatmap, the same 0–9 digits as the IR |
| `residual` | real master minus the forward-model render (same μ through h_k, no FX), gain-normalised per band and per bar, plus the overall spectral tilt: what FX, sidechain and mastering added |

Without `--audio`, `wave`/`mel`/`bands` use the forward-model render. `residual` needs
`--audio`. `--bars A-B` zooms in. Examples are in `docs/examples/*.png`.

### MCP server (`[mcp]` extra)

`beats-py-mcp` is a stdio server built on the official `mcp` SDK (v2
`MCPServer`, falling back to v1 `FastMCP`). Its tools are `describe`,
`trace` (a bounded summary that never returns the raw trace),
`render` (≤ 120 s, to a path), `diff`, `calibrate` (≤ 50 seeds) and `plot`
(returns the PNG as an MCP image and saves it to a path). Text outputs are
capped at 60 k chars. It is not registered in any MCP config; to try it:

```bash
.venv/bin/pip install -e '.[analysis,mcp,viz]'
BEATS_PUBLIC=/path/to/beats-bitwrap-io/public .venv/bin/beats-py-mcp
```

## Tests

```bash
.venv/bin/python -m pytest
```

- `tests/test_offline.py` needs no node. Its golden values (PRNG outputs, a
  trace hash and an audio hash for a small conflict net) were produced by the
  JS engine, so it checks JS parity offline. The audio-hash test is the one
  that would trip on a host whose libm differs from V8's (see the caveat).
- `tests/test_describe.py` holds golden-ish tests for describe/diff/calibrate
  on a hand-built fixture (no node). Its audio alignment, viz (every view)
  and MCP tool-list smoke tests skip when their extras are missing.
- `tests/test_js_parity.py` runs the full harness against live JS, shortened
  (300 ticks, 0.5 s audio). It is skipped unless `node` is on `PATH` and
  `BEATS_PUBLIC` is set.

The full-length harness is still a standalone script; it writes
`parity/results.json` (or `results-audio.json`):

```bash
BEATS_PUBLIC=/path/to/beats-bitwrap-io/public python3 parity/test_parity.py
BEATS_PUBLIC=/path/to/beats-bitwrap-io/public python3 parity/test_parity.py --only-audio --audio-seconds 10
```

## What is ported

| JS (beats-bitwrap-io `public/`) | Python | notes |
|---|---|---|
| `lib/pflow.js` parse / arc index / `resetState` | `wave_py/project.py` | JS object-key order (integer-like keys first) and `Math.round` semantics |
| `wave-engine/net.js` | `wave_py/net.py` | executor (enabled → seeded conflict resolution → fire → immediate controls), incidence matrix, P-invariants (exact `Fraction`, sparse RREF) |
| `deterministicRand`, `strHash`, `createRng` (mulberry32, `generator/core.js`), `noiseSeed` + xorshift32 | `wave_py/prng.py` | int arithmetic with explicit uint32 masking / int32 reinterpretation; no `random` |
| `wave-engine/synth.js` | `wave_py/synth.py` | pulses, kits, voice families, float32 wavetables, RBJ biquad, `ringOf`, `envelope`, `ringSpectrum` |
| `wave-engine/runner.js` + `offline.js` | `wave_py/runner.py`, `wav.py` | sample clock, gate IIRs, carriers, tonal voice pools, AM/FM `mods`, tanh master, `stop-transport` |

**Instruments** — every voice the JS wave engine has:

- drum lanes on channels 10–15, by note: **kick** (sine swept from
  `48·octaves` Hz to 48 Hz), **snare** (band-passed noise + 180 Hz body),
  **clap**, **closed hat** and **open hat**, in all six kits (`drums`,
  `-breakbeat`, `-cr78`, `-v8`, `-808`, `-lofi`);
- tonal lanes, one wavetable family per instrument: **saw**, **sine**
  (sub/808), **square**, **soft** (pad/strings/choir/organ/harmony),
  **bell**, plus the pluck/stab/piano saw variant; voice pool 4 (6 for
  harmony) with quietest-voice stealing;
- the `dc` gate-only voice and the `voices` / `mods` runner overrides.

## What is not ported

- **The composer** — see above. Only its PRNG (mulberry32) is ported and
  parity-tested; it is what a future composer port would build on.
- **Block-wise numpy synthesis.** The scalar port is the reference a
  vectorised version would be tested against. The noise xorshift and biquad
  recursions are sequential and would stay scalar.
- Features absent from the JS wave engine too: macros (`fire-macro`),
  loop/seek, swing/humanize, drift.
- Runner features offline rendering never uses: mid-play project swap, the
  worklet message layer, real-time playback.

## Layout

```
wave_py/   prng project net synth runner trace compose wav __main__
           describe analysis calibrate diff viz mcp_server   — the IR, audio rows, plots, MCP
docs/examples/   real IR text, stats, diff and PNGs referenced above
tests/     test_offline.py, test_describe.py (no node)   test_js_parity.py (node + BEATS_PUBLIC)
parity/    test_parity.py — full harness
           js.mjs compose.mjs trace.mjs render.mjs prng.mjs — node shims that
           import the unmodified JS from $BEATS_PUBLIC
```

## License

MIT, matching beats-bitwrap-io. See [LICENSE](LICENSE).
