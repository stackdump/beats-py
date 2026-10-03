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

## Tests

```bash
.venv/bin/python -m pytest
```

- `tests/test_offline.py` needs no node. Its golden values (PRNG outputs, a
  trace hash and an audio hash for a small conflict net) were produced by the
  JS engine, so it checks JS parity offline. The audio-hash test is the one
  that would trip on a host whose libm differs from V8's (see the caveat).
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
tests/     test_offline.py (no node)   test_js_parity.py (node + BEATS_PUBLIC)
parity/    test_parity.py — full harness
           js.mjs compose.mjs trace.mjs render.mjs prng.mjs — node shims that
           import the unmodified JS from $BEATS_PUBLIC
```

## License

MIT, matching beats-bitwrap-io. See [LICENSE](LICENSE).
