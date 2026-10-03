"""Port of public/wave-engine/runner.js (WaveRunner) + offline.js
(renderOffline): the tick clock, the lanes and the mix, sample by sample.

The arithmetic follows the JS operation for operation and in the same order
(IEEE doubles on both sides), so the sample clock and every tick boundary
agree exactly. What can differ is the last ulp of exp/sin/tanh/pow: V8 ships
its own fdlibm port, CPython calls the platform libm. Output is rounded to
float32 exactly as the JS Float32Array does. Hence the contract: the trace
is exact, the samples agree within a tolerance (see parity/).

Not ported: mid-play project swap (`pending` / `fading`); offline rendering
loads once before play, which is the only path offline.js uses.
"""

import math
from array import array

from .net import compile_project, reset_graph, tick as tick_graph
from .synth import (PPQ, TABLE_SIZE, is_drum_channel, drum_kind, drum_voice, tonal_voice,
                    build_table, biquad)
from .prng import noise_seed

L_GATE, L_TONAL = 0, 1
C_DC, C_SWEEP, C_NOISE, C_NOISE_TONE = 0, 1, 2, 3
SILENT = 1e-6
TWO_PI = 2 * math.pi

_TABLES = {}


def _table(wave):
    if wave not in _TABLES:
        _TABLES[wave] = build_table(wave)
    return _TABLES[wave]


class Lane:
    __slots__ = ('type', 'voice', 'active', 'E', 'sr', 'alpha', 'mul', 's', 'carrier', 'rng', 'bq',
                 'am', 'fm', 'table', 'K', 'inc', 'phase', 'sd', 'sa', 'md', 'net_id',
                 'env', 'prev', 'level', 'ph', 'sweep', 'sweep_mul', 'f0', 'f1', 'tone_inc',
                 'tone_mix', 'z1', 'z2', 'am_depth', 'fm_depth', 'inv_sr', 'att_mul')

    def __init__(self, voice, sr):
        self.voice, self.sr, self.active = voice, sr, False
        self.env = self.prev = self.ph = self.sweep = self.sweep_mul = 0.0
        self.f0 = self.f1 = self.tone_inc = self.tone_mix = self.z1 = self.z2 = 0.0
        self.am_depth = self.fm_depth = self.att_mul = 0.0
        self.level = voice['level']
        self.inv_sr = 1 / sr
        self.am = self.fm = None
        self.alpha = self.mul = self.s = self.rng = self.bq = None
        self.table = self.inc = self.phase = self.sd = self.sa = self.md = None
        self.E = self.K = 0
        self.carrier = C_DC
        self.net_id = ''


def gate_lane(voice, sr, seed):
    ln = Lane(voice, sr)
    ln.type = L_GATE
    ln.E = len(voice['pulse']['alpha'])
    ln.alpha = list(voice['pulse']['alpha'])
    ln.mul = [math.exp(-1 / (tau * sr)) for tau in voice['pulse']['tau']]
    ln.s = [0.0] * ln.E
    c = voice.get('carrier')
    ln.carrier = C_DC if c == 'dc' else C_SWEEP if c == 'sweep' else C_NOISE_TONE if c == 'noise+tone' else C_NOISE
    ln.rng = noise_seed(seed)
    ln.bq = [0.0] * 5
    ln.f0 = voice.get('f0') or 0
    ln.f1 = voice.get('f1') or 0
    ln.tone_inc = (voice.get('tone') or 0) / sr
    ln.tone_mix = voice.get('toneMix') or 0
    if ln.carrier == C_SWEEP:
        ln.sweep_mul = math.exp(-1 / (voice['sweepTau'] * sr))
    if ln.carrier in (C_NOISE, C_NOISE_TONE):
        ln.bq = biquad(voice['filter'], voice['fc'], voice['q'], sr)
    return ln


def tonal_lane(voice, sr, poly):
    ln = Lane(voice, sr)
    ln.type = L_TONAL
    ln.att_mul = math.exp(-1 / (voice['attack'] * sr))
    ln.table = _table(voice['wave'])
    ln.K = poly
    ln.inc, ln.phase = [0.0] * poly, [0.0] * poly
    ln.sd, ln.sa, ln.md = [0.0] * poly, [0.0] * poly, [0.0] * poly
    return ln


def _hit(runner, ln, n, t):
    ln.active = True
    w = n.vel[t] / 127
    if ln.type == L_GATE:
        s, a = ln.s, ln.alpha
        for e in range(ln.E):
            s[e] += a[e] * w
        if ln.carrier == C_SWEEP:
            ln.sweep = 1.0
        return
    steps = n.dur_steps[t]
    dur_ms = steps * 60000 / (runner.tempo * PPQ) if steps > 0 else n.dur_ms[t]
    v, best = 0, math.inf
    for k in range(ln.K):
        a = abs(ln.sd[k] + ln.sa[k])
        if a < best:
            best, v = a, k
    cur = ln.sd[v] + ln.sa[v]
    ln.inc[v] = 440 * math.pow(2, (n.note[t] - 69) / 12) * ln.inv_sr
    tau = (dur_ms if dur_ms > 0 else 100) / 1000 * ln.voice['decayScale']
    tau = 0.03 if tau < 0.03 else ln.voice['maxDecay'] if tau > ln.voice['maxDecay'] else tau
    ln.md[v] = math.exp(-ln.inv_sr / tau)
    ln.sd[v] = w
    ln.sa[v] = cur - w


def _lane_sample(ln, sin=math.sin, floor=math.floor):
    if ln.type == L_GATE:
        s, mul = ln.s, ln.mul
        env = 0.0
        for e in range(ln.E):
            env += s[e]
            s[e] *= mul[e]
        ln.env = env
        c = 1.0
        car = ln.carrier
        if car == C_SWEEP:
            f = ln.f1 + (ln.f0 - ln.f1) * ln.sweep
            ln.sweep *= ln.sweep_mul
            if ln.fm is not None:
                f *= 1 + ln.fm_depth * ln.fm.prev
            ph = ln.ph + f * ln.inv_sr
            ph -= floor(ph)
            ln.ph = ph
            c = sin(TWO_PI * ph)
        elif car != C_DC:
            x = ln.rng & 0xFFFFFFFF
            x ^= (x << 13) & 0xFFFFFFFF
            x ^= x >> 17
            x ^= (x << 5) & 0xFFFFFFFF
            x = x - 0x100000000 if x & 0x80000000 else x
            ln.rng = x
            nz = x / 2147483648
            b = ln.bq
            y = b[0] * nz + ln.z1
            ln.z1 = b[1] * nz - b[3] * y + ln.z2
            ln.z2 = b[2] * nz - b[4] * y
            c = y
            if car == C_NOISE_TONE:
                ph = ln.ph + ln.tone_inc
                ph -= floor(ph)
                ln.ph = ph
                c = (1 - ln.tone_mix) * y + ln.tone_mix * sin(TWO_PI * ph)
        out = env * c * ln.level
        if ln.am is not None:
            out *= 1 - ln.am_depth + ln.am_depth * ln.am.prev
        if abs(s[0]) < SILENT and (ln.E < 2 or abs(s[1]) < SILENT):
            ln.active = False
            ln.env = 0.0
        return out
    out = 0.0
    env_sum = 0.0
    live = False
    tb = ln.table
    fmk = 1 + ln.fm_depth * ln.fm.prev if ln.fm is not None else 1
    am = ln.att_mul
    sd_, sa_, md_, inc_, phase_ = ln.sd, ln.sa, ln.md, ln.inc, ln.phase
    for k in range(ln.K):
        sd, sa = sd_[k], sa_[k]
        if abs(sd) < SILENT and abs(sa) < SILENT:
            continue
        live = True
        env = sd + sa
        env_sum += env
        sd_[k] = sd * md_[k]
        sa_[k] = sa * am
        ph = phase_[k] + inc_[k] * fmk
        ph -= floor(ph)
        phase_[k] = ph
        x = ph * TABLE_SIZE
        i = int(x)
        fr = x - i
        out += env * (tb[i] + (tb[i + 1] - tb[i]) * fr)
    ln.env = env_sum
    if not live:
        ln.active = False
    out *= ln.level
    if ln.am is not None:
        out *= 1 - ln.am_depth + ln.am_depth * ln.am.prev
    return out


def build_lanes(g, sr, opts):
    lanes, by_net = [], {}
    voices = opts.get('voices') or {}
    for n in g.nets:
        if n.bundle.role == 'control':
            continue
        tr = n.bundle.track
        override = voices.get(n.id)
        kind_lane = {}
        for t in range(n.T):
            if not n.midi[t]:
                continue
            ch = n.bundle.bindings[n.trans_ids[t]]['channel'] or tr['channel']
            key = 'v' if override else drum_kind(n.note[t]) if is_drum_channel(ch) else 'tonal'
            if key not in kind_lane:
                if override:
                    ln = tonal_lane(override, sr, 4) if override.get('kind') == 'tonal' else gate_lane(override, sr, len(lanes))
                elif key == 'tonal':
                    ln = tonal_lane(tonal_voice(tr.get('instrument'), tr.get('group')), sr,
                                    6 if tr.get('group') == 'harmony' else 4)
                else:
                    ln = gate_lane(drum_voice(key, tr.get('instrument')), sr, len(lanes))
                ln.net_id = n.id
                kind_lane[key] = len(lanes)
                lanes.append(ln)
            n.lane[t] = kind_lane[key]
        if kind_lane:
            by_net[n.id] = list(kind_lane.values())
    sources = []
    for m in opts.get('mods') or []:
        src, dst = by_net.get(m['source']), by_net.get(m['target'])
        if not src or not dst:
            continue
        for li in dst:
            if m.get('kind') == 'fm':
                lanes[li].fm, lanes[li].fm_depth = lanes[src[0]], m['depth']
            else:
                lanes[li].am, lanes[li].am_depth = lanes[src[0]], m['depth']
        if m.get('silentSource'):
            for li in src:
                lanes[li].level = 0
        if lanes[src[0]] not in sources:
            sources.append(lanes[src[0]])
    return lanes, by_net, sources


class WaveRunner:
    """opts: master 'soft' (tanh, default) | 'linear'; masterGain (0.8);
    voices {netId: voice}; mods [{target, source, kind, depth, silentSource}];
    on_tick(runner) after every tick."""

    def __init__(self, sr, opts=None):
        self.sr = sr
        self.opts = opts or {}
        mg = self.opts.get('masterGain')
        self.master_gain = 0.8 if mg is None else mg
        self.linear = self.opts.get('master') == 'linear'
        self.graph, self.lanes, self.by_net, self.sources = None, [], {}, []
        self.playing = False
        self.stopped = False
        self.tempo = 120
        self.spt = self._spt(120)
        self.tick_pos = 0.0
        self.samples = 0

    def _spt(self, bpm):
        return self.sr * 60 / (bpm * PPQ)

    def load(self, json_obj):
        g = compile_project(json_obj)
        self.lanes, self.by_net, self.sources = build_lanes(g, self.sr, self.opts)
        self.graph = g
        self.set_tempo(g.tempo)

    def play(self):
        if self.graph is None or self.playing:
            return
        self.playing = True
        self.stopped = False
        self.tick_pos = self.spt - 1

    def stop(self):
        self.playing = False
        if self.graph is not None:
            reset_graph(self.graph)

    def set_tempo(self, bpm):
        if not (bpm > 0):
            return
        frac = self.tick_pos / self.spt
        self.tempo = bpm
        if self.graph is not None:
            self.graph.tempo = bpm
        self.spt = self._spt(bpm)
        self.tick_pos = frac * self.spt

    def lanes_of(self, net_id):
        return [self.lanes[i] for i in self.by_net.get(net_id, [])]

    def _on_note(self, n, t):
        li = n.lane[t]
        if li >= 0:
            _hit(self, self.lanes[li], n, t)

    def _tick(self):
        g = self.graph
        tick_graph(g, self._on_note)
        cb = self.opts.get('on_tick')
        if cb:
            cb(self)
        if g.stop_requested:
            self.stopped = True
            self.stop()

    def process(self, out, frames):
        """Append `frames` float32-rounded mono samples to `out` (array('f'))."""
        gain, linear, tanh = self.master_gain, self.linear, math.tanh
        lanes, sources = self.lanes, self.sources
        for _ in range(frames):
            if self.playing:
                self.tick_pos += 1
                if self.tick_pos >= self.spt:
                    self.tick_pos -= self.spt
                    self._tick()
            acc = 0.0
            for ln in lanes:
                if ln.active:
                    acc += _lane_sample(ln)
                else:
                    ln.env = 0.0
            for s in sources:
                s.prev = s.env
            out.append(acc if linear else tanh(acc * gain))
            self.samples += 1


def js_round(x):
    r = math.floor(x)
    return r + 1 if x - r >= 0.5 else r


def render_offline(project, seconds=10, sr=48000, tempo=None, runner_opts=None):
    """→ (samples: array('f'), runner). Mono: the JS writes the same y to L and R."""
    runner = WaveRunner(sr, runner_opts or {})
    runner.load(project)
    if tempo:
        runner.set_tempo(tempo)
    runner.play()
    frames = js_round(seconds * sr)
    out = array('f')
    runner.process(out, frames)
    return out, runner
