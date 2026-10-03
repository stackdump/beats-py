"""Node-free tests: these run anywhere, with no JS and no node installed.

The golden values were produced by the unmodified JS engine (beats-bitwrap-io
3ebaf60, node 25) via parity/*.mjs, so they pin JS parity offline.
"""

import hashlib
import json
import wave
from array import array

import pytest

from wave_py import compose
from wave_py.__main__ import main
from wave_py.net import compile_project, incidence_matrix, p_invariants, marking_ints
from wave_py.prng import mulberry32, str_hash, deterministic_rand, noise_seed, xorshift32
from wave_py.project import parse_project
from wave_py.runner import render_offline
from wave_py.trace import iter_trace, _dumps

from test_parity import CONFLICT  # parity/test_parity.py, via conftest

JS_CONFLICT_TRACE_200 = '187ce3f612fe32ee2c8e7ae20605806c689c28bb93100437f7cd78c7831f4afd'
JS_CONFLICT_AUDIO_1S = 'aaf1386fb0c5f5034c9e41610e9a4dcf60b707cb055aeeea27784b56f5a14755'


def test_prng_matches_js():
    r = mulberry32(42)
    assert [r.next() for _ in range(4)] == [
        0.6011037519201636, 0.44829055899754167, 0.8524657934904099, 0.6697340414393693]
    assert [str_hash(s) for s in ['p0', 'p63', 'struct-kick-d3', 'a', '']] == [
        3520, 109357, -1028925634, 97, 0]
    assert [deterministic_rand(t, 3520) for t in [0, 1, 2, 17, 928, 100000, -3]] == [
        0.8387001678347588, 0.903145048301667, 0.9003193594980985, 0.9827221378218383,
        0.039261315716430545, 0.7973033734597266, 0.6742764981463552]
    x, out = noise_seed(0), []
    for _ in range(5):
        out.append(x)
        x = xorshift32(x)
    assert out == [467448786, 963646753, 1095840171, -1947721415, 1982307614]


def test_conflict_trace_matches_js():
    text = ''.join(_dumps(r) + '\n' for r in iter_trace(CONFLICT, 200))
    assert hashlib.sha256(text.encode()).hexdigest() == JS_CONFLICT_TRACE_200


def test_conflict_audio_matches_js():
    samples, _ = render_offline(CONFLICT, seconds=1, sr=48000)
    digest = hashlib.sha256(array('f', samples).tobytes()).hexdigest()
    # Bit identity relies on CPython's libm agreeing with V8 on exp/sin/tanh/pow;
    # the contract is 1e-5 and this host-dependent check is the stronger claim.
    assert digest == JS_CONFLICT_AUDIO_1S, 'audio differs from JS bit-for-bit (libm vs V8?)'


def test_p_invariants_hold_every_tick():
    g = compile_project(CONFLICT)
    net = g.nets[0]
    basis = p_invariants(incidence_matrix(net))
    assert basis
    start = [sum(y * m for y, m in zip(v, marking_ints(net))) for v in basis]
    for rec in list(iter_trace(CONFLICT, 300))[1:]:
        assert [sum(y * m for y, m in zip(v, rec['m'][0])) for v in basis] == start


def test_cli_project_file_needs_no_node(tmp_path, monkeypatch):
    monkeypatch.setenv('PATH', str(tmp_path))       # no node reachable
    monkeypatch.delenv('BEATS_PUBLIC', raising=False)
    pj = tmp_path / 'p.json'
    pj.write_text(json.dumps(CONFLICT))
    assert main(['trace', '--project', str(pj), '--ticks', '50', '--out', str(tmp_path / 't.jsonl')]) == 0
    assert len((tmp_path / 't.jsonl').read_text().splitlines()) == 51
    out = tmp_path / 'x.wav'
    assert main(['render', '--project', str(pj), '--seconds', '0.25', '--out', str(out)]) == 0
    with wave.open(str(out)) as w:
        assert (w.getnchannels(), w.getframerate(), w.getnframes()) == (2, 48000, 12000)


def test_genre_without_node_fails_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('PATH', str(tmp_path))
    with pytest.raises(RuntimeError, match='--project'):
        compose.compose_project('techno', 42)
    assert main(['trace', '--genre', 'techno', '--out', str(tmp_path / 't.jsonl')]) == 2
    assert 'node not found' in capsys.readouterr().err


def test_parse_project_roundtrip():
    assert parse_project(CONFLICT)
