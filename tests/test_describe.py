"""describe / diff / calibrate on a hand-built fixture (no node), plus
analysis, viz and MCP smoke tests that skip when their extras are absent."""

import asyncio
import copy
import json

import pytest

from wave_py.__main__ import main
from wave_py.calibrate import calibrate, percentile_of
from wave_py.describe import build_ir, render_text, _seq_text, describe
from wave_py.diff import diff_ir, render_diff_text


def ring(n, midi=None, channel=10, instrument='drums', group=''):
    """Single-token ring: step s fires t(s % n). midi = {t: {note, velocity, ...}}."""
    nb = {'track': {'channel': channel, 'instrument': instrument}, 'places': {}, 'transitions': {}, 'arcs': []}
    if group:
        nb['track']['group'] = group
    for i in range(n):
        nb['places'][f'p{i}'] = {'initial': [1 if i == 0 else 0]}
        t = {}
        if midi and i in midi:
            t['midi'] = {'channel': channel, **midi[i]}
        nb['transitions'][f't{i}'] = t
        nb['arcs'] += [{'source': f'p{i}', 'target': f't{i}'},
                       {'source': f't{i}', 'target': f'p{(i + 1) % n}'}]
    return nb


def chain(n, controls):
    """Control net: a one-shot chain of n transitions; controls = {t: control}."""
    nb = {'role': 'control', 'track': {'channel': 1}, 'places': {}, 'transitions': {}, 'arcs': []}
    for i in range(n + 1):
        nb['places'][f'c{i}'] = {'initial': [1 if i == 0 else 0]}
    for i in range(n):
        nb['transitions'][f'u{i}'] = {'control': controls[i]} if i in controls else {}
        nb['arcs'] += [{'source': f'c{i}', 'target': f'u{i}'}, {'source': f'u{i}', 'target': f'c{i + 1}'}]
    return nb


def fixture():
    return {
        'name': 'fixture', 'tempo': 120, 'rootNote': 36, 'scaleName': 'Minor',
        'nets': {
            'kick': ring(4, {0: {'note': 36, 'velocity': 120}}),
            'hihat': ring(4, {2: {'note': 42, 'velocity': 100}}, channel=12),
            'bass': ring(8, {0: {'note': 36, 'velocity': 100, 'durationSteps': 2},
                             4: {'note': 43, 'velocity': 100, 'durationSteps': 1}},
                         channel=2, instrument='acid', group='bass'),
            'struct': chain(32, {31: {'action': 'unmute-track', 'targetNet': 'bass'}}),
        },
        'initialMutes': ['bass'],
        'structure': [{'name': 'intro', 'steps': 32}, {'name': 'drop', 'steps': 32}],
    }


def test_seq_text_compresses_runs_and_cycles():
    assert _seq_text('A A A A'.split()) == 'A×4'
    assert _seq_text('A B A B'.split()) == '(A B)×2'
    assert _seq_text('A B C B C B C B C B C B'.split()) == 'A (B C)×5 B'
    assert _seq_text('C A A A A A D B'.split()) == 'C A×5 D B'


def test_describe_fixture_ir():
    ir = build_ir(fixture(), {'project': 'fixture'})
    m = ir['meta']
    assert (m['bars'], m['steps'], m['tempo'], m['root'], m['seconds']) == (4, 64, 120, 'C2', 8.0)
    assert list(ir['rows']) == ['kick', 'hihat', 'bass']
    assert ir['patterns']['kick'] == {'A': {'grid': 'X...X...X...X...', 'notes': [], 'hits': 4, 'first_bar': 1}}
    assert ir['patterns']['hihat']['A']['grid'] == '..x...x...x...x.'
    assert ir['patterns']['bass']['A'] == {'grid': 'x-..x...x-..x...', 'notes': ['C2', 'G2', 'C2', 'G2'],
                                           'hits': 4, 'first_bar': 3}
    intro, drop = ir['sections']
    assert (intro['name'], intro['bars'], drop['bars']) == ('intro', [1, 2], [3, 4])
    assert 'bass' not in intro['parts'] and drop['enter'] == ['bass']
    assert drop['controls'] == {} and intro['control_events'] == [
        {'bar': 2, 'step': 16, 'action': 'unmute-track', 'target': 'bass'}]
    assert drop['parts']['bass']['notes'] == ['C2', 'G2']
    assert ir['totals']['hihat']['offbeat8'] == 1.0 and ir['totals']['kick']['onbeat'] == 1.0
    assert ir['kick_bass'] == {'bass_onsets': 8, 'on_kick': 8, 'ratio': 1.0}


def test_describe_fixture_text_golden():
    text = render_text(build_ir(fixture(), {'project': 'fixture'}))
    lines = text.splitlines()
    assert lines[0] == '# beats-ir/1 project=fixture | 120 BPM C2 Minor | 4 bars 0:08 | 16 steps/bar'
    for want in ('  kick       A  X...|X...|X...|X...   4',
                 '  hihat      A  ..x.|..x.|..x.|..x.   4',
                 '  bass       A  x-..|x...|x-..|x...   4 C2 G2 C2 G2',
                 '§1 intro  bars 1-2 (2)  8.0 hits/bar',
                 '  ctl        unmute-track×1  unmute bass@b2',
                 '  in: bass',
                 '  kick/bass  8/8 bass onsets land on a kick (100%)'):
        assert want in lines, want
    assert len(lines) < 40


def test_describe_cli_json(tmp_path, capsys):
    p = tmp_path / 'f.json'
    p.write_text(json.dumps(fixture()))
    assert main(['describe', '--project', str(p), '--format', 'json', '--out', str(tmp_path / 'ir.json')]) == 0
    ir = json.loads((tmp_path / 'ir.json').read_text())
    assert ir['meta']['bars'] == 4 and len(ir['bars']) == 4


def test_diff_reports_changed_part(tmp_path, capsys):
    a = fixture()
    b = copy.deepcopy(a)
    b['nets']['hihat'] = ring(2, {1: {'note': 42, 'velocity': 60}}, channel=12)   # 16ths, ghosted
    pa, pb = tmp_path / 'a.json', tmp_path / 'b.json'
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    d = diff_ir(build_ir(a), build_ir(b))
    hh = d['sections'][0]['parts']['hihat']
    assert hh['bars_changed'] == '2/2' and hh['density'] == [4, 8]
    assert 'kick' not in d['sections'][0]['parts']
    text = render_diff_text(d)
    assert '..x.|..x.|..x.|..x.' in text and '.g.g|.g.g|.g.g|.g.g' in text
    assert main(['diff', str(pa), str(pb)]) == 0
    assert 'hihat' in capsys.readouterr().out


def test_calibrate_from_projects_gives_percentiles(tmp_path):
    paths = []
    for i, n in enumerate((2, 4, 8)):
        f = fixture()
        f['nets']['hihat'] = ring(n, {0: {'note': 42, 'velocity': 100}}, channel=12)
        paths.append(str(tmp_path / f'p{i}.json'))
        with open(paths[-1], 'w') as fh:
            json.dump(f, fh)
    st = calibrate(projects=paths)
    assert st['n'] == 3 and st['rows']['hihat']['density'] == [2, 4, 8]
    assert percentile_of(st, 'hihat', 'density', 8) == 83
    assert percentile_of(st, 'hihat', 'density', 1) == 0
    # unknown row name falls back to its kind
    assert percentile_of(st, 'hats2', 'density', 4, {'kind': 'hat'}) == 50
    text = render_text(build_ir(fixture()), st)
    assert '(p' in text


# --- optional extras ---------------------------------------------------------------

def _render_fixture_wav(tmp_path):
    from wave_py.runner import render_offline
    from wave_py.wav import write_wav
    s, _ = render_offline(fixture(), seconds=8, sr=44100)
    path = str(tmp_path / 'fixture.wav')
    write_wav(path, s, 44100, channels=1)
    return path


def test_audio_rows_align_to_forward_model(tmp_path):
    pytest.importorskip('numpy')
    pytest.importorskip('scipy')
    wav = _render_fixture_wav(tmp_path)
    ir, text, notes = describe(fixture(), audio=wav)
    assert notes == []
    a = ir['audio']
    assert abs(a['align']['offset_s']) < 0.02 and abs(a['align']['scale'] - 1) <= 0.002
    assert a['summary']['kick']['match'] >= 0.9
    assert len(a['rows']) == 4 and all(len(r['bands']) == 5 for r in a['rows'])
    assert a['rows'][0]['kick_onsets'] == 'x...x...x...x...'
    assert 'AUDIO  fixture.wav' in text


def test_audio_missing_file_degrades(tmp_path):
    pytest.importorskip('numpy')
    pytest.importorskip('scipy')
    ir, text, notes = describe(fixture(), audio=str(tmp_path / 'nope.wav'))
    assert 'audio' not in ir and notes and notes[0].startswith('audio skipped')
    assert text.startswith('# beats-ir/1')


@pytest.mark.parametrize('view', ['raster', 'wave', 'mel', 'bands', 'residual'])
def test_viz_views_render(tmp_path, view):
    pytest.importorskip('matplotlib')
    pytest.importorskip('scipy')
    from wave_py.viz import plot
    audio = _render_fixture_wav(tmp_path) if view == 'residual' else None
    out = plot(view, str(tmp_path / f'{view}.png'), project=fixture(), audio=audio,
               bars='1-4' if view != 'raster' else None)
    with open(out, 'rb') as f:
        head = f.read(24)
    assert head[:8] == b'\x89PNG\r\n\x1a\n'
    width = int.from_bytes(head[16:20], 'big')
    assert 400 < width <= 1600


def test_mcp_server_lists_tools():
    pytest.importorskip('mcp')
    from wave_py.mcp_server import mcp
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert {'describe', 'trace', 'render', 'diff', 'calibrate', 'plot'} <= names


def test_mcp_describe_tool_on_fixture(tmp_path):
    pytest.importorskip('mcp')
    from wave_py import mcp_server
    p = tmp_path / 'f.json'
    p.write_text(json.dumps(fixture()))
    out = mcp_server.describe(project=str(p))
    assert out.startswith('# beats-ir/1') and 'PATTERNS' in out
    tr = mcp_server.trace(project=str(p), ticks=100000)
    assert '4096 ticks' in tr and 'kick' in tr
