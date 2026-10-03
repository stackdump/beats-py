"""The full Python ↔ JS harness (parity/test_parity.py), shortened, under pytest.

Skipped unless node is on PATH and $BEATS_PUBLIC names a beats-bitwrap-io
checkout's public/ directory. For the full-length run use the script directly.
"""

import os
import shutil

import pytest

import test_parity as harness  # parity/test_parity.py, via conftest

pytestmark = pytest.mark.skipif(
    not shutil.which('node') or not os.path.isfile(
        os.path.join(os.environ.get('BEATS_PUBLIC', ''), 'wave-engine', 'offline.js')),
    reason='needs node and BEATS_PUBLIC=<beats-bitwrap-io>/public')


@pytest.fixture(autouse=True)
def _reset():
    harness.failures.clear()
    yield
    assert harness.failures == []


def test_prng():
    harness.test_prng()


def test_traces_and_invariants(tmp_path):
    harness.test_traces(300, str(tmp_path))


def test_spectrum():
    harness.test_spectrum()


def test_audio(tmp_path):
    harness.test_audio(0.5, str(tmp_path))
