"""wave_py — offline Python port of beats-bitwrap-io's wave engine (the beats-py package).

The canonical artifact is the integer marking trace (net.py); audio is a
projection of it (runner.py). See README.md.
"""

from .prng import mulberry32, deterministic_rand, str_hash, noise_seed, xorshift32
from .project import parse_project
from .net import compile_project, reset_graph, tick, incidence_matrix, p_invariants
from .runner import WaveRunner, render_offline

__version__ = '0.1.0'

__all__ = [
    'mulberry32', 'deterministic_rand', 'str_hash', 'noise_seed', 'xorshift32',
    'parse_project', 'compile_project', 'reset_graph', 'tick',
    'incidence_matrix', 'p_invariants', 'WaveRunner', 'render_offline',
]
