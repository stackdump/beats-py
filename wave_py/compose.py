"""(genre, seed, structure) → project JSON.

The composer (public/lib/generator/* in beats-bitwrap-io, ~5k lines, held
byte-identical to the Go generator) is NOT ported, and its JS is not vendored
here. compose_project() runs the unmodified JS composer through node, from a
beats-bitwrap-io checkout named by $BEATS_PUBLIC (its `public/` directory).

Node is optional: nothing in this module runs at import time, and
load_project() — the --project path — never touches it.
"""

import json
import os
import shutil
import subprocess

# Inline ES module: imports offline.js from $BEATS_PUBLIC, prints project JSON.
_SHIM = """
import { pathToFileURL } from 'node:url';
import { join } from 'node:path';
const [pub, genre, seed, structure = ''] = process.argv.slice(1);
const { composeProject } = await import(pathToFileURL(join(pub, 'wave-engine', 'offline.js')).href);
process.stdout.write(JSON.stringify(composeProject(genre, Number(seed), structure)));
"""

_HINT = ('the composer is JS-only; pass --project <file.json> instead, or install node '
         'and set BEATS_PUBLIC to a beats-bitwrap-io checkout\'s public/ directory')


def beats_public():
    """The beats-bitwrap-io public/ directory from $BEATS_PUBLIC, validated."""
    pub = os.environ.get('BEATS_PUBLIC', '')
    if not pub:
        raise RuntimeError('BEATS_PUBLIC is not set: ' + _HINT)
    if not os.path.isfile(os.path.join(pub, 'wave-engine', 'offline.js')):
        raise RuntimeError(f'BEATS_PUBLIC={pub} has no wave-engine/offline.js: ' + _HINT)
    return os.path.abspath(pub)


def compose_project(genre, seed, structure=''):
    node = shutil.which('node')
    if not node:
        raise RuntimeError('node not found: ' + _HINT)
    args = [node, '--input-type=module', '-e', _SHIM, '--', beats_public(), genre, str(seed)]
    if structure and structure != 'loop':
        args.append(structure)
    r = subprocess.run(args, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError('JS composer failed: ' + r.stderr.strip()[-500:])
    return json.loads(r.stdout)


def load_project(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)
