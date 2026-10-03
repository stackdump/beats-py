// Locates the unmodified beats JS: $BEATS_PUBLIC = a beats-bitwrap-io checkout's public/.
// The JS is never vendored into this repo; the shims import it from there.
import { existsSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const pub = process.env.BEATS_PUBLIC;
if (!pub || !existsSync(join(pub, 'wave-engine', 'offline.js'))) {
    process.stderr.write('BEATS_PUBLIC must name a beats-bitwrap-io checkout\'s public/ directory\n');
    process.exit(2);
}
export const load = rel => import(pathToFileURL(resolve(pub, rel)).href);
