#!/usr/bin/env node
// node compose.mjs <genre> <seed> [structure] → project JSON on stdout.
// Imports the JS composer unmodified (via wave-engine/offline.js under $BEATS_PUBLIC).
import { load } from './js.mjs';
const { composeProject } = await load('wave-engine/offline.js');

const [genre = 'techno', seed = '42', structure = ''] = process.argv.slice(2);
process.stdout.write(JSON.stringify(composeProject(genre, Number(seed), structure)));
