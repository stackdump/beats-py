#!/usr/bin/env node
// node prng.mjs → JSON of reference PRNG outputs from the unmodified JS.
import { load } from './js.mjs';
const { createRng } = await load('lib/generator/core.js');
const { deterministicRand, strHash } = await load('wave-engine/net.js');
const { noiseSeed } = await load('wave-engine/synth.js');

const seeds = [0, 1, 42, 1234, -7, 2147483647, 4294967295, 3000000000.7];
const mulberry = seeds.map(s => { const r = createRng(s); return Array.from({ length: 64 }, () => r.next()); });
const labels = ['p0', 'p63', 'struct-kick-d3', 'a', '', 'ß→𝄞', 'x'.repeat(40)];
const hashes = labels.map(strHash);
const rands = [];
for (const h of hashes) for (const t of [0, 1, 2, 17, 928, 100000, -3]) rands.push(deterministicRand(t, h));
const noise = [0, 1, 2, 5, 52].map(i => {
    const x = Int32Array.of(noiseSeed(i)), out = [x[0]];
    for (let k = 0; k < 64; k++) { let v = x[0]; v ^= v << 13; v ^= v >>> 17; v ^= v << 5; x[0] = v; out.push(x[0]); }
    return out;
});
process.stdout.write(JSON.stringify({ seeds, mulberry, labels, hashes, rands, noise }));
