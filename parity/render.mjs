#!/usr/bin/env node
// node render.mjs <project.json> <seconds> <out.f32> [sampleRate] [runnerOptsJSON]
// Offline render through the unmodified JS WaveRunner (wave-engine/offline.js);
// writes the left channel as raw little-endian float32.
import { readFileSync, writeFileSync } from 'node:fs';
import { load } from './js.mjs';
const { renderOffline } = await load('wave-engine/offline.js');

const [projectPath, seconds, out, rate = '48000', opts = '{}'] = process.argv.slice(2);
const project = JSON.parse(readFileSync(projectPath, 'utf8'));
const r = renderOffline({ project, seconds: Number(seconds), sampleRate: Number(rate), runner: JSON.parse(opts) });
writeFileSync(out, Buffer.from(r.left.buffer, r.left.byteOffset, r.left.byteLength));
