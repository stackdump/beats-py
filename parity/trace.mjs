#!/usr/bin/env node
// node trace.mjs <project.json> <ticks> <out.jsonl>
// Dumps the marking trace from the unmodified JS executor (wave-engine/net.js),
// in exactly the shape wave_py/trace.py writes.
import { readFileSync, writeFileSync } from 'node:fs';
import { load } from './js.mjs';
const { compileProject, tick } = await load('wave-engine/net.js');

const [projectPath, ticksArg, out] = process.argv.slice(2);
const project = JSON.parse(readFileSync(projectPath, 'utf8'));
const ticks = Number(ticksArg);
const g = compileProject(project);
const lines = [JSON.stringify({
    nets: g.nets.map(n => n.id), places: g.nets.map(n => n.placeIds), transitions: g.nets.map(n => n.transIds),
})];
for (let i = 0; i < ticks; i++) {
    tick(g, null);
    const fired = [], ctl = [], muted = [], mn = [];
    for (let k = 0; k < g.firedCount; k += 2) fired.push([g.fired[k], g.fired[k + 1]]);
    for (let k = 0; k < g.controlCount; k += 2) ctl.push([g.controls[k], g.controls[k + 1]]);
    g.muted.forEach((m, j) => { if (m) muted.push(j); });
    g.mutedNotes.forEach((m, j) => { if (m) mn.push(j); });
    lines.push(JSON.stringify({
        t: g.tick, fired, ctl, muted, mn, stop: g.stopRequested ? 1 : 0,
        m: g.nets.map(n => Array.from(n.state)),
    }));
}
writeFileSync(out, lines.join('\n') + '\n');
