// Test helpers: read a dataset folder from disk through the same interface the browser uses.
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));

export const DATA_DIR = process.env.QG_DATA || path.resolve(here, '../../docs/data/default');

export class NodeSource {
  constructor(dir) { this.dir = dir; this.label = dir; }
  async json(name) { return JSON.parse(await readFile(path.join(this.dir, name), 'utf8')); }
  async buffer(name) {
    const b = await readFile(path.join(this.dir, name));
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  }
}

/** Deterministic pseudo-random numbers (mulberry32) so tests do not depend on Math.random. */
export function rng(seed = 1) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gauss(r) {
  return Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
}

export function close(actual, expected, rel = 1e-9, abs = 0) {
  const tol = Math.max(abs, rel * Math.abs(expected));
  return Math.abs(actual - expected) <= tol;
}
