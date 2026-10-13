// Loading a dataset (manifest.json + binary arrays) and serving derived fields from it.
// The data can come from a URL (the bundled datasets) or from a folder the user picked.

import { Grid, Model, Snapshot, Spectral, computeDiagnostics } from './physics.js';

// ---- sources ------------------------------------------------------------------------------
/** Files served by a web server below `base` (e.g. "data/default/"). */
export class UrlSource {
  constructor(base) { this.base = base.endsWith('/') ? base : base + '/'; this.label = this.base; }
  async #get(name) {
    const r = await fetch(this.base + name);
    if (!r.ok) throw new Error(`could not load ${this.base + name} (HTTP ${r.status})`);
    return r;
  }
  async json(name) { return (await this.#get(name)).json(); }
  async buffer(name) { return (await this.#get(name)).arrayBuffer(); }
}

/** Files from a folder chosen with <input type="file" webkitdirectory> (a FileList). */
export class FolderSource {
  constructor(fileList) {
    this.files = new Map();
    for (const f of fileList) this.files.set(f.name, f);
    const any = fileList[0];
    this.label = any && any.webkitRelativePath ? any.webkitRelativePath.split('/')[0] : 'local folder';
  }
  #file(name) {
    const f = this.files.get(name);
    if (!f) throw new Error(`file '${name}' is not in the selected folder`);
    return f;
  }
  async json(name) { return JSON.parse(await this.#file(name).text()); }
  async buffer(name) { return this.#file(name).arrayBuffer(); }
}

// ---- arrays ---------------------------------------------------------------------------------
/** Descriptor -> { data (Int16Array | Float32Array), shape, scale, scaleAxis }. */
export async function readArray(source, desc) {
  const buf = await source.buffer(desc.file);
  const data = desc.dtype === 'int16' ? new Int16Array(buf) : new Float32Array(buf);
  const n = desc.shape.reduce((a, b) => a * b, 1);
  if (data.length !== n) throw new Error(`${desc.file}: expected ${n} values, found ${data.length}`);
  return { data, shape: desc.shape, scale: desc.scale ?? 1, scaleAxis: desc.scale_axis ?? null, dtype: desc.dtype };
}

/** Decode a whole array to Float64Array applying the stored scale. */
export function decode(arr) {
  const out = new Float64Array(arr.data.length);
  if (Array.isArray(arr.scale)) {
    const inner = arr.shape.slice(arr.scaleAxis + 1).reduce((a, b) => a * b, 1);
    const len = arr.shape[arr.scaleAxis];
    for (let i = 0; i < out.length; i++) out[i] = arr.data[i] * arr.scale[Math.floor(i / inner) % len];
  } else {
    for (let i = 0; i < out.length; i++) out[i] = arr.data[i] * arr.scale;
  }
  return out;
}

class LRU {
  constructor(max) { this.max = max; this.map = new Map(); }
  get(k) { if (!this.map.has(k)) return undefined; const v = this.map.get(k); this.map.delete(k); this.map.set(k, v); return v; }
  set(k, v) { this.map.delete(k); this.map.set(k, v); if (this.map.size > this.max) this.map.delete(this.map.keys().next().value); }
}

// ---- the dataset -------------------------------------------------------------------------------
export class Dataset {
  static async open(source) {
    const manifest = await source.json('manifest.json');
    if (manifest.format !== 'qg2l-dataset') throw new Error('this folder does not contain a qg2l dataset (manifest.json has the wrong format)');
    const ds = new Dataset(manifest, source);
    await ds.#loadControl();
    return ds;
  }

  constructor(manifest, source) {
    this.manifest = manifest;
    this.source = source;
    this.model = new Model(manifest.model);
    this.grid = new Grid(manifest.grid);
    this.spec = new Spectral(this.grid);
    this._json = new Map();
    this._arrays = new Map();
    this._snaps = new LRU(6);
    this._series = new LRU(3);
    this._diag = null;
  }

  async #loadControl() {
    const c = this.manifest.control;
    this.control = c;
    this.nt = c.n;
    this.dtDays = c.step_hours / 24;
    this.psiArr = await readArray(this.source, c.psi);
    if (this.psiArr.shape.length !== 4 || this.psiArr.shape[0] !== this.nt) throw new Error('control psi has an unexpected shape');
  }

  has(product) { return Boolean(this.manifest[product]); }
  get title() { return this.manifest.title; }
  timeDays(t) { return this.control.start_days + t * this.dtDays; }
  indexOfDay(day) { return Math.max(0, Math.min(this.nt - 1, Math.round((day - this.control.start_days) / this.dtDays))); }

  /** [psi1, psi2] (m2/s, Float64Array) at time index t. */
  getPsi(t) {
    const N = this.grid.N, { data, scale } = this.psiArr;
    const out = [new Float64Array(N), new Float64Array(N)];
    for (let L = 0; L < 2; L++) {
      const o = (t * 2 + L) * N, s = scale[L];
      for (let p = 0; p < N; p++) out[L][p] = data[o + p] * s;
    }
    return out;
  }

  snapshot(t) {
    let s = this._snaps.get(t);
    if (!s) {
      const [p1, p2] = this.getPsi(t);
      s = new Snapshot(this.model, this.spec, p1, p2);
      this._snaps.set(t, s);
    }
    return s;
  }

  async json(name) {
    if (!this._json.has(name)) this._json.set(name, await this.source.json(name));
    return this._json.get(name);
  }

  /** Decoded Float64Array of an array descriptor (cached by file name). */
  async array(desc) {
    if (!this._arrays.has(desc.file)) {
      const a = await readArray(this.source, desc);
      this._arrays.set(desc.file, { arr: a, f64: decode(a) });
    }
    return this._arrays.get(desc.file).f64;
  }

  /** Scalar series and zonal-mean profiles for the whole control record (computed once). */
  async diagnostics(onProgress) {
    if (!this._diag) {
      this._diag = await computeDiagnostics({
        model: this.model, grid: this.grid, nt: this.nt, getPsi: (t) => this.getPsi(t), onProgress,
      });
    }
    return this._diag;
  }

  /** The cached series for (variable, layer), or undefined (does not compute anything). */
  cachedSeries(variable, layer, opts = {}) { return this._series.get(`${variable}|${layer}|${opts.total ? 1 : 0}`); }

  /**
   * Time series of a derived field at every grid point: Float32Array (nt * N).
   * Cached for the three most recent (variable, layer) combinations.
   */
  async fieldSeries(variable, layer, opts = {}, onProgress = null) {
    const key = `${variable}|${layer}|${opts.total ? 1 : 0}`;
    const hit = this._series.get(key);
    if (hit) return hit;
    const N = this.grid.N;
    const out = new Float32Array(this.nt * N);
    let last = performance.now();
    for (let t = 0; t < this.nt; t++) {
      const [p1, p2] = this.getPsi(t);                       // temporary snapshot: do not thrash the cache
      out.set(new Snapshot(this.model, this.spec, p1, p2).field(variable, layer, opts), t * N);
      if (onProgress && performance.now() - last > 40) { last = performance.now(); await onProgress((t + 1) / this.nt); }
    }
    this._series.set(key, out);
    return out;
  }
}
