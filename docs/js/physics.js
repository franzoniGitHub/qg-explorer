// Two-layer QG diagnostics computed in the browser from the streamfunction data.
// Pure functions and classes (no DOM): the same definitions as qg2l/analysis.py and
// qg2l/model.py, cross-checked in tests/js against numbers computed with numpy.

import { FFT2 } from './fft.js';

const TWO_PI = 2 * Math.PI;

/** Model parameters + derived quantities, from manifest.model. */
export class Model {
  constructor(m) {
    Object.assign(this, m);
    this.kd2 = 1 / (m.Ld * m.Ld);
    this.F1 = this.kd2 * (1 - m.delta);
    this.F2 = this.kd2 * m.delta;
    this.G = this.kd2 * m.delta * (1 - m.delta);
    this.dU = m.U1 - m.U2;
    this.beta1 = m.beta + this.F1 * this.dU;
    this.beta2 = m.beta - this.F2 * this.dU;
    this.dUcrit = m.beta / this.F2;
  }
  /** geopotential height anomaly (m) per unit streamfunction (m2/s) */
  get zFactor() { return this.f0 / this.g; }
}

/** Grid, wavenumbers (full complex-FFT layout, row = y, column = x) and shell indices. */
export class Grid {
  constructor({ nx, ny, Lx, Ly }) {
    Object.assign(this, { nx, ny, Lx, Ly });
    this.dx = Lx / nx;
    this.dy = Ly / ny;
    this.N = nx * ny;
    const N = this.N;
    this.kx = new Float64Array(N);
    this.ky = new Float64Array(N);
    this.kxd = new Float64Array(N);   // derivative wavenumbers (Nyquist set to 0)
    this.kyd = new Float64Array(N);
    this.K2 = new Float64Array(N);
    this.shell = new Int16Array(N);
    const dk = TWO_PI / Lx;
    for (let j = 0; j < ny; j++) {
      const my = j <= ny / 2 ? j : j - ny;
      for (let i = 0; i < nx; i++) {
        const mx = i <= nx / 2 ? i : i - nx;
        const p = j * nx + i;
        const kx = (TWO_PI * mx) / Lx;
        const ky = (TWO_PI * my) / Ly;
        this.kx[p] = kx;
        this.ky[p] = ky;
        this.kxd[p] = i === nx / 2 ? 0 : kx;
        this.kyd[p] = j === ny / 2 ? 0 : ky;
        this.K2[p] = kx * kx + ky * ky;
        this.shell[p] = Math.round(Math.sqrt(this.K2[p]) / dk);
      }
    }
    this.nShell = nx / 2;     // shells 1 .. nx/2
    this.kCut = Math.floor(nx / 3);   // 2/3 truncation: shells above this are empty in the model
    this.x = Float64Array.from({ length: nx }, (_, i) => (i + 0.5) * this.dx);
    this.y = Float64Array.from({ length: ny }, (_, j) => (j + 0.5) * this.dy);
  }
}

/** Spectral helper: owns the FFT plan and scratch buffers. */
export class Spectral {
  constructor(grid) {
    this.grid = grid;
    this.fft = new FFT2(grid.ny, grid.nx);
    this._re = new Float64Array(grid.N);
    this._im = new Float64Array(grid.N);
  }
  forward(f) { return this.fft.forwardReal(f); }
  inverse(hat) { return this.fft.inverseReal(hat.re, hat.im); }
  /** real field whose transform is c * (i*kx|i*ky|1) * hat, as selected by `op` */
  apply(hat, op) {
    const g = this.grid, N = g.N;
    const re = this._re, im = this._im;
    for (let p = 0; p < N; p++) {
      const a = hat.re[p], b = hat.im[p];
      switch (op) {
        case 'dx': re[p] = -g.kxd[p] * b; im[p] = g.kxd[p] * a; break;          // i kx
        case 'dy': re[p] = -g.kyd[p] * b; im[p] = g.kyd[p] * a; break;          // i ky
        case 'u': re[p] = g.kyd[p] * b; im[p] = -g.kyd[p] * a; break;           // -i ky
        case 'lap': re[p] = -g.K2[p] * a; im[p] = -g.K2[p] * b; break;
        default: re[p] = a; im[p] = b;
      }
    }
    return this.fft.inverseReal(re, im);
  }
}

// ---- layer selections ---------------------------------------------------------------
// Every selection is a linear combination  a*psi1 + b*psi2.
export function layerWeights(model, layer) {
  switch (layer) {
    case 'upper': return [1, 0];
    case 'lower': return [0, 1];
    case 'barotropic': return [model.delta, 1 - model.delta];
    case 'baroclinic': return [1, -1];
    default: throw new Error(`unknown layer selection '${layer}'`);
  }
}

export const LAYER_LABELS = {
  upper: 'Upper layer', lower: 'Lower layer',
  barotropic: 'Barotropic (depth-weighted mean)', baroclinic: 'Baroclinic (upper − lower)',
};

/** Variables the Fields / Statistics tabs can show. `scale` converts SI -> display unit. */
export const VARIABLES = {
  Z:     { label: 'Geopotential height anomaly Z′', short: 'Z′', unit: 'm', scale: 1, diverging: true },
  psi:   { label: 'Streamfunction ψ', short: 'ψ', unit: '10⁶ m² s⁻¹', scale: 1e-6, diverging: true },
  u:     { label: 'Zonal wind u′ (perturbation)', short: 'u′', unit: 'm s⁻¹', scale: 1, diverging: true },
  v:     { label: 'Meridional wind v′', short: 'v′', unit: 'm s⁻¹', scale: 1, diverging: true },
  zeta:  { label: 'Relative vorticity ζ', short: 'ζ', unit: '10⁻⁵ s⁻¹', scale: 1e5, diverging: true },
  q:     { label: 'PV anomaly q′', short: 'q′', unit: '10⁻⁵ s⁻¹', scale: 1e5, diverging: true },
  speed: { label: 'Total wind speed |U + u′|', short: '|V|', unit: 'm s⁻¹', scale: 1, diverging: false },
  ke:    { label: 'Eddy kinetic energy ½|u′|²', short: 'KE′', unit: 'm² s⁻²', scale: 1, diverging: false },
};

/**
 * One snapshot of the streamfunction (psi1, psi2 in m2/s, Float64Array ny*nx).
 * Spectral coefficients and derived fields are computed on demand and cached.
 */
export class Snapshot {
  constructor(model, spec, psi1, psi2) {
    this.model = model;
    this.spec = spec;
    this.grid = spec.grid;
    this.psi = [psi1, psi2];
    this._hat = [null, null];
    this._cache = new Map();
  }

  hat(layer) {   // layer index 0 | 1
    if (!this._hat[layer]) this._hat[layer] = this.spec.forward(this.psi[layer]);
    return this._hat[layer];
  }

  /** hat of a*psi1 + b*psi2 */
  hatCombo(a, b) {
    const N = this.grid.N;
    const re = new Float64Array(N), im = new Float64Array(N);
    if (a !== 0) { const h = this.hat(0); for (let p = 0; p < N; p++) { re[p] += a * h.re[p]; im[p] += a * h.im[p]; } }
    if (b !== 0) { const h = this.hat(1); for (let p = 0; p < N; p++) { re[p] += b * h.re[p]; im[p] += b * h.im[p]; } }
    return { re, im };
  }

  /** psi of a layer selection (physical space) */
  psiLayer(layer) {
    const [a, b] = layerWeights(this.model, layer);
    const N = this.grid.N, out = new Float64Array(N);
    for (let p = 0; p < N; p++) out[p] = a * this.psi[0][p] + b * this.psi[1][p];
    return out;
  }

  /**
   * Derived field in SI units.  options.total adds the imposed mean wind (u, speed).
   */
  field(variable, layer, { total = false } = {}) {
    const key = `${variable}|${layer}|${total ? 1 : 0}`;
    if (this._cache.has(key)) return this._cache.get(key);
    const m = this.model, N = this.grid.N;
    const [a, b] = layerWeights(m, layer);
    let out;
    if (variable === 'psi') out = this.psiLayer(layer);
    else if (variable === 'Z') { out = this.psiLayer(layer); for (let p = 0; p < N; p++) out[p] *= m.zFactor; }
    else if (variable === 'u' || variable === 'v' || variable === 'zeta') {
      const hat = this.hatCombo(a, b);
      out = this.spec.apply(hat, variable === 'u' ? 'u' : variable === 'v' ? 'dx' : 'lap');
      if (variable === 'u' && total) {
        const U = a * m.U1 + b * m.U2;
        for (let p = 0; p < N; p++) out[p] += U;
      }
    } else if (variable === 'q') {
      // q_i = lap psi_i + F_i (psi_j - psi_i), combined linearly
      const h1 = this.hat(0), h2 = this.hat(1), K2 = this.grid.K2;
      const re = new Float64Array(N), im = new Float64Array(N);
      for (let p = 0; p < N; p++) {
        const q1r = -K2[p] * h1.re[p] + m.F1 * (h2.re[p] - h1.re[p]);
        const q1i = -K2[p] * h1.im[p] + m.F1 * (h2.im[p] - h1.im[p]);
        const q2r = -K2[p] * h2.re[p] + m.F2 * (h1.re[p] - h2.re[p]);
        const q2i = -K2[p] * h2.im[p] + m.F2 * (h1.im[p] - h2.im[p]);
        re[p] = a * q1r + b * q2r;
        im[p] = a * q1i + b * q2i;
      }
      out = this.spec.fft.inverseReal(re, im);
    } else if (variable === 'ke' || variable === 'speed') {
      const u = this.field('u', layer, { total: variable === 'speed' });
      const v = this.field('v', layer);
      out = new Float64Array(N);
      if (variable === 'ke') for (let p = 0; p < N; p++) out[p] = 0.5 * (u[p] * u[p] + v[p] * v[p]);
      else for (let p = 0; p < N; p++) out[p] = Math.hypot(u[p], v[p]);
    } else throw new Error(`unknown variable '${variable}'`);
    this._cache.set(key, out);
    return out;
  }
}

// ---- domain-integrated diagnostics -----------------------------------------------------
/** Sum of |hat|^2 weighted by w[p], divided by N^2: the domain mean of f^2 (Parseval). */
function meanSq(hat, w, N) {
  let s = 0;
  for (let p = 0; p < N; p++) s += (w ? w[p] : 1) * (hat.re[p] * hat.re[p] + hat.im[p] * hat.im[p]);
  return s / (N * N);
}

/** KE1, KE2, APE, E, heat flux <v tau>, conversion C_B and Ekman dissipation D_E (m2 s-3). */
export function energetics(model, grid, h1, h2) {
  const N = grid.N, K2 = grid.K2;
  const ke1 = 0.5 * meanSq(h1, K2, N);
  const ke2 = 0.5 * meanSq(h2, K2, N);
  const tau = { re: new Float64Array(N), im: new Float64Array(N) };
  for (let p = 0; p < N; p++) { tau.re[p] = h1.re[p] - h2.re[p]; tau.im[p] = h1.im[p] - h2.im[p]; }
  const ape = 0.5 * model.G * meanSq(tau, null, N);
  // <v1 tau>, v1 = d psi1 / dx -> sum Re(conj(i kx h1) tau)
  let vt = 0;
  for (let p = 0; p < N; p++) {
    const k = grid.kxd[p];
    const vr = -k * h1.im[p], vi = k * h1.re[p];
    vt += vr * tau.re[p] + vi * tau.im[p];
  }
  vt /= N * N;
  const delta = model.delta;
  return {
    KE1: ke1, KE2: ke2, APE: ape, KE: delta * ke1 + (1 - delta) * ke2,
    E: delta * ke1 + (1 - delta) * ke2 + ape,
    heat_flux: vt, C_B: model.G * model.dU * vt, D_E: (1 - delta) * model.r_ek * 2 * ke2,
  };
}

/** Kinetic-energy spectrum in integer wavenumber shells 1..nx/2 for one layer's psi hat. */
export function keSpectrum(grid, hat, out = null) {
  const N = grid.N;
  const spec = out || new Float64Array(grid.nShell);
  for (let p = 0; p < N; p++) {
    const s = grid.shell[p];
    if (s >= 1 && s <= grid.nShell) spec[s - 1] += (0.5 * grid.K2[p] * (hat.re[p] * hat.re[p] + hat.im[p] * hat.im[p])) / (N * N);
  }
  return spec;
}

/** Spectrum of an arbitrary scalar-field hat, weighted by `w(p)` (e.g. APE of tau). */
export function shellSum(grid, hat, weight) {
  const N = grid.N, out = new Float64Array(grid.nShell);
  for (let p = 0; p < N; p++) {
    const s = grid.shell[p];
    if (s >= 1 && s <= grid.nShell) out[s - 1] += (weight * (hat.re[p] * hat.re[p] + hat.im[p] * hat.im[p])) / (N * N);
  }
  return out;
}

export function zonalMean(f, ny, nx) {
  const out = new Float64Array(ny);
  for (let j = 0; j < ny; j++) {
    let s = 0;
    for (let i = 0; i < nx; i++) s += f[j * nx + i];
    out[j] = s / nx;
  }
  return out;
}

/**
 * Diagnostics for a whole record: scalar series and (nt x ny) zonal-mean profiles.
 * `getPsi(t)` must return [psi1, psi2]. Async so the UI can show progress.
 */
export async function computeDiagnostics({ model, grid, nt, getPsi, onProgress = null }) {
  const spec = new Spectral(grid);
  const { nx, ny, N } = grid;
  const names = ['KE1', 'KE2', 'APE', 'E', 'heat_flux', 'C_B', 'D_E',
                 'EKE1', 'EKE2', 'ZKE1', 'ZKE2', 'EAPE', 'ZAPE'];
  const s = Object.fromEntries(names.map((k) => [k, new Float64Array(nt)]));
  const pnames = ['ubar1', 'ubar2', 'uv1', 'uv2', 'vt', 'eke1', 'eke2', 'tbar'];
  const prof = Object.fromEntries(pnames.map((k) => [k, new Float64Array(nt * ny)]));
  let tLast = performance_now();
  for (let t = 0; t < nt; t++) {
    const [p1, p2] = getPsi(t);
    const snap = new Snapshot(model, spec, p1, p2);
    const h1 = snap.hat(0), h2 = snap.hat(1);
    const e = energetics(model, grid, h1, h2);
    for (const k of ['KE1', 'KE2', 'APE', 'E', 'heat_flux', 'C_B', 'D_E']) s[k][t] = e[k];
    const u1 = snap.field('u', 'upper'), v1 = snap.field('v', 'upper');
    const u2 = snap.field('u', 'lower'), v2 = snap.field('v', 'lower');
    const tau = new Float64Array(N);
    for (let p = 0; p < N; p++) tau[p] = p1[p] - p2[p];
    const ub1 = zonalMean(u1, ny, nx), ub2 = zonalMean(u2, ny, nx), tb = zonalMean(tau, ny, nx);
    let eke1 = 0, eke2 = 0, zke1 = 0, zke2 = 0, zape = 0;
    for (let j = 0; j < ny; j++) {
      let a1 = 0, a2 = 0, c1 = 0, c2 = 0, vt = 0;
      for (let i = 0; i < nx; i++) {
        const p = j * nx + i;
        const up1 = u1[p] - ub1[j], up2 = u2[p] - ub2[j], tp = tau[p] - tb[j];
        a1 += 0.5 * (up1 * up1 + v1[p] * v1[p]);
        a2 += 0.5 * (up2 * up2 + v2[p] * v2[p]);
        c1 += up1 * v1[p];
        c2 += up2 * v2[p];
        vt += v1[p] * tp;
      }
      const o = t * ny + j;
      prof.ubar1[o] = ub1[j]; prof.ubar2[o] = ub2[j];
      prof.eke1[o] = a1 / nx; prof.eke2[o] = a2 / nx;
      prof.uv1[o] = c1 / nx; prof.uv2[o] = c2 / nx; prof.vt[o] = vt / nx;
      prof.tbar[o] = tb[j];
      eke1 += a1 / nx; eke2 += a2 / nx;
      zke1 += 0.5 * ub1[j] * ub1[j]; zke2 += 0.5 * ub2[j] * ub2[j];
      zape += 0.5 * model.G * tb[j] * tb[j];
    }
    s.EKE1[t] = eke1 / ny; s.EKE2[t] = eke2 / ny;
    s.ZKE1[t] = zke1 / ny; s.ZKE2[t] = zke2 / ny;
    s.ZAPE[t] = zape / ny; s.EAPE[t] = e.APE - zape / ny;
    if (onProgress && (t % 16 === 15 || t === nt - 1)) {
      const now = performance_now();
      if (now - tLast > 40) { tLast = now; await onProgress((t + 1) / nt); }
    }
  }
  return { scalars: s, profiles: prof, nt, ny };
}

function performance_now() {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}
