// The browser-side analysis, run on a real dataset, against numbers computed with numpy
// (manifest.reference, written by qg2l/reference.py from the same exported data).
// Dataset: docs/data/default, or the folder in $QG_DATA.
import test from 'node:test';
import assert from 'node:assert/strict';

import { Dataset } from '../../docs/js/dataset.js';
import { energetics, keSpectrum, zonalMean } from '../../docs/js/physics.js';
import { eof } from '../../docs/js/eof.js';
import * as S from '../../docs/js/stats.js';
import { linearParams, mode, criticalShear } from '../../docs/js/linear.js';
import { NodeSource, DATA_DIR, close } from './helpers.js';

const ds = await Dataset.open(new NodeSource(DATA_DIR));
const ref = ds.manifest.reference;

test('energetics of single snapshots match numpy', () => {
  for (const r of ref.snapshots) {
    const s = ds.snapshot(r.index);
    const e = energetics(ds.model, ds.grid, s.hat(0), s.hat(1));
    for (const k of ['KE1', 'KE2', 'APE', 'heat_flux', 'C_B', 'D_E']) {
      assert.ok(close(e[k], r[k], 1e-8, 1e-14), `${k}: ${e[k]} vs ${r[k]} (snapshot ${r.index})`);
    }
  }
});

test('kinetic-energy spectrum, vorticity and psi rms match numpy', () => {
  for (const r of ref.snapshots) {
    const s = ds.snapshot(r.index);
    const spec = keSpectrum(ds.grid, s.hat(0));
    r.spec_ke1.forEach((v, i) => assert.ok(close(spec[i], v, 1e-8, 1e-14), `shell ${i + 1}`));
    const zeta = s.field('zeta', 'upper');
    const rms = Math.sqrt(zeta.reduce((a, v) => a + v * v, 0) / zeta.length);
    assert.ok(close(rms, r.zeta1_rms, 1e-8));
    const psi = s.field('psi', 'upper');
    assert.ok(close(Math.sqrt(psi.reduce((a, v) => a + v * v, 0) / psi.length), r.psi1_rms, 1e-9));
  }
});

test('Parseval: shell-summed spectrum equals the layer kinetic energy (up to the corner modes)', () => {
  const s = ds.snapshot(ref.snapshots[0].index);
  const e = energetics(ds.model, ds.grid, s.hat(0), s.hat(1));
  const spec = keSpectrum(ds.grid, s.hat(0));
  const tot = spec.reduce((a, b) => a + b, 0);
  assert.ok(tot <= e.KE1 * (1 + 1e-9) && tot > 0.999 * e.KE1);
});

test('whole-record diagnostics: zonal-mean wind, EOF variance fractions, energy budget', async () => {
  const d = await ds.diagnostics();
  const { ny } = ds.grid, nt = ds.nt;
  // time-mean zonal-mean upper-layer wind, sampled like the reference
  const step = Math.max(1, ny / 16);
  const tm = [];
  for (let j = 0; j < ny; j += step) { let a = 0; for (let t = 0; t < nt; t++) a += d.profiles.ubar1[t * ny + j]; tm.push(a / nt); }
  ref.zonal_mean_u1_timemean.forEach((v, i) => assert.ok(close(tm[i], v, 1e-6, 1e-6), `u profile ${i}`));
  // EOFs of the zonal-mean wind of both layers (2*ny dims)
  const X = new Float64Array(nt * 2 * ny);
  for (let t = 0; t < nt; t++) for (let j = 0; j < ny; j++) { X[t * 2 * ny + j] = d.profiles.ubar1[t * ny + j]; X[t * 2 * ny + ny + j] = d.profiles.ubar2[t * ny + j]; }
  const res = eof(X, nt, 2 * ny, 4);
  ref.eof_zonal_wind_variance_fractions.forEach((v, i) => assert.ok(close(res.fractions[i], v, 1e-6, 1e-8), `EOF ${i + 1}: ${res.fractions[i]} vs ${v}`));
  assert.ok(close(res.total, ref.eof_zonal_wind_total_variance, 1e-8));
  // the model's own float64 energetics (written by the Python model) agree with the quantised data
  const me = ds.control.model_energetics;
  for (const k of ['KE1', 'KE2', 'APE', 'heat_flux']) {
    let maxrel = 0;
    for (let t = 0; t < nt; t++) maxrel = Math.max(maxrel, Math.abs(d.scalars[k][t] - me[k][t]) / Math.abs(me[k][t]));
    assert.ok(maxrel < 2e-3, `${k} max relative difference ${maxrel}`);
  }
  // partition: ZKE + EKE == KE_i
  for (let t = 0; t < nt; t += 17) {
    assert.ok(close(d.scalars.EKE1[t] + d.scalars.ZKE1[t], d.scalars.KE1[t], 1e-5, 1e-6), `KE partition t=${t}: ${d.scalars.EKE1[t] + d.scalars.ZKE1[t]} vs ${d.scalars.KE1[t]}`);
    assert.ok(close(d.scalars.EAPE[t] + d.scalars.ZAPE[t], d.scalars.APE[t], 1e-9, 1e-9));
  }
});

test('vorticity moments and histogram match numpy', async () => {
  const series = await ds.fieldSeries('zeta', 'upper');
  const m = S.moments(series);
  const r = ref.zeta1_moments;
  assert.ok(close(m.std, r.std, 1e-4) && close(m.skew, r.skew, 1e-2, 2e-3) && close(m.kurt, r.kurt, 1e-3));
  const z = Float64Array.from(series, (v) => (v - m.mean) / m.std);
  const h = S.histogram(z, -4, 4, 16);
  ref.zeta1_hist_fraction.forEach((f, i) => assert.ok(close(h.counts[i] / z.length, f, 1e-3, 2e-5), `bin ${i}`));
  const zm = S.moments(await ds.fieldSeries('Z', 'upper'));
  assert.ok(close(zm.std, ref.z1_moments.std, 1e-6) && close(zm.kurt, ref.z1_moments.kurt, 1e-4));
});

test('linear-stability closed form matches numpy eigenvalues (growth rate, phase speed, tilt)', () => {
  const p = linearParams(ds.manifest.model);
  for (const r of ref.linear_modes) {
    const m = mode(p, r.kx, r.ky);
    assert.ok(close(m.sigma, r.sigma, 1e-8, 1e-14), `sigma m=${r.mx},${r.my}: ${m.sigma} vs ${r.sigma}`);
    if (r.sigma > 1e-8) {
      assert.ok(close(m.c, r.c, 1e-6, 1e-8), 'phase speed');
      assert.ok(close(m.ratio, r.ratio, 1e-6), 'amplitude ratio');
      assert.ok(Math.abs(Math.atan2(Math.sin(m.lag - r.lag), Math.cos(m.lag - r.lag))) < 1e-6, 'vertical phase lag');
    }
  }
  assert.ok(close(criticalShear(p), ds.manifest.model.dU_crit, 1e-12));
});

test('zonal mean helper', () => {
  const f = Float64Array.from({ length: 4 * 4 }, (_, p) => Math.floor(p / 4));
  assert.deepEqual(Array.from(zonalMean(f, 4, 4)), [0, 1, 2, 3]);
});
