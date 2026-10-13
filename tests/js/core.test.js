import test from 'node:test';
import assert from 'node:assert/strict';

import { FFT, FFT2 } from '../../docs/js/fft.js';
import { symmetricEigen, eof } from '../../docs/js/eof.js';
import * as S from '../../docs/js/stats.js';
import { welch, kOmega } from '../../docs/js/spectra.js';
import { rng, gauss, close } from './helpers.js';

test('FFT matches a naive DFT', () => {
  const n = 16, r = rng(3);
  const re = Float64Array.from({ length: n }, () => r() - 0.5);
  const im = Float64Array.from({ length: n }, () => r() - 0.5);
  const xr = Float64Array.from(re), xi = Float64Array.from(im);
  new FFT(n).transform(xr, xi, false);
  for (let k = 0; k < n; k++) {
    let sr = 0, si = 0;
    for (let j = 0; j < n; j++) {
      const a = (-2 * Math.PI * j * k) / n;
      sr += re[j] * Math.cos(a) - im[j] * Math.sin(a);
      si += re[j] * Math.sin(a) + im[j] * Math.cos(a);
    }
    assert.ok(Math.abs(sr - xr[k]) < 1e-12 && Math.abs(si - xi[k]) < 1e-12, `bin ${k}`);
  }
});

test('2-D FFT: round trip, Parseval and a single plane wave', () => {
  const ny = 8, nx = 16, f = new FFT2(ny, nx), r = rng(5);
  const field = Float64Array.from({ length: ny * nx }, () => r() - 0.5);
  const h = f.forwardReal(field);
  const back = f.inverseReal(h.re, h.im);
  for (let i = 0; i < field.length; i++) assert.ok(Math.abs(back[i] - field[i]) < 1e-12);
  let e = 0, eh = 0;
  for (let i = 0; i < field.length; i++) { e += field[i] ** 2; eh += h.re[i] ** 2 + h.im[i] ** 2; }
  assert.ok(close(e, eh / field.length, 1e-12));
  const wave = Float64Array.from({ length: ny * nx }, (_, p) => Math.cos((2 * Math.PI * 3 * (p % nx)) / nx + (2 * Math.PI * 2 * Math.floor(p / nx)) / ny));
  const hw = f.forwardReal(wave);
  const idx = 2 * nx + 3;                      // (ky=2, kx=3)
  assert.ok(close(hw.re[idx], (nx * ny) / 2, 1e-12));
});

test('symmetricEigen: residual, orthonormality and trace', () => {
  const n = 24, r = rng(9);
  const A = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) A[i * n + j] = A[j * n + i] = gauss(r);
  const { values, vectors } = symmetricEigen(A, n);
  let tr = 0, sum = 0;
  for (let i = 0; i < n; i++) { tr += A[i * n + i]; sum += values[i]; }
  assert.ok(Math.abs(tr - sum) < 1e-10);
  for (let k = 1; k < n; k++) assert.ok(values[k - 1] >= values[k]);
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < n; i++) {
      let av = 0;
      for (let j = 0; j < n; j++) av += A[i * n + j] * vectors[j * n + k];
      assert.ok(Math.abs(av - values[k] * vectors[i * n + k]) < 1e-9, `residual k=${k}`);
    }
    for (let l = 0; l < n; l++) {
      let d = 0;
      for (let i = 0; i < n; i++) d += vectors[i * n + k] * vectors[i * n + l];
      assert.ok(Math.abs(d - (k === l ? 1 : 0)) < 1e-10);
    }
  }
});

test('EOF recovers two orthogonal patterns with known variances (both covariance branches)', () => {
  const r = rng(11);
  for (const [T, N] of [[200, 12], [12, 200]]) {
    const p1 = Float64Array.from({ length: N }, (_, i) => Math.sin((2 * Math.PI * i) / N));
    const p2 = Float64Array.from({ length: N }, (_, i) => Math.cos((4 * Math.PI * i) / N));
    const n1 = Math.hypot(...p1), n2 = Math.hypot(...p2);
    const X = new Float64Array(T * N);
    const a1 = [], a2 = [];
    for (let t = 0; t < T; t++) { a1.push(3 * gauss(r)); a2.push(1 * gauss(r)); }
    for (let t = 0; t < T; t++) for (let i = 0; i < N; i++) X[t * N + i] = 5 + (a1[t] * p1[i]) / n1 + (a2[t] * p2[i]) / n2;
    const res = eof(X, T, N, 3);
    assert.ok(res.fractions[0] > 0.8 && res.fractions[0] < 0.97);
    const dot = res.patterns[0].reduce((s, v, i) => s + v * p1[i] / n1, 0);
    assert.ok(Math.abs(dot) > 0.95, `pattern 1 projection ${dot}`);
    const var1 = res.pcs[0].reduce((s, v) => s + v * v, 0) / T;
    assert.ok(close(var1, res.values[0], 1e-9));
    assert.ok(res.fractions[0] + res.fractions[1] + res.fractions[2] <= 1 + 1e-9);
  }
});

test('moments, histogram, autocorrelation, linear fit', () => {
  const r = rng(2);
  const x = Float64Array.from({ length: 200000 }, () => gauss(r));
  const m = S.moments(x);
  assert.ok(Math.abs(m.mean) < 0.01 && Math.abs(m.std - 1) < 0.01 && Math.abs(m.skew) < 0.03 && Math.abs(m.kurt - 3) < 0.05);
  const h = S.histogram(x, -4, 4, 16);
  assert.equal(h.counts.reduce((a, b) => a + b, 0) + h.outside, x.length);
  // AR(1) with phi = 0.8: acf(k) = 0.8^k
  const ar = new Float64Array(100000);
  for (let i = 1; i < ar.length; i++) ar[i] = 0.8 * ar[i - 1] + gauss(r);
  const acf = S.autocorrelation(ar, 5);
  for (let k = 0; k <= 5; k++) assert.ok(Math.abs(acf[k] - 0.8 ** k) < 0.02);
  assert.ok(Math.abs(S.efoldingLag(acf) - 1 / -Math.log(0.8)) < 0.3);   // 1/e at lag 4.48
  const fit = S.linearFit([0, 1, 2, 3], [1, 3, 5, 7]);
  assert.ok(close(fit.a, 1) && close(fit.b, 2) && close(fit.r2, 1));
  assert.ok(close(S.pearson([1, 2, 3, 4], [2, 4, 6, 8]), 1));
});

test('Welch spectrum finds the sinusoid frequency and integrates to the variance', () => {
  const n = 2048, dt = 0.5, f0 = 0.2;                 // cycles per unit time
  const r = rng(4);
  const x = Float64Array.from({ length: n }, (_, i) => 2 * Math.sin(2 * Math.PI * f0 * i * dt) + 0.3 * gauss(r));
  const { freq, psd } = welch(x, 256, dt);
  let imax = 0;
  for (let k = 1; k < psd.length; k++) if (psd[k] > psd[imax]) imax = k;
  assert.ok(Math.abs(freq[imax] - f0) < 1 / (256 * dt));
  let tot = 0;
  for (let k = 0; k < psd.length; k++) tot += psd[k] * (freq[1] - freq[0]);
  assert.ok(Math.abs(tot - S.variance(x)) / S.variance(x) < 0.1);
});

test('k-omega spectrum puts an eastward wave at (m > 0, freq > 0)', () => {
  const nt = 256, nx = 32, dt = 0.5;
  const m0 = 3, f0 = 0.25;
  for (const dir of [+1, -1]) {
    const d = new Float64Array(nt * nx);
    for (let t = 0; t < nt; t++) for (let i = 0; i < nx; i++) {
      d[t * nx + i] = Math.cos((2 * Math.PI * m0 * i) / nx - dir * 2 * Math.PI * f0 * t * dt);
    }
    const { m, freq, power, nFreq } = kOmega(d, nt, nx, 64, dt);
    let best = [0, 0], bv = -1;
    for (let k = 1; k < nFreq; k++) for (let i = 0; i < nx; i++) if (power[k * nx + i] > bv) { bv = power[k * nx + i]; best = [k, i]; }
    assert.equal(Math.sign(m[best[1]]), dir, 'direction of propagation');
    assert.equal(Math.abs(m[best[1]]), m0);
    assert.ok(Math.abs(freq[best[0]] - f0) <= 1 / (64 * dt));
  }
});

test('normalQuantile inverts the normal CDF', () => {
  const q = S.normalQuantile;
  assert.ok(Math.abs(q(0.5)) < 1e-12);
  assert.ok(Math.abs(q(0.975) - 1.959963984540054) < 1e-8);
  assert.ok(Math.abs(q(0.001) + 3.090232306167813) < 1e-8);
  assert.ok(Math.abs(q(1e-5) + 4.264890793922825) < 1e-7);
  for (const p of [0.01, 0.2, 0.7, 0.99]) assert.ok(Math.abs(q(p) + q(1 - p)) < 1e-9);
  // round trip through the Gaussian pdf: the cumulative integral of the pdf up to q(p) is p
  const z = q(0.8413447460685429);
  assert.ok(Math.abs(z - 1) < 1e-8);
  assert.equal(q(0), -Infinity);
  assert.equal(q(1), Infinity);
});
