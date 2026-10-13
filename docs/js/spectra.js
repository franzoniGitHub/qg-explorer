// Time/space spectral analysis: Welch power spectra and wavenumber-frequency (k-omega) spectra.
import { FFT } from './fft.js';

export function hann(n) {
  return Float64Array.from({ length: n }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n));
}

/**
 * Welch power spectral density of a real series sampled every `dt` (any time unit).
 * Segments of length `seg` (power of 2), 50 % overlap, Hann window, mean removed per segment.
 * Returns { freq (cycles per time unit), psd, nSeg } for freq >= 0 up to Nyquist.
 */
export function welch(series, seg, dt = 1) {
  const n = series.length;
  if (seg > n) seg = 1 << Math.floor(Math.log2(n));
  const fft = new FFT(seg);
  const w = hann(seg);
  let wss = 0;
  for (let i = 0; i < seg; i++) wss += w[i] * w[i];
  const half = seg / 2;
  const psd = new Float64Array(half + 1);
  let nSeg = 0;
  const re = new Float64Array(seg), im = new Float64Array(seg);
  for (let s = 0; s + seg <= n; s += seg / 2) {
    let m = 0;
    for (let i = 0; i < seg; i++) m += series[s + i];
    m /= seg;
    for (let i = 0; i < seg; i++) { re[i] = (series[s + i] - m) * w[i]; im[i] = 0; }
    fft.transform(re, im, false);
    for (let k = 0; k <= half; k++) {
      // one-sided density; integrates (sum psd * df) to the variance of the series
      psd[k] += ((k === 0 || k === half ? 1 : 2) * dt * (re[k] * re[k] + im[k] * im[k])) / wss;
    }
    nSeg++;
  }
  for (let k = 0; k <= half; k++) psd[k] /= nSeg;
  const freq = Float64Array.from({ length: half + 1 }, (_, k) => k / (seg * dt));
  return { freq, psd, nSeg };
}

/**
 * Wavenumber-frequency power spectrum of f(t, x) (rows = time, columns = x, periodic in x).
 * Convention: a wave cos(m x - omega t) with m > 0, omega > 0 (eastward) shows up at
 * (m > 0, freq > 0); a westward wave shows up at (m < 0, freq > 0).
 * Returns { m: wavenumbers -nx/2..nx/2-1, freq: 0..1/(2 dt) cycles per time unit, power[iw][im] }.
 */
export function kOmega(data, nt, nx, seg, dt = 1) {
  if (seg > nt) seg = 1 << Math.floor(Math.log2(nt));
  const fx = new FFT(nx), ft = new FFT(seg);
  const w = hann(seg);
  const half = seg / 2;
  const power = new Float64Array((half + 1) * nx);
  const re = new Float64Array(seg * nx), im = new Float64Array(seg * nx);
  const cr = new Float64Array(seg), ci = new Float64Array(seg);
  const rr = new Float64Array(nx), ri = new Float64Array(nx);
  let nSeg = 0;
  for (let s0 = 0; s0 + seg <= nt; s0 += seg / 2) {
    // remove the segment mean at each x, apply the time window
    for (let i = 0; i < nx; i++) {
      let m = 0;
      for (let t = 0; t < seg; t++) m += data[(s0 + t) * nx + i];
      m /= seg;
      for (let t = 0; t < seg; t++) { re[t * nx + i] = (data[(s0 + t) * nx + i] - m) * w[t]; im[t * nx + i] = 0; }
    }
    for (let t = 0; t < seg; t++) {            // forward transform in x
      rr.set(re.subarray(t * nx, (t + 1) * nx)); ri.set(im.subarray(t * nx, (t + 1) * nx));
      fx.transform(rr, ri, false);
      re.set(rr, t * nx); im.set(ri, t * nx);
    }
    for (let i = 0; i < nx; i++) {             // *inverse* transform in time (sign convention above)
      for (let t = 0; t < seg; t++) { cr[t] = re[t * nx + i]; ci[t] = im[t * nx + i]; }
      ft.transform(cr, ci, true);
      for (let k = 0; k <= half; k++) power[k * nx + i] += cr[k] * cr[k] + ci[k] * ci[k];
    }
    nSeg++;
  }
  // reorder x axis to -nx/2 .. nx/2-1
  const out = new Float64Array((half + 1) * nx);
  const m = Int16Array.from({ length: nx }, (_, i) => i - nx / 2);
  for (let k = 0; k <= half; k++) {
    for (let i = 0; i < nx; i++) {
      const src = (i + nx / 2) % nx;         // index of wavenumber m[i] in FFT layout
      out[k * nx + i] = power[k * nx + src] / nSeg;
    }
  }
  const freq = Float64Array.from({ length: half + 1 }, (_, k) => k / (seg * dt));
  return { m, freq, power: out, nx, nFreq: half + 1, nSeg };
}
