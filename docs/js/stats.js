// Basic statistics on typed arrays.  Pure functions, no DOM.

export function mean(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
}

export function variance(a, m = mean(a)) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - m; s += d * d; }
  return s / a.length;
}

/** mean, std, skewness, (non-excess) kurtosis -- population moments, like the Python reference. */
export function moments(a) {
  const n = a.length;
  const m = mean(a);
  let m2 = 0, m3 = 0, m4 = 0;
  for (let i = 0; i < n; i++) { const d = a[i] - m; const d2 = d * d; m2 += d2; m3 += d2 * d; m4 += d2 * d2; }
  m2 /= n; m3 /= n; m4 /= n;
  const sd = Math.sqrt(m2);
  return { mean: m, std: sd, skew: sd > 0 ? m3 / (sd * sd * sd) : 0, kurt: m2 > 0 ? m4 / (m2 * m2) : 0, n };
}

export function minMax(a) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < a.length; i++) { const v = a[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
  return [lo, hi];
}

/** Histogram with `nbins` equal bins on [lo, hi]; values outside are ignored (counted in `outside`). */
export function histogram(a, lo, hi, nbins) {
  const counts = new Float64Array(nbins);
  const w = (hi - lo) / nbins;
  let outside = 0;
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (!(v >= lo && v < hi)) { if (v === hi) counts[nbins - 1]++; else outside++; continue; }
    counts[Math.floor((v - lo) / w)]++;
  }
  const centers = Float64Array.from({ length: nbins }, (_, k) => lo + (k + 0.5) * w);
  return { counts, centers, width: w, outside, total: a.length };
}

/** Empirical quantile (linear interpolation) from a *sorted* array. */
export function quantileSorted(sorted, p) {
  const x = p * (sorted.length - 1);
  const i = Math.floor(x), f = x - i;
  return i + 1 < sorted.length ? sorted[i] * (1 - f) + sorted[i + 1] * f : sorted[i];
}

export function sortedCopy(a) {
  const c = Float64Array.from(a);
  c.sort();
  return c;
}

/** Pearson correlation of two equal-length series. */
export function pearson(a, b) {
  const ma = mean(a), mb = mean(b);
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] - ma, y = b[i] - mb;
    sab += x * y; saa += x * x; sbb += y * y;
  }
  return saa > 0 && sbb > 0 ? sab / Math.sqrt(saa * sbb) : 0;
}

/** Normalised autocorrelation function for lags 0..maxLag (biased estimator). */
export function autocorrelation(series, maxLag) {
  const n = series.length;
  const m = mean(series);
  const out = new Float64Array(maxLag + 1);
  let c0 = 0;
  for (let i = 0; i < n; i++) c0 += (series[i] - m) * (series[i] - m);
  for (let lag = 0; lag <= maxLag; lag++) {
    let s = 0;
    for (let i = 0; i + lag < n; i++) s += (series[i] - m) * (series[i + lag] - m);
    out[lag] = c0 > 0 ? s / c0 : 0;
  }
  return out;
}

/** First lag at which acf drops below 1/e (linear interpolation), or NaN. */
export function efoldingLag(acf) {
  const target = Math.exp(-1);
  for (let k = 1; k < acf.length; k++) {
    if (acf[k] <= target) {
      const f = (acf[k - 1] - target) / (acf[k - 1] - acf[k]);
      return k - 1 + f;
    }
  }
  return NaN;
}

/** Least-squares line y = a + b x; returns { a, b, r2 }. */
export function linearFit(x, y) {
  const n = x.length;
  let sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
  for (let i = 0; i < n; i++) { sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; sxy += x[i] * y[i]; syy += y[i] * y[i]; }
  const det = n * sxx - sx * sx;
  const b = det !== 0 ? (n * sxy - sx * sy) / det : 0;
  const a = (sy - b * sx) / n;
  const ssTot = syy - (sy * sy) / n;
  let ssRes = 0;
  for (let i = 0; i < n; i++) { const r = y[i] - a - b * x[i]; ssRes += r * r; }
  return { a, b, r2: ssTot > 0 ? 1 - ssRes / ssTot : 0 };
}

/** Normal probability density with the given mean and std, evaluated at x. */
export function normalPdf(x, mu, sd) {
  const z = (x - mu) / sd;
  return Math.exp(-0.5 * z * z) / (sd * Math.sqrt(2 * Math.PI));
}

/** Round-robin 'nice' axis limits symmetric about zero. */
export function symmetricLimit(a, pct = 0.995) {
  const abs = Float64Array.from(a, Math.abs);
  abs.sort();
  const v = quantileSorted(abs, pct);
  return v > 0 ? v : 1;
}

/** Inverse of the standard normal CDF (Acklam's rational approximation, relative error ~1e-9). */
export function normalQuantile(p) {
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  const pl = 0.02425;
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  if (p < pl) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - pl) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = p - 0.5, r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}
