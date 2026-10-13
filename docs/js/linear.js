// Linear (normal-mode) stability of the two-layer basic state, evaluated in closed form.
// Same problem as qg2l/stability.py:  d psi_hat/dt = A^-1 M psi_hat  with
//   A = [[-K2-F1, F1], [F2, -K2-F2]],  M = -i kx (diag(U) A + diag(beta_i)) + diag(0, r K2).

function csqrt(re, im) {
  const r = Math.hypot(re, im);
  const a = Math.sqrt(Math.max(0, (r + re) / 2));
  const b = Math.sqrt(Math.max(0, (r - re) / 2));
  return { re: a, im: im < 0 ? -b : b };
}

/** Plain parameter bag from the model object or manifest.model. */
export function linearParams(m) {
  const kd2 = 1 / (m.Ld * m.Ld);
  const F1 = kd2 * (1 - m.delta), F2 = kd2 * m.delta;
  const dU = m.U1 - m.U2;
  return { U1: m.U1, U2: m.U2, beta: m.beta, F1, F2, kd2, r: m.r_ek, beta1: m.beta + F1 * dU, beta2: m.beta - F2 * dU, dU };
}

/** The 2x2 complex matrix N (psi_hat tendency matrix) as [[a,b],[c,d]] of {re,im}. */
function tendencyMatrix(p, kx, ky) {
  const K2 = kx * kx + ky * ky;
  if (K2 === 0) return null;
  const det = K2 * (K2 + p.kd2);
  const i11 = -(K2 + p.F2) / det, i12 = -p.F1 / det, i21 = -p.F2 / det, i22 = -(K2 + p.F1) / det;
  const A11 = -K2 - p.F1, A12 = p.F1, A21 = p.F2, A22 = -K2 - p.F2;
  const B11 = p.U1 * A11 + p.beta1, B12 = p.U1 * A12, B21 = p.U2 * A21, B22 = p.U2 * A22 + p.beta2;
  const C11 = i11 * B11 + i12 * B21, C12 = i11 * B12 + i12 * B22;
  const C21 = i21 * B11 + i22 * B21, C22 = i21 * B12 + i22 * B22;
  const rK = p.r * K2;
  const D12 = i12 * rK, D22 = i22 * rK;
  return [[{ re: 0, im: -kx * C11 }, { re: D12, im: -kx * C12 }],
          [{ re: 0, im: -kx * C21 }, { re: D22, im: -kx * C22 }]];
}

/** Both eigenvalues of the normal-mode problem: [{re, im}, {re, im}]  (re = growth rate, s^-1). */
export function eigenvalues(p, kx, ky) {
  const N = tendencyMatrix(p, kx, ky);
  if (!N) return [{ re: 0, im: 0 }, { re: 0, im: 0 }];
  const [[a, b], [c, d]] = N;
  const trRe = a.re + d.re, trIm = a.im + d.im;
  const detRe = a.re * d.re - a.im * d.im - (b.re * c.re - b.im * c.im);
  const detIm = a.re * d.im + a.im * d.re - (b.re * c.im + b.im * c.re);
  const discRe = trRe * trRe - trIm * trIm - 4 * detRe;
  const discIm = 2 * trRe * trIm - 4 * detIm;
  const s = csqrt(discRe, discIm);
  return [{ re: (trRe + s.re) / 2, im: (trIm + s.im) / 2 }, { re: (trRe - s.re) / 2, im: (trIm - s.im) / 2 }];
}

/** Growth rate (s^-1) of the fastest-growing mode at (kx, ky). */
export function growthRate(p, kx, ky) {
  const e = eigenvalues(p, kx, ky);
  return Math.max(e[0].re, e[1].re);
}

/** Growing mode: eigenvalue, phase speed (m/s) and the vertical structure of psi. */
export function mode(p, kx, ky) {
  const N = tendencyMatrix(p, kx, ky);
  const e = eigenvalues(p, kx, ky);
  const lam = e[0].re >= e[1].re ? e[0] : e[1];
  if (!N) return { lambda: lam, sigma: 0, c: 0, ratio: NaN, lag: NaN };
  const [[a, b]] = N;
  // eigenvector (v1, v2) = (b, lambda - a)
  const v1 = { re: b.re, im: b.im };
  const v2 = { re: lam.re - a.re, im: lam.im - a.im };
  const amp1 = Math.hypot(v1.re, v1.im), amp2 = Math.hypot(v2.re, v2.im);
  // phase of psi2 relative to psi1 (radians, in (-pi, pi]): psi ~ Re[v exp(i kx x)]
  const lag = Math.atan2(v2.im * v1.re - v2.re * v1.im, v2.re * v1.re + v2.im * v1.im);
  return { lambda: lam, sigma: lam.re, c: kx !== 0 ? -lam.im / kx : NaN, ratio: amp1 > 0 ? amp2 / amp1 : NaN, lag };
}

/** Growth-rate curve over zonal wavenumbers (ky fixed). */
export function growthCurve(p, kxs, ky = 0) {
  return Float64Array.from(kxs, (k) => growthRate(p, k, ky));
}

/** Growth rate on a (kx, ky) grid: result[j*nkx + i]. */
export function growthMap(p, kxs, kys) {
  const out = new Float64Array(kxs.length * kys.length);
  for (let j = 0; j < kys.length; j++) for (let i = 0; i < kxs.length; i++) out[j * kxs.length + i] = growthRate(p, kxs[i], kys[j]);
  return out;
}

/** Fastest-growing zonal mode on a fine grid up to kmax. */
export function fastestMode(p, kmax, n = 600) {
  let best = { k: 0, sigma: -Infinity };
  for (let i = 1; i <= n; i++) {
    const k = (kmax * i) / n;
    const s = growthRate(p, k, 0);
    if (s > best.sigma) best = { k, sigma: s };
  }
  return best;
}

/** Phillips (1954): necessary and sufficient condition for equal-depth layers, shear > beta / F2. */
export function criticalShear(p) {
  return p.beta / p.F2;
}
