// Empirical orthogonal functions (EOFs / principal components) in plain JavaScript.
// The symmetric eigen-solver is the classic Householder + implicit-QL scheme (EISPACK tred2/tql2,
// as in the public-domain JAMA library).

/** Eigen-decomposition of a symmetric n x n matrix (row-major Float64Array).
 *  Returns { values (descending), vectors } where eigenvector k is column k: vectors[r*n + k]. */
export function symmetricEigen(A, n) {
  const V = Float64Array.from(A);
  const d = new Float64Array(n);
  const e = new Float64Array(n);
  tred2(V, d, e, n);
  tql2(V, d, e, n);
  // sort descending
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => d[b] - d[a]);
  const values = Float64Array.from(order, (i) => d[i]);
  const vectors = new Float64Array(n * n);
  for (let k = 0; k < n; k++) for (let r = 0; r < n; r++) vectors[r * n + k] = V[r * n + order[k]];
  return { values, vectors };
}

function tred2(V, d, e, n) {
  const at = (r, c) => r * n + c;
  for (let j = 0; j < n; j++) d[j] = V[at(n - 1, j)];
  for (let i = n - 1; i > 0; i--) {
    let scale = 0, h = 0;
    for (let k = 0; k < i; k++) scale += Math.abs(d[k]);
    if (scale === 0) {
      e[i] = d[i - 1];
      for (let j = 0; j < i; j++) { d[j] = V[at(i - 1, j)]; V[at(i, j)] = 0; V[at(j, i)] = 0; }
    } else {
      for (let k = 0; k < i; k++) { d[k] /= scale; h += d[k] * d[k]; }
      let f = d[i - 1];
      let g = Math.sqrt(h);
      if (f > 0) g = -g;
      e[i] = scale * g;
      h -= f * g;
      d[i - 1] = f - g;
      for (let j = 0; j < i; j++) e[j] = 0;
      for (let j = 0; j < i; j++) {
        f = d[j];
        V[at(j, i)] = f;
        g = e[j] + V[at(j, j)] * f;
        for (let k = j + 1; k <= i - 1; k++) { g += V[at(k, j)] * d[k]; e[k] += V[at(k, j)] * f; }
        e[j] = g;
      }
      f = 0;
      for (let j = 0; j < i; j++) { e[j] /= h; f += e[j] * d[j]; }
      const hh = f / (h + h);
      for (let j = 0; j < i; j++) e[j] -= hh * d[j];
      for (let j = 0; j < i; j++) {
        f = d[j]; g = e[j];
        for (let k = j; k <= i - 1; k++) V[at(k, j)] -= f * e[k] + g * d[k];
        d[j] = V[at(i - 1, j)];
        V[at(i, j)] = 0;
      }
    }
    d[i] = h;
  }
  for (let i = 0; i < n - 1; i++) {
    V[at(n - 1, i)] = V[at(i, i)];
    V[at(i, i)] = 1;
    const h = d[i + 1];
    if (h !== 0) {
      for (let k = 0; k <= i; k++) d[k] = V[at(k, i + 1)] / h;
      for (let j = 0; j <= i; j++) {
        let g = 0;
        for (let k = 0; k <= i; k++) g += V[at(k, i + 1)] * V[at(k, j)];
        for (let k = 0; k <= i; k++) V[at(k, j)] -= g * d[k];
      }
    }
    for (let k = 0; k <= i; k++) V[at(k, i + 1)] = 0;
  }
  for (let j = 0; j < n; j++) { d[j] = V[at(n - 1, j)]; V[at(n - 1, j)] = 0; }
  V[at(n - 1, n - 1)] = 1;
  e[0] = 0;
}

function tql2(V, d, e, n) {
  const at = (r, c) => r * n + c;
  for (let i = 1; i < n; i++) e[i - 1] = e[i];
  e[n - 1] = 0;
  let f = 0, tst1 = 0;
  const eps = Math.pow(2, -52);
  for (let l = 0; l < n; l++) {
    tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]));
    let m = l;
    while (m < n) { if (Math.abs(e[m]) <= eps * tst1) break; m++; }
    if (m > l) {
      let iter = 0;
      do {
        if (++iter > 200) throw new Error('symmetricEigen: no convergence');
        let g = d[l];
        let p = (d[l + 1] - g) / (2 * e[l]);
        let r = Math.hypot(p, 1);
        if (p < 0) r = -r;
        d[l] = e[l] / (p + r);
        d[l + 1] = e[l] * (p + r);
        const dl1 = d[l + 1];
        let h = g - d[l];
        for (let i = l + 2; i < n; i++) d[i] -= h;
        f += h;
        p = d[m];
        let c = 1, c2 = c, c3 = c;
        const el1 = e[l + 1];
        let s = 0, s2 = 0;
        for (let i = m - 1; i >= l; i--) {
          c3 = c2; c2 = c; s2 = s;
          g = c * e[i];
          h = c * p;
          r = Math.hypot(p, e[i]);
          e[i + 1] = s * r;
          s = e[i] / r;
          c = p / r;
          p = c * d[i] - s * g;
          d[i + 1] = h + s * (c * g + s * d[i]);
          for (let k = 0; k < n; k++) {
            h = V[at(k, i + 1)];
            V[at(k, i + 1)] = s * V[at(k, i)] + c * h;
            V[at(k, i)] = c * V[at(k, i)] - s * h;
          }
        }
        p = (-s * s2 * c3 * el1 * e[l]) / dl1;
        e[l] = s * p;
        d[l] = c * p;
      } while (Math.abs(e[l]) > eps * tst1);
    }
    d[l] += f;
    e[l] = 0;
  }
}

/**
 * EOFs of the data matrix X (T rows = times, N columns = space), row-major.
 * The time mean is removed.  Returns the first `nModes` modes:
 *   values[k]    variance explained by mode k (same units as X squared)
 *   fractions[k] fraction of the total variance
 *   patterns[k]  unit-norm spatial pattern (length N); sign: largest |loading| positive
 *   pcs[k]       principal-component time series (length T), variance = values[k]
 *   total        total variance
 * Uses the smaller of the N x N and T x T covariance matrices.
 */
export function eof(X, T, N, nModes = 4) {
  const mean = new Float64Array(N);
  for (let t = 0; t < T; t++) for (let i = 0; i < N; i++) mean[i] += X[t * N + i];
  for (let i = 0; i < N; i++) mean[i] /= T;
  const A = new Float64Array(T * N);
  for (let t = 0; t < T; t++) for (let i = 0; i < N; i++) A[t * N + i] = X[t * N + i] - mean[i];
  let total = 0;
  for (let k = 0; k < A.length; k++) total += A[k] * A[k];
  total /= T;
  const spaceFirst = N <= T;
  const M = spaceFirst ? N : T;
  const C = new Float64Array(M * M);
  if (spaceFirst) {
    for (let t = 0; t < T; t++) {
      const row = t * N;
      for (let i = 0; i < N; i++) {
        const a = A[row + i];
        if (a === 0) continue;
        for (let j = i; j < N; j++) C[i * N + j] += a * A[row + j];
      }
    }
    for (let i = 0; i < N; i++) for (let j = i; j < N; j++) { C[i * N + j] /= T; C[j * N + i] = C[i * N + j]; }
  } else {
    for (let s = 0; s < T; s++) {
      for (let t = s; t < T; t++) {
        let acc = 0;
        for (let i = 0; i < N; i++) acc += A[s * N + i] * A[t * N + i];
        C[s * T + t] = C[t * T + s] = acc / T;
      }
    }
  }
  const { values, vectors } = symmetricEigen(C, M);
  const nm = Math.min(nModes, M);
  const out = { values: [], fractions: [], patterns: [], pcs: [], total, mean };
  for (let k = 0; k < nm; k++) {
    const lam = Math.max(values[k], 0);
    const pat = new Float64Array(N);
    const pc = new Float64Array(T);
    if (spaceFirst) {
      for (let i = 0; i < N; i++) pat[i] = vectors[i * M + k];
    } else {
      const sc = lam > 0 ? 1 / Math.sqrt(lam * T) : 0;
      for (let t = 0; t < T; t++) { const w = vectors[t * M + k]; for (let i = 0; i < N; i++) pat[i] += A[t * N + i] * w * sc; }
    }
    // sign convention
    let big = 0;
    for (let i = 0; i < N; i++) if (Math.abs(pat[i]) > Math.abs(big)) big = pat[i];
    if (big < 0) for (let i = 0; i < N; i++) pat[i] = -pat[i];
    for (let t = 0; t < T; t++) { let a = 0; for (let i = 0; i < N; i++) a += A[t * N + i] * pat[i]; pc[t] = a; }
    out.values.push(lam);
    out.fractions.push(total > 0 ? lam / total : 0);
    out.patterns.push(pat);
    out.pcs.push(pc);
  }
  return out;
}
