// Radix-2 fast Fourier transforms (1-D and 2-D, complex, in place).
// No dependencies, no DOM: usable from the browser and from the Node tests.
//
// Convention (same as numpy.fft):  forward  X_k = sum_n x_n exp(-2 pi i n k / N)
//                                  inverse  x_n = (1/N) sum_k X_k exp(+2 pi i n k / N)
// The inverse here is *unscaled*; use scale() or the 2-D helpers below.

export class FFT {
  constructor(n) {
    if (n < 2 || (n & (n - 1)) !== 0) throw new Error(`FFT length must be a power of two, got ${n}`);
    this.n = n;
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      const a = (2 * Math.PI * i) / n;
      this.cos[i] = Math.cos(a);
      this.sin[i] = Math.sin(a);
    }
    this.rev = new Uint32Array(n);
    const bits = Math.log2(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
      this.rev[i] = r;
    }
  }

  /** In-place transform of re/im (length n). `inverse` flips the sign of the exponent (unscaled). */
  transform(re, im, inverse = false) {
    const { n, cos, sin, rev } = this;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    const sg = inverse ? 1 : -1;
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1;
      const step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = i, k = 0; j < i + half; j++, k += step) {
          const l = j + half;
          const c = cos[k];
          const s = sg * sin[k];
          const tre = re[l] * c - im[l] * s;
          const tim = re[l] * s + im[l] * c;
          re[l] = re[j] - tre;
          im[l] = im[j] - tim;
          re[j] += tre;
          im[j] += tim;
        }
      }
    }
  }
}

/** 2-D complex FFT on row-major (ny rows, nx columns) arrays. */
export class FFT2 {
  constructor(ny, nx) {
    this.ny = ny;
    this.nx = nx;
    this.fx = new FFT(nx);
    this.fy = ny === nx ? this.fx : new FFT(ny);
    this.cr = new Float64Array(ny);
    this.ci = new Float64Array(ny);
  }

  /** In place. The inverse is fully scaled by 1/(nx*ny), so inverse(forward(f)) == f. */
  transform(re, im, inverse = false) {
    const { ny, nx, fx, fy, cr, ci } = this;
    for (let j = 0; j < ny; j++) {
      const o = j * nx;
      fx.transform(re.subarray(o, o + nx), im.subarray(o, o + nx), inverse);
    }
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) { cr[j] = re[j * nx + i]; ci[j] = im[j * nx + i]; }
      fy.transform(cr, ci, inverse);
      for (let j = 0; j < ny; j++) { re[j * nx + i] = cr[j]; im[j * nx + i] = ci[j]; }
    }
    if (inverse) {
      const s = 1 / (nx * ny);
      for (let k = 0; k < re.length; k++) { re[k] *= s; im[k] *= s; }
    }
  }

  /** Forward transform of a real field; returns new {re, im} arrays. */
  forwardReal(f) {
    const re = Float64Array.from(f);
    const im = new Float64Array(f.length);
    this.transform(re, im, false);
    return { re, im };
  }

  /** Real part of the inverse transform of (re, im); inputs are not modified. */
  inverseReal(re, im, out = null) {
    const r = Float64Array.from(re);
    const i = Float64Array.from(im);
    this.transform(r, i, true);
    if (out) { out.set(r); return out; }
    return r;
  }
}
