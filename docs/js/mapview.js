// A canvas map of a doubly periodic (ny x nx) field: shading, contour lines, markers, hover readout,
// click-to-pick and a colour bar.  Built for fast animation (one draw call per frame).

import { lut, theme } from './colors.js';
import { el, fmt, niceStep, niceTicks, tickLabel } from './util.js';

const MARGIN = { l: 40, r: 62, t: 24, b: 34 };

// marching squares: edge pairs per case; edges 0 bottom, 1 right, 2 top, 3 left
const CASES = [[], [[3, 0]], [[0, 1]], [[3, 1]], [[1, 2]], 'S5', [[0, 2]], [[3, 2]], [[2, 3]], [[0, 2]], 'S10', [[1, 2]], [[3, 1]], [[0, 1]], [[3, 0]], []];

function contourPath(vals, nx, ny, level, toPx, ctx) {
  const at = (i, j) => vals[((j + ny) % ny) * nx + ((i + nx) % nx)];
  const lerp = (a, b) => (level - a) / (b - a);
  for (let j = -1; j < ny; j++) {
    for (let i = -1; i < nx; i++) {
      const v0 = at(i, j), v1 = at(i + 1, j), v2 = at(i + 1, j + 1), v3 = at(i, j + 1);
      const idx = (v0 > level ? 1 : 0) | (v1 > level ? 2 : 0) | (v2 > level ? 4 : 0) | (v3 > level ? 8 : 0);
      if (idx === 0 || idx === 15) continue;
      let segs = CASES[idx];
      if (segs === 'S5') segs = (v0 + v1 + v2 + v3) / 4 > level ? [[0, 1], [2, 3]] : [[3, 0], [1, 2]];
      else if (segs === 'S10') segs = (v0 + v1 + v2 + v3) / 4 > level ? [[3, 0], [1, 2]] : [[0, 1], [2, 3]];
      const pt = (e) => {
        switch (e) {
          case 0: return [i + lerp(v0, v1), j];
          case 1: return [i + 1, j + lerp(v1, v2)];
          case 2: return [i + lerp(v3, v2), j + 1];
          default: return [i, j + lerp(v0, v3)];
        }
      };
      for (const [e1, e2] of segs) {
        const a = pt(e1), b = pt(e2);
        const [x1, y1] = toPx(a[0], a[1]), [x2, y2] = toPx(b[0], b[1]);
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
      }
    }
  }
}

export class MapView {
  /**
   * @param {HTMLElement} container
   * @param {{grid: object, onPick?: Function, axes?: boolean, readoutDigits?: number}} opts
   */
  constructor(container, opts) {
    this.grid = opts.grid;
    this.onPick = opts.onPick || null;
    this.axes = opts.axes !== false;
    this.container = container;
    container.classList.add('map');
    this.title = el('div', { class: 'map-title' });
    this.readout = el('div', { class: 'map-readout', 'aria-live': 'off' });
    this.canvas = el('canvas', { role: 'img' });
    container.append(this.title, this.readout, this.canvas);
    this.ctx = this.canvas.getContext('2d');
    this.off = document.createElement('canvas');
    this.off.width = this.grid.nx;
    this.off.height = this.grid.ny;
    this.data = null;
    this.markers = [];
    this.pick = null;
    this.lutCache = new Map();
    this._ro = new ResizeObserver(() => this.draw());
    this._ro.observe(container);
    this.canvas.addEventListener('mousemove', (e) => this.#hover(e));
    this.canvas.addEventListener('mouseleave', () => { this.readout.textContent = ''; });
    this.canvas.addEventListener('click', (e) => this.#click(e));
    if (this.onPick) this.canvas.style.cursor = 'crosshair';
  }

  setTitle(t) { this.title.textContent = t; }

  /**
   * data: { values (SI), scale, vmin, vmax (display units), kind, unit, name,
   *         contour?: { values (display units), step } }
   */
  setData(data) { this.data = data; this.draw(); }
  setMarkers(m) { this.markers = m; this.draw(); }
  setPick(p) { this.pick = p; this.draw(); }
  destroy() { this._ro.disconnect(); }

  #layout() {
    const w = Math.max(160, this.container.clientWidth);
    const pw = w - MARGIN.l - MARGIN.r;
    const ph = pw * (this.grid.Ly / this.grid.Lx);
    return { w, h: ph + MARGIN.t + MARGIN.b, pw, ph };
  }

  #lutFor(kind, dark) {
    const k = kind + dark;
    if (!this.lutCache.has(k)) this.lutCache.set(k, lut(kind, dark));
    return this.lutCache.get(k);
  }

  draw() {
    if (!this.data || !this.container.clientWidth) return;
    const th = theme();
    const { w, h, pw, ph } = this.#layout();
    const dpr = window.devicePixelRatio || 1;
    const c = this.canvas;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
      c.style.width = w + 'px'; c.style.height = h + 'px';
    }
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const { nx, ny } = this.grid;
    const d = this.data;
    const L = this.#lutFor(d.kind, th.dark);
    const img = this.off.getContext('2d').createImageData(nx, ny);
    const s = d.scale ?? 1;
    const span = d.vmax - d.vmin || 1;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const v = d.values[j * nx + i] * s;
        const f = Math.min(1, Math.max(0, (v - d.vmin) / span));
        const k = Math.round(f * 255) * 3;
        const o = ((ny - 1 - j) * nx + i) * 4;      // y = 0 at the bottom
        img.data[o] = L[k]; img.data[o + 1] = L[k + 1]; img.data[o + 2] = L[k + 2]; img.data[o + 3] = 255;
      }
    }
    this.off.getContext('2d').putImageData(img, 0, 0);
    const x0 = MARGIN.l, y0 = MARGIN.t;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(this.off, x0, y0, pw, ph);

    // contours
    if (d.contour) {
      ctx.save();
      ctx.beginPath(); ctx.rect(x0, y0, pw, ph); ctx.clip();
      const toPx = (gi, gj) => [x0 + ((gi + 0.5) / nx) * pw, y0 + ph - ((gj + 0.5) / ny) * ph];
      const cv = d.contour.values;
      let lo = Infinity, hi = -Infinity;
      for (let p = 0; p < cv.length; p++) { if (cv[p] < lo) lo = cv[p]; if (cv[p] > hi) hi = cv[p]; }
      const step = d.contour.step || niceStep(hi - lo, 12);
      ctx.lineWidth = 1;
      ctx.strokeStyle = th.dark ? 'rgba(235,240,245,0.78)' : 'rgba(18,28,40,0.72)';
      for (let pass = 0; pass < 2; pass++) {
        ctx.beginPath();
        for (let lev = Math.ceil(lo / step) * step; lev <= hi; lev += step) {
          if ((lev < 0) !== (pass === 1) && !(lev === 0 && pass === 0)) continue;
          contourPath(cv, nx, ny, lev + 1e-9 * step, toPx, ctx);
        }
        ctx.setLineDash(pass === 1 ? [4, 3] : []);       // dashed = below zero
        ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.restore();
    }

    // frame + axes
    ctx.strokeStyle = th.axis; ctx.lineWidth = 1;
    ctx.strokeRect(x0 + 0.5, y0 + 0.5, pw - 1, ph - 1);
    ctx.fillStyle = th.muted;
    ctx.font = `11px ${th.fontSans}`;
    if (this.axes) {
      const Lkm = this.grid.Lx / 1e3, Lykm = this.grid.Ly / 1e3;
      const step = niceStep(Lkm, 4);
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const v of niceTicks(0, Lkm, 4)) {
        const x = x0 + (v / Lkm) * pw;
        ctx.fillText(tickLabel(v / 1000, step / 1000), x, y0 + ph + 5);
        ctx.fillRect(x, y0 + ph, 1, 3);
      }
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of niceTicks(0, Lykm, 4)) {
        const y = y0 + ph - (v / Lykm) * ph;
        ctx.fillText(tickLabel(v / 1000, step / 1000), x0 - 7, y);
        ctx.fillRect(x0 - 3, y, 3, 1);
      }
      ctx.textAlign = 'right'; ctx.textBaseline = 'alphabetic';
      ctx.fillText('10³ km', x0 + pw, y0 + ph + 29);
    }

    // markers and picked point
    const toPxKm = (xm, ym) => [x0 + (xm / this.grid.Lx) * pw, y0 + ph - (ym / this.grid.Ly) * ph];
    for (const m of this.markers) {
      const [x, y] = toPxKm(m.x, m.y);
      ctx.fillStyle = m.fill || th.ink; ctx.strokeStyle = m.color || th.surface; ctx.lineWidth = m.lw || 1.2;
      const r = m.r || 2.5;
      ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); if (m.fill !== null) ctx.fill(); ctx.stroke();
    }
    if (this.pick) {
      const [x, y] = toPxKm((this.pick.i + 0.5) * this.grid.dx, (this.pick.j + 0.5) * this.grid.dy);
      ctx.strokeStyle = th.surface; ctx.lineWidth = 3.5;
      this.#cross(ctx, x, y, 6);
      ctx.strokeStyle = th.ink; ctx.lineWidth = 1.6;
      this.#cross(ctx, x, y, 6);
    }

    // colour bar
    const bx = x0 + pw + 14, bw = 10;
    const g = ctx.createLinearGradient(0, y0 + ph, 0, y0);
    for (let k = 0; k <= 16; k++) {
      const f = k / 16, i = Math.round(f * 255) * 3;
      g.addColorStop(f, `rgb(${L[i]},${L[i + 1]},${L[i + 2]})`);
    }
    ctx.fillStyle = g; ctx.fillRect(bx, y0, bw, ph);
    ctx.strokeStyle = th.axis; ctx.strokeRect(bx + 0.5, y0 + 0.5, bw - 1, ph - 1);
    ctx.fillStyle = th.ink2; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.font = `11px ${th.fontSans}`;
    const bstep = niceStep(span, 5);
    for (const v of niceTicks(d.vmin, d.vmax, 5)) {
      const y = y0 + ph - ((v - d.vmin) / span) * ph;
      ctx.fillRect(bx + bw, y, 3, 1);
      ctx.fillText(tickLabel(v, bstep), bx + bw + 6, y);
    }
    if (d.unit) {
      ctx.fillStyle = th.muted; ctx.textBaseline = 'alphabetic';
      ctx.fillText(d.unit, bx - 2, y0 - 8);
    }
    this._geom = { x0, y0, pw, ph };
  }

  #cross(ctx, x, y, r) {
    ctx.beginPath();
    ctx.moveTo(x - r, y); ctx.lineTo(x + r, y);
    ctx.moveTo(x, y - r); ctx.lineTo(x, y + r);
    ctx.stroke();
  }

  #cell(e) {
    if (!this._geom) return null;
    const r = this.canvas.getBoundingClientRect();
    const { x0, y0, pw, ph } = this._geom;
    const fx = (e.clientX - r.left - x0) / pw, fy = 1 - (e.clientY - r.top - y0) / ph;
    if (fx < 0 || fx >= 1 || fy < 0 || fy >= 1) return null;
    return { i: Math.floor(fx * this.grid.nx), j: Math.floor(fy * this.grid.ny) };
  }

  #hover(e) {
    const c = this.#cell(e);
    if (!c || !this.data) { this.readout.textContent = ''; return; }
    const v = this.data.values[c.j * this.grid.nx + c.i] * (this.data.scale ?? 1);
    const xk = Math.round(((c.i + 0.5) * this.grid.dx) / 1e3), yk = Math.round(((c.j + 0.5) * this.grid.dy) / 1e3);
    this.readout.textContent = `${fmt(v, 3)} ${this.data.unit || ''}  ·  x ${xk}, y ${yk} km`;
  }

  #click(e) {
    if (!this.onPick) return;
    const c = this.#cell(e);
    if (c) this.onPick(c.i, c.j);
  }
}
