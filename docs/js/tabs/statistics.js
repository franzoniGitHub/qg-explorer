// Statistics: distribution, tails, persistence and frequency spectrum of any derived field.

import { el, fmt } from '../util.js';
import { VARIABLES, LAYER_LABELS } from '../physics.js';
import { moments, minMax, histogram, normalPdf, normalQuantile, quantileSorted, autocorrelation, efoldingLag } from '../stats.js';
import { welch } from '../spectra.js';
import { theme } from '../colors.js';
import { chart, layout, lineTrace, layoutRail, group, selectCtl, checkCtl, busy, stats, stat, cols, panel } from '../ui.js';

const LAYERS = ['upper', 'lower', 'barotropic', 'baroclinic'].map((value) => ({ value, label: LAYER_LABELS[value] }));

export default function create({ ds }, root) {
  const { nx, ny, N } = ds.grid;
  const nt = ds.nt;

  const varCtl = selectCtl({ label: 'Variable', options: Object.entries(VARIABLES).filter(([k]) => k !== 'psi').map(([value, v]) => ({ value, label: v.label })), value: 'zeta', onchange: () => update() });
  const layerCtl = selectCtl({ label: 'Layer', options: LAYERS, value: 'upper', onchange: () => update() });
  const stdCtl = checkCtl({ label: 'Standardise (subtract mean, divide by σ)', checked: true, onchange: () => draw() });
  const logCtl = checkCtl({ label: 'Logarithmic probability axis', checked: true, onchange: () => draw() });
  const rail = [
    group(null, varCtl.el, layerCtl.el),
    group('Histogram', stdCtl.el, logCtl.el),
    el('p', { class: 'hint' }, 'All grid points and all times of the control run are pooled for the distribution. Persistence and spectrum are averaged over 64 grid points spread evenly over the domain.'),
  ];

  const summary = el('div');
  const hist = chart({ title: 'Probability density', sub: '', height: 300 });
  const qq = chart({ title: 'Quantile–quantile plot against a Gaussian', sub: 'Points on the line: Gaussian. Curling up at the ends: heavy tails.', height: 300 });
  const acfC = chart({ title: 'Autocorrelation in time', sub: '', height: 280 });
  const psdC = chart({ title: 'Frequency spectrum', sub: 'Power spectral density, Welch estimate', height: 280 });
  root.append(layoutRail(rail, [panel('Summary', '', summary), cols(hist.el, qq.el), cols(acfC.el, psdC.el)]));

  let cur = null;
  let token = 0;

  async function update() {
    const mine = ++token;
    const v = varCtl.value, l = layerCtl.value;
    const data = ds.cachedSeries(v, l) || await busy('Computing the field for every time…', (prog) => ds.fieldSeries(v, l, {}, prog));
    if (mine !== token) return;
    const info = VARIABLES[v];
    const mom = moments(data);
    const stride = Math.max(1, Math.floor(data.length / 400000));
    const sub = new Float32Array(Math.ceil(data.length / stride));
    for (let i = 0, k = 0; i < data.length; i += stride) sub[k++] = data[i];
    sub.sort();
    const [lo, hi] = minMax(data);
    // points for time statistics
    const pts = [];
    for (let j = 4; j < ny; j += 8) for (let i = 4; i < nx; i += 8) pts.push(j * nx + i);
    const maxLag = Math.min(nt - 2, Math.round(30 / ds.dtDays));
    const acf = new Float64Array(maxLag + 1);
    let seg = 1; while (seg * 2 <= Math.min(256, nt / 2)) seg *= 2;
    let psd = null, freq = null;
    const s = new Float64Array(nt);
    for (const p of pts) {
      for (let t = 0; t < nt; t++) s[t] = data[t * N + p];
      const a = autocorrelation(s, maxLag);
      for (let k = 0; k <= maxLag; k++) acf[k] += a[k] / pts.length;
      if (seg >= 8) {
        const w = welch(s, seg, ds.dtDays);
        if (!psd) { psd = new Float64Array(w.psd.length); freq = w.freq; }
        for (let k = 0; k < psd.length; k++) psd[k] += w.psd[k] / pts.length;
      }
    }
    cur = { v, l, info, mom, sub, lo, hi, acf, psd, freq, data };
    draw();
  }

  function draw() {
    if (!cur) return;
    const th = theme();
    const { info, mom, sub, lo, hi, acf, psd, freq, data } = cur;
    const sc = info.scale, unit = info.unit;
    const std = stdCtl.value;
    const shift = std ? mom.mean : 0, div = std ? mom.std : 1;
    const f = (v) => ((v - shift) / div) * (std ? 1 : sc);
    const xUnit = std ? 'σ' : unit;

    // histogram
    let a = f(lo), b = f(hi);
    const zlim = std ? 6 : null;
    if (zlim) { a = Math.max(a, -zlim); b = Math.min(b, zlim); }
    const nb = 70;
    const raw = histogram(data, a * (std ? div : 1 / sc) + (std ? shift : 0), b * (std ? div : 1 / sc) + (std ? shift : 0), nb);
    const centers = Array.from(raw.centers, f);
    const wid = (b - a) / nb;
    const pdf = Array.from(raw.counts, (c) => c / (raw.total * wid));
    const gx = Array.from({ length: 200 }, (_, i) => a + ((b - a) * i) / 199);
    const mu = f(mom.mean), sd = std ? 1 : mom.std * sc;
    const gy = gx.map((x) => normalPdf(x, mu, sd));
    const floor = Math.min(...pdf.filter((p) => p > 0)) / 3;
    hist.setSub(`${info.short} in ${xUnit}`);
    hist.set([
      { type: 'bar', name: 'Model', x: centers, y: pdf.map((p) => (p > 0 ? p : null)), width: wid, marker: { color: th.upper, line: { color: th.upper, width: 0.5 } }, hovertemplate: '%{x:.3g} ' + xUnit + '<br>density %{y:.3g}<extra></extra>' },
      lineTrace('Gaussian, same mean and σ', gx, gy, th.fourth, { line: { width: 2 } }),
    ], layout(th, { x: { title: `${info.short} (${xUnit})` }, y: { title: 'Probability density', type: logCtl.value ? 'log' : 'linear', dtick: logCtl.value ? 1 : undefined, range: logCtl.value ? [Math.log10(floor), Math.log10(Math.max(...pdf, ...gy) * 1.6)] : undefined },
      bargap: 0 }), { xTitle: xUnit });

    // QQ
    const tails = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3];
    const ps = [...tails];
    for (let p = 0.01; p < 0.995; p += 0.01) ps.push(p);
    for (const t of tails.slice().reverse()) ps.push(1 - t);
    const xq = [], yq = [];
    for (const p of ps) {
      const q = Math.min(Math.max(p, 1 / sub.length), 1 - 1 / sub.length);
      xq.push(normalQuantile(q));
      yq.push((quantileSorted(sub, q) - mom.mean) / mom.std);
    }
    const ext = Math.max(...xq.map(Math.abs));
    qq.set([
      lineTrace('Identity', [-ext, ext], [-ext, ext], th.muted, { line: { dash: 'dot', width: 1.2 }, hoverinfo: 'skip', meta: { table: false } }),
      { type: 'scatter', mode: 'markers', name: 'Model quantiles', x: xq, y: yq, marker: { size: 5, color: th.upper }, hovertemplate: 'Gaussian %{x:.2f} σ<br>Model %{y:.2f} σ<extra></extra>' },
    ], layout(th, { x: { title: 'Gaussian quantile (σ)' }, y: { title: 'Model quantile (σ)', scaleanchor: 'x' }, hover: 'closest' }), { xTitle: 'Gaussian quantile' });

    // ACF
    const lags = Array.from(acf, (_, k) => k * ds.dtDays);
    const ef = efoldingLag(acf) * ds.dtDays;
    acfC.setSub(Number.isFinite(ef) ? `1/e time ${fmt(ef, 3)} days` : 'no 1/e crossing within 30 days');
    acfC.set([lineTrace('Autocorrelation', lags, Array.from(acf), th.upper), ...(Number.isFinite(ef) ? [] : [])],
      layout(th, { x: { title: 'Lag (days)' }, y: { title: 'Correlation', range: [Math.min(-0.5, Math.min(...acf) - 0.05), 1.02] }, legend: false,
        shapes: [{ type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: Math.exp(-1), y1: Math.exp(-1), line: { color: th.muted, width: 1, dash: 'dot' } }] }), { xTitle: 'lag (days)' });

    // PSD
    if (psd) {
      const fr = Array.from(freq).slice(1), pw = Array.from(psd).slice(1);
      let kmax = 0; for (let k = 1; k < pw.length; k++) if (pw[k] > pw[kmax]) kmax = k;
      psdC.setSub(`maximum at a period of ${fmt(1 / fr[kmax], 3)} days`);
      psdC.set([lineTrace('Power spectral density', fr, pw, th.upper)],
        layout(th, { x: { title: 'Frequency (cycles per day)', type: 'log' }, y: { title: `PSD (${xUnitPow(unit)} per cycle/day)`, type: 'log', exponentformat: 'power', dtick: 1 }, legend: false }), { xTitle: 'frequency' });
    } else { psdC.set([], layout(th, { legend: false }), { table: false }); psdC.setSub('The record is too short for a spectrum.'); }

    // summary
    const q = (p) => quantileSorted(sub, p) * sc;
    summary.replaceChildren(stats(
      stat('Mean', `${fmt(Math.abs(mom.mean) < 1e-6 * mom.std ? 0 : mom.mean * sc, 3)} ${unit}`),
      stat('Standard deviation', `${fmt(mom.std * sc, 3)} ${unit}`),
      stat('Skewness', fmt(mom.skew, 3)),
      stat('Kurtosis (Gaussian: 3)', fmt(mom.kurt, 3)),
      stat('1 % / 99 % quantiles', `${fmt(q(0.01), 3)} / ${fmt(q(0.99), 3)}`),
      stat('Extremes', `${fmt(cur.lo * sc, 3)} / ${fmt(cur.hi * sc, 3)}`),
      stat('Samples', mom.n.toLocaleString('en')),
    ));
  }
  const xUnitPow = (u) => `(${u})²`;

  return {
    async activate() { if (!cur) await update(); else [hist, qq, acfC, psdC].forEach((c) => c.resize()); },
    refresh() { draw(); },
    destroy() { [hist, qq, acfC, psdC].forEach((c) => c.purge()); },
  };
}
