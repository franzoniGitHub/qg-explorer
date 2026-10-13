// Waves and stability: Hovmoeller diagram, wavenumber-frequency spectrum and an interactive
// normal-mode (Phillips) stability analysis of the two-layer basic state.

import { el, fmt, linspace, niceStep } from '../util.js';
import { VARIABLES } from '../physics.js';
import { kOmega } from '../spectra.js';
import { linearParams, growthCurve, growthMap, fastestMode, mode } from '../linear.js';
import { theme, plotlyScale } from '../colors.js';
import { symmetricLimit } from '../stats.js';
import { chart, layout, lineTrace, layoutRail, group, selectCtl, sliderCtl, busy, stats, stat, cols, panel, prose } from '../ui.js';

const DAY = 86400;

export default function create({ ds }, root) {
  const { nx, ny, N } = ds.grid;
  const base = ds.manifest.model;
  const dk = (2 * Math.PI) / base.Lx;
  const mMax = nx / 2, mShow = Math.min(16, nx / 2);

  // ---- controls ---------------------------------------------------------------------------------------
  const layerCtl = selectCtl({ label: 'Layer', options: [{ value: 'upper', label: 'Upper layer' }, { value: 'lower', label: 'Lower layer' }], value: 'upper', onchange: () => drawObserved() });
  const varCtl = selectCtl({ label: 'Variable', options: ['v', 'Z', 'zeta', 'u'].map((v) => ({ value: v, label: VARIABLES[v].label })), value: 'v', onchange: () => drawObserved() });
  const latCtl = sliderCtl({ label: 'Latitude of the section', min: 0, max: ny - 1, step: 1, value: Math.floor(ny / 2), format: (j) => `y = ${Math.round(((j + 0.5) * ds.grid.dy) / 1e3)} km`, oninput: () => drawObserved() });

  const ex = { ratio: base.dU / (base.beta * base.Ld * base.Ld / base.delta), Ld: base.Ld / 1e3, drag: Math.log10(1 / (base.r_ek * DAY)), n: 0, m: 0 };
  const dcrit0 = base.beta * base.Ld * base.Ld / base.delta;
  const ratioCtl = sliderCtl({ label: 'Shear / critical shear', min: 0.4, max: 3.5, step: 0.05, value: Math.round(ex.ratio * 20) / 20, format: (v) => fmt(v, 3), oninput: (v) => { ex.ratio = v; drawLinear(); } });
  const LdCtl = sliderCtl({ label: 'Deformation radius', min: 250, max: 1000, step: 10, value: Math.round(ex.Ld / 10) * 10, format: (v) => `${v} km`, oninput: (v) => { ex.Ld = v; drawLinear(); } });
  const dragCtl = sliderCtl({ label: 'Ekman drag time 1/r', min: -0.5, max: 2, step: 0.05, value: ex.drag, format: (v) => `${fmt(10 ** v, 3)} days`, oninput: (v) => { ex.drag = v; drawLinear(); } });
  const nCtl = sliderCtl({ label: 'Meridional wavenumber n', min: 0, max: 8, step: 1, value: 0, format: (v) => String(v), oninput: (v) => { ex.n = v; drawLinear(); } });
  const mCtl = sliderCtl({ label: 'Mode shown: zonal wavenumber m', min: 1, max: 24, step: 1, value: 4, format: (v) => String(v), oninput: (v) => { ex.m = v; drawLinear(); } });
  const reset = el('button', { class: 'btn', type: 'button' }, 'Reset to the dataset');
  reset.addEventListener('click', () => {
    ex.ratio = base.dU / dcrit0; ratioCtl.value = Math.round(ex.ratio * 20) / 20; ex.ratio = ratioCtl.value;
    ex.Ld = base.Ld / 1e3; LdCtl.value = Math.round(ex.Ld / 10) * 10; ex.Ld = LdCtl.value;
    ex.drag = Math.log10(1 / (base.r_ek * DAY)); dragCtl.value = ex.drag;
    drawLinear();
  });
  ex.ratio = ratioCtl.value; ex.Ld = LdCtl.value; ex.drag = dragCtl.value; ex.n = 0; ex.m = mCtl.value;

  const rail = [
    group('Observed waves', varCtl.el, layerCtl.el, latCtl.el),
    group('Linear theory explorer', ratioCtl.el, LdCtl.el, dragCtl.el, nCtl.el, mCtl.el, reset),
    el('p', { class: 'hint' }, 'The explorer changes only the linear calculation. The dashed line in the growth-rate plot is the dataset’s own basic state.'),
  ];

  // ---- charts ---------------------------------------------------------------------------------------------
  const hov = chart({ title: 'Hovmöller diagram', sub: '', height: 420, tall: true });
  const kw = chart({ title: 'Wavenumber–frequency spectrum', sub: 'log₁₀ of power relative to its maximum; diamonds mark the linear-theory phase speeds', height: 420, tall: true });
  const gcurve = chart({ title: 'Growth rate against zonal wavenumber', sub: '', height: 280 });
  const gmap = chart({ title: 'Growth rate over (m, n)', sub: 'day⁻¹, unstable region only', height: 280 });
  const gmode = chart({ title: 'Structure of the selected mode', sub: '', height: 280 });
  const readout = el('div');

  root.append(layoutRail(rail, [
    cols(hov.el, kw.el),
    panel('Linear stability', 'Normal modes of the two-layer flow with the explorer’s parameters', readout),
    cols(gcurve.el, gmap.el, gmode.el),
    panel(null, null, prose(`<p>Reading the diagrams. Eastward-travelling features slope up and to the right in the Hovmöller diagram. In the spectrum they sit at positive m. Growing baroclinic waves move at roughly the speed of the flow at the steering level, between the two layer winds, so they appear on the line ω = c·k with c between U₂ and U₁. Increase the shear in the explorer past the critical value β/F₂ and watch the instability appear at long wavelengths first.</p>`)),
  ]));

  // ---- observed ---------------------------------------------------------------------------------------------
  let row = null;       // nt x nx values at the chosen latitude
  let seqRow = 0;
  async function drawObserved() {
    const mine = ++seqRow;
    const info = VARIABLES[varCtl.value], th = theme();
    const data = ds.cachedSeries(varCtl.value, layerCtl.value)
      || await busy('Computing the section…', (prog) => ds.fieldSeries(varCtl.value, layerCtl.value, {}, prog));
    if (mine !== seqRow) return;
    const j = latCtl.value;
    const rowv = new Float64Array(ds.nt * nx);
    for (let t = 0; t < ds.nt; t++) for (let i = 0; i < nx; i++) rowv[t * nx + i] = data[t * N + j * nx + i];
    row = rowv;
    const days = Array.from({ length: ds.nt }, (_, t) => ds.timeDays(t));
    const xs = Array.from(ds.grid.x, (v) => v / 1e3);
    const z = Array.from({ length: ds.nt }, (_, t) => Array.from(rowv.subarray(t * nx, (t + 1) * nx), (v) => v * info.scale));
    const lim = symmetricLimit(rowv, 0.995) * info.scale;
    hov.setSub(`${info.short} (${info.unit}) at y = ${Math.round(((j + 0.5) * ds.grid.dy) / 1e3)} km, ${layerCtl.value} layer`);
    hov.set([{ type: 'heatmap', x: xs, y: days, z, zmin: -lim, zmax: lim, zmid: 0, colorscale: plotlyScale('diverging', th.dark), hovertemplate: `x %{x:.0f} km<br>day %{y:.1f}<br>%{z:.3g} ${info.unit}<extra></extra>`,
      colorbar: { thickness: 10, len: 0.92, tickfont: { size: 11, color: th.muted }, outlinewidth: 0 } }],
      layout(th, { x: { title: 'x (km)' }, y: { title: 'Time (days)' }, legend: false, hover: 'closest', margin: { r: 6 } }), { table: false });

    // k-omega
    const dtDays = ds.dtDays;
    let seg = 1; while (seg * 2 <= Math.min(256, ds.nt / 2)) seg *= 2;
    if (seg < 16) { kw.set([], layout(th, { legend: false }), { table: false }); kw.setSub('The record is too short for a spectrum.'); return; }
    const ko = kOmega(rowv, ds.nt, nx, seg, dtDays);
    const mlim = Math.min(16, nx / 2 - 1), fmaxIdx = ko.freq.findIndex((f) => f > 0.5);
    const nf = fmaxIdx < 0 ? ko.nFreq : fmaxIdx;
    const mIdx = [], mVals = [];
    for (let i = 0; i < nx; i++) if (Math.abs(ko.m[i]) <= mlim) { mIdx.push(i); mVals.push(ko.m[i]); }
    let mx = -Infinity;
    for (let k = 1; k < nf; k++) for (const i of mIdx) mx = Math.max(mx, ko.power[k * nx + i]);
    const zz = [];
    for (let k = 0; k < nf; k++) zz.push(mIdx.map((i) => Math.log10(Math.max(ko.power[k * nx + i], mx * 1e-9) / mx)));
    const lp = linearParams(ds.manifest.model);
    const lm = [], lf = [], lt = [];
    for (let m = 1; m <= mlim; m++) {
      const md = mode(lp, m * dk, 0);
      if (md.sigma > 0) { lm.push(m); lf.push((md.c * m * DAY) / base.Lx); lt.push(`m=${m}: c=${fmt(md.c, 3)} m/s, e-folding ${fmt(1 / (md.sigma * DAY), 3)} d`); }
    }
    kw.set([
      { type: 'heatmap', x: mVals, y: Array.from(ko.freq.subarray(0, nf)), z: zz, zmin: -3.5, zmax: 0, colorscale: plotlyScale('sequential', th.dark), hovertemplate: 'm %{x}<br>%{y:.3f} cycles/day<br>power 10^%{z:.2f}<extra></extra>',
        colorbar: { thickness: 10, len: 0.92, tickfont: { size: 11, color: th.muted }, outlinewidth: 0 } },
      { type: 'scatter', mode: 'markers', name: 'Linear theory', x: lm, y: lf, text: lt, hoverinfo: 'text',
        marker: { symbol: 'diamond', size: 9, color: th.fourth, line: { color: th.surface, width: 1.5 } } },
    ], layout(th, { x: { title: 'Zonal wavenumber m (eastward > 0)' }, y: { title: 'Frequency (cycles per day)', range: [0, 0.5] }, legend: true, hover: 'closest', margin: { r: 6 } }), { table: false });
  }

  // ---- linear theory -----------------------------------------------------------------------------------------
  function explorerModel() {
    const Ld = ex.Ld * 1e3;
    const dcrit = base.beta * Ld * Ld / base.delta;
    const U2 = base.U2;
    return { ...base, Ld, U1: U2 + ex.ratio * dcrit, U2, r_ek: 1 / (10 ** ex.drag * DAY) };
  }

  function drawLinear() {
    const th = theme();
    const p = linearParams(explorerModel());
    const p0 = linearParams(base);
    const ky = ex.n * dk;
    const ms = linspace(0.05, mShow, 320);
    const kxs = Array.from(ms, (m) => m * dk);
    const sig = growthCurve(p, kxs, ky), sig0 = growthCurve(p0, kxs, ky);
    const toDay = (a) => Array.from(a, (s) => s * DAY);
    gcurve.set([
      lineTrace('Dataset', Array.from(ms), toDay(sig0), th.muted, { line: { dash: 'dash', width: 1.5 } }),
      lineTrace('Explorer', Array.from(ms), toDay(sig), th.accent, { line: { width: 2.2 } }),
    ], layout(th, { x: { title: 'Zonal wavenumber m' }, y: { title: 'Growth rate (day⁻¹)', zeroline: true } }), { xTitle: 'm' });
    gcurve.setSub(`n = ${ex.n}`);

    const mk = Array.from({ length: mShow }, (_, i) => i + 1), nk = Array.from({ length: Math.min(16, nx / 2) + 1 }, (_, j) => j);
    const g = growthMap(p, mk.map((m) => m * dk), nk.map((n) => n * dk));
    const z = nk.map((n, j) => mk.map((m, i) => Math.max(0, g[j * mk.length + i] * DAY)));
    gmap.set([{ type: 'heatmap', x: mk, y: nk, z, zmin: 0, colorscale: plotlyScale('sequential', th.dark), hovertemplate: 'm %{x}, n %{y}<br>%{z:.3f} day⁻¹<extra></extra>',
      colorbar: { thickness: 10, tickfont: { size: 11, color: th.muted }, outlinewidth: 0 } }],
      layout(th, { x: { title: 'Zonal wavenumber m' }, y: { title: 'Meridional wavenumber n' }, legend: false, hover: 'closest', margin: { r: 6 } }), { table: false });

    const best = fastestMode(p, mMax * dk, 4000);
    const md = mode(p, ex.m * dk, ex.n * dk);
    const lam = (2 * Math.PI) / (ex.m * dk);
    const unstable = md.sigma > 0;
    readout.replaceChildren(stats(
      stat('Critical shear β/F₂', `${fmt(p.beta / p.F2, 3)} m s⁻¹`),
      stat('Shear', `${fmt(p.dU, 3)} m s⁻¹`),
      stat('Fastest zonal mode (n = 0)', best.sigma > 0 ? `m = ${fmt(best.k / dk, 3)}, ${fmt(1 / (best.sigma * DAY), 3)} d e-folding` : 'stable'),
      stat(`Selected mode m = ${ex.m}, n = ${ex.n}`, unstable ? `${fmt(1 / (md.sigma * DAY), 3)} d e-folding` : 'stable or neutral'),
      stat('Phase speed', `${fmt(md.c, 3)} m s⁻¹`),
      stat('Lower / upper amplitude', fmt(md.ratio, 3)),
    ));

    const xx = linspace(0, 1.5 * lam, 160);
    const kx = ex.m * dk;
    const u1 = Array.from(xx, (x) => Math.cos(kx * x));
    const u2 = Array.from(xx, (x) => md.ratio * Math.cos(kx * x + md.lag));
    gmode.setSub(`${fmt(lam / 1e3, 4)} km wavelength, phase lag ${fmt((md.lag * 180) / Math.PI, 3)}°`);
    gmode.set([
      lineTrace('Upper layer ψ₁', Array.from(xx, (x) => x / 1e3), u1, th.upper),
      lineTrace('Lower layer ψ₂', Array.from(xx, (x) => x / 1e3), u2, th.lower),
    ], layout(th, { x: { title: 'x (km)' }, y: { title: 'ψ (relative to ψ₁ amplitude)' } }), { xTitle: 'x (km)' });
  }

  drawLinear();

  return {
    async activate() {
      if (!row) await drawObserved(); else { hov.resize(); kw.resize(); gcurve.resize(); gmap.resize(); gmode.resize(); }
    },
    refresh() { drawObserved(); drawLinear(); },
    destroy() { [hov, kw, gcurve, gmap, gmode].forEach((c) => c.purge()); },
  };
}
