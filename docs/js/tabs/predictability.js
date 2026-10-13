// Predictability: twin-experiment ensemble forecasts. Error growth, spread, doubling times,
// the upscale cascade of error and forecast "postage stamps".

import { el, fmt, niceStep, clamp } from '../util.js';
import { MapView } from '../mapview.js';
import { linearParams, fastestMode } from '../linear.js';
import { linearFit, symmetricLimit, mean as avg } from '../stats.js';
import { theme, sequentialColor, hex } from '../colors.js';
import { chart, layout, lineTrace, layoutRail, group, selectCtl, sliderCtl, checkCtl, busy, stats, stat, cols3, cols, panel, dataTable } from '../ui.js';

const DAY = 86400;

export default function create({ ds }, root) {
  const m = ds.model, N = ds.grid.N;
  const E = ds.manifest.ensemble;
  const zf = m.zFactor;
  let ens = null, F = null;      // ensemble.json and decoded fields
  let maps = [];

  const metricCtl = selectCtl({ label: 'Error measure', options: [
    { value: 'member', label: 'Member error (rms of Z′ against the truth)' },
    { value: 'mean', label: 'Error of the ensemble mean' },
    { value: 'spread', label: 'Ensemble spread' },
    { value: 'dte', label: 'Difference total energy of the members' }], value: 'member', onchange: () => draw() });
  const layerCtl = selectCtl({ label: 'Layer', options: [{ value: '1', label: 'Upper layer' }, { value: '2', label: 'Lower layer' }], value: '1', onchange: () => { draw(); drawMaps(); } });
  const groupCtl = selectCtl({ label: 'Initial-error group for detail', options: [], onchange: () => draw() });
  const logCtl = checkCtl({ label: 'Logarithmic error axis', checked: true, onchange: () => draw() });
  const fit0 = sliderCtl({ label: 'Doubling-time fit starts', min: 0, max: E.lead_days, step: 0.5, value: 1, format: (v) => `day ${v}`, oninput: () => draw() });
  const fit1 = sliderCtl({ label: 'Doubling-time fit ends', min: 0, max: E.lead_days, step: 0.5, value: Math.min(5, E.lead_days), format: (v) => `day ${v}`, oninput: () => draw() });
  const leadCtl = sliderCtl({ label: 'Lead time for the maps', min: 0, max: E.lead_days, step: E.field_step_hours / 24, value: Math.min(5, E.lead_days), format: (v) => `day ${v}`, oninput: () => drawMaps() });
  const rail = [
    group(null, metricCtl.el, layerCtl.el, groupCtl.el, logCtl.el),
    group('Doubling-time window', fit0.el, fit1.el),
    group('Postage stamps', leadCtl.el),
    el('p', { class: 'hint' }, `${E.members} members per group, started from the control run at day ${E.start_day}, with random initial errors of different amplitude and scale. Members are perturbed copies of the truth: errors measure how far a perturbed run drifts from it.`),
  ];

  const grow = chart({ title: 'Error growth', sub: '', height: 320 });
  const cascade = chart({ title: 'Where the error lives in wavenumber', sub: '', height: 320 });
  const tableBox = el('div');
  const mapGrid = el('div', { class: 'cols three' });
  const note = el('p', { class: 'hint' });
  root.append(layoutRail(rail, [panel('Doubling times', 'Fastest early growth rate of the error against linear theory', tableBox), cols(grow.el, cascade.el), panel('Forecast maps', 'Z′ (m)', mapGrid, note)]));

  // Saturation level of the member error: the rms difference of two unrelated states of the control run.
  // (Not sqrt(2) times the standard deviation: the control run has a persistent pattern shared by all states.)
  let pairCache = null;
  function pairRms() {
    if (pairCache) return pairCache;
    const nt = ds.nt, minSep = Math.round(30 / ds.dtDays);
    let seed = 12345;
    const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
    const acc = [0, 0];
    let n = 0;
    for (let tries = 0; n < 300 && tries < 20000; tries++) {
      const i = Math.floor(rnd() * nt), j = Math.floor(rnd() * nt);
      if (Math.abs(i - j) < minSep) continue;
      const a = ds.getPsi(i), b = ds.getPsi(j);
      for (let L = 0; L < 2; L++) { let s = 0; for (let p = 0; p < N; p++) { const d = a[L][p] - b[L][p]; s += d * d; } acc[L] += s / N; }
      n++;
    }
    pairCache = n ? acc.map((v) => Math.sqrt(v / n) * zf) : [NaN, NaN];
    return pairCache;
  }
  function groupColor(i, th) { return th.series(i); }
  const lead = () => Array.from({ length: ens.groups[0].spread_z1.length }, (_, k) => (k * ens.lead_hours) / 24);

  async function load() {
    ens = await ds.json(E.file);
    groupCtl.setOptions(ens.groups.map((g, i) => ({ value: String(i), label: g.name })), '0');
    F = {};
    await busy('Loading forecast fields…', async (prog) => {
      const keys = ['truth', 'mean', 'spread', 'stamps'].filter((k) => E.fields[k]);
      for (let i = 0; i < keys.length; i++) { F[keys[i]] = await ds.array(E.fields[keys[i]]); prog((i + 1) / keys.length); }
    });
    buildMaps();
  }

  function metricOf(g, kind, l) {
    if (kind === 'member') return { all: g['rms_z' + l], y: avg_cols(g['rms_z' + l]), unit: 'm', ref: pairRms()[Number(l) - 1], refLabel: 'rms difference of two unrelated states of the control run' };
    if (kind === 'mean') return { y: g['rmse_mean_z' + l], unit: 'm' };
    if (kind === 'spread') return { y: g['spread_z' + l], unit: 'm' };
    return { all: g.dte, y: avg_cols(g.dte), unit: 'm² s⁻²' };
  }
  function avg_cols(a) { return a[0].map((_, k) => avg(a.map((r) => r[k]))); }

  function draw() {
    if (!ens) return;
    const th = theme(), l = layerCtl.value, kind = metricCtl.value, t = lead();
    const sel = ens.groups[Number(groupCtl.value)];
    const traces = [];
    const mt = metricOf(sel, kind, l);
    if (mt.all) mt.all.forEach((row, i) => traces.push(lineTrace(`Member ${i + 1}`, t, row, groupColor(Number(groupCtl.value), th), { line: { width: 0.8 }, opacity: 0.28, showlegend: false, hoverinfo: 'skip', meta: { table: false } })));
    ens.groups.forEach((g, i) => traces.push(lineTrace(g.name, t, metricOf(g, kind, l).y, groupColor(i, th), { line: { width: 2.2 } })));
    const shapes = [];
    if (Number.isFinite(mt.ref)) shapes.push({ type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: mt.ref, y1: mt.ref, line: { color: th.muted, width: 1.2, dash: 'dot' }, layer: 'below' });
    const fa = fit0.value, fb = fit1.value;
    if (fb > fa) shapes.push({ type: 'rect', xref: 'x', yref: 'paper', x0: fa, x1: fb, y0: 0, y1: 1, fillcolor: th.accent, opacity: 0.08, line: { width: 0 }, layer: 'below' });
    const unit = mt.unit;
    grow.setSub(mt.refLabel ? `Dotted line: ${mt.refLabel}. Shaded: doubling-time fit window` : 'Shaded: doubling-time fit window');
    grow.set(traces, layout(th, { x: { title: 'Lead time (days)' }, y: { title: unit === 'm' ? 'Error in Z′ (m)' : `Energy (${unit})`, type: logCtl.value ? 'log' : 'linear', exponentformat: 'power' }, shapes, hover: 'x' }), { xTitle: 'lead (days)' });

    // spectrum of the error
    const ks = Array.from({ length: ds.grid.kCut }, (_, k) => k + 1);
    const nLead = sel.err_spectrum.length;
    const days = [0, 1, 2, 4, 7, 10, 15, 20, 30].filter((d) => (d * 24) / ens.lead_hours <= nLead - 1);
    const picks = days.length > 6 ? days.filter((_, i) => i !== 6 && i !== 7) : days;
    cascade.setSub(`${sel.name}: error energy per wavenumber shell at selected lead times`);
    cascade.set(picks.map((d, i) => {
      const k = Math.round((d * 24) / ens.lead_hours);
      const c = hex(sequentialColor(picks.length > 1 ? 0.25 + (0.75 * i) / (picks.length - 1) : 0.7, th.dark));
      return lineTrace(`day ${d}`, ks, sel.err_spectrum[k].slice(0, ks.length).map((v) => Math.max(v, 1e-30)), c, { line: { width: 1.8 } });
    }), layout(th, { x: { title: 'Total wavenumber k (units of 2π / L)', type: 'log', dtick: 'D2' }, y: { title: 'Error energy per shell (m² s⁻²)', type: 'log', exponentformat: 'power' }, hover: 'x' }), { xTitle: 'k' });

    // doubling table
    const lp = linearParams(m), dk = (2 * Math.PI) / ds.grid.Lx;
    const best = fastestMode(lp, (ds.grid.nx / 2) * dk, 4000);
    const linDouble = best.sigma > 0 ? Math.LN2 / (best.sigma * DAY) : NaN;
    const head = el('tr', {}, el('th', {}, 'Initial error'), el('th', {}, 'Initial rms (m)'), el('th', {}, 'Doubling time in window (days)'), el('th', {}, 'Day the member error reaches half its saturation'));
    const rows = ens.groups.map((g) => {
      const y = metricOf(g, 'member', l).y;
      const xs = [], ys = [];
      t.forEach((tt, k) => { if (tt >= fa && tt <= fb && y[k] > 0) { xs.push(tt); ys.push(Math.log(y[k])); } });
      const slope = xs.length >= 2 ? linearFit(xs, ys).b : NaN;
      const sat = pairRms()[Number(l) - 1] * 0.5;
      const kHalf = y.findIndex((v) => v >= sat);
      return el('tr', {}, el('td', {}, g.name), el('td', {}, fmt(y[0], 3)), el('td', {}, Number.isFinite(slope) && slope > 0 ? fmt(Math.LN2 / slope, 3) : '–'), el('td', {}, kHalf >= 0 ? fmt(t[kHalf], 3) : '> ' + fmt(t[t.length - 1], 3)));
    });
    tableBox.replaceChildren(dataTable( el('thead', {}, head), el('tbody', {}, ...rows)),
      stats(stat('Linear theory: doubling time of the fastest mode', Number.isFinite(linDouble) ? `${fmt(linDouble, 3)} days` : 'stable'),
        stat('Saturation: rms difference of unrelated states (upper layer)', `${fmt(pairRms()[0], 3)} m`)));
  }

  // ---- maps ---------------------------------------------------------------------------------------------------
  function buildMaps() {
    maps.forEach((q) => q.mv.destroy());
    mapGrid.replaceChildren();
    const defs = [
      { id: 'truth', label: 'Truth', kind: 'diverging' }, { id: 'mean', label: 'Ensemble mean', kind: 'diverging' }, { id: 'spread', label: 'Ensemble spread', kind: 'sequential' },
      ...(F.stamps ? Array.from({ length: E.stamps }, (_, i) => ({ id: 'stamp' + i, label: `Member ${i + 1}`, kind: 'diverging', i })) : []),
    ];
    maps = defs.map((d) => { const box = el('div'); mapGrid.append(box); return { ...d, mv: new MapView(box, { grid: ds.grid }) }; });
  }

  function slice(arr, shape, idx) { const n = N; return arr.subarray(idx * n, (idx + 1) * n); }
  function drawMaps() {
    if (!maps.length) return;
    const l = Number(layerCtl.value) - 1, tIdx = Math.round((leadCtl.value * 24) / E.field_step_hours);
    const nT = E.fields.truth.shape[0];
    const ti = clamp(tIdx, 0, nT - 1);
    const field = (id) => {
      if (id.startsWith('stamp')) { const i = Number(id.slice(5)); return F.stamps.subarray((i * nT + ti) * N, (i * nT + ti + 1) * N); }
      return F[id].subarray((ti * 2 + l) * N, (ti * 2 + l + 1) * N);
    };
    const truth = field('truth');
    const lim = Math.max(10, Math.ceil((symmetricLimit(truth, 0.999) * zf) / 10) * 10);
    let slim = 0; const sp = field('spread'); for (let p = 0; p < N; p++) slim = Math.max(slim, sp[p] * zf);
    slim = Math.max(5, Math.ceil(slim / 5) * 5);
    for (const q of maps) {
      if (q.id.startsWith('stamp') && l !== 0) { q.mv.container.hidden = true; continue; }
      q.mv.container.hidden = false;
      const v = field(q.id);
      q.mv.setTitle(`${q.label} · day ${leadCtl.value}`);
      q.mv.setData({ values: v, scale: zf, vmin: q.kind === 'diverging' ? -lim : 0, vmax: q.kind === 'diverging' ? lim : slim, kind: q.kind, unit: 'm',
        contour: q.kind === 'diverging' ? { values: Float64Array.from(v, (x) => x * zf), step: niceStep(2 * lim, 9) } : undefined });
    }
    note.textContent = l !== 0 ? 'Individual members are stored for the upper layer only.' : `Members are drawn from the “${ens.groups[0].name}” group.`;
  }

  return {
    async activate() { if (!ens) { await load(); draw(); drawMaps(); } else { grow.resize(); cascade.resize(); maps.forEach((q) => q.mv.draw()); } },
    refresh() { draw(); drawMaps(); },
    destroy() { grow.purge(); cascade.purge(); maps.forEach((q) => q.mv.destroy()); },
  };
}
