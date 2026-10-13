// Parameter sweep: how the statistically steady state responds to the shear and the drag.

import { el, fmt, niceStep } from '../util.js';
import { MapView } from '../mapview.js';
import { Snapshot } from '../physics.js';
import { symmetricLimit } from '../stats.js';
import { theme, sequentialColor, hex } from '../colors.js';
import { chart, layout, lineTrace, layoutRail, group, selectCtl, checkCtl, busy, stats, stat, cols, panel, dataTable } from '../ui.js';

const PARAMS = {
  shear_ratio: { label: 'Shear / critical shear', log: false, marker: (m) => m.dU_ratio, critical: 1 },
  r_ek_days: { label: 'Ekman drag time 1/r (days)', log: true, marker: (m) => m.r_ek_days },
};
const RESP = [
  { k: 'E', label: 'Total energy E', unit: 'm² s⁻²', s: 1 },
  { k: 'EKE1', label: 'Eddy kinetic energy, upper layer', unit: 'm² s⁻²', s: 1 },
  { k: 'EKE2', label: 'Eddy kinetic energy, lower layer', unit: 'm² s⁻²', s: 1 },
  { k: 'ZKE1', label: 'Zonal-mean kinetic energy, upper layer', unit: 'm² s⁻²', s: 1 },
  { k: 'APE', label: 'Available potential energy', unit: 'm² s⁻²', s: 1 },
  { k: 'C_B', label: 'Conversion from the mean shear', unit: '10⁻⁴ m² s⁻³', s: 1e4 },
  { k: 'D_E', label: 'Ekman dissipation', unit: '10⁻⁴ m² s⁻³', s: 1e4 },
  { k: 'v_rms1', label: 'rms meridional wind, upper layer', unit: 'm s⁻¹', s: 1 },
  { k: 'u_rms1', label: 'rms zonal wind perturbation, upper layer', unit: 'm s⁻¹', s: 1 },
  { k: 'z_rms1', label: 'rms height anomaly Z′, upper layer', unit: 'm', s: 1 },
  { k: 'tau_rms', label: 'rms of ψ₁ − ψ₂ (thermal wind)', unit: '10⁶ m² s⁻¹', s: 1e-6 },
];

export default function create({ ds }, root) {
  const m = ds.model, { nx, ny, N } = ds.grid;
  const yKm = Array.from(ds.grid.y, (v) => v / 1e3);
  let sw = null, finalPsi = null;

  const axisCtl = selectCtl({ label: 'Parameter varied', options: [], onchange: () => { fillCases(); draw(); } });
  const respCtl = selectCtl({ label: 'Response', options: RESP.map((r) => ({ value: r.k, label: r.label })), value: 'E', onchange: () => draw() });
  const caseCtl = selectCtl({ label: 'Case shown on the map', options: [], onchange: () => draw() });
  const logCtl = checkCtl({ label: 'Logarithmic response and energy axes', checked: true, onchange: () => draw() });
  const rail = [group(null, axisCtl.el, respCtl.el, logCtl.el), group(null, caseCtl.el),
    el('p', { class: 'hint' }, 'Each point is a separate model run: a spin-up followed by an averaging window. The dotted vertical line marks the control run; for the shear, 1 is the Phillips critical value below which there is no linear instability.')];

  const resp = chart({ title: 'Response to the parameter', sub: '', height: 300 });
  const ts = chart({ title: 'Total energy through time', sub: 'Spin-up and averaging window of every case', height: 300 });
  const prof = chart({ title: 'Time-mean upper-layer wind', sub: '', height: 300 });
  const spec = chart({ title: 'Mean upper-layer kinetic energy spectrum', sub: '', height: 300 });
  const mapBox = el('div', { style: 'max-width:620px' });
  let map = null;
  const tableBox = el('div');
  root.append(layoutRail(rail, [panel('Cases', '', tableBox), cols(resp.el, ts.el), cols(prof.el, spec.el), panel('Final state of the selected case', 'Upper-layer Z′ (m) at the end of the run', mapBox)]));

  const cases = () => sw.cases.filter((c) => c.meta.axis === Number(axisCtl.value));
  function fillCases() {
    const cs = cases();
    const ctrl = PARAMS[sw.axes[Number(axisCtl.value)].parameter];
    let best = 0;
    if (ctrl && ctrl.marker) cs.forEach((c, i) => { if (Math.abs(Math.log(c.meta.value / ctrl.marker(m))) < Math.abs(Math.log(cs[best].meta.value / ctrl.marker(m)))) best = i; });
    caseCtl.setOptions(cs.map((c, i) => ({ value: String(i), label: c.meta.label })), String(best));
  }

  async function load() {
    sw = await ds.json(ds.manifest.sweep.file);
    axisCtl.setOptions(sw.axes.map((a, i) => ({ value: String(i), label: (PARAMS[a.parameter] || {}).label || a.parameter })), '0');
    const have = RESP.filter((r) => sw.cases[0].mean[r.k] !== undefined);
    respCtl.setOptions(have.map((r) => ({ value: r.k, label: r.label })), 'E');
    const fp = ds.manifest.sweep.final_psi;
    if (fp) finalPsi = await busy('Loading final states…', () => ds.array(fp));
    fillCases();
    map = new MapView(mapBox, { grid: ds.grid });
  }

  function draw() {
    if (!sw) return;
    const th = theme();
    const cs = cases();
    const pname = sw.axes[Number(axisCtl.value)].parameter;
    const P = PARAMS[pname] || { label: pname, log: false };
    const r = RESP.find((q) => q.k === respCtl.value);
    const xs = cs.map((c) => c.meta.value);
    const shapes = [];
    if (P.marker && P.marker(m) !== undefined) shapes.push({ type: 'line', xref: 'x', yref: 'paper', x0: P.marker(m), x1: P.marker(m), y0: 0, y1: 1, line: { color: th.muted, width: 1.2, dash: 'dot' }, layer: 'below' });
    if (P.critical !== undefined) shapes.push({ type: 'line', xref: 'x', yref: 'paper', x0: P.critical, x1: P.critical, y0: 0, y1: 1, line: { color: th.red, width: 1.2, dash: 'dash' }, layer: 'below' });
    resp.setSub(r.label);
    resp.set([{ type: 'scatter', mode: 'lines+markers', name: r.label, x: xs, y: cs.map((c) => c.mean[r.k] * r.s), line: { color: th.upper, width: 2 }, marker: { size: 8, color: th.upper, line: { color: th.surface, width: 2 } },
      hovertemplate: `${P.label} %{x}<br>%{y:.4g} ${r.unit}<extra></extra>` }],
      layout(th, { x: { title: P.label, type: P.log ? 'log' : 'linear' }, y: { title: r.unit, type: logCtl.value && xs.length ? 'log' : 'linear', rangemode: logCtl.value ? undefined : 'tozero' }, legend: false, hover: 'closest', shapes }), { xTitle: P.label });

    const col = (i) => hex(sequentialColor(cs.length > 1 ? 0.3 + (0.7 * i) / (cs.length - 1) : 0.7, th.dark));
    ts.set(cs.map((c, i) => lineTrace(c.meta.label, c.t_days, c.series.E, col(i), { line: { width: 1.4 } })),
      layout(th, { x: { title: 'Time (days)' }, y: { title: 'E (m² s⁻²)', type: logCtl.value ? 'log' : 'linear', rangemode: logCtl.value ? undefined : 'tozero' }, hover: 'x' }), { xTitle: 'day' });
    prof.set(cs.map((c, i) => lineTrace(c.meta.label, c.mean.u1_total, yKm, col(i), { line: { width: 1.6 } })),
      layout(th, { x: { title: 'Total zonal wind (m s⁻¹)' }, y: { title: 'y (km)' }, hover: 'y' }), { xTitle: 'u' });
    const ks = Array.from({ length: ds.grid.kCut }, (_, k) => k + 1);
    spec.set(cs.map((c, i) => lineTrace(c.meta.label, ks, c.mean.spec_ke1.slice(0, ks.length), col(i), { line: { width: 1.6 } })),
      layout(th, { x: { title: 'Total wavenumber k (units of 2π / L)', type: 'log', dtick: 'D2' }, y: { title: 'Energy per shell (m² s⁻²)', type: 'log', exponentformat: 'power' }, hover: 'x' }), { xTitle: 'k' });

    // table
    const head = el('tr', {}, el('th', {}, 'Case'), el('th', {}, 'E'), el('th', {}, 'Eddy KE (upper)'), el('th', {}, 'C (10⁻⁴)'), el('th', {}, 'E / D (days)'));
    const body = cs.map((c) => el('tr', {}, el('td', {}, c.meta.label), el('td', {}, fmt(c.mean.E, 3)), el('td', {}, fmt(c.mean.EKE1, 3)), el('td', {}, fmt(c.mean.C_B * 1e4, 3)),
      el('td', {}, fmt(c.mean.E / (c.mean.D_E + (c.mean.D_h || 0)) / 86400, 3))));
    tableBox.replaceChildren(dataTable( el('thead', {}, head), el('tbody', {}, ...body)),
      el('p', { class: 'hint' }, `Averages over the last ${ds.manifest.sweep.average_days} days of each run (after ${ds.manifest.sweep.spinup_days} days of spin-up).`));

    // map
    const sel = cs[Number(caseCtl.value)];
    if (finalPsi && sel && map) {
      const gi = sw.cases.indexOf(sel);
      const p1 = finalPsi.slice(gi * 2 * N, gi * 2 * N + N), p2 = finalPsi.slice(gi * 2 * N + N, (gi + 1) * 2 * N);
      const snap = new Snapshot(m, ds.spec, Float64Array.from(p1), Float64Array.from(p2));
      const z = snap.field('Z', 'upper');
      const lim = Math.max(10, Math.ceil(symmetricLimit(z, 0.995) / 10) * 10);
      map.setTitle(sel.meta.label);
      map.setData({ values: z, scale: 1, vmin: -lim, vmax: lim, kind: 'diverging', unit: 'm', contour: { values: z, step: niceStep(2 * lim, 9) } });
    }
  }

  return {
    async activate() { if (!sw) { await load(); draw(); } else { [resp, ts, prof, spec].forEach((c) => c.resize()); map && map.draw(); } },
    refresh() { draw(); },
    destroy() { [resp, ts, prof, spec].forEach((c) => c.purge()); map && map.destroy(); },
  };
}
