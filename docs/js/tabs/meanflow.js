// Mean flow and EOFs: zonal-mean jets, eddy fluxes and the dominant patterns of jet variability.

import { el, fmt } from '../util.js';
import { eof } from '../eof.js';
import { autocorrelation, efoldingLag, mean as avg } from '../stats.js';
import { theme, plotlyScale } from '../colors.js';
import { chart, layout, lineTrace, layoutRail, group, selectCtl, busy, stats, stat, cols, panel } from '../ui.js';

export default function create({ ds }, root) {
  const { ny } = ds.grid, nt = ds.nt, m = ds.model;
  const yKm = Array.from(ds.grid.y, (v) => v / 1e3);
  const days = Array.from({ length: nt }, (_, t) => ds.timeDays(t));
  let diag = null;

  const layerCtl = selectCtl({ label: 'Layer for the Hovmöller diagram and EOFs', options: [{ value: 'upper', label: 'Upper layer' }, { value: 'lower', label: 'Lower layer' }, { value: 'barotropic', label: 'Barotropic mean' }], value: 'upper', onchange: () => draw() });
  const rail = [group(null, layerCtl.el), el('p', { class: 'hint' }, 'Zonal means are averages along x at each y. The imposed mean wind is added back for the profile plot; the other panels show departures from the time mean.')];

  const prof = chart({ title: 'Time-mean zonal wind', sub: 'Lines: time mean. Bands: ±1 standard deviation in time', height: 300 });
  const flux = chart({ title: 'Eddy fluxes', sub: 'Time-mean momentum flux ⟨u′v′⟩ of each layer (m² s⁻²)', height: 300 });
  const hovC = chart({ title: 'Zonal-mean wind anomaly through time', sub: '', height: 340 });
  const scree = chart({ title: 'Variance explained by the EOFs', sub: '', height: 340 });
  const pat = chart({ title: 'EOF patterns', sub: 'Amplitude for a one-standard-deviation principal component', height: 300 });
  const pcs = chart({ title: 'Principal components', sub: '', height: 300 });
  const strip = el('div');
  root.append(layoutRail(rail, [panel('Summary', '', strip), cols(prof.el, flux.el), cols(hovC.el, scree.el), cols(pat.el, pcs.el)]));

  const timeMean = (a) => { const out = new Float64Array(ny); for (let t = 0; t < nt; t++) for (let j = 0; j < ny; j++) out[j] += a[t * ny + j] / nt; return out; };
  const timeStd = (a, mu) => { const out = new Float64Array(ny); for (let t = 0; t < nt; t++) for (let j = 0; j < ny; j++) { const d = a[t * ny + j] - mu[j]; out[j] += (d * d) / nt; } return out.map(Math.sqrt); };
  const combo = (l) => {
    const { ubar1, ubar2 } = diag.profiles;
    if (l === 'upper') return ubar1;
    if (l === 'lower') return ubar2;
    return ubar1.map((v, k) => m.delta * v + (1 - m.delta) * ubar2[k]);
  };
  const meanWind = (l) => (l === 'upper' ? m.U1 : l === 'lower' ? m.U2 : m.delta * m.U1 + (1 - m.delta) * m.U2);

  function band(name, color, mu, sd, off, th) {
    const hi = Array.from(mu, (v, j) => v + off + sd[j]), lo = Array.from(mu, (v, j) => v + off - sd[j]);
    return [
      { type: 'scatter', mode: 'lines', x: lo, y: yKm, line: { width: 0 }, hoverinfo: 'skip', showlegend: false, meta: { table: false } },
      { type: 'scatter', mode: 'lines', x: hi, y: yKm, fill: 'tonextx', fillcolor: color + '33', line: { width: 0 }, hoverinfo: 'skip', showlegend: false, meta: { table: false } },
      lineTrace(name, Array.from(mu, (v) => v + off), yKm, color, { legendrank: name.startsWith('Upper') ? 1 : 2 }),
    ];
  }

  function draw() {
    if (!diag) return;
    const th = theme();
    const { ubar1, ubar2, uv1, uv2, vt } = diag.profiles;
    const mu1 = timeMean(ubar1), mu2 = timeMean(ubar2), s1 = timeStd(ubar1, mu1), s2 = timeStd(ubar2, mu2);
    prof.set([...band('Upper layer', th.upper, mu1, s1, m.U1, th), ...band('Lower layer', th.lower, mu2, s2, m.U2, th)],
      layout(th, { x: { title: 'Zonal wind (m s⁻¹)' }, y: { title: 'y (km)' }, hover: 'y unified' }), { xTitle: 'u' });
    flux.set([lineTrace('Upper layer', Array.from(timeMean(uv1)), yKm, th.upper), lineTrace('Lower layer', Array.from(timeMean(uv2)), yKm, th.lower)],
      layout(th, { x: { title: '⟨u′v′⟩ (m² s⁻²)', zeroline: true }, y: { title: 'y (km)' }, hover: 'y unified' }), { xTitle: 'flux' });

    const l = layerCtl.value;
    const U = combo(l);
    const mu = timeMean(U);
    const anomaly = Array.from({ length: nt }, (_, t) => Array.from({ length: ny }, (_, j) => U[t * ny + j] - mu[j]));
    let lim = 0; for (const r of anomaly) for (const v of r) lim = Math.max(lim, Math.abs(v));
    lim = Math.max(1e-6, lim * 0.9);
    hovC.set([{ type: 'heatmap', x: yKm, y: days, z: anomaly, zmin: -lim, zmax: lim, zmid: 0, colorscale: plotlyScale('diverging', th.dark), hovertemplate: 'y %{x:.0f} km<br>day %{y:.1f}<br>%{z:.2f} m/s<extra></extra>',
      colorbar: { thickness: 10, tickfont: { size: 11, color: th.muted }, outlinewidth: 0, title: { text: 'm s⁻¹', side: 'top', font: { size: 11, color: th.muted } } } }],
      layout(th, { x: { title: 'y (km)' }, y: { title: 'Time (days)' }, legend: false, hover: 'closest', margin: { r: 6 } }), { table: false });

    const e = eof(U, nt, ny, 4);
    const names = e.values.map((_, k) => `EOF ${k + 1}`);
    scree.setSub(`${e.fractions.slice(0, 2).map((f) => fmt(100 * f, 3) + ' %').join(' and ')} in the first two`);
    scree.set([{ type: 'bar', x: names, y: e.fractions.map((f) => 100 * f), marker: { color: th.series(0) }, hovertemplate: '%{x}: %{y:.1f} %<extra></extra>', name: 'Variance explained' }],
      layout(th, { x: { title: '' }, y: { title: 'Variance explained (%)' }, legend: false, hover: 'closest' }), { xTitle: 'mode' });
    pat.set(e.patterns.slice(0, 3).map((p, k) => lineTrace(`EOF ${k + 1}`, Array.from(p, (v) => v * Math.sqrt(e.values[k])), yKm, th.series(k))),
      layout(th, { x: { title: 'Wind anomaly per σ of the PC (m s⁻¹)', zeroline: true }, y: { title: 'y (km)' }, hover: 'y unified' }), { xTitle: 'amplitude' });
    pcs.set(e.pcs.slice(0, 2).map((pc, k) => lineTrace(`PC ${k + 1}`, days, Array.from(pc, (v) => v / Math.sqrt(e.values[k])), th.series(k), { line: { width: 1.5 } })),
      layout(th, { x: { title: 'Time (days)' }, y: { title: 'Standardised amplitude' } }), { xTitle: 'day' });

    const acf = autocorrelation(e.pcs[0], Math.min(nt - 2, Math.round(60 / ds.dtDays)));
    const ef = efoldingLag(acf) * ds.dtDays;
    const jet = (arr) => { let best = 0; const tm = timeMean(arr); for (let j = 1; j < ny; j++) if (tm[j] > tm[best]) best = j; return yKm[best]; };
    strip.replaceChildren(stats(
      stat('Upper-layer jet maximum at', `y ≈ ${fmt(jet(ubar1), 4)} km`),
      stat('Time-mean wind change, upper layer (max)', `${fmt(Math.max(...mu1.map(Math.abs)), 3)} m s⁻¹`),
      stat('EOF 1 variance', `${fmt(100 * e.fractions[0], 3)} %`),
      stat('PC 1 decorrelation time', Number.isFinite(ef) ? `${fmt(ef, 3)} days` : '> 60 days'),
      stat('Layer', l === 'barotropic' ? 'Barotropic' : l === 'upper' ? 'Upper' : 'Lower'),
    ));
  }

  return {
    async activate() {
      if (!diag) {
        diag = await busy('Computing zonal means…', (prog) => ds.diagnostics(prog));
        draw();
      } else [prof, flux, hovC, scree, pat, pcs].forEach((c) => c.resize());
    },
    refresh() { draw(); },
    destroy() { [prof, flux, hovC, scree, pat, pcs].forEach((c) => c.purge()); },
  };
}
