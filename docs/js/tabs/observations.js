// Observations: an observing-system experiment with a synthetic station network and an
// ensemble Kalman filter. Error against network density, maps of analysis and error, station series.

import { el, fmt, niceStep, clamp } from '../util.js';
import { MapView } from '../mapview.js';
import { symmetricLimit } from '../stats.js';
import { theme } from '../colors.js';
import { chart, layout, lineTrace, layoutRail, group, selectCtl, sliderCtl, busy, stats, stat, cols, panel, dataTable } from '../ui.js';

export default function create({ ds }, root) {
  const m = ds.model, N = ds.grid.N, zf = m.zFactor;
  const A0 = ds.manifest.assimilation;
  let A = null, snaps = null, series = null;
  let maps = [];

  const layerCtl = selectCtl({ label: 'Layer', options: [{ value: '0', label: 'Upper layer' }, { value: '1', label: 'Lower layer' }], value: '0', onchange: () => { drawErrors(); drawMaps(); drawStation(); } });
  const measureCtl = selectCtl({ label: 'Measure in the error plot', options: [
    { value: 'an_rmse', label: 'Analysis error (rms of Z′)' },
    { value: 'bg_rmse', label: 'Background (forecast) error' },
    { value: 'bg_spread', label: 'Background ensemble spread' }], value: 'an_rmse', onchange: () => drawErrors() });
  const cyc = A0.snapshot_cycles;
  const cycCtl = sliderCtl({ label: 'Cycle for the maps', min: 0, max: cyc.length - 1, step: 1, value: Math.min(2, cyc.length - 1), format: (i) => `day ${fmt(A0.start_day + (cyc[i] * A0.cycle_hours) / 24, 4)}`, oninput: () => drawMaps() });
  const stCtl = sliderCtl({ label: 'Station for the time series', min: 0, max: A0.primary - 1, step: 1, value: 0, format: (i) => `station ${i + 1}`, oninput: () => drawStation() });
  const rail = [
    group(null, layerCtl.el, measureCtl.el),
    group('Maps', cycCtl.el),
    group('Station', stCtl.el, el('p', { class: 'hint' }, 'You can also click a station on the truth map.')),
    el('p', { class: 'hint' }, `Stations measure the height anomaly of each layer with an error of ${(ds.manifest.config && ds.manifest.config.assimilation && ds.manifest.config.assimilation.obs_error_z) || 10} m (equivalent); an ensemble of ${A0.members} members is updated every ${A0.cycle_hours} h with covariance localisation of ${A0.localization_km} km and inflation ${A0.inflation}.`),
  ];

  const errC = chart({ title: 'Error against time for different network sizes', sub: '', height: 320 });
  const stC = chart({ title: 'At one station', sub: '', height: 320 });
  const tableBox = el('div');
  const mapGrid = el('div', { class: 'cols' });
  root.append(layoutRail(rail, [panel('Scores', 'Averaged over the cycles after spin-up', tableBox), cols(errC.el, stC.el), panel('Maps at the chosen cycle', `Z′ (m), network of ${A0.primary} stations`, mapGrid)]));

  async function load() {
    A = await ds.json(A0.file);
    await busy('Loading observation-experiment fields…', async (prog) => {
      snaps = await ds.array(A0.snapshots); prog(0.7);
      series = await ds.array(A0.station_series); prog(1);
    });
    const defs = [{ id: 0, label: 'Truth', kind: 'diverging' }, { id: 2, label: 'Analysis', kind: 'diverging' }, { id: 'err', label: 'Analysis − truth', kind: 'diverging' }, { id: 3, label: 'Analysis spread', kind: 'sequential' }];
    maps = defs.map((d) => {
      const box = el('div'); mapGrid.append(box);
      return { ...d, mv: new MapView(box, { grid: ds.grid, onPick: d.id === 0 ? (i, j) => nearestStation(i, j) : null }) };
    });
    stCtl.setRange(0, A.positions_km.length > A0.primary ? A0.primary - 1 : A.positions_km.length - 1);
  }

  function nearestStation(i, j) {
    const x = (i + 0.5) * ds.grid.dx, y = (j + 0.5) * ds.grid.dy;
    let best = 0, bd = Infinity;
    for (let s = 0; s < A0.primary; s++) {
      let dx = Math.abs(A.positions_km[s][0] * 1e3 - x), dy = Math.abs(A.positions_km[s][1] * 1e3 - y);
      dx = Math.min(dx, ds.grid.Lx - dx); dy = Math.min(dy, ds.grid.Ly - dy);
      const d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = s; }
    }
    stCtl.value = best;
    drawStation();
  }

  const days = (n, off = 0) => Array.from({ length: n }, (_, k) => A0.start_day + ((k + off) * A0.cycle_hours) / 24);

  function drawErrors() {
    if (!A) return;
    const th = theme(), l = Number(layerCtl.value), key = measureCtl.value;
    const n = A.free.series[key][l].length, t = days(n);
    const traces = [];
    traces.push(lineTrace('No observations', t, A.free.series[key][l], th.muted, { line: { dash: 'dash', width: 1.6 } }));
    A.runs.forEach((r, i) => traces.push(lineTrace(`${r.stations} stations`, t, r.series[key][l], th.series(i), { line: { width: 2 } })));
    const shapes = [];
    if (l === 0 && A0.climatological_std_z1) shapes.push({ type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: A0.climatological_std_z1, y1: A0.climatological_std_z1, line: { color: th.muted, width: 1, dash: 'dot' }, layer: 'below' });
    errC.setSub(l === 0 ? 'Dotted line: climatological standard deviation of Z′' : '');
    errC.set(traces, layout(th, { x: { title: 'Time (days)' }, y: { title: 'm of Z′', type: 'log', exponentformat: 'power' }, shapes }), { xTitle: 'day' });

    const head = el('tr', {}, ...['Network', 'Analysis error (m)', 'Background error (m)', 'Background spread (m)', 'Analysis error / no observations'].map((h) => el('th', {}, h)));
    const base = A.free.score[l];
    const rows = [el('tr', {}, el('td', {}, 'No observations'), el('td', {}, fmt(base, 3)), el('td', {}, fmt(base, 3)), el('td', {}, '–'), el('td', {}, '1'))]
      .concat(A.runs.map((r) => el('tr', {}, el('td', {}, `${r.stations} stations`), el('td', {}, fmt(r.score_an[l], 3)), el('td', {}, fmt(r.score_bg[l], 3)), el('td', {}, fmt(r.score_spread_bg[l], 3)), el('td', {}, fmt(r.score_an[l] / base, 2)))));
    tableBox.replaceChildren(dataTable( el('thead', {}, head), el('tbody', {}, ...rows)),
      el('p', { class: 'hint' }, 'Spread close to the error means the ensemble is statistically consistent. Spread much larger than the error means the network constrains the state better than the ensemble knows.'));
  }

  function drawMaps() {
    if (!A || !maps.length) return;
    const th = theme(), l = Number(layerCtl.value), c = cycCtl.value;
    const get = (name) => { const o = ((c * 4 + name) * 2 + l) * N; return snaps.subarray(o, o + N); };
    const truth = get(0), an = get(2), sp = get(3);
    const err = Float64Array.from(an, (v, p) => v - truth[p]);
    const lim = Math.max(10, Math.ceil((symmetricLimit(truth, 0.999) * zf) / 10) * 10);
    let slim = 0; for (let p = 0; p < N; p++) slim = Math.max(slim, sp[p] * zf);
    slim = Math.max(5, Math.ceil(slim / 5) * 5);
    const elim = Math.max(5, Math.ceil((symmetricLimit(err, 0.999) * zf) / 5) * 5);
    const pos = A.positions_km.slice(0, A0.primary).map(([x, y]) => ({ x: x * 1e3, y: y * 1e3, r: 2.6, fill: th.surface, color: th.ink, lw: 1.2 }));
    for (const q of maps) {
      const v = q.id === 'err' ? err : get(q.id);
      const range = q.id === 'err' ? elim : q.kind === 'diverging' ? lim : slim;
      q.mv.setTitle(`${q.label} · day ${fmt(A0.start_day + (cyc[c] * A0.cycle_hours) / 24, 4)}`);
      q.mv.setMarkers(q.id === 0 || q.id === 'err' ? pos : []);
      q.mv.setData({ values: v, scale: zf, vmin: q.kind === 'diverging' ? -range : 0, vmax: range, kind: q.kind, unit: 'm',
        contour: q.id === 'err' || q.kind === 'sequential' ? undefined : { values: Float64Array.from(v, (x) => x * zf), step: niceStep(2 * Math.max(10, symmetricLimit(v, 0.999) * zf), 9) } });
    }
  }

  function drawStation() {
    if (!A || !series) return;
    const th = theme(), l = Number(layerCtl.value), s = stCtl.value;
    const nvar = A0.station_series.shape[1], nc = A0.cycles, nst = A0.station_series.shape[3];
    const vi = A.variables.indexOf(`psi${l + 1}`);
    if (vi < 0 || vi >= nvar) {
      stC.setSub(`The ${l === 0 ? 'upper' : 'lower'} layer's height was not observed in this experiment (observed: ${A.variables.join(', ')}).`);
      stC.set([], layout(th, { legend: false }), { table: false });
      return;
    }
    const get = (name) => Array.from({ length: nc }, (_, k) => series[(((name * nvar + vi) * nc) + k) * nst + s]);
    const t = days(nc, 1);
    const truth = get(0), obs = get(1), bg = get(2), an = get(3), spr = get(4);
    const hi = bg.map((v, k) => v + spr[k]), lo = bg.map((v, k) => v - spr[k]);
    const [xk, yk] = A.positions_km[s];
    stC.setSub(`Station ${s + 1} at x = ${Math.round(xk)} km, y = ${Math.round(yk)} km; band: background ± spread`);
    stC.set([
      { type: 'scatter', mode: 'lines', x: t, y: lo, line: { width: 0 }, hoverinfo: 'skip', showlegend: false, meta: { table: false } },
      { type: 'scatter', mode: 'lines', x: t, y: hi, fill: 'tonexty', fillcolor: th.series(0) + '2e', line: { width: 0 }, hoverinfo: 'skip', showlegend: false, meta: { table: false } },
      lineTrace('Truth', t, truth, th.ink, { line: { width: 2 } }),
      { type: 'scatter', mode: 'markers', name: 'Observation', x: t, y: obs, marker: { size: 6, color: th.series(1), line: { color: th.surface, width: 1 } } },
      lineTrace('Background', t, bg, th.series(0), { line: { width: 1.6, dash: 'dot' } }),
      lineTrace('Analysis', t, an, th.series(2), { line: { width: 1.8 } }),
    ], layout(th, { x: { title: 'Time (days)' }, y: { title: 'Z′ equivalent (m)' } }), { xTitle: 'day' });
  }

  function drawAll() { drawErrors(); drawMaps(); drawStation(); }

  return {
    async activate() { if (!A) { await load(); drawAll(); } else { errC.resize(); stC.resize(); maps.forEach((q) => q.mv.draw()); } },
    refresh() { drawAll(); },
    destroy() { errC.purge(); stC.purge(); maps.forEach((q) => q.mv.destroy()); },
  };
}
