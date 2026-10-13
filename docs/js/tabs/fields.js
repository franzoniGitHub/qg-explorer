// Fields: animated maps of the model state with click-to-pick time series.

import { el, fmt, niceStep, clamp } from '../util.js';
import { MapView } from '../mapview.js';
import { VARIABLES, LAYER_LABELS, Snapshot } from '../physics.js';
import { symmetricLimit, quantileSorted, sortedCopy } from '../stats.js';
import { theme } from '../colors.js';
import { chart, layout, lineTrace, refLine, layoutRail, group, selectCtl, segCtl, checkCtl, sliderCtl, busy, panel, cols } from '../ui.js';

function niceCeil(x) {
  if (!(x > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(x));
  const f = x / p;
  return [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].find((v) => v >= f - 1e-9) * p;
}

const LAYERS = [
  { value: 'both', label: 'Upper and lower layer' },
  { value: 'upper', label: LAYER_LABELS.upper },
  { value: 'lower', label: LAYER_LABELS.lower },
  { value: 'barotropic', label: LAYER_LABELS.barotropic },
  { value: 'baroclinic', label: LAYER_LABELS.baroclinic },
];

export default function create({ ds }, root) {
  const grid = ds.grid, nt = ds.nt, N = grid.N;
  const st = { variable: 'Z', layer: 'both', total: false, contour: true, t: 0, playing: false, fps: 12, pick: null };
  const rangeCache = new Map();
  let maps = [], raf = 0, lastFrame = 0;

  // ---- colour range from a sample of the record --------------------------------------------------
  function sample(variable, layerKey, total) {
    const idx = Array.from({ length: 14 }, (_, k) => Math.round((k * (nt - 1)) / 13));
    const vals = [];
    for (const t of idx) {
      const [p1, p2] = ds.getPsi(t);
      const f = new Snapshot(ds.model, ds.spec, p1, p2).field(variable, layerKey, { total });
      for (let p = 0; p < N; p += 3) vals.push(f[p]);
    }
    return Float64Array.from(vals);
  }
  function rangeFor(variable, layerKeys, total) {
    const key = `${variable}|${layerKeys.join()}|${total}`;
    if (rangeCache.has(key)) return rangeCache.get(key);
    const info = VARIABLES[variable];
    const diverging = info.diverging && !(variable === 'u' && total);
    const all = layerKeys.map((l) => sample(variable, l, total));
    let r;
    if (diverging) {
      const lim = niceCeil(Math.max(...all.map((a) => symmetricLimit(a, 0.995))) * info.scale);
      r = { kind: 'diverging', vmin: -lim, vmax: lim };
    } else if (variable === 'u' && total) {
      const s = sortedCopy(Float64Array.from(all.flatMap((a) => Array.from(a))));
      const lo = quantileSorted(s, 0.003) * info.scale, hi = quantileSorted(s, 0.997) * info.scale;
      const step = niceStep(hi - lo, 5);
      r = { kind: 'sequential', vmin: Math.floor(lo / step) * step, vmax: Math.ceil(hi / step) * step };
    } else {
      const s = sortedCopy(Float64Array.from(all.flatMap((a) => Array.from(a))));
      r = { kind: 'sequential', vmin: 0, vmax: niceCeil(quantileSorted(s, 0.997) * info.scale) };
    }
    rangeCache.set(key, r);
    return r;
  }
  const layerKeys = () => (st.layer === 'both' ? ['upper', 'lower'] : [st.layer]);

  // ---- controls -------------------------------------------------------------------------------------
  const varCtl = selectCtl({ label: 'Variable', options: Object.entries(VARIABLES).map(([value, v]) => ({ value, label: v.label })), value: st.variable,
    onchange: (v) => { st.variable = v; totalCtl.el.hidden = !(v === 'u' || v === 'speed'); if (v === 'speed') st.total = true; else if (v !== 'u') st.total = false; totalCtl.value = st.total; series = null; redraw(); drawSeries(); } });
  const layerCtl = selectCtl({ label: 'Layer', options: LAYERS, value: st.layer, onchange: (v) => { st.layer = v; buildMaps(); series = null; drawSeries(); } });
  const totalCtl = checkCtl({ label: 'Add the imposed mean wind', checked: false, onchange: (v) => { st.total = v; redraw(); drawSeries(); } });
  totalCtl.el.hidden = true;
  const contourCtl = checkCtl({ label: 'Contour Z′ of each layer', checked: true, onchange: (v) => { st.contour = v; redraw(); } });
  const fpsCtl = selectCtl({ label: 'Animation speed', options: [{ value: 4, label: 'Slow' }, { value: 12, label: 'Medium' }, { value: 30, label: 'Fast' }], value: 12, onchange: (v) => { st.fps = Number(v); } });
  const pickInfo = el('p', { class: 'hint' }, 'Click a map to see the time series at that point.');

  const rail = [
    group(null, varCtl.el, layerCtl.el, totalCtl.el),
    group(null, contourCtl.el, fpsCtl.el),
    group('Point', pickInfo),
    el('p', { class: 'hint' }, 'With both layers shown the two maps share one colour scale, so the upper layer’s larger amplitude is visible. Dashed contours are negative.'),
  ];

  // ---- maps ---------------------------------------------------------------------------------------------
  const mapsBox = el('div', { class: 'cols' });
  function buildMaps() {
    maps.forEach((m) => m.destroy());
    mapsBox.replaceChildren();
    maps = layerKeys().map((key) => {
      const box = el('div');
      mapsBox.append(box);
      const mv = new MapView(box, { grid, onPick: (i, j) => pick(i, j) });
      mv.layerKey = key;
      mv.setPick(st.pick);
      return mv;
    });
    redraw();
  }

  function redraw() {
    const snap = ds.snapshot(st.t);
    const info = VARIABLES[st.variable];
    const rng = rangeFor(st.variable, layerKeys(), st.total);
    for (const mv of maps) {
      const f = snap.field(st.variable, mv.layerKey, { total: st.total });
      const data = { values: f, scale: info.scale, vmin: rng.vmin, vmax: rng.vmax, kind: rng.kind, unit: info.unit, name: info.short };
      if (st.contour) {
        const z = snap.field('Z', mv.layerKey);
        const zr = rangeFor('Z', [mv.layerKey], false);
        data.contour = { values: z, step: niceStep(2 * zr.vmax, 9) };
      }
      mv.title.textContent = `${mv.layerKey === 'upper' ? 'Upper layer' : mv.layerKey === 'lower' ? 'Lower layer' : LAYER_LABELS[mv.layerKey]} · ${info.short}`;
      mv.setData(data);
    }
    clock.textContent = `day ${fmt(ds.timeDays(st.t), 4)}`;
    marker();
  }

  // ---- timeline ---------------------------------------------------------------------------------------------
  const clock = el('span', { class: 'clock' });
  const playBtn = el('button', { class: 'btn', type: 'button', 'aria-label': 'Play animation', style: 'min-width:74px' }, 'Play');
  const slider = el('input', { type: 'range', min: 0, max: nt - 1, step: 1, value: 0, 'aria-label': 'Time' });
  const E = (ds.manifest.control.model_energetics || {}).E || [];
  const svgNS = 'http://www.w3.org/2000/svg';
  const spark = document.createElementNS(svgNS, 'svg');
  spark.setAttribute('viewBox', '0 0 100 40'); spark.setAttribute('preserveAspectRatio', 'none');
  const path = document.createElementNS(svgNS, 'polyline');
  const cursor = document.createElementNS(svgNS, 'line');
  function sparkline() {
    if (E.length < 2) return;
    let lo = Infinity, hi = -Infinity;
    for (const v of E) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const pts = E.map((v, k) => `${((100 * k) / (E.length - 1)).toFixed(2)},${(36 - (32 * (v - lo)) / (hi - lo || 1)).toFixed(2)}`).join(' ');
    path.setAttribute('points', pts);
    path.setAttribute('fill', 'none'); path.setAttribute('stroke', 'var(--muted)'); path.setAttribute('stroke-width', '1.2'); path.setAttribute('vector-effect', 'non-scaling-stroke');
    cursor.setAttribute('y1', 0); cursor.setAttribute('y2', 40); cursor.setAttribute('stroke', 'var(--accent)'); cursor.setAttribute('stroke-width', '2'); cursor.setAttribute('vector-effect', 'non-scaling-stroke');
    spark.append(path, cursor);
  }
  sparkline();
  function marker() {
    const x = nt > 1 ? (100 * st.t) / (nt - 1) : 0;
    cursor.setAttribute('x1', x); cursor.setAttribute('x2', x);
    slider.value = st.t;
    moveSeriesMarker();
  }
  function setTime(t) { st.t = clamp(Math.round(t), 0, nt - 1); redraw(); }
  slider.addEventListener('input', () => setTime(Number(slider.value)));
  function tick(now) {
    if (!st.playing) return;
    if (now - lastFrame >= 1000 / st.fps) { lastFrame = now; setTime(st.t + 1 >= nt ? 0 : st.t + 1); }
    raf = requestAnimationFrame(tick);
  }
  function play(on) {
    st.playing = on;
    playBtn.textContent = on ? 'Pause' : 'Play';
    playBtn.setAttribute('aria-label', on ? 'Pause animation' : 'Play animation');
    cancelAnimationFrame(raf);
    if (on) raf = requestAnimationFrame(tick);
  }
  playBtn.addEventListener('click', () => play(!st.playing));
  const timelinePanel = el('section', { class: 'panel' },
    el('div', { class: 'play' }, playBtn, clock, el('span', { class: 'sub' }, `${nt} fields, one every ${ds.control.step_hours} h`)),
    el('div', { class: 'timeline', style: 'margin-top:6px' }, spark, slider),
    el('p', { class: 'hint' }, 'The grey line is the total energy of the control run; the purple marker is the time shown.'));

  // ---- picked point ---------------------------------------------------------------------------------------------
  const seriesChart = chart({ title: 'Time series at the picked point', sub: '', height: 250 });
  seriesChart.el.hidden = true;
  let series = null;   // { key, data: [Float32Array...] }

  async function pick(i, j) {
    st.pick = { i, j };
    maps.forEach((m) => m.setPick(st.pick));
    const xk = Math.round(((i + 0.5) * grid.dx) / 1e3), yk = Math.round(((j + 0.5) * grid.dy) / 1e3);
    pickInfo.textContent = `x = ${xk} km, y = ${yk} km`;
    await drawSeries();
  }

  async function drawSeries() {
    if (!st.pick) return;
    const key = `${st.variable}|${st.layer}|${st.total}`;
    if (!series || series.key !== key) {
      const data = await busy('Computing the time series…', async (prog) => {
        const out = [];
        const keys = layerKeys();
        for (let k = 0; k < keys.length; k++) out.push(await ds.fieldSeries(st.variable, keys[k], { total: st.total }, (f) => prog((k + f) / keys.length)));
        return out;
      });
      series = { key, data };
    }
    const info = VARIABLES[st.variable], th = theme();
    const p = st.pick.j * grid.nx + st.pick.i;
    const days = Array.from({ length: nt }, (_, t) => ds.timeDays(t));
    const traces = layerKeys().map((lk, k) => {
      const y = Array.from({ length: nt }, (_, t) => series.data[k][t * N + p] * info.scale);
      const color = lk === 'upper' ? th.upper : lk === 'lower' ? th.lower : th.accent;
      return lineTrace(lk === 'upper' ? 'Upper layer' : lk === 'lower' ? 'Lower layer' : lk, days, y, color, { line: { width: 1.5 }, hovertemplate: `%{y:.3g} ${info.unit}` });
    });
    const lay = layout(th, { x: { title: 'Time (days)' }, y: { title: `${info.short} (${info.unit})` }, legend: traces.length > 1,
      shapes: [refLine(th, { x: ds.timeDays(st.t), dash: 'solid', color: th.accent })] });
    seriesChart.el.hidden = false;
    seriesChart.setSub(pickInfo.textContent);
    seriesChart.set(traces, lay, { xTitle: 'day' });
  }
  let pending = false;
  function moveSeriesMarker() {
    if (!st.pick || seriesChart.el.hidden || !seriesChart.div.layout || pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      const d = ds.timeDays(st.t);
      Plotly.relayout(seriesChart.div, { 'shapes[0].x0': d, 'shapes[0].x1': d });
    });
  }

  // ---- assemble -----------------------------------------------------------------------------------------------
  root.append(layoutRail(rail, [panel(null, null, mapsBox), timelinePanel, seriesChart.el]));
  buildMaps();

  return {
    activate() { maps.forEach((m) => m.draw()); seriesChart.resize(); },
    refresh() { redraw(); drawSeries(); },
    destroy() { play(false); maps.forEach((m) => m.destroy()); seriesChart.purge(); },
  };
}
