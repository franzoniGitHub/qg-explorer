// Shared interface pieces: Plotly theming, chart panels with a table view, form controls and the
// loading overlay.  No physics in here.

import { el, fmt } from './util.js';
import { theme } from './colors.js';

// ---- Plotly ---------------------------------------------------------------------------------
function axis(th, a = {}) {
  const { title, ...rest } = a;
  if (rest.type === 'log') { if (rest.dtick === undefined) rest.dtick = 1; if (!rest.exponentformat) rest.exponentformat = 'power'; }
  return {
    title: title ? { text: title, standoff: 8, font: { size: 12, color: th.ink2 } } : undefined,
    gridcolor: th.grid, gridwidth: 1, zerolinecolor: th.axis, zerolinewidth: 1,
    linecolor: th.axis, tickcolor: th.axis, ticks: 'outside', ticklen: 3,
    tickfont: { size: 11, color: th.muted }, automargin: true, showline: true, mirror: false,
    ...rest,
  };
}

/** Plotly layout in the active theme.  `x` and `y` are axis overrides ({title, type, range, ...}). */
export function layout(th, { x = {}, y = {}, legend = true, margin = {}, hover = 'x unified', ...rest } = {}) {
  return {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: th.fontSans, size: 12, color: th.ink2 },
    margin: { l: 56, r: 14, t: legend ? 30 : 8, b: 44, ...margin },
    xaxis: axis(th, x), yaxis: axis(th, y),
    showlegend: legend,
    legend: { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', font: { size: 12, color: th.ink2 }, bgcolor: 'rgba(0,0,0,0)' },
    hovermode: hover,
    hoverlabel: { bgcolor: th.surface, bordercolor: th.axis, font: { family: th.fontSans, size: 12, color: th.ink } },
    ...rest,
  };
}

export const PLOT_CONFIG = {
  displaylogo: false, responsive: true, displayModeBar: 'hover',
  modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toggleSpikelines'],
};

/** Thin line trace. */
export function lineTrace(name, x, y, color, extra = {}) {
  const { line, ...rest } = extra;
  return { type: 'scatter', mode: 'lines', name, x, y, line: { color, width: 2, ...line }, ...rest };
}

/** Horizontal or vertical reference line as a shape, drawn behind the data. */
export function refLine(th, { x, y, dash = 'dot', color }) {
  const c = color || th.muted;
  if (x !== undefined) return { type: 'line', xref: 'x', yref: 'paper', x0: x, x1: x, y0: 0, y1: 1, line: { color: c, width: 1.2, dash }, layer: 'below' };
  return { type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: y, y1: y, line: { color: c, width: 1.2, dash }, layer: 'below' };
}

// ---- table view ------------------------------------------------------------------------------
function tableFromTraces(traces, xTitle) {
  const ts = traces.filter((t) => t.x && t.y && !(t.meta && t.meta.table === false) && t.type !== 'heatmap');
  if (!ts.length) return null;
  const same = ts.every((t) => t.x.length === ts[0].x.length && t.x[0] === ts[0].x[0] && t.x[t.x.length - 1] === ts[0].x[ts[0].x.length - 1]);
  const blocks = same ? [ts] : ts.map((t) => [t]);
  const wrap = el('div');
  const LIMIT = 600;
  for (const block of blocks) {
    const n = block[0].x.length;
    const head = el('tr', {}, el('th', {}, xTitle || 'x'), ...block.map((t) => el('th', {}, t.name || 'series')));
    const body = [];
    for (let i = 0; i < Math.min(n, LIMIT); i++) {
      body.push(el('tr', {}, el('td', {}, fmt(Number(block[0].x[i]), 5)), ...block.map((t) => el('td', {}, fmt(Number(t.y[i]), 4)))));
    }
    wrap.append(el('div', { style: 'max-height:320px;overflow:auto;margin-top:6px' },
      el('table', { class: 'data' }, el('thead', {}, head), el('tbody', {}, ...body))));
    if (n > LIMIT) wrap.append(el('p', { class: 'hint' }, `First ${LIMIT} of ${n} rows.`));
  }
  return wrap;
}

/**
 * A panel holding one Plotly chart with a "Table" switch.
 * chart.set(traces, layoutObject, { xTitle, hover }) draws it; layout is rebuilt by the caller
 * on theme changes.
 */
export function chart({ title, sub = '', height = 260, tall = false } = {}) {
  const plotDiv = el('div', { class: 'plot' + (tall ? ' tall' : '') });
  plotDiv.style.height = height + 'px';
  const tableDiv = el('div', { hidden: true });
  const toggle = el('button', { class: 'btn', type: 'button', 'aria-pressed': 'false', style: 'padding:2px 8px;font-weight:400' }, 'Table');
  const subEl = el('span', { class: 'sub' }, sub);
  const root = el('section', { class: 'panel' },
    el('header', {}, el('div', {}, el('h2', {}, title), subEl), toggle), plotDiv, tableDiv);
  let last = null, showTable = false;

  function draw() {
    if (!last) return;
    const tbl = showTable ? tableFromTraces(last.traces, last.opts.xTitle) : null;
    toggle.hidden = last.opts.table === false || !last.traces.some((t) => t.x && t.y && t.type !== 'heatmap');
    if (showTable && tbl) {
      plotDiv.hidden = true; tableDiv.hidden = false; tableDiv.replaceChildren(tbl);
    } else {
      plotDiv.hidden = false; tableDiv.hidden = true;
      Plotly.react(plotDiv, last.traces, last.layout, PLOT_CONFIG);
    }
  }
  toggle.addEventListener('click', () => {
    showTable = !showTable;
    toggle.setAttribute('aria-pressed', String(showTable));
    toggle.textContent = showTable ? 'Chart' : 'Table';
    draw();
  });
  return {
    el: root, div: plotDiv,
    set(traces, lay, opts = {}) { last = { traces, layout: lay, opts }; draw(); },
    setSub(t) { subEl.textContent = t; },
    resize() { if (!plotDiv.hidden && plotDiv.data) Plotly.Plots.resize(plotDiv); },
    purge() { try { Plotly.purge(plotDiv); } catch { /* ignore */ } },
  };
}

// ---- panels and layout -------------------------------------------------------------------------
export function panel(title, sub, ...body) {
  return el('section', { class: 'panel' }, title ? el('header', {}, el('div', {}, el('h2', {}, title), sub ? el('span', { class: 'sub' }, sub) : null)) : null, ...body);
}

/** A data table that scrolls sideways instead of widening the page on narrow screens. */
export function dataTable(...children) {
  return el('div', { class: 'tablewrap' }, el('table', { class: 'data' }, ...children));
}

export const prose = (html) => el('div', { class: 'prose', html });

// ---- controls ---------------------------------------------------------------------------------
let uid = 0;
const nextId = (p) => `${p}-${++uid}`;

export function selectCtl({ label, options, value, onchange }) {
  const id = nextId('sel');
  const sel = el('select', { id });
  for (const o of options) sel.append(el('option', { value: o.value }, o.label));
  if (value !== undefined) sel.value = value;
  sel.addEventListener('change', () => onchange && onchange(sel.value));
  return {
    el: el('div', { class: 'ctl' }, el('label', { for: id }, label), sel),
    get value() { return sel.value; }, set value(v) { sel.value = v; },
    setOptions(opts, v) { sel.replaceChildren(...opts.map((o) => el('option', { value: o.value }, o.label))); if (v !== undefined) sel.value = v; },
  };
}

export function sliderCtl({ label, min, max, step = 1, value, format = (v) => String(v), oninput }) {
  const id = nextId('rng');
  const input = el('input', { id, type: 'range', min, max, step, value });
  const val = el('span', { class: 'val' }, format(Number(value)));
  input.addEventListener('input', () => { val.textContent = format(Number(input.value)); oninput && oninput(Number(input.value)); });
  return {
    el: el('div', { class: 'ctl' }, el('label', { for: id }, label, val), input), input,
    get value() { return Number(input.value); },
    set value(v) { input.value = v; val.textContent = format(Number(input.value)); },
    setRange(lo, hi, st) { input.min = lo; input.max = hi; if (st) input.step = st; },
  };
}

export function segCtl({ label, options, value, onchange }) {
  let cur = value ?? options[0].value;
  const wrap = el('div', { class: 'seg', role: 'group', 'aria-label': label });
  const btns = options.map((o) => {
    const b = el('button', { type: 'button', 'aria-pressed': String(o.value === cur) }, o.label);
    b.addEventListener('click', () => { set(o.value); onchange && onchange(cur); });
    wrap.append(b);
    return b;
  });
  function set(v) { cur = v; btns.forEach((b, i) => b.setAttribute('aria-pressed', String(options[i].value === v))); }
  return { el: el('div', { class: 'ctl' }, el('span', { class: 'lbl' }, label), wrap), get value() { return cur; }, set value(v) { set(v); } };
}

export function checkCtl({ label, checked = false, onchange }) {
  const input = el('input', { type: 'checkbox' });
  input.checked = checked;
  input.addEventListener('change', () => onchange && onchange(input.checked));
  return { el: el('label', { class: 'check' }, input, label), get value() { return input.checked; }, set value(v) { input.checked = v; } };
}

export const group = (title, ...kids) => el('div', { class: 'ctl-group' }, title ? el('h3', {}, title) : null, ...kids);

/** rail + content layout */
export function layoutRail(rail, content) {
  return el('div', { class: 'layout' }, el('aside', { class: 'rail' }, ...rail), el('div', { class: 'stack' }, ...content));
}

export const cols = (...kids) => el('div', { class: 'cols' }, ...kids);
export const cols3 = (...kids) => el('div', { class: 'cols three' }, ...kids);

export function stat(label, value) { return el('div', { class: 'stat' }, el('span', {}, label), el('span', {}, value)); }
export function stats(...items) { return el('div', { class: 'stats' }, ...items); }

export function notAvailable(what, how) {
  return el('div', { class: 'panel empty' }, el('p', {}, `This dataset has no ${what}.`),
    how ? el('p', { class: 'hint', html: how }) : null);
}

// ---- loading overlay ----------------------------------------------------------------------------
export const progress = {
  show(text) {
    document.getElementById('loading').hidden = false;
    document.getElementById('loading-text').textContent = text;
    document.getElementById('loading-bar').style.width = '0%';
  },
  set(f, text) {
    document.getElementById('loading-bar').style.width = `${Math.round(100 * Math.min(1, Math.max(0, f)))}%`;
    if (text) document.getElementById('loading-text').textContent = text;
  },
  hide() { document.getElementById('loading').hidden = true; },
};

/** Run `fn(setFraction)` behind the overlay.  Always hides the overlay. */
export async function busy(text, fn) {
  progress.show(text);
  await new Promise((r) => requestAnimationFrame(() => r()));
  try { return await fn((f) => progress.set(f)); } finally { progress.hide(); }
}

export function errorBox(err) {
  console.error(err);
  return el('div', { class: 'error', role: 'alert' }, el('strong', {}, 'Something went wrong. '), String(err && err.message ? err.message : err));
}

export { theme };
