// Small helpers shared by the UI modules.

/** 1-2-5 "nice" step so that about `count` steps cover `range`. */
export function niceStep(range, count = 6) {
  if (!(range > 0)) return 1;
  const raw = range / count;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const f = raw / p;
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p;
}

export function niceTicks(lo, hi, count = 5) {
  const step = niceStep(hi - lo, count);
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

/** Compact number formatting for readouts and tick labels. */
export function fmt(v, digits = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) {
    const e = Math.floor(Math.log10(a));
    return `${(v / 10 ** e).toFixed(digits - 1)}×10${sup(e)}`;
  }
  const d = Math.max(0, digits - 1 - Math.floor(Math.log10(a)));
  return v.toFixed(Math.min(d, 6));
}

const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
export function sup(s) { return String(s).split('').map((c) => SUP[c] ?? c).join(''); }

/** Tick-label formatting (no exponent) with the step in mind. */
export function tickLabel(v, step) {
  const decimals = step >= 1 ? 0 : Math.min(4, Math.ceil(-Math.log10(step)));
  return v.toFixed(decimals);
}

/** Create an element:  el('div', {class: 'x', onclick: fn}, child, 'text', ...) */
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
    else if (k === 'dataset') Object.assign(node.dataset, v);
    else node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return node;
}

export function debounce(fn, ms = 80) {
  let t = null;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

/** Yield to the browser so the UI can repaint. */
export const nextFrame = () => new Promise((r) => requestAnimationFrame(() => r()));

export function clamp(x, lo, hi) { return Math.min(hi, Math.max(lo, x)); }

export function linspace(a, b, n) { return Float64Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1)); }

export function range(n) { return Array.from({ length: n }, (_, i) => i); }
