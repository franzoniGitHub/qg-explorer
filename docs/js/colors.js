// Colour ramps built in OKLCH, plus the categorical palette and theme helpers.
// Follows three rules: sequential = one hue, light -> dark; diverging = two hues around a neutral
// midpoint with equal steps per arm; categorical colours are assigned in a fixed order and
// follow the entity (upper layer is always blue, lower layer always orange, ...).

function oklchToSrgb(L, C, hDeg) {
  const h = (hDeg * Math.PI) / 180;
  const a = C * Math.cos(h), b = C * Math.sin(h);
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.291485548 * b;
  const l = l_ ** 3, m = m_ ** 3, s = s_ ** 3;
  const lin = [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
  return lin;
}

function gamma(c) { return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055; }

/** OKLCH -> [r,g,b] 0..255, reducing chroma until the colour is inside the sRGB gamut. */
export function oklch(L, C, h) {
  let c = C;
  for (let i = 0; i < 40; i++) {
    const lin = oklchToSrgb(L, c, h);
    if (lin.every((v) => v >= -0.0005 && v <= 1.0005)) {
      return lin.map((v) => Math.round(255 * Math.min(1, Math.max(0, gamma(Math.min(1, Math.max(0, v)))))));
    }
    c *= 0.94;
  }
  const v = Math.round(255 * gamma(Math.min(1, Math.max(0, L ** 3))));
  return [v, v, v];
}

export const hex = (rgb) => '#' + rgb.map((v) => v.toString(16).padStart(2, '0')).join('');

const BLUE = 255, RED = 28;

/** Diverging colour at t in [-1, 1]  (negative = blue, positive = red, 0 = neutral). */
export function divergingColor(t, dark = false) {
  const a = Math.min(1, Math.abs(t));
  const e = Math.pow(a, 0.85);
  const hue = t < 0 ? BLUE : RED;
  if (!dark) return oklch(0.955 - 0.56 * e, 0.185 * Math.pow(a, 0.9) + 0.004, hue);
  return oklch(0.31 + 0.5 * e, 0.17 * Math.pow(a, 0.9) + 0.004, hue);
}

/** Sequential colour at t in [0, 1] (single hue). */
export function sequentialColor(t, dark = false, hue = BLUE) {
  const a = Math.min(1, Math.max(0, t));
  if (!dark) return oklch(0.965 - 0.62 * Math.pow(a, 0.9), 0.14 * Math.pow(a, 0.8) + 0.004, hue);
  return oklch(0.27 + 0.55 * Math.pow(a, 0.9), 0.14 * Math.pow(a, 0.8) + 0.01, hue);
}

/** 256-entry lookup table (Uint8ClampedArray rgb triplets) for a map colour scale. */
export function lut(kind, dark = false, n = 256) {
  const out = new Uint8ClampedArray(n * 3);
  for (let i = 0; i < n; i++) {
    const x = i / (n - 1);
    const c = kind === 'diverging' ? divergingColor(2 * x - 1, dark) : sequentialColor(x, dark);
    out[3 * i] = c[0]; out[3 * i + 1] = c[1]; out[3 * i + 2] = c[2];
  }
  return out;
}

/** Plotly colorscale array from the same ramps. */
export function plotlyScale(kind, dark = false, n = 11) {
  return Array.from({ length: n }, (_, i) => {
    const x = i / (n - 1);
    const c = kind === 'diverging' ? divergingColor(2 * x - 1, dark) : sequentialColor(x, dark);
    return [x, hex(c)];
  });
}

// ---- categorical palette (fixed order, light / dark steps) -------------------------------------
const SERIES = [
  ['#2a78d6', '#3987e5'], // blue
  ['#eb6834', '#d95926'], // orange
  ['#1baf7a', '#199e70'], // aqua
  ['#eda100', '#c98500'], // yellow
  ['#e87ba4', '#d55181'], // magenta
  ['#008300', '#008300'], // green
  ['#4a3aa7', '#9085e9'], // violet
  ['#e34948', '#e66767'], // red
];

export function isDark() {
  const t = document.documentElement.dataset.theme;
  if (t === 'dark') return true;
  if (t === 'light') return false;
  return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
}

/** Colours for charts, read from the active theme. `series(i)` is the i-th categorical colour. */
export function theme() {
  const dark = isDark();
  const css = getComputedStyle(document.documentElement);
  const v = (name, fallback) => (css.getPropertyValue(name).trim() || fallback);
  return {
    dark,
    surface: v('--surface', dark ? '#151b22' : '#fbfcfd'),
    ink: v('--ink', dark ? '#e7edf3' : '#17212b'),
    ink2: v('--ink-2', dark ? '#b4bfca' : '#4b5866'),
    muted: v('--muted', dark ? '#7d8996' : '#6d7a87'),
    grid: v('--grid', dark ? '#262f39' : '#e3e8ed'),
    axis: v('--axis', dark ? '#3a4552' : '#c3ccd5'),
    accent: v('--accent', dark ? '#a99bff' : '#5b3cc4'),
    series: (i) => SERIES[i % SERIES.length][dark ? 1 : 0],
    // named roles: colour follows the entity, never its rank
    upper: SERIES[0][dark ? 1 : 0],
    lower: SERIES[1][dark ? 1 : 0],
    third: SERIES[2][dark ? 1 : 0],
    fourth: SERIES[3][dark ? 1 : 0],
    violet: SERIES[6][dark ? 1 : 0],
    red: SERIES[7][dark ? 1 : 0],
    fontSans: v('--font-sans', 'system-ui, sans-serif'),
  };
}
