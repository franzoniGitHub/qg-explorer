// Energetics: reservoirs, the conversion/dissipation budget and wavenumber spectra.

import { el, fmt } from '../util.js';
import { keSpectrum, shellSum, Snapshot } from '../physics.js';
import { linearParams, fastestMode } from '../linear.js';
import { mean as avg } from '../stats.js';
import { theme } from '../colors.js';
import { chart, layout, lineTrace, layoutRail, group, segCtl, busy, stats, stat, cols, panel, dataTable } from '../ui.js';

const DAY = 86400;

export default function create({ ds }, root) {
  const m = ds.model, nt = ds.nt, grid = ds.grid, dtS = ds.control.step_hours * 3600;
  const days = Array.from({ length: nt }, (_, t) => ds.timeDays(t));
  const me = ds.manifest.control.model_energetics || {};
  let diag = null, spectra = null;

  const modeCtl = segCtl({ label: 'Spectrum axes', options: [{ value: 'loglog', label: 'Log–log' }, { value: 'semilog', label: 'Log–linear' }], value: 'loglog', onchange: () => draw() });
  const rail = [group(null, modeCtl.el), el('p', { class: 'hint' }, 'Energies are per unit mass (m² s⁻²), averaged over the domain and weighted by layer depth. “Eddy” means the departure from the zonal mean.')];

  const res = chart({ title: 'Energy reservoirs', sub: 'Zonal-mean and eddy, kinetic and available potential', height: 300 });
  const bud = chart({ title: 'Energy budget', sub: 'Conversion from the imposed shear against dissipation (10⁻⁴ m² s⁻³)', height: 300 });
  const spec = chart({ title: 'Spectra against total wavenumber', sub: 'Time-mean energy in each wavenumber shell', height: 340 });
  const tableBox = el('div');
  root.append(layoutRail(rail, [panel('Time-mean budget', '', tableBox), cols(res.el, bud.el), spec.el]));

  async function compute() {
    diag = await busy('Computing energy partition…', (prog) => ds.diagnostics(prog));
    spectra = await busy('Computing spectra…', async (prog) => {
      const n = grid.nShell, k1 = new Float64Array(n), k2 = new Float64Array(n), ap = new Float64Array(n);
      const step = Math.max(1, Math.floor(nt / 150));
      let cnt = 0;
      for (let t = 0; t < nt; t += step) {
        const [p1, p2] = ds.getPsi(t);
        const s = new Snapshot(m, ds.spec, p1, p2);
        keSpectrum(grid, s.hat(0), k1); keSpectrum(grid, s.hat(1), k2);
        const tau = s.hatCombo(1, -1);
        const a = shellSum(grid, tau, 0.5 * m.G);
        for (let k = 0; k < n; k++) ap[k] += a[k];
        cnt++;
        if (cnt % 20 === 0) prog(t / nt);
      }
      for (let k = 0; k < n; k++) { k1[k] /= cnt; k2[k] /= cnt; ap[k] /= cnt; }
      return { k1, k2, ap };
    });
  }

  function draw() {
    if (!diag) return;
    const th = theme(), s = diag.scalars, d = m.delta;
    const EKE = Array.from(s.EKE1, (v, t) => d * v + (1 - d) * s.EKE2[t]);
    const ZKE = Array.from(s.ZKE1, (v, t) => d * v + (1 - d) * s.ZKE2[t]);
    const L = (name, y, i, ex = {}) => lineTrace(name, days, y, th.series(i), { line: { width: 1.8 }, ...ex });
    res.set([L('Eddy kinetic', EKE, 0), L('Eddy potential', Array.from(s.EAPE), 1), L('Zonal-mean kinetic', ZKE, 2), L('Zonal-mean potential', Array.from(s.ZAPE), 3)],
      layout(th, { x: { title: 'Time (days)' }, y: { title: 'Energy (m² s⁻²)' } }), { xTitle: 'day' });

    // budget
    const CB = me.C_B, DE = me.D_E, DH = me.D_h, E = me.E;
    if (CB && DE && E) {
      const D = DE.map((v, t) => v + (DH ? DH[t] : 0));
      const dE = Array.from(E, (_, t) => (t === 0 ? NaN : t === nt - 1 ? NaN : (E[t + 1] - E[t - 1]) / (2 * dtS)));
      const sc = 1e4;
      bud.set([
        L('Conversion C', CB.map((v) => v * sc), 0), L('Dissipation D', D.map((v) => v * sc), 1),
        L('dE/dt', dE.map((v) => v * sc), 4, { line: { width: 1.4, dash: 'dot', color: th.muted } }),
      ], layout(th, { x: { title: 'Time (days)' }, y: { title: '10⁻⁴ m² s⁻³', zeroline: true } }), { xTitle: 'day' });

      const row = (name, a) => { const mu = avg(a), sd = Math.sqrt(avg(a.map((v) => (v - mu) ** 2))); return [name, mu * sc, sd * sc]; };
      const mC = avg(CB), mD = avg(D);
      const rows = [row('Conversion from mean shear, C', CB), row('Ekman drag, D_E', DE), ...(DH ? [row('Hyperviscous filter, D_h', DH)] : []), row('Total dissipation, D', D)];
      const head = el('tr', {}, el('th', {}, 'Term (10⁻⁴ m² s⁻³)'), el('th', {}, 'Mean'), el('th', {}, 'σ in time'), el('th', {}, '% of C'));
      const body = rows.map(([n, mu, sd]) => el('tr', {}, el('td', {}, n), el('td', {}, fmt(mu, 3)), el('td', {}, fmt(sd, 3)), el('td', {}, fmt((100 * mu) / (mC * sc), 3))));
      const residual = mC - mD;
      tableBox.replaceChildren(
        dataTable( el('thead', {}, head), el('tbody', {}, ...body)),
        stats(
          stat('Mean total energy E', `${fmt(avg(E), 3)} m² s⁻²`),
          stat('Energy turnover time E / D', `${fmt(avg(E) / mD / DAY, 3)} days`),
          stat('Budget imbalance C − D', `${fmt((100 * residual) / mC, 2)} % of C`),
          stat('Eddy share of kinetic energy', `${fmt(100 * avg(EKE) / (avg(EKE) + avg(ZKE)), 3)} %`),
        ));
    } else {
      bud.set([], layout(th, { legend: false }), { table: false });
      bud.setSub('The dataset does not contain the budget terms.');
    }

    // spectra
    if (spectra) {
      const ks = Array.from({ length: grid.kCut }, (_, k) => k + 1);
      const log = modeCtl.value === 'loglog';
      const k0 = 6, y0 = spectra.k1[k0 - 1];
      const slope = (p) => ks.map((k) => y0 * (k / k0) ** p);
      const lp = linearParams(m);
      const kd = grid.Lx / (2 * Math.PI * m.Ld);
      const bestK = fastestMode(lp, (grid.nx / 2) * ((2 * Math.PI) / grid.Lx), 4000).k / ((2 * Math.PI) / grid.Lx);
      const vline = (x, text) => ({ shape: { type: 'line', xref: 'x', yref: 'paper', x0: x, x1: x, y0: 0, y1: 1, line: { color: th.muted, width: 1, dash: 'dot' } }, ann: { x: log ? Math.log10(x) : x, xref: 'x', y: 0.98, yref: 'paper', text, showarrow: false, yanchor: 'top', xanchor: 'left', font: { size: 11, color: th.muted } } });
      const marks = [vline(kd, 'k = 1/L_d'), ...(bestK > 0 ? [vline(bestK, 'fastest linear mode')] : [])];
      spec.set([
        lineTrace('Kinetic, upper layer', ks, Array.from(spectra.k1.subarray(0, ks.length)), th.upper),
        lineTrace('Kinetic, lower layer', ks, Array.from(spectra.k2.subarray(0, ks.length)), th.lower),
        lineTrace('Available potential', ks, Array.from(spectra.ap.subarray(0, ks.length)), th.third),
        ...(log ? [lineTrace('k⁻³', ks, slope(-3), th.muted, { line: { dash: 'dash', width: 1.2 }, meta: { table: false } }), lineTrace('k⁻⁵ᐟ³', ks, slope(-5 / 3), th.muted, { line: { dash: 'dot', width: 1.2 }, meta: { table: false } })] : []),
      ], layout(th, { x: { title: 'Total wavenumber k (units of 2π / L)', type: log ? 'log' : 'linear', dtick: log ? 'D2' : undefined }, y: { title: 'Energy per shell (m² s⁻²)', type: 'log', exponentformat: 'power', dtick: 1 },
        shapes: marks.map((q) => q.shape), annotations: marks.map((q) => q.ann) }), { xTitle: 'k' });
    }
  }

  return {
    async activate() { if (!diag) { await compute(); draw(); } else [res, bud, spec].forEach((c) => c.resize()); },
    refresh() { draw(); },
    destroy() { [res, bud, spec].forEach((c) => c.purge()); },
  };
}
