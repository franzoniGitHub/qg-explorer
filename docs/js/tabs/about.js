// About: what this dataset is, its parameters and what linear theory says about it.

import { el, fmt } from '../util.js';
import { panel, prose, stat, stats, cols } from '../ui.js';
import { linearParams, fastestMode, mode } from '../linear.js';

const DAY = 86400;

export default function create({ ds }, root) {
  const m = ds.model, mf = ds.manifest;
  const lp = linearParams(m);
  const dk = (2 * Math.PI) / m.Lx;
  const best = fastestMode(lp, (m.nx / 2) * dk, 4000);
  const bm = mode(lp, best.k, 0);
  const mBest = best.k / dk;

  const row = (k, v) => [el('dt', {}, k), el('dd', {}, v)];
  const params = el('dl', { class: 'kv' },
    ...row('Grid', `${m.nx} × ${m.ny}`),
    ...row('Domain', `${fmt(m.Lx / 1e6, 3)} × ${fmt(m.Ly / 1e6, 3)} ·10³ km`),
    ...row('Deformation radius Ld', `${fmt(m.Ld / 1e3, 3)} km`),
    ...row('Layer depth ratio δ = H₁/(H₁+H₂)', fmt(m.delta, 3)),
    ...row('β', `${fmt(m.beta * 1e11, 3)} ×10⁻¹¹ m⁻¹ s⁻¹`),
    ...row('Upper / lower mean wind', `${fmt(m.U1, 3)} / ${fmt(m.U2, 3)} m s⁻¹`),
    ...row('Shear U₁ − U₂', `${fmt(m.dU, 3)} m s⁻¹`),
    ...row('Critical shear β/F₂', `${fmt(lp.beta / lp.F2, 3)} m s⁻¹`),
    ...row('Shear / critical', `${fmt(m.dU / (lp.beta / lp.F2), 3)}`),
    ...row('Ekman drag time 1/r', `${fmt(1 / (m.r_ek * DAY), 3)} days`),
    ...row('Time step', `${m.dt} s`),
    ...row('Hyperviscosity (∇⁸) e-fold at cut-off', `${fmt(m.filter_tau / 3600, 3)} h`),
  );

  const c = ds.control;
  const products = el('dl', { class: 'kv' },
    ...row('Control run', `${c.n} fields every ${c.step_hours} h (${fmt(c.n * c.step_hours / 24, 4)} days)${c.spinup_days !== undefined ? `, after ${c.spinup_days} days of spin-up` : ''}`),
    ...row('Parameter sweep', ds.has('sweep') ? 'available' : 'not in this dataset'),
    ...row('Ensemble forecasts', ds.has('ensemble') ? `${mf.ensemble.members} members, ${mf.ensemble.lead_days}-day lead` : 'not in this dataset'),
    ...row('Observing-system experiments', ds.has('assimilation') ? `${mf.assimilation.members} members, ${mf.assimilation.cycles} cycles of ${mf.assimilation.cycle_hours} h` : 'not in this dataset'),
    ...row('Created', mf.created || '–'),
    ...row('Generator', mf.generator || '–'),
  );

  root.append(el('div', { class: 'stack' },
    panel(ds.title, '', prose(`<p>${mf.description || ''}</p>
      <p>The fields are the streamfunctions of a two-layer quasi-geostrophic channel on a doubly periodic β-plane, driven by a fixed vertical shear and damped by Ekman drag on the lower layer. Baroclinic waves grow on the shear, saturate, occlude and decay, and the cycle repeats: a small-scale storm track. Use the tabs to look at the fields, test linear theory against the waves, quantify the statistics and energy budget, and examine how the system responds to its parameters, to errors in the initial state and to observations.</p>`)),
    panel('Linear theory for this basic state', 'Normal modes of the two-layer system with the imposed winds and drag',
      stats(
        stat('Shear / critical', fmt(m.dU / (lp.beta / lp.F2), 3)),
        stat('Fastest-growing zonal wavenumber', `m = ${fmt(mBest, 3)}`),
        stat('Wavelength', `${fmt((2 * Math.PI / best.k) / 1e3, 4)} km`),
        stat('e-folding time', best.sigma > 0 ? `${fmt(1 / (best.sigma * DAY), 3)} days` : 'stable'),
        stat('Phase speed', `${fmt(bm.c, 3)} m s⁻¹`),
      )),
    cols(panel('Parameters', '', params), panel('Data products', '', products)),
  ));

  return { refresh() {} };
}
