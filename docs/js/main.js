// Application shell: dataset picker, local-folder loading, tabs, theme.

import { Dataset, UrlSource, FolderSource } from './dataset.js';
import { el } from './util.js';
import { busy, errorBox } from './ui.js';
import about from './tabs/about.js';
import fields from './tabs/fields.js';
import waves from './tabs/waves.js';
import statistics from './tabs/statistics.js';
import meanflow from './tabs/meanflow.js';
import energetics from './tabs/energetics.js';
import sweep from './tabs/sweep.js';
import predictability from './tabs/predictability.js';
import observations from './tabs/observations.js';

const TABS = [
  { id: 'about', title: 'About', create: about },
  { id: 'fields', title: 'Fields', create: fields },
  { id: 'waves', title: 'Waves and stability', create: waves },
  { id: 'stats', title: 'Statistics', create: statistics },
  { id: 'meanflow', title: 'Mean flow and EOFs', create: meanflow },
  { id: 'energy', title: 'Energetics', create: energetics },
  { id: 'sweep', title: 'Parameter sweep', create: sweep, needs: 'sweep' },
  { id: 'predict', title: 'Predictability', create: predictability, needs: 'ensemble' },
  { id: 'obs', title: 'Observations', create: observations, needs: 'assimilation' },
];

const $ = (id) => document.getElementById(id);
const main = $('main'), tabsEl = $('tabs'), select = $('dataset-select');
const state = { ds: null, tabs: new Map(), active: null, token: 0, index: [] };

// ---- theme ----------------------------------------------------------------------------------------
function storedTheme() { try { return localStorage.getItem('qg-theme'); } catch { return null; } }
function applyTheme(t) {
  if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
  for (const tab of state.tabs.values()) tab.dirty = true;
  const cur = state.tabs.get(state.active);
  if (cur && cur.instance) refreshTab(cur);
}
const initial = storedTheme();
if (initial === 'dark' || initial === 'light') document.documentElement.dataset.theme = initial;
$('theme-toggle').addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme === 'dark'
    || (!document.documentElement.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);
  const next = dark ? 'light' : 'dark';
  try { localStorage.setItem('qg-theme', next); } catch { /* private mode */ }
  applyTheme(next);
});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (!document.documentElement.dataset.theme) applyTheme(null); });

// ---- tabs -------------------------------------------------------------------------------------------
function refreshTab(tab) {
  try { tab.instance.refresh && tab.instance.refresh(); tab.dirty = false; } catch (e) { tab.panel.replaceChildren(errorBox(e)); }
}

async function showTab(id, { push = true } = {}) {
  const spec = TABS.find((t) => t.id === id);
  if (!spec || !state.ds) return;
  const tab = state.tabs.get(id);
  if (!tab || tab.hidden) return;
  state.active = id;
  for (const [tid, t] of state.tabs) {
    t.panel.hidden = tid !== id;
    t.button.setAttribute('aria-selected', String(tid === id));
    t.button.tabIndex = tid === id ? 0 : -1;
  }
  if (push && location.hash !== '#' + id) history.replaceState(null, '', '#' + id);
  const token = state.token;
  if (!tab.instance && !tab.failed) {
    try {
      tab.instance = tab.spec.create({ ds: state.ds }, tab.panel);
      if (tab.instance.activate) await tab.instance.activate();
    } catch (e) {
      tab.failed = true;
      tab.panel.replaceChildren(errorBox(e));
    }
  } else if (tab.instance) {
    if (tab.dirty) refreshTab(tab);
    if (tab.instance.activate) await tab.instance.activate();
  }
  if (token === state.token) window.dispatchEvent(new Event('resize'));
}

function buildTabs(ds) {
  for (const t of state.tabs.values()) { try { t.instance && t.instance.destroy && t.instance.destroy(); } catch { /* ignore */ } }
  state.tabs.clear();
  tabsEl.replaceChildren();
  main.replaceChildren();
  for (const spec of TABS) {
    const hidden = Boolean(spec.needs && !ds.has(spec.needs));
    const button = el('button', { class: 'tab', role: 'tab', type: 'button', id: 'tab-' + spec.id, 'aria-controls': 'panel-' + spec.id, 'aria-selected': 'false', tabindex: -1 }, spec.title);
    button.hidden = hidden;
    button.addEventListener('click', () => showTab(spec.id));
    button.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      const vis = TABS.filter((t) => !state.tabs.get(t.id).hidden);
      const k = vis.findIndex((t) => t.id === spec.id) + (e.key === 'ArrowRight' ? 1 : -1);
      const nxt = vis[(k + vis.length) % vis.length];
      state.tabs.get(nxt.id).button.focus();
      showTab(nxt.id);
    });
    const panel = el('div', { class: 'tabpanel', role: 'tabpanel', id: 'panel-' + spec.id, 'aria-labelledby': 'tab-' + spec.id });
    panel.hidden = true;
    tabsEl.append(button);
    main.append(panel);
    state.tabs.set(spec.id, { spec, button, panel, instance: null, hidden, dirty: false, failed: false });
  }
}

// ---- datasets -----------------------------------------------------------------------------------------
async function openDataset(source, label) {
  const token = ++state.token;
  try {
    const ds = await busy(`Loading ${label}…`, () => Dataset.open(source));
    if (token !== state.token) return;
    state.ds = ds;
    $('dataset-title').textContent = ds.title || label;
    document.title = `QG explorer · ${ds.title || label}`;
    buildTabs(ds);
    const wanted = location.hash.slice(1);
    const first = state.tabs.has(wanted) && !state.tabs.get(wanted).hidden ? wanted : 'about';
    await showTab(first, { push: false });
  } catch (e) {
    state.ds = null;
    tabsEl.replaceChildren();
    main.replaceChildren(el('div', { class: 'panel' }, errorBox(e), el('p', { class: 'hint', style: 'margin-top:8px' },
      'Choose the folder that contains manifest.json (written by `python -m qg2l run`), or pick another dataset from the list.')));
  }
}

function fillSelect(selected) {
  select.replaceChildren(...state.index.map((d) => el('option', { value: d.id }, d.title || d.id)));
  if (selected) select.value = selected;
}

async function init() {
  let index = [];
  try {
    const r = await fetch('data/index.json');
    if (r.ok) index = (await r.json()).datasets || [];
  } catch { /* opened from file:// or no data folder */ }
  state.index = index.map((d) => ({ ...d, source: new UrlSource('data/' + d.path) }));
  const q = new URLSearchParams(location.search).get('d');
  const start = state.index.find((d) => d.id === q) || state.index[0];
  fillSelect(start && start.id);
  select.addEventListener('change', () => {
    const d = state.index.find((x) => x.id === select.value);
    if (d) openDataset(d.source, d.title || d.id);
  });
  $('folder-input').addEventListener('change', (ev) => {
    const files = ev.target.files;
    if (!files || !files.length) return;
    const source = new FolderSource(files);
    const id = 'local:' + source.label;
    state.index = state.index.filter((d) => !d.id.startsWith('local:'));
    state.index.unshift({ id, title: `${source.label} (local folder)`, source });
    fillSelect(id);
    openDataset(source, source.label);
    ev.target.value = '';
  });
  window.addEventListener('hashchange', () => { const h = location.hash.slice(1); if (state.tabs.has(h)) showTab(h, { push: false }); });
  if (start) await openDataset(start.source, start.title || start.id);
  else {
    select.disabled = true;
    main.replaceChildren(el('div', { class: 'panel empty' },
      el('p', {}, 'No bundled datasets were found.'),
      el('p', { class: 'hint' }, 'Run python -m qg2l run config/default.toml --out docs/data/default, or use “Open a folder…” to load a dataset from your computer.')));
  }
}

init();
