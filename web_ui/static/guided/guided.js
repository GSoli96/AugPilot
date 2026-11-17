(function () {
'use strict';

/* ---------- utilita' ---------- */
const $ = (id) => document.getElementById(id);
function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const k in (attrs || {})) {
    const v = attrs[k];
    if (v === false || v == null) continue;
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v === true) e.setAttribute(k, '');
    else e.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) e.append(c.nodeType ? c : document.createTextNode(String(c)));
  return e;
}
function clear(e) { while (e.firstChild) e.removeChild(e.firstChild); return e; }
function fmt(v) {
  if (v == null) return '';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : (Math.abs(v) >= 100 ? v.toFixed(1) : v.toPrecision(4).replace(/\.?0+$/, ''));
  return String(v);
}
async function api(path, opts) {
  const r = await fetch(path, opts);
  let j = null;
  try { j = await r.json(); } catch (e) { /* risposta non JSON */ }
  if (!r.ok) { const err = new Error((j && (j.error || j.message)) || ('HTTP ' + r.status)); err.data = j; err.status = r.status; throw err; }
  return j;
}
const postJSON = (p, body) => api(p, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

/* ---------- stato ---------- */
const STEPS = [
  { n: 1, label: 'Goal' }, { n: 2, label: 'Data' }, { n: 3, label: 'Algorithms' },
  { n: 4, label: 'Run' }, { n: 5, label: 'Results' }, { n: 6, label: 'Explain' }
];
const S = {
  step: 1, mods: null, modsError: null, mod: null,
  priority: 'balanced', flags: { fairness: false, privacy: false, constraints: false, nogpu: false },
  datasets: [], algos: [], onlyRunnable: true,
  params: { epochs: 5, n_samples: '', device: 'cpu' },
  cache: {}, task: null, session: null,
  results: null, resFile: '', resOnlySel: true, resCols: null, sortCol: null, sortDir: 1,
  explain: { dataset: '', algorithm: '' },
  assistant: { mode: 'unknown', busy: false },
  reco: null
};
const MODNAME = () => (S.mods && S.mods.find((m) => m.id === S.mod) || {}).name || S.mod;

function canEnter(n) {
  if (n === 1) return true;
  if (!S.mod) return false;
  if (n === 2 || n === 5 || n === 6) return true;
  if (n === 3) return S.datasets.length > 0;
  if (n === 4) return S.datasets.length > 0 && S.algos.length > 0;
  return false;
}
function whyBlocked(n) {
  if (!S.mod) return 'Choose a modality first.';
  if (n >= 3 && !S.datasets.length) return 'Choose at least one dataset.';
  if (n >= 4 && !S.algos.length) return 'Choose at least one algorithm.';
  return '';
}

/* ---------- stepper, striscia, pager ---------- */
function renderStepper() {
  const ol = clear($('stepper'));
  STEPS.forEach((s) => {
    const cur = s.n === S.step;
    const done = !cur && ((s.n === 1 && S.mod) || (s.n === 2 && S.datasets.length) || (s.n === 3 && S.algos.length) || (s.n === 4 && S.task && S.task.finished));
    const li = h('li', { class: done ? 'done' : '', 'data-cur': cur ? '1' : null },
      h('button', { type: 'button', 'aria-current': cur ? 'step' : null, disabled: !canEnter(s.n), onclick: () => go(s.n),
        'aria-label': 'Step ' + s.n + ': ' + s.label + (done ? ' (completed)' : '') },
        h('span', { class: 'num', 'aria-hidden': 'true' }, done ? '✓' : s.n), h('span', { class: 'lbl' }, s.label)));
    ol.append(li);
  });
}
function renderStrip() {
  const strip = clear($('strip'));
  const chip = (label, text, step) => strip.append(h('button', { type: 'button', class: 'chip' + (text ? '' : ' empty'), onclick: () => canEnter(step) && go(step),
    title: 'Go to step ' + step }, h('b', null, label), h('span', null, text || 'to choose')));
  chip('Modality', S.mod ? MODNAME() : '', 1);
  chip('Dataset', S.datasets.join(', '), 2);
  chip('Algorithms', S.algos.join(', '), 3);
  const t = S.task;
  chip('Run', t ? (t.id + ' - ' + t.status) : '', 4);
}
function renderPager() {
  $('btn-back').disabled = S.step === 1;
  const nx = $('btn-next');
  const last = S.step === 6;
  nx.hidden = last;
  const target = S.step + 1;
  const ok = !last && canEnter(target);
  nx.disabled = !ok;
  $('pager-hint').textContent = last ? 'End of the path: you can go back or start over from step 1.' : (ok ? '' : whyBlocked(target));
}
function go(n) {
  if (!canEnter(n)) return;
  S.step = n;
  document.querySelectorAll('.panel').forEach((p) => { p.hidden = Number(p.dataset.step) !== n; });
  renderStepper(); renderStrip(); renderPager(); renderSuggestions();
  ({ 1: r1, 2: r2, 3: r3, 4: r4, 5: r5, 6: r6 })[n]();
  const hd = document.querySelector('#step-' + n + ' > h2');
  if (hd) { hd.setAttribute('tabindex', '-1'); hd.focus({ preventScroll: false }); }
}
function refreshChrome() { renderStepper(); renderStrip(); renderPager(); renderSuggestions(); }

/* ---------- caricamento dati ---------- */
async function loadMods() {
  try { S.mods = await api('/api/modalities'); S.modsError = null; } catch (e) { S.modsError = e.message; S.mods = []; }
}
async function loadMod(mod) {
  if (S.cache[mod] && S.cache[mod].ok) return S.cache[mod];
  const c = { ok: false, error: null, algos: [], datasets: [], catalog_error: null, info: { algorithms: {}, datasets: {} } };
  try {
    const [a, d] = await Promise.all([api('/api/modality/' + mod + '/algorithms'), api('/api/modality/' + mod + '/datasets')]);
    c.algos = a.algorithms || []; c.catalog_error = a.catalog_error; c.datasets = d.datasets || []; c.ok = true;
  } catch (e) { c.error = e.message; }
  c.info = await CatalogInfo.load(mod);
  S.cache[mod] = c;
  return c;
}

/* ---------- passo 1: obiettivo ---------- */
const PRIORITIES = (window.Evidence && Evidence.PROFILES) || [['balanced', 'Balanced'], ['utility', 'Utility'], ['fidelity', 'Fidelity'], ['diversity', 'Diversity'], ['privacy', 'Privacy'], ['speed', 'Speed']];
const FLAGS = [['fairness', 'Fairness'], ['privacy', 'Privacy'], ['constraints', 'Data constraints'], ['nogpu', 'No GPU']];
function r1() {
  const p = clear($('step-1'));
  p.append(h('h2', null, '1. What is your goal?'),
    h('p', { class: 'lead' }, 'Choose the type of data you want to augment. The rest of the path adapts.'));
  if (!S.mods) { p.append(h('p', { class: 'empty' }, 'Loading modalities...')); return; }
  if (S.modsError) p.append(h('div', { class: 'note bad', role: 'alert' }, 'Cannot read /api/modalities: ' + S.modsError + '. ',
    h('button', { type: 'button', class: 'btn small', onclick: async () => { await loadMods(); r1(); } }, 'Retry')));
  const cards = h('div', { class: 'cards', role: 'group', 'aria-label': 'Modality' });
  S.mods.forEach((m) => {
    const c = m.counts || {};
    cards.append(h('button', { type: 'button', class: 'card', 'aria-pressed': String(S.mod === m.id), onclick: () => chooseMod(m.id) },
      h('h3', null, m.name || m.id),
      h('p', null, (c.executable || 0) + ' runnable algorithms out of ' + (c.total || 0) + (c.described ? ' (' + c.described + ' described only)' : '')),
      h('p', null, (m.n_datasets || 0) + ((m.n_datasets || 0) === 1 ? ' dataset' : ' datasets') + ' on disk'),
      m.catalog_error ? h('p', null, 'Incomplete catalog: ' + m.catalog_error) : null));
  });
  if (!S.mods.length && !S.modsError) cards.append(h('p', { class: 'empty' }, 'No modality available.'));
  p.append(cards);

  p.append(h('h3', { style: 'margin-top:18px' }, 'Preferences (optional, used by the recommender in step 3)'));
  const sel = h('select', { id: 'prio', onchange: (e) => { S.priority = e.target.value; } },
    PRIORITIES.map(([v, l]) => h('option', { value: v, selected: S.priority === v }, l)));
  p.append(h('div', { class: 'row' }, h('label', { class: 'field' }, 'What matters most', sel)));
  const chk = h('div', { class: 'checks', role: 'group', 'aria-label': 'Requirements' });
  FLAGS.forEach(([k, l]) => chk.append(h('label', null, h('input', { type: 'checkbox', checked: S.flags[k], onchange: (e) => { S.flags[k] = e.target.checked; } }), l)));
  p.append(chk);
  p.append(h('p', { class: 'hint', style: 'text-align:left' }, 'These preferences filter or rank the recommendations; they do not change what you can run.'));
}
async function chooseMod(id) {
  if (S.mod !== id) {
    S.mod = id; S.datasets = []; S.algos = []; S.reco = null; S.results = null; S.resFile = ''; S.resCols = null; S.explain = { dataset: '', algorithm: '' };
    if (S.task && !S.task.finished) { /* il task in corso resta monitorato */ } else { S.task = null; S.eval = null; }
  }
  refreshChrome(); r1();
  await loadMod(id);
  go(2);
}

/* ---------- passo 2: dati ---------- */
let dsFilter = '';
async function r2() {
  const p = clear($('step-2'));
  p.append(h('h2', null, '2. Choose the data'),
    h('p', { class: 'lead' }, 'Select one or more ' + MODNAME() + ' datasets. Use "Analyze" to see the statistics before deciding.'));
  const c = await loadMod(S.mod);
  if (S.step !== 2) return;
  if (!c.ok) { p.append(h('div', { class: 'note bad', role: 'alert' }, 'Datasets could not be read: ' + c.error + '. ', h('button', { type: 'button', class: 'btn small', onclick: () => { delete S.cache[S.mod]; r2(); } }, 'Retry'))); return; }
  if (window.Upload) p.append(Upload.box(S.mod, (d) => { delete S.cache[S.mod]; if (!S.datasets.includes(d.name)) S.datasets.push(d.name); S.reco = null; refreshChrome(); r2(); }));
  if (!c.datasets.length) { p.append(h('p', { class: 'empty' }, 'No dataset found on disk for this modality.')); return; }
  const search = h('input', { type: 'search', id: 'ds-q', value: dsFilter, placeholder: 'Filter by name', 'aria-label': 'Filter datasets',
    oninput: (e) => { dsFilter = e.target.value; fillList(); } });
  p.append(h('div', { class: 'row' }, search, h('span', { class: 'hint', id: 'ds-count', style: 'text-align:left' })));
  const list = h('div', { class: 'list', id: 'ds-list' });
  p.append(list, h('div', { id: 'ds-info' }));
  function fillList() {
    clear(list);
    const q = dsFilter.toLowerCase();
    const rows = c.datasets.filter((d) => d.stem.toLowerCase().includes(q));
    rows.forEach((d) => {
      const meta = d.rows != null ? (d.rows + ' rows x ' + d.columns + ' columns, ') : '';
      const id = 'ds-' + d.stem;
      list.append(h('div', { class: 'item' },
        h('input', { type: 'checkbox', id, checked: S.datasets.includes(d.stem), onchange: (e) => {
          if (e.target.checked) { if (!S.datasets.includes(d.stem)) S.datasets.push(d.stem); } else S.datasets = S.datasets.filter((x) => x !== d.stem);
          S.reco = null; refreshChrome(); count();
        } }),
        h('div', { class: 't' }, h('label', { for: id }, h('b', null, d.stem, d.user ? h('span', { class: 'badge ok', style: 'margin-left:8px' }, 'uploaded by you') : null), h('small', null, meta + d.size_kb + ' KB')),
          h('details', null, h('summary', null, 'Description and uses'), CatalogInfo.datasetBlock((c.info.datasets || {})[d.stem]))),
        h('button', { type: 'button', class: 'btn small', onclick: () => analyze(d.stem) }, 'Analyze')));
    });
    if (!rows.length) list.append(h('p', { class: 'empty' }, 'No dataset matches the filter.'));
    count();
  }
  function count() { const e = $('ds-count'); if (e) e.textContent = S.datasets.length + ' selected of ' + c.datasets.length; }
  fillList();
}
async function analyze(ds) {
  const box = clear($('ds-info'));
  box.append(h('p', { class: 'hint', style: 'text-align:left' }, 'Analyzing ' + ds + '...'));
  try {
    const r = await tool('analyze_dataset', { modality: S.mod, dataset: ds });
    clear(box);
    if (r.error) { box.append(h('div', { class: 'note bad', role: 'alert' }, r.error)); return; }
    box.append(h('h3', null, 'Statistics for ' + ds), kv(r, ['modality']));
  } catch (e) { clear(box).append(h('div', { class: 'note bad', role: 'alert' }, e.message)); }
}
function kv(obj, skip) {
  // etichette leggibili e distribuzioni come barre: funzione condivisa Stats (site/shared.js)
  return Stats.list(obj, { skip: skip || [], cls: 'kv', maxArray: 12 });
}

/* ---------- passo 3: algoritmi ---------- */
let alFilter = '';
async function r3() {
  const p = clear($('step-3'));
  p.append(h('h2', null, '3. Choose the algorithms'),
    h('p', { class: 'lead' }, 'Ask for a recommendation based on the measured runs, or choose by hand. "Described only" algorithms cannot be run.'));
  const c = await loadMod(S.mod);
  if (S.step !== 3) return;
  if (!c.ok) { p.append(h('div', { class: 'note bad', role: 'alert' }, 'Algorithms could not be read: ' + c.error)); return; }
  if (c.catalog_error) p.append(h('div', { class: 'note warn' }, 'Partial catalog: ' + c.catalog_error));
  p.append(h('div', { class: 'row' },
    h('button', { type: 'button', class: 'btn primary', id: 'btn-reco', onclick: recommend }, 'Show recommendations'),
    h('input', { type: 'search', value: alFilter, placeholder: 'Filter by name or strategy', 'aria-label': 'Filter algorithms', oninput: (e) => { alFilter = e.target.value; fill(); } }),
    h('label', { class: 'checks' }, h('input', { type: 'checkbox', checked: S.onlyRunnable, onchange: (e) => { S.onlyRunnable = e.target.checked; fill(); } }), 'Runnable only')));
  p.append(profileBar());
  p.append(h('div', { id: 'reco' }));
  const list = h('div', { class: 'list' });
  p.append(h('span', { class: 'hint', id: 'al-count', style: 'text-align:left;display:block' }), list);
  function fill() {
    clear(list);
    const q = alFilter.toLowerCase();
    const rows = c.algos.filter((a) => (!S.onlyRunnable || a.runnable) && ((a.key + ' ' + (a.name || '') + ' ' + (a.strategy || '')).toLowerCase().includes(q)));
    rows.forEach((a) => {
      const id = 'al-' + a.key;
      list.append(h('div', { class: 'item' + (a.runnable ? '' : ' disabled') },
        h('input', { type: 'checkbox', id, disabled: !a.runnable, checked: S.algos.includes(a.key), onchange: (e) => {
          if (e.target.checked) { if (!S.algos.includes(a.key)) S.algos.push(a.key); } else S.algos = S.algos.filter((x) => x !== a.key);
          refreshChrome(); cnt();
        } }),
        h('div', { class: 't' }, h('label', { for: id }, h('b', null, a.name || a.key), h('small', null, [a.key, a.strategy, a.needs_gpu ? 'needs GPU' : ''].filter(Boolean).join(' - '))),
          h('details', null, h('summary', null, 'What it does, when to use it' + (CatalogInfo.unverified((c.info.algorithms || {})[a.key]) ? ' (to be verified)' : '')), CatalogInfo.algoBlock((c.info.algorithms || {})[a.key]))),
        h('span', { class: 'badge ' + (a.runnable ? 'ok' : 'warn') }, a.runnable ? 'runnable' : 'described only')));
    });
    if (!rows.length) list.append(h('p', { class: 'empty' }, 'No algorithm matches the filters.'));
    cnt();
  }
  function cnt() { const e = $('al-count'); if (e) e.textContent = S.algos.length + ' selected'; }
  fill();
  if (S.reco) showReco(S.reco);
}
/* Selettore del profilo di priorita': rilancia il consiglio con il profilo scelto (stesse misure, pesi diversi). */
function profileBar() {
  const bar = h('div', { class: 'evx-profile', role: 'group', 'aria-label': 'Priority profile', id: 'profile-bar' }, h('span', null, 'Profile:'));
  PRIORITIES.forEach(([v, l]) => bar.append(h('button', { type: 'button', class: 'evx-pbtn', 'data-profile': v, 'aria-pressed': String(S.priority === v),
    title: 'Recompute the recommendation giving more weight to: ' + l.toLowerCase(), onclick: () => { S.priority = v; syncProfileBar(); recommend(); } }, l)));
  return bar;
}
function syncProfileBar() {
  document.querySelectorAll('#profile-bar .evx-pbtn').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.profile === S.priority)));
  const sel = $('prio'); if (sel) sel.value = S.priority;
}
function recoArgs() {
  const args = { modality: S.mod, priority: S.priority, executable_only: true };
  if (S.datasets.length === 1) args.dataset = S.datasets[0];
  if (S.flags.fairness) args.need_fairness = true;
  if (S.flags.privacy) args.need_privacy = true;
  if (S.flags.constraints) args.need_constraints = true;
  if (S.flags.nogpu) args.gpu = false;
  return args;
}
async function recommend() {
  const args = Object.assign(recoArgs(), { top_k: 8 });
  const box = $('reco'); if (box) clear(box).append(h('p', { class: 'hint', style: 'text-align:left' }, 'Computing recommendations (profile ' + Evidence.profileLabel(S.priority) + ')...'));
  try {
    const r = await tool('recommend_algorithms', args);
    if (r.error) { if (box) clear(box).append(h('div', { class: 'note bad', role: 'alert' }, r.error)); return; }
    S.reco = r; S.why = {}; if (S.step === 3) showReco(r);
  } catch (e) { if (box) clear(box).append(h('div', { class: 'note bad', role: 'alert' }, e.message)); }
}
/* «Perché?»: chiama explain_recommendation con gli stessi filtri del consiglio (cosi' il classificato e' lo stesso). */
async function showWhy(x, slot, btn) {
  const key = x.algorithm + '|' + S.priority;
  S.why = S.why || {};
  if (slot.dataset.open === '1') { clear(slot); slot.dataset.open = ''; btn.setAttribute('aria-expanded', 'false'); return; }
  slot.dataset.open = '1'; btn.setAttribute('aria-expanded', 'true');
  clear(slot).append(h('p', { class: 'hint', style: 'text-align:left' }, 'Computing the explanation...'));
  const r = S.reco || {};
  const ctx = { evidence: r.evidence, method: r.method, weights: r.weights };
  if (!S.why[key]) {
    const a = recoArgs(); a.algorithm = x.algorithm;
    if (!a.dataset) S.why[key] = { error: 'For the full breakdown (metrics, comparison, similar datasets) select a single dataset.' };
    else {
      try { S.why[key] = await tool('explain_recommendation', a); }
      catch (e) { S.why[key] = { error: e.message }; }
    }
  }
  if (slot.dataset.open !== '1') return;
  clear(slot).append(Evidence.explainPanel(S.why[key], x, ctx));
}
function showReco(r) {
  const box = $('reco'); if (!box) return; clear(box);
  const rank = r.ranking || [];
  box.append(h('h3', null, 'Recommendations (' + (r.scope || 'scope not specified') + ') - profile ' + Evidence.profileLabel(r.profile || S.priority)));
  if (r.evidence === 'nessuna') box.append(Evidence.fallbackNote(r.method));
  if (!rank.length) box.append(h('p', { class: 'empty' }, 'No algorithm can be ranked: measured results are missing for these criteria.'));
  if (r.evidence === 'kb' && r.weights) box.append(Evidence.axisLegend(r.weights), h('p', { class: 'hint', style: 'text-align:left' }, 'The bar of each algorithm is as long as its score (0 to 1), split by axis.'));
  const ol = h('div', { class: 'list reco' });
  rank.forEach((x) => {
    const checked = S.algos.includes(x.algorithm);
    const slot = h('div', { class: 'why-slot' });
    const why = h('button', { type: 'button', class: 'btn small ghost', 'aria-expanded': 'false', onclick: (e) => showWhy(x, slot, e.currentTarget) }, 'Why?');
    const bar = (r.evidence === 'kb' && typeof x.score === 'number' && x.contributions)
      ? Evidence.contribBars({ score: x.score, contributions: x.contributions, axis_scores: x.axis_scores }, r.weights).querySelector('.evx-track') : null;
    ol.append(h('div', { class: 'item' },
      h('span', { class: 'badge' }, '#' + x.rank),
      h('div', { class: 't' }, h('b', null, x.name || x.algorithm, ' ', h('span', { class: 'sc', style: 'font-weight:400;font-size:.85rem' }, r.evidence === 'kb' && typeof x.score === 'number' ? 'score ' + x.score.toFixed(3) : '')),
        bar, h('small', null, (x.reasons || []).join(' ')), (x.warnings || []).length ? h('small', null, 'Warnings: ' + x.warnings.join(' ')) : null, slot),
      h('div', { class: 'row', style: 'margin:0;gap:6px;flex-wrap:nowrap' }, why,
        h('button', { type: 'button', class: 'btn small', disabled: checked || x.status === 'described', onclick: () => { if (!S.algos.includes(x.algorithm)) S.algos.push(x.algorithm); refreshChrome(); r3(); } }, checked ? 'Selected' : 'Select'))));
  });
  box.append(ol);
  if ((r.unranked || []).length) box.append(h('p', { class: 'hint', style: 'text-align:left' }, r.unranked.length + ' algorithms are not ranked because they have no measurements (or are described only).'));
  (r.notes || []).forEach((n) => box.append(h('div', { class: 'note' }, String(n))));
  box.append(rawDetails(r, 'Full recommender details'));
}
function rawDetails(obj, label) { return h('details', null, h('summary', null, label || 'Details'), h('pre', { class: 'log' }, JSON.stringify(obj, null, 2))); }

/* ---------- passo 4: esegui ---------- */
const RUN_STATE = { SUCCESS: 'completed', FAILED: 'failed', STOPPED: 'stopped', UNKNOWN: 'unknown', RUNNING: 'running', PENDING: 'queued' };
const ROW_STATE = { SUCCESS: 'succeeded', FAILED: 'failed', FALLBACK: 'fallback (FALLBACK)' };
function fmtDur(s) {
  s = Number(s); if (!isFinite(s) || s < 0) return '';
  if (s < 60) return (s < 10 ? s.toFixed(1) : Math.round(s)) + ' s';
  const m = Math.floor(s / 60), r = Math.round(s - m * 60);
  if (m < 60) return m + ' min ' + r + ' s';
  return Math.floor(m / 60) + ' h ' + (m % 60) + ' min';
}
function genCount(x) {
  for (const k of ['n_generated', 'n_augmented', 'n_syn']) if (typeof x[k] === 'number' && isFinite(x[k])) return x[k];
  return null;
}
function launchForm(p) {
  const nRuns = S.algos.length * S.datasets.length;
  const bigEpochs = Number(S.params.epochs) > 5;
  p.append(h('ul', { class: 'sum' },
    h('li', null, h('b', null, 'Modality: '), MODNAME()),
    h('li', null, h('b', null, 'Dataset (' + S.datasets.length + '): '), S.datasets.join(', ')),
    h('li', null, h('b', null, 'Algorithms (' + S.algos.length + '): '), S.algos.join(', ')),
    h('li', null, h('b', null, 'Combinations: '), String(nRuns))));
  p.append(h('div', { class: 'row' },
    h('label', { class: 'field' }, 'Epochs', h('input', { type: 'number', id: 'p-ep', min: 1, max: 5000, value: S.params.epochs, onchange: (e) => { S.params.epochs = e.target.value; r4(); } })),
    h('label', { class: 'field' }, 'Samples to generate (opt.)', h('input', { type: 'number', id: 'p-ns', min: 1, value: S.params.n_samples, onchange: (e) => { S.params.n_samples = e.target.value; } })),
    h('label', { class: 'field' }, 'Device', h('select', { id: 'p-dev', onchange: (e) => { S.params.device = e.target.value; } },
      ['cpu', 'cuda'].map((d) => h('option', { value: d, selected: S.params.device === d }, d === 'cpu' ? 'CPU' : 'GPU (cuda)'))))));
  const why = [];
  if (S.algos.length > 1) why.push(S.algos.length + ' algorithms');
  if (S.datasets.length > 1) why.push(S.datasets.length + ' datasets');
  if (bigEpochs) why.push(S.params.epochs + ' epochs');
  if (why.length) p.append(h('div', { class: 'note warn' }, 'Heavy run: ' + why.join(', ') + '. It may take a long time. Only one run can be active at a time.'));
  const ok = h('input', { type: 'checkbox', id: 'p-ok', onchange: (e) => { $('btn-run').disabled = !e.target.checked; } });
  p.append(h('div', { class: 'checks' }, h('label', null, ok, 'I have checked the summary and confirm the start')));
  p.append(h('div', { class: 'row' }, h('button', { type: 'button', class: 'btn primary', id: 'btn-run', disabled: true, onclick: launch }, 'Confirm and start'),
    h('span', { id: 'run-msg', role: 'status' })));
}
function r4() {
  const p = clear($('step-4'));
  const t = S.task;
  p.append(h('h2', null, '4. Check and start'));
  if (t && t.finished) {
    doneBox(p, t);
    // il modulo di avvio resta disponibile, ma in secondo piano
    const det = h('details', { class: 'again', open: S.againOpen ? true : null, ontoggle: (e) => { S.againOpen = e.target.open; } },
      h('summary', null, 'Start another run'));
    launchForm(det);
    p.append(det);
    return;
  }
  p.append(h('p', { class: 'lead' }, 'Check the summary. Nothing starts until you confirm.'));
  if (!t) launchForm(p); else monitorUI(p);
}
/* Riquadro di fine esecuzione: esito, tempo, righe generate, errori leggibili e il passo successivo. */
function doneBox(p, t) {
  const sm = t.summary;                       // undefined: lettura in corso; altrimenti {rows, missing, error}
  const rows = sm ? sm.rows : [];
  const bad = rows.filter((x) => x.status && x.status !== 'SUCCESS');
  const okRows = rows.filter((x) => x.status === 'SUCCESS');
  let kind, title;
  if (t.status === 'SUCCESS') {
    if (sm && rows.length && !okRows.length) { kind = 'bad'; title = 'Run finished, but no combination produced a valid result'; }
    else if (bad.length || (sm && sm.missing.length)) { kind = 'warn'; title = 'Run completed with warnings'; }
    else { kind = 'ok'; title = 'Run completed'; }
  } else if (t.status === 'STOPPED') { kind = 'warn'; title = 'Run stopped'; }
  else { kind = 'bad'; title = 'Run failed'; }
  const box = h('section', { class: 'done ' + kind, role: kind === 'bad' ? 'alert' : 'status', 'aria-labelledby': 'done-t' });
  box.append(h('h3', { id: 'done-t' }, title));
  const info = { stato: RUN_STATE[t.status] || t.status, esecuzione: t.id };
  if (t.duration_s != null) info.tempo = fmtDur(t.duration_s);
  if (sm && rows.length) {
    const gen = rows.map(genCount).filter((v) => v !== null);
    if (gen.length) info.generati = gen.reduce((a, b) => a + b, 0);
    info.combinazioni = okRows.length + ' succeeded out of ' + (t.datasets.length * t.algos.length);
  }
  const dl = h('dl', { class: 'kv' });
  dl.append(h('dt', null, 'Status'), h('dd', null, info.stato),
    h('dt', null, 'Run'), h('dd', null, info.esecuzione));
  if (info.tempo) dl.append(h('dt', null, 'Time'), h('dd', null, info.tempo));
  if (info.generati != null) dl.append(h('dt', null, S.mod === 'tabular' ? 'Rows generated' : 'Examples generated'), h('dd', null, Stats.format('x', info.generati)));
  if (info.combinazioni) dl.append(h('dt', null, 'Combinations'), h('dd', null, info.combinazioni));
  box.append(dl);
  if (!sm) box.append(h('p', { class: 'hint', style: 'text-align:left' }, 'Reading the results summary...'));
  if (sm && sm.error) box.append(h('p', { class: 'hint', style: 'text-align:left' }, 'Results summary could not be read: ' + sm.error));
  // errori per combinazione, in forma leggibile
  bad.forEach((x) => {
    const msg = String(x.error_message || x.eval_error || '').trim();
    box.append(h('div', { class: 'note ' + (x.status === 'FAILED' ? 'bad' : 'warn') },
      h('b', null, x.algorithm + ' on ' + x.dataset + ' - ' + (ROW_STATE[x.status] || x.status) + (x.method_used ? ' (method: ' + x.method_used + ')' : '') + ': '),
      msg ? msg.slice(0, 500) : 'no error message recorded.'));
  });
  if (sm && sm.missing.length) box.append(h('div', { class: 'note warn' }, 'No new result row for: ' + sm.missing.join(', ') + '. They may already exist from previous runs (automatic resume) or the run stopped earlier.'));
  if (t.status !== 'SUCCESS') {
    const tail = t.lines.filter((l) => l.trim()).slice(-8);
    if (t.pollError) box.append(h('p', { class: 'hint', style: 'text-align:left' }, 'Reading the log: ' + t.pollError));
    if (tail.length) box.append(h('details', { open: true }, h('summary', null, 'Last log lines'), h('pre', { class: 'log' }, tail.join('\n'))));
    else box.append(h('p', { class: 'hint', style: 'text-align:left' }, 'No output recorded.'));
  } else if (t.lines.length) {
    box.append(h('details', null, h('summary', null, 'Run log'), h('pre', { class: 'log' }, t.lines.slice(-60).join('\n'))));
  }
  const usable = t.status === 'SUCCESS' && !(sm && rows.length && !okRows.length);
  box.append(h('div', { class: 'row' },
    usable ? h('button', { type: 'button', class: 'btn primary', id: 'btn-eval', onclick: () => go(5) }, 'Evaluate and compare →')
      : h('button', { type: 'button', class: 'btn', onclick: () => go(5) }, 'See the saved results anyway'),
    usable ? h('span', { class: 'hint' }, S.mod === 'tabular' ? 'Step 5 starts the evaluation (fidelity and diversity) on the generated outputs.' : 'Step 5 shows the measurements computed during the run.') : null));
  p.append(box);
}
/* Dopo la fine del task: legge dai CSV le righe di risultato di questa esecuzione (per la modalita' tabulare, solo quelle scritte dopo l'avvio). */
async function finalizeTask(t) {
  try {
    const r = await api('/api/modality/' + t.mod + '/results');
    let recs = Array.isArray(r) ? [] : (r.records || []);
    recs = recs.filter((x) => t.datasets.includes(x.dataset) && t.algos.includes(x.algorithm));
    if (t.mod === 'tabular') {
      recs = recs.filter((x) => { const ts = Date.parse(String(x.timestamp || '').replace(' ', 'T')); return isNaN(ts) || ts >= t.startedAt - 5000; });
    }
    const last = {};
    recs.forEach((x) => { last[x.algorithm + '|' + x.dataset] = x; });      // l'ultima riga per coppia
    const rows = Object.keys(last).map((k) => last[k]);
    const missing = [];
    t.datasets.forEach((d) => t.algos.forEach((a) => { if (!last[a + '|' + d]) missing.push(a + ' / ' + d); }));
    t.summary = { rows, missing };
  } catch (e) { t.summary = { rows: [], missing: [], error: e.message }; }
}
async function launch() {
  const btn = $('btn-run'); btn.disabled = true;
  const msg = $('run-msg'); msg.textContent = 'Starting...';
  const args = { modality: S.mod, algorithms: S.algos.slice(), datasets: S.datasets.slice(), epochs: parseInt(S.params.epochs, 10) || 5, device: S.params.device };
  if (S.params.n_samples) args.n_samples = parseInt(S.params.n_samples, 10);
  try {
    let r = await tool('run_augmentation', args);
    if (r.status === 'needs_confirmation') {
      // la conferma umana e' appena stata data in questa pagina: la inoltro alla rotta di conferma
      r = await postJSON('/api/assistant/confirm', { session_id: S.session, pending_id: r.pending_id, approve: true });
    }
    if (r.status === 'started') { startMonitor(r.task_id, r.command); say('sys', 'Run started: ' + r.task_id); r4(); }
    else { msg.textContent = ''; $('step-4').append(h('div', { class: 'note bad', role: 'alert' }, 'Not started: ' + (r.error || r.message || JSON.stringify(r)))); btn.disabled = false; }
  } catch (e) {
    msg.textContent = '';
    $('step-4').append(h('div', { class: 'note bad', role: 'alert' }, 'Not started: ' + e.message));
    btn.disabled = false;
  }
}
let pollTimer = null;
function startMonitor(id, cmd) {
  S.task = { id, command: cmd || '', status: 'RUNNING', finished: false, lines: [], idx: 0, mod: S.mod, datasets: S.datasets.slice(), algos: S.algos.slice(), startedAt: Date.now(), duration_s: null };
  S.eval = null; S.againOpen = false;
  refreshChrome();
  if (pollTimer) clearTimeout(pollTimer);
  poll();
}
async function poll() {
  const t = S.task; if (!t || t.finished) return;
  try {
    const r = await api('/api/logs/' + encodeURIComponent(t.id) + '?start=' + t.idx);
    (r.lines || []).forEach((l) => t.lines.push(String(l).replace(/\s+$/, '')));
    if (t.lines.length > 300) t.lines = t.lines.slice(-300);
    t.idx = r.next_index != null ? r.next_index : t.idx;
    t.status = r.status || t.status;
    if (r.duration_s != null) t.duration_s = r.duration_s;
    if (['SUCCESS', 'FAILED', 'STOPPED', 'UNKNOWN'].includes(t.status)) {
      t.finished = true;
      say('sys', 'Run ' + t.id + ' finished: ' + t.status);
      S.results = null;
    }
  } catch (e) { t.pollError = e.message; }
  refreshChrome();
  if (S.step === 4) updateMonitor();
  if (!t.finished) pollTimer = setTimeout(poll, 2000);
  else {
    if (S.step === 4) r4();                                     // subito il riquadro, poi il riepilogo dei risultati
    await finalizeTask(t);
    if (S.step === 4 && S.task === t) r4();
  }
}
function monitorUI(p) {
  const t = S.task;
  p.append(h('div', { class: 'note' }, h('b', null, 'Running: '), t.id, h('br'), h('small', { id: 'mon-cmd' }, t.command)));
  p.append(h('div', { class: 'progress', id: 'mon-bar', role: 'progressbar', 'aria-label': 'Run in progress' }, h('i')));
  p.append(h('p', { id: 'mon-status', role: 'status' }), h('pre', { class: 'log', id: 'mon-log', tabindex: '0', 'aria-label': 'Log' }));
  p.append(h('div', { class: 'row' }, h('button', { type: 'button', class: 'btn danger', onclick: stopTask }, 'Stop the run')));
  updateMonitor();
}
function updateMonitor() {
  const t = S.task; if (!t || !$('mon-status')) return;
  $('mon-status').textContent = 'Status: ' + t.status + (t.pollError ? ' (log read: ' + t.pollError + ')' : '');
  const log = $('mon-log'); log.textContent = t.lines.length ? t.lines.slice(-80).join('\n') : 'No output yet.';
  log.scrollTop = log.scrollHeight;
}
async function stopTask() {
  if (!S.task) return;
  if (!window.confirm('Stop run ' + S.task.id + '?')) return;
  try { await postJSON('/api/stop_task/' + encodeURIComponent(S.task.id), {}); say('sys', 'Stop requested for ' + S.task.id); } catch (e) { say('err', 'Stop failed: ' + e.message); }
}

/* ---------- valutazione (passo 5) ----------
 * Tabulare: POST /api/run_evaluation (evaluate_metrics.py), un dataset per volta, in coda; poi confronto con compare_results.
 * Serie temporali e grafi: run_modality.py calcola gia' le metriche durante l'esecuzione, quindi non c'e' nulla da lanciare. */
let evalTimer = null;
function evalTargets() {
  const t = S.task;
  return { datasets: (t && t.datasets) || S.datasets, algos: (t && t.algos) || S.algos };
}
function startEval(auto) {
  const { datasets, algos } = evalTargets();
  if (!datasets.length || !algos.length) return;
  if (evalTimer) clearTimeout(evalTimer);
  S.eval = { taskId: S.task ? S.task.id : null, auto: !!auto, state: 'running', algos: algos.slice(), cmp: null,
    items: datasets.map((d) => ({ dataset: d, state: 'queued', task_id: null, lines: [], idx: 0, duration_s: null, error: '' })) };
  evalNext(S.eval);
}
async function evalNext(E) {
  if (S.eval !== E) return;
  const it = E.items.find((x) => x.state === 'queued');
  if (!it) {
    E.state = E.items.some((x) => x.state === 'failed') ? 'failed' : 'done';
    drawEval();
    if (E.items.some((x) => x.state === 'done')) loadCmp(E);
    return;
  }
  it.state = 'running'; drawEval();
  try {
    const r = await postJSON('/api/run_evaluation', { dataset: it.dataset, algorithms: E.algos });
    if (!r || r.success === false) throw new Error((r && r.error) || 'unexpected response');
    it.task_id = r.task_id;
    say('sys', 'Evaluation started for ' + it.dataset + ': ' + r.task_id);
    evalPoll(E, it);
  } catch (e) {
    it.state = 'failed'; it.error = e.message;
    evalNext(E);
  }
}
async function evalPoll(E, it) {
  if (S.eval !== E) return;
  try {
    const r = await api('/api/logs/' + encodeURIComponent(it.task_id) + '?start=' + it.idx);
    (r.lines || []).forEach((l) => it.lines.push(String(l).replace(/\s+$/, '')));
    if (it.lines.length > 200) it.lines = it.lines.slice(-200);
    it.idx = r.next_index != null ? r.next_index : it.idx;
    if (r.duration_s != null) it.duration_s = r.duration_s;
    const st = r.status || 'UNKNOWN';
    if (['SUCCESS', 'FAILED', 'STOPPED', 'UNKNOWN'].includes(st)) {
      it.state = st === 'SUCCESS' ? 'done' : 'failed';
      if (it.state === 'failed') it.error = 'the evaluation ended with status ' + (RUN_STATE[st] || st) + '. ' + it.lines.filter((l) => l.trim()).slice(-3).join(' | ');
      evalNext(E);
      return;
    }
  } catch (e) { it.pollError = e.message; }
  drawEval();
  evalTimer = setTimeout(() => evalPoll(E, it), 2000);
}
async function loadCmp(E) {
  const args = { modality: S.mod };
  if (E.items.length === 1) args.dataset = E.items[0].dataset;
  try { E.cmp = await tool('compare_results', args); } catch (e) { E.cmp = { error: e.message }; }
  drawEval();
}
function renderCmp(box, r) {
  if (r.error) { box.append(h('div', { class: 'note bad', role: 'alert' }, r.error)); return; }
  const rows = (r.rows || []).slice(0, 100);
  box.append(h('h3', null, 'Comparison with sources'));
  box.append(h('p', { class: 'hint', style: 'text-align:left' }, (r.n_rows != null ? r.n_rows : rows.length) + ' rows in total; each row shows its source file.'));
  if (!rows.length) { box.append(h('p', { class: 'empty' }, r.message || 'No comparison rows.')); return; }
  const cols = Object.keys(rows[0]);
  box.append(h('div', { class: 'tablewrap' }, h('table', null, h('thead', null, h('tr', null, cols.map((c) => h('th', { scope: 'col' }, c)))),
    h('tbody', null, rows.map((x) => h('tr', null, cols.map((c) => h('td', { class: typeof x[c] === 'number' ? 'n' : '' }, fmt(x[c])))))))));
}
function drawEval() {
  const box = $('eval-box'); if (!box || S.step !== 5) return;
  clear(box);
  const E = S.eval, t = S.task;
  box.append(h('h3', null, 'Metrics evaluation'));
  if (S.mod !== 'tabular') {
    box.append(h('div', { class: 'note ok' }, 'For this modality the metrics are already computed during the run (run_modality.py includes them): you can find them in the table below. There is no separate evaluation to start.'));
    return;
  }
  const { datasets, algos } = evalTargets();
  if (!E) {
    let why = '';
    if (!datasets.length || !algos.length) why = 'Choose at least one dataset and one algorithm (steps 2 and 3) to evaluate them.';
    else if (t && !t.finished) why = 'The run is still in progress: the evaluation starts after it ends.';
    else if (t && t.status !== 'SUCCESS') why = 'The last run did not succeed (' + (RUN_STATE[t.status] || t.status) + '), so the evaluation did not start automatically.';
    else why = 'There is no completed run in this session: the evaluation does not start automatically.';
    box.append(h('div', { class: 'note' }, why + ' You can still evaluate the outputs already in augmented_datasets/ for ' + (datasets.join(', ') || '...') + ' (' + (algos.join(', ') || '...') + ').'));
    box.append(h('div', { class: 'row' }, h('button', { type: 'button', class: 'btn primary', disabled: !datasets.length || !algos.length || (t && !t.finished), onclick: () => { startEval(false); drawEval(); } }, 'Evaluate existing outputs')));
    return;
  }
  const running = E.state === 'running';
  box.append(h('p', { class: 'hint', style: 'text-align:left' }, (E.auto ? 'Started automatically at the end of the run. ' : '') + 'Fidelity and diversity metrics (evaluate_metrics.py), one dataset at a time.'));
  E.items.forEach((it) => {
    const label = { queued: 'queued', running: 'running', done: 'completed', failed: 'failed' }[it.state];
    box.append(h('div', { class: 'note ' + (it.state === 'done' ? 'ok' : it.state === 'failed' ? 'bad' : '') },
      h('b', null, it.dataset + ': '), label + (it.duration_s ? ' (' + fmtDur(it.duration_s) + ')' : ''),
      it.state === 'failed' && it.error ? h('div', null, it.error) : null,
      it.lines.length ? h('details', null, h('summary', null, 'Log'), h('pre', { class: 'log' }, it.lines.slice(-30).join('\n'))) : null));
  });
  if (running) box.append(h('div', { class: 'progress', role: 'progressbar', 'aria-label': 'Evaluation in progress' }, h('i')));
  if (E.state === 'failed') box.append(h('div', { class: 'row' }, h('button', { type: 'button', class: 'btn', onclick: () => { startEval(false); drawEval(); } }, 'Retry the evaluation')));
  if (E.state === 'done') box.append(h('div', { class: 'note ok' }, 'Evaluation completed.'));
  if (E.cmp) renderCmp(box, E.cmp);
  else if (E.state !== 'running' && E.items.some((x) => x.state === 'done')) box.append(h('p', { class: 'hint' }, 'Loading the comparison...'));
}

/* ---------- passo 5: risultati ---------- */
const PREF = ['status', 'time_s', 'total_time_s', 'n_generated', 'n_augmented', 'clf_acc_delta', 'delta_accuracy', 'delta_macro_f1', 'peak_mem_mb', 'total_mem_peak_mb'];
async function r5() {
  const p = clear($('step-5'));
  p.append(h('h2', null, '5. Results'),
    h('p', { class: 'lead' }, 'Measurements saved on disk for ' + MODNAME() + '. These are the benchmark CSV files: no value is computed here.'));
  // tabulare: dopo un'esecuzione riuscita la valutazione parte da sola (una volta per esecuzione)
  const t = S.task, sm = t && t.summary;
  if (S.mod === 'tabular' && t && t.finished && t.status === 'SUCCESS' && !S.eval && !(sm && sm.rows.length && !sm.rows.some((x) => x.status === 'SUCCESS'))) startEval(true);
  p.append(h('div', { id: 'eval-box', class: 'evalbox' }));
  drawEval();
  p.append(h('div', { id: 'res-body' }, h('p', { class: 'empty' }, 'Loading results...')));
  try {
    const q = S.resFile && S.mod !== 'tabular' ? '?file=' + encodeURIComponent(S.resFile) : '';
    const r = await api('/api/modality/' + S.mod + '/results' + q);
    S.results = Array.isArray(r) ? { records: [], files: [] } : r;
  } catch (e) { S.results = { records: [], files: (e.data && e.data.files) || [], error: e.message }; }
  if (S.step === 5) drawResults();
}
function drawResults() {
  const body = clear($('res-body')); const R = S.results;
  if (R.error) body.append(h('div', { class: 'note bad', role: 'alert' }, R.error));
  const files = R.files || [];
  if (files.length > 1 && S.mod !== 'tabular') {
    body.append(h('div', { class: 'row' }, h('label', { class: 'field' }, 'Results file', h('select', { onchange: (e) => { S.resFile = e.target.value; r5(); } },
      files.map((f) => h('option', { value: f, selected: f === (R.filename || files[0]) }, f))))));
  } else if (R.filename) body.append(h('p', { class: 'hint', style: 'text-align:left' }, 'File: ' + R.filename));
  let rows = R.records || [];
  if (!rows.length) { body.append(h('p', { class: 'empty' }, 'There are no results for this modality yet. Start a run in step 4.')); return; }
  const selOnly = h('input', { type: 'checkbox', checked: S.resOnlySel, onchange: (e) => { S.resOnlySel = e.target.checked; drawResults(); } });
  body.append(h('div', { class: 'checks' }, h('label', null, selOnly, 'Only the selected datasets and algorithms')));
  const total = rows.length;
  if (S.resOnlySel && (S.datasets.length || S.algos.length)) {
    rows = rows.filter((x) => (!S.datasets.length || S.datasets.includes(x.dataset)) && (!S.algos.length || S.algos.includes(x.algorithm)));
  }
  body.append(h('p', { class: 'hint', style: 'text-align:left' }, rows.length + ' rows shown out of ' + total + '.'));
  const keys = Object.keys(R.records[0]).filter((k) => k !== 'error_message' && k !== 'output_path' && k !== 'modality');
  if (!S.resCols || S.resColsMod !== S.mod) { S.resCols = ['algorithm', 'dataset'].concat(PREF.filter((k) => keys.includes(k)).slice(0, 5)); S.resColsMod = S.mod; }
  const det = h('details', null, h('summary', null, 'Columns (' + S.resCols.length + ')'));
  const cg = h('div', { class: 'checks' });
  keys.forEach((k) => cg.append(h('label', null, h('input', { type: 'checkbox', checked: S.resCols.includes(k), onchange: (e) => {
    S.resCols = e.target.checked ? keys.filter((x) => S.resCols.includes(x) || x === k) : S.resCols.filter((x) => x !== k); drawResults(); } }), k)));
  det.append(cg); body.append(det);
  if (S.sortCol) rows = rows.slice().sort((a, b) => { const x = a[S.sortCol], y = b[S.sortCol]; if (x == null) return 1; if (y == null) return -1; return (x > y ? 1 : x < y ? -1 : 0) * S.sortDir; });
  const thead = h('tr', null, S.resCols.map((k) => h('th', { scope: 'col', 'aria-sort': S.sortCol === k ? (S.sortDir > 0 ? 'ascending' : 'descending') : 'none' },
    h('button', { type: 'button', onclick: () => { S.sortDir = S.sortCol === k ? -S.sortDir : 1; S.sortCol = k; drawResults(); } }, k + (S.sortCol === k ? (S.sortDir > 0 ? ' ▲' : ' ▼') : '')))));
  const tb = h('tbody');
  rows.slice(0, 500).forEach((x) => tb.append(h('tr', null, S.resCols.map((k) => h('td', { class: typeof x[k] === 'number' ? 'n' : '' },
    k === 'status' ? h('span', { class: 'badge ' + (x[k] === 'SUCCESS' ? 'ok' : 'bad'), title: x.error_message || '' }, x[k]) : fmt(x[k]))))));
  body.append(h('div', { class: 'tablewrap' }, h('table', null, h('caption', { class: 'sr' }, 'Results'), h('thead', null, thead), tb)));
  if (rows.length > 500) body.append(h('p', { class: 'hint' }, 'Showing the first 500 rows.'));
  const fails = rows.filter((x) => x.status && x.status !== 'SUCCESS' && x.error_message);
  if (fails.length) body.append(h('details', null, h('summary', null, fails.length + ' failed runs: messages'),
    fails.slice(0, 20).map((x) => h('div', { class: 'note bad' }, h('b', null, x.algorithm + ' / ' + x.dataset + ': '), String(x.error_message).slice(0, 400)))));
  body.append(h('div', { class: 'row' }, h('button', { type: 'button', class: 'btn', onclick: compare }, 'Comparison with sources'), h('button', { type: 'button', class: 'btn primary', onclick: () => go(6) }, 'Go to the explanation')));
  body.append(h('div', { id: 'cmp' }));
}
async function compare() {
  const box = $('cmp') || $('eval-box'); if (!box) return;
  const out = $('cmp') ? clear(box) : box;
  out.append(h('p', { class: 'hint' }, 'Loading the comparison...'));
  const args = { modality: S.mod };
  if (S.datasets.length === 1) args.dataset = S.datasets[0];
  try {
    const r = await tool('compare_results', args);
    if ($('cmp')) clear($('cmp'));
    renderCmp($('cmp') || out, r);
  } catch (e) { out.append(h('div', { class: 'note bad', role: 'alert' }, e.message)); }
}

/* ---------- passo 6: spiega ---------- */
async function r6() {
  const p = clear($('step-6'));
  p.append(h('h2', null, '6. Understand the results'),
    h('p', { class: 'lead' }, 'Choose a dataset-algorithm pair that has already been measured: you see the real values with the metric definitions.'));
  if (!S.results) {
    try { const r = await api('/api/modality/' + S.mod + '/results'); S.results = Array.isArray(r) ? { records: [] } : r; } catch (e) { S.results = { records: [], error: e.message }; }
    if (S.step !== 6) return;
  }
  const recs = (S.results.records || []);
  if (!recs.length) { p.append(h('p', { class: 'empty' }, 'No result to explain: start a run first (step 4).')); return; }
  const dss = [...new Set(recs.map((x) => x.dataset))].sort();
  if (!S.explain.dataset || !dss.includes(S.explain.dataset)) S.explain.dataset = S.datasets.find((d) => dss.includes(d)) || dss[0];
  const algsFor = () => [...new Set(recs.filter((x) => x.dataset === S.explain.dataset).map((x) => x.algorithm))].sort();
  if (!algsFor().includes(S.explain.algorithm)) S.explain.algorithm = algsFor()[0];
  const selD = h('select', { id: 'ex-ds', onchange: (e) => { S.explain.dataset = e.target.value; r6(); } }, dss.map((d) => h('option', { value: d, selected: d === S.explain.dataset }, d)));
  const selA = h('select', { id: 'ex-al', onchange: (e) => { S.explain.algorithm = e.target.value; } }, algsFor().map((a) => h('option', { value: a, selected: a === S.explain.algorithm }, a)));
  p.append(h('div', { class: 'row' }, h('label', { class: 'field' }, 'Dataset', selD), h('label', { class: 'field' }, 'Algorithm', selA),
    h('button', { type: 'button', class: 'btn primary', onclick: explain }, 'Explain the metrics')));
  p.append(h('div', { id: 'ex-out' }));
  p.append(h('div', { class: 'row' }, h('button', { type: 'button', class: 'btn', onclick: () => { S.mod = null; S.datasets = []; S.algos = []; S.task = null; S.eval = null; S.results = null; S.reco = null; go(1); } }, 'Start a new path')));
}
async function explain() {
  const out = clear($('ex-out')); out.append(h('p', { class: 'hint' }, 'Reading the measured values...'));
  try {
    const r = await tool('explain_metrics', { modality: S.mod, dataset: S.explain.dataset, algorithm: S.explain.algorithm });
    clear(out);
    if (r.error) { out.append(h('div', { class: 'note bad', role: 'alert' }, r.error)); return; }
    out.append(h('h3', null, r.algorithm + ' on ' + r.dataset), h('p', { class: 'hint', style: 'text-align:left' }, 'Source: ' + (r.source || 'not specified')));
    if (r.values) out.append(kv(r.values));
    if (r.metrics) out.append(h('details', { open: true }, h('summary', null, 'Explained metrics'), h('pre', { class: 'log' }, JSON.stringify(r.metrics, null, 2))));
    if (r.metrics_doc) out.append(h('details', null, h('summary', null, 'Column definitions'), h('pre', { class: 'log' }, r.metrics_doc)));
    if (S.assistant.mode === 'online') out.append(h('button', { type: 'button', class: 'btn', onclick: () => { openAssistant(); sendToAssistant('Explain in plain words the results of ' + r.algorithm + ' on ' + r.dataset + ', using only the measured values.'); } }, 'Ask the assistant to comment on them'));
    else out.append(h('div', { class: 'note' }, 'The natural-language comment requires a model: the assistant is offline now, so here you only see the values and definitions.'));
  } catch (e) { clear(out).append(h('div', { class: 'note bad', role: 'alert' }, e.message)); }
}

/* ---------- assistente ---------- */
function say(kind, text) {
  const log = $('as-log'); const m = h('div', { class: 'msg ' + kind });
  if (kind === 'bot' && window.Md) Md.render(m, text); else m.textContent = text;   // markdown reso solo per le risposte dell'assistente
  log.append(m); log.scrollTop = log.scrollHeight; return m;
}
async function refreshStatus() {
  const b = $('as-badge'), off = $('as-offline');
  try {
    const st = await api('/api/assistant/status');
    S.assistant.mode = st.mode === 'online' ? 'online' : 'offline';
    if (st.mode === 'online') { b.textContent = 'online'; b.className = 'badge ok'; off.hidden = true; }
    else {
      b.textContent = 'offline'; b.className = 'badge warn'; off.hidden = false;
      clear(off).append(h('b', null, 'No model available. '), 'I do not make up answers: the shortcuts below call the read-only tools directly. ',
        h('details', null, h('summary', null, 'Reason'), String(st.reason || 'not specified')));
    }
  } catch (e) { S.assistant.mode = 'offline'; b.textContent = 'unreachable'; b.className = 'badge bad'; off.hidden = false; off.textContent = 'Assistant status could not be read: ' + e.message; }
  $('as-input').disabled = S.assistant.mode !== 'online';
  $('as-send').disabled = S.assistant.mode !== 'online';
  $('as-input').placeholder = S.assistant.mode === 'online' ? 'Write to the assistant (Enter to send, Shift+Enter for a new line)' : 'Chat disabled: use the shortcuts';
  renderSuggestions();
}
function ctxText() {
  const bits = ['Step ' + S.step + ' of 6 (' + STEPS[S.step - 1].label + ')'];
  if (S.mod) bits.push('modality ' + S.mod);
  if (S.datasets.length) bits.push('dataset ' + S.datasets.join(', '));
  if (S.algos.length) bits.push('algorithms ' + S.algos.join(', '));
  return bits.join('; ');
}
const SUGG = {
  1: [{ l: 'Which modality is right for me?', p: 'I do not know which modality to choose (tabular, time series or graphs). Ask me two questions to work it out.', t: () => api('/api/modalities').then((m) => say('bot', m.map((x) => x.name + ': ' + x.counts.executable + ' runnable algorithms, ' + x.n_datasets + ' datasets').join('\n'))), tl: 'Show what is available for each modality' }],
  2: [{ l: 'How do I choose the dataset?', p: 'Help me choose among the datasets available for this modality. Use the tools to analyze them.', t: () => S.mod && listDatasets(), tl: 'List datasets with their sizes' }],
  3: [{ l: 'Recommend algorithms', p: 'Recommend algorithms for the chosen datasets, with the measured evidence.', t: () => recommend(), tl: 'Show the measured recommendations' }],
  4: [{ l: 'Is the plan reasonable?', p: 'Check the current run plan (algorithms, datasets, epochs): is it reasonable in terms of time?', t: null }],
  5: [{ l: 'How did it go?', p: 'Summarize how the last run went and which results to look at first.', t: () => compare(), tl: 'Show the comparison with sources' }],
  6: [{ l: 'Explain these numbers', p: 'Explain in plain words what the metrics of the chosen pair mean.', t: () => explain(), tl: 'Show values and definitions' }]
};
async function listDatasets() {
  const c = await loadMod(S.mod);
  say('bot', c.ok ? c.datasets.map((d) => d.stem + (d.rows != null ? ' (' + d.rows + 'x' + d.columns + ')' : '')).join(', ') || 'No datasets.' : 'Error: ' + c.error);
}
function renderSuggestions() {
  const box = clear($('as-sugg')); $('as-ctx').textContent = 'Where we are: ' + ctxText() + '.';
  const online = S.assistant.mode === 'online';
  box.append(h('span', { class: 'lab' }, online ? 'Ask the assistant' : 'Shortcuts (no model)'));
  (SUGG[S.step] || []).forEach((s) => {
    if (online) box.append(h('button', { type: 'button', class: 'btn small', onclick: () => { openAssistant(); sendToAssistant(s.p); } }, s.l));
    else if (s.t) box.append(h('button', { type: 'button', class: 'btn small', onclick: async () => { say('sys', 'Offline: running the tool directly.'); try { await s.t(); } catch (e) { say('err', e.message); } } }, s.tl));
  });
  if (!online && !(SUGG[S.step] || []).some((s) => s.t)) box.append(h('span', { class: 'lab' }, 'There are no model-free shortcuts for this step.'));
}
function openAssistant() { $('assistant').classList.add('open'); $('as-open').setAttribute('aria-expanded', 'true'); $('as-input').focus(); }
function closeAssistant() { $('assistant').classList.remove('open'); $('as-open').setAttribute('aria-expanded', 'false'); $('as-open').focus(); }

/* chiamata diretta a uno strumento (anche offline); ritorna il risultato dello strumento */
async function tool(name, args) {
  const r = await postJSON('/api/assistant/chat?stream=0', { session_id: S.session, offline_action: { tool: name, args } });
  if (r.session_id) S.session = r.session_id;
  const ev = (r.events || []).find((e) => e.type === 'tool_result');
  return ev ? ev.result : { error: 'no result from the tool' };
}

/* chat con SSE */
async function sendToAssistant(text) {
  if (S.assistant.busy || S.assistant.mode !== 'online') return;
  S.assistant.busy = true; $('as-send').disabled = true;
  say('user', text);
  const body = { session_id: S.session, message: '[Context: ' + ctxText() + '] ' + text };
  let botMsg = null;
  try {
    const resp = await fetch('/api/assistant/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!resp.ok || !resp.body) throw new Error('HTTP ' + resp.status);
    const reader = resp.body.getReader(); const dec = new TextDecoder(); let buf = '';
    for (;;) {
      const { value, done } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf('\n\n')) >= 0) {
        const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
        const line = chunk.split('\n').find((l) => l.startsWith('data:')); if (!line) continue;
        let ev; try { ev = JSON.parse(line.slice(5)); } catch (e) { continue; }
        botMsg = handleEvent(ev, botMsg);
      }
    }
  } catch (e) { say('err', 'Communication error: ' + e.message); }
  S.assistant.busy = false; $('as-send').disabled = S.assistant.mode !== 'online';
}
function handleEvent(ev) {
  switch (ev.type) {
    case 'session': S.session = ev.session_id; break;
    case 'text': say('bot', ev.text); break;
    case 'offline': say('sys', ev.text); refreshStatus(); break;
    case 'tool_call': { const m = say('sys', 'Tool: ' + ev.name); m.title = JSON.stringify(ev.input); break; }
    case 'tool_result': {
      const m = say('sys', (ev.ok ? 'Result of ' : 'Error in ') + ev.name);
      m.append(rawDetails(ev.result, 'Details'));
      if (ev.name === 'run_augmentation' && ev.result && ev.result.status === 'started') { startMonitor(ev.result.task_id, ev.result.command); go(4); }
      break;
    }
    case 'pending': pendingCard(ev); break;
    case 'warning': case 'limit': say('sys', ev.text); break;
    case 'error': say('err', ev.text); break;
    default: break;
  }
}
function pendingCard(ev) {
  const plan = ev.plan || {};
  const card = h('div', { class: 'msg bot' });
  const status = h('div', { role: 'status' });
  const yes = h('button', { type: 'button', class: 'btn small primary', onclick: () => decide(true) }, 'Confirm and start');
  const no = h('button', { type: 'button', class: 'btn small', onclick: () => decide(false) }, 'Reject');
  card.append(h('b', null, 'The assistant proposes a run'), h('div', null, 'Reason for confirmation: ' + ev.reason),
    h('div', null, [plan.modality, (plan.algorithms || []).join(', '), (plan.datasets || []).join(', '), plan.epochs + ' epochs'].join(' - ')),
    h('div', { class: 'row' }, yes, no), status);
  $('as-log').append(card); $('as-log').scrollTop = $('as-log').scrollHeight;
  async function decide(ok) {
    yes.disabled = no.disabled = true;
    try {
      const r = await postJSON('/api/assistant/confirm', { session_id: S.session, pending_id: ev.pending_id, approve: ok });
      if (r.status === 'started') { status.textContent = 'Started: ' + r.task_id; startMonitor(r.task_id, r.command); go(4); }
      else status.textContent = ok ? ('Not started: ' + (r.error || r.message || '')) : 'Rejected: nothing was started.';
    } catch (e) { status.textContent = 'Error: ' + e.message; }
  }
}

/* ---------- prefill dal Catalogo (?mod=&datasets=&algos=) ---------- */
async function applyPrefill(pf) {
  Prefill.clean();
  S.mod = pf.mod; S.datasets = []; S.algos = []; S.reco = null;
  const c = await loadMod(pf.mod);
  const notes = [];
  if (c.ok) {
    const dsOk = new Set(c.datasets.map((d) => d.stem));
    S.datasets = pf.datasets.filter((d) => dsOk.has(d));
    const miss = pf.datasets.filter((d) => !dsOk.has(d));
    if (miss.length) notes.push('datasets not found on disk: ' + miss.join(', '));
    const byKey = {}; c.algos.forEach((a) => { byKey[a.key] = a; });
    S.algos = pf.algos.filter((k) => byKey[k] && byKey[k].runnable);
    const desc = pf.algos.filter((k) => byKey[k] && !byKey[k].runnable);
    const unk = pf.algos.filter((k) => !byKey[k]);
    if (desc.length) notes.push('described only, not runnable (excluded): ' + desc.join(', '));
    if (unk.length) notes.push('unknown algorithms: ' + unk.join(', '));
  } else notes.push('catalog could not be read: ' + c.error);
  const box = $('pf-note');
  box.hidden = false;
  box.textContent = 'Selection received from the Catalog (' + MODNAME() + '): ' + S.datasets.length + ' datasets, ' + S.algos.length + ' algorithms.' + (notes.length ? ' Warning - ' + notes.join('; ') + '.' : '');
  say('sys', box.textContent);
  go(!S.datasets.length ? 2 : (!S.algos.length ? 3 : 4));
}

/* ---------- avvio ---------- */
function init() {
  $('btn-back').addEventListener('click', () => go(Math.max(1, S.step - 1)));
  $('btn-next').addEventListener('click', () => go(Math.min(6, S.step + 1)));
  $('as-open').addEventListener('click', openAssistant);
  $('as-close').addEventListener('click', closeAssistant);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && $('assistant').classList.contains('open')) closeAssistant(); });
  const form = $('as-form'), inp = $('as-input');
  form.addEventListener('submit', (e) => { e.preventDefault(); const t = inp.value.trim(); if (!t) return; inp.value = ''; sendToAssistant(t); });
  inp.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); } });
  say('sys', 'Hello. I will guide you step by step: tell me if anything is unclear. The assistant only uses tools and measured results.');
  go(1);
  loadMods().then(() => {
    const pf = window.Prefill && Prefill.read();
    if (pf && (S.mods || []).some((m) => m.id === pf.mod)) return applyPrefill(pf);
    if (S.step === 1) r1();
  });
  refreshStatus();
}
init();
})();
