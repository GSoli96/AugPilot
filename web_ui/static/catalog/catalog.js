/* catalog.js - Esploratore del catalogo. Vanilla JS, solo API reali del tool. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const MOD_NAME = { tabular: 'Tabular', ts: 'Time series', graph: 'Graphs' };
  const GENERATIVE = /gan|vae|diffusion|llm|generativ|flow|transformer|autoregress/i;

  // Metriche mostrate nel confronto. dir: 'up' = piu' alto e' meglio, 'down' = piu' basso e' meglio, null = non assegnata.
  const METRICS = {
    ts: [['clf_acc_delta', 'Classifier accuracy delta (aug - base)', 'up'], ['clf_acc_aug', 'Accuracy with augmentation', 'up'],
      ['DTW', 'DTW', null], ['ED', 'ED', null], ['ACD', 'ACD', null], ['KD', 'KD', null], ['MDD', 'MDD', null], ['SD', 'SD', null],
      ['PS', 'PS', null], ['DS', 'DS', null], ['FD_feat', 'FD (features)', null],
      ['time_s', 'Time (s)', 'down'], ['peak_mem_mb', 'Peak RAM (MB)', 'down'], ['peak_gpu_mb', 'Peak GPU (MB)', 'down']],
    graph: [['delta_accuracy', 'Delta accuracy', 'up'], ['delta_macro_f1', 'Delta macro-F1', 'up'], ['delta_auc', 'Delta AUC', 'up'],
      ['struct_deg_mmd', 'Degree MMD', null], ['struct_spec_mmd', 'Spectrum MMD', null], ['struct_deg_w1', 'Degree W1', null], ['struct_clust_w1', 'Clustering W1', null],
      ['div_syn_novelty', 'Synthetic novelty', null], ['div_syn_uniqueness', 'Synthetic uniqueness', null], ['div_syn_pairwise_dist', 'Mean pairwise distance', null],
      ['time_s', 'Time (s)', 'down'], ['peak_mem_mb', 'Peak RAM (MB)', 'down'], ['peak_gpu_mb', 'Peak GPU (MB)', 'down']],
    tabular: [['total_time_s', 'Total time (s)', 'down'], ['train_time_s', 'Training time (s)', 'down'], ['gen_time_s', 'Generation time (s)', 'down'],
      ['total_mem_peak_mb', 'Peak RAM (MB)', 'down'], ['train_gpu_peak_mb', 'Training GPU (MB)', 'down'], ['gen_gpu_peak_mb', 'Generation GPU (MB)', 'down'],
      ['n_augmented', 'Augmented rows', null]],
  };
  const NOT_HERE = { tabular: 'Fidelity and Diversity for tabular data are not in this endpoint (only time and memory from runtime_results): for now they appear as not measured here.' };

  const S = {
    mods: [], algos: {}, ds: {}, res: {}, idx: {}, loadErr: [], info: {}, selDs: new Set(), dsq: '',
    sel: new Set(), view: 'catalog', mxMod: 'ts', cmpCommon: true,
    f: { q: '', mods: new Set(), status: new Set(), strat: new Set(), gpu: '', lic: new Set(), task: new Set(), withRes: false },
    status: null, lastTask: null,
  };

  function h(tag, attrs, kids) {
    const e = document.createElement(tag);
    for (const k in (attrs || {})) {
      const v = attrs[k];
      if (v == null || v === false) continue;
      if (k === 'class') e.className = v; else if (k === 'text') e.textContent = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? '' : v);
    }
    [].concat(kids == null ? [] : kids).forEach(c => { if (c != null) e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); });
    return e;
  }
  const key = a => a.mod + ':' + a.key;
  const num = v => (typeof v === 'number' && isFinite(v));
  const pl = (n, one, many) => n + ' ' + (n === 1 ? one : many);
  const ND = v => (v === 'n/d' ? 'n/a' : v);
  function fmt(v) {
    if (!num(v)) return '-';
    const a = Math.abs(v);
    if (a !== 0 && (a < 0.001 || a >= 1e5)) return v.toExponential(2);
    return (Math.round(v * 10000) / 10000).toString();
  }
  async function getJSON(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(url + ' -> HTTP ' + r.status);
    return r.json();
  }

  // ---------- Caricamento ----------
  function classLic(a) {
    const l = (a.license || '').trim();
    if (!l || l === '-') return 'nd';
    if (/nessuna licenza|no licen[sc]e|\bNC\b|non.?commercial|GPL|AGPL|proprietar/i.test(l)) return 'warn';
    return 'ok';
  }
  const LIC_LABEL = { ok: 'Declared license', warn: 'License to verify', nd: 'License n/a' };

  async function load() {
    const ban = $('load-banner');
    try { S.mods = await getJSON('/api/modalities'); }
    catch (e) { ban.hidden = false; ban.className = 'banner bad'; ban.textContent = 'Cannot read /api/modalities: ' + e.message; return; }
    await Promise.all(S.mods.map(async m => {
      const id = m.id;
      try {
        const a = await getJSON('/api/modality/' + id + '/algorithms');
        S.algos[id] = (a.algorithms || []).map(x => Object.assign({}, x, { mod: id, strat: x.strategy || x.category || 'n/a' }));
        if (a.catalog_error) S.loadErr.push(MOD_NAME[id] + ': ' + a.catalog_error);
      } catch (e) { S.algos[id] = []; S.loadErr.push('Algorithms ' + id + ': ' + e.message); }
      try { S.ds[id] = (await getJSON('/api/modality/' + id + '/datasets')).datasets || []; }
      catch (e) { S.ds[id] = []; S.loadErr.push('Datasets ' + id + ': ' + e.message); }
      try { S.res[id] = (await getJSON('/api/modality/' + id + '/results?all=1')).records || []; }
      catch (e) { S.res[id] = []; S.loadErr.push('Results ' + id + ': ' + e.message); }
      indexResults(id);
      S.info[id] = await CatalogInfo.load(id);
    }));
    if (S.loadErr.length) { ban.hidden = false; ban.className = 'banner warn'; ban.textContent = 'Partial data: ' + S.loadErr.join(' | '); }
    S.mxMod = (S.mods.find(m => m.id === 'ts') || S.mods[0] || {}).id || 'ts';
    buildFacets(); renderAll();
  }

  // idx[mod][algo] = { ok: {dataset: [records]}, fail: {dataset: n}, nok, nfail }
  function indexResults(mod) {
    const ix = {};
    (S.res[mod] || []).forEach(r => {
      if (!r.algorithm || !r.dataset) return;
      const ds = String(r.dataset).replace(/_min$/, '');  // same dataset for the KB (abalone9-18) and the UI files (abalone9-18_min)
      const e = ix[r.algorithm] || (ix[r.algorithm] = { ok: {}, fail: {}, nok: 0, nfail: 0, nfb: 0 });
      const st = String(r.status).toUpperCase();
      if (st === 'FALLBACK') e.nfb++;  // completed with a declared substitute: counted as measured, shown apart
      if (st === 'SUCCESS' || st === 'FALLBACK') { (e.ok[ds] = e.ok[ds] || []).push(r); e.nok++; }
      else { e.fail[ds] = (e.fail[ds] || 0) + 1; e.nfail++; }
    });
    S.idx[mod] = ix;
  }
  const info = a => (S.idx[a.mod] || {})[a.key] || { ok: {}, fail: {}, nok: 0, nfail: 0, nfb: 0 };
  const allAlgos = () => S.mods.flatMap(m => S.algos[m.id] || []);
  const byKey = k => allAlgos().find(a => key(a) === k);

  // ---------- Filtri ----------
  function matches(a) {
    const f = S.f;
    if (f.mods.size && !f.mods.has(a.mod)) return false;
    if (f.status.size && !f.status.has(a.status)) return false;
    if (f.strat.size && !f.strat.has(a.strat)) return false;
    if (f.lic.size && !f.lic.has(classLic(a))) return false;
    if (f.task.size && !f.task.has(a.task || 'n/a')) return false;
    if (f.gpu === 'yes' && a.needs_gpu !== true) return false;
    if (f.gpu === 'no' && a.needs_gpu !== false) return false;
    if (f.withRes && info(a).nok === 0) return false;
    if (f.q) {
      const t = [a.name, a.key, a.strat, a.paper, a.desc, a.notes, a.source].join(' ').toLowerCase();
      if (!f.q.toLowerCase().split(/\s+/).every(w => t.indexOf(w) >= 0)) return false;
    }
    return true;
  }
  const visible = () => allAlgos().filter(matches);

  function group(title, items, set, onChange, cls) {
    const box = $(cls);
    box.textContent = '';
    items.forEach(([val, label, count]) => {
      const id = cls + '-' + val.replace(/[^a-z0-9]/gi, '_');
      const cb = h('input', { type: 'checkbox', id, value: val });
      cb.checked = set.has(val);
      cb.addEventListener('change', () => { cb.checked ? set.add(val) : set.delete(val); onChange(); });
      box.appendChild(h('label', { for: id }, [cb, h('span', { text: label }), h('span', { class: 'c', text: String(count) })]));
    });
    if (!items.length) box.appendChild(h('p', { class: 'note', text: 'No values available.' }));
  }
  function countBy(fn) { const m = {}; allAlgos().forEach(a => { const v = fn(a); m[v] = (m[v] || 0) + 1; }); return m; }

  function buildFacets() {
    const onC = () => { renderCatalog(); syncFacetInputs(); };
    group('mod', S.mods.map(m => [m.id, MOD_NAME[m.id] || m.name, (S.algos[m.id] || []).length]), S.f.mods, onC, 'fx-mod');
    const st = countBy(a => a.status);
    group('status', Object.keys(st).sort().map(k => [k, ND(k), st[k]]), S.f.status, onC, 'fx-status');
    const sc = countBy(a => a.strat);
    group('strat', Object.keys(sc).sort((a, b) => sc[b] - sc[a] || a.localeCompare(b)).map(k => [k, ND(k), sc[k]]), S.f.strat, onC, 'fx-strat');
    const lc = countBy(classLic);
    group('lic', ['ok', 'warn', 'nd'].filter(k => lc[k]).map(k => [k, LIC_LABEL[k], lc[k]]), S.f.lic, onC, 'fx-lic');
    const tc = countBy(a => a.task || 'n/a');
    group('task', Object.keys(tc).sort().map(k => [k, ND(k), tc[k]]), S.f.task, onC, 'fx-task');
    const g = countBy(a => a.needs_gpu === true ? 'yes' : a.needs_gpu === false ? 'no' : 'nd');
    const gb = $('fx-gpu'); gb.textContent = '';
    [['', 'Any', allAlgos().length], ['no', 'Does not require GPU', g.no || 0], ['yes', 'Requires GPU', g.yes || 0]].forEach(([v, l, c]) => {
      const id = 'gpu-' + (v || 'any');
      const r = h('input', { type: 'radio', name: 'gpu', id, value: v });
      r.checked = S.f.gpu === v;
      r.addEventListener('change', () => { S.f.gpu = v; renderCatalog(); });
      gb.appendChild(h('label', { for: id }, [r, h('span', { text: l }), h('span', { class: 'c', text: String(c) })]));
    });
    if (g.nd) gb.appendChild(h('p', { class: 'note', text: g.nd + ' algorithms with no GPU data (n/a).' }));
    $('f-q').value = S.f.q; $('f-res').checked = S.f.withRes;
  }
  function syncFacetInputs() {
    [['fx-mod', S.f.mods], ['fx-status', S.f.status], ['fx-strat', S.f.strat], ['fx-lic', S.f.lic], ['fx-task', S.f.task]].forEach(([id, set]) => {
      $(id).querySelectorAll('input').forEach(i => { i.checked = set.has(i.value); });
    });
    document.querySelectorAll('input[name="gpu"]').forEach(r => { r.checked = S.f.gpu === r.value; });
    $('f-q').value = S.f.q; $('f-res').checked = S.f.withRes;
  }
  function resetFilters() {
    const f = S.f; f.q = ''; f.gpu = ''; f.withRes = false;
    ['mods', 'status', 'strat', 'lic', 'task'].forEach(k => f[k].clear());
    syncFacetInputs(); renderCatalog();
  }

  // ---------- Catalogo ----------
  function card(a) {
    const i = info(a), k = key(a), lc = classLic(a);
    const cb = h('input', { type: 'checkbox', id: 'sel-' + k.replace(/\W/g, '_'), 'aria-label': 'Select ' + a.name });
    cb.checked = S.sel.has(k);
    cb.addEventListener('change', () => { cb.checked ? S.sel.add(k) : S.sel.delete(k); updateSel(); el.classList.toggle('sel', cb.checked); });
    const st = a.status === 'executable' ? 'ok' : a.status === 'wired' ? 'warn' : 'line';
    const badges = h('div', { class: 'badges' }, [
      h('span', { class: 'b line', text: MOD_NAME[a.mod] || a.mod }),
      h('span', { class: 'b ' + st, text: a.status }),
      h('span', { class: 'b', text: ND(a.strat) }),
      a.needs_gpu === true ? h('span', { class: 'b warn', text: 'GPU' }) : a.needs_gpu === false ? h('span', { class: 'b ok', text: 'CPU only' }) : h('span', { class: 'b line', text: 'GPU n/a' }),
      h('span', { class: 'b ' + (lc === 'ok' ? 'ok' : lc === 'warn' ? 'warn' : 'line'), text: lc === 'nd' ? 'license n/a' : lc === 'warn' ? 'license to verify' : (a.license || '').slice(0, 24) }),
    ]);
    const nds = Object.keys(i.ok).length;
    const meas = i.nok ? h('p', { class: 'meas', text: pl(i.nok, 'completed run', 'completed runs') + ' on ' + pl(nds, 'dataset', 'datasets') + (i.nfb ? ' (' + i.nfb + ' as fallback)' : '') + (i.nfail ? ' (' + i.nfail + ' failed)' : '') })
      : h('p', { class: 'meas none', text: i.nfail ? 'No valid measurements (' + i.nfail + ' failed runs).' : 'No measurements available.' });
    const rows = [['License', a.license], ['Task', a.task], ['Paper', a.paper], ['Source', a.source], ['Environment', a.env], ['Notes', a.notes || a.desc]].filter(r => r[1]);
    const det = h('details', null, [h('summary', { text: 'Details' }), h('dl', null, rows.flatMap(r => [h('dt', { text: r[0] }), h('dd', { text: ND(String(r[1])) })]))]);
    const el = h('article', { class: 'card' + (cb.checked ? ' sel' : '') }, [
      h('h3', { text: a.name }), h('span', { class: 'pick' }, cb), badges,
      h('p', { text: a.notes || a.desc || 'No description in the catalog.' }), CatalogInfo.algoBlock(((S.info[a.mod] || {}).algorithms || {})[a.key]), meas, det]);
    return el;
  }
  function renderCatalog() {
    const list = visible();
    $('cnt-visible').textContent = String(list.length);
    $('cat-title').textContent = list.length + ' of ' + allAlgos().length + ' algorithms';
    const grid = $('grid'); grid.textContent = '';
    if (!allAlgos().length) grid.appendChild(h('div', { class: 'empty', text: 'The catalog is not available (see the notice above).' }));
    else if (!list.length) grid.appendChild(h('div', { class: 'empty' }, ['No algorithm matches the filters. ', h('button', { class: 'btn sm', type: 'button', onclick: resetFilters, text: 'Clear filters' })]));
    list.forEach(a => grid.appendChild(card(a)));
    const act = ['mods', 'status', 'strat', 'lic', 'task'].filter(k => S.f[k].size).length + (S.f.gpu ? 1 : 0) + (S.f.q ? 1 : 0) + (S.f.withRes ? 1 : 0);
    $('cat-note').textContent = act ? pl(act, 'active filter', 'active filters') : 'No active filters';
  }

  // ---------- Selezione ----------
  function updateSel() {
    const n = S.sel.size, nd = S.selDs.size;
    $('cnt-sel').textContent = String(n);
    $('cnt-ds').textContent = String(nd);
    $('tray').hidden = n === 0 && nd === 0;
    $('tray-n').textContent = pl(n, 'algorithm', 'algorithms') + ', ' + pl(nd, 'dataset', 'datasets');
    $('tray-cmp').disabled = n < 1;
    $('tray-run').disabled = n < 1;
    $('tray-msg').textContent = '';
    if (S.view === 'compare') renderCompare();
    if (S.view === 'run') renderRun();
  }
  function selectVisible(max) {
    const list = visible(); let n = 0;
    list.forEach(a => { if (n < (max || 12) && !S.sel.has(key(a))) { S.sel.add(key(a)); } n++; });
    renderCatalog(); updateSel();
    return { shown: list.length, taken: Math.min(list.length, max || 12) };
  }

  // ---------- Dataset ----------
  const dsKey = (mod, stem) => mod + ':' + stem;
  function renderDatasets() {
    const root = $('view-datasets'); root.textContent = '';
    root.appendChild(h('h2', { text: 'Dataset', style: 'margin-top:0' }));
    const q = h('input', { type: 'search', id: 'ds-q', placeholder: 'Search by name', value: S.dsq, 'aria-label': 'Search for a dataset', autocomplete: 'off' });
    q.addEventListener('input', () => { S.dsq = q.value.trim(); drawDs(); });
    root.appendChild(h('div', { class: 'ctl' }, [q, h('span', { class: 'note', text: 'Tick the datasets and use the bar at the bottom to send them to the Guided path or the Copilot. Numbers and uses come from catalog_info; if missing they appear as undocumented.' })]));
    const body = h('div', { id: 'ds-body' }); root.appendChild(body);
    function drawDs() {
      body.textContent = '';
      let total = 0;
      S.mods.forEach(m => {
        const list = (S.ds[m.id] || []).filter(d => !S.dsq || d.stem.toLowerCase().indexOf(S.dsq.toLowerCase()) >= 0);
        if (!list.length) return;
        total += list.length;
        const sec = h('section', { class: 'sect' }, [h('h3', { text: (MOD_NAME[m.id] || m.name) + ' (' + list.length + ')' })]);
        const grid = h('div', { class: 'grid' });
        list.forEach(d => {
          const k = dsKey(m.id, d.stem);
          const cb = h('input', { type: 'checkbox', id: 'dsel-' + k.replace(/\W/g, '_'), 'aria-label': 'Select ' + d.stem });
          cb.checked = S.selDs.has(k);
          const el = h('article', { class: 'card' + (cb.checked ? ' sel' : '') });
          cb.addEventListener('change', () => { cb.checked ? S.selDs.add(k) : S.selDs.delete(k); el.classList.toggle('sel', cb.checked); updateSel(); });
          const meta = [d.rows != null ? d.rows + ' rows' : '', d.columns != null ? d.columns + ' columns' : '', d.size_kb != null ? d.size_kb + ' KB' : ''].filter(Boolean).join(' - ');
          el.appendChild(h('h3', { text: d.stem })); el.appendChild(h('span', { class: 'pick' }, cb));
          if (meta) el.appendChild(h('p', { text: 'On disk: ' + meta }));
          el.appendChild(CatalogInfo.datasetBlock(((S.info[m.id] || {}).datasets || {})[d.stem]));
          grid.appendChild(el);
        });
        sec.appendChild(grid); body.appendChild(sec);
      });
      if (!total) body.appendChild(h('div', { class: 'empty', text: S.dsq ? 'No dataset matches the search.' : 'No dataset on disk.' }));
    }
    drawDs();
  }

  // ---------- Invio ad altre pagine ----------
  function sendTo(page) {
    const algos = [...S.sel].map(byKey).filter(Boolean);
    const dss = [...S.selDs].map(k => { const i = k.indexOf(':'); return { mod: k.slice(0, i), stem: k.slice(i + 1) }; });
    const mods = [...new Set(algos.map(a => a.mod).concat(dss.map(d => d.mod)))];
    const msg = $('tray-msg');
    if (!mods.length) { msg.textContent = 'Select at least one algorithm or dataset.'; return; }
    if (mods.length > 1) { msg.textContent = 'The selection mixes several modalities (' + mods.map(m => MOD_NAME[m] || m).join(', ') + '): only one is allowed.'; return; }
    Prefill.go(page, { mod: mods[0], datasets: dss.map(d => d.stem), algos: algos.map(a => a.key) });
  }

  // ---------- Confronto ----------
  function mean(xs) { return xs.length ? xs.reduce((s, v) => s + v, 0) / xs.length : null; }
  function renderCompare() {
    const root = $('view-compare'); root.textContent = '';
    root.appendChild(h('h2', { text: 'Side-by-side comparison', style: 'margin-top:0' }));
    const sel = [...S.sel].map(byKey).filter(Boolean);
    if (sel.length < 1) {
      root.appendChild(h('div', { class: 'empty' }, ['No algorithm selected. Tick the cards in the catalog (or ask the assistant: \"select the visible ones\").']));
      return;
    }
    root.appendChild(h('div', { class: 'ctl' }, [
      h('label', null, [(() => { const c = h('input', { type: 'checkbox' }); c.checked = S.cmpCommon; c.addEventListener('change', () => { S.cmpCommon = c.checked; renderCompare(); }); return c; })(),
        'Averages only over datasets common to all algorithms in the group']),
      h('span', { class: 'note', text: 'Hatched "not measured" cells mean missing data, not zeros. Averages cover successful runs only.' })]));
    ['tabular', 'ts', 'graph'].forEach(mod => {
      const g = sel.filter(a => a.mod === mod);
      if (!g.length) return;
      const sec = h('section', { class: 'sect' });
      sec.appendChild(h('h3', { text: (MOD_NAME[mod] || mod) + ' (' + g.length + ')' }));
      if (NOT_HERE[mod]) sec.appendChild(h('p', { class: 'note', text: NOT_HERE[mod] }));
      // dataset coperti
      const sets = g.map(a => new Set(Object.keys(info(a).ok)));
      let common = null; sets.forEach(s => { common = common === null ? new Set(s) : new Set([...common].filter(x => s.has(x))); });
      const useCommon = S.cmpCommon && g.length > 1;
      if (useCommon && common.size === 0 && sets.some(s => s.size)) sec.appendChild(h('div', { class: 'banner warn', text: 'No dataset is covered by all algorithms in this group: with the "common datasets" filter the metrics cannot be compared. Turn it off to see averages over different datasets (not directly comparable).' }));
      const head = h('tr', null, [h('th', { scope: 'col', text: 'Item' })].concat(g.map(a => h('th', { scope: 'col', text: a.name }))));
      const tb = h('tbody');
      const stat = (label, fn) => tb.appendChild(h('tr', null, [h('th', { scope: 'row', text: label })].concat(g.map(a => h('td', { text: ND(String(fn(a) || 'n/a')) })))));
      stat('Status', a => a.status); stat('Strategy', a => a.strat); stat('GPU', a => a.needs_gpu === true ? 'required' : a.needs_gpu === false ? 'not required' : null);
      stat('License', a => a.license); stat('Paper', a => a.paper);
      tb.appendChild(h('tr', null, [h('th', { scope: 'row', text: 'Datasets with measurements' })].concat(g.map(a => {
        const n = Object.keys(info(a).ok).length;
        return h('td', n ? null : { class: 'miss' }, n ? n + ' (' + Object.keys(info(a).ok).slice(0, 6).join(', ') + (n > 6 ? ', ...' : '') + ')' : 'not measured');
      }))));
      if (useCommon && common.size) tb.appendChild(h('tr', null, [h('th', { scope: 'row', text: 'Common datasets' })].concat([h('td', { colspan: g.length, text: [...common].join(', ') })])));
      (METRICS[mod] || []).forEach(([m, label, dir]) => {
        const vals = g.map(a => {
          const i = info(a), vs = [];
          Object.keys(i.ok).forEach(d => { if (useCommon && !common.has(d)) return; i.ok[d].forEach(r => { if (num(r[m])) vs.push(r[m]); }); });
          return { v: mean(vs), n: vs.length };
        });
        const present = vals.filter(x => x.v !== null);
        let best = null;
        if (dir && present.length > 1) best = (dir === 'up' ? Math.max : Math.min).apply(null, present.map(x => x.v));
        tb.appendChild(h('tr', null, [h('th', { scope: 'row' }, [label, dir ? h('small', { class: 'note', text: dir === 'up' ? ' (higher is better)' : ' (lower is better)' }) : h('small', { class: 'note', text: ' (direction not assigned)' })])]
          .concat(vals.map(x => x.v === null ? h('td', { class: 'miss', text: 'not measured' })
            : h('td', { class: 'num' + (best !== null && x.v === best ? ' best' : '') }, [fmt(x.v), h('small', { text: 'n=' + x.n })])))));
      });
      sec.appendChild(h('div', { class: 'tw' }, h('table', null, [h('caption', { class: 'sr', text: 'Comparison: ' + (MOD_NAME[mod] || mod) }), h('thead', null, head), tb])));
      root.appendChild(sec);
    });
  }

  // ---------- Matrice ----------
  function renderMatrix() {
    const root = $('view-matrix'); root.textContent = '';
    root.appendChild(h('h2', { text: 'Algorithm by dataset matrix', style: 'margin-top:0' }));
    const sel = h('select', { id: 'mx-mod', 'aria-label': 'Modality' });
    S.mods.forEach(m => { const o = h('option', { value: m.id, text: MOD_NAME[m.id] || m.name }); if (m.id === S.mxMod) o.selected = true; sel.appendChild(o); });
    sel.addEventListener('change', () => { S.mxMod = sel.value; renderMatrix(); });
    const onlyF = h('input', { type: 'checkbox', id: 'mx-f' }); onlyF.checked = S.mxFiltered !== false;
    onlyF.addEventListener('change', () => { S.mxFiltered = onlyF.checked; renderMatrix(); });
    root.appendChild(h('div', { class: 'ctl' }, [h('label', null, ['Modality ', sel]), h('label', null, [onlyF, 'Only algorithms that pass the catalog filters'])]));
    root.appendChild(h('div', { class: 'legend' }, [h('span', { class: 'b ok', text: 'N = successful runs' }), h('span', { class: 'b bad', text: 'x = failed only' }), h('span', { class: 'b line', text: '. = not measured (no data)' })]));
    const mod = S.mxMod, ds = (S.ds[mod] || []).map(d => d.stem);
    const extra = new Set(); Object.values(S.idx[mod] || {}).forEach(e => { Object.keys(e.ok).concat(Object.keys(e.fail)).forEach(d => { if (ds.indexOf(d) < 0) extra.add(d); }); });
    const cols = ds.concat([...extra]);
    let rows = (S.algos[mod] || []).filter(a => S.mxFiltered === false || matches(a));
    if (!cols.length || !rows.length) { root.appendChild(h('div', { class: 'empty', text: !rows.length ? 'No algorithm to show with the current filters.' : 'No dataset available for this modality.' })); return; }
    const detail = h('div', { id: 'mx-detail', class: 'panel', hidden: true, 'aria-live': 'polite' });
    const thead = h('tr', null, [h('th', { scope: 'col', text: 'Algorithm' })].concat(cols.map(d => h('th', { scope: 'col', title: d }, h('span', { text: d + (ds.indexOf(d) < 0 ? ' *' : '') })))));
    const tb = h('tbody');
    rows.forEach(a => {
      const i = info(a);
      tb.appendChild(h('tr', null, [h('th', { scope: 'row', text: a.name })].concat(cols.map(d => {
        const ok = (i.ok[d] || []).length, fl = i.fail[d] || 0;
        const lab = a.name + ' on ' + d + ': ' + (ok ? ok + ' successful' : fl ? fl + ' failed only' : 'not measured');
        if (!ok && !fl) return h('td', null, h('span', { class: 'cell none', role: 'img', 'aria-label': lab, title: lab, text: '.' }));
        return h('td', null, h('button', { type: 'button', class: 'cell ' + (ok ? 'ok' : 'fail'), 'aria-label': lab, title: lab, text: ok ? String(ok) : 'x',
          onclick: () => showCell(detail, a, d) }));
      }))));
    });
    root.appendChild(h('div', { class: 'tw' }, h('table', { class: 'mx' }, [h('caption', { class: 'sr', text: 'Algorithm by dataset matrix' }), h('thead', null, thead), tb])));
    if (extra.size) root.appendChild(h('p', { class: 'note', text: '* dataset present in the results but no longer in the data folder.' }));
    root.appendChild(detail);
  }
  function showCell(box, a, d) {
    box.hidden = false; box.textContent = '';
    const i = info(a), recs = (S.res[a.mod] || []).filter(r => r.algorithm === a.key && r.dataset === d);
    box.appendChild(h('h3', { text: a.name + ' on ' + d }));
    const metrics = (METRICS[a.mod] || []).map(m => m[0]);
    const cols = ['status', 'file', 'timestamp', 'epochs'].concat(metrics.slice(0, 6)).filter(c => recs.some(r => r[c] != null));
    const tb = h('tbody');
    recs.forEach(r => tb.appendChild(h('tr', null, cols.map(c => h('td', { class: num(r[c]) ? 'num' : null, text: num(r[c]) ? fmt(r[c]) : String(r[c] == null ? '-' : r[c]) })))));
    box.appendChild(h('div', { class: 'tw' }, h('table', null, [h('thead', null, h('tr', null, cols.map(c => h('th', { scope: 'col', text: c })))), tb])));
    const err = recs.find(r => r.error_message);
    if (err) box.appendChild(h('p', { class: 'note', text: 'Error: ' + String(err.error_message).slice(0, 300) }));
    if (!i.ok[d] && !recs.length) box.appendChild(h('p', { text: 'No records.' }));
  }

  // ---------- Esegui ----------
  function renderRun() {
    const root = $('view-run'); root.textContent = '';
    root.appendChild(h('h2', { text: 'Run the selection', style: 'margin-top:0' }));
    const sel = [...S.sel].map(byKey).filter(Boolean);
    const mods = [...new Set(sel.map(a => a.mod))];
    if (!sel.length) { root.appendChild(h('div', { class: 'empty', text: 'Select one or more algorithms in the catalog.' })); root.appendChild(taskPanel()); return; }
    if (mods.length > 1) { root.appendChild(h('div', { class: 'banner warn', text: 'The selection mixes several modalities (' + mods.map(m => MOD_NAME[m]).join(', ') + '). A run covers a single modality: narrow the selection.' })); root.appendChild(taskPanel()); return; }
    const mod = mods[0];
    if (mod === 'tabular') {
      root.appendChild(h('div', { class: 'banner' }, ['For tabular data this API does not launch runs (run_modality is only for time series and graphs). Use the assistant (with the key active it asks for confirmation) or the', h('a', { href: '/guided', text: ' Guided path' }), '.']));
      root.appendChild(taskPanel()); return;
    }
    const run = sel.filter(a => a.status === 'executable' || a.runnable), skip = sel.filter(a => !(a.status === 'executable' || a.runnable));
    if (skip.length) root.appendChild(h('div', { class: 'banner warn', text: 'Not runnable (described only): ' + skip.map(a => a.name).join(', ') + '. They will be excluded.' }));
    const dsList = S.ds[mod] || [];
    const form = h('div', { class: 'panel' });
    form.appendChild(h('h3', { text: 'Algorithms (' + run.length + ')' }));
    form.appendChild(h('p', { text: run.map(a => a.name).join(', ') || 'none' }));
    form.appendChild(h('h3', { text: 'Datasets' }));
    const chk = h('div', { class: 'chk', role: 'group', 'aria-label': 'Datasets' });
    if (!dsList.length) chk.appendChild(h('p', { class: 'note', text: 'No dataset available for this modality.' }));
    dsList.forEach(d => {
      const id = 'rd-' + d.stem.replace(/\W/g, '_');
      chk.appendChild(h('label', { for: id }, [h('input', { type: 'checkbox', id, value: d.stem, name: 'rds' }), d.stem + ' (' + d.size_kb + ' KB)']));
    });
    form.appendChild(chk);
    const f = (id, label, el) => h('label', { for: id, style: 'display:flex;gap:8px;align-items:center;margin-top:8px' }, [label, el]);
    form.appendChild(f('r-ep', 'Epochs', h('input', { type: 'number', id: 'r-ep', min: '1', max: '5000', value: '5' })));
    form.appendChild(f('r-dev', 'Device', (() => { const s = h('select', { id: 'r-dev' }); ['auto', 'cpu', 'cuda'].forEach(v => s.appendChild(h('option', { value: v, text: v }))); return s; })()));
    form.appendChild(f('r-seed', 'Seed', h('input', { type: 'number', id: 'r-seed', value: '0' })));
    form.appendChild(f('r-n', 'Samples (optional)', h('input', { type: 'number', id: 'r-n', min: '0', placeholder: 'default' })));
    const go = h('button', { type: 'button', class: 'btn primary', style: 'margin-top:12px', text: 'Prepare the run...' });
    go.disabled = !run.length || !dsList.length;
    go.addEventListener('click', () => {
      const dsSel = [...root.querySelectorAll('input[name="rds"]:checked')].map(i => i.value);
      if (!dsSel.length) { alert('Select at least one dataset.'); return; }
      confirmRun({ modality: mod, algorithms: run.map(a => a.key), datasets: dsSel, epochs: +$('r-ep').value || 5, device: $('r-dev').value, seed: +$('r-seed').value || 0, n_samples: $('r-n').value ? +$('r-n').value : undefined });
    });
    form.appendChild(go);
    const wrap = h('div', { class: 'run' }, [form, taskPanel()]);
    root.appendChild(wrap);
  }
  function taskPanel() {
    const p = h('div', { class: 'panel', id: 'task-panel' }, [h('h3', { text: 'Task' })]);
    p.appendChild(h('p', { id: 'task-state', class: 'note', text: S.lastTask ? 'Task ' + S.lastTask : 'No task started from this page.' }));
    p.appendChild(h('div', { class: 'ctl' }, [h('button', { type: 'button', class: 'btn sm', id: 'task-stop', text: 'Stop the task', disabled: !S.lastTask, onclick: stopTask })]));
    p.appendChild(h('pre', { class: 'log', id: 'task-log', 'aria-live': 'off', text: '' }));
    return p;
  }
  function confirmRun(plan) {
    const n = plan.algorithms.length * plan.datasets.length;
    const b = $('dlg-body'); b.textContent = '';
    b.appendChild(h('p', { text: 'Modality: ' + MOD_NAME[plan.modality] + '. ' + plan.algorithms.length + ' algorithms x ' + plan.datasets.length + ' datasets = ' + n + ' combinations, ' + plan.epochs + ' epochs, device ' + plan.device + '.' }));
    b.appendChild(h('p', null, ['Algorithms: ', h('code', { text: plan.algorithms.join(', ') })]));
    b.appendChild(h('p', null, ['Datasets: ', h('code', { text: plan.datasets.join(', ') })]));
    b.appendChild(h('p', { class: 'note', text: 'The run may take a lot of compute time (GPU). It starts only after your confirmation.' }));
    const dlg = $('dlg'), ok = $('dlg-ok'), no = $('dlg-no');
    ok.disabled = false;
    ok.onclick = async () => {
      ok.disabled = true;
      try {
        const r = await fetch('/api/run_modality', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(Object.assign({ confirmed: true }, plan)) });
        const j = await r.json();
        dlg.close();
        if (!j.success) { setTask(null, 'Error: ' + (j.error || r.status)); return; }
        S.lastTask = j.task_id; setTask(j.task_id, 'Task ' + j.task_id + ' started.'); pollTask(j.task_id, 0);
      } catch (e) { dlg.close(); setTask(null, 'Connection: ' + e); }
    };
    no.onclick = () => dlg.close();
    if (dlg.showModal) dlg.showModal(); else dlg.setAttribute('open', '');
    ok.focus();
  }
  function setTask(id, msg) {
    const st = $('task-state'); if (st) st.textContent = msg;
    const sb = $('task-stop'); if (sb) sb.disabled = !id;
  }
  let pollTimer = null;
  async function pollTask(id, from) {
    clearTimeout(pollTimer);
    try {
      const j = await getJSON('/api/logs/' + encodeURIComponent(id) + '?start=' + from);
      const lg = $('task-log');
      if (lg && j.lines && j.lines.length) { lg.textContent += j.lines.join('\n') + '\n'; lg.scrollTop = lg.scrollHeight; }
      setTask(id, 'Task ' + id + ': ' + j.status + (j.duration_s ? ' (' + j.duration_s + ' s)' : ''));
      if (['RUNNING', 'PENDING', 'QUEUED'].indexOf(String(j.status).toUpperCase()) >= 0) { pollTimer = setTimeout(() => pollTask(id, j.next_index), 2000); }
      else if (String(j.status).toUpperCase() !== 'UNKNOWN') { setTask(id, 'Task ' + id + ' finished: ' + j.status + '. Reloading results.'); await reloadResults(); }
    } catch (e) { setTask(id, 'Reading logs: ' + e.message); }
  }
  async function stopTask() {
    if (!S.lastTask) return;
    await fetch('/api/stop_task/' + encodeURIComponent(S.lastTask), { method: 'POST' }).catch(() => {});
  }
  async function reloadResults() {
    await Promise.all(S.mods.map(async m => {
      try { S.res[m.id] = (await getJSON('/api/modality/' + m.id + '/results?all=1')).records || []; indexResults(m.id); } catch (e) { /* mantiene i vecchi */ }
    }));
    renderAll();
  }

  // ---------- Viste ----------
  function setView(v) {
    S.view = v;
    document.querySelectorAll('.tab').forEach(t => { const on = t.dataset.view === v; t.setAttribute('aria-selected', on); });
    ['catalog', 'datasets', 'compare', 'matrix', 'run'].forEach(n => { $('view-' + n).hidden = n !== v; });
    if (v === 'datasets') renderDatasets(); if (v === 'compare') renderCompare(); if (v === 'matrix') renderMatrix(); if (v === 'run') renderRun();
  }
  function renderAll() { renderCatalog(); updateSel(); setView(S.view); }

  // ---------- Assistente: comandi locali + chat ----------
  const A = { chat: null, offline: true, status: null };
  function say(cls, text) { const d = h('div', { class: 'msg ' + cls, text }); $('a-log').appendChild(d); $('a-log').scrollTop = $('a-log').scrollHeight; }
  function describeFilters() {
    const f = S.f, p = [];
    if (f.mods.size) p.push('modality: ' + [...f.mods].map(m => MOD_NAME[m]).join(', '));
    if (f.status.size) p.push('status: ' + [...f.status].join(', '));
    if (f.strat.size) p.push('strategies: ' + [...f.strat].join(', '));
    if (f.gpu) p.push(f.gpu === 'no' ? 'without GPU' : 'with GPU');
    if (f.lic.size) p.push('license: ' + [...f.lic].map(k => LIC_LABEL[k]).join(', '));
    if (f.withRes) p.push('with results');
    if (f.q) p.push('text "' + f.q + '"');
    return p.length ? p.join('; ') : 'no filters';
  }
  // Interpreta pochi comandi deterministici. Restituisce il testo di risposta oppure null se non riconosciuto.
  function localCommand(raw) {
    const t = ' ' + raw.toLowerCase().replace(/['`]/g, ' ') + ' ';
    const verb = /(mostra|mostrami|filtra|filtrami|seleziona|azzera|resetta|togli|confronta|solo|fammi vedere|vedi|elenca|esegui|lancia|\b(?:show|display|filter|select|reset|clear|remove|compare|only|list|launch|run)\b)/.test(t);
    if (!verb) return null;
    let acted = [];
    const sel = /(seleziona|confronta|\bselect\b|\bcompare\b)/.test(t);
    if (/(azzera|resetta|togli i filtri|rimuovi i filtri|\b(?:clear|reset|remove)\s+(?:the\s+|all\s+)*filters\b|\breset\b)/.test(t) && !sel) { resetFilters(); acted.push('filters cleared'); }
    else {
      if (/\btabular/.test(t) || /\btabellar/.test(t)) { S.f.mods = new Set(['tabular']); acted.push('tabular modality'); }
      else if (/(serie|temporal|\bts\b|time.?series)/.test(t)) { S.f.mods = new Set(['ts']); acted.push('time series modality'); }
      else if (/\b(?:graf|graph)/.test(t)) { S.f.mods = new Set(['graph']); acted.push('graphs modality'); }
      if (/(senza gpu|no gpu|solo cpu|su cpu|non richied\w+ gpu|without gpu|cpu.?only|only cpu|on cpu|\b(?:not|t)\s+(?:need|require)\w*\s+gpu)/.test(t)) { S.f.gpu = 'no'; acted.push('without GPU'); }
      else if (/(con gpu|richied\w+ gpu|with gpu|(?:need|require)\w*\s+gpu|gpu required)/.test(t)) { S.f.gpu = 'yes'; acted.push('with GPU'); }
      if (/generativ/.test(t)) {
        S.f.strat = new Set([...new Set(allAlgos().map(a => a.strat))].filter(s => GENERATIVE.test(s)));
        acted.push('generative strategies (' + [...S.f.strat].join(', ') + ')');
      }
      if (/(senza licenza problematic|licenza non problematic|non.?problematic\s+licen[sc]e|no\s+problematic\s+licen[sc]e|licen[sc]e\s+(?:ok|free|permissive|clear)|(?:no|without)\s+licen[sc]e\s+(?:problems|issues)|licenza (ok|libera|permissiva|chiara)|senza problemi di licenza)/.test(t)) {
        S.f.lic = new Set(['ok', 'nd']); acted.push('excluded cases with a license to verify (declared license and n/a remain: missing data is not a guarantee)');
      }
      if (/(eseguibil|avviabil|executable|runnable)/.test(t)) { S.f.status = new Set(['executable']); acted.push('executable only'); }
      if (/(con risultati|misurat|con misure|con dati|with results|measured|with measurements|with data)/.test(t)) { S.f.withRes = true; acted.push('only with measured results'); }
    }
    if (!acted.length && !sel) return null;
    syncFacetInputs(); renderCatalog();
    let msg = acted.length ? 'Filters applied: ' + acted.join('; ') + '. ' : '';
    if (sel) {
      if (/(visibil|tutti|questi|quelli|\bvisible\b|\ball\b|\bthese\b|\bthose\b)/.test(t) || acted.length) {
        const r = selectVisible(12);
        msg += 'Selected ' + r.taken + ' of ' + r.shown + ' visible' + (r.shown > 12 ? ' (limit of 12 for a readable comparison)' : '') + '.';
        if (/confronta|\bcompare\b/.test(t)) { setView('compare'); msg += ' Opening the comparison.'; }
      } else if (/confronta|\bcompare\b/.test(t)) { setView('compare'); msg += 'Opening the comparison of the current selection (' + S.sel.size + ').'; }
      else msg += 'Tell me which ones, e.g. "select the visible ones".';
    }
    return msg + ' Now: ' + visible().length + ' algorithms (' + describeFilters() + ').';
  }
  function chips() {
    const box = $('a-chips'); box.textContent = '';
    const add = (label, fn) => box.appendChild(h('button', { type: 'button', class: 'chip', text: label, onclick: fn }));
    const cmd = txt => () => { say('user', txt); const r = localCommand(txt); say('loc', r || 'Command not recognized.'); };
    add('Generative', cmd('show the generative ones'));
    add('No GPU', cmd('show me those without GPU'));
    add('Non-problematic license', cmd('show me those with no problematic license'));
    add('Only with results', cmd('show only those with measured results'));
    add('Select visible', cmd('select the visible ones'));
    add('Clear', cmd('clear the filters'));
    add('Recommender (tool)', () => {
      const mod = S.f.mods.size === 1 ? [...S.f.mods][0] : (S.sel.size ? byKey([...S.sel][0]).mod : 'ts');
      A.chat.runTool('recommend_algorithms', { modality: mod, executable_only: true, top_k: 5 });
    });
    add('Measured comparison (tool)', () => {
      const sel = [...S.sel].map(byKey).filter(Boolean); const mod = sel.length ? sel[0].mod : (S.f.mods.size === 1 ? [...S.f.mods][0] : 'ts');
      const a = sel.filter(x => x.mod === mod).map(x => x.key);
      A.chat.runTool('compare_results', a.length ? { modality: mod, algorithms: a } : { modality: mod });
    });
  }
  function initAssistant() {
    const log = $('a-log'), inp = $('a-input'), send = $('a-send');
    // Intercettazione: i comandi locali passano prima della chat (listener registrati prima di AssistantChat.create).
    const tryLocal = ev => {
      const t = inp.value.trim(); if (!t) return false;
      const r = localCommand(t);
      if (r !== null) { ev.preventDefault(); ev.stopImmediatePropagation(); say('user', t); say('loc', r); inp.value = ''; return true; }
      if (A.offline) { ev.preventDefault(); ev.stopImmediatePropagation(); say('user', t); inp.value = '';
        say('warn', 'Assistant offline: no model is responding and I do not simulate answers. I can run commands on the filters (e.g. "show me the generative ones without GPU", "select the visible ones with results and compare") and the shortcuts above.'); return true; }
      return false;
    };
    inp.addEventListener('keydown', ev => { if (ev.key === 'Enter' && !ev.shiftKey && !ev.isComposing) tryLocal(ev); });
    send.addEventListener('click', ev => { tryLocal(ev); });
    A.chat = window.AssistantChat.create({
      log, input: inp, sendBtn: send, welcome: false,
      sysText: 'I can filter and select from the catalog (local commands) and, if the model is active, reason about your problem and launch runs with your confirmation.',
      onResults: mod => { S.mxMod = mod; setView('matrix'); },
      onStatus: st => {
        A.status = st; A.offline = !st || st.mode === 'offline';
        const s = $('a-state');
        s.className = 'a-state' + (A.offline ? '' : ' on');
        s.textContent = !st ? 'Status not readable: the assistant is not responding. Local filter commands still work.'
          : A.offline ? 'Offline: ' + (st.reason || 'no model available') + '. Filter/selection commands and the deterministic tools still work.'
            : 'Online (' + (st.backend || '') + ' ' + (st.model || '') + ').';
        $('a-dot').classList.toggle('on', !A.offline);
      },
    });
    A.chat.intro(); chips(); A.chat.refreshStatus();
    const tg = $('a-toggle'), box = $('assistant');
    tg.addEventListener('click', () => {
      const min = box.classList.toggle('min');
      tg.setAttribute('aria-expanded', String(!min)); tg.textContent = min ? 'Open' : 'Minimize';
      if (!min) inp.focus();
    });
    box.addEventListener('keydown', ev => { if (ev.key === 'Escape' && !box.classList.contains('min')) { tg.click(); tg.focus(); } });
  }

  // ---------- Avvio ----------
  function init() {
    document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => setView(t.dataset.view)));
    $('f-q').addEventListener('input', e => { S.f.q = e.target.value.trim(); renderCatalog(); });
    $('f-res').addEventListener('change', e => { S.f.withRes = e.target.checked; renderCatalog(); });
    $('f-reset').addEventListener('click', resetFilters);
    $('sel-visible').addEventListener('click', () => selectVisible(12));
    $('sel-none').addEventListener('click', () => { S.sel.clear(); renderCatalog(); updateSel(); });
    $('tray-clear').addEventListener('click', () => { S.sel.clear(); S.selDs.clear(); renderCatalog(); if (S.view === 'datasets') renderDatasets(); updateSel(); });
    $('tray-guided').addEventListener('click', () => sendTo('/guided'));
    $('tray-copilot').addEventListener('click', () => sendTo('/copilot'));
    $('tray-cmp').addEventListener('click', () => setView('compare'));
    $('tray-run').addEventListener('click', () => setView('run'));
    if (window.matchMedia('(max-width: 900px)').matches) $('facet-wrap').open = false;
    initAssistant();
    load();
  }
  init();
})();
