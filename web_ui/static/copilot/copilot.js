/* Proposta 1 - Copilota a tutto schermo. JS vanilla, nessuna dipendenza.
 * API usate: /api/modalities, /api/modality/<m>/algorithms|datasets, /api/assistant/status|chat|confirm|task|reset, /api/stop_task. */
(function () {
  'use strict';

  var MODS = [];                       // [{id,name,counts,n_datasets}]
  var S = { mod: 'tabular', algos: [], datasets: [], dataset: null, picked: [], sid: null, busy: false,
            status: null, tab: 'algo', offlineShown: false, info: { algorithms: {}, datasets: {} } };
  var $ = function (id) { return document.getElementById(id); };
  var log = $('log'), input = $('composer-input');

  /* ---------- utilita' DOM ---------- */
  function h(tag, attrs, kids) {
    var e = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      var v = attrs[k];
      if (v === null || v === undefined || v === false) return;
      if (k === 'class') e.className = v;
      else if (k === 'text') e.textContent = v;
      else if (k.indexOf('on') === 0 && typeof v === 'function') e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? '' : v);
    });
    (kids || []).forEach(function (c) { if (c !== null && c !== undefined && c !== false) e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); });
    return e;
  }
  function scrollEnd() { log.scrollTop = log.scrollHeight; }
  function put(node) { var hello = log.querySelector('.hello'); if (hello) hello.remove(); log.appendChild(node); scrollEnd(); return node; }
  function jget(url) { return fetch(url).then(function (r) { return r.json().then(function (j) { if (!r.ok && !j.error) j.error = 'HTTP ' + r.status; return j; }); }); }
  function jpost(url, body) {
    return fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) })
      .then(function (r) { return r.json().then(function (j) { if (!r.ok && !j.error) j.error = 'HTTP ' + r.status; return j; }); });
  }
  function fmt(v) {
    if (v === null || v === undefined || v === '') return '-';
    if (typeof v === 'number') return Number.isInteger(v) ? String(v) : (Math.abs(v) >= 100 ? v.toFixed(1) : v.toPrecision(4).replace(/\.?0+$/, ''));
    if (typeof v === 'object') return Array.isArray(v) ? v.join(', ') : Object.keys(v).map(function (k) { return k + ': ' + fmt(v[k]); }).join(' | ');
    return String(v);
  }
  var STATUS_TXT = { executable: 'executable', wired: 'wired', described: 'described only' };
  function badge(st) { return h('span', { class: 'badge b-' + st, text: STATUS_TXT[st] || st || '?' }); }
  var MOD_LABEL = { tabular: 'Tabular', ts: 'Time series', graph: 'Graphs' };
  function modName(id) { var m = MODS.filter(function (x) { return x.id === id; })[0]; return MOD_LABEL[id] || (m ? m.name : id); }

  /* ---------- barra: modalita' e stato del modello ---------- */
  function renderMods() {
    var seg = $('mod-seg'); seg.innerHTML = '';
    MODS.forEach(function (m) {
      var b = h('button', { type: 'button', role: 'radio', 'aria-checked': String(m.id === S.mod), text: MOD_LABEL[m.id] || m.name,
        title: m.counts ? (m.counts.total + ' algorithms, ' + m.n_datasets + ' datasets') : '', onclick: function () { setMod(m.id); } });
      seg.appendChild(b);
    });
  }
  function setMod(id) {
    if (id === S.mod && S.algos.length) return;
    S.mod = id; S.dataset = null; S.picked = [];
    try { localStorage.setItem('copilot_mod', id); } catch (e) { /* ignora */ }
    renderMods(); renderCtx(); loadCatalog();
  }
  function refreshStatus() {
    return jget('/api/assistant/status').then(function (st) {
      S.status = st; var p = $('model-pill');
      if (st.mode === 'online') { p.className = 'pill pill-on'; p.textContent = 'Model: ' + (st.model || st.backend || 'active'); p.title = st.backend || ''; }
      else { p.className = 'pill pill-off'; p.textContent = 'Offline: no model'; p.title = st.reason || ''; }
      $('foot-note').textContent = st.mode === 'online' ? '' : 'Offline: use the shortcuts above.';
      $('composer-input').placeholder = st.mode === 'online' ? 'Describe your problem: data, goal, constraints...' : 'Offline: the assistant does not respond. Use the shortcuts.';
    }).catch(function () {
      var p = $('model-pill'); p.className = 'pill pill-off'; p.textContent = 'Status unavailable';
    });
  }

  /* ---------- cassetto: catalogo ---------- */
  /* Caricamento di un dataset proprio (tabulare CSV, serie temporali .npz; per i grafi solo un avviso). */
  function renderUpload() {
    var slot = $('upload-slot'); if (!slot || !window.Upload) return; slot.innerHTML = '';
    slot.appendChild(Upload.box(S.mod, function (d) {
      S.dataset = d.name; renderCtx();
      note('Dataset "' + d.name + '" uploaded and selected: you can now analyze it or ask for a recommendation.', 'ok');
      jget('/api/modality/' + S.mod + '/datasets').then(function (r) { S.datasets = r.datasets || []; renderDrawer(); });
    }));
  }
  function loadCatalog() {
    $('panel-algo').innerHTML = ''; $('panel-data').innerHTML = '';
    renderUpload();
    $('panel-algo').appendChild(h('div', { class: 'empty', text: 'Loading...' }));
    var modNow = S.mod;
    CatalogInfo.load(modNow).then(function (i) { if (S.mod === modNow) { S.info = i; renderDrawer(); } });
    jget('/api/modality/' + S.mod + '/algorithms').then(function (r) {
      S.algos = r.algorithms || []; S.algoErr = r.error || r.catalog_error || null; renderDrawer();
    }).catch(function (e) { S.algos = []; S.algoErr = String(e); renderDrawer(); });
    jget('/api/modality/' + S.mod + '/datasets').then(function (r) {
      S.datasets = r.datasets || []; S.dataErr = r.error || null; renderDrawer();
    }).catch(function (e) { S.datasets = []; S.dataErr = String(e); renderDrawer(); });
  }
  function renderDrawer() {
    var q = ($('drawer-filter').value || '').toLowerCase();
    var pa = $('panel-algo'), pd = $('panel-data'); pa.innerHTML = ''; pd.innerHTML = '';
    var al = S.algos.filter(function (a) { return (a.key + ' ' + (a.name || '') + ' ' + (a.strategy || '')).toLowerCase().indexOf(q) >= 0; });
    if (S.algoErr) pa.appendChild(h('div', { class: 'empty', text: 'Catalog unavailable: ' + S.algoErr }));
    else if (!al.length) pa.appendChild(h('div', { class: 'empty', text: S.algos.length ? 'No algorithm matches the filter.' : 'No algorithm in the catalog for this modality.' }));
    al.forEach(function (a) {
      var on = S.picked.indexOf(a.key) >= 0;
      pa.appendChild(h('button', { type: 'button', class: 'item', 'aria-pressed': String(on),
        title: (CatalogInfo.algoTitle(S.info.algorithms[a.key]) ? CatalogInfo.algoTitle(S.info.algorithms[a.key]) + (CatalogInfo.unverified(S.info.algorithms[a.key]) ? ' (to be verified). ' : '. ') : '') + (a.runnable ? 'Click: add to or remove from the context' : 'Described only: it cannot be launched, but you can ask about it'),
        onclick: function () { togglePick(a.key); } }, [
        h('span', { class: 't' }, [h('span', { text: a.name || a.key }), badge(a.status)]),
        h('span', { class: 's', text: [a.strategy, a.needs_gpu ? 'GPU' : ''].filter(Boolean).join(' - ') || a.key })]));
    });
    var dl = S.datasets.filter(function (d) { return d.stem.toLowerCase().indexOf(q) >= 0; });
    if (S.dataErr) pd.appendChild(h('div', { class: 'empty', text: 'Datasets unavailable: ' + S.dataErr }));
    else if (!dl.length) pd.appendChild(h('div', { class: 'empty', text: S.datasets.length ? 'No dataset matches the filter.' : 'No dataset on disk for this modality.' }));
    dl.forEach(function (d) {
      var on = S.dataset === d.stem;
      var b = h('button', { type: 'button', class: 'item', draggable: 'true', 'aria-pressed': String(on), title: ((S.info.datasets[d.stem] || {}).uses ? 'Uses: ' + (Array.isArray(S.info.datasets[d.stem].uses) ? S.info.datasets[d.stem].uses.join('; ') : S.info.datasets[d.stem].uses) + '. ' : '') + 'Click: select (or drag into the conversation)',
        onclick: function () { pickDataset(on ? null : d.stem); } }, [
        h('span', { class: 't' }, [h('span', { text: d.stem }), d.user ? h('span', { class: 'badge b-executable', text: 'yours' }) : null]),
        h('span', { class: 's', text: [d.rows ? d.rows + ' rows' : '', d.columns ? d.columns + ' columns' : '', d.size_kb !== undefined ? d.size_kb + ' KB' : ''].filter(Boolean).join(' - ') })]);
      b.addEventListener('dragstart', function (ev) { ev.dataTransfer.setData('text/plain', 'dataset:' + d.stem); ev.dataTransfer.effectAllowed = 'copy'; });
      pd.appendChild(b);
    });
  }
  function selectTab(t) {
    S.tab = t;
    [['algo', 'tab-algo', 'panel-algo'], ['data', 'tab-data', 'panel-data']].forEach(function (x) {
      var on = x[0] === t; $(x[1]).setAttribute('aria-selected', String(on)); $(x[1]).tabIndex = on ? 0 : -1; $(x[2]).hidden = !on;
    });
  }
  function togglePick(k) { var i = S.picked.indexOf(k); if (i >= 0) S.picked.splice(i, 1); else S.picked.push(k); renderCtx(); renderDrawer(); }
  function pickDataset(name) { S.dataset = name; renderCtx(); renderDrawer(); }

  /* ---------- contesto sopra il campo di scrittura ---------- */
  function renderCtx() {
    var c = $('ctx'); c.innerHTML = '';
    if (S.dataset) c.appendChild(h('span', { class: 'chip' }, ['Dataset: ' + S.dataset, h('button', { type: 'button', 'aria-label': 'Remove dataset ' + S.dataset, onclick: function () { pickDataset(null); }, text: '×' })]));
    S.picked.forEach(function (k) {
      c.appendChild(h('span', { class: 'chip' }, [k, h('button', { type: 'button', 'aria-label': 'Remove algorithm ' + k, onclick: function () { togglePick(k); }, text: '×' })]));
    });
  }
  function ctxText() {
    var p = ['modality: ' + S.mod];
    if (S.dataset) p.push('dataset: ' + S.dataset);
    if (S.picked.length) p.push('algorithms: ' + S.picked.join(', '));
    return p.join('; ');
  }

  /* ---------- scorciatoie senza modello ---------- */
  var SHORT = [
    ['Algorithms', function () { tool('list_algorithms', { modality: S.mod }); }],
    ['Dataset', function () { tool('list_datasets', { modality: S.mod }); }],
    ['Recommend', function () { var a = { modality: S.mod, top_k: 5 }; if (S.dataset) a.dataset = S.dataset; tool('recommend_algorithms', a); }],
    ['Analyze dataset', function () {
      if (!S.dataset) return note('Choose a dataset first from the drawer (Datasets tab).');
      tool('analyze_dataset', { modality: S.mod, dataset: S.dataset }); }],
    ['Compare results', function () { var a = { modality: S.mod }; if (S.dataset) a.dataset = S.dataset; if (S.picked.length) a.algorithms = S.picked.slice(); tool('compare_results', a); }],
    ['Prepare run', function () { planCard(); }]
  ];
  function renderShortcuts() {
    var box = $('shortcuts'); box.innerHTML = '';
    SHORT.forEach(function (s) { box.appendChild(h('button', { type: 'button', class: 'chip', text: s[0], onclick: s[1] })); });
  }
  function note(t, cls) { return put(h('div', { class: 'msg ' + (cls || 'sys'), text: t })); }

  /* ---------- chat: SSE ---------- */
  function setBusy(b) { S.busy = b; $('send').disabled = b; $('shortcuts').querySelectorAll('button').forEach(function (x) { x.disabled = b; }); }
  var workingCards = {};
  function post(body) {
    setBusy(true);
    var payload = Object.assign({ session_id: S.sid }, body);
    return fetch('/api/assistant/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      .then(function (resp) {
        if (!resp.body) throw new Error('stream unavailable');
        var rd = resp.body.getReader(), dec = new TextDecoder(), buf = '';
        function pump() {
          return rd.read().then(function (x) {
            if (x.done) return;
            buf += dec.decode(x.value, { stream: true });
            var i;
            while ((i = buf.indexOf('\n\n')) >= 0) {
              var chunk = buf.slice(0, i); buf = buf.slice(i + 2);
              if (chunk.indexOf('data: ') === 0) { try { onEvent(JSON.parse(chunk.slice(6))); } catch (e) { /* chunk non valido */ } }
            }
            return pump();
          });
        }
        return pump();
      })
      .catch(function (e) { note('Connection: ' + e, 'err'); })
      .then(function () { setBusy(false); });
  }
  function onEvent(ev) {
    switch (ev.type) {
      case 'session': S.sid = ev.session_id; break;
      case 'text': put(window.Md ? Md.render(h('div', { class: 'msg bot' }), ev.text) : h('div', { class: 'msg bot', text: ev.text })); break;
      case 'offline': put(h('div', { class: 'msg warn', text: ev.text })); break;
      case 'warning': note(ev.text, 'warn'); break;
      case 'limit': note('Stopped: ' + ev.text, 'warn'); break;
      case 'error': note('Error: ' + ev.text, 'err'); break;
      case 'tool_call': workingCards[ev.id] = put(h('div', { class: 'card working', role: 'status' }, [
        h('header', null, [h('h2', { text: 'Using: ' + toolLabel(ev.name) })])])); break;
      case 'tool_result': var w = workingCards[ev.id]; delete workingCards[ev.id];
        var card = renderResult(ev.name, ev.result || {}, ev.ok);
        if (card) { if (w && w.isConnected) { w.replaceWith(card); scrollEnd(); } else put(card); } else if (w) w.remove(); break;
      case 'pending': put(pendingCard(ev)); break;
    }
  }
  var LABELS = { list_algorithms: 'algorithm catalog', describe_algorithm: 'algorithm sheet', list_datasets: 'dataset list', analyze_dataset: 'dataset analysis',
    recommend_algorithms: 'recommender', run_augmentation: 'run', get_task_status: 'task status', evaluate: 'evaluation', compare_results: 'results comparison', explain_metrics: 'metrics explanation', explain_recommendation: 'why behind a recommendation' };
  function toolLabel(n) { return LABELS[n] || n; }
  function tool(name, args) { if (S.busy) return; return post({ offline_action: { tool: name, args: args } }); }

  function send(text) {
    text = (text || '').trim(); if (!text || S.busy) return;
    var bub = h('div', { class: 'msg user' }, [text]);
    var c = ctxText(); if (S.dataset || S.picked.length) bub.appendChild(h('span', { class: 'uctx', text: 'Context: ' + c }));
    put(bub);
    var body = (S.dataset || S.picked.length) ? text + '\n\n[Context chosen by the user: ' + c + ']' : text;
    input.value = ''; autosize();
    post({ message: body });
  }

  /* ---------- carte ---------- */
  function card(title, tag, body, acts, cls) {
    return h('section', { class: 'card ' + (cls || ''), 'aria-label': title }, [
      h('header', null, [h('h2', { text: title }), tag ? h('span', { class: 'tag', text: tag }) : null]),
      h('div', { class: 'body' }, body || []), acts && acts.length ? h('div', { class: 'acts' }, acts) : null]);
  }
  function btn(t, fn, primary) { return h('button', { type: 'button', class: 'btn sm' + (primary ? ' primary' : ''), text: t, onclick: fn }); }
  function rawDetails(obj) { return h('details', { class: 'more' }, [h('summary', { text: 'Raw data' }), h('pre', { text: JSON.stringify(obj, null, 2) })]); }

  function renderResult(name, r, ok) {
    if (name === 'explain_recommendation' && r.error && window.Evidence) return card('Why? - no explanation', null, [Evidence.explainPanel(r, null, {})], null, 'bad');
    if (!ok || r.error) return card('Error: ' + toolLabel(name), null, [h('p', { text: r.error || 'The tool returned no result.' }), r.available_for_dataset ? h('p', { class: 'note', text: 'Available: ' + r.available_for_dataset.join(', ') }) : null], null, 'bad');
    switch (name) {
      case 'list_algorithms': return algoListCard(r);
      case 'list_datasets': return dataListCard(r);
      case 'analyze_dataset': return analyzeCard(r);
      case 'recommend_algorithms': return recCard(r);
      case 'explain_recommendation': return card('Why ' + r.algorithm + ' - ' + modName(r.modality), 'profile ' + Evidence.profileLabel(r.profile), [Evidence.explainPanel(r, null, { weights: r.weights })]);
      case 'compare_results': return compareCard(r);
      case 'run_augmentation': case 'evaluate':
        if (r.status === 'started' && r.task_id) return runCard(r.task_id, r.command);
        if (r.status === 'needs_confirmation') return null;   // arriva l'evento 'pending'
        return card(toolLabel(name), r.status || null, [h('p', { text: r.message || 'No details.' }), rawDetails(r)]);
      case 'get_task_status': return card('Task status ' + (r.task_id || ''), r.status, [kv({ status: r.status, duration_s: r.duration_s, error: r.error }), r.log_tail && r.log_tail.length ? h('pre', { class: 'tail', text: r.log_tail.join('\n') }) : null]);
      default: return card(toolLabel(name), null, [kv(r), rawDetails(r)]);
    }
  }
  function kv(o) {
    var dl = Stats.list(o, { cls: 'kv' });
    return dl.children.length ? dl : h('p', { class: 'note', text: 'No details to show.' });
  }
  function algoListCard(r) {
    var items = r.algorithms || []; var c = r.counts_by_status || {};
    var chips = h('div', { class: 'chips' });
    items.forEach(function (a) {
      var b = h('button', { type: 'button', class: 'chip', title: (STATUS_TXT[a.status] || a.status) + (a.strategy ? ' - ' + a.strategy : ''), onclick: function () { if (S.picked.indexOf(a.key) < 0) togglePick(a.key); } },
        [a.name || a.key, ' ', badge(a.status)]);
      chips.appendChild(b);
    });
    return card('Algorithms - ' + modName(r.modality), Object.keys(c).map(function (k) { return c[k] + ' ' + (STATUS_TXT[k] || k); }).join(', '),
      [items.length ? chips : h('p', { class: 'empty', text: 'No algorithm in the catalog.' }), r.warnings && r.warnings.length ? h('p', { class: 'note', text: r.warnings.join(' ') }) : null]);
  }
  function dataListCard(r) {
    var items = r.datasets || []; var chips = h('div', { class: 'chips' });
    items.forEach(function (d) {
      var b = h('button', { type: 'button', class: 'chip', draggable: 'true', title: 'Select', onclick: function () { pickDataset(d.name || d.stem); } }, [d.name || d.stem, d.size_kb !== undefined ? ' (' + d.size_kb + ' KB)' : '']);
      chips.appendChild(b);
    });
    return card('Dataset - ' + modName(r.modality || S.mod), items.length + ' on disk', [items.length ? chips : h('p', { class: 'empty', text: 'No dataset on disk for this modality.' })],
      items.length ? [btn('Analyze the selected one', function () { if (S.dataset) tool('analyze_dataset', { modality: S.mod, dataset: S.dataset }); else note('Choose a dataset first.'); })] : null);
  }
  function analyzeCard(r) {
    var keep = {}; Object.keys(r).forEach(function (k) { if (['column_names', 'modality', 'file', 'note', 'label_column_basis'].indexOf(k) < 0) keep[k] = r[k]; });
    var cols = r.column_names && r.column_names.length ? h('details', { class: 'more' }, [h('summary', { text: 'Columns (' + r.column_names.length + ')' }), h('pre', { text: r.column_names.join(', ') })]) : null;
    var ds = r.file ? r.file.replace(/^.*[\\/]/, '').replace(/\.[^.]+$/, '') : null;
    return card('Dataset' + (ds ? ': ' + ds : ''), modName(r.modality || S.mod),
      [kv(keep), r.note ? h('p', { class: 'note', text: r.note }) : null, cols],
      [ds ? btn('Use in context', function () { pickDataset(ds); }) : null, btn('Recommend algorithms', function () { var a = { modality: S.mod, top_k: 5 }; if (ds) a.dataset = ds; tool('recommend_algorithms', a); }, true)].filter(Boolean));
  }
  function recCard(r) {
    var rk = r.ranking || [], un = r.unranked || [];
    var max = rk.reduce(function (m, x) { return Math.max(m, Number(x.score) || 0); }, 0) || 1;
    var ol = h('ol', { class: 'rank' });
    var kb = r.evidence === 'kb';
    rk.forEach(function (x) {
      var ev = (x.evidence || []).map(function (e) { return e.metric + ' = ' + fmt(e.value) + '  (' + e.source + ')'; }).join('\n');
      var hasC = kb && x.contributions && typeof x.score === 'number';
      var cb = hasC ? Evidence.contribBars({ score: x.score, contributions: x.contributions, axis_scores: x.axis_scores }, r.weights) : null;
      var ds = r.inputs && r.inputs.dataset;
      var whyBtn = kb ? btn('Why?', function () {
        if (!ds) { note('The explanation needs a dataset: choose one in the drawer and ask for the recommendation again.'); return; }
        var a = { modality: r.modality, dataset: ds, priority: r.profile || 'balanced', algorithm: x.algorithm };
        ['task', 'gpu', 'time_budget_s', 'need_privacy', 'need_fairness', 'need_constraints', 'executable_only'].forEach(function (k) { if (r.inputs && r.inputs[k] !== undefined) a[k] = r.inputs[k]; });
        tool('explain_recommendation', a);
      }) : null;
      var li = h('li', null, [h('span', { class: 'n', text: String(x.rank) }),
        h('span', { class: 'name' }, [x.name || x.algorithm, ' ', badge(x.status), hasC ? h('span', { class: 'note', text: ' score ' + fmt(x.score) }) : null]),
        h('span', { class: 'sc' }, [whyBtn, btn(S.picked.indexOf(x.algorithm) >= 0 ? 'In context' : 'Add', function () { if (S.picked.indexOf(x.algorithm) < 0) togglePick(x.algorithm); this.textContent = 'In context'; })]),
        hasC ? h('span', { style: 'grid-column:2/4' }, [cb.querySelector('.evx-track')])
          : h('span', { class: 'bar-bg', role: 'img', 'aria-label': 'score ' + fmt(x.score) }, [h('span', { class: 'bar-fg', style: 'display:block;width:' + Math.max(3, Math.round(100 * (Number(x.score) || 0) / max)) + '%' })]),
        h('span', { class: 'why', text: (x.reasons || []).join(' ') + (x.warnings && x.warnings.length ? ' Warning: ' + x.warnings.join(' ') : '') })]);
      if (hasC) li.appendChild(h('details', { class: 'more', style: 'grid-column:2/4' }, [h('summary', { text: 'Contributions per axis (sum = score)' }), cb]));
      if (ev) li.appendChild(h('details', { class: 'more', style: 'grid-column:2/4' }, [h('summary', { text: 'Source of the numbers' }), h('pre', { text: ev })]));
      ol.appendChild(li);
    });
    var body = [kb && r.weights ? Evidence.axisLegend(r.weights) : null, r.evidence === 'nessuna' && window.Evidence ? Evidence.fallbackNote(r.method) : null, rk.length ? ol : h('p', { class: 'empty', text: 'No algorithm has measured results for these criteria: I cannot rank.' })];
    if (un.length) body.push(h('details', { class: 'more' }, [h('summary', { text: 'Not ranked, no measurements (' + un.length + ')' }),
      h('pre', { text: un.map(function (u) { return (u.algorithm || u.name || u) + (u.reason ? ' - ' + u.reason : u.status ? ' - ' + u.status : ''); }).join('\n') })]));
    (r.notes || []).forEach(function (n) { body.push(h('p', { class: 'note', text: n })); });
    return card('Recommendation - ' + modName(r.modality), rk.length ? 'top ' + rk.length + ' of ' + (r.n_ranked_total || rk.length) : null, body,
      [btn('Prepare run', function () { if (!S.picked.length && rk.length) { togglePick(rk[0].algorithm); } planCard(); }, true), rawDetailsBtn(r)]);
  }
  function rawDetailsBtn(r) { return btn('Raw data', function () { note(JSON.stringify(r, null, 1).slice(0, 4000)); }); }

  var PREF = ['KL_mean', 'JS_mean', 'WD_mean', 'KS_mean', 'MMD', 'pMSE', 'Precision', 'Recall', 'Coverage', 'total_time_s_median'];
  function compareCard(r) {
    var rows = r.rows || [];
    if (!rows.length) return card('Results comparison', modName(r.modality), [h('p', { class: 'empty', text: r.message || 'No measured results.' })]);
    var idKeys = ['dataset', 'algorithm', 'algo', 'ds', 'file', 'status'];
    var all = []; rows.forEach(function (x) { Object.keys(x).forEach(function (k) { if (all.indexOf(k) < 0 && typeof x[k] === 'number') all.push(k); }); });
    var shown = PREF.filter(function (k) { return all.indexOf(k) >= 0; }); if (!shown.length) shown = all.slice(0, 6);
    var sortKey = null, asc = true, expanded = false;
    var wrap = h('div', { class: 'tablewrap' });
    function draw() {
      var cols = expanded ? all : shown;
      var ids = idKeys.filter(function (k) { return rows.some(function (x) { return x[k] !== undefined; }); }).slice(0, 3);
      var data = rows.slice();
      if (sortKey) data.sort(function (a, b) { var x = a[sortKey], y = b[sortKey]; if (x === undefined) return 1; if (y === undefined) return -1; return asc ? x - y : y - x; });
      var thead = h('tr', null, ids.map(function (k) { return h('th', { scope: 'col', text: k }); }).concat(cols.map(function (k) {
        return h('th', { scope: 'col', 'aria-sort': sortKey === k ? (asc ? 'ascending' : 'descending') : 'none' }, [h('button', { type: 'button', title: 'Sort', text: k + (sortKey === k ? (asc ? ' ▲' : ' ▼') : ''), onclick: function () { if (sortKey === k) asc = !asc; else { sortKey = k; asc = true; } draw(); } })]);
      })));
      var tb = h('tbody'); data.forEach(function (x) { tb.appendChild(h('tr', null, ids.map(function (k) { return h('td', { text: fmt(x[k]) }); }).concat(cols.map(function (k) { return h('td', { text: fmt(x[k]) }); })))); });
      wrap.innerHTML = ''; wrap.appendChild(h('table', { class: 't' }, [h('caption', { class: 'visually-hidden', text: 'Comparison of the measured metrics' }), h('thead', null, [thead]), tb]));
    }
    draw();
    var toggle = btn('Show all columns', function () { expanded = !expanded; toggle.textContent = expanded ? 'Show main columns' : 'Show all columns'; draw(); });
    var srcs = {}; rows.forEach(function (x) { if (x.sources) srcs[x.sources] = 1; if (x.source) srcs[x.source] = 1; });
    return card('Results comparison - ' + modName(r.modality), rows.length + ' rows', [wrap,
      r.truncated ? h('p', { class: 'note', text: r.truncated }) : null,
      h('p', { class: 'note', text: 'The "better" direction is not given for every metric: ask the assistant for the verified explanation. Sources: ' + (Object.keys(srcs).slice(0, 3).join('; ') || 'not given') + '.' })], [toggle]);
  }

  /* ---------- esecuzione: piano, conferma, avanzamento ---------- */
  function planCard() {
    var runnable = S.picked.filter(function (k) { var a = S.algos.filter(function (x) { return x.key === k; })[0]; return a && a.runnable; });
    var skipped = S.picked.filter(function (k) { return runnable.indexOf(k) < 0; });
    var miss = []; if (!S.dataset) miss.push('a dataset (Datasets tab of the drawer)'); if (!runnable.length) miss.push('at least one executable algorithm (Algorithms tab)');
    if (miss.length) return put(card('Prepare run', modName(S.mod), [h('p', { class: 'empty', text: 'To launch you still need: ' + miss.join(' and ') + '.' }), skipped.length ? h('p', { class: 'note', text: 'Excluded because they are described only: ' + skipped.join(', ') }) : null]));
    var ep = h('input', { type: 'number', min: '1', max: '5000', value: '5', id: 'ep' + Date.now() });
    var ns = h('input', { type: 'number', min: '1', placeholder: 'default' });
    var dv = h('select', null, [h('option', { value: 'cpu', text: 'CPU' }), h('option', { value: 'cuda', text: 'GPU (cuda)' })]);
    var out = h('div'), box;
    var go = btn('Confirm and launch', function () {
      var nEp = parseInt(ep.value, 10); if (!(nEp >= 1)) { out.textContent = 'Invalid epochs.'; return; }
      go.disabled = true; cancel.disabled = true;
      var args = { modality: S.mod, algorithms: runnable, datasets: [S.dataset], epochs: nEp, device: dv.value }; if (ns.value) args.n_samples = parseInt(ns.value, 10);
      out.textContent = 'Sending...';
      tool('run_augmentation', args); out.textContent = 'Request sent.';
    }, true);
    var cancel = btn('Cancel', function () { box.remove(); note('Run cancelled, nothing was launched.'); });
    box = card('Confirm run', modName(S.mod), [
      kv({ algorithms: runnable.join(', '), dataset: S.dataset }),
      h('div', { class: 'form' }, [h('label', null, ['Epochs', ep]), h('label', null, ['Samples to generate', ns]), h('label', null, ['Device', dv])]),
      skipped.length ? h('p', { class: 'note', text: 'Excluded because they are described only: ' + skipped.join(', ') }) : null,
      h('p', { class: 'note', text: 'A real process starts on your computer and may take a long time. Nothing starts until you confirm. Beyond the server thresholds a second confirmation is requested.' }), out], [go, cancel], 'ask');
    return put(box);
  }
  function pendingCard(ev) {
    var out = h('p', { class: 'note' }), ok, no, box;
    function answer(approve) {
      ok.disabled = no.disabled = true;
      jpost('/api/assistant/confirm', { session_id: S.sid, pending_id: ev.pending_id, approve: approve }).then(function (j) {
        if (j.error) { out.textContent = 'Error: ' + j.error; return; }
        if (!approve) { out.textContent = 'Cancelled: nothing was launched.'; return; }
        out.textContent = 'Confirmed.'; if (j.task_id) put(runCard(j.task_id, j.command));
      }).catch(function (e) { out.textContent = 'Connection: ' + e; });
    }
    ok = btn('Confirm and launch', function () { answer(true); }, true); no = btn('Cancel', function () { answer(false); });
    box = card('Your confirmation is needed', ev.reason, [kv(ev.plan || {}), h('p', { class: 'note', text: 'Threshold exceeded (' + ev.reason + '). Nothing has started yet.' }), out], [ok, no], 'ask');
    return box;
  }
  function runCard(taskId, command) {
    var st = h('span', { class: 'tag', text: 'running...' }), prog = h('div', { class: 'prog', role: 'progressbar', 'aria-label': 'Progress' }, [h('i')]), tail = h('pre', { class: 'tail', text: 'Waiting for logs...' }), meta = h('p', { class: 'note' });
    var stop = btn('Stop', function () { jpost('/api/stop_task/' + encodeURIComponent(taskId)); stop.disabled = true; });
    var cmp = btn('Compare results', function () { var a = { modality: S.mod }; if (S.dataset) a.dataset = S.dataset; tool('compare_results', a); }, true); cmp.hidden = true;
    var box = h('section', { class: 'card', 'aria-label': 'Run ' + taskId }, [
      h('header', null, [h('h2', { text: 'Run ' + taskId }), st]), h('div', { class: 'body' }, [prog, meta, tail,
        h('details', { class: 'more' }, [h('summary', { text: 'Command' }), h('pre', { text: command || 'not given' })])]), h('div', { class: 'acts' }, [stop, cmp])]);
    var fails = 0;
    (function poll() {
      if (!box.isConnected && fails > 0) return;
      jget('/api/assistant/task/' + encodeURIComponent(taskId) + '?session_id=' + encodeURIComponent(S.sid || '')).then(function (t) {
        if (t.error) { fails++; st.textContent = 'status unreadable'; meta.textContent = t.error; if (fails < 4) setTimeout(poll, 4000); return; }
        fails = 0;
        var done = ['SUCCESS', 'FAILED', 'STOPPED'].indexOf(t.status) >= 0;
        st.textContent = { PENDING: 'queued', RUNNING: 'running', SUCCESS: 'completed', FAILED: 'failed', STOPPED: 'stopped' }[t.status] || t.status;
        meta.textContent = 'Duration: ' + fmt(t.duration_s) + ' s' + (t.error ? ' - ' + t.error : '');
        if (t.log_tail && t.log_tail.length) { tail.textContent = t.log_tail.join('\n'); tail.scrollTop = tail.scrollHeight; }
        if (done) { prog.className = 'prog ' + (t.status === 'SUCCESS' ? 'done' : 'fail'); stop.hidden = true; cmp.hidden = t.status !== 'SUCCESS'; }
        else setTimeout(poll, 2500);
      }).catch(function () { fails++; if (fails < 4) setTimeout(poll, 4000); });
    })();
    return box;
  }

  /* ---------- avvio e eventi ---------- */
  function autosize() { input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 160) + 'px'; }
  function hello() {
    var st = h('div', { class: 'starters' });
    ['I have an imbalanced dataset: which algorithm do you recommend?', 'Which datasets can I augment in this modality?', 'Compare the algorithms using the measured results'].forEach(function (t) {
      st.appendChild(h('button', { type: 'button', class: 'chip', text: t, onclick: function () { input.value = t; send(t); } }));
    });
    log.appendChild(h('div', { class: 'hello' }, [h('h1', { text: 'What do you want to augment today?' }),
      h('p', { text: 'Describe the problem: I help you choose data and algorithm, launch, evaluate and understand the results. I only use measured results; if a piece of data is missing, I will tell you.' }), st,
      h('p', { id: 'hello-off', class: 'note' })]));
  }
  function bind() {
    $('composer').addEventListener('submit', function (e) { e.preventDefault(); send(input.value); });
    input.addEventListener('keydown', function (e) { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(input.value); } });
    input.addEventListener('input', autosize);
    $('drawer-filter').addEventListener('input', renderDrawer);
    $('tab-algo').addEventListener('click', function () { selectTab('algo'); });
    $('tab-data').addEventListener('click', function () { selectTab('data'); });
    document.querySelector('.tabs').addEventListener('keydown', function (e) {
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { var t = S.tab === 'algo' ? 'data' : 'algo'; selectTab(t); $('tab-' + t).focus(); }
    });
    function drawer(open) {
      $('drawer').hidden = !open; $('drawer-btn').setAttribute('aria-expanded', String(open));
      $('scrim').hidden = !(open && window.matchMedia('(max-width: 860px)').matches);
    }
    $('drawer-btn').addEventListener('click', function () { drawer($('drawer').hidden); });
    $('scrim').addEventListener('click', function () { drawer(false); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && !$('drawer').hidden && window.matchMedia('(max-width: 860px)').matches) { drawer(false); $('drawer-btn').focus(); } });
    if (window.matchMedia('(max-width: 860px)').matches) drawer(false);
    $('reset').addEventListener('click', function () {
      var old = S.sid; S.sid = null;
      if (old) jpost('/api/assistant/reset', { session_id: old }).catch(function () {});
      log.innerHTML = ''; hello(); fillHelloOff();
    });
    // trascinamento dei dataset
    [log, $('composer')].forEach(function (t) {
      t.addEventListener('dragover', function (e) { e.preventDefault(); t.classList.add(t === log ? 'dropping' : 'drop'); });
      t.addEventListener('dragleave', function () { t.classList.remove('dropping', 'drop'); });
      t.addEventListener('drop', function (e) {
        e.preventDefault(); t.classList.remove('dropping', 'drop');
        var d = (e.dataTransfer.getData('text/plain') || ''); if (d.indexOf('dataset:') === 0) { pickDataset(d.slice(8)); note('Dataset added to the context: ' + d.slice(8)); }
      });
    });
  }
  function fillHelloOff() {
    var el = $('hello-off'); if (!el || !S.status) return;
    el.textContent = S.status.mode === 'online' ? '' : 'No model is responding at the moment (' + (S.status.reason || 'reason not given') + '). The shortcuts below use the tools directly, without a model.';
  }
  /* Selezione ricevuta dal Catalogo (?mod=&datasets=&algos=): riempie il contesto e prepara un primo messaggio che la cita.
   * Il messaggio NON parte da solo: resta nel campo di scrittura, lo invia l'utente. */
  function applyPrefill(pf) {
    Prefill.clean();
    S.dataset = pf.datasets[0] || null; S.picked = pf.algos.slice();
    renderCtx(); renderDrawer();
    var parts = ['From the Catalog I chose the modality ' + modName(pf.mod)];
    if (pf.datasets.length) parts.push((pf.datasets.length > 1 ? 'the datasets ' : 'the dataset ') + pf.datasets.join(', '));
    if (pf.algos.length) parts.push((pf.algos.length > 1 ? 'the algorithms ' : 'the algorithm ') + pf.algos.join(', '));
    var ask = pf.algos.length && pf.datasets.length ? 'Is this choice suitable? What do you recommend I use, and why?'
      : pf.datasets.length ? 'Which algorithms do you recommend for these data?' : 'For which data is it worth using them, and when not?';
    input.value = parts.join('; ') + '. ' + ask; autosize();
    var p = h('p', { class: 'note', id: 'hello-pf', text: 'Selection received from the Catalog: the first message is already in the input field, press Send whenever you want.' });
    var hello = log.querySelector('.hello'); if (hello) hello.appendChild(p);
    input.focus();
  }
  function init() {
    renderShortcuts(); hello(); bind(); renderCtx();
    var saved = null; try { saved = localStorage.getItem('copilot_mod'); } catch (e) { /* ignora */ }
    var pf = Prefill.read();
    jget('/api/modalities').then(function (m) {
      MODS = Array.isArray(m) ? m : [];
      if (!MODS.length) { note('No modality available from the server.', 'err'); return; }
      S.mod = (pf && MODS.some(function (x) { return x.id === pf.mod; })) ? pf.mod : (MODS.some(function (x) { return x.id === saved; }) ? saved : MODS[0].id);
      renderMods(); loadCatalog();
      if (pf && pf.mod === S.mod) applyPrefill(pf);
    }).catch(function (e) { note('Cannot read the modalities: ' + e, 'err'); });
    refreshStatus().then(fillHelloOff);
    setInterval(refreshStatus, 60000);
  }
  init();
})();
