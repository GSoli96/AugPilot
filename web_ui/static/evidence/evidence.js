/* Pagina «Prove»: legge /api/kb/summary?modality=...&level=... e /api/modality/<m>/algorithms (famiglia di ogni algoritmo).
 * Grafico qualita'/tempo in SVG scritto a mano (nessuna libreria da CDN: funziona anche offline), matrice dataset x algoritmo in una tabella HTML.
 * Il punteggio e' quello del consigliere: somma peso x punteggio d'asse, con i pesi del profilo rinormalizzati sugli assi che hanno misure
 * (i pesi arrivano dall'API: assistant/recommender.PROFILES). Un asse senza misura per un algoritmo vale 0, come nel consigliere. */
(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';
  var $ = function (id) { return document.getElementById(id); };
  var MODS = [['tabular', 'Tabular'], ['ts', 'Time series'], ['graph', 'Graphs']];
  var LEVELS = [['graph', 'Whole graphs'], ['node', 'Nodes of a graph']];
  var PROFILES = [['balanced', 'Balanced'], ['utility', 'Utility'], ['fidelity', 'Fidelity'], ['diversity', 'Diversity'], ['privacy', 'Privacy'], ['speed', 'Speed']];
  var AXES = ['utility', 'fidelity', 'diversity', 'privacy', 'speed'];
  var AXIS_LABEL = { utility: 'Utility', fidelity: 'Fidelity', diversity: 'Diversity', privacy: 'Privacy', speed: 'Speed' };
  var SERIES = ['var(--s1)', 'var(--s2)', 'var(--s3)'];
  var OTHER = 'var(--s-other)';
  var SHAPES = ['circle', 'square', 'triangle', 'diamond'];
  var S = { mod: null, level: 'graph', profile: 'balanced', counts: {}, summary: null, catalog: {}, hidden: {}, w: {}, seq: 0 };
  var NF = new Intl.NumberFormat('en-US', { maximumFractionDigits: 3, minimumFractionDigits: 3 });
  var NI = new Intl.NumberFormat('en-US');
  function n3(v) { return v === null || v === undefined || !isFinite(v) ? '–' : NF.format(v); }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined && text !== null) e.textContent = text; return e; }
  function sv(tag, attrs, text) {
    var e = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function fmtTime(s) { if (s === null || s === undefined) return '–'; if (s < 10) return NF.format(s).replace(/\.?0+$/, '') + ' s'; if (s < 3600) return Math.round(s) + ' s'; return (s / 3600).toFixed(1) + ' h'; }
  function api(url) { return fetch(url).then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(j.error || ('HTTP ' + r.status)); return j; }); }); }

  function shortName(n) { var t = String(n).replace(/\s+\(.*$/, ''); return t.length > 22 ? t.slice(0, 21) + '…' : t; }

  /* ---------- punteggio ---------- */
  function availableAxes(sum) {
    return AXES.filter(function (a) { return sum.algorithms.some(function (x) { return x.axis_scores && x.axis_scores[a] !== null && x.axis_scores[a] !== undefined; }); });
  }
  function weights(sum, profile) {
    var decl = (sum.profiles || {})[profile] || {}, av = availableAxes(sum), w = {}, tot = 0;
    av.forEach(function (a) { w[a] = decl[a] || 0; tot += w[a]; });
    if (!tot) { av.forEach(function (a) { w[a] = 1 / av.length; }); return w; }
    av.forEach(function (a) { w[a] /= tot; });
    return w;
  }
  function score(axisScores, w) { var t = 0; Object.keys(w).forEach(function (a) { t += w[a] * ((axisScores || {})[a] || 0); }); return t; }

  /* ---------- colori ---------- */
  var RAMP_L = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];
  var RAMP_D = ['#1d2535', '#184f95', '#1c5cab', '#2a78d6', '#3987e5', '#6da7ec', '#9ec5f4'];
  function isDark() {
    var r = document.documentElement.getAttribute('data-theme');
    if (r) return r === 'dark';
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  function hex(h) { return [1, 3, 5].map(function (i) { return parseInt(h.slice(i, i + 2), 16); }); }
  function rampAt(v, dark) {
    var R = dark ? RAMP_D : RAMP_L, t = Math.max(0, Math.min(1, v)) * (R.length - 1), i = Math.min(R.length - 2, Math.floor(t)), f = t - i;
    var a = hex(R[i]), b = hex(R[i + 1]);
    return a.map(function (x, k) { return Math.round(x + (b[k] - x) * f); });
  }
  function lum(c) { var f = function (x) { x /= 255; return x <= 0.03928 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4); }; return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]); }

  /* ---------- controlli ---------- */
  function radio(box, items, current, onPick, counts) {
    box.innerHTML = '';
    items.forEach(function (it) {
      var b = el('button'); b.type = 'button'; b.setAttribute('role', 'radio'); b.setAttribute('aria-checked', String(it[0] === current));
      b.appendChild(document.createTextNode(it[1]));
      if (counts && counts[it[0]] !== undefined) b.appendChild(el('span', 'cnt', counts[it[0]] === null ? '' : '(' + NI.format(counts[it[0]]) + ' runs)'));
      b.addEventListener('click', function () { onPick(it[0]); });
      box.appendChild(b);
    });
  }
  function drawControls() {
    radio($('seg-mod'), MODS, S.mod, setMod, S.counts);
    $('fld-level').hidden = S.mod !== 'graph';
    radio($('seg-level'), LEVELS, S.level, function (l) { S.level = l; load(); });
    radio($('seg-prof'), PROFILES, S.profile, function (p) { S.profile = p; render(); drawControls(); });
  }
  function setMod(m) { if (m === S.mod) return; S.mod = m; S.hidden = {}; drawControls(); load(); }

  /* ---------- dati ---------- */
  function loadCatalog(mod) {
    if (S.catalog[mod]) return Promise.resolve(S.catalog[mod]);
    return api('/api/modality/' + mod + '/algorithms').then(function (j) {
      var m = {}; (j.algorithms || []).forEach(function (a) { m[String(a.key).toLowerCase()] = { name: a.name || a.key, strategy: a.strategy || '' }; });
      S.catalog[mod] = m; return m;
    }).catch(function () { S.catalog[mod] = {}; return {}; });
  }
  function load() {
    var seq = ++S.seq, mod = S.mod;
    $('ev-status').textContent = 'Loading evidence...';
    var q = '/api/kb/summary?modality=' + mod + (mod === 'graph' ? '&level=' + S.level : '');
    Promise.all([api(q), loadCatalog(mod)]).then(function (r) {
      if (seq !== S.seq) return;
      S.summary = r[0]; $('ev-status').textContent = ''; render();
    }).catch(function (e) {
      if (seq !== S.seq) return;
      S.summary = null; $('ev-body').hidden = true; $('ev-empty').hidden = false;
      $('ev-empty').innerHTML = ''; $('ev-empty').appendChild(el('h2', null, 'Cannot read the evidence'));
      $('ev-empty').appendChild(el('p', null, String(e.message || e)));
      $('ev-status').textContent = '';
    });
  }

  /* ---------- stato vuoto ---------- */
  function drawEmpty(sum) {
    var box = $('ev-empty'); box.innerHTML = ''; box.hidden = false; $('ev-body').hidden = true;
    var name = (MODS.filter(function (m) { return m[0] === S.mod; })[0] || [])[1] || S.mod;
    box.appendChild(el('h2', null, 'No evidence for "' + name + '"' + (S.mod === 'graph' ? ' (' + (LEVELS.filter(function (l) { return l[0] === S.level; })[0][1]).toLowerCase() + ')' : '')));
    box.appendChild(el('p', null, sum && sum.message ? sum.message + '.' : 'The benchmark has not measured anything here yet.'));
    box.appendChild(el('p', null, 'Without evidence the recommender cannot rank by measurements: for this modality the recommendations in the Guided path and the Copilot are a fallback, ordered with run data only, and they say so.'));
    var p = el('p'); p.appendChild(document.createTextNode('The evidence comes from the benchmark in progress; when the results are ready, rebuild the knowledge base with '));
    p.appendChild(el('code', null, 'python -m assistant.kb.build_kb')); p.appendChild(document.createTextNode('.'));
    box.appendChild(p);
    var others = MODS.filter(function (m) { return m[0] !== S.mod && S.counts[m[0]]; });
    if (others.length) box.appendChild(el('p', null, 'With evidence: ' + others.map(function (m) { return m[1] + ' (' + NI.format(S.counts[m[0]]) + ' runs)'; }).join(', ') + '.'));
  }

  /* ---------- rendering ---------- */
  function render() {
    var sum = S.summary; if (!sum) return;
    if (!sum.available || !sum.algorithms.length) { drawEmpty(sum); return; }
    $('ev-empty').hidden = true; $('ev-body').hidden = false;
    var w = S.w = weights(sum, S.profile), cat = S.catalog[S.mod] || {};
    var algos = sum.algorithms.map(function (a) {
      var c = cat[a.algorithm] || {};
      return { key: a.algorithm, name: c.name || a.algorithm, short: shortName(c.name || a.algorithm), family: c.strategy || 'Not in catalog', score: score(a.axis_scores, w), time: a.time_s_median, n_runs: a.n_runs, n_ds: a.n_datasets, fail: a.failure_rate, axes: a.axis_scores };
    }).sort(function (a, b) { return b.score - a.score; });
    drawKpis(sum, algos);
    var wtxt = Object.keys(w).map(function (a) { return AXIS_LABEL[a] + ' ' + Math.round(100 * w[a]) + '%'; }).join(', ');
    var prof = (PROFILES.filter(function (p) { return p[0] === S.profile; })[0] || [])[1];
    $('sub-scatter').textContent = 'Each point is an algorithm: mean score across datasets (profile ' + prof + ': ' + wtxt + ') against median run time, on a logarithmic scale. Top left is better. Color indicates the algorithm\'s family (strategy).';
    drawScatter(algos);
    drawHeat(sum, algos, w);
    var foot = 'Source: ' + ((sum.kb || {}).file || 'benchmark KB') + '. ' + (sum.note ? sum.note.charAt(0).toUpperCase() + sum.note.slice(1) + '. ' : '') + 'The score of an axis with no measurements for an algorithm is 0, as in the recommender; axes with no measurements at all are dropped and the weights renormalized.';
    var miss = AXES.filter(function (a) { return !(a in w); });
    if (miss.length) foot += ' Axes without measurements in this selection: ' + miss.map(function (a) { return AXIS_LABEL[a]; }).join(', ') + '.';
    $('ev-foot').textContent = foot;
  }
  function drawKpis(sum, algos) {
    var k = $('ev-kpis'); k.innerHTML = '';
    var runs = (sum.kb || {}).runs;
    [[runs !== undefined ? runs : algos.reduce(function (t, a) { return t + a.n_runs; }, 0), 'runs measured'], [sum.datasets.length, 'datasets covered'], [algos.length, 'algorithms covered'],
      [Math.round(100 * (algos.reduce(function (t, a) { return t + (a.fail || 0) * a.n_runs; }, 0) / Math.max(1, algos.reduce(function (t, a) { return t + a.n_runs; }, 0)))) + '%', 'failed runs']].forEach(function (x) {
      var d = el('div', 'ev-kpi'); d.appendChild(el('div', 'v', typeof x[0] === 'number' ? NI.format(x[0]) : x[0])); d.appendChild(el('div', 'l', x[1])); k.appendChild(d);
    });
  }

  /* ---------- tooltip ---------- */
  var tip = null;
  function showTip(ev, title, rows, anchor) {
    tip = tip || $('ev-tip'); tip.innerHTML = ''; tip.appendChild(el('b', null, title));
    var dl = el('dl'); rows.forEach(function (r) { dl.appendChild(el('dt', null, r[0])); dl.appendChild(el('dd', null, r[1])); }); tip.appendChild(dl);
    tip.hidden = false;
    var x, y;
    if (ev && ev.clientX !== undefined && ev.type !== 'focus') { x = ev.clientX + 14; y = ev.clientY + 14; }
    else { var r = anchor.getBoundingClientRect(); x = r.right + 8; y = r.top; }
    var tw = tip.offsetWidth, th = tip.offsetHeight;
    if (x + tw > window.innerWidth - 8) x = Math.max(8, (ev && ev.clientX !== undefined ? ev.clientX : x) - tw - 14);
    if (y + th > window.innerHeight - 8) y = Math.max(8, window.innerHeight - th - 8);
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }
  function hideTip() { if (tip) tip.hidden = true; }

  /* ---------- grafico qualita'/tempo ---------- */
  function famOrder(algos) {
    var cnt = {}; algos.forEach(function (a) { cnt[a.family] = (cnt[a.family] || 0) + 1; });
    return Object.keys(cnt).sort(function (a, b) { return cnt[b] - cnt[a] || a.localeCompare(b); });
  }
  function marker(shape, cx, cy, r, fill, extra) {
    var m;
    if (shape === 'square') m = sv('rect', { x: cx - r, y: cy - r, width: 2 * r, height: 2 * r, rx: 2 });
    else if (shape === 'triangle') m = sv('path', { d: 'M' + cx + ' ' + (cy - r * 1.2) + ' L' + (cx + r * 1.15) + ' ' + (cy + r * 0.9) + ' L' + (cx - r * 1.15) + ' ' + (cy + r * 0.9) + ' Z' });
    else if (shape === 'diamond') m = sv('path', { d: 'M' + cx + ' ' + (cy - r * 1.25) + ' L' + (cx + r * 1.25) + ' ' + cy + ' L' + cx + ' ' + (cy + r * 1.25) + ' L' + (cx - r * 1.25) + ' ' + cy + ' Z' });
    else m = sv('circle', { cx: cx, cy: cy, r: r });
    m.setAttribute('fill', fill); m.setAttribute('stroke', 'var(--surface)'); m.setAttribute('stroke-width', '2');
    Object.keys(extra || {}).forEach(function (k) { m.setAttribute(k, extra[k]); });
    return m;
  }
  function drawScatter(all) {
    var host = $('scatter'); host.innerHTML = '';
    var fams = famOrder(all), famIdx = {};
    fams.forEach(function (f, i) { famIdx[f] = i; });
    function style(f) { var i = famIdx[f]; return i < 3 ? { color: SERIES[i], shape: SHAPES[i], label: f } : { color: OTHER, shape: SHAPES[3], label: 'Other families' }; }
    // legenda
    var leg = $('scatter-legend'); leg.innerHTML = '';
    var legItems = fams.slice(0, 3).map(function (f) { return { key: f, label: f, st: style(f) }; });
    if (fams.length > 3) legItems.push({ key: '__other', label: 'Other families (' + (fams.length - 3) + ')', st: style(fams[3]) });
    legItems.forEach(function (it) {
      var b = el('button'); b.type = 'button'; b.setAttribute('aria-pressed', String(!S.hidden[it.key]));
      var s = sv('svg', { width: 14, height: 14, viewBox: '0 0 14 14', 'aria-hidden': 'true' }); s.appendChild(marker(it.st.shape, 7, 7, 4.5, it.st.color, { stroke: 'none' })); b.appendChild(s);
      b.appendChild(document.createTextNode(it.label));
      b.addEventListener('click', function () { S.hidden[it.key] = !S.hidden[it.key]; drawScatter(all); });
      leg.appendChild(b);
    });
    function famKey(f) { return famIdx[f] < 3 ? f : '__other'; }
    var pts = all.filter(function (a) { return a.time !== null && a.time !== undefined && a.time > 0 && !S.hidden[famKey(a.family)]; });
    var noTime = all.filter(function (a) { return a.time === null || a.time === undefined || a.time <= 0; });
    var W = Math.max(320, host.clientWidth || 760), H = W < 520 ? 340 : 420;
    var m = { l: 52, r: 24, t: 14, b: 48 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
    var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Scatter plot: mean score against median time per algorithm. The data are in the table below.' });
    var tmin = 1, tmax = 10;
    if (pts.length) { tmin = Math.min.apply(null, pts.map(function (p) { return p.time; })); tmax = Math.max.apply(null, pts.map(function (p) { return p.time; })); }
    var e0 = Math.floor(Math.log10(tmin) - 0.05), e1 = Math.ceil(Math.log10(tmax) + 0.05); if (e1 <= e0) e1 = e0 + 1;
    var X = function (t) { return m.l + iw * (Math.log10(t) - e0) / (e1 - e0); }, Y = function (q) { return m.t + ih * (1 - q); };
    // griglia e assi
    for (var q = 0; q <= 1.0001; q += 0.25) {
      svg.appendChild(sv('line', { x1: m.l, x2: m.l + iw, y1: Y(q), y2: Y(q), stroke: q === 0 ? 'var(--axis)' : 'var(--grid)', 'stroke-width': 1 }));
      svg.appendChild(sv('text', { x: m.l - 8, y: Y(q) + 4, 'text-anchor': 'end' }, String(q)));
    }
    for (var e = e0; e <= e1; e++) {
      var tx = X(Math.pow(10, e));
      svg.appendChild(sv('line', { x1: tx, x2: tx, y1: m.t, y2: m.t + ih, stroke: 'var(--grid)', 'stroke-width': 1 }));
      svg.appendChild(sv('text', { x: tx, y: m.t + ih + 16, 'text-anchor': 'middle' }, e >= 3 ? NI.format(Math.pow(10, e)) + ' s' : (e < 0 ? String(Math.pow(10, e)) : Math.pow(10, e)) + ' s'));
    }
    svg.appendChild(sv('text', { class: 'axis-title', x: m.l + iw / 2, y: H - 8, 'text-anchor': 'middle' }, 'Median time per run (logarithmic scale)'));
    svg.appendChild(sv('text', { class: 'axis-title', transform: 'rotate(-90)', x: -(m.t + ih / 2), y: 14, 'text-anchor': 'middle' }, 'Mean score (0-1)'));
    // punti
    var frontier = pts.filter(function (p) { return !pts.some(function (o) { return o !== p && o.time <= p.time && o.score >= p.score && (o.time < p.time || o.score > p.score); }); });
    var placed = [];
    pts.forEach(function (p) {
      var st = style(p.family), cx = X(p.time), cy = Y(p.score);
      svg.appendChild(marker(st.shape, cx, cy, 5.5, st.color));
    });
    // etichette dirette: solo i punti sulla frontiera (migliore compromesso), senza sovrapposizioni
    var top4 = pts.slice().sort(function (a, b) { return b.score - a.score; }).slice(0, 4);
    frontier.concat(top4.filter(function (p) { return frontier.indexOf(p) < 0; })).sort(function (a, b) { return b.score - a.score; }).forEach(function (p) {
      var cx = X(p.time), cy = Y(p.score), wTxt = p.short.length * 6.4 + 4, right = cx + 10 + wTxt < W - 6;
      var x0 = right ? cx + 9 : cx - 9 - wTxt, box = { x0: x0, x1: x0 + wTxt, y0: cy - 8, y1: cy + 8 };
      var clash = placed.some(function (b) { return !(box.x1 < b.x0 || box.x0 > b.x1 || box.y1 < b.y0 || box.y0 > b.y1); });
      if (clash) return;
      placed.push(box);
      svg.appendChild(sv('text', { class: 'lab', x: right ? cx + 9 : cx - 9, y: cy + 4, 'text-anchor': right ? 'start' : 'end' }, p.short));
    });
    // aree di contatto (piu' grandi dei marcatori), anche da tastiera
    pts.forEach(function (p) {
      var h = sv('circle', { class: 'hit', cx: X(p.time), cy: Y(p.score), r: 14, tabindex: '0', role: 'img', 'aria-label': p.name + ': score ' + n3(p.score) + ', median time ' + fmtTime(p.time) });
      var rows = tipRows(p);
      h.addEventListener('mousemove', function (ev) { showTip(ev, p.name, rows, h); });
      h.addEventListener('mouseleave', hideTip); h.addEventListener('blur', hideTip);
      h.addEventListener('focus', function (ev) { showTip(ev, p.name, rows, h); });
      h.addEventListener('click', function (ev) { showTip(ev, p.name, rows, h); });
      svg.appendChild(h);
    });
    host.appendChild(svg);
    if (!pts.length) host.appendChild(el('p', 'ev-sub', all.length ? 'No points to show: hidden families or time not measured.' : ''));
    if (noTime.length) host.appendChild(el('p', 'ev-sub', 'Without measured time (not in the chart): ' + noTime.map(function (a) { return a.name; }).join(', ') + '.'));
    // tabella
    var t = el('table', 'ev-t'), hd = el('tr');
    ['Algorithm', 'Family', 'Score', 'Median time', 'Datasets', 'Runs', 'Failed'].forEach(function (x, i) { var th = el('th', null, x); th.scope = 'col'; hd.appendChild(th); });
    var th0 = el('thead'); th0.appendChild(hd); t.appendChild(th0);
    var tb = el('tbody');
    all.forEach(function (a) {
      var r = el('tr'); [a.name, a.family, n3(a.score), fmtTime(a.time), a.n_ds, a.n_runs, a.fail === null ? '–' : Math.round(100 * a.fail) + '%'].forEach(function (x, i) { var c = el(i === 0 ? 'th' : 'td', null, String(x)); if (i === 0) c.scope = 'row'; r.appendChild(c); });
      tb.appendChild(r);
    });
    t.appendChild(tb); var tt = $('scatter-table'); tt.innerHTML = ''; tt.appendChild(t);
  }
  function tipRows(p) {
    var rows = [['Family', p.family], ['Score', n3(p.score)], ['Median time', fmtTime(p.time)], ['Datasets', String(p.n_ds)], ['Runs', String(p.n_runs)],
      ['Failed', p.fail === null ? '–' : Math.round(100 * p.fail) + '%']];
    AXES.forEach(function (a) { rows.push([AXIS_LABEL[a], p.axes && p.axes[a] !== null && p.axes[a] !== undefined ? n3(p.axes[a]) : 'no measurement']); });
    return rows;
  }

  /* ---------- matrice dataset x algoritmo ---------- */
  function drawHeat(sum, algos, w) {
    var host = $('heat'); host.innerHTML = '';
    var dark = isDark(), dsList = sum.datasets.map(function (d) { return d.dataset; }).sort(function (a, b) { return a.toLowerCase().localeCompare(b.toLowerCase()); });
    var cell = {}; sum.matrix.forEach(function (m) { cell[m.dataset + '|' + m.algorithm] = m; });
    var cols = algos.map(function (a) { return a; });
    $('sub-heat').textContent = 'Rows: datasets (' + dsList.length + '). Columns: algorithms, from the highest mean score. Each cell is the algorithm\'s score on that dataset with the chosen profile, between 0 and 1; hatched cells are pairs with no measured runs.';
    var ramp = $('heat-ramp'); ramp.innerHTML = '';
    ramp.appendChild(el('span', null, '0'));
    var bar = el('span', 'bar'); bar.style.background = 'linear-gradient(90deg,' + [0, 0.25, 0.5, 0.75, 1].map(function (v) { return 'rgb(' + rampAt(v, dark).join(',') + ')'; }).join(',') + ')'; ramp.appendChild(bar);
    ramp.appendChild(el('span', null, '1')); var ms = el('span', 'miss hatch'); ramp.appendChild(ms); ramp.appendChild(el('span', null, 'no evidence'));
    var t = el('table', 'ev-heat'); t.setAttribute('aria-label', 'Score by dataset and algorithm');
    var hd = el('tr'); hd.appendChild(el('td')); hd.firstChild.style.cssText = 'width:auto;height:auto';
    cols.forEach(function (a) { var th = el('th'); th.scope = 'col'; th.title = a.name; th.appendChild(el('span', null, a.short)); hd.appendChild(th); });
    var thead = el('thead'); thead.appendChild(hd); t.appendChild(thead);
    var tb = el('tbody');
    dsList.forEach(function (d) {
      var tr = el('tr'), rh = el('th', null, d); rh.scope = 'row'; tr.appendChild(rh);
      cols.forEach(function (a) {
        var m = cell[d + '|' + a.key], td = el('td');
        if (!m) { td.className = 'hatch'; td.setAttribute('aria-label', d + ', ' + a.name + ': no evidence'); td.setAttribute('role', 'cell'); tr.appendChild(td); bindCell(td, d, a, null); return; }
        var sc = score(m.axis_scores, w), c = rampAt(sc, dark);
        td.style.background = 'rgb(' + c.join(',') + ')'; td.style.color = lum(c) > 0.3 ? '#101418' : '#ffffff';
        td.textContent = sc.toFixed(2);
        td.setAttribute('aria-label', d + ', ' + a.name + ': score ' + n3(sc)); td.tabIndex = 0;
        tr.appendChild(td); bindCell(td, d, a, m, sc);
      });
      tb.appendChild(tr);
    });
    t.appendChild(tb); host.appendChild(t);
  }
  function bindCell(td, d, a, m, sc) {
    var rows;
    if (!m) rows = [['Status', 'no measured run']];
    else {
      rows = [['Score', n3(sc)], ['Median time', fmtTime(m.time_s_median)], ['Runs', String(m.n_runs)], ['Seeds', String(m.n_seeds)], ['Failed', m.failure_rate === null ? '–' : Math.round(100 * m.failure_rate) + '%']];
      AXES.forEach(function (x) { rows.push([AXIS_LABEL[x], m.axis_scores && m.axis_scores[x] !== null && m.axis_scores[x] !== undefined ? n3(m.axis_scores[x]) : 'no measurement']); });
    }
    var title = d + ' · ' + a.name;
    td.addEventListener('mousemove', function (ev) { showTip(ev, title, rows, td); });
    td.addEventListener('mouseleave', hideTip);
    td.addEventListener('focus', function (ev) { showTip(ev, title, rows, td); });
    td.addEventListener('blur', hideTip);
    td.addEventListener('click', function (ev) { showTip(ev, title, rows, td); });
  }

  /* ---------- avvio ---------- */
  var resizeT = null;
  window.addEventListener('resize', function () { clearTimeout(resizeT); resizeT = setTimeout(function () { if (S.summary && S.summary.available) render(); }, 150); });
  document.addEventListener('scroll', hideTip, true);
  var q = new URLSearchParams(location.search);
  Promise.all(MODS.map(function (m) {
    return api('/api/kb/summary?modality=' + m[0]).then(function (j) { S.counts[m[0]] = j.available ? ((j.kb || {}).runs || 0) : 0; }).catch(function () { S.counts[m[0]] = null; });
  })).then(function () {
    var want = q.get('mod');
    S.mod = MODS.some(function (m) { return m[0] === want; }) ? want : (MODS.filter(function (m) { return S.counts[m[0]]; })[0] || MODS[0])[0];
    if (q.get('profile') && PROFILES.some(function (p) { return p[0] === q.get('profile'); })) S.profile = q.get('profile');
    drawControls(); load();
  });
})();
