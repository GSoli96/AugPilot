/* Pagina «Sfida»: legge /api/challenge/datasets, /api/challenge/candidates e fa POST /api/challenge/reveal.
 * Qui non si calcola nulla: classifica, consiglio leave-one-out e regret arrivano da assistant/kb/challenge.py.
 * Le barre dei contributi sono quelle di window.Evidence (site/shared.js). Il conteggio delle partite sta in localStorage
 * (try/catch: serve solo come comodità, la pagina funziona anche senza). */
(function () {
  'use strict';
  var $ = function (id) { return document.getElementById(id); };
  var Ev = window.Evidence || null;
  var MODS = [['tabular', 'Tabular'], ['ts', 'Time series'], ['graph', 'Graphs']];
  var LEVELS = [['graph', 'Whole graphs'], ['node', 'Nodes of a graph']];
  var AXES = ['utility', 'fidelity', 'diversity', 'privacy', 'speed'];
  var AXIS_LABEL = { utility: 'Utility', fidelity: 'Fidelity', diversity: 'Diversity', privacy: 'Privacy', speed: 'Speed' };
  var PROFILES = (Ev && Ev.PROFILES) || [['balanced', 'Balanced'], ['utility', 'Utility'], ['fidelity', 'Fidelity'], ['diversity', 'Diversity'], ['privacy', 'Privacy'], ['speed', 'Speed']];
  var STORE_KEY = 'augpilot.challenge.v1';
  var S = { data: null, mod: null, level: 'graph', ds: null, profile: 'balanced', cands: null, guess: null, catalog: {}, result: null, seq: 0, counted: null, busy: false };
  var NF = new Intl.NumberFormat('en-US', { maximumFractionDigits: 3, minimumFractionDigits: 3 });
  var NI = new Intl.NumberFormat('en-US');
  function n3(v) { return v === null || v === undefined || !isFinite(v) ? '–' : NF.format(v); }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined && text !== null) e.textContent = text; return e; }
  function api(url, opt) { return fetch(url, opt).then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(j.error || ('HTTP ' + r.status)); return j; }); }); }
  function label(list, key) { var f = list.filter(function (x) { return x[0] === key; })[0]; return f ? f[1] : String(key); }
  function pctTxt(v) { return Math.round(100 * v) + '%'; }

  /* ---------- conteggio delle partite (solo comodità) ---------- */
  function emptyBoard() { return { rounds: 0, visitor: 0, augpilot: 0, tie: 0, sumV: 0, sumA: 0 }; }
  function loadBoard() {
    try { var b = JSON.parse(localStorage.getItem(STORE_KEY) || 'null'); if (b && typeof b.rounds === 'number') return Object.assign(emptyBoard(), b); } catch (e) { /* ignora */ }
    return emptyBoard();
  }
  function saveBoard(b) { try { localStorage.setItem(STORE_KEY, JSON.stringify(b)); } catch (e) { /* ignora: il conteggio vale solo per questa pagina aperta */ } }
  var BOARD = loadBoard();
  function drawBoard() {
    var box = $('board'); box.innerHTML = '';
    function stat(v, l) { var d = el('div', 'ch-stat'); d.appendChild(el('div', 'v', v)); d.appendChild(el('div', 'l', l)); box.appendChild(d); }
    stat(NI.format(BOARD.rounds), 'rounds with advice');
    stat(NI.format(BOARD.visitor), 'won by you');
    stat(NI.format(BOARD.augpilot), 'won by AugPilot');
    stat(NI.format(BOARD.tie), 'ties');
    stat(BOARD.rounds ? n3(BOARD.sumV / BOARD.rounds) : '–', 'your mean regret');
    stat(BOARD.rounds ? n3(BOARD.sumA / BOARD.rounds) : '–', 'AugPilot mean regret');
  }

  /* ---------- controlli ---------- */
  function radio(box, items, current, onPick, disabled) {
    box.innerHTML = '';
    items.forEach(function (it) {
      var b = el('button'); b.type = 'button'; b.setAttribute('role', 'radio'); b.setAttribute('aria-checked', String(it[0] === current));
      b.appendChild(document.createTextNode(it[1]));
      if (it[2]) b.appendChild(el('span', 'cnt', it[2]));
      if (disabled && disabled[it[0]]) { b.disabled = true; b.title = 'No data in the KB for this choice'; }
      b.addEventListener('click', function () { onPick(it[0]); });
      box.appendChild(b);
    });
  }
  function modDatasets(mod, level) {
    return (S.data.datasets || []).filter(function (d) { return d.modality === mod && (mod !== 'graph' || d.level === level); });
  }
  function levelsOf(mod) {
    var s = {}; (S.data.datasets || []).forEach(function (d) { if (d.modality === mod) s[d.level] = true; });
    return LEVELS.filter(function (l) { return s[l[0]]; });
  }
  function drawControls() {
    var counts = {}; MODS.forEach(function (m) { counts[m[0]] = (S.data.datasets || []).filter(function (d) { return d.modality === m[0]; }).length; });
    radio($('seg-mod'), MODS.map(function (m) { return [m[0], m[1], '(' + counts[m[0]] + ')']; }), S.mod, setMod,
      (function () { var o = {}; MODS.forEach(function (m) { o[m[0]] = !counts[m[0]]; }); return o; })());
    var lv = S.mod === 'graph' ? levelsOf('graph') : [];
    $('fld-level').hidden = lv.length < 2;
    if (lv.length >= 2) radio($('seg-level'), lv, S.level, function (l) { S.level = l; pickDefaultDs(); drawControls(); onSelection(); });
    var sel = $('sel-ds'); sel.innerHTML = '';
    modDatasets(S.mod, S.level).forEach(function (d) {
      var o = el('option', null, d.dataset + ' — ' + d.n_algorithms + ' algorithms measured' + (d.playable ? '' : ' (challenge not possible)'));
      o.value = d.dataset; if (d.dataset === S.ds) o.selected = true; sel.appendChild(o);
    });
    radio($('seg-prof'), PROFILES, S.profile, function (p) { S.profile = p; drawControls(); resetResult(); });
    var w = (S.data.profiles || {})[S.profile] || {};
    $('prof-hint').textContent = 'Profile ' + label(PROFILES, S.profile) + ': ' + AXES.filter(function (a) { return w[a] !== undefined; })
      .map(function (a) { return AXIS_LABEL[a] + ' ' + pctTxt(w[a]); }).join(', ') + '. Axes with no measurements on a dataset are dropped and the weights renormalized.';
  }
  function curDs() { return modDatasets(S.mod, S.level).filter(function (d) { return d.dataset === S.ds; })[0] || null; }
  function pickDefaultDs() {
    var l = modDatasets(S.mod, S.level), pl = l.filter(function (d) { return d.advice_possible; })[0] || l.filter(function (d) { return d.playable; })[0] || l[0];
    S.ds = pl ? pl.dataset : null;
  }
  function setMod(m) {
    if (m === S.mod) return;
    S.mod = m;
    if (m === 'graph') { var lv = levelsOf('graph'); S.level = lv.length && !lv.some(function (l) { return l[0] === S.level; }) ? lv[0][0] : S.level; }
    pickDefaultDs(); drawControls(); onSelection();
  }

  /* ---------- scelta e candidati ---------- */
  function resetResult() { S.result = null; $('ch-result').hidden = true; updateGo(); }
  function onSelection() { S.guess = null; S.cands = null; resetResult(); loadCands(); }
  function algoName(k) { var c = S.catalog[S.mod] && S.catalog[S.mod][String(k).toLowerCase()]; return (c && c.name) || k; }
  function algoFamily(k) { var c = S.catalog[S.mod] && S.catalog[S.mod][String(k).toLowerCase()]; return (c && c.strategy) || ''; }
  function loadCatalog(mod) {
    if (S.catalog[mod]) return Promise.resolve();
    return api('/api/modality/' + mod + '/algorithms').then(function (j) {
      var m = {}; (j.algorithms || []).forEach(function (a) { m[String(a.key).toLowerCase()] = { name: a.name || a.key, strategy: a.strategy || '' }; });
      S.catalog[mod] = m;
    }).catch(function () { S.catalog[mod] = {}; });
  }
  function loadCands() {
    var seq = ++S.seq, d = curDs(), note = $('ds-note');
    note.hidden = true; $('chips').innerHTML = '';
    if (!d) { $('guess-sub').textContent = 'No dataset available for this choice.'; updateGo(); return; }
    if (d.note) { note.hidden = false; note.textContent = d.note.charAt(0).toUpperCase() + d.note.slice(1) + '.' + (d.playable ? ' You can play, but after "Reveal" you will only see the measured ranking, without AugPilot advice.' : ''); }
    if (!d.playable) { $('guess-sub').textContent = 'With fewer than ' + S.data.min_algorithms + ' measured algorithms the challenge makes no sense: choose another dataset.'; updateGo(); return; }
    $('guess-sub').textContent = 'Loading the measured algorithms...';
    var q = '/api/challenge/candidates?modality=' + encodeURIComponent(S.mod) + '&dataset=' + encodeURIComponent(d.dataset) + (S.mod === 'graph' ? '&level=' + S.level : '');
    Promise.all([api(q), loadCatalog(S.mod)]).then(function (r) {
      if (seq !== S.seq) return;
      S.cands = r[0]; drawChips();
    }).catch(function (e) {
      if (seq !== S.seq) return;
      $('guess-sub').textContent = 'Cannot read the algorithms: ' + (e.message || e);
    });
  }
  function drawChips() {
    var box = $('chips'); box.innerHTML = '';
    $('guess-sub').textContent = 'These are the ' + S.cands.n_algorithms + ' algorithms measured on "' + S.cands.dataset + '". There are no scores here: numbers appear only after "Reveal".';
    S.cands.algorithms.slice().sort(function (a, b) { return algoName(a.algorithm).toLowerCase() < algoName(b.algorithm).toLowerCase() ? -1 : 1; }).forEach(function (c) {
      var b = el('button', 'ch-chip'); b.type = 'button'; b.setAttribute('role', 'radio'); b.setAttribute('aria-checked', String(c.algorithm === S.guess));
      b.dataset.algo = c.algorithm;
      b.appendChild(el('span', null, algoName(c.algorithm)));
      var fam = algoFamily(c.algorithm); if (fam) b.appendChild(el('small', null, fam));
      b.addEventListener('click', function () { S.guess = c.algorithm; Array.prototype.forEach.call(box.children, function (x) { x.setAttribute('aria-checked', String(x.dataset.algo === S.guess)); }); resetResult(); });
      box.appendChild(b);
    });
    updateGo();
  }
  function updateGo() {
    var btn = $('btn-go'), ok = !!(S.guess && S.cands && !S.busy);
    btn.disabled = !ok;
    $('picked').textContent = S.guess ? 'Your choice: ' + algoName(S.guess) : (S.cands ? 'Choose an algorithm to enable "Reveal".' : '');
  }

  /* ---------- risultato ---------- */
  function selKey() { return [S.mod, S.mod === 'graph' ? S.level : '', S.ds, S.profile, S.guess].join('|'); }
  function reveal() {
    if (!S.guess || !S.cands) return;
    S.busy = true; updateGo(); $('ch-status').textContent = 'Computing the AugPilot advice...';
    var key = selKey();
    api('/api/challenge/reveal', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ modality: S.mod, level: S.mod === 'graph' ? S.level : '', dataset: S.ds, profile: S.profile, guess: S.guess }) })
      .then(function (r) { S.result = r; $('ch-status').textContent = ''; render(r, key); })
      .catch(function (e) { $('ch-status').textContent = 'Error: ' + (e.message || e); })
      .then(function () { S.busy = false; updateGo(); });
  }
  function tag(text, cls) { return el('span', 'ch-tag' + (cls ? ' ' + cls : ''), text); }
  function entryCard(kicker, entry, r, extra) {
    var c = el('section', 'ev-card'); c.setAttribute('aria-label', kicker);
    c.appendChild(el('div', 'ch-kick', kicker));
    c.appendChild(el('div', 'ch-algo', algoName(entry.algorithm)));
    var tags = el('div', 'ch-tags');
    tags.appendChild(tag('rank ' + entry.rank + ' of ' + r.n_algorithms, entry.rank === 1 ? 'good' : ''));
    tags.appendChild(tag('measured score ' + n3(entry.score)));
    tags.appendChild(tag('regret ' + n3(entry.regret), entry.regret === 0 ? 'good' : ''));
    c.appendChild(tags);
    if (extra) extra(c);
    if (Ev) c.appendChild(Ev.contribBars(entry, r.weights));
    return c;
  }
  function render(r, key) {
    $('ch-result').hidden = false;
    var verdict = $('verdict'), duo = $('duo'), rc = $('ranking-card');
    verdict.className = 'ch-verdict'; verdict.innerHTML = ''; duo.innerHTML = '';
    if (r.status === 'too_few_algorithms') {
      verdict.classList.add('v-none'); verdict.appendChild(el('b', null, 'Challenge not possible on this dataset')); verdict.appendChild(el('p', null, r.message));
      rc.hidden = true; return;
    }
    var o = r.outcome, g = r.guess, a = r.advice;
    if (o) {
      verdict.classList.add('v-' + (o.result === 'tie' ? 'tie' : o.result));
      verdict.appendChild(el('b', null, o.label));
      verdict.appendChild(el('p', null, 'Normalized regret (0 = optimal choice, 1 = the worst among the ' + r.n_algorithms + ' measured algorithms): you ' + n3(o.visitor_regret) + ', AugPilot ' + n3(o.augpilot_regret) +
        '; a random choice would have ' + n3(o.chance_regret) + ' on average.'));
      if (S.counted !== key) {
        S.counted = key; BOARD.rounds += 1; BOARD.sumV += o.visitor_regret || 0; BOARD.sumA += o.augpilot_regret || 0;
        if (o.result === 'visitor') BOARD.visitor += 1; else if (o.result === 'augpilot') BOARD.augpilot += 1; else BOARD.tie += 1;
        saveBoard(BOARD); drawBoard();
      }
    } else {
      verdict.classList.add('v-none');
      verdict.appendChild(el('b', null, g.rank === 1 ? 'You picked the best one, but there is no comparison with AugPilot' : 'The comparison with AugPilot is not available for this dataset'));
      verdict.appendChild(el('p', null, 'Your normalized regret is ' + n3(g.regret) + ' (0 = optimal choice, 1 = the worst). This round is not counted.'));
    }
    duo.appendChild(entryCard('Your choice', g, r, function (c) {
      if (r.guess_is_best) c.appendChild(el('p', 'ch-text', 'It is the best measured' + (r.n_ties_at_top > 1 ? ' (tied with ' + (r.n_ties_at_top - 1) + ' other' + (r.n_ties_at_top - 1 === 1 ? '' : 's') + ')' : '') + '.'));
      else c.appendChild(el('p', 'ch-text', 'The best measured is ' + algoName(r.best) + ' (score ' + n3(r.best_score) + ').'));
    }));
    if (a) {
      duo.appendChild(entryCard('AugPilot advice', a, r, function (c) {
        c.appendChild(el('p', 'ch-text', 'Computed as for a new dataset: "' + r.dataset + '" is removed from the knowledge base and the ' + a.neighbors.length + ' closest datasets by meta-feature are used. Estimate from the neighbors: score ' + n3(a.predicted_score) + ', first among ' + a.n_compared + ' comparable algorithms. The contributions below are those measured on "' + r.dataset + '".'));
        var ul = el('ul', 'ch-nb');
        a.neighbors.forEach(function (n) { ul.appendChild(el('li', null, n.dataset + ' (distance ' + n3(n.distance) + ', weight ' + pctTxt(n.weight) + ')')); });
        c.appendChild(ul);
        if (a.n_not_comparable) c.appendChild(el('p', 'ch-text', a.n_not_comparable + ' algorithms measured on "' + r.dataset + '" are not measured on the neighbors and AugPilot could not recommend them.'));
      }));
    } else {
      var c = el('section', 'ev-card'); c.setAttribute('aria-label', 'AugPilot advice');
      c.appendChild(el('div', 'ch-kick', 'AugPilot advice'));
      c.appendChild(el('div', 'ch-algo', 'No advice'));
      c.appendChild(el('div', 'ch-fallback', r.message || 'There is not enough data for advice.'));
      duo.appendChild(c);
    }
    drawRanking(r);
    var foot = 'Source: benchmark KB. Axis scores = 0-1 percentiles among the algorithms measured on the same dataset; score = sum of weight × axis score with the profile weights. Same logic as the leave-one-dataset-out evaluation (E1).';
    if (a && r.weights && a.weights && JSON.stringify(Object.keys(r.weights).sort()) !== JSON.stringify(Object.keys(a.weights).sort())) foot += ' Note: the axes with measurements in the neighboring datasets differ from those of the chosen dataset, so the advice weights are renormalized over a different set of axes.';
    $('ch-foot').textContent = foot;
    try { $('h-result').scrollIntoView({ behavior: 'smooth', block: 'start' }); } catch (e) { /* ignora */ }
  }
  function drawRanking(r) {
    var card = $('ranking-card'); card.hidden = false;
    var shown = r.ranking.length;
    $('rank-sub').textContent = 'Top ' + shown + ' rows of ' + r.n_algorithms + ' algorithms measured on "' + r.dataset + '", profile ' + label(PROFILES, r.profile) + '. Bar length is the score (0-1); colors are the contributions per axis.';
    var lg = $('rank-legend'); lg.innerHTML = ''; if (Ev) lg.appendChild(Ev.axisLegend(r.weights));
    var ol = $('rank-list'); ol.innerHTML = '';
    var list = r.ranking.slice();
    [r.guess, r.advice].forEach(function (x) { if (x && !list.some(function (y) { return y.algorithm === x.algorithm; })) list.push(x); });
    list.forEach(function (e) {
      var isG = e.algorithm === r.guess.algorithm, isA = r.advice && e.algorithm === r.advice.algorithm;
      var li = el('li', isG || isA ? 'me' : '');
      li.appendChild(el('span', 'pos', String(e.rank)));
      var nm = el('span', 'nm', algoName(e.algorithm));
      var marks = el('span', 'marks');
      if (isG) marks.appendChild(el('span', 'tag', 'your choice'));
      if (isA) marks.appendChild(el('span', 'tag', 'AugPilot'));
      if (marks.children.length) nm.appendChild(marks);
      li.appendChild(nm);
      var track = el('div', 'evx-track'); track.setAttribute('role', 'img');
      var ax = AXES.filter(function (k) { return e.contributions && e.contributions[k] !== null && e.contributions[k] !== undefined; });
      track.setAttribute('aria-label', 'Score ' + n3(e.score) + ': ' + ax.map(function (k) { return AXIS_LABEL[k] + ' ' + n3(e.contributions[k]); }).join(', '));
      ax.forEach(function (k) {
        var seg = el('span', 'evx-seg evx-' + k); seg.style.width = (100 * Math.max(0, e.contributions[k])) + '%';
        seg.title = AXIS_LABEL[k] + ': ' + n3(e.contributions[k]); track.appendChild(seg);
      });
      li.appendChild(track);
      li.appendChild(el('span', 'sc', n3(e.score)));
      ol.appendChild(li);
    });
    if (list.length > shown) ol.insertBefore(el('li', 'sep'), ol.children[shown]);
  }

  /* ---------- avvio ---------- */
  function drawEmpty(msg) {
    var box = $('ch-empty'); box.innerHTML = ''; box.hidden = false; $('ch-app').hidden = true;
    box.appendChild(el('h2', null, 'No data for the challenge'));
    box.appendChild(el('p', null, msg || 'The benchmark knowledge base is empty.'));
    var p = el('p'); p.appendChild(document.createTextNode('When the benchmark results are ready, rebuild the base with '));
    p.appendChild(el('code', null, 'python -m assistant.kb.build_kb')); p.appendChild(document.createTextNode('.'));
    box.appendChild(p);
  }
  function init() {
    $('ch-status').textContent = 'Loading the benchmark datasets...';
    api('/api/challenge/datasets').then(function (j) {
      $('ch-status').textContent = '';
      S.data = j;
      if (!j.available || !(j.datasets || []).length) { drawEmpty(j.message || 'The benchmark has not measured anything yet.'); return; }
      $('ch-app').hidden = false;
      var first = MODS.filter(function (m) { return modDatasets(m[0], m[0] === 'graph' ? 'graph' : '').length || (m[0] === 'graph' && levelsOf('graph').length); })[0];
      S.mod = first ? first[0] : 'tabular';
      if (S.mod === 'graph') { var lv = levelsOf('graph'); S.level = lv.length ? lv[0][0] : 'graph'; }
      pickDefaultDs(); drawControls(); drawBoard(); loadCands();
    }).catch(function (e) { $('ch-status').textContent = ''; drawEmpty('Cannot read the datasets: ' + (e.message || e)); });
  }
  $('sel-ds').addEventListener('change', function (ev) { S.ds = ev.target.value; onSelection(); });
  $('btn-go').addEventListener('click', reveal);
  $('btn-reset').addEventListener('click', function () { BOARD = emptyBoard(); saveBoard(BOARD); S.counted = null; drawBoard(); });
  init();
})();
