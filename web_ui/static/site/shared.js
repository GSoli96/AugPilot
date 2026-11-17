/* shared.js - funzioni comuni alle pagine: prefill tra pagine (query string), descrizioni brevi (/api/catalog_info/<modalita>),
 * etichette e formattazione delle statistiche dei dataset (window.Stats) e rendering del markdown (window.Md).
 * Espone window.Prefill, window.CatalogInfo, window.Stats, window.Md, window.Evidence (profili, barre dei contributi, pannello «Perché?»)
 * e window.Upload (caricamento di un dataset proprio). Nessuna dipendenza obbligatoria: Md usa marked e DOMPurify se presenti. */
(function () {
  'use strict';
  var SAFE = /^[A-Za-z0-9_.\-]+$/;

  /* ---------- Prefill ----------
   * Formato in query string: ?mod=ts&datasets=a,b&algos=x,y  (opzionale &from=catalog). */
  var Prefill = {
    read: function () {
      var p; try { p = new URLSearchParams(window.location.search); } catch (e) { return null; }
      var mod = p.get('mod');
      var list = function (k) { return (p.get(k) || '').split(',').map(function (x) { return x.trim(); }).filter(function (x) { return SAFE.test(x); }); };
      var datasets = list('datasets'), algos = list('algos');
      if (!mod || !SAFE.test(mod) || (!datasets.length && !algos.length)) return null;
      return { mod: mod, datasets: datasets, algos: algos, from: p.get('from') || '' };
    },
    url: function (page, sel) {
      var q = ['mod=' + encodeURIComponent(sel.mod)];
      if (sel.datasets && sel.datasets.length) q.push('datasets=' + sel.datasets.map(encodeURIComponent).join(','));
      if (sel.algos && sel.algos.length) q.push('algos=' + sel.algos.map(encodeURIComponent).join(','));
      q.push('from=catalog');
      return page + '?' + q.join('&');
    },
    go: function (page, sel) { window.location.href = Prefill.url(page, sel); },
    /* Toglie il prefill dalla barra degli indirizzi, cosi' un ricaricamento non lo riapplica. */
    clean: function () { try { window.history.replaceState(null, '', window.location.pathname); } catch (e) { /* ignora */ } }
  };

  /* ---------- Markdown (marked + DOMPurify, se caricati; altrimenti testo semplice) ---------- */
  var mdHooked = false;
  var Md = {
    available: function () {
      return typeof window.marked !== 'undefined' && typeof window.marked.parse === 'function' &&
        typeof window.DOMPurify !== 'undefined' && typeof window.DOMPurify.sanitize === 'function';
    },
    /* HTML ripulito, oppure null se le librerie mancano o falliscono. */
    html: function (text) {
      if (!Md.available()) return null;
      try {
        if (!mdHooked && window.DOMPurify.addHook) {
          window.DOMPurify.addHook('afterSanitizeAttributes', function (n) {
            if (n.tagName === 'A') { n.setAttribute('target', '_blank'); n.setAttribute('rel', 'noopener noreferrer'); }
          });
          mdHooked = true;
        }
        var raw = window.marked.parse(String(text == null ? '' : text), { gfm: true, breaks: true });
        return window.DOMPurify.sanitize(raw, { USE_PROFILES: { html: true } });
      } catch (e) { return null; }
    },
    /* Mette il testo nell'elemento: markdown reso se possibile, testo semplice altrimenti. Si puo' richiamare piu' volte (streaming). */
    render: function (el, text) {
      var s = String(text == null ? '' : text), h = Md.html(s);
      if (h === null) { el.classList.remove('md'); el.textContent = s; }
      else { el.classList.add('md'); el.innerHTML = h; }
      return el;
    }
  };

  /* ---------- Statistiche dei dataset: etichette leggibili e formattazione ---------- */
  var STAT_LABELS = {
    // tabulari (catalog_info/datasets_tabular.json)
    n_rows: 'Rows', n_cols: 'Columns', class_column: 'Class column', class_column_is_named_class: 'Column named "class"',
    n_numeric_features: 'Numeric features', n_categorical_features: 'Categorical features',
    n_numeric_low_cardinality_le10: 'Numeric with at most 10 values', n_missing_values: 'Missing values',
    n_classes: 'Classes', class_distribution: 'Class distribution', imbalance_ratio: 'Imbalance ratio',
    // serie temporali
    n_series: 'Series', length_T: 'Length of each series (T)', dims_D: 'Channels (D)', has_labels: 'Has labels',
    value_min: 'Minimum value', value_max: 'Maximum value',
    // grafi
    level: 'Level', n_nodes: 'Nodes', n_edges: 'Edges', n_features: 'Features per node', avg_degree: 'Average degree',
    n_train: 'Training nodes', n_val: 'Validation nodes', n_test: 'Test nodes',
    class_distribution_train: 'Class distribution (training)', imbalance_ratio_train: 'Imbalance ratio (training)',
    n_graphs: 'Graphs', avg_nodes: 'Average nodes per graph', avg_edges: 'Average edges per graph', max_nodes: 'Maximum nodes in a graph',
    // risultato di analyze_dataset
    rows: 'Rows', columns: 'Columns', column_names: 'Column names', missing_values_total: 'Missing values',
    columns_with_missing: 'Columns with missing values', numeric_columns: 'Numeric columns', categorical_columns: 'Categorical columns',
    duplicate_rows: 'Duplicate rows', label_column: 'Class column', label_column_basis: 'How it was detected',
    class_counts: 'Class distribution', minority_class: 'Minority class', sampled: 'Sample', note: 'Note', warning: 'Warning',
    size_kb: 'Size (KB)', file: 'File', modality: 'Modality', series: 'Series', length: 'Length', channels: 'Channels',
    kind: 'Type', nodes: 'Nodes', edges: 'Edges', features: 'Features', graphs: 'Graphs', mean_nodes: 'Average nodes per graph',
    mean_edges: 'Average edges per graph', arrays: 'Arrays in the file',
    // piani e stato dei task
    stato: 'Status', durata_s: 'Duration (s)', errore: 'Error', algoritmi: 'Algorithms', dataset: 'Dataset', algorithms: 'Algorithms',
    datasets: 'Datasets', epochs: 'Epochs', n_samples: 'Samples to generate', device: 'Device'
  };
  var STAT_HINTS = {
    imbalance_ratio: 'Most frequent class divided by the least frequent: 1 = balanced.',
    imbalance_ratio_train: 'Like the imbalance ratio, computed on the training set only.'
  };
  var STAT_VALUES = { level: { node: 'nodes', graph: 'graphs' } };
  var NUM_FMT = null;
  function numFmt() { if (!NUM_FMT) { try { NUM_FMT = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }); } catch (e) { NUM_FMT = { format: String }; } } return NUM_FMT; }
  function isDist(v) {
    if (!v || typeof v !== 'object' || Array.isArray(v)) return false;
    var k = Object.keys(v); if (!k.length) return false;
    return k.every(function (x) { return typeof v[x] === 'number' && isFinite(v[x]) && v[x] >= 0; });
  }
  var Stats = {
    /* Etichetta in italiano; per le chiavi sconosciute ripulisce il nome (n_foo_bar -> "Foo bar"). */
    label: function (k) {
      if (Object.prototype.hasOwnProperty.call(STAT_LABELS, k)) return STAT_LABELS[k];
      var t = String(k).replace(/^n_/, '').replace(/_+/g, ' ').replace(/([a-z])([A-Z])/g, '$1 $2').trim();
      return t ? t.charAt(0).toUpperCase() + t.slice(1) : String(k);
    },
    hint: function (k) { return STAT_HINTS[k] || ''; },
    /* Testo di un valore semplice (non distribuzione). */
    format: function (k, v) {
      if (v === null || v === undefined || v === '') return '–';
      if (typeof v === 'boolean') return v ? 'yes' : 'no';
      if (typeof v === 'number') return numFmt().format(v);
      if (Array.isArray(v)) return v.map(function (x) { return typeof x === 'number' ? numFmt().format(x) : String(x); }).join(', ');
      if (typeof v === 'object') return Object.keys(v).map(function (x) { return x + ': ' + (typeof v[x] === 'number' ? numFmt().format(v[x]) : String(v[x])); }).join(' | ');
      var m = STAT_VALUES[k]; if (m && m[String(v)]) return m[String(v)];
      return String(v);
    },
    isDistribution: isDist,
    /* Barre orizzontali con etichetta della classe, conteggio e percentuale (classi ordinate per numerosita'). */
    distBars: function (dist) {
      var keys = Object.keys(dist).sort(function (a, b) { return dist[b] - dist[a]; });
      var total = keys.reduce(function (s, x) { return s + dist[x]; }, 0), max = keys.length ? dist[keys[0]] : 0;
      var box = document.createElement('div'); box.className = 'dist'; box.setAttribute('role', 'list');
      var shown = keys.slice(0, 12);
      shown.forEach(function (c) {
        var pct = total ? 100 * dist[c] / total : 0;
        var row = document.createElement('div'); row.className = 'dist-row'; row.setAttribute('role', 'listitem');
        row.title = 'Class ' + c + ': ' + numFmt().format(dist[c]) + ' (' + numFmt().format(pct) + '%)';
        var lab = document.createElement('span'); lab.className = 'dist-lab'; lab.textContent = c;
        var bar = document.createElement('span'); bar.className = 'dist-bar';
        var fill = document.createElement('i'); fill.style.width = Math.max(2, max ? Math.round(100 * dist[c] / max) : 0) + '%'; bar.appendChild(fill);
        var val = document.createElement('span'); val.className = 'dist-val'; val.textContent = numFmt().format(dist[c]) + ' · ' + numFmt().format(pct) + '%';
        row.appendChild(lab); row.appendChild(bar); row.appendChild(val); box.appendChild(row);
      });
      if (keys.length > shown.length) {
        var more = document.createElement('p'); more.className = 'dist-more'; more.textContent = '... and ' + (keys.length - shown.length) + ' more classes';
        box.appendChild(more);
      }
      return box;
    },
    /* Elenco di definizioni <dl> con etichette leggibili. opts.skip = chiavi da omettere, opts.cls = classe del <dl>,
     * opts.maxArray = oltre questa lunghezza gli array non si mostrano. Le distribuzioni diventano barre. */
    list: function (obj, opts) {
      opts = opts || {};
      var skip = opts.skip || [], dl = document.createElement('dl'); dl.className = opts.cls || 'ci-stats';
      var order = Object.keys(STAT_LABELS);
      var pos = function (k) { var i = order.indexOf(k); return i < 0 ? 1e6 : i; };
      Object.keys(obj || {}).sort(function (a, b) { return pos(a) - pos(b); }).forEach(function (k) {   // sort stabile: prima le chiavi note, nell'ordine delle etichette
        if (skip.indexOf(k) >= 0) return;
        var v = obj[k];
        if (v === null || v === undefined || v === '') return;
        if (Array.isArray(v) && v.length > (opts.maxArray || 12)) return;
        var dt = document.createElement('dt'); dt.textContent = Stats.label(k);
        var hint = Stats.hint(k); if (hint) dt.title = hint;
        var dd = document.createElement('dd');
        if (isDist(v)) { dd.className = 'dist-dd'; dd.appendChild(Stats.distBars(v)); dt.className = 'dist-dt'; }
        else if (typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length > 12) return;
        else dd.textContent = Stats.format(k, v);
        dl.appendChild(dt); dl.appendChild(dd);
      });
      return dl;
    }
  };

  /* ---------- Descrizioni brevi ---------- */
  var cache = {};
  var EMPTY = { algorithms: {}, datasets: {} };
  var CatalogInfo = {
    NA: 'not documented',
    load: function (mod) {
      if (!cache[mod]) {
        cache[mod] = fetch('/api/catalog_info/' + encodeURIComponent(mod))
          .then(function (r) { return r.ok ? r.json() : EMPTY; })
          .then(function (j) { return { algorithms: (j && j.algorithms) || {}, datasets: (j && j.datasets) || {} }; })
          .catch(function () { return EMPTY; });
      }
      return cache[mod];
    },
    /* Righe [etichetta, testo|null] di un algoritmo. */
    algoRows: function (e) {
      e = e || {};
      return [['What it does', e.what], ['When to use it', e.good_for], ['When to avoid it', e.avoid_when]].map(function (r) {
        return [r[0], (typeof r[1] === 'string' && r[1].trim()) ? r[1].trim() : null];
      });
    },
    unverified: function (e) { return !!e && e.verified === false; },
    /* Elemento DOM con le tre righe di un algoritmo. */
    algoBlock: function (e) {
      var box = document.createElement('div'); box.className = 'ci';
      CatalogInfo.algoRows(e).forEach(function (r) {
        var p = document.createElement('p'); p.className = 'ci-row';
        var b = document.createElement('b'); b.textContent = r[0] + ': '; p.appendChild(b);
        var s = document.createElement('span'); s.textContent = r[1] || CatalogInfo.NA; if (!r[1]) s.className = 'ci-na'; p.appendChild(s);
        box.appendChild(p);
      });
      if (e && CatalogInfo.unverified(e)) {
        var u = document.createElement('p'); u.className = 'ci-unv'; u.textContent = 'to be verified'; u.title = 'Description not yet verified against the source'; box.appendChild(u);
      }
      if (e && typeof e.source === 'string' && e.source.trim()) {
        var f = document.createElement('p'); f.className = 'ci-src'; f.textContent = 'Source: ' + e.source.trim(); box.appendChild(f);
      }
      return box;
    },
    /* Una riga di testo breve per tooltip. */
    algoTitle: function (e) { return e && typeof e.what === 'string' && e.what.trim() ? e.what.trim() : ''; },
    /* Elemento DOM con numeri e 'usi' di un dataset. */
    datasetBlock: function (e) {
      var box = document.createElement('div'); box.className = 'ci';
      var st = e && e.stats && typeof e.stats === 'object' ? e.stats : null;
      var keys = st ? Object.keys(st) : [];
      if (keys.length) {
        box.appendChild(Stats.list(st, { cls: 'ci-stats' }));
      } else { var n = document.createElement('p'); n.className = 'ci-row ci-na'; n.textContent = 'Statistics: ' + CatalogInfo.NA; box.appendChild(n); }
      var u = document.createElement('p'); u.className = 'ci-row';
      var b = document.createElement('b'); b.textContent = 'Uses: '; u.appendChild(b);
      var s = document.createElement('span'); var uses = e && e.uses;
      if (Array.isArray(uses)) uses = uses.join('; ');
      s.textContent = (typeof uses === 'string' && uses.trim()) ? uses.trim() : CatalogInfo.NA; if (s.textContent === CatalogInfo.NA) s.className = 'ci-na';
      u.appendChild(s); box.appendChild(u);
      if (e && typeof e.source === 'string' && e.source.trim()) { var f = document.createElement('p'); f.className = 'ci-src'; f.textContent = 'Source: ' + e.source.trim(); box.appendChild(f); }
      return box;
    }
  };


  /* ---------- Prove del consigliere: profili, barre dei contributi, pannello «Perché?» ----------
   * Usati da Percorso guidato e Copilota (stessa resa). I dati sono quelli di recommend_algorithms / explain_recommendation
   * (assistant/recommender.py, assistant/tools.py): qui non si calcola nulla, si formatta soltanto. Stili: site.css (.evx). */
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined && text !== null) e.textContent = text; return e; }
  function n3(v) { return (v === null || v === undefined || !isFinite(v)) ? '–' : Number(v).toFixed(3); }
  function pct(v) { return (v === null || v === undefined || !isFinite(v)) ? '–' : Math.round(100 * v) + '%'; }
  var AXES = ['utility', 'fidelity', 'diversity', 'privacy', 'speed'];
  var AXIS_LABEL = { utility: 'Utility', fidelity: 'Fidelity', diversity: 'Diversity', privacy: 'Privacy', speed: 'Speed' };
  var AXIS_HINT = {
    utility: 'How much the generated data helps a classifier (TRTR vs TR+S).',
    fidelity: 'How closely the generated data resembles the real data.',
    diversity: 'How well the generated data covers the variety of the real data.',
    privacy: 'How far the generated data stays from individual real records.',
    speed: 'Percentile of the run time: faster means higher.'
  };
  var PROFILES = [['balanced', 'Balanced'], ['utility', 'Utility'], ['fidelity', 'Fidelity'], ['diversity', 'Diversity'], ['privacy', 'Privacy'], ['speed', 'Speed']];
  var DIR_TXT = { lower: 'lower is better', higher: 'higher is better', target1: 'ideal is 1', neutral: 'not rankable' };
  var Evidence = {
    AXES: AXES, AXIS_LABEL: AXIS_LABEL, PROFILES: PROFILES,
    profileLabel: function (p) { var f = PROFILES.filter(function (x) { return x[0] === p; })[0]; return f ? f[1] : String(p || ''); },
    /* Quando il consiglio non viene dal KB (evidence === 'nessuna') o la spiegazione non esiste. */
    fallbackNote: function (method) {
      var d = el('div', 'evx-fallback'); d.setAttribute('role', 'note');
      d.appendChild(el('b', null, 'No benchmark evidence. '));
      d.appendChild(document.createTextNode('This recommendation is a fallback: it ranks the algorithms using the older available results or the runs alone, ' +
        'but there is no score measured by the benchmark for this modality, so there is nothing to break down or explain.'));
      if (method) { var m = el('details', 'evx-more'); m.appendChild(el('summary', null, 'How it was ranked')); m.appendChild(el('p', null, method)); d.appendChild(m); }
      return d;
    },
    /* Barra impilata dei contributi per asse: lunghezza totale = score (somma dei contributi), scala 0..1.
     * La legenda è anche la tabella dei numeri: peso, punteggio d'asse, contributo. */
    contribBars: function (item, weights) {
      var contrib = (item && item.contributions) || {}, ax = (item && item.axis_scores) || {}, w = weights || {};
      var axes = AXES.filter(function (a) { return contrib[a] !== undefined && contrib[a] !== null; });
      var score = item && typeof item.score === 'number' ? item.score : axes.reduce(function (t, a) { return t + (contrib[a] || 0); }, 0);
      var box = el('div', 'evx-contrib');
      if (!axes.length) { box.appendChild(el('p', 'evx-empty', 'No per-axis contribution in this result.')); return box; }
      var track = el('div', 'evx-track'); track.setAttribute('role', 'img');
      track.setAttribute('aria-label', 'Score ' + n3(score) + ' out of 1: ' + axes.map(function (a) { return AXIS_LABEL[a] + ' ' + n3(contrib[a]); }).join(', '));
      axes.forEach(function (a) {
        var seg = el('span', 'evx-seg evx-' + a); seg.style.width = (100 * Math.max(0, contrib[a])) + '%';
        seg.title = AXIS_LABEL[a] + ': weight ' + pct(w[a]) + ' × score ' + n3(ax[a]) + ' = ' + n3(contrib[a]);
        if (contrib[a] >= 0.1) seg.appendChild(el('span', 'evx-seg-t', n3(contrib[a])));
        track.appendChild(seg);
      });
      box.appendChild(track);
      var scale = el('div', 'evx-scale'); scale.setAttribute('aria-hidden', 'true');
      ['0', '0.5', '1'].forEach(function (t) { scale.appendChild(el('span', null, t)); });
      box.appendChild(scale);
      var tbl = el('table', 'evx-legend');
      tbl.appendChild(el('caption', 'evx-sr', 'Contributions by axis: profile weight, axis score (0-1 percentile among the measured algorithms) and contribution'));
      var head = el('tr'); ['Axis', 'Weight', 'Score', 'Contribution'].forEach(function (t) { var th = el('th', null, t); th.scope = 'col'; head.appendChild(th); });
      var thead = el('thead'); thead.appendChild(head); tbl.appendChild(thead);
      var tb = el('tbody');
      axes.forEach(function (a) {
        var tr = el('tr'), th = el('th'); th.scope = 'row';
        var sw = el('i', 'evx-sw evx-' + a); sw.setAttribute('aria-hidden', 'true'); th.appendChild(sw);
        var nm = el('span', null, AXIS_LABEL[a]); nm.title = AXIS_HINT[a]; th.appendChild(nm); tr.appendChild(th);
        [pct(w[a]), n3(ax[a]), n3(contrib[a])].forEach(function (t) { tr.appendChild(el('td', null, t)); });
        tb.appendChild(tr);
      });
      var tot = el('tr', 'evx-total'); var tt = el('th', null, 'Score (sum)'); tt.scope = 'row'; tot.appendChild(tt); tot.appendChild(el('td')); tot.appendChild(el('td'));
      tot.appendChild(el('td', null, n3(score))); tb.appendChild(tot);
      tbl.appendChild(tb); box.appendChild(tbl);
      var miss = AXES.filter(function (a) { return axes.indexOf(a) < 0; });
      if (miss.length) box.appendChild(el('p', 'evx-note', 'Axes without measurements (removed from the score, weights renormalized): ' + miss.map(function (a) { return AXIS_LABEL[a]; }).join(', ') + '.'));
      return box;
    },
    /* Legenda dei colori degli assi (per le barre piccole di un elenco, dove non c'e' la tabella). */
    axisLegend: function (weights) {
      var w = weights || {}, ul = el('ul', 'evx-axleg'); ul.setAttribute('aria-label', 'Axis colors');
      AXES.filter(function (a) { return w[a] !== undefined; }).forEach(function (a) {
        var li = el('li'); var sw = el('i', 'evx-sw evx-' + a); sw.setAttribute('aria-hidden', 'true'); li.appendChild(sw);
        li.appendChild(document.createTextNode(AXIS_LABEL[a] + ' (weight ' + pct(w[a]) + ')')); li.title = AXIS_HINT[a]; ul.appendChild(li);
      });
      return ul;
    },
    /* Elenco di avvisi (null se non ce ne sono). */
    warnings: function (list) {
      list = (list || []).filter(Boolean); if (!list.length) return null;
      var ul = el('ul', 'evx-warn'); list.forEach(function (t) { ul.appendChild(el('li', null, String(t))); }); return ul;
    },
    /* Pannello «Perché?». `expl` = risultato di explain_recommendation (o {error, method} senza prove);
     * `fallback` = voce del ranking di recommend_algorithms, usata per mostrare almeno i contributi quando la spiegazione completa non c'e';
     * `ctx` = {evidence, method, weights} dal risultato di recommend_algorithms. */
    explainPanel: function (expl, fallback, ctx) {
      var root = el('section', 'evx-panel'); root.setAttribute('aria-label', 'Why this recommendation');
      ctx = ctx || {};
      if (!expl || expl.error) {
        var noEv = ctx.evidence === 'nessuna' || (expl && /nessuna prova|no evidence|no benchmark evidence/i.test(expl.error || ''));
        if (noEv) { root.appendChild(Evidence.fallbackNote((expl && expl.method) || ctx.method)); return root; }
        if (fallback && fallback.contributions) {
          root.appendChild(el('p', 'evx-note', (expl && expl.error ? expl.error + ' ' : '') + 'Showing only what the recommendation already contains: the per-axis contributions.'));
          root.appendChild(Evidence.contribBars(fallback, ctx.weights));
          var wf = Evidence.warnings(fallback.warnings); if (wf) root.appendChild(wf);
        } else root.appendChild(el('div', 'evx-fallback', (expl && expl.error) || 'No explanation available.'));
        return root;
      }
      var head = el('div', 'evx-head');
      head.appendChild(el('b', null, expl.algorithm));
      head.appendChild(el('span', 'evx-tag', '#' + expl.rank + ' of ' + expl.n_ranked));
      head.appendChild(el('span', 'evx-tag', 'profile ' + Evidence.profileLabel(expl.profile)));
      head.appendChild(el('span', 'evx-tag', 'score ' + n3(expl.score)));
      root.appendChild(head);
      root.appendChild(el('p', 'evx-note', 'Scope: ' + (expl.scope || '–') + '.'));
      root.appendChild(Evidence.contribBars(expl, expl.weights));
      function metricList(title, arr) {
        var d = el('div', 'evx-col'); d.appendChild(el('h4', null, title));
        if (!arr || !arr.length) { d.appendChild(el('p', 'evx-empty', 'no metrics')); return d; }
        var ul = el('ul', 'evx-metrics');
        arr.forEach(function (m) {
          var li = el('li'); li.appendChild(el('code', null, m.metric));
          li.appendChild(el('span', 'evx-ax', AXIS_LABEL[m.axis] || m.axis || ''));
          li.appendChild(el('span', 'evx-v', 'score ' + n3(m.score) + (m.value !== undefined && m.value !== null ? ' · value ' + n3(m.value) : '')));
          li.title = DIR_TXT[m.direction] || '';
          ul.appendChild(li);
        });
        d.appendChild(ul); return d;
      }
      var two = el('div', 'evx-two');
      two.appendChild(metricList('Strongest metrics', expl.strongest_metrics));
      two.appendChild(metricList('Weakest metrics', expl.weakest_metrics));
      root.appendChild(two);
      var cmp = expl.comparison || [];
      var cs = el('div', 'evx-sec'); cs.appendChild(el('h4', null, 'Comparison with the next ranked'));
      if (!cmp.length) cs.appendChild(el('p', 'evx-empty', 'No other ranked algorithm to compare with.'));
      cmp.forEach(function (c) {
        var row = el('div', 'evx-cmp');
        var diff = c.score_diff;
        row.appendChild(el('b', null, '#' + c.rank + ' ' + c.algorithm));
        row.appendChild(el('span', null, ' score ' + n3(c.score) + ' (' + (diff >= 0 ? expl.algorithm + ' is ahead by ' + n3(diff) : 'is ahead by ' + n3(-diff)) + ')'));
        var d = c.contribution_diff_by_axis || {};
        var parts = Object.keys(d).map(function (a) { var v = d[a]; return (AXIS_LABEL[a] || a) + ' ' + (v >= 0 ? '+' : '−') + n3(Math.abs(v)); });
        if (parts.length) row.appendChild(el('div', 'evx-note', 'Contribution difference (this − the other): ' + parts.join(', ') + '.'));
        cs.appendChild(row);
      });
      root.appendChild(cs);
      var ds = el('div', 'evx-sec'); ds.appendChild(el('h4', null, 'Datasets used as evidence'));
      var sim = expl.similar_datasets || [];
      if (sim.length) {
        var t = el('table', 'evx-sim'); var th = el('tr'); ['Similar dataset', 'Distance', 'Weight'].forEach(function (x) { var c = el('th', null, x); c.scope = 'col'; th.appendChild(c); });
        var thead = el('thead'); thead.appendChild(th); t.appendChild(thead);
        var tb = el('tbody'); sim.forEach(function (x) { var r = el('tr'); r.appendChild(el('td', null, x.dataset)); r.appendChild(el('td', null, n3(x.distance))); r.appendChild(el('td', null, pct(x.weight))); tb.appendChild(r); });
        t.appendChild(tb); ds.appendChild(t);
        ds.appendChild(el('p', 'evx-note', 'The dataset is not in the benchmark: the score is the weighted average (1/distance) of the closest datasets by standardized meta-features.'));
      } else ds.appendChild(el('p', null, 'The dataset is in the benchmark: the score uses its runs (' + 'runs: ' + (expl.n_runs || '?') + ', seeds: ' + (expl.n_seeds || '?') + ').'));
      var fr = expl.failure_rate;
      if (fr !== undefined && fr !== null) ds.appendChild(el('p', 'evx-note', 'Failure rate: ' + pct(fr) + (fr > 0 ? ' (failed runs count as 0)' : '') + '. Source: ' + (expl.source || 'benchmark KB') + '.'));
      root.appendChild(ds);
      var w = Evidence.warnings(expl.warnings);
      if (w) { var ws = el('div', 'evx-sec'); ws.appendChild(el('h4', null, 'Warnings')); ws.appendChild(w); root.appendChild(ws); }
      if (expl.note) root.appendChild(el('p', 'evx-note', expl.note));
      return root;
    }
  };

  /* ---------- Caricamento di un dataset proprio (POST /api/datasets/upload) ----------
   * Upload.box(modalita, onDone) -> elemento. Tabulare: CSV con la classe nell'ultima colonna; serie temporali: .npz con `data` (N,T,D)
   * e `labels` opzionale; grafi: nessun caricamento (lo dice l'interfaccia). */
  var Upload = {
    MAX_MB: 50,
    accept: function (mod) { return mod === 'tabular' ? '.csv' : mod === 'ts' ? '.npz' : ''; },
    help: function (mod) {
      if (mod === 'tabular') return 'CSV with a header row, comma separator and the class in the last column. Maximum 50 MB. The file is used as is.';
      if (mod === 'ts') return '.npz file with the "data" array of shape (N series, T time steps, D channels) and, optionally, "labels" of shape (N). Maximum 50 MB.';
      return 'For graphs there is no upload: use the datasets already on disk.';
    },
    send: function (mod, file) {
      var fd = new FormData(); fd.append('modality', mod); fd.append('file', file);
      return fetch('/api/datasets/upload', { method: 'POST', body: fd }).then(function (r) {
        return r.json().catch(function () { return {}; }).then(function (j) {
          if (!r.ok || !j.ok) throw new Error(j.error || ('HTTP ' + r.status));
          return j.dataset;
        });
      });
    },
    box: function (mod, onDone) {
      var box = el('div', 'evx-upload');
      box.appendChild(el('h4', null, 'Upload your own dataset'));
      box.appendChild(el('p', 'evx-note', Upload.help(mod)));
      if (mod !== 'tabular' && mod !== 'ts') return box;
      var id = 'up-' + Math.random().toString(36).slice(2, 8);
      var lab = el('label', 'evx-sr', 'Dataset file (' + Upload.accept(mod) + ')'); lab.htmlFor = id;
      var inp = document.createElement('input'); inp.type = 'file'; inp.id = id; inp.accept = Upload.accept(mod);
      var btn = el('button', 'evx-btn', 'Upload'); btn.type = 'button';
      var msg = el('div', 'evx-upmsg'); msg.setAttribute('role', 'status');
      if (Upload.last && Upload.last.mod === mod) { msg.className = 'evx-upmsg ok'; msg.textContent = Upload.last.text; }  // la scheda si ridisegna dopo il caricamento: il messaggio resta
      btn.addEventListener('click', function () {
        var f = inp.files && inp.files[0];
        if (!f) { msg.className = 'evx-upmsg bad'; msg.textContent = 'Choose a ' + Upload.accept(mod) + ' file first.'; return; }
        if (f.size > Upload.MAX_MB * 1024 * 1024) { msg.className = 'evx-upmsg bad'; msg.textContent = 'File too large: maximum ' + Upload.MAX_MB + ' MB.'; return; }
        btn.disabled = true; msg.className = 'evx-upmsg'; msg.textContent = 'Uploading and checking the file...';
        Upload.send(mod, f).then(function (d) {
          var v = d.validation || {};
          msg.className = 'evx-upmsg ok';
          msg.textContent = 'Uploaded as "' + d.name + '"' + (d.renamed ? ' (name already in use: a suffix was added)' : '') + '. ' +
            (mod === 'tabular' ? v.rows + ' rows, ' + v.columns + ' columns, ' + v.n_classes + ' classes.' : v.series + ' series, T=' + v.length + ', D=' + v.channels + '.') +
            ((v.warnings || []).length ? ' Warning: ' + v.warnings.join(' ') : '');
          inp.value = ''; Upload.last = { mod: mod, text: msg.textContent };
          if (onDone) onDone(d);
        }).catch(function (e) { msg.className = 'evx-upmsg bad'; msg.textContent = 'Not uploaded: ' + e.message; })
          .then(function () { btn.disabled = false; });
      });
      var row = el('div', 'evx-uprow'); row.appendChild(lab); row.appendChild(inp); row.appendChild(btn);
      box.appendChild(row); box.appendChild(msg);
      return box;
    }
  };

  window.Evidence = Evidence;
  window.Upload = Upload;
  window.Prefill = Prefill;
  window.CatalogInfo = CatalogInfo;
  window.Stats = Stats;
  window.Md = Md;
})();
