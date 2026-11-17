/* Proposta 4 - Cabina di regia. Vanilla JS, nessuna libreria. Tutti i dati arrivano dalle API reali. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const MODS = {tabular: 'Tabular', ts: 'Time series', graph: 'Graphs'};
  const STATE_LABEL = {online: 'online', offline: 'offline', non_verificata: 'not checked', errore: 'error', non_raggiungibile: 'unreachable', inattiva: 'idle', libera: 'free', occupata: 'busy', sconosciuta: 'unknown', sconosciuto: 'unknown'};
  const stateLabel = st => STATE_LABEL[st] || String(st).replace(/_/g, ' ');
  const S = {mods: [], mod: 'tabular', algs: [], dss: [], rows: [], rowsLoaded: false, machines: null, sshClient: null,
             sys: null, ai: null, session: null, tasks: [], busy: false, filterMod: '', pollTimer: null};

  function h(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  async function jget(url) {
    const r = await fetch(url);
    let j = null; try { j = await r.json(); } catch (e) { /* non JSON */ }
    if (!r.ok) throw new Error((j && (j.error || j.message)) || ('HTTP ' + r.status));
    return j;
  }
  async function jpost(url, body) {
    const r = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})});
    let j = null; try { j = await r.json(); } catch (e) { /* non JSON */ }
    return {ok: r.ok, status: r.status, data: j};
  }
  const fmtT = s => s == null || isNaN(s) ? '-' : s < 1 ? (s * 1000).toFixed(0) + ' ms' : s < 120 ? s.toFixed(1) + ' s' : s < 7200 ? (s / 60).toFixed(1) + ' min' : (s / 3600).toFixed(1) + ' h';
  function setCell(id, dotCls, text, sub) {
    const dd = $(id).querySelector('dd'); dd.textContent = '';
    if (dotCls) dd.appendChild(h('span', 'dot ' + dotCls));
    dd.appendChild(document.createTextNode(text));
    if (sub) { dd.appendChild(h('div', 'muted', sub)); dd.lastChild.style.fontWeight = '400'; dd.lastChild.style.fontSize = '12px'; }
  }

  /* ---------------- console ---------------- */
  const log = $('con-log');
  function line(cls, text) { const d = h('div', 'l ' + cls, text); log.appendChild(d); log.scrollTop = log.scrollHeight; return d; }
  function localLine(text) { return line('s loc', text); }

  function addStep(ev) {
    const d = h('div', 'l'); const det = h('details'); const s = h('summary', null, '> ' + ev.name + '(' + JSON.stringify(ev.input || {}).slice(0, 80) + ')');
    det.appendChild(s); det.appendChild(h('pre', null, JSON.stringify(ev.input, null, 2))); d.appendChild(det); d.dataset.id = ev.id;
    log.appendChild(d); log.scrollTop = log.scrollHeight; return d;
  }
  function addResult(ev) {
    const steps = [...log.querySelectorAll('div.l[data-id]')].reverse().find(x => x.dataset.id === ev.id && !x.dataset.done);
    if (steps) { steps.dataset.done = 1; const det = steps.querySelector('details'); det.querySelector('summary').textContent += ev.ok ? '  [ok]' : '  [error]'; det.appendChild(h('pre', null, JSON.stringify(ev.result, null, 2).slice(0, 6000))); }
    const r = ev.result || {};
    if (ev.ok && r.status === 'started' && r.task_id) trackTask(r.task_id, r);
  }
  function addPending(ev) {
    const box = h('div', 'cbox');
    const p = ev.plan || {};
    box.appendChild(h('div', null, 'Confirmation required (' + ev.reason + '). Modality ' + p.modality + ', algorithms: ' + (p.algorithms || []).join(', ') +
      '; datasets: ' + (p.datasets || []).join(', ') + '; epochs: ' + p.epochs + '; device: ' + p.device + '. This may take a lot of compute time.'));
    const ok = h('button', 'btn primary', 'Confirm and launch'); ok.type = 'button';
    const no = h('button', 'btn', 'Cancel'); no.type = 'button';
    const answer = async approve => {
      ok.disabled = no.disabled = true;
      const r = await jpost('/api/assistant/confirm', {session_id: S.session, pending_id: ev.pending_id, approve});
      const j = r.data || {};
      if (!r.ok || j.error) line('e', 'Error: ' + (j.error || r.status));
      else if (!approve) line('s', 'Run cancelled: nothing was started.');
      else { line('s', 'Task launched: ' + j.task_id); if (j.task_id) trackTask(j.task_id, j); }
    };
    ok.onclick = () => answer(true); no.onclick = () => answer(false);
    box.appendChild(ok); box.appendChild(no); log.appendChild(box); log.scrollTop = log.scrollHeight;
    ok.focus();
  }
  function handle(ev) {
    switch (ev.type) {
      case 'session': S.session = ev.session_id; break;
      case 'text': line('b', ev.text); break;
      case 'offline': line('w', ev.text); break;
      case 'tool_call': addStep(ev); break;
      case 'tool_result': addResult(ev); break;
      case 'pending': addPending(ev); break;
      case 'warning': line('w', ev.text); break;
      case 'limit': line('w', 'Stopped: ' + ev.text); break;
      case 'error': line('e', 'Error: ' + ev.text); break;
    }
  }
  function setBusy(b) { S.busy = b; $('con-send').disabled = b; }
  async function stream(body) {
    setBusy(true);
    try {
      const resp = await fetch('/api/assistant/chat', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({session_id: S.session, ...body})});
      const rd = resp.body.getReader(), dec = new TextDecoder(); let buf = '';
      for (;;) {
        const {done, value} = await rd.read(); if (done) break;
        buf += dec.decode(value, {stream: true}); let i;
        while ((i = buf.indexOf('\n\n')) >= 0) {
          const c = buf.slice(0, i); buf = buf.slice(i + 2);
          if (c.startsWith('data: ')) { try { handle(JSON.parse(c.slice(6))); } catch (e) { /* chunk non valido */ } }
        }
      }
    } catch (e) { line('e', 'Connection: ' + e); }
    setBusy(false);
  }
  async function toolSync(name, args) {   // strumento diretto, risposta JSON, funziona anche offline
    const r = await jpost('/api/assistant/chat?stream=0', {session_id: S.session, offline_action: {tool: name, args}});
    const j = r.data || {}; if (j.session_id) S.session = j.session_id;
    return j.events || [];
  }

  const online = () => S.ai && S.ai.mode === 'online';
  const OFFLINE_HELP = 'The model is unreachable' + '. The buttons above stay active: they read the APIs and the results directly.';

  function failures() { return S.rows.filter(r => r.status !== 'SUCCESS'); }
  function lastLine(t) { const ls = String(t || '').split('\n').map(x => x.trim()).filter(Boolean); return ls.length ? ls[ls.length - 1].slice(0, 200) : '(no reason recorded)'; }
  function whatWentWrong() {
    if (online()) { send('What went wrong in the recorded runs? For each FAILED or FALLBACK outcome, report the reason you find in the results, without inventing causes.'); return; }
    if (!S.rowsLoaded) { localLine('Results not loaded yet.'); return; }
    if (!S.rows.length) { localLine('No runs recorded: there is nothing to analyze.'); return; }
    const bad = failures();
    if (!bad.length) { localLine('All ' + S.rows.length + ' recorded runs are SUCCESS.'); return; }
    const groups = new Map();
    bad.forEach(r => { const k = r.status + ' | ' + lastLine(r.err); const g = groups.get(k) || {n: 0, a: new Set(), d: new Set(), m: new Set()}; g.n++; g.a.add(r.algo); g.d.add(r.ds); g.m.add(r.mod); groups.set(k, g); });
    const out = [bad.length + ' of ' + S.rows.length + ' runs are not SUCCESS. Grouped by the last line of the recorded message:'];
    [...groups.entries()].sort((a, b) => b[1].n - a[1].n).forEach(([k, g]) => out.push('- ' + g.n + 'x ' + k + '\n    algorithms: ' + [...g.a].slice(0, 8).join(', ') + (g.a.size > 8 ? '...' : '') + ' (' + [...g.m].join(', ') + ')'));
    out.push('The reason is the one written in the CSVs; the actual cause must be checked in the log. For a narrative analysis the online assistant is needed.');
    localLine(out.join('\n'));
    S.filterMod = ''; $('r-st').value = 'FAILED'; renderRows();
  }
  function summaryLine() {
    if (!S.rowsLoaded) return localLine('Results not loaded yet.');
    const c = {}; S.rows.forEach(r => { c[r.status] = (c[r.status] || 0) + 1; });
    localLine(S.rows.length ? S.rows.length + ' runs recorded: ' + Object.entries(c).map(([k, v]) => v + ' ' + k).join(', ') + '.' : 'No runs recorded.');
  }
  function machinesLine() {
    if (S.machines == null) return localLine('Machine list unavailable.');
    if (!S.machines.length) return localLine('No remote machines registered. They are added in the Machines tab. Automatic workload assignment does not exist yet.');
    localLine(S.machines.map(m => m.name + ': ' + stateLabel(m.status && m.status.state || '?') + (m.gpu_count ? ', ' + m.gpu_count + ' GPU' : ', no GPU')).join('\n') + '\n(Automatic workload assignment does not exist yet.)');
  }
  async function advise() {
    const ds = $('f-ds').value;
    if (!ds) { localLine('Choose a dataset first: the advice depends on its data.'); return; }
    line('u', 'Recommend algorithms for ' + ds + ' (' + MODS[S.mod] + ')');
    const evs = await toolSync('recommend_algorithms', {modality: S.mod, dataset: ds, executable_only: true});
    const res = evs.find(e => e.type === 'tool_result');
    if (!res || !res.ok || !res.result.ranking) { line('e', 'Recommendation not available: ' + ((res && res.result && res.result.error) || 'no response')); return; }
    const rk = res.result.ranking.slice(0, 5);
    localLine('Top ' + rk.length + ' from the deterministic recommender (based on the measured CSVs, not on the model):\n' + rk.map((r, i) => (i + 1) + '. ' + r.algorithm).join('\n'));
    const b = h('button', 'btn', 'Select these in the form'); b.type = 'button';
    b.onclick = () => { const set = new Set(rk.map(r => r.algorithm)); document.querySelectorAll('#f-algs input').forEach(i => { i.checked = set.has(i.value); }); planText(); };
    const d = h('div', 'l'); d.appendChild(b); log.appendChild(d); log.scrollTop = log.scrollHeight;
  }
  function send(text) {
    text = (text || '').trim(); if (!text || S.busy) return;
    line('u', text);
    if (online()) { stream({message: text}); return; }
    if (/storto|sbagli|errori|fallit|falliment|failed|wrong|error|fail/i.test(text)) return whatWentWrong();
    if (/macchin|machine/i.test(text)) return machinesLine();
    if (/riepilog|esiti|quante|summary|outcome|how many/i.test(text)) return summaryLine();
    line('w', 'Offline mode: ' + ((S.ai && S.ai.reason) ? S.ai.reason.slice(0, 220) : 'unknown status') + '. ' + OFFLINE_HELP + ' No response is simulated.');
  }

  /* ---------------- stato ---------------- */
  async function loadSys() {
    try { S.sys = await jget('/api/status'); } catch (e) { S.sys = null; }
    const s = S.sys;
    if (!s) setCell('c-local', 'off', 'Unreachable');
    else setCell('c-local', 'on', s.gpu_available ? 'GPU available' : 'CPU only', s.device_name);
    const act = s && s.active_task;
    if (act) setCell('c-run', 'mid', '1 task ' + act.status, act.task_type + ' ' + act.task_id + ', ' + fmtT(act.duration_s));
    else setCell('c-run', '', S.sys ? 'No active task' : 'Unknown');
    return act;
  }
  async function loadAI() {
    try { S.ai = await jget('/api/assistant/status'); } catch (e) { S.ai = null; }
    const pill = $('con-mode');
    if (S.ai && S.ai.mode === 'online') {
      setCell('c-ai', 'on', 'Online', [S.ai.backend, S.ai.model].filter(Boolean).join(' / '));
      pill.textContent = 'online'; pill.className = 'pill on';
    } else {
      setCell('c-ai', 'mid', 'Offline', 'shortcuts only');
      pill.textContent = 'offline: shortcuts only'; pill.className = 'pill off';
      if (!S._offWarned) { S._offWarned = true; line('w', 'Assistant offline' + (S.ai && S.ai.reason ? ': ' + S.ai.reason.slice(0, 200) : '') + '. No response is simulated. The buttons above work (outcomes, errors, machines, advice), and so does launching from the form on the left.'); }
    }
  }
  async function loadMachines() {
    try {
      const j = await jget('/api/machines'); S.machines = j.machines || []; S.sshClient = j.ssh_client;
      const on = S.machines.filter(m => m.status && m.status.state === 'online').length;
      if (!S.machines.length) setCell('c-remote', '', 'None registered', 'workload not distributed');
      else setCell('c-remote', on ? 'on' : 'off', on + ' online of ' + S.machines.length, S.machines.reduce((a, m) => a + (m.gpu_count || 0), 0) + ' GPUs declared');
    } catch (e) { S.machines = null; setCell('c-remote', 'off', 'Unavailable'); }
    renderMachines();
  }

  /* ---------------- risultati ---------------- */
  function whenFromFile(f) { const m = /(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/.exec(f || ''); return m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}:${m[6]}` : ''; }
  async function loadRows() {
    const rows = []; const errs = [];
    await Promise.all(['tabular', 'ts', 'graph'].map(async mod => {
      try {
        const j = await jget(mod === 'tabular' ? '/api/runtime_results' : '/api/modality/' + mod + '/results?all=1');
        (Array.isArray(j) ? [] : (j.records || [])).forEach(r => rows.push({
          mod, algo: r.algorithm, ds: r.dataset, status: r.status || '?', err: r.error_message,
          t: mod === 'tabular' ? r.total_time_s : r.time_s, when: mod === 'tabular' ? (r.timestamp || '') : whenFromFile(r.file), file: r.file || j.filename || ''}));
      } catch (e) { errs.push(MODS[mod] + ': ' + e.message); }
    }));
    rows.sort((a, b) => (b.when || '').localeCompare(a.when || ''));
    S.rows = rows; S.rowsLoaded = true; S.rowsErr = errs;
    const c = {}; rows.forEach(r => { c[r.status] = (c[r.status] || 0) + 1; });
    if (!rows) setCell('c-out', '', 'None');
    else if (!rows.length) setCell('c-out', '', 'None', errs.join('; '));
    else setCell('c-out', c.FAILED ? 'mid' : 'on', rows.length + ' total', Object.entries(c).map(([k, v]) => v + ' ' + k).join(', '));
    renderRows();
  }
  function renderRows() {
    const q = $('r-q').value.trim().toLowerCase(), st = $('r-st').value;
    const rows = S.rows.filter(r => (!S.filterMod || r.mod === S.filterMod) && (!st || r.status === st) && (!q || (r.algo + ' ' + r.ds).toLowerCase().includes(q)));
    const body = $('r-body'); body.textContent = '';
    const maxL = Math.log10(1 + Math.max(1, ...rows.map(r => +r.t || 0)));
    rows.slice(0, 400).forEach(r => {
      const tr = h('tr', 'row'); tr.tabIndex = 0; tr.setAttribute('role', 'button'); tr.setAttribute('aria-expanded', 'false');
      tr.appendChild(h('td', null, r.when || '-')); tr.appendChild(h('td', null, MODS[r.mod])); tr.appendChild(h('td', null, r.algo)); tr.appendChild(h('td', null, r.ds));
      const ts = h('td'); ts.appendChild(h('span', 'badge ' + r.status, r.status)); tr.appendChild(ts);
      const tt = h('td', null, fmtT(+r.t)); const bar = h('span', 'bar'); bar.style.width = Math.max(2, 100 * Math.log10(1 + (+r.t || 0)) / maxL) + '%'; tt.appendChild(bar); tr.appendChild(tt);
      let det = null;
      const toggle = () => {
        if (det) { det.remove(); det = null; tr.setAttribute('aria-expanded', 'false'); return; }
        det = h('tr', 'det'); const td = h('td'); td.colSpan = 6;
        td.appendChild(h('div', 'muted', 'Source: ' + (r.file || 'unknown')));
        td.appendChild(h('pre', null, r.err ? String(r.err).slice(0, 3000) : (r.status === 'SUCCESS' ? 'No error recorded.' : 'No reason recorded in the CSV.')));
        det.appendChild(td); tr.after(det); tr.setAttribute('aria-expanded', 'true');
      };
      tr.onclick = toggle; tr.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } };
      body.appendChild(tr);
    });
    if (!rows.length) { const tr = h('tr'); const td = h('td', 'muted', S.rows.length ? 'No runs match the filters.' : 'No runs recorded. Launch one from the form on the left.'); td.colSpan = 6; tr.appendChild(td); body.appendChild(tr); }
    $('r-info').textContent = (S.rowsErr && S.rowsErr.length ? 'Read errors: ' + S.rowsErr.join('; ') + '. ' : '') + rows.length + ' of ' + S.rows.length + ' runs' + (rows.length > 400 ? ' (first 400 shown)' : '') + '. Tabular shows only the most recent CSV.';
    const seg = $('r-mod'); seg.textContent = '';
    [['', 'All'], ...Object.entries(MODS)].forEach(([k, v]) => { const b = h('button', null, v); b.type = 'button'; b.setAttribute('aria-pressed', String(S.filterMod === k)); b.onclick = () => { S.filterMod = k; renderRows(); }; seg.appendChild(b); });
  }

  /* ---------------- task lanciati da qui ---------------- */
  function trackTask(id, info) {
    if (S.tasks.find(t => t.id === id)) return;
    S.tasks.push({id, d: null}); renderTasks(); pollTasks();
  }
  async function pollTasks() {
    clearTimeout(S.pollTimer);
    let running = false;
    for (const t of S.tasks) {
      if (t.d && !['RUNNING', 'PENDING'].includes(t.d.status)) continue;
      try { t.d = await jget('/api/assistant/task/' + encodeURIComponent(t.id) + '?session_id=' + encodeURIComponent(S.session || '')); } catch (e) { t.d = {status: '?', error: e.message}; }
      if (t.d && ['RUNNING', 'PENDING'].includes(t.d.status)) running = true; else if (!t.done) { t.done = true; loadRows(); }
    }
    renderTasks(); loadSys();
    S.pollTimer = setTimeout(pollTasks, running ? 4000 : 15000);
    if (!running && S.tasks.every(t => t.done)) { clearTimeout(S.pollTimer); }
  }
  function renderTasks() {
    const box = $('tasks'); box.textContent = '';
    S.tasks.slice().reverse().forEach(t => {
      const d = t.d || {status: 'PENDING'}; const w = h('div', 't');
      w.appendChild(h('span', 'badge ' + d.status, d.status)); w.appendChild(h('b', null, 'Task ' + t.id));
      w.appendChild(h('span', 'muted', fmtT(d.duration_s))); if (d.command) w.appendChild(h('code', null, d.command.slice(-200)));
      if (d.error) w.appendChild(h('span', 'err', d.error));
      if (['RUNNING', 'PENDING'].includes(d.status)) { const b = h('button', 'btn small', 'Stop'); b.type = 'button'; b.style.margin = '0'; b.onclick = async () => { if (confirm('Stop task ' + t.id + '?')) { await jpost('/api/stop_task/' + encodeURIComponent(t.id)); pollTasks(); } }; w.appendChild(b); }
      box.appendChild(w);
    });
  }

  /* ---------------- modulo Nuova esecuzione ---------------- */
  function renderModSeg() {
    const seg = $('f-mod'); seg.textContent = '';
    Object.entries(MODS).forEach(([k, v]) => {
      const b = h('button', null, v); b.type = 'button'; b.setAttribute('role', 'radio'); b.setAttribute('aria-checked', String(S.mod === k));
      const m = S.mods.find(x => x.id === k); if (m) b.title = m.counts.executable + ' runnable algorithms, ' + m.n_datasets + ' datasets';
      b.onclick = () => { S.mod = k; renderModSeg(); loadForm(); }; seg.appendChild(b);
    });
  }
  async function loadForm() {
    const dsSel = $('f-ds'), box = $('f-algs'); dsSel.textContent = ''; box.textContent = '';
    try {
      const [a, d] = await Promise.all([jget('/api/modality/' + S.mod + '/algorithms'), jget('/api/modality/' + S.mod + '/datasets')]);
      S.algs = (a.algorithms || []).filter(x => x.runnable); S.dss = d.datasets || [];
      if (a.catalog_error) box.appendChild(h('p', 'err', 'Catalog: ' + a.catalog_error));
    } catch (e) { box.appendChild(h('p', 'err', 'Cannot read the catalog: ' + e.message)); S.algs = []; S.dss = []; }
    if (!S.dss.length) dsSel.appendChild(new Option('No datasets found', ''));
    else { dsSel.appendChild(new Option('Choose a dataset...', '')); S.dss.forEach(x => dsSel.appendChild(new Option(x.stem + (x.rows ? ' (' + x.rows + ' rows)' : ''), x.stem))); }
    renderAlgs();
  }
  function renderAlgs() {
    const box = $('f-algs'); const q = $('f-q').value.trim().toLowerCase();
    const keep = new Set([...box.querySelectorAll('input:checked')].map(i => i.value));
    box.textContent = '';
    const list = S.algs.filter(a => !q || (a.key + ' ' + (a.name || '')).toLowerCase().includes(q));
    list.forEach(a => {
      const l = h('label'); const i = h('input'); i.type = 'checkbox'; i.value = a.key; i.checked = keep.has(a.key); i.onchange = planText;
      const t = h('span'); t.textContent = (a.name || a.key); if (a.needs_gpu) { const s = h('small', null, ' (uses GPU)'); t.appendChild(s); }
      l.appendChild(i); l.appendChild(t); box.appendChild(l);
    });
    if (!list.length) box.appendChild(h('p', 'muted', S.algs.length ? 'No algorithm matches.' : 'No runnable algorithm.'));
    $('f-nalg').textContent = '(' + S.algs.length + ' runnable)';
    planText();
  }
  const picked = () => [...document.querySelectorAll('#f-algs input:checked')].map(i => i.value);
  function planText() {
    const a = picked(), ds = $('f-ds').value, ep = +$('f-ep').value || 0;
    $('f-plan').textContent = !ds ? 'Choose a dataset.' : !a.length ? 'Choose at least one algorithm.' :
      `Plan: ${a.length} algorithm(s) on ${ds}, ${ep} epochs, ${$('f-dev').value}. ` + (a.length > 1 || ep > 5 ? 'It will require your confirmation in the console.' : 'It starts right away.');
  }
  async function launch(e) {
    e.preventDefault();
    const a = picked(), ds = $('f-ds').value, ep = +$('f-ep').value;
    if (!ds || !a.length) { planText(); return; }
    if (!(ep >= 1 && ep <= 5000)) { $('f-plan').textContent = 'Epochs must be between 1 and 5000.'; return; }
    const args = {modality: S.mod, algorithms: a, datasets: [ds], epochs: ep, device: $('f-dev').value};
    if ($('f-ns').value) args.n_samples = +$('f-ns').value;
    line('u', 'Launch ' + a.join(', ') + ' on ' + ds + ' (' + MODS[S.mod] + ', ' + ep + ' epochs)');
    $('f-go').disabled = true;
    const evs = await toolSync('run_augmentation', args);
    evs.forEach(ev => handle(ev));
    const res = evs.find(x => x.type === 'tool_result');
    if (res && !res.ok) line('e', 'Not launched: ' + (res.result.error || 'error'));
    else if (res && res.result.status === 'started') line('s', 'Task launched: ' + res.result.task_id);
    $('f-go').disabled = false;
  }
  async function adviseBtn() { await advise(); }

  /* ---------------- macchine ---------------- */
  function renderMachines() {
    const box = $('m-list'); box.textContent = '';
    if (S.machines == null) { box.appendChild(h('p', 'err', 'Machine list unavailable.')); return; }
    if (!S.machines.length) box.appendChild(h('p', 'muted', 'No machines registered. Use the form below.'));
    if (S.sshClient === false) box.appendChild(h('p', 'err', 'ssh client not found on the server: the check cannot work.'));
    S.machines.forEach(m => {
      const w = h('div', 'm'); const i = h('div', 'i'); const st = (m.status && m.status.state) || 'non_verificata';
      const dot = h('span', 'dot ' + (st === 'online' ? 'on' : st === 'non_verificata' ? '' : 'off'));
      const b = h('b'); b.appendChild(dot); b.appendChild(document.createTextNode(m.name + ' - ' + stateLabel(st)));
      i.appendChild(b);
      i.appendChild(h('div', 'muted', m.user + '@' + m.host + ':' + m.port + ' | ' + m.os + ' | ' + (m.gpu_count ? m.gpu_count + ' GPU ' + (m.gpu_model || '') : 'no GPU') + (m.ram_gb ? ' | ' + m.ram_gb + ' GB RAM' : '') + (m.enabled ? '' : ' | disabled')));
      if (m.status && m.status.message) i.appendChild(h('div', 'muted', m.status.message.slice(0, 200)));
      const chk = h('button', 'btn small', 'Check'); chk.type = 'button'; chk.style.margin = '0';
      chk.onclick = async () => { chk.disabled = true; chk.textContent = 'Checking...'; await jpost('/api/machines/' + encodeURIComponent(m.id) + '/check'); await loadMachines(); };
      const del = h('button', 'btn small', 'Delete'); del.type = 'button'; del.style.margin = '0';
      del.onclick = async () => { if (confirm('Delete machine "' + m.name + '" from the list?')) { await fetch('/api/machines/' + encodeURIComponent(m.id), {method: 'DELETE'}); loadMachines(); } };
      w.appendChild(i); w.appendChild(chk); w.appendChild(del); box.appendChild(w);
    });
  }
  async function addMachine(e) {
    e.preventDefault(); const f = e.target, msg = $('m-msg'); msg.className = ''; msg.textContent = 'Saving...';
    const data = {}; new FormData(f).forEach((v, k) => { if (v !== '') data[k] = v; });
    const r = await jpost('/api/machines', data);
    if (r.ok) { msg.textContent = 'Machine saved. Use "Check" to test the connection.'; f.reset(); loadMachines(); }
    else { msg.className = 'err'; const j = r.data || {}; msg.textContent = (j.error || 'Error') + (j.fields ? ': ' + Object.entries(j.fields).map(([k, v]) => k + ' - ' + v).join('; ') : ''); }
  }

  /* ---------------- tab, avvio ---------------- */
  function tab(which) {
    const runs = which === 'runs';
    $('t-runs').setAttribute('aria-selected', runs); $('t-mach').setAttribute('aria-selected', !runs);
    $('t-runs').tabIndex = runs ? 0 : -1; $('t-mach').tabIndex = runs ? -1 : 0;
    $('p-runs').hidden = !runs; $('p-mach').hidden = runs;
  }
  function init() {
    $('t-runs').onclick = () => tab('runs'); $('t-mach').onclick = () => tab('mach');
    document.querySelector('.tabs').addEventListener('keydown', e => { if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { const r = $('t-runs').getAttribute('aria-selected') === 'true'; tab(r ? 'mach' : 'runs'); ($(r ? 't-mach' : 't-runs')).focus(); } });
    $('r-q').oninput = renderRows; $('r-st').onchange = renderRows; $('r-refresh').onclick = () => { loadRows(); loadSys(); loadMachines(); loadAI(); };
    $('f-q').oninput = renderAlgs; $('f-ep').oninput = planText; $('f-dev').onchange = planText; $('f-ds').onchange = planText;
    $('run-form').onsubmit = launch; $('f-advise').onclick = adviseBtn; $('m-form').onsubmit = addMachine;
    $('con-form').onsubmit = e => { e.preventDefault(); const v = $('con-inp').value; $('con-inp').value = ''; send(v); };
    $('con-reset').onclick = async () => { if (S.session) await jpost('/api/assistant/reset', {session_id: S.session}); S.session = null; log.textContent = ''; line('s', 'New session.'); };
    $('con-size').onclick = e => { const big = $('con-log').closest('.console').classList.toggle('big'); e.target.setAttribute('aria-pressed', big); e.target.textContent = big ? 'Shrink' : 'Enlarge'; };
    const chips = [['What went wrong?', whatWentWrong], ['Outcomes summary', summaryLine], ['Machine status', machinesLine], ['Recommend algorithms', adviseBtn]];
    chips.forEach(([t, fn]) => { const b = h('button', null, t); b.type = 'button'; b.onclick = fn; $('chips').appendChild(b); });
    line('s', 'Assistant console. Ask what went wrong, the machine status or a recommendation; to launch, use the form on the left.');
    jget('/api/modalities').then(m => { S.mods = m; renderModSeg(); }).catch(() => renderModSeg());
    renderModSeg(); loadForm(); loadSys().then(act => { if (act) trackTask(act.task_id); }); loadAI(); loadMachines(); loadRows();
    setInterval(() => { if (!document.hidden) loadSys(); }, 20000);
  }
  init();
})();
