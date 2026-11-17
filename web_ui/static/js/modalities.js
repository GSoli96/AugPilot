/* modalities.js
 * Selettore di modalita' (Tabulare / Serie temporali / Grafi) e pagina per ts e graph.
 * Usa le funzioni globali di app.js (switchTab, clearConsole, openConsoleDrawer, startLogPolling, state).
 * Il tabulare continua a usare il codice esistente di app.js.
 */
const modState = {
    mode: 'tabular',
    algos: [],
    datasets: [],
    selAlgos: new Set(),
    selDatasets: new Set(),
    device: 'auto',
    watchTimer: null,
};

const MOD_LABELS = { tabular: 'Tabular', ts: 'Time series', graph: 'Graphs' };

function modEsc(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

document.addEventListener('DOMContentLoaded', () => {
    let saved = 'tabular';
    try { saved = localStorage.getItem('dataaug_mode') || 'tabular'; } catch (e) { /* storage non disponibile */ }
    document.body.classList.add('mode-tabular');
    loadModalityOverview();
    if (saved !== 'tabular' && MOD_LABELS[saved]) setMode(saved);
});

async function loadModalityOverview() {
    try {
        const res = await fetch('/api/modalities');
        const data = await res.json();
        data.forEach(m => {
            const el = document.getElementById('mode-count-' + m.id);
            if (!el) return;
            const c = m.counts;
            el.textContent = `(${c.executable} exe / ${c.wired} wired / ${c.described} desc.)`;
            el.title = `${c.total} algorithms, ${m.n_datasets} datasets` + (m.catalog_error ? ` - CATALOG ERROR: ${m.catalog_error}` : '');
        });
    } catch (e) {
        console.error('Errore /api/modalities', e);
    }
}

function setMode(mode) {
    modState.mode = mode;
    try { localStorage.setItem('dataaug_mode', mode); } catch (e) { /* ignora */ }
    document.querySelectorAll('#mode-switch .btn-toggle').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
    document.body.classList.toggle('mode-tabular', mode === 'tabular');
    document.body.classList.toggle('mode-other', mode !== 'tabular');
    if (mode === 'tabular') {
        switchTab('tab-augment');
    } else {
        switchTab('tab-modality');
        loadModality(mode);
    }
}

async function loadModality(mode) {
    const nameEl = document.getElementById('mod-name');
    if (nameEl) nameEl.textContent = MOD_LABELS[mode] || mode;
    modState.selAlgos.clear();
    modState.selDatasets.clear();
    document.getElementById('mod-algos').innerHTML = '<div class="loading-state"><i class="fa-solid fa-spinner fa-spin"></i> Loading...</div>';
    document.getElementById('mod-datasets').innerHTML = '<div class="loading-state"><i class="fa-solid fa-spinner fa-spin"></i> Loading...</div>';
    const errBox = document.getElementById('mod-catalog-error');
    errBox.style.display = 'none';
    try {
        const [ra, rd] = await Promise.all([
            fetch(`/api/modality/${mode}/algorithms`).then(r => r.json()),
            fetch(`/api/modality/${mode}/datasets`).then(r => r.json()),
        ]);
        if (modState.mode !== mode) return;   // l'utente ha cambiato modalita' nel frattempo
        modState.algos = ra.algorithms || [];
        modState.datasets = rd.datasets || [];
        if (ra.catalog_error) {
            errBox.style.display = 'block';
            errBox.textContent = 'Catalog could not be loaded (it may be being written): ' + ra.catalog_error;
        }
        const counts = { executable: 0, wired: 0, described: 0 };
        modState.algos.forEach(a => { counts[a.status] = (counts[a.status] || 0) + 1; });
        document.getElementById('mod-counts').innerHTML =
            `<span class="badge-status executable">executable ${counts.executable}</span>` +
            `<span class="badge-status wired">wired ${counts.wired}</span>` +
            `<span class="badge-status described">described ${counts.described}</span>` +
            `<span class="help-text">${modState.datasets.length} datasets</span>`;
        renderModDatasets();
        renderModAlgos();
        updateModSummary();
        loadModResults(true);
    } catch (e) {
        document.getElementById('mod-algos').innerHTML = `<div class="mod-empty">Error: ${modEsc(e)}</div>`;
    }
}

function renderModDatasets() {
    const box = document.getElementById('mod-datasets');
    if (!modState.datasets.length) {
        box.innerHTML = '<div class="mod-empty">No datasets in this modality&#39;s folder.</div>';
        return;
    }
    box.innerHTML = modState.datasets.map(d => `
        <label class="mod-algo-card">
            <input type="checkbox" data-stem="${modEsc(d.stem)}" ${modState.selDatasets.has(d.stem) ? 'checked' : ''} onchange="toggleModSel('datasets', this)">
            <div><strong>${modEsc(d.stem)}</strong><div class="mod-algo-meta">${modEsc(d.filename)} - ${d.size_kb} KB</div></div>
        </label>`).join('');
}

function renderModAlgos() {
    const box = document.getElementById('mod-algos');
    const q = (document.getElementById('mod-algo-search').value || '').toLowerCase().trim();
    let list = modState.algos;
    if (q) list = list.filter(a => [a.key, a.name, a.strategy, a.status, a.task].join(' ').toLowerCase().includes(q));
    if (!list.length) {
        box.innerHTML = '<div class="mod-empty">No algorithms in the catalog (this modality&#39;s catalog may still be under construction).</div>';
        return;
    }
    box.innerHTML = list.map(a => {
        const dis = !a.runnable;
        const tip = [a.notes, a.source, a.paper].filter(Boolean).join(' | ');
        return `
        <label class="mod-algo-card ${dis ? 'disabled' : ''}" title="${modEsc(tip)}">
            <input type="checkbox" data-key="${modEsc(a.key)}" ${dis ? 'disabled' : ''} ${modState.selAlgos.has(a.key) ? 'checked' : ''} onchange="toggleModSel('algos', this)">
            <div>
                <strong>${modEsc(a.name)}</strong> <span class="badge-status ${modEsc(a.status)}">${modEsc(a.status)}</span>
                ${a.needs_gpu ? '<span class="badge-status wired">GPU</span>' : ''}
                <div class="mod-algo-meta">${modEsc(a.key)} - ${modEsc(a.strategy)}${a.task && a.task !== 'generation' ? ' - ' + modEsc(a.task) : ''}${dis ? ' - not runnable' : ''}</div>
            </div>
        </label>`;
    }).join('');
}

function toggleModSel(which, input) {
    const set = which === 'algos' ? modState.selAlgos : modState.selDatasets;
    const id = which === 'algos' ? input.dataset.key : input.dataset.stem;
    if (input.checked) set.add(id); else set.delete(id);
    updateModSummary();
}

function modSelectAll(which, on) {
    if (which === 'algos') {
        modState.selAlgos.clear();
        if (on) modState.algos.filter(a => a.runnable).forEach(a => modState.selAlgos.add(a.key));
        renderModAlgos();
    } else {
        modState.selDatasets.clear();
        if (on) modState.datasets.forEach(d => modState.selDatasets.add(d.stem));
        renderModDatasets();
    }
    updateModSummary();
}

function updateModSummary() {
    const a = modState.selAlgos.size, d = modState.selDatasets.size;
    const el = document.getElementById('mod-selection-summary');
    if (el) el.textContent = `${a} algorithms x ${d} datasets = ${a * d} runs.`;
}

function setModDevice(dev) {
    modState.device = dev;
    document.querySelectorAll('#mod-device-group .btn-toggle').forEach(b => b.classList.toggle('active', b.dataset.device === dev));
}

async function startModalityRun() {
    const algos = Array.from(modState.selAlgos), datasets = Array.from(modState.selDatasets);
    if (!algos.length || !datasets.length) {
        alert('Select at least one dataset and one runnable algorithm.');
        return;
    }
    let confirmed = false;
    if (algos.length > 1 || datasets.length > 1) {
        const ok = confirm(`You are about to start ${algos.length} algorithms on ${datasets.length} datasets (${algos.length * datasets.length} runs, ${MOD_LABELS[modState.mode]}).\n` +
            'They may take a long time and need a GPU. Continue?');
        if (!ok) return;
        confirmed = true;
    }
    const payload = {
        modality: modState.mode,
        algorithms: algos,
        datasets: datasets,
        epochs: parseInt(document.getElementById('mod-epochs').value) || 5,
        n_samples: document.getElementById('mod-n-samples').value || null,
        seed: parseInt(document.getElementById('mod-seed').value) || 0,
        device: modState.device,
        no_eval: document.getElementById('mod-no-eval').checked,
        confirmed: confirmed,
    };
    try {
        const res = await fetch('/api/run_modality', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (data.success) {
            state.activeTaskId = data.task_id;
            clearConsole();
            openConsoleDrawer();
            startLogPolling(data.task_id);
            document.getElementById('stop-task-btn').disabled = false;
            watchModalityTask(data.task_id);
        } else {
            alert('Error: ' + (data.error || 'unknown'));
        }
    } catch (e) {
        alert('Network error: ' + e);
    }
}

function watchModalityTask(taskId) {
    if (modState.watchTimer) clearInterval(modState.watchTimer);
    modState.watchTimer = setInterval(async () => {
        try {
            const r = await fetch(`/api/logs/${taskId}?start=1000000000`).then(x => x.json());
            if (r.status !== 'RUNNING' && r.status !== 'PENDING') {
                clearInterval(modState.watchTimer);
                modState.watchTimer = null;
                if (modState.mode !== 'tabular') loadModResults(true);
            }
        } catch (e) { /* riprova al giro dopo */ }
    }, 3000);
}

async function loadModResults(latest) {
    const mode = modState.mode;
    if (mode === 'tabular') return;
    const sel = document.getElementById('mod-results-file');
    const file = (!latest && sel.value) ? `?file=${encodeURIComponent(sel.value)}` : '';
    const head = document.getElementById('mod-results-head');
    const body = document.getElementById('mod-results-body');
    try {
        const data = await fetch(`/api/modality/${mode}/results${file}`).then(r => r.json());
        if (modState.mode !== mode) return;
        const files = data.files || [];
        sel.innerHTML = files.map(f => `<option value="${modEsc(f)}" ${f === data.filename ? 'selected' : ''}>${modEsc(f)}</option>`).join('');
        const recs = data.records || [];
        document.getElementById('mod-results-count').textContent = `${recs.length} records`;
        if (!recs.length) {
            head.innerHTML = '';
            body.innerHTML = '<tr><td class="text-center">No results for this modality.</td></tr>';
            return;
        }
        const first = ['status', 'dataset', 'algorithm', 'epochs', 'n_generated', 'time_s', 'peak_mem_mb', 'error_message'];
        const all = [];
        recs.forEach(r => Object.keys(r).forEach(k => { if (!all.includes(k)) all.push(k); }));
        const cols = first.filter(k => all.includes(k)).concat(all.filter(k => !first.includes(k) && k !== 'modality'));
        head.innerHTML = '<tr>' + cols.map(c => `<th>${modEsc(c)}</th>`).join('') + '</tr>';
        body.innerHTML = recs.map(r => '<tr>' + cols.map(c => {
            let v = r[c];
            if (c === 'status') {
                const cls = v === 'SUCCESS' ? 'success' : (v === 'FALLBACK' ? 'warning' : 'failed');
                return `<td><span class="status-pill ${cls}">${modEsc(v)}</span></td>`;
            }
            if (typeof v === 'number') v = Number.isInteger(v) ? v : v.toFixed(4);
            const txt = v == null ? '-' : String(v);
            return `<td title="${modEsc(txt)}" style="${c === 'error_message' ? 'max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;' : ''}">${modEsc(txt)}</td>`;
        }).join('') + '</tr>').join('');
    } catch (e) {
        body.innerHTML = `<tr><td class="text-danger">Error: ${modEsc(e)}</td></tr>`;
    }
}
