/**
 * app.js
 * ======
 * Main frontend application logic for the Data Augmentation Dashboard.
 */

// Application State
const state = {
    datasets: [],
    selectedDatasets: new Set(),
    algorithms: [],
    selectedAlgos: new Set(),
    device: 'auto',
    activeTaskId: null,
    logOffset: 0,
    logTimer: null,
    currentAnalytics: null,
    // Interactive Datatables
    rankingsData: [],
    rankingsFilter: '',
    rankingsSortKey: 'rank',
    rankingsSortAsc: true,
    runtimeData: [],
    runtimeFilter: '',
    runtimeSortKey: 'timestamp',
    runtimeSortAsc: false,
    // Checkpoints
    checkpoints: [],
    checkpointsFilter: '',
    selectedCheckpoint: null,
    ckptDevice: 'auto',
};

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initApp();
});

async function initApp() {
    await Promise.all([
        fetchSystemStatus(),
        loadDatasets(),
        loadAlgorithms(),
        loadDcFiles(),
        loadEvaluatedDatasetsDropdown(),
    ]);

    // Check if task is already running in background
    checkActiveTask();
    setInterval(checkActiveTask, 4000);
}

// --------------------------------------------------------------------------
// Theme & Appearance Customizer
// --------------------------------------------------------------------------
function initTheme() {
    const savedTheme = localStorage.getItem('ui_theme') || 'dark';
    if (savedTheme === 'light') {
        document.body.classList.add('light-theme');
        updateThemeButtonUI(true);
    } else {
        document.body.classList.remove('light-theme');
        updateThemeButtonUI(false);
    }

    const savedAccent = localStorage.getItem('ui_accent');
    if (savedAccent) {
        try {
            const parsed = JSON.parse(savedAccent);
            setAccentColor(parsed.name, parsed.hex, false);
        } catch (e) {}
    }
}

function toggleTheme() {
    const isLight = document.body.classList.toggle('light-theme');
    localStorage.setItem('ui_theme', isLight ? 'light' : 'dark');
    updateThemeButtonUI(isLight);

    // Re-render plots if analytics data is loaded
    if (state.currentAnalytics) {
        renderTradeoffScatter('plot-tradeoff', state.currentAnalytics.scatter);
        renderRadarProfile('plot-radar', state.currentAnalytics.radar);
        renderDiversityBarChart();
        renderMetricsHeatmap('plot-heatmap', state.currentAnalytics.heatmap);
    }
}

function updateThemeButtonUI(isLight) {
    const icon = document.getElementById('theme-icon');
    const label = document.getElementById('theme-label');
    if (!icon || !label) return;
    if (isLight) {
        icon.className = 'fa-solid fa-moon';
        label.innerText = 'Dark';
    } else {
        icon.className = 'fa-solid fa-sun';
        label.innerText = 'Light';
    }
}

function setAccentColor(name, hex, save = true) {
    document.documentElement.style.setProperty('--accent-primary', hex);
    document.documentElement.style.setProperty('--accent-cyan', hex);
    document.querySelectorAll('.color-dot').forEach(dot => {
        dot.classList.toggle('active', dot.classList.contains(`dot-${name}`));
    });
    if (save) {
        localStorage.setItem('ui_accent', JSON.stringify({ name, hex }));
    }
}

// --------------------------------------------------------------------------
// System Status & Background Tasks
// --------------------------------------------------------------------------
async function fetchSystemStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const gpuBadge = document.getElementById('gpu-badge');
        if (data.gpu_available) {
            gpuBadge.className = 'badge badge-device';
            gpuBadge.innerHTML = `<i class="fa-solid fa-bolt"></i> ${data.device_name}`;
        } else {
            gpuBadge.className = 'badge badge-device';
            gpuBadge.innerHTML = `<i class="fa-solid fa-microchip"></i> CPU Only`;
        }
    } catch (e) {
        console.error('Error fetching system status:', e);
    }
}

async function checkActiveTask() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const taskBadge = document.getElementById('task-status-badge');
        const stopBtn = document.getElementById('stop-task-btn');

        if (data.active_task && data.active_task.status === 'RUNNING') {
            state.activeTaskId = data.active_task.task_id;
            taskBadge.className = 'badge badge-running';
            taskBadge.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Running: ${data.active_task.task_id}`;
            stopBtn.disabled = false;
            document.getElementById('console-task-id').innerText = `Task ID: ${state.activeTaskId}`;
            if (!state.logTimer) {
                startLogPolling(state.activeTaskId);
            }
        } else {
            if (state.activeTaskId && !data.active_task) {
                // Task just finished
                taskBadge.className = 'badge badge-idle';
                taskBadge.innerHTML = `<i class="fa-solid fa-circle-check"></i> System Idle`;
                stopBtn.disabled = true;
                state.activeTaskId = null;
                // Auto refresh evaluated datasets and runtime results
                loadEvaluatedDatasetsDropdown();
                loadRuntimeResults();
            }
        }
    } catch (e) {
        console.error('Error checking active task:', e);
    }
}

// --------------------------------------------------------------------------
// Tab Navigation
// --------------------------------------------------------------------------
function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
    });
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === tabId);
    });

    if (tabId === 'tab-analytics' && !state.currentAnalytics) {
        loadAnalyticsData();
    } else if (tabId === 'tab-runtime') {
        loadRuntimeResults();
    } else if (tabId === 'tab-checkpoints') {
        loadCheckpoints();
        populateCkptDatasetsDropdown();
    }
}

// --------------------------------------------------------------------------
// Dataset Discovery & Selection
// --------------------------------------------------------------------------
async function loadDatasets() {
    const container = document.getElementById('datasets-grid');
    const evalSelect = document.getElementById('eval-dataset-select');
    try {
        const res = await fetch('/api/datasets');
        state.datasets = await res.json();

        if (state.datasets.length === 0) {
            container.innerHTML = '<div class="text-center p-3 text-muted">No CSV files found in datasets/</div>';
            return;
        }

        evalSelect.innerHTML = '<option value="all">All datasets (global evaluation)</option>';

        container.innerHTML = '';
        state.datasets.forEach(ds => {
            const isSel = state.selectedDatasets.has(ds.stem);
            const item = document.createElement('div');
            item.className = `dataset-item ${isSel ? 'selected' : ''}`;
            item.id = `ds-card-${ds.stem}`;
            item.onclick = (e) => toggleDataset(ds.stem, e);
            item.innerHTML = `
                <input type="checkbox" id="chk-ds-${ds.stem}" ${isSel ? 'checked' : ''} onclick="event.stopPropagation(); toggleDataset('${ds.stem}')">
                <div style="flex:1">
                    <div class="ds-name">${ds.stem}</div>
                    <div class="ds-meta">${ds.rows} rows • ${ds.columns} cols • ${ds.size_kb} KB</div>
                </div>
            `;
            container.appendChild(item);

            // Add to evaluation select dropdown
            const opt = document.createElement('option');
            opt.value = ds.stem;
            opt.innerText = `${ds.stem} (${ds.rows} rows)`;
            evalSelect.appendChild(opt);
        });

    } catch (e) {
        container.innerHTML = `<div class="text-center text-danger p-3">Error loading datasets: ${e}</div>`;
    }
}

function toggleDataset(stem, event) {
    const chk = document.getElementById(`chk-ds-${stem}`);
    const card = document.getElementById(`ds-card-${stem}`);
    if (state.selectedDatasets.has(stem)) {
        state.selectedDatasets.delete(stem);
        if (chk) chk.checked = false;
        if (card) card.classList.remove('selected');
    } else {
        state.selectedDatasets.add(stem);
        if (chk) chk.checked = true;
        if (card) card.classList.add('selected');
    }
}

function selectAllDatasets(select) {
    state.datasets.forEach(ds => {
        const chk = document.getElementById(`chk-ds-${ds.stem}`);
        const card = document.getElementById(`ds-card-${ds.stem}`);
        if (select) {
            state.selectedDatasets.add(ds.stem);
            if (chk) chk.checked = true;
            if (card) card.classList.add('selected');
        } else {
            state.selectedDatasets.delete(ds.stem);
            if (chk) chk.checked = false;
            if (card) card.classList.remove('selected');
        }
    });
}

function filterDatasets() {
    const q = document.getElementById('dataset-search').value.toLowerCase();
    state.datasets.forEach(ds => {
        const el = document.getElementById(`ds-card-${ds.stem}`);
        if (el) {
            el.style.display = ds.stem.toLowerCase().includes(q) ? 'flex' : 'none';
        }
    });
}

// --------------------------------------------------------------------------
// Algorithm Discovery & Selection
// --------------------------------------------------------------------------
async function loadAlgorithms() {
    const container = document.getElementById('algos-container');
    try {
        const res = await fetch('/api/algorithms');
        state.algorithms = await res.json();

        container.innerHTML = '';
        state.algorithms.forEach(algo => {
            const isSel = state.selectedAlgos.has(algo.id);
            const card = document.createElement('div');
            card.className = `algo-card ${isSel ? 'selected' : ''}`;
            card.id = `algo-card-${algo.id}`;
            card.setAttribute('data-category', algo.category);
            card.onclick = () => toggleAlgorithm(algo.id);
            card.innerHTML = `
                <div class="algo-card-top">
                    <input type="checkbox" id="chk-algo-${algo.id}" ${isSel ? 'checked' : ''} onclick="event.stopPropagation(); toggleAlgorithm('${algo.id}')">
                    <div class="algo-title">${algo.name}</div>
                </div>
                <div class="algo-desc">${algo.desc}</div>
            `;
            container.appendChild(card);
        });

    } catch (e) {
        container.innerHTML = `<div class="text-center text-danger p-3">Error loading algorithms: ${e}</div>`;
    }
}

function toggleAlgorithm(id) {
    const chk = document.getElementById(`chk-algo-${id}`);
    const card = document.getElementById(`algo-card-${id}`);
    if (state.selectedAlgos.has(id)) {
        state.selectedAlgos.delete(id);
        if (chk) chk.checked = false;
        if (card) card.classList.remove('selected');
    } else {
        state.selectedAlgos.add(id);
        if (chk) chk.checked = true;
        if (card) card.classList.add('selected');
    }
}

function selectAlgosByCategory(cat) {
    state.algorithms.forEach(algo => {
        const chk = document.getElementById(`chk-algo-${algo.id}`);
        const card = document.getElementById(`algo-card-${algo.id}`);
        const shouldSelect = (cat === 'all') || (cat !== 'none' && algo.category === cat);

        if (shouldSelect) {
            state.selectedAlgos.add(algo.id);
            if (chk) chk.checked = true;
            if (card) card.classList.add('selected');
        } else if (cat === 'none') {
            state.selectedAlgos.delete(algo.id);
            if (chk) chk.checked = false;
            if (card) card.classList.remove('selected');
        }
    });
}

function filterAlgoCards(category, chipEl) {
    document.querySelectorAll('.cat-chip').forEach(c => c.classList.remove('active'));
    if (chipEl) chipEl.classList.add('active');

    state.algorithms.forEach(algo => {
        const el = document.getElementById(`algo-card-${algo.id}`);
        if (el) {
            el.style.display = (category === 'all' || algo.category === category) ? 'block' : 'none';
        }
    });
}

// --------------------------------------------------------------------------
// Parameter Setters
// --------------------------------------------------------------------------
function setDevice(dev) {
    state.device = dev;
    document.querySelectorAll('#device-group .btn-toggle').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-device') === dev);
    });
}

function toggleConstraintDetails() {
    const body = document.getElementById('constraint-body');
    const chevron = document.getElementById('constraint-chevron');
    body.classList.toggle('hidden');
    chevron.classList.toggle('fa-chevron-down');
    chevron.classList.toggle('fa-chevron-up');
}

async function loadDcFiles() {
    const select = document.getElementById('kamino-dc-select');
    try {
        const res = await fetch('/api/dc_files');
        const files = await res.json();
        files.forEach(f => {
            const opt = document.createElement('option');
            opt.value = f.path;
            opt.innerText = `${f.filename} (${Math.round(f.size_bytes / 1024)} KB)`;
            select.appendChild(opt);
        });
    } catch (e) {
        console.error('Error loading DC files:', e);
    }
}

function syncDcPathInput() {
    const sel = document.getElementById('kamino-dc-select');
    const custom = document.getElementById('kamino-dc-custom');
    if (sel.value) {
        custom.value = sel.value;
    }
}

// --------------------------------------------------------------------------
// Execution: Augmentation & Evaluation
// --------------------------------------------------------------------------
async function startAugmentation() {
    const selectedDatasets = Array.from(state.selectedDatasets);
    const selectedAlgos = Array.from(state.selectedAlgos);

    if (selectedDatasets.length === 0) {
        alert('Select at least one dataset before starting the benchmark.');
        return;
    }
    if (selectedAlgos.length === 0) {
        alert('Select at least one algorithm before starting the benchmark.');
        return;
    }

    const nSamplesVal = parseInt(document.getElementById('param-n-samples').value) || 1000;
    const epochsVal = parseInt(document.getElementById('param-epochs').value) || 1000;
    const resumeVal = document.getElementById('param-resume').checked;
    const dcFileVal = document.getElementById('kamino-dc-custom').value.trim() || null;
    const kaminoWeightVal = document.querySelector('input[name="kamino_weight"]:checked')?.value || 'toggled';

    const payload = {
        algorithms: selectedAlgos,
        datasets: selectedDatasets,
        n_samples: nSamplesVal,
        epochs: epochsVal,
        device: state.device,
        use_gpu: state.device === 'cuda' ? true : (state.device === 'cpu' ? false : null),
        resume: resumeVal,
        dc_file: dcFileVal,
        kamino_weights: kaminoWeightVal,
    };

    try {
        const res = await fetch('/api/run_augmentation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await res.json();

        if (data.success) {
            state.activeTaskId = data.task_id;
            clearConsole();
            openConsoleDrawer();
            startLogPolling(data.task_id);
            document.getElementById('stop-task-btn').disabled = false;
        } else {
            alert('Error starting the benchmark: ' + data.error);
        }
    } catch (e) {
        alert('Connection error: ' + e);
    }
}

async function startEvaluation() {
    const datasetVal = document.getElementById('eval-dataset-select').value;
    const algoMode = document.querySelector('input[name="eval_algo_mode"]:checked')?.value || 'all';
    let algorithms = ['all'];

    if (algoMode === 'sydc') {
        algorithms = ['sydc'];
    } else if (algoMode === 'custom') {
        const customInput = document.getElementById('eval-custom-algos').value.trim();
        if (customInput) {
            algorithms = customInput.split(/\s+/);
        }
    }

    const binsVal = parseInt(document.getElementById('eval-bins').value) || 30;

    const payload = {
        dataset: datasetVal,
        algorithms: algorithms,
        bins: binsVal,
    };

    try {
        const res = await fetch('/api/run_evaluation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await res.json();

        if (data.success) {
            state.activeTaskId = data.task_id;
            clearConsole();
            openConsoleDrawer();
            startLogPolling(data.task_id);
            document.getElementById('stop-task-btn').disabled = false;
        } else {
            alert('Error starting the evaluation: ' + data.error);
        }
    } catch (e) {
        alert('Connection error: ' + e);
    }
}

function toggleEvalAlgoCustom(show) {
    document.getElementById('eval-custom-algo-box').classList.toggle('hidden', !show);
}

async function stopCurrentTask() {
    if (!state.activeTaskId) return;
    if (!confirm(`Are you sure you want to stop task ${state.activeTaskId}?`)) return;

    try {
        const res = await fetch(`/api/stop_task/${state.activeTaskId}`, { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            document.getElementById('stop-task-btn').disabled = true;
        }
    } catch (e) {
        alert('Error sending the stop request: ' + e);
    }
}

// --------------------------------------------------------------------------
// Console Drawer & Live Log Streaming
// --------------------------------------------------------------------------
function toggleConsoleDrawer() {
    const drawer = document.getElementById('console-drawer');
    drawer.classList.toggle('open');
}

function openConsoleDrawer() {
    document.getElementById('console-drawer').classList.add('open');
}

function clearConsole() {
    document.getElementById('console-output').innerText = '';
    state.logOffset = 0;
}

function startLogPolling(taskId) {
    if (state.logTimer) clearInterval(state.logTimer);
    state.logOffset = 0;

    state.logTimer = setInterval(async () => {
        try {
            const res = await fetch(`/api/logs/${taskId}?start=${state.logOffset}`);
            const data = await res.json();

            if (data.lines && data.lines.length > 0) {
                const consoleEl = document.getElementById('console-output');
                consoleEl.innerText += data.lines.join('');
                consoleEl.scrollTop = consoleEl.scrollHeight;
                state.logOffset = data.next_index;
            }

            if (data.status !== 'RUNNING' && data.status !== 'PENDING') {
                clearInterval(state.logTimer);
                state.logTimer = null;
                document.getElementById('stop-task-btn').disabled = true;
                checkActiveTask();
            }
        } catch (e) {
            console.error('Error fetching logs:', e);
        }
    }, 800);
}

// --------------------------------------------------------------------------
// Visual Analytics & Plotly Charts Integration
// --------------------------------------------------------------------------
async function loadEvaluatedDatasetsDropdown() {
    const select = document.getElementById('analytics-dataset-select');
    try {
        const res = await fetch('/api/evaluated_datasets');
        const datasets = await res.json();

        select.innerHTML = '';
        if (datasets.length === 0) {
            select.innerHTML = '<option value="">No evaluated dataset in evaluation_metrics/</option>';
            return;
        }

        datasets.forEach(ds => {
            const opt = document.createElement('option');
            opt.value = ds;
            opt.innerText = ds;
            select.appendChild(opt);
        });

        // Trigger loading for first dataset
        loadAnalyticsData();
    } catch (e) {
        console.error('Error loading evaluated datasets:', e);
    }
}

async function loadAnalyticsData() {
    const select = document.getElementById('analytics-dataset-select');
    const dsName = select.value;
    if (!dsName) return;

    try {
        const res = await fetch(`/api/analytics/${encodeURIComponent(dsName)}`);
        const data = await res.json();

        if (!data.success) {
            alert(data.error);
            return;
        }

        state.currentAnalytics = data;

        // Update KPI Pills
        document.getElementById('kpi-algos').innerText = data.n_algorithms;
        const bestQ = data.rankings.reduce((prev, curr) => curr.quality_score > prev.quality_score ? curr : prev, data.rankings[0]);
        const bestD = data.rankings.reduce((prev, curr) => curr.diversity_score > prev.diversity_score ? curr : prev, data.rankings[0]);

        document.getElementById('kpi-best-q').innerText = bestQ ? bestQ.algo_short : '-';
        document.getElementById('kpi-best-d').innerText = bestD ? bestD.algo_short : '-';

        // 1. Render Quality vs Diversity Scatter
        renderTradeoffScatter('plot-tradeoff', data.scatter);

        // 2. Render Polar Radar Profile
        renderRadarProfile('plot-radar', data.radar);

        // 3. Render Diversity Bar Chart
        renderDiversityBarChart();

        // 4. Render Full Heatmap
        renderMetricsHeatmap('plot-heatmap', data.heatmap);

        // 5. Populate Multi-Criteria Rankings Table
        populateRankingsTable(data.rankings);

    } catch (e) {
        console.error('Error loading analytics:', e);
    }
}

function renderDiversityBarChart() {
    if (!state.currentAnalytics || !state.currentAnalytics.diversity_bars) return;
    const picker = document.getElementById('diversity-metric-picker');
    const metric = picker.value;
    renderDiversityBar('plot-diversity-bars', state.currentAnalytics.diversity_bars, metric);
}

// --------------------------------------------------------------------------
// Multi-Criteria Rankings Interactive Datatable
// --------------------------------------------------------------------------
function populateRankingsTable(rankings) {
    state.rankingsData = rankings || [];
    state.rankingsFilter = '';
    state.rankingsSortKey = 'rank';
    state.rankingsSortAsc = true;
    const searchInput = document.getElementById('rankings-search');
    if (searchInput) searchInput.value = '';
    renderRankingsTable();
}

function filterRankingsTable(query) {
    state.rankingsFilter = (query || '').toLowerCase().trim();
    renderRankingsTable();
}

function sortRankingsTable(columnKey) {
    if (state.rankingsSortKey === columnKey) {
        state.rankingsSortAsc = !state.rankingsSortAsc;
    } else {
        state.rankingsSortKey = columnKey;
        state.rankingsSortAsc = true;
    }
    renderRankingsTable();
}

function renderRankingsTable() {
    const tbody = document.getElementById('rankings-table-body') || document.querySelector('#rankings-table tbody');
    const countBadge = document.getElementById('rankings-count');
    if (!tbody) return;

    let rows = [...state.rankingsData];

    // Filter across algorithm name, short tag, and rank
    if (state.rankingsFilter) {
        rows = rows.filter(r => 
            (r.algo_short || '').toLowerCase().includes(state.rankingsFilter) ||
            (r.full_name || '').toLowerCase().includes(state.rankingsFilter) ||
            String(r.rank).includes(state.rankingsFilter)
        );
    }

    // Sort
    const key = state.rankingsSortKey;
    const asc = state.rankingsSortAsc;
    rows.sort((a, b) => {
        let valA = a[key];
        let valB = b[key];
        if (typeof valA === 'string') {
            return asc ? valA.localeCompare(valB) : valB.localeCompare(valA);
        }
        valA = Number(valA) || 0;
        valB = Number(valB) || 0;
        return asc ? (valA - valB) : (valB - valA);
    });

    // Update Header Sort Icons
    document.querySelectorAll('#rankings-table th.sortable').forEach(th => {
        th.classList.remove('asc', 'desc');
        const onclickAttr = th.getAttribute('onclick') || '';
        if (onclickAttr.includes(`'${key}'`)) {
            th.classList.add(asc ? 'asc' : 'desc');
        }
    });

    // Update Count Badge
    if (countBadge) {
        countBadge.innerText = `${rows.length} of ${state.rankingsData.length} algorithms`;
    }

    // Render Rows
    if (rows.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No algorithm matches the current filter.</td></tr>';
        return;
    }

    tbody.innerHTML = '';
    rows.forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>#${r.rank}</strong></td>
            <td><code>${r.algo_short}</code></td>
            <td><span class="badge ${r.quality_score >= 0.5 ? 'badge-idle' : 'badge-device'}">${r.quality_score}</span></td>
            <td><span class="badge ${r.diversity_score >= 0.5 ? 'badge-idle' : 'badge-device'}">${r.diversity_score}</span></td>
            <td>${r.avg_rank}</td>
            <td>${r.n_generated}</td>
        `;
        tbody.appendChild(tr);
    });
}

function exportRankingsCSV() {
    if (!state.rankingsData || state.rankingsData.length === 0) return;
    let csvContent = 'data:text/csv;charset=utf-8,Rank,Algorithm,Full_Name,Quality_Score,Diversity_Score,Avg_Rank,N_Generated\n';
    state.rankingsData.forEach(r => {
        csvContent += `${r.rank},"${r.algo_short}","${r.full_name}",${r.quality_score},${r.diversity_score},${r.avg_rank},${r.n_generated}\n`;
    });
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `rankings_${state.currentAnalytics ? state.currentAnalytics.dataset : 'export'}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// --------------------------------------------------------------------------
// Tab 4: Runtime Results Interactive Datatable
// --------------------------------------------------------------------------
async function loadRuntimeResults() {
    const tbody = document.getElementById('runtime-table-body');
    tbody.innerHTML = '<tr><td colspan="11" class="text-center"><i class="fa-solid fa-spinner fa-spin"></i> Loading runtime results...</td></tr>';

    try {
        const res = await fetch('/api/runtime_results');
        const data = await res.json();

        if (!data.records || data.records.length === 0) {
            tbody.innerHTML = '<tr><td colspan="11" class="text-center">No benchmark recorded yet.</td></tr>';
            const countBadge = document.getElementById('runtime-count');
            if (countBadge) countBadge.innerText = '0 records';
            state.runtimeData = [];
            return;
        }

        state.runtimeData = data.records;
        state.runtimeFilter = '';
        state.runtimeSortKey = 'timestamp';
        state.runtimeSortAsc = false;
        const searchInput = document.getElementById('runtime-search');
        if (searchInput) searchInput.value = '';
        renderRuntimeTable();

    } catch (e) {
        tbody.innerHTML = `<tr><td colspan="11" class="text-center text-danger">Error: ${e}</td></tr>`;
    }
}

function filterRuntimeTable(query) {
    state.runtimeFilter = (query || '').toLowerCase().trim();
    renderRuntimeTable();
}

function sortRuntimeTable(columnKey) {
    if (state.runtimeSortKey === columnKey) {
        state.runtimeSortAsc = !state.runtimeSortAsc;
    } else {
        state.runtimeSortKey = columnKey;
        state.runtimeSortAsc = false; // Descending by default for metrics
    }
    renderRuntimeTable();
}

function renderRuntimeTable() {
    const tbody = document.getElementById('runtime-table-body');
    const countBadge = document.getElementById('runtime-count');
    if (!tbody) return;

    let rows = [...state.runtimeData];

    // Filter across multiple fields
    if (state.runtimeFilter) {
        rows = rows.filter(r => 
            (r.algorithm || '').toLowerCase().includes(state.runtimeFilter) ||
            (r.dataset || '').toLowerCase().includes(state.runtimeFilter) ||
            (r.status || '').toLowerCase().includes(state.runtimeFilter) ||
            (r.timestamp || '').toLowerCase().includes(state.runtimeFilter) ||
            (r.method_used || '').toLowerCase().includes(state.runtimeFilter) ||
            (r.error_message || '').toLowerCase().includes(state.runtimeFilter)
        );
    }

    // Sort
    const key = state.runtimeSortKey;
    const asc = state.runtimeSortAsc;
    rows.sort((a, b) => {
        let valA = a[key];
        let valB = b[key];
        if (['total_time_s', 'train_time_s', 'gen_time_s', 'total_mem_peak_mb', 'n_orig', 'n_augmented'].includes(key)) {
            valA = Number(valA) || 0;
            valB = Number(valB) || 0;
            return asc ? (valA - valB) : (valB - valA);
        }
        valA = String(valA || '');
        valB = String(valB || '');
        return asc ? valA.localeCompare(valB) : valB.localeCompare(valA);
    });

    // Update Header Sort Icons
    document.querySelectorAll('#runtime-table th.sortable').forEach(th => {
        th.classList.remove('asc', 'desc');
        const onclickAttr = th.getAttribute('onclick') || '';
        if (onclickAttr.includes(`'${key}'`)) {
            th.classList.add(asc ? 'asc' : 'desc');
        }
    });

    // Update Count Badge
    if (countBadge) {
        countBadge.innerText = `${rows.length} of ${state.runtimeData.length} benchmarks`;
    }

    if (rows.length === 0) {
        tbody.innerHTML = '<tr><td colspan="11" class="text-center text-muted">No benchmark matches the current filter.</td></tr>';
        return;
    }

    tbody.innerHTML = '';
    rows.forEach(r => {
        const tr = document.createElement('tr');
        const pillClass = r.status === 'SUCCESS' ? 'success' : (r.status === 'FALLBACK' ? 'warning' : 'failed');
        const noteText = [r.method_used ? ('method: ' + r.method_used) : '', r.error_message || ''].filter(Boolean).join(' | ');
        const fmt = (v, d) => (v != null && !isNaN(Number(v)) && v !== '') ? Number(v).toFixed(d) : '-';
        tr.innerHTML = `
            <td>${r.timestamp || '-'}</td>
            <td><strong>${r.algorithm}</strong></td>
            <td>${r.dataset}</td>
            <td><span class="status-pill ${pillClass}">${r.status}</span></td>
            <td>${fmt(r.total_time_s, 2)}</td>
            <td>${fmt(r.train_time_s, 2)}</td>
            <td>${fmt(r.gen_time_s, 2)}</td>
            <td>${fmt(r.total_mem_peak_mb, 1)}</td>
            <td>${r.n_orig || '0'}</td>
            <td>${r.n_augmented || '0'}</td>
            <td style="max-width:250px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${noteText.replace(/"/g, '&quot;')}">
                ${noteText || '-'}
            </td>
        `;
        tbody.appendChild(tr);
    });
}

function exportRuntimeCSV() {
    if (!state.runtimeData || state.runtimeData.length === 0) return;
    const cols = ['timestamp', 'algorithm', 'dataset', 'status', 'total_time_s', 'train_time_s', 'gen_time_s', 'total_mem_peak_mb', 'n_orig', 'n_augmented', 'error_message'];
    let csvContent = 'data:text/csv;charset=utf-8,' + cols.join(',') + '\n';
    state.runtimeData.forEach(r => {
        const line = cols.map(c => `"${String(r[c] || '').replace(/"/g, '""')}"`).join(',');
        csvContent += line + '\n';
    });
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', 'runtime_benchmark_export.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// --------------------------------------------------------------------------
// Tab 5: Checkpoints & Generation
// --------------------------------------------------------------------------
async function loadCheckpoints() {
    const list = document.getElementById('checkpoints-list');
    const countBadge = document.getElementById('checkpoints-count');
    if (!list) return;

    list.innerHTML = '<div class="loading-state"><i class="fa-solid fa-spinner fa-spin"></i> Scanning checkpoints...</div>';

    try {
        const res = await fetch('/api/checkpoints');
        state.checkpoints = await res.json();
        state.checkpointsFilter = '';
        renderCheckpointsList();
    } catch (e) {
        list.innerHTML = `<div class="text-center text-danger p-3">Error loading checkpoints: ${e}</div>`;
    }
}

function filterCheckpoints(query) {
    state.checkpointsFilter = (query || '').toLowerCase().trim();
    renderCheckpointsList();
}

function renderCheckpointsList() {
    const list = document.getElementById('checkpoints-list');
    const countBadge = document.getElementById('checkpoints-count');
    if (!list) return;

    let items = state.checkpoints || [];
    if (state.checkpointsFilter) {
        items = items.filter(c => 
            (c.filename || '').toLowerCase().includes(state.checkpointsFilter) ||
            (c.algorithm || '').toLowerCase().includes(state.checkpointsFilter) ||
            (c.dataset_hint || '').toLowerCase().includes(state.checkpointsFilter) ||
            (c.rel_path || '').toLowerCase().includes(state.checkpointsFilter)
        );
    }

    if (countBadge) {
        countBadge.innerText = `${items.length} of ${state.checkpoints.length} checkpoints`;
    }

    if (items.length === 0) {
        list.innerHTML = '<div class="text-center text-muted p-3">No checkpoint matches the current filters.</div>';
        return;
    }

    list.innerHTML = '';
    items.forEach(c => {
        const isSel = state.selectedCheckpoint && state.selectedCheckpoint.full_path === c.full_path;
        const el = document.createElement('div');
        el.className = `ckpt-item ${isSel ? 'selected' : ''}`;
        el.onclick = () => selectCheckpoint(c);
        el.innerHTML = `
            <div style="flex:1">
                <div class="ckpt-title">${c.filename}</div>
                <div class="ckpt-meta">
                    <span class="badge badge-info" style="font-size:10px; padding:2px 8px;">${c.algorithm.toUpperCase()}</span>
                    ${c.dataset_hint ? `<span class="badge badge-device" style="font-size:10px; padding:2px 8px;">${c.dataset_hint}</span>` : ''}
                    <span>${c.size_kb} KB</span> • <span>${c.mtime}</span>
                </div>
                <div class="ckpt-meta" style="font-size:10px; opacity:0.7; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                    ${c.rel_path}
                </div>
            </div>
            <i class="fa-solid fa-chevron-right text-muted"></i>
        `;
        list.appendChild(el);
    });
}

function selectCheckpoint(ckpt) {
    state.selectedCheckpoint = ckpt;
    renderCheckpointsList();

    const box = document.getElementById('selected-ckpt-box');
    const manualInput = document.getElementById('manual-ckpt-path');
    const algoSelect = document.getElementById('ckpt-algo-select');
    const datasetSelect = document.getElementById('ckpt-dataset-select');
    const outputNameInput = document.getElementById('ckpt-output-name');

    if (box) {
        box.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <strong style="font-size:14px; color:var(--text-primary);">${ckpt.filename}</strong>
                    <div style="margin-top:6px; display:flex; gap:6px; flex-wrap:wrap;">
                        <span class="badge badge-info">${ckpt.algorithm.toUpperCase()}</span>
                        ${ckpt.dataset_hint ? `<span class="badge badge-device">${ckpt.dataset_hint}</span>` : ''}
                        <span class="badge badge-idle">${ckpt.size_kb} KB</span>
                    </div>
                    <div class="help-text mt-2" style="word-break:break-all;"><strong>Path:</strong> ${ckpt.rel_path}</div>
                </div>
                <button class="btn btn-outline btn-xs" onclick="clearSelectedCheckpoint()" title="Deselect"><i class="fa-solid fa-xmark"></i></button>
            </div>
        `;
    }

    if (manualInput) manualInput.value = ckpt.full_path;
    if (algoSelect && ckpt.algorithm) {
        algoSelect.value = ckpt.algorithm;
    }

    // Auto-match dataset if hint is present
    if (datasetSelect && ckpt.dataset_hint) {
        for (let i = 0; i < datasetSelect.options.length; i++) {
            const opt = datasetSelect.options[i];
            if (opt.value && opt.value.toLowerCase().includes(ckpt.dataset_hint.toLowerCase())) {
                datasetSelect.selectedIndex = i;
                break;
            }
        }
    }

    if (outputNameInput) {
        const dsName = datasetSelect ? datasetSelect.value || 'data' : 'data';
        outputNameInput.value = `${dsName}_from_${ckpt.algorithm}.csv`;
    }
}

function clearSelectedCheckpoint() {
    state.selectedCheckpoint = null;
    const box = document.getElementById('selected-ckpt-box');
    if (box) {
        box.innerHTML = '<div class="ckpt-empty-msg"><i class="fa-solid fa-circle-info"></i> No checkpoint selected. Choose one from the list on the left or upload a new one.</div>';
    }
    const manualInput = document.getElementById('manual-ckpt-path');
    if (manualInput) manualInput.value = '';
    renderCheckpointsList();
}

function handleManualCkptInput(path) {
    const p = (path || '').trim();
    if (p.length > 0) {
        state.selectedCheckpoint = {
            filename: p.split('/').pop() || 'custom_model',
            full_path: p,
            rel_path: p,
            algorithm: 'auto',
            dataset_hint: '',
            size_kb: 0,
            mtime: 'Manual'
        };
    } else {
        state.selectedCheckpoint = null;
    }
}

async function handleCheckpointUpload(files) {
    if (!files || files.length === 0) return;
    const file = files[0];
    const formData = new FormData();
    formData.append('file', file);

    const dropzone = document.getElementById('checkpoint-dropzone');
    const origHtml = dropzone.innerHTML;
    dropzone.innerHTML = `<i class="fa-solid fa-spinner fa-spin upload-icon"></i><div class="upload-text">Uploading ${file.name}...</div>`;

    try {
        const res = await fetch('/api/upload_checkpoint', {
            method: 'POST',
            body: formData,
        });
        const data = await res.json();
        dropzone.innerHTML = origHtml;

        if (data.success && data.checkpoint) {
            state.checkpoints.unshift(data.checkpoint);
            selectCheckpoint(data.checkpoint);
            alert(`✅ Checkpoint "${file.name}" uploaded successfully to checkpoint/`);
        } else {
            alert('Upload error: ' + (data.error || 'Unknown error'));
        }
    } catch (e) {
        dropzone.innerHTML = origHtml;
        alert('Upload error: ' + e);
    }
}

function populateCkptDatasetsDropdown() {
    const sel = document.getElementById('ckpt-dataset-select');
    if (!sel) return;
    if (state.datasets && state.datasets.length > 0) {
        sel.innerHTML = '<option value="">-- Select a dataset from datasets/ --</option>';
        state.datasets.forEach(ds => {
            const opt = document.createElement('option');
            opt.value = ds.stem;
            opt.innerText = `${ds.stem} (${ds.rows} rows)`;
            sel.appendChild(opt);
        });
    }
}

function setCkptDevice(dev) {
    state.ckptDevice = dev;
    document.querySelectorAll('#ckpt-device-group .btn-toggle').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-device') === dev);
    });
}

async function startCheckpointGeneration() {
    const ckptPath = (document.getElementById('manual-ckpt-path')?.value || state.selectedCheckpoint?.full_path || '').trim();
    if (!ckptPath) {
        alert('Select a checkpoint or specify a valid path.');
        return;
    }

    const dataset = document.getElementById('ckpt-dataset-select')?.value;
    if (!dataset) {
        alert('Select a reference dataset from datasets/');
        return;
    }

    const algo = document.getElementById('ckpt-algo-select')?.value || 'auto';
    const nSamples = parseInt(document.getElementById('ckpt-n-samples')?.value) || 1000;
    const outputName = document.getElementById('ckpt-output-name')?.value.trim();

    const payload = {
        checkpoint_path: ckptPath,
        dataset: dataset,
        algorithm: algo,
        n_samples: nSamples,
        device: state.ckptDevice,
        output_filename: outputName || null,
    };

    try {
        const res = await fetch('/api/generate_from_checkpoint', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await res.json();

        if (data.success) {
            state.activeTaskId = data.task_id;
            clearConsole();
            openConsoleDrawer();
            startLogPolling(data.task_id);
            document.getElementById('stop-task-btn').disabled = false;
        } else {
            alert('Error starting the generation: ' + data.error);
        }
    } catch (e) {
        alert('Connection error: ' + e);
    }
}
