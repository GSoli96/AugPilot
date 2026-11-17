/* assistant_chat.js
 * Logica condivisa della chat dell'assistente (home "/" e pagina "/assistant").
 * Nessuna variabile globale tranne window.AssistantChat.
 * Usa solo gli endpoint di web_ui/assistant_bp.py:
 *   GET  /api/assistant/status
 *   POST /api/assistant/chat            (SSE; con ?stream=0 risposta JSON)
 *   POST /api/assistant/confirm
 *   POST /api/assistant/reset
 *
 * AssistantChat.create(opts) -> controller
 *   opts.log        elemento (o id) del registro messaggi          [obbligatorio]
 *   opts.input      textarea (o id)                                 [obbligatorio]
 *   opts.sendBtn    pulsante di invio (o id)
 *   opts.steps      contenitore dei passi; se assente i passi vanno nel registro
 *   opts.onStatus   funzione(st | null) chiamata dopo la lettura dello stato
 *   opts.onResults  funzione(modality, info) chiamata al click su "Vedi risultati"
 *   opts.welcome    se true mostra il testo di benvenuto con 3 suggerimenti
 *   opts.sysText    testo introduttivo semplice (se non c'e' welcome)
 */
(function () {
    'use strict';

    const SUGGESTIONS = [
        'I have an imbalanced graph dataset: what do you recommend?',
        'Which time series can I augment, and with which methods?',
        'Compare the available tabular algorithms',
    ];
    const WELCOME = 'I am the tool\'s assistant. I can help you understand your problem, choose the dataset, ' +
        'recommend algorithms, launch runs (with your confirmation), compare the results and explain them. ' +
        'I only use tools and measured results: if a piece of data is missing, I say so.';

    function el(x) { return typeof x === 'string' ? document.getElementById(x) : x; }

    function create(opts) {
        const log = el(opts.log), inp = el(opts.input), sendBtn = el(opts.sendBtn), stepsBox = el(opts.steps);
        let sessionId = null, busy = false;
        const lastCalls = {};   // id chiamata -> {name, input}

        function scroll() { log.scrollTop = log.scrollHeight; }

        function addMsg(cls, text) {
            const d = document.createElement('div');
            d.className = 'msg ' + cls;
            // le risposte del modello sono in markdown: reso con window.Md (site/shared.js, marked + DOMPurify);
            // senza le librerie (offline) resta testo semplice
            if (cls === 'bot' && window.Md) window.Md.render(d, text); else d.textContent = text;
            log.appendChild(d); scroll();
            return d;
        }

        function addWelcome() {
            addMsg('sys', WELCOME);
            const box = document.createElement('div');
            box.className = 'ac-suggestions';
            SUGGESTIONS.forEach(t => {
                const b = document.createElement('button');
                b.type = 'button'; b.className = 'ac-suggestion'; b.textContent = t;
                b.onclick = () => { inp.value = t; send(); };
                box.appendChild(b);
            });
            log.appendChild(box); scroll();
        }

        function intro() {
            if (opts.welcome) addWelcome();
            else if (opts.sysText) addMsg('sys', opts.sysText);
        }

        function addStep(ev) {
            lastCalls[ev.id] = { name: ev.name, input: ev.input };
            const det = document.createElement('details');
            det.className = 'step'; det.dataset.id = ev.id;
            const s = document.createElement('summary');
            s.textContent = '> ' + ev.name + '(' + JSON.stringify(ev.input).slice(0, 90) + ')';
            const pre = document.createElement('pre');
            pre.textContent = JSON.stringify(ev.input, null, 2);
            det.appendChild(s); det.appendChild(pre);
            if (stepsBox) {
                stepsBox.appendChild(det);
                if (stepsBox.firstChild && stepsBox.firstChild.nodeType === 3) stepsBox.firstChild.remove();
            } else {
                log.appendChild(det); scroll();
            }
            return det;
        }

        function addResult(ev) {
            const root = stepsBox || log;
            const det = [...root.querySelectorAll('details.step')].reverse()
                .find(d => d.dataset.id === ev.id && !d.dataset.done);
            if (det) {
                det.dataset.done = 1; det.classList.add(ev.ok ? 'ok' : 'fail');
                det.querySelector('summary').textContent += ev.ok ? '  [ok]' : '  [error]';
                const pre = document.createElement('pre');
                pre.textContent = JSON.stringify(ev.result, null, 2);
                det.appendChild(pre);
            }
            // Azione che avvia un'esecuzione: avviso "Vedi risultati"
            const call = lastCalls[ev.id] || {};
            const r = ev.result || {};
            if (ev.ok && r.status === 'started' && r.task_id) {
                resultsHint((call.input && call.input.modality) || r.modality, r.task_id);
            }
        }

        function resultsHint(modality, taskId) {
            if (!opts.onResults || !modality) return;
            const d = document.createElement('div');
            d.className = 'msg sys ac-hint';
            const a = document.createElement('a');
            a.href = '#'; a.className = 'ac-hint-link';
            a.innerHTML = '<i class="fa-solid fa-chart-line"></i> ';
            a.appendChild(document.createTextNode('See results (' + modality + (taskId ? ', task ' + taskId : '') + ')'));
            a.onclick = e => { e.preventDefault(); opts.onResults(modality, { task_id: taskId }); };
            d.appendChild(a); log.appendChild(d); scroll();
        }

        function addPending(ev) {
            const box = document.createElement('div');
            box.className = 'msg bot confirm-box';
            const t = document.createElement('div');
            t.textContent = 'Confirmation required (' + ev.reason + '). Plan: ' + JSON.stringify(ev.plan) +
                '. The run may take many hours of computation.';
            box.appendChild(t);
            const ok = document.createElement('button'); ok.className = 'btn btn-sm btn-primary'; ok.textContent = 'Confirm and launch';
            const no = document.createElement('button'); no.className = 'btn btn-sm btn-secondary'; no.textContent = 'Cancel';
            const answer = async approve => {
                ok.disabled = no.disabled = true;
                try {
                    const r = await fetch('/api/assistant/confirm', {
                        method: 'POST', headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ session_id: sessionId, pending_id: ev.pending_id, approve }),
                    });
                    const j = await r.json();
                    addMsg(j.error ? 'err' : 'sys', approve ? (j.error ? 'Error: ' + j.error : 'Task launched: ' + j.task_id) : 'Run cancelled.');
                    if (!j.error && approve) {
                        inp.value = 'I confirmed: the task has started. Follow it and tell me its status.';
                        const mod = (ev.plan && ev.plan.modality) || (j && j.modality);
                        resultsHint(mod, j.task_id);
                    }
                } catch (e) { addMsg('err', 'Connection: ' + e); }
            };
            ok.onclick = () => answer(true); no.onclick = () => answer(false);
            box.appendChild(ok); box.appendChild(no); log.appendChild(box); scroll();
        }

        function handle(ev) {
            switch (ev.type) {
                case 'session': sessionId = ev.session_id; break;
                case 'text': addMsg('bot', ev.text); break;
                case 'offline': addMsg('warn', ev.text); break;
                case 'tool_call': addStep(ev); break;
                case 'tool_result': addResult(ev); break;
                case 'pending': addPending(ev); break;
                case 'warning': addMsg('warn', ev.text); break;
                case 'limit': addMsg('warn', 'Stopped: ' + ev.text); break;
                case 'error': addMsg('err', 'Error: ' + ev.text); break;
            }
        }

        function setBusy(b) { busy = b; if (sendBtn) sendBtn.disabled = b; }

        async function post(body) {
            setBusy(true);
            try {
                const resp = await fetch('/api/assistant/chat', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId, ...body }),
                });
                const reader = resp.body.getReader(), dec = new TextDecoder();
                let buf = '';
                for (;;) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buf += dec.decode(value, { stream: true });
                    let i;
                    while ((i = buf.indexOf('\n\n')) >= 0) {
                        const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
                        if (chunk.startsWith('data: ')) { try { handle(JSON.parse(chunk.slice(6))); } catch (e) { /* chunk non valido */ } }
                    }
                }
            } catch (e) { addMsg('err', 'Connection: ' + e); }
            setBusy(false);
        }

        function send() {
            const t = inp.value.trim();
            if (!t || busy) return;
            addMsg('user', t); inp.value = ''; post({ message: t });
        }

        // Strumento diretto, con visualizzazione dei passi (come "tool" di assistant.html)
        function runTool(name, args) { if (!busy) post({ offline_action: { tool: name, args } }); }

        // Strumento diretto senza scrivere nulla in chat; restituisce il risultato
        async function callToolSilent(name, args) {
            const r = await fetch('/api/assistant/chat?stream=0', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, offline_action: { tool: name, args } }),
            });
            const j = await r.json(); sessionId = j.session_id;
            const e = j.events.find(x => x.type === 'tool_result');
            return e ? e.result : null;
        }

        async function reset() {
            if (sessionId) {
                try {
                    await fetch('/api/assistant/reset', {
                        method: 'POST', headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ session_id: sessionId }),
                    });
                } catch (e) { /* ignora */ }
            }
            sessionId = null; log.innerHTML = '';
            if (stepsBox) stepsBox.textContent = 'No steps yet.';
            addMsg('sys', 'New session.');
            if (opts.welcome) addWelcome();
        }

        async function refreshStatus() {
            let st = null;
            try { st = await (await fetch('/api/assistant/status')).json(); } catch (e) { st = null; }
            if (opts.onStatus) opts.onStatus(st);
            return st;
        }

        inp.addEventListener('keydown', ev => {
            if (ev.key === 'Enter' && !ev.shiftKey) { ev.preventDefault(); send(); }
        });
        if (sendBtn) sendBtn.addEventListener('click', send);

        return { send, reset, runTool, callToolSilent, refreshStatus, intro, addMsg, isBusy: () => busy, getSession: () => sessionId };
    }

    window.AssistantChat = { create, SUGGESTIONS };
})();
