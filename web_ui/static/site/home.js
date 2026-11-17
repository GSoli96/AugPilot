/* home.js - riga di stato della pagina iniziale (solo lettura di /api/modalities e /api/assistant/status). */
(function () {
  'use strict';
  var el = document.getElementById('home-status');
  function get(u) { return fetch(u).then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); }); }
  Promise.all([
    get('/api/modalities').catch(function () { return null; }),
    get('/api/assistant/status').catch(function () { return null; })
  ]).then(function (r) {
    var parts = [];
    if (Array.isArray(r[0])) {
      r[0].forEach(function (m) {
        var c = m.counts || {};
        parts.push((m.name || m.id) + ': ' + (c.executable || 0) + ' executable algorithms out of ' + (c.total || 0) + ', ' + (m.n_datasets || 0) + ' datasets');
      });
    } else parts.push('Modalities unreadable');
    parts.push(r[1] ? (r[1].mode === 'online' ? 'assistant online' : 'assistant offline (' + (r[1].reason || 'no model') + ')') : 'assistant unreachable');
    el.textContent = parts.join(' | ');
  });
})();
