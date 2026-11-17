# Assistente agentico di data augmentation

Pacchetto `assistant/` + blueprint `web_ui/assistant_bp.py` + pagina `web_ui/templates/assistant.html`.
Claude (SDK `anthropic`) guida l'utente con un ciclo `tool_use`/`tool_result`; i consigli vengono dal consigliere
deterministico e dai CSV misurati, mai dalla memoria del modello.

## Chiave API
Ordine di lettura: variabile `ANTHROPIC_API_KEY`, poi il file indicato da `CLAUDE_KEY_FILE` (default `~/Desktop/prog/ChiaveClaude.json`)
(`{"api_key": "..."}`). Non si cercano altre chiavi. La chiave non e' mai stampata ne' restituita (lo stato mostra solo
l'origine: env/file). Modello: variabile `ASSISTANT_MODEL`, default `claude-sonnet-5-5`.
Senza chiave (o senza SDK) l'assistente e' in **modalita' offline**: lo dichiara in UI e in `/api/assistant/status`, non
simula risposte; restano il pannello "Strumenti diretti" (consigliere, dataset, confronto) e le chiamate dirette agli strumenti.

## Avvio
- Prova isolata (porta 5057): `python tests/test_assistant_server.py` (aggiungi `--mock` per un client simulato, senza API),
  poi http://127.0.0.1:5057/assistant.
- Nell'app principale: `app.register_blueprint(web_ui.assistant_bp.bp)` (riusa `task_manager` di `web_ui/app.py` se presente).
- Test: `python -m pytest tests/test_assistant.py -q` (nessuna chiamata all'API).

## Strumenti
`list_algorithms`, `describe_algorithm`, `list_datasets`, `analyze_dataset`, `recommend_algorithms`, `run_augmentation`,
`get_task_status`, `evaluate`, `compare_results`, `explain_metrics`.
- `run_augmentation` e' asincrona (task_manager, restituisce `task_id`). Con piu' di 1 algoritmo, piu' di 1 dataset o piu' di 5
  epoche NON parte: restituisce `needs_confirmation` e la UI mostra "Conferma e lancia". Solo `/api/assistant/confirm`
  (azione umana) esegue; il modello non puo' confermare da solo. Rifiuta algoritmi `described` e nomi fuori catalogo.
- `explain_metrics` rilegge i valori dai CSV e, se si passano `claimed_values`, segnala le discrepanze.
- Controllo a posteriori: i numeri della risposta finale che non compaiono nei risultati degli strumenti o nei messaggi
  dell'utente generano un avviso in UI (controllo euristico, non una prova).
- Lettura file: solo nomi di dataset o percorsi sotto `datasets*/`, `augmented*/`, `runtime_results*/`, `evaluation_metrics/`.
- Limiti del ciclo: 10 passi, 120 s, 80.000 token (`assistant/config.py: LIMITS`). Errori API: eventi `error` espliciti.

## Consigliere (`recommender.py`)
Regole R1-R7 nel docstring e nel campo `rules` di ogni risposta. In breve: classifica solo chi ha misure; il tabulare usa
rango medio su fidelity (JS, KS, WD, pMSE, MMD) e diversity (Precision, Recall, Coverage); ts/graph per tasso di successo e
tempo mediano; i `described` non sono mai classificati; filtri duri per fairness/privacy/vincoli/budget di tempo.

## Limiti noti
- Dati misurati tabulari: pochi dataset (iris0, abalone9-18, cleveland-0_vs_4, migraine_0), tempi dal Mac del collega,
  nessun dato GPU (tutti 0): il fabbisogno di GPU dei tabulari e' "ignoto". ts/graph: nessun CSV di risultati finche' non si esegue.
- I tag fairness/privacy/vincoli sono derivati da descrizioni e titoli (euristica); `imbalance_ratio`/`n_cols` non pilotano regole.
- I dataset `*_min` del tabulare contengono la sola classe minoritaria: lo sbilanciamento non si ricava dal file.
- ts/graph: la valutazione richiede `modalities/<mod>/metrics.py`; per ts/graph `evaluate` e' un algoritmo per volta.
- Sessioni in memoria (max 50, perse al riavvio). Un solo task alla volta (limite del `TaskManager`).
- Mai provato con l'API reale (nessuna chiave a disposizione durante lo sviluppo).
