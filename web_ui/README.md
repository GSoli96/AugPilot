# 🎛️ Data Augmentation & Metric Evaluation Web UI

Dashboard web moderna basata su **Flask**, **Vanilla JavaScript** e **Plotly.js** per orchestrare visivamente l'esecuzione degli algoritmi di Data Augmentation (`main.py`), la valutazione delle metriche multidimensionali (`evaluate_metrics.py`), e l'esplorazione dei grafici analitici (`plot/plot_results.py`).

Tutti gli script CLI sottostanti rimangono **100% indipendenti** ed eseguibili da terminale.

---

## 🚀 Avvio Rapido

Dalla cartella principale del benchmark:

```bash
cd <cartella del progetto>
python web_ui/app.py
```

La dashboard sarà disponibile all'indirizzo:
👉 **[http://127.0.0.1:5005](http://127.0.0.1:5005)**

*(Porta: variabile d'ambiente `PORT`; indirizzo: `HOST`, default `127.0.0.1`, `HOST=0.0.0.0` per aprire alla rete locale.)*

---

## 🌟 Funzionalità Principali

### 1. 🧪 Augmentation Runner (`main.py`)
- **Selezione Dataset**: visualizzazione interattiva dei dataset disponibili (`datasets/*.csv`) con conteggio righe, colonne e dimensione.
- **Griglia Algoritmi**: selezione rapida multi-categoria:
  - *Traditional & Fair* (SMOTE, Fair-SMOTE, SMOTE-CDNN, Prefair)
  - *GAN & VAE* (CasTGAN, TabFairGAN, TVAE, CTAB-GAN, CTAB-GAN+)
  - *Diffusion & Advanced* (Tab-DDPM, TabSyn, TabDiff, OCTree, HARMONIC, Synthetic-LLM, CUTS, Tabular-ARGN)
  - *Constraint-Aware* (Kamino, SyRFDRes, SyDC)
- **Configurazione Denial Constraints per Kamino / SyDC**:
  - Auto-discovery dei file di vincoli (`.out`, `.ic`, `.txt`) nei percorsi configurati.
  - Possibilità di inserire un percorso arbitrario di file DC o cartella DC (`--dc-dir`).
  - Selezione strategia pesi Kamino (`toggled`, `hard`, `raw`, `dp`, `uniform` oppure `all` per sweep completo).
- **Controllo Iperparametri**: Campioni sintetici (`n_samples`), epoche (`epochs`), selezione device (`Auto`, `CUDA`, `MPS`, `CPU`), flag Resume/Overwrite.
- **Esecuzione Asincrona e Streaming Terminale**:
  - Esecuzione non bloccante in background con process-group isolation.
  - Streaming in tempo reale di stdout/stderr nella console a scomparsa.
  - Pulsante **Termina Esecuzione** (termina tutto l'albero di processi: `killpg` su Linux/macOS, `taskkill /T /F` su Windows).

### 2. 📊 Metrics Evaluation (`evaluate_metrics.py`)
- Selezione del dataset da valutare (supporta anche l'opzione `all`).
- Selezione rapida dei modelli/algoritmi da includere nella valutazione (inclusi i dataset generati da SyDC con sweep `coverage`/`interestingness`).
- Configurazione granularità bin istogrammi (`bins`, default: 30).
- Monitoraggio streaming avanzamento calcolo metriche (Fidelity, Diversity, Distance to Closest Record, Privacy, Runtime).

### 3. 📈 Visual Analytics & Interactive Plots
- **Grafico di Trade-off Quality vs. Diversity**: Scatter plot bidimensionale interattivo (Quality Score vs Diversity Score con raggio Pareto ottimale).
- **Profilo Radar Multi-Criterio**: Confronto di tutti i modelli su 5 assi normalizzati (`Fidelity`, `Diversity`, `Privacy / DCR`, `Coverage`, `Efficiency`).
- **Diversity Bar Chart**: Visualizzazione comparativa ordinata della diversità sintetica rispetto al dataset reale.
- **Matrice Heatmap delle Metriche**: Heatmap normalizzata (0-1, con inversione automatica per le metriche in cui "lower is better") per identificare istantaneamente punti di forza e debolezza di ciascun algoritmo.
- **Tabella Leaderboard Completa**: Ranking aggregato con export interattivo.

### 4. ⏱️ Runtime Benchmark History
- Ispezione dell'ultimo file CSV generato in `runtime_results/` con tempi di addestramento, campionamento, e stato di ogni algoritmo.

### 5. Modalità serie temporali e grafi
Il selettore **Modalità** (Tabulare / Serie temporali / Grafi) è in cima alla pagina. Per ts e grafi compare una pagina con dataset (`datasets_ts/`, `datasets_graph/`), algoritmi con badge di stato (`executable`, `wired`, `described`; i `described` non sono avviabili), parametri, avvio e risultati (`runtime_results_ts/`, `runtime_results_graph/`). Con più di un algoritmo o dataset il client chiede conferma e il server la ricontrolla. Il tabulare resta com'era. L'avvio usa `run_modality.py` come task asincrono (console live e stop come per `main.py`). I cataloghi si leggono all'avvio dell'app: dopo l'aggiunta di algoritmi serve riavviarla.

Nella barra in alto ci sono i link **Assistente** (`/assistant`) e **Macchine** (`/machines`): i blueprint `web_ui/assistant_bp.py` e `web_ui/machines_bp.py` si registrano se importabili; altrimenti il link è disattivato e l'app parte lo stesso.

### 6. Stato reale delle esecuzioni tabulari
I record di `main.py` (e la tabella Runtime Metrics) hanno `status` = `SUCCESS` (metodo richiesto), `FALLBACK` (risultato prodotto da un metodo diverso o semplificato, vedi `method_used` ed `error_message`) o `FAILED`. Il ripiego automatico su un metodo più semplice, quando il metodo reale fallisce, si attiva solo con `main.py --allow-fallback`.

## 🔌 Rotte aggiunte

| Rotta | Metodo | Descrizione |
|---|---|---|
| `/api/modalities` | GET | Per ogni modalità: conteggi degli algoritmi per stato (`executable`/`wired`/`described`/`total`), numero di dataset, `catalog_error` |
| `/api/modality/<mod>/algorithms` | GET | Algoritmi della modalità (`tabular`, `ts`, `graph`) con `key`, `name`, `strategy`, `status`, `runnable`, `notes`, ... |
| `/api/modality/<mod>/datasets` | GET | Dataset disponibili (CSV in `datasets/`, `.npz` in `datasets_ts/`, `.pt` in `datasets_graph/`) |
| `/api/run_modality` | POST | Lancia `run_modality.py`. Corpo: `modality`, `algorithms[]`, `datasets[]`, `epochs`, `n_samples?`, `device` (`auto`/`cpu`/`cuda`), `seed`, `no_eval`, `confirmed`. Risponde 409 con `needs_confirmation` se gli algoritmi sono più di uno e `confirmed` manca; 400 per algoritmi `described`, nomi non validi o dataset assenti. Per `tabular` usare `/api/run_augmentation` |
| `/api/modality/<mod>/results` | GET | Ultimo CSV dei risultati; `?file=<nome.csv>` per un altro file, `?all=1` per tutti con colonna `file`; `files` elenca i CSV disponibili |
| `/assistant`, `/machines` | GET | Pagine dei blueprint opzionali (registrati se importabili) |

Rotte già presenti: `/api/status`, `/api/datasets`, `/api/algorithms`, `/api/dc_files`, `/api/checkpoints`, `/api/upload_checkpoint`, `/api/generate_from_checkpoint`, `/api/run_augmentation`, `/api/run_evaluation`, `/api/stop_task/<id>`, `/api/logs/<id>`, `/api/evaluated_datasets`, `/api/analytics/<dataset>`, `/api/runtime_results`.

---

## 🏗️ Architettura

```
web_ui/
├── app.py                # Server Flask con endpoint REST per processi, file e analisi
├── task_manager.py       # Gestore asincrono dei task con streaming dei log e stop sicuro
├── analytics_engine.py   # Motore di calcolo metriche e formattazione payload Plotly
├── requirements.txt      # Dipendenze Python (Flask, Flask-CORS, pandas, numpy)
├── templates/
│   └── index.html        # Single-page dashboard con design dark moderno
└── static/
    ├── css/
    │   └── style.css     # Stile moderno dark mode con glassmorphism
    └── js/
        ├── app.js        # Controller UI, polling asincrono e gestione form
        ├── modalities.js # Selettore di modalità e pagina serie temporali / grafi
        └── charts.js     # Render grafici interattivi Plotly.js
```
