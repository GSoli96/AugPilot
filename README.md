# AugPilot

AugPilot turns a data-augmentation benchmark into advice. It catalogues 106 augmentation algorithms for tables, time
series, and graphs, runs them in isolated Python environments, scores every run, and stores the results in a
knowledge base. A recommender ranks algorithms for a dataset by the user's priorities (utility, fidelity, diversity,
privacy, speed), borrows evidence from similar datasets, and explains each ranking as a decomposition of its score. A
tool-using LLM agent answers through these tools and flags any number that no tool returned.

This is the companion code of a demonstration paper submitted to ICDE 2027.

| Modality | Algorithms in the catalogue | Executable (at least one completed run) | Described only |
|---|---|---|---|
| Tables | 20 | 17 | 3 (OCTree, HARMONIC, synthetic-LLM) |
| Time series | 41 | 39 | 2 |
| Graphs | 45 | 41 | 4 |

The benchmark behind the knowledge base covers 12 tabular, 6 time-series, and 8 graph datasets, every executable
algorithm, and seeds 0, 1, and 2: 1788 planned combinations, of which 1728 succeeded, 36 (all Kamino, a
reimplementation) completed as fallbacks, 19 failed, and 5 were not run (`bench/COPERTURA.md`).

## What this repository contains, and what it does not

Included: the web interface, the assistant and recommender, the evaluation code, the job runners, the raw benchmark
results (`bench/results/`, one JSON line per run), the knowledge base built from them (`assistant/kb/`), and the
experiments of the paper (`bench/experiments/`).

Not included, and why:

- **The code of the algorithms** (`Algorithms/`) belongs to other people and has its own licences. Download it with
  `scripts/fetch_algorithms.py` (see below) and read `THIRD_PARTY.md` first.
- **The datasets**, which keep the terms of their original sources. Only a 912-byte Iris fragment
  (`datasets/iris0_min.csv`) is included, as a test fixture.
- **Model weights.** Benchmark jobs do not store trained generators, only the synthetic samples they produce, which are
  not included.

## Requirements

Verified: Windows 11, Python 3.10 (the lock files in `envs_lock/` come from 3.10), `git`, PowerShell.
The benchmark ran on lab machines with NVIDIA GPUs (RTX 5090, RTX 4090, RTX 5000 Ada, RTX 3070); the generative
methods need a CUDA GPU, the classical ones do not. Linux and macOS were not tested.
For the language agent you need one of: a local [Ollama](https://ollama.com) server (the paper used `gpt-oss:120b` and
`gpt-oss:20b`), an Ollama Cloud key, or an Anthropic API key. The interface works without any of them, with the
agent switched off.

## Install

**1. Interface, recommender, knowledge base, tests (no algorithm code needed).**

```powershell
git clone https://github.com/GSoli96/AugPilot.git
cd AugPilot
python -m venv .venv
.venv\Scripts\python -m pip install flask flask-cors pandas numpy scikit-learn scipy pyyaml anthropic pytest
.venv\Scripts\python -m pytest tests -q          # 170 passed, 12 skipped without the full datasets
.venv\Scripts\python -m web_ui.app               # http://127.0.0.1:5005  (set PORT to change it)
```

The 12 skipped tests read the full benchmark datasets and historical CSV files, which are not redistributed. The
Catalog, Evidence, and Challenge pages and the recommender read the knowledge base in `assistant/kb/benchmark.csv`.

Agent backend, through environment variables:

| Variable | Meaning |
|---|---|
| `ASSISTANT_BACKEND` | `ollama` (local server), `ollama-cloud`, `claude`, or `auto` (default) |
| `OLLAMA_HOST_URL` | address of the local Ollama server, for example `http://127.0.0.1:11434` |
| `ASSISTANT_OLLAMA_MODEL` | model name, for example `gpt-oss:120b` |
| `ASSISTANT_OLLAMA_NUM_CTX` | context size (the paper used 32768) |
| `OLLAMA_KEYS_FILE`, `CLAUDE_KEY_FILE`, `ANTHROPIC_API_KEY` | where the API keys are read from; keys are never printed |

**2. Run the algorithms.** The algorithms live in five isolated environments because their dependencies conflict
(`tabular`, `argn`, `ts`, `ts_tf`, `graph`). `scripts/setup_envs.ps1` recreates them in `envs/` from the lock files with
[uv](https://docs.astral.sh/uv/) (it installs uv if missing), then download the algorithm code:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_envs.ps1            # all five, or: -Envs tabular,ts
python scripts\fetch_algorithms.py --dry-run                                 # what would be downloaded
python scripts\fetch_algorithms.py                                           # clone the 30 folders the benchmark uses
```

`fetch_algorithms.py` clones every repository at the commit used in the benchmark and applies our patches. It cannot
restore SyRFDRes (no public repository), SMOTE-CDNN, tab-ddpm/CTAB-GAN-Plus (our patches to code without a licence are
not published), the TabSyn helper `main_custom.py`, and the Kamino Denial Constraint files, so the corresponding
algorithms need manual steps or stay unavailable. The script prints this list at the end.

**3. Datasets.** Put them where the code looks: tabular minority-class files as `datasets/<name>_min.csv` and full
datasets as `datasets_full/<name>.csv`, time series as `datasets_ts/<name>.npz`, graphs as `datasets_graph/<name>.pt`.
Names, sizes, and our notes are in `catalog_info/datasets_*.json`. Datasets can also be uploaded in the interface.

## Use

```powershell
envs\tabular\Scripts\python main.py --algorithms smote --datasets iris0_min      # tables (17 runnable; "all" skips the 3 described-only)
python run_modality.py --list                                                    # time-series and graph algorithms
python run_modality.py --modality ts --algorithms jitter --datasets sine_small --epochs 5
python -m assistant.cli tools                                                    # the agent's tools, from the shell
```

Reproduce the benchmark (one job per GPU; each combination is written as soon as it finishes and skipped on restart):

```powershell
envs\tabular\Scripts\python -m bench.run_tab_job --datasets pima yeast1 --algorithms all --seeds 0 1 2 --epochs 1000 --results-file bench\results\my_tab.jsonl
envs\ts\Scripts\python -m bench.run_mod_job --modality ts --datasets sine_small cbf --algorithms all --seeds 0 1 2 --epochs 50 --results-file bench\results\my_ts.jsonl
python -m assistant.kb.build_kb        # rebuild the knowledge base from bench/results/**/*.jsonl
python -m bench.copertura              # coverage table, written to bench/COPERTURA.md
python -m bench.experiments.e1_loo     # E1 of the paper: leave-one-dataset-out recommender evaluation
```

E1 runs from the files in this repository and reproduces `bench/experiments/out/e1_loo_*.csv`. E2 and E3
(`bench/experiments/e2_e3_scenarios.py`) call an LLM server and so depend on the model you configure.

## Layout

| Folder | Content |
|---|---|
| `web_ui/` | Flask interface: Guided path, Copilot, Catalog, Evidence, Challenge, Machines |
| `assistant/` | agent loop, tools, recommender, score decomposition, knowledge base (`kb/`) |
| `modalities/` | catalogues, wrappers, and metrics for time series and graphs |
| `workers/`, `algorithms_runner.py`, `main.py` | tabular wrappers and runner |
| `evaluate_metrics.py`, `evaluate_utility.py`, `bench/privacy.py` | fidelity, diversity, utility, privacy |
| `bench/` | job runners with resume, raw results, coverage, experiments |
| `distributed/` | remote-machine registry used by the Machines page |
| `scripts/` | environment setup, algorithm download, patches |
| `catalog_info/` | dataset and algorithm descriptions |

Many comments and some documents are in Italian, the working language of the project.

## Licence

The code in this repository is released under the MIT licence (`LICENSE`). Third-party code and data keep their own
terms: see `THIRD_PARTY.md`.
