"""Catalogo unificato degli algoritmi (tabulare, ts, graph) con stato executable/wired/described.

Fonti (tutte lette dai file del progetto, nulla di inventato):
- tabulare: i 20 id di `config.CORE_ALGORITHMS` (letti via AST); nome/categoria/descrizione da
  `web_ui/app.py: ALGORITHM_CATALOG` (AST), paper e repository da `Algorithms/Algorithms.md` (solo per quei 20);
  "executable" se il KB finale (`assistant/kb/benchmark.csv`) ha almeno una esecuzione SUCCESS o FALLBACK per la
  modalita', altrimenti "described" (non avviabile); senza KB "wired" (non verificato). Le righe di Algorithms.md senza id nel catalogo non sono algoritmi.
- ts/graph: `modalities.list_specs()` (cataloghi scritti da altri; se vuoti il catalogo e' vuoto, dichiarato).
"""
from __future__ import annotations

import ast
import re
from typing import Any, Dict, List

from . import config
from .kb import store

MD_ALIAS = {"ddpm": "tabddpm", "synthetic-llm": "epic", "syrfd": "syrfdres"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower().replace("+", "plus"))


def _ast_assign(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return node.value
    raise KeyError(f"{name} not found in {path.name}")


def parse_algorithms_md() -> List[Dict[str, str]]:
    f = config.ROOT / "Algorithms" / "Algorithms.md"
    rows = []
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or set(line.replace("|", "").strip()) <= set("-: "):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if c[0] in ("ID", "Algoritmo"):
            continue
        if c[0].isdigit() and len(c) >= 4:
            rows.append({"name": c[1], "paper": c[2], "repo": c[3], "note": ""})
        elif len(c) >= 5:
            rows.append({"name": c[0], "paper": c[2], "repo": c[3], "note": c[4]})
    return rows


def tabular_catalog() -> Dict[str, Any]:
    """Catalogo tabellare = i 20 id di `config.CORE_ALGORITHMS` (come nel paper). Stato dal KB finale: executable se
    c'e' almeno una esecuzione SUCCESS/FALLBACK per la modalita' tabular, altrimenti described (mai avviabile)."""
    warnings: List[str] = []
    try:
        keys = list(ast.literal_eval(_ast_assign(config.ROOT / "config.py", "CORE_ALGORITHMS")))
    except Exception as e:  # noqa: BLE001
        return {"algorithms": [], "warnings": [f"tabular catalog unreadable: {type(e).__name__}: {e}"]}
    try:
        ui = {d["id"]: d for d in ast.literal_eval(_ast_assign(config.ROOT / "web_ui" / "app.py", "ALGORITHM_CATALOG"))}
    except Exception as e:  # noqa: BLE001
        ui = {}
        warnings.append(f"ALGORITHM_CATALOG in web_ui/app.py unreadable ({type(e).__name__}): names/categories missing")
    try:
        md = parse_algorithms_md()
    except Exception as e:  # noqa: BLE001
        md = []
        warnings.append(f"Algorithms.md unreadable ({type(e).__name__}): papers missing")
    md_by = {_norm(r["name"]): r for r in md}
    done = store.completed_algos("tabular")
    if done is None:
        warnings.append("knowledge base (assistant/kb/benchmark.csv) not found: status cannot be verified, "
                        "all algorithms are reported as wired")
    out = []
    for k in keys:
        u = ui.get(k, {})
        m = md_by.get(_norm(MD_ALIAS.get(k, k)), {})
        desc = u.get("desc", "")
        paper = m.get("paper", "")
        tags = []
        if "fair" in (desc + u.get("name", "")).lower():
            tags.append("fairness")
        if "privacy" in desc.lower() or "privately" in paper.lower():
            tags.append("privacy")
        if u.get("category") == "Constraint-Aware":
            tags.append("constraints")
        ok = done is not None and store.canon_algo(k) in done
        st = "wired" if done is None else "executable" if ok else "described"
        out.append({"key": k, "modality": "tabular", "name": u.get("name", k),
                    "status": st,
                    "strategy": u.get("category", "n/a"), "task": "generation/oversampling (minority class)",
                    "needs_gpu": None, "paper": paper or "not found in Algorithms.md",
                    "source": m.get("repo", "") or "n/a", "notes": desc, "tags": tags,
                    "status_source": "assistant/kb/benchmark.csv (at least one SUCCESS/FALLBACK run)" if ok
                    else "knowledge base missing (status not verified)" if done is None
                    else "assistant/kb/benchmark.csv (no completed run: described only)"})
    return {"algorithms": out, "warnings": warnings}


def modality_catalog(modality: str) -> Dict[str, Any]:
    try:
        import modalities
        specs = modalities.list_specs(modality)
    except Exception as e:  # noqa: BLE001
        return {"algorithms": [], "warnings": [f"modalities could not be loaded: {type(e).__name__}: {e}"]}
    out = []
    for s in specs:
        out.append({"key": s.key, "modality": modality, "name": s.name, "status": s.status, "strategy": s.strategy,
                    "task": s.task, "needs_gpu": s.needs_gpu, "paper": s.paper or "n/a", "source": s.source or "n/a",
                    "notes": s.notes, "tags": [], "license": (s.license.replace("nessuna licenza", "no license") if isinstance(s.license, str) else s.license), "status_source": "modalities.list_specs()",
                    "level": (s.extra.get("level") or ("node" if s.key.startswith("nl_") else "graph")) if modality == "graph" else None})
    w = [] if out else [f"catalog '{modality}' is empty for now (catalogs are being written by other agents)"]
    return {"algorithms": out, "warnings": w}


def get_catalog(modality: str) -> Dict[str, Any]:
    if modality not in config.MODALITIES:
        raise ValueError(f"unknown modality: {modality!r}; allowed: {config.MODALITIES}")
    return tabular_catalog() if modality == "tabular" else modality_catalog(modality)


def find(modality: str, key: str) -> Dict[str, Any]:
    key = (key or "").lower().strip()
    for a in get_catalog(modality)["algorithms"]:
        if a["key"] == key:
            return a
    raise KeyError(f"algorithm {key!r} not in the '{modality}' catalog")
