"""Registro degli algoritmi per modalità e lancio dei worker in ambienti isolati."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent

# Stati: "executable" = testato con almeno una esecuzione reale; "wired" = collegato ma non ancora
# testato; "described" = solo nel catalogo (non si presenta come eseguibile).
STATUSES = ("executable", "wired", "described")


@dataclass
class AlgoSpec:
    key: str                 # identificatore breve, es. "timevae"
    modality: str            # "ts" | "graph"
    name: str                # nome leggibile
    strategy: str            # famiglia: noise, warping, interpolation, vae, gan, diffusion, mixup, ...
    task: str = "generation"  # generation | view (vista perturbata) | oversampling
    env: str = "main"        # nome della cartella in envs/ ("ts", "graph") o "main"
    worker: str = ""         # percorso relativo a ROOT del worker
    status: str = "described"
    source: str = ""         # repo / paper
    license: str = ""
    paper: str = ""
    needs_gpu: bool = False
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


REGISTRY: Dict[tuple, AlgoSpec] = {}


def register(spec: AlgoSpec) -> AlgoSpec:
    assert spec.status in STATUSES, spec.status
    REGISTRY[(spec.modality, spec.key)] = spec
    return spec


def get_spec(modality: str, key: str) -> AlgoSpec:
    return REGISTRY[(modality, key.lower().strip())]


def list_specs(modality: Optional[str] = None) -> List[AlgoSpec]:
    _load_catalogs()
    return [s for (m, _), s in sorted(REGISTRY.items()) if modality in (None, m)]


def _load_catalogs() -> None:
    # Ogni modalità espone il proprio catalogo in modalities/<mod>/catalog.py (funzione `load()`).
    import importlib
    for mod in ("ts", "graph"):
        try:
            importlib.import_module(f"modalities.{mod}.catalog").load()
        except ModuleNotFoundError as e:  # catalogo non ancora scritto
            if e.name != f"modalities.{mod}.catalog":
                raise


def env_python(env: str) -> str:
    if env and env != "main":
        for rel in (f"envs/{env}/Scripts/python.exe", f"envs/{env}/bin/python"):
            p = ROOT / rel
            if p.exists():
                return str(p)
    return sys.executable


def run_algorithm(modality: str, key: str, data: str, out: str, n_samples: Optional[int] = None,
                  epochs: int = 50, device: str = "cpu", seed: int = 0, timeout: int = 3600,
                  extra_args: Optional[List[str]] = None) -> Dict[str, Any]:
    """Lancia il worker e restituisce il record JSON; non solleva mai eccezioni."""
    _load_catalogs()
    t0 = time.time()
    try:
        spec = get_spec(modality, key)
        if not spec.worker:
            return {"status": "FAILED", "error_message": f"{key}: described only, no worker"}
        cmd = [env_python(spec.env), str(ROOT / spec.worker), "--data", str(data), "--out", str(out),
               "--epochs", str(epochs), "--device", device, "--seed", str(seed)]
        if n_samples:
            cmd += ["--n_samples", str(n_samples)]
        if spec.extra.get("method"):  # worker condiviso da più metodi
            cmd += ["--method", str(spec.extra["method"])]
        cmd += list(extra_args or [])
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=str(ROOT),
                              env={**__import__("os").environ, "KERAS_BACKEND": "torch",
                                   "PYTHONIOENCODING": "utf-8"})
        lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
        rec: Dict[str, Any] = {}
        if lines:
            try:
                rec = json.loads(lines[-1])
            except json.JSONDecodeError:
                pass
        if not rec:
            rec = {"status": "FAILED", "error_message": (proc.stderr or proc.stdout)[-1500:]}
        elif proc.returncode != 0 and rec.get("status") == "SUCCESS":
            rec = {"status": "FAILED", "error_message": f"exit code {proc.returncode}"}
        rec.setdefault("time_s", time.time() - t0)
        return rec
    except subprocess.TimeoutExpired:
        return {"status": "FAILED", "error_message": f"timeout {timeout}s", "time_s": time.time() - t0}
    except Exception as e:  # noqa: BLE001
        return {"status": "FAILED", "error_message": f"{type(e).__name__}: {e}", "time_s": time.time() - t0}
