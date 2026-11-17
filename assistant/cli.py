"""Interfaccia a riga di comando agli strumenti dell'assistente, per usarli senza chiave API.

Nel «modo chat» (D11) l'intelligenza è Claude Code (la sessione o un suo agente); gli strumenti sono gli stessi
dell'assistente web (assistant/tools.py), con gli stessi controlli (percorsi ammessi, algoritmi del catalogo, nessun
algoritmo solo descritto). Uso:

  python -m assistant.cli tools                      # elenco degli strumenti e dei parametri
  python -m assistant.cli call <strumento> '<json>'  # chiama uno strumento di sola lettura
  python -m assistant.cli run  <modalita> --algorithms a b --datasets d --epochs 2 --device cuda --yes
                                                     # esegue in primo piano (serve --yes: la conferma dell'utente)
Le uscite sono JSON su stdout. Nessuna chiamata all'API, nessuna chiave.
"""
import argparse
import json
import subprocess
import sys

from . import tools
from .tools import ToolContext, dispatch

READ_ONLY = {"list_algorithms", "describe_algorithm", "list_datasets", "analyze_dataset", "recommend_algorithms",
             "compare_results", "explain_metrics", "explain_recommendation", "evaluate"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tools")
    c = sub.add_parser("call"); c.add_argument("name"); c.add_argument("args", nargs="?", default="{}")
    r = sub.add_parser("run"); r.add_argument("modality", choices=["tabular", "ts", "graph"])
    r.add_argument("--algorithms", nargs="+", required=True); r.add_argument("--datasets", nargs="+", required=True)
    r.add_argument("--epochs", type=int, default=2); r.add_argument("--n_samples", type=int)
    r.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    r.add_argument("--yes", action="store_true", help="conferma dell'utente ricevuta in chat")
    r.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args(argv)

    if a.cmd == "tools":
        out = [{"name": t["name"], "description": t["description"], "parameters": list(t["input_schema"]["properties"])}
               for t in tools.TOOL_SCHEMAS]
        print(json.dumps(out, ensure_ascii=False, indent=1)); return 0
    if a.cmd == "call":
        if a.name not in READ_ONLY:
            print(json.dumps({"error": f"{a.name}: usa il sottocomando 'run' per le esecuzioni"})); return 2
        print(json.dumps(dispatch(ToolContext(), a.name, json.loads(a.args)), ensure_ascii=False, default=str)); return 0
    # run: stessa validazione dello strumento run_augmentation
    try:
        v = tools._validate_run({"modality": a.modality, "algorithms": a.algorithms, "datasets": a.datasets,
                                 "epochs": a.epochs, "n_samples": a.n_samples, "device": a.device})
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": f"{type(e).__name__}: {e}"})); return 2
    why = tools.needs_confirmation(v)
    if why and not a.yes:
        print(json.dumps({"status": "needs_confirmation", "reason": why, "plan": v,
                          "message": "Chiedi conferma all'utente in chat, poi ripeti con --yes."})); return 3
    cmd = tools._build_cmd(v)
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=a.timeout, cwd=str(tools.config.ROOT))
    print(json.dumps({"status": "done" if p.returncode == 0 else "failed", "returncode": p.returncode,
                      "command": " ".join(cmd[1:]), "stdout_tail": p.stdout[-3000:], "stderr_tail": p.stderr[-1500:]},
                     ensure_ascii=False)); return p.returncode


if __name__ == "__main__":
    sys.exit(main())
