"""E2 (ancoraggio dei numeri) ed E3 (confronto tra LLM) di WP12 (the project plan).

Esegue gli scenari di `scenarios.yaml` con l'agente di AugPilot e uno o più modelli Ollama cloud, senza
intervento umano (le esecuzioni che chiederebbero conferma restano in sospeso e contano come "pending").
Per ogni (modello, scenario) salva una riga JSONL in `bench/experiments/out/e2_e3.jsonl` (ripresa: le righe già
scritte si saltano) con: strumenti chiamati, errori, numeri nella risposta finale, numeri non ancorati (stesso
controllo di `assistant.agent.ungrounded_numbers`), passi, token, latenza, eventi di limite.

Condizioni (D21): `tools` = agente AugPilot con strumenti; `notools` = stesso modello, stessa domanda, nessuno strumento
(risposta dalla sola conoscenza del modello). In `notools` i numeri della risposta si confrontano con quelli prodotti dagli
strumenti per lo stesso modello e scenario (campo `tool_numbers` della riga `tools`): misura quanti numeri citati
corrispondono a valori misurati dal benchmark.

Uso:  python -m bench.experiments.e2_e3_scenarios --models gpt-oss:120b gpt-oss:20b [--conditions tools notools] [--only tab1 ts1] [--local]
Chiamate a ollama.com: coperte da D16/D12 (chiavi lette da ChaviOllama.json, mai stampate).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from bench.orchlib import done_ids, save_result  # noqa: E402

OUT = ROOT / "bench" / "experiments" / "out" / "e2_e3.jsonl"
# backend: "cloud" (ollama.com con le chiavi di ChaviOllama.json) o "local" (server Ollama locale, OLLAMA_HOST_URL)
BACKEND = {"cloud": True}


def make_client(model: str):
    from assistant.ollama_client import OllamaClient
    return OllamaClient(model, cloud=BACKEND["cloud"])


def wait_task(tm, timeout: float = 600.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not tm.get_active_task():
            return
        time.sleep(2)


def run_scenario(model: str, sc: dict) -> dict:
    from assistant.agent import Agent, Session, _extract_numbers, ungrounded_numbers
    from web_ui.task_manager import TaskManager

    client = make_client(model)
    agent = Agent(client)
    session = Session()
    session.ctx.task_manager = TaskManager()
    calls, errors, pending, limits, texts = [], [], 0, [], []
    t0 = time.time()
    steps = tokens = 0
    fatal = None
    for ev in agent.run_turn(session, sc["prompt"]):
        t = ev["type"]
        if t == "tool_call":
            calls.append(ev["name"])
        elif t == "tool_result" and not ev.get("ok", True):
            errors.append(ev["name"])
        elif t == "pending":
            pending += 1
        elif t == "limit":
            limits.append(ev["text"])
        elif t == "text":
            texts.append(ev["text"])
        elif t == "done":
            steps, tokens = ev.get("steps", 0), ev.get("tokens", 0)
        elif t == "error":
            fatal = ev["text"]
    latency = time.time() - t0
    if "run_augmentation" in calls:
        wait_task(session.ctx.task_manager)
    final = texts[-1] if texts else ""
    nums = _extract_numbers(final)
    bad = ungrounded_numbers(final, session.numbers) if final else []
    exp = sc.get("expected_tools", [])
    return {
        "id": f"{model}|{sc['id']}", "model": model, "scenario": sc["id"], "modality": sc["modality"],
        "tools_called": calls, "tool_errors": errors, "pending": pending, "limits": limits, "error": fatal,
        "expected_tools": exp, "expected_hit": sum(t in calls for t in exp) / len(exp) if exp else None,
        "completed": bool(final) and fatal is None and not limits,
        "n_numbers": len(nums), "n_ungrounded": len(bad), "ungrounded": bad,
        "steps": steps, "tokens": tokens, "latency_s": round(latency, 2), "final_text": final[:4000],
        "condition": "tools", "backend": "cloud" if BACKEND["cloud"] else "local", "tool_numbers": sorted(set(round(x, 6) for x in session.numbers))[:5000],
    }


NO_TOOLS_SYSTEM = ("You are an expert assistant on data augmentation for tabular data, time series and graphs. "
                   "Answer the user's question as well as you can.")


def run_scenario_notools(model: str, sc: dict, tool_numbers: list) -> dict:
    from assistant.agent import _extract_numbers, ungrounded_numbers

    client = make_client(model)
    t0 = time.time()
    resp = client.create(NO_TOOLS_SYSTEM, [{"role": "user", "content": sc["prompt"]}], [], 4096)
    latency = time.time() - t0
    final = "".join(b.get("text", "") for b in resp.content if b.get("type") == "text").strip()
    nums = _extract_numbers(final)
    bad = ungrounded_numbers(final, tool_numbers) if final else []
    return {
        "id": f"{model}|{sc['id']}|notools", "model": model, "scenario": sc["id"], "modality": sc["modality"],
        "condition": "notools", "backend": "cloud" if BACKEND["cloud"] else "local", "completed": bool(final), "n_numbers": len(nums), "n_ungrounded": len(bad),
        "ungrounded": bad, "n_reference_numbers": len(tool_numbers), "latency_s": round(latency, 2),
        "tokens": resp.input_tokens + resp.output_tokens, "final_text": final[:4000],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["gpt-oss:120b", "gpt-oss:20b"])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--conditions", nargs="+", default=["tools", "notools"], choices=["tools", "notools"])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--local", action="store_true",
                    help="usa il server Ollama locale (OLLAMA_HOST_URL, default 127.0.0.1:11434) invece di ollama.com")
    a = ap.parse_args()
    BACKEND["cloud"] = not a.local
    scs = yaml.safe_load((Path(__file__).parent / "scenarios.yaml").read_text(encoding="utf-8"))["scenarios"]
    if a.only:
        scs = [s for s in scs if s["id"] in a.only]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    done = done_ids(a.out)
    rows = {}
    if Path(a.out).exists():
        for line in Path(a.out).read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                rows[r["id"]] = r
            except Exception:
                pass
    for m in a.models:
        for sc in scs:
            jid = f"{m}|{sc['id']}"
            if "tools" in a.conditions and jid not in done:
                try:
                    rec = run_scenario(m, sc)
                except Exception as e:  # registrato come fallimento dello scenario
                    rec = {"id": jid, "model": m, "scenario": sc["id"], "modality": sc["modality"], "condition": "tools",
                           "error": f"{type(e).__name__}: {e}"[:500], "completed": False}
                save_result(a.out, rec)
                rows[jid] = rec
                print(f"[e2e3] {jid}: completed={rec.get('completed')} tools={rec.get('tools_called')} "
                      f"ungrounded={rec.get('n_ungrounded')}/{rec.get('n_numbers')} {rec.get('latency_s')}s", flush=True)
            nid = jid + "|notools"
            if "notools" in a.conditions and nid not in done:
                ref = (rows.get(jid) or {}).get("tool_numbers")
                if ref is None:
                    print(f"[e2e3] {nid}: salto, manca la riga tools di riferimento", flush=True)
                    continue
                try:
                    rec = run_scenario_notools(m, sc, ref)
                except Exception as e:
                    rec = {"id": nid, "model": m, "scenario": sc["id"], "modality": sc["modality"], "condition": "notools",
                           "error": f"{type(e).__name__}: {e}"[:500], "completed": False}
                save_result(a.out, rec)
                print(f"[e2e3] {nid}: ungrounded={rec.get('n_ungrounded')}/{rec.get('n_numbers')} "
                      f"{rec.get('latency_s')}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
