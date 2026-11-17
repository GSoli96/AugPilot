"""Prova da terminale del ciclo agentico con il backend scelto: python -m assistant.chat_test "domanda" [modello]."""
import json
import os
import sys
import time

from . import client as cl
from .agent import Agent, Session


def main():
    q = sys.argv[1] if len(sys.argv) > 1 else "Quanti algoritmi eseguibili ci sono per i grafi?"
    if len(sys.argv) > 2:
        os.environ["ASSISTANT_OLLAMA_MODEL"] = sys.argv[2]
    st = cl.get_status()
    print("stato:", json.dumps(st, ensure_ascii=False))
    c = cl.make_client()
    if c is None:
        print("offline: niente da provare"); return 2
    from .tools import ToolContext
    s = Session()
    t0 = time.time()
    for ev in Agent(c).run_turn(s, q):
        k = ev["type"]
        if k == "tool_call":
            print(f"  [strumento] {ev['name']} {json.dumps(ev['input'], ensure_ascii=False)[:160]}")
        elif k == "tool_result":
            print(f"  [esito] {'ok' if ev['ok'] else 'ERRORE'} {json.dumps(ev['result'], ensure_ascii=False)[:140]}")
        elif k == "text":
            print("\nRISPOSTA:\n" + ev["text"])
        elif k in ("warning", "limit", "error", "pending"):
            print(f"  [{k}] {ev.get('text') or ev}")
        elif k == "done":
            print(f"\n(passi {ev['steps']}, token {ev['tokens']}, {time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
