# Piccola libreria per gli script degli esperimenti (solo ASCII, nessuna dipendenza).
# Uso:
#   from orchlib import done_ids, save_result, progress
#   gia = done_ids("results.jsonl")            # id gia calcolati (ripresa dopo un'interruzione)
#   for i in items:
#       if i["id"] in gia: continue
#       save_result("results.jsonl", {"id": i["id"], ...})   # salva subito, riga per riga
#       progress(len(gia) + n, total)
import json
import os
import time

JOB_DIR = os.environ.get("ORCH_JOB_DIR", ".")


def progress(done, total, note=""):
    # Su Windows il runner puo' leggere il file nello stesso istante: si riprova e, se non va, si salta
    # (l'avanzamento e' solo informativo e non deve mai far morire l'esperimento).
    p = os.path.join(JOB_DIR, "progress.json")
    for _ in range(10):
        try:
            with open(p + ".tmp", "w") as f:
                json.dump({"done": done, "total": total, "note": note}, f)
            os.replace(p + ".tmp", p)
            return
        except OSError:
            time.sleep(0.2)


def done_ids(path, key="id"):
    ids = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ids.add(json.loads(line)[key])
                except Exception:
                    pass  # riga troncata da uno spegnimento: viene rifatta
    return ids


def save_result(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())
