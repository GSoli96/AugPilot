# Modo chat: l'assistente senza chiave API (D11)

Finché non c'è una chiave API (e credito), l'assistente è Claude Code (la sessione o un agente da essa creato),
che usa gli STESSI strumenti e le STESSE regole dell'assistente web.

- Regole: `assistant/prompts.py` (`SYSTEM_PROMPT`), in particolare le regole anti-invenzione: ogni numero viene da uno
  strumento o dall'utente, con la fonte; «non lo so» quando il dato manca; mai presentare come eseguibile ciò che è
  `wired` o `described`.
- Strumenti: `python -m assistant.cli tools` (elenco), `python -m assistant.cli call <strumento> '<json>'` (sola lettura),
  `python -m assistant.cli run <modalità> --algorithms ... --datasets ... --epochs N --device cuda --yes`
  (esecuzione; `--yes` solo dopo la conferma dell'utente in chat se si superano 1 algoritmo, 1 dataset o 5 epoche).
- Flusso: obiettivo → dataset (`list_datasets`, `analyze_dataset`) → consiglio (`recommend_algorithms`) → piano e conferma
  → `run` → `compare_results` / `explain_metrics` → onestà sui limiti (campioni piccoli, poche epoche).
- Con la chiave: l'assistente web (`/assistant`) fa lo stesso con il modello via API; gli strumenti non cambiano.
