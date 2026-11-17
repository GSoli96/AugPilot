# Copertura del benchmark

Fonte: `assistant/kb` (KB costruito con `python -m assistant.kb.build_kb`). Previste 1788 combinazioni (dataset x algoritmi x 3 seed).

| Modalità | Livello | Previste | SUCCESS | FALLBACK | FAILED | Mancanti |
|---|---|---|---|---|---|---|
| graph | graph | 342 | 342 | 0 | 0 | 0 |
| graph | node | 132 | 132 | 0 | 0 | 0 |
| tabular | - | 612 | 573 | 36 | 0 | 3 |
| ts | - | 702 | 681 | 0 | 19 | 2 |
| **totale** | | 1788 | 1728 | 36 | 19 | 5 |

## Eccezioni note (da dichiarare nel paper)

- ts / sigcwgan / energy: oltre il limite di 2 ore per combinazione (s0 FAILED per timeout; s1, s2 non eseguiti). Esiti: MANCANTE 2, FAILED 1.
- tabular / prefair / vowel0: oltre il limite di 2 ore per combinazione (fermato sui 3 seed, nessuna riga). Esiti: MANCANTE 3.
- ts / tsgm-cgan / energy: non applicabile: dataset senza etichette. Esiti: FAILED 3.
- ts / tsgm-cbetavae / energy: non applicabile: dataset senza etichette. Esiti: FAILED 3.
- ts / tsgm-cgan / stock: non applicabile: dataset senza etichette. Esiti: FAILED 3.
- ts / tsgm-cbetavae / stock: non applicabile: dataset senza etichette. Esiti: FAILED 3.
- ts / tsgm-cgan / tv_air: non applicabile: dataset senza etichette. Esiti: FAILED 3.
- ts / tsgm-cbetavae / tv_air: non applicabile: dataset senza etichette. Esiti: FAILED 3.

## FAILED e mancanti fuori dalle eccezioni (0)

Nessuno.

## FALLBACK (reimplementazioni dichiarate)

- tabular / kamino: 36 combinazioni

## Righe del KB fuori dal sottoinsieme previsto (0)

Corse di prova o dataset non del sottoinsieme: non entrano nel conteggio.
