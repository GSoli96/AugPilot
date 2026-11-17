# distributed/ - macchine per l'esecuzione distribuita

Stato: solo interfaccia e modello dati. Lo scheduler (assegnazione dei task, ribilanciamento, checkpoint) NON esiste e si decide a parte.

## Fatto
- `models.py`: campi di una macchina, validazione, persistenza in `distributed/machines.json` (creato al primo salvataggio, scrittura atomica; un file corrotto non viene sovrascritto), import/export JSON.
- `remote.py`: verifica connessione (TCP sulla porta SSH, poi `ssh ... echo ok` se il client esiste) e profilazione (sistema, python, `nvidia-smi`, RAM, disco).
- `web_ui/machines_bp.py` (blueprint `bp`) e `web_ui/templates/machines.html` (pagina `/machines`).
- Test: `tests/test_machines.py` (solo localhost e mock).

Campi per macchina: nome, host/IP, porta, utente, autenticazione (`agent` oppure `key` con SOLO il percorso della chiave privata), politica chiave host (`strict` di default, `accept-new` solo se scelta), sistema operativo, numero e modello di GPU, RAM (GB), cartella di lavoro remota, etichette, note, abilitata, stato (`non_verificata`, `online`, `offline`, `errore` + messaggio + data), profilo rilevato.

## API
| Metodo e percorso | Funzione |
|---|---|
| `GET /machines` | pagina |
| `GET /api/machines` | elenco, `count`, `enabled_count`, `ssh_client` |
| `POST /api/machines` | aggiunge (201; 400 con `fields` se non valido) |
| `PUT /api/machines/<id>` | modifica parziale |
| `DELETE /api/machines/<id>` | rimuove (la conferma e' nella pagina) |
| `POST /api/machines/<id>/check` | verifica connessione e salva lo stato |
| `POST /api/machines/<id>/profile` | profila e salva il profilo (501 senza client ssh, 502 se la connessione fallisce) |
| `GET /api/machines/export` | scarica il JSON |
| `POST /api/machines/import` | aggiunge macchine da JSON; tutto o niente, nessuna voce esistente viene sovrascritta |

Registrazione in `web_ui/app.py` (a cura di chi lo possiede): `from web_ui.machines_bp import bp as machines_bp; app.register_blueprint(machines_bp)`.

## Segnaposto (da definire)
La sezione "Assegnazione dei task" della pagina: scelta delle macchine abilitate (sola visualizzazione), tabella algoritmo x dataset x macchina vuota, pulsante di avvio disabilitato. Nessuna API, nessuna logica.

## Sicurezza
- Mai password ne' contenuto di chiavi nel file: i campi `password`, `passphrase`, `private_key` e simili sono rifiutati; della chiave si salva il percorso, che non viene mai letto.
- Comandi remoti: solo la lista bianca fissa `REMOTE_COMMANDS` in `remote.py`, scelti per nome interno; il client non puo' inviare comandi.
- `subprocess` con lista di argomenti, mai shell. Host, utente, porta e percorso chiave sono validati nel modello e rivalidati prima di costruire `ssh`; niente valori che iniziano con `-`; `--` prima dell'host.
- `BatchMode=yes`, `PasswordAuthentication=no`, `ConnectTimeout=5`: nessuna richiesta di password. `StrictHostKeyChecking=yes` salvo scelta esplicita `accept-new` della macchina.
- Le richieste che modificano dati con `Origin` diverso dall'host sono rifiutate (403).
- Limiti noti: nessuna autenticazione sulle API (come il resto della web UI, pensata per uso locale); "Verifica connessione" apre una connessione TCP verso l'host indicato da chi usa la pagina, quindi la UI non va esposta su reti non fidate. I messaggi di errore di ssh sono mostrati nella pagina e possono contenere il percorso della chiave. `machines.json` contiene IP e utenti del laboratorio: non va pubblicato.
- La profilazione usa comandi Linux/macOS (`uname`, `free`, `df`): su macchine Windows i campi corrispondenti risultano in errore, mostrato nel profilo.
