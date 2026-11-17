"""Modello dati, validazione e persistenza delle macchine (distributed/machines.json).

Il file non contiene mai password ne' contenuto di chiavi: solo il PERCORSO della chiave privata.
"""
from __future__ import annotations

import ipaddress
import json
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_FILE = Path(__file__).resolve().parent / "machines.json"
MAX_MACHINES = 200

AUTH_METHODS = ("key", "agent")
HOST_KEY_POLICIES = ("strict", "accept-new")   # accept-new solo se scelto esplicitamente
OS_CHOICES = ("linux", "windows", "macos", "altro")
STATES = ("non_verificata", "online", "offline", "errore")

FORBIDDEN_KEYS = {"password", "passwd", "passphrase", "private_key", "privatekey", "secret", "token", "key_content"}

_HOST_LABEL = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
USER_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.@-]{0,63}$")
PATH_RE = re.compile(r"^[A-Za-z0-9_ ./\\:~@+=,-]{1,260}$")
_NAME_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,60}$")
_CTRL = re.compile(r"[\x00-\x1f\x7f]")

EDITABLE = ("name", "host", "port", "user", "auth_method", "key_path", "host_key_policy", "os",
            "gpu_count", "gpu_model", "ram_gb", "workdir", "labels", "notes", "enabled")


class ValidationError(ValueError):
    def __init__(self, errors: Dict[str, str]):
        self.errors = errors
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def valid_host(host: str) -> bool:
    if not host or len(host) > 253 or host.startswith("-"):
        return False
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    if host.endswith("."):
        host = host[:-1]
    return bool(host) and all(_HOST_LABEL.match(p) for p in host.split("."))


def _int(v: Any, lo: int, hi: int, name: str, errors: Dict[str, str], default: Optional[int] = None) -> Optional[int]:
    if v is None or v == "":
        return default
    if isinstance(v, bool):
        errors[name] = "invalid number"
        return default
    try:
        n = int(v)
    except (TypeError, ValueError):
        errors[name] = "invalid integer"
        return default
    if not lo <= n <= hi:
        errors[name] = f"must be between {lo} and {hi}"
        return default
    return n


def _float(v: Any, lo: float, hi: float, name: str, errors: Dict[str, str]) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        n = float(v)
    except (TypeError, ValueError):
        errors[name] = "invalid number"
        return None
    if not lo <= n <= hi:
        errors[name] = f"must be between {lo} and {hi}"
        return None
    return n


def validate_machine(data: Dict[str, Any]) -> Dict[str, Any]:
    """Valida e normalizza i campi modificabili. Solleva ValidationError con un messaggio per campo."""
    if not isinstance(data, dict):
        raise ValidationError({"_": "JSON object expected"})
    errors: Dict[str, str] = {}
    bad = sorted(str(k) for k in data if str(k).lower() in FORBIDDEN_KEYS)
    if bad:
        raise ValidationError({k: "field not allowed: passwords and keys are not stored" for k in bad})

    out: Dict[str, Any] = {}
    name = str(data.get("name", "") or "").strip()
    if not _NAME_RE.match(name):
        errors["name"] = "name required (max 60 characters, no control characters)"
    out["name"] = name

    host = str(data.get("host", "") or "").strip()
    if not valid_host(host):
        errors["host"] = "invalid IP address or host name"
    out["host"] = host

    out["port"] = _int(data.get("port"), 1, 65535, "port", errors, 22)

    user = str(data.get("user", "") or "").strip()
    if not USER_RE.match(user):
        errors["user"] = "invalid user (letters, digits, _ . @ -; must not start with '-')"
    out["user"] = user

    auth = str(data.get("auth_method", "agent") or "agent").strip()
    if auth not in AUTH_METHODS:
        errors["auth_method"] = f"must be one of {', '.join(AUTH_METHODS)}"
    out["auth_method"] = auth

    key_path = str(data.get("key_path", "") or "").strip()
    if auth == "key":
        if not key_path:
            errors["key_path"] = "private key path required"
        elif key_path.startswith("-") or not PATH_RE.match(key_path) or "BEGIN" in key_path:
            errors["key_path"] = "invalid path (enter the file path, not the key content)"
    else:
        key_path = ""
    out["key_path"] = key_path

    pol = str(data.get("host_key_policy", "strict") or "strict").strip()
    if pol not in HOST_KEY_POLICIES:
        errors["host_key_policy"] = f"must be one of {', '.join(HOST_KEY_POLICIES)}"
    out["host_key_policy"] = pol

    osv = str(data.get("os", "linux") or "linux").strip().lower()
    if osv not in OS_CHOICES:
        errors["os"] = f"must be one of {', '.join(OS_CHOICES)}"
    out["os"] = osv

    out["gpu_count"] = _int(data.get("gpu_count"), 0, 64, "gpu_count", errors, 0)
    gm = str(data.get("gpu_model", "") or "").strip()
    if len(gm) > 80 or _CTRL.search(gm):
        errors["gpu_model"] = "invalid GPU model"
    out["gpu_model"] = gm
    out["ram_gb"] = _float(data.get("ram_gb"), 0, 65536, "ram_gb", errors)

    wd = str(data.get("workdir", "") or "").strip()
    if wd and (wd.startswith("-") or not PATH_RE.match(wd)):
        errors["workdir"] = "invalid working directory"
    out["workdir"] = wd

    labels = data.get("labels", [])
    if isinstance(labels, str):
        labels = [x.strip() for x in labels.split(",")]
    if not isinstance(labels, list):
        errors["labels"] = "list of labels expected"
        labels = []
    clean: List[str] = []
    for l in labels:
        l = str(l).strip()
        if not l:
            continue
        if len(l) > 40 or _CTRL.search(l):
            errors["labels"] = "invalid label (max 40 characters)"
        elif l not in clean:
            clean.append(l)
    out["labels"] = clean[:20]

    notes = str(data.get("notes", "") or "")
    if len(notes) > 2000:
        errors["notes"] = "notes too long (max 2000)"
    out["notes"] = notes

    en = data.get("enabled", True)
    if isinstance(en, str):
        en = en.strip().lower() in ("1", "true", "si", "yes")
    out["enabled"] = bool(en)

    if errors:
        raise ValidationError(errors)
    return out


def _new_record(fields: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": uuid.uuid4().hex[:12], **fields,
            "status": {"state": "non_verificata", "message": "", "checked_at": None},
            "profile": None, "created_at": now_iso(), "updated_at": now_iso()}


class MachineStore:
    """Persistenza JSON thread-safe. Il file viene creato al primo salvataggio (scrittura atomica)."""

    def __init__(self, path: Optional[os.PathLike] = None):
        self.path = Path(path or os.environ.get("MACHINES_FILE") or DEFAULT_FILE)
        self._lock = threading.RLock()

    def _load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise RuntimeError(f"{self.path} is unreadable or corrupted: {e}. Not overwriting it: fix it by hand.") from e
        if not isinstance(data, dict) or not isinstance(data.get("machines"), list):
            raise RuntimeError(f"{self.path}: invalid format (expected an object with the 'machines' key)")
        return data["machines"]

    def _save(self, machines: List[Dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"version": 1, "machines": machines}, indent=2, ensure_ascii=False)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".machines_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @staticmethod
    def _dup_errors(fields: Dict[str, Any], others: List[Dict[str, Any]]) -> Dict[str, str]:
        err: Dict[str, str] = {}
        for m in others:
            if m["name"].casefold() == fields["name"].casefold():
                err["name"] = "a machine with this name already exists"
            if (m["host"].casefold(), m["port"], m["user"]) == (fields["host"].casefold(), fields["port"], fields["user"]):
                err["host"] = "a machine with the same host, port and user already exists"
        return err

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._load()

    def get(self, mid: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return next((m for m in self._load() if m["id"] == mid), None)

    def add(self, data: Dict[str, Any]) -> Dict[str, Any]:
        fields = validate_machine(data)
        with self._lock:
            ms = self._load()
            if len(ms) >= MAX_MACHINES:
                raise ValidationError({"_": f"maximum {MAX_MACHINES} machines"})
            dup = self._dup_errors(fields, ms)
            if dup:
                raise ValidationError(dup)
            m = _new_record(fields)
            ms.append(m)
            self._save(ms)
            return m

    def update(self, mid: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            ms = self._load()
            cur = next((m for m in ms if m["id"] == mid), None)
            if cur is None:
                return None
            merged = {k: cur.get(k) for k in EDITABLE}
            merged.update({k: v for k, v in data.items() if k in EDITABLE or str(k).lower() in FORBIDDEN_KEYS})
            fields = validate_machine(merged)
            dup = self._dup_errors(fields, [m for m in ms if m["id"] != mid])
            if dup:
                raise ValidationError(dup)
            changed = any(fields[k] != cur.get(k) for k in ("host", "port", "user", "auth_method", "key_path", "host_key_policy"))
            cur.update(fields)
            cur["updated_at"] = now_iso()
            if changed:
                cur["status"] = {"state": "non_verificata", "message": "connection settings changed", "checked_at": None}
            self._save(ms)
            return cur

    def delete(self, mid: str) -> bool:
        with self._lock:
            ms = self._load()
            new = [m for m in ms if m["id"] != mid]
            if len(new) == len(ms):
                return False
            self._save(new)
            return True

    def set_status(self, mid: str, state: str, message: str) -> Optional[Dict[str, Any]]:
        if state not in STATES:
            raise ValueError(f"invalid state: {state}")
        with self._lock:
            ms = self._load()
            cur = next((m for m in ms if m["id"] == mid), None)
            if cur is None:
                return None
            cur["status"] = {"state": state, "message": message[:500], "checked_at": now_iso()}
            self._save(ms)
            return cur

    def set_profile(self, mid: str, profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            ms = self._load()
            cur = next((m for m in ms if m["id"] == mid), None)
            if cur is None:
                return None
            cur["profile"] = profile
            self._save(ms)
            return cur

    def export(self) -> Dict[str, Any]:
        with self._lock:
            return {"version": 1, "exported_at": now_iso(), "machines": self._load()}

    def import_(self, payload: Any) -> List[Dict[str, Any]]:
        """Tutto o niente. Le voci si AGGIUNGONO, nessuna esistente viene sovrascritta. Stato e profilo non si importano."""
        items = payload.get("machines") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise ValidationError({"_": "expected {'machines': [...]} or a list"})
        errors: Dict[str, str] = {}
        with self._lock:
            ms = self._load()
            if len(ms) + len(items) > MAX_MACHINES:
                raise ValidationError({"_": f"maximum {MAX_MACHINES} machines"})
            added: List[Dict[str, Any]] = []
            for i, it in enumerate(items):
                label = f"entry {i + 1}"
                try:
                    f = validate_machine(it)
                    label += f" ({f['name']})"
                    dup = self._dup_errors(f, ms + added)
                    if dup:
                        raise ValidationError(dup)
                except ValidationError as e:
                    errors.update({f"{label} / {k}": v for k, v in e.errors.items()})
                    continue
                added.append(_new_record(f))
            if errors:
                raise ValidationError(errors)
            self._save(ms + added)
            return added
