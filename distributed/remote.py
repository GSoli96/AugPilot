"""Accesso remoto: verifica connessione e profilazione. Solo comandi a LISTA BIANCA FISSA.

Nessun comando arriva dal client. subprocess sempre con lista di argomenti (mai tramite shell), BatchMode=yes
(mai password interattive). StrictHostKeyChecking=accept-new solo se la macchina lo ha scelto esplicitamente.
"""
from __future__ import annotations

import re
import shutil
import socket
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from .models import PATH_RE, USER_RE, now_iso, valid_host

# Unici comandi eseguibili sul remoto (stringhe costanti, nessuna interpolazione).
REMOTE_COMMANDS: Dict[str, str] = {
    "ping": "echo ok",
    "system": "uname -srm",
    "python": "python3 --version 2>&1 || python --version 2>&1",
    "gpu": "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits",
    "ram": "free -m",
    "disk": "df -Pk ~",
}
PROFILE_KEYS = ("system", "python", "gpu", "ram", "disk")
TCP_TIMEOUT = 5
SSH_TIMEOUT = 20


def ssh_binary() -> Optional[str]:
    return shutil.which("ssh")


def check_tcp(host: str, port: int, timeout: float = TCP_TIMEOUT) -> Tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, "port reachable"
    except socket.timeout:
        return False, f"timeout ({timeout}s) on port {port}"
    except OSError as e:
        return False, f"connection refused or host unreachable: {e.strerror or e}"


def build_ssh_args(m: Dict[str, Any], remote_key: str, ssh_bin: str = "ssh") -> List[str]:
    """Costruisce la lista argv. Rivalida host/utente/chiave (difesa in profondita' contro injection)."""
    if remote_key not in REMOTE_COMMANDS:
        raise ValueError(f"remote command not in the allowlist: {remote_key!r}")
    host, user, port = str(m["host"]), str(m["user"]), m["port"]
    if not valid_host(host) or not USER_RE.match(user) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("invalid host, user or port")
    policy = "accept-new" if m.get("host_key_policy") == "accept-new" else "yes"
    args = [ssh_bin, "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
            "-o", f"StrictHostKeyChecking={policy}", "-o", "PasswordAuthentication=no",
            "-o", "KbdInteractiveAuthentication=no", "-p", str(port), "-l", user]
    if m.get("auth_method") == "key":
        kp = str(m.get("key_path", ""))
        if not kp or kp.startswith("-") or not PATH_RE.match(kp):
            raise ValueError("invalid key path")
        args += ["-o", "IdentitiesOnly=yes", "-i", kp]
    args += ["--", host, REMOTE_COMMANDS[remote_key]]
    return args


def run_remote(m: Dict[str, Any], remote_key: str, timeout: int = SSH_TIMEOUT) -> Tuple[int, str, str]:
    exe = ssh_binary()
    if exe is None:
        raise FileNotFoundError("ssh client not found in PATH")
    args = build_ssh_args(m, remote_key, exe)
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout ({timeout}s)"
    return p.returncode, p.stdout, p.stderr


def _short(s: str) -> str:
    return " ".join(s.split())[:300]


def check_machine(m: Dict[str, Any]) -> Tuple[str, str]:
    """Ritorna (stato, messaggio). Stati: online / offline / errore."""
    ok, msg = check_tcp(m["host"], m["port"])
    if not ok:
        return "offline", msg
    if ssh_binary() is None:
        return "errore", "SSH port reachable, but no ssh client: authentication NOT verified"
    try:
        rc, out, err = run_remote(m, "ping", timeout=15)
    except ValueError as e:
        return "errore", str(e)
    if rc == 0 and out.strip() == "ok":
        return "online", "port reachable and SSH authentication succeeded"
    return "errore", _short(err) or f"ssh returned code {rc}"


def parse_profile(raw: Dict[str, Tuple[int, str, str]]) -> Dict[str, Any]:
    prof: Dict[str, Any] = {"collected_at": now_iso(), "errors": {}, "gpus": []}
    for k, (rc, _out, err) in raw.items():
        if rc != 0:
            prof["errors"][k] = _short(err) or f"exit code {rc}"
    g = raw.get("system")
    if g and g[0] == 0:
        prof["system"] = g[1].strip()
    g = raw.get("python")
    if g and g[0] == 0 and g[1].strip():
        prof["python"] = g[1].strip().splitlines()[0]
    g = raw.get("gpu")
    if g and g[0] == 0:
        for line in g[1].strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                prof["gpus"].append({"name": parts[0], "mem_total_mb": int(parts[1]), "mem_free_mb": int(parts[2])})
    g = raw.get("ram")
    if g and g[0] == 0:
        mm = re.search(r"^Mem:\s+(\d+)\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)", g[1], re.M)
        if mm:
            prof["ram_total_mb"], prof["ram_available_mb"] = int(mm.group(1)), int(mm.group(2))
        else:
            prof["errors"]["ram"] = "unrecognized 'free' output"
    g = raw.get("disk")
    if g and g[0] == 0:
        lines = g[1].strip().splitlines()
        f = lines[-1].split() if len(lines) >= 2 else []
        if len(f) >= 4 and f[1].isdigit() and f[3].isdigit():
            prof["disk_total_gb"] = round(int(f[1]) / 1024 / 1024, 1)
            prof["disk_free_gb"] = round(int(f[3]) / 1024 / 1024, 1)
        else:
            prof["errors"]["disk"] = "unrecognized 'df' output"
    return prof


def profile_machine(m: Dict[str, Any]) -> Dict[str, Any]:
    """Esegue i comandi della lista bianca. Solleva FileNotFoundError (niente ssh) o ConnectionError."""
    raw: Dict[str, Tuple[int, str, str]] = {}
    for k in PROFILE_KEYS:
        raw[k] = run_remote(m, k)
        if k == PROFILE_KEYS[0] and raw[k][0] in (124, 255):   # connessione/autenticazione fallita
            raise ConnectionError(_short(raw[k][2]) or "SSH connection failed")
    return parse_profile(raw)
