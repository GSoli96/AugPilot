"""
task_manager.py
===============
Asynchronous background task runner for the Data Augmentation Benchmark Web UI.
Handles non-blocking subprocess execution, live stdout/stderr capturing,
process group termination, and task state tracking.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import signal
import threading
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional


IS_WINDOWS = sys.platform == "win32"


def kill_process_tree(proc: subprocess.Popen, grace_s: float = 0.3) -> str:
    """Termina il processo e tutti i suoi discendenti. Restituisce il metodo usato.

    - POSIX: killpg sul gruppo creato con start_new_session=True (SIGTERM, poi SIGKILL).
    - Windows: killpg non esiste; si usa `taskkill /T /F` (albero di processi) e, se non disponibile,
      psutil; come ultima risorsa proc.kill() (solo il processo principale).
    """
    if proc.poll() is not None:
        return "gia' terminato"
    if not IS_WINDOWS:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(grace_s)
        if proc.poll() is None:
            os.killpg(pgid, signal.SIGKILL)
        return "killpg"
    try:
        r = subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, text=True, timeout=15)
        if r.returncode == 0 or proc.poll() is not None:
            return "taskkill /T /F"
    except Exception:
        pass
    try:
        import psutil
        parent = psutil.Process(proc.pid)
        for c in parent.children(recursive=True):
            try:
                c.kill()
            except psutil.Error:
                pass
        parent.kill()
        return "psutil"
    except Exception:
        pass
    proc.kill()
    return "proc.kill (main process only)"


class Task:
    def __init__(self, task_id: str, task_type: str, cmd: List[str], cwd: Path):
        self.task_id = task_id
        self.task_type = task_type  # 'augmentation' or 'evaluation'
        self.cmd = cmd
        self.cwd = cwd
        self.status = "PENDING"     # PENDING, RUNNING, SUCCESS, FAILED, STOPPED
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.exit_code: Optional[int] = None
        self.logs: List[str] = []
        self.error: Optional[str] = None
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def add_log(self, line: str):
        with self._lock:
            self.logs.append(line)

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "task_id": self.task_id,
                "task_type": self.task_type,
                "command": " ".join(self.cmd),
                "status": self.status,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "duration_s": round((self.finished_at or time.time()) - (self.started_at or self.created_at), 2)
                if self.started_at else 0.0,
                "exit_code": self.exit_code,
                "log_lines_count": len(self.logs),
                "error": self.error,
            }


class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.current_task_id: Optional[str] = None
        self._lock = threading.Lock()

    def create_task(self, task_type: str, cmd: List[str], cwd: Path) -> Task:
        with self._lock:
            task_id = str(uuid.uuid4())[:8]
            task = Task(task_id, task_type, cmd, cwd)
            self.tasks[task_id] = task
            return task

    def start_task(self, task: Task) -> None:
        def _runner():
            with self._lock:
                self.current_task_id = task.task_id
                task.status = "RUNNING"
                task.started_at = time.time()

            task.add_log(f"[{time.strftime('%H:%M:%S')}] 🚀 Task started: {' '.join(task.cmd)}\n")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            try:
                # POSIX: start_new_session=True crea un gruppo di processi da terminare con killpg.
                # Windows: nuovo gruppo di processi; l'albero si termina con taskkill /T /F.
                popen_kwargs = {}
                if IS_WINDOWS:
                    popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                else:
                    popen_kwargs["start_new_session"] = True
                env.setdefault("PYTHONIOENCODING", "utf-8")
                task.proc = subprocess.Popen(
                    task.cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=str(task.cwd),
                    env=env,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    **popen_kwargs,
                )

                if task.proc.stdout:
                    for line in iter(task.proc.stdout.readline, ""):
                        if not line:
                            break
                        task.add_log(line)

                task.proc.wait()
                with self._lock:
                    task.exit_code = task.proc.returncode
                    task.finished_at = time.time()
                    if task.status != "STOPPED":
                        if task.proc.returncode == 0:
                            task.status = "SUCCESS"
                            task.add_log(f"\n[{time.strftime('%H:%M:%S')}] ✅ Task completed successfully with exit code 0.\n")
                        else:
                            task.status = "FAILED"
                            task.add_log(f"\n[{time.strftime('%H:%M:%S')}] ❌ Task exited with error code {task.proc.returncode}.\n")

            except Exception as e:
                with self._lock:
                    task.status = "FAILED"
                    task.finished_at = time.time()
                    task.error = str(e)
                    task.add_log(f"\n[{time.strftime('%H:%M:%S')}] ❌ Exception during execution: {e}\n")
            finally:
                with self._lock:
                    if self.current_task_id == task.task_id:
                        self.current_task_id = None

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()

    def stop_task(self, task_id: str) -> bool:
        with self._lock:
            task = self.tasks.get(task_id)
            if not task or not task.proc or task.status not in ("PENDING", "RUNNING"):
                return False

            task.status = "STOPPED"
            task.finished_at = time.time()
            task.add_log(f"\n[{time.strftime('%H:%M:%S')}] ⚠️  Task termination requested by user.\n")

            try:
                # Terminate the whole process tree so child processes (workers) stop too
                how = kill_process_tree(task.proc)
                task.add_log(f"Process tree terminated: {how}\n")
            except Exception as e:
                task.add_log(f"Warning during process termination: {e}\n")

            return True

    def get_task(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self.tasks.get(task_id)

    def get_active_task(self) -> Optional[Task]:
        with self._lock:
            if self.current_task_id:
                return self.tasks.get(self.current_task_id)
            return None

    def get_logs_from(self, task_id: str, start_index: int = 0) -> Dict[str, Any]:
        task = self.get_task(task_id)
        if not task:
            return {"lines": [], "next_index": 0, "status": "UNKNOWN"}
        with task._lock:
            lines = task.logs[start_index:]
            return {
                "lines": lines,
                "next_index": len(task.logs),
                "status": task.status,
                "duration_s": round((task.finished_at or time.time()) - (task.started_at or task.created_at), 2)
                if task.started_at else 0.0,
            }
