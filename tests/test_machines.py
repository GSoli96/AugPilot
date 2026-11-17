"""Test di machines_bp e distributed/: solo localhost o mock, nessuna connessione a host esterni.

Avvia un'app Flask minimale su una porta libera scelta dal sistema (mai 5005 o 5057) e usa richieste HTTP reali.
Esecuzione: python -m pytest tests/test_machines.py   (oppure python tests/test_machines.py)
"""
import json
import socket
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask  # noqa: E402
from werkzeug.serving import make_server  # noqa: E402

from distributed import remote  # noqa: E402
from distributed.models import MachineStore, valid_host  # noqa: E402
from web_ui import machines_bp  # noqa: E402

GOOD = {"name": "lab-1", "host": "10.0.0.20", "port": 22, "user": "labuser", "auth_method": "agent",
        "os": "linux", "gpu_count": 2, "gpu_model": "RTX 4090", "ram_gb": 64, "workdir": "/home/labuser/da",
        "labels": "gpu, lab", "notes": "prova", "enabled": True}


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.file = Path(cls.tmp.name) / "machines.json"
        machines_bp.set_store(MachineStore(cls.file))
        app = Flask(__name__, template_folder=str(ROOT / "web_ui" / "templates"),
                    static_folder=str(ROOT / "web_ui" / "static"))
        app.register_blueprint(machines_bp.bp)
        cls.srv = make_server("127.0.0.1", 0, app, threaded=True)
        cls.port = cls.srv.server_port
        assert cls.port not in (5005, 5057)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        machines_bp.set_store(None)
        cls.tmp.cleanup()

    def setUp(self):
        if self.file.exists():
            self.file.unlink()

    def call(self, method, path, body=None, headers=None):
        data = json.dumps(body).encode() if body is not None else None
        h = {"Content-Type": "application/json"} if body is not None else {}
        h.update(headers or {})
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read().decode()), r.headers
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode()), e.headers

    # --- CRUD e validazione ---
    def test_page(self):
        with urllib.request.urlopen(self.base + "/machines") as r:
            html = r.read().decode()
        self.assertIn("TO BE DEFINED", html)

    def test_empty_then_crud(self):
        s, j, _ = self.call("GET", "/api/machines")
        self.assertEqual((s, j["count"]), (200, 0))
        self.assertFalse(self.file.exists(), "il file nasce al primo salvataggio")
        s, j, _ = self.call("POST", "/api/machines", GOOD)
        self.assertEqual(s, 201)
        mid = j["machine"]["id"]
        self.assertEqual(j["machine"]["labels"], ["gpu", "lab"])
        self.assertEqual(j["machine"]["status"]["state"], "non_verificata")
        self.assertTrue(self.file.exists())
        s, j, _ = self.call("PUT", f"/api/machines/{mid}", {"gpu_count": 4, "enabled": False})
        self.assertEqual((s, j["machine"]["gpu_count"], j["machine"]["enabled"]), (200, 4, False))
        s, j, _ = self.call("GET", "/api/machines")
        self.assertEqual((j["count"], j["enabled_count"]), (1, 0))
        self.assertEqual(self.call("DELETE", f"/api/machines/{mid}")[0], 200)
        self.assertEqual(self.call("DELETE", f"/api/machines/{mid}")[0], 404)
        self.assertEqual(self.call("PUT", "/api/machines/nonesiste", {"name": "x"})[0], 404)

    def test_validation(self):
        for patch, field in [({"host": "-oProxyCommand=evil"}, "host"), ({"host": "a b"}, "host"),
                             ({"host": "host;rm -rf /"}, "host"), ({"port": 70000}, "port"), ({"port": "abc"}, "port"),
                             ({"user": "-oFoo"}, "user"), ({"user": "a;b"}, "user"), ({"name": ""}, "name"),
                             ({"auth_method": "key", "key_path": ""}, "key_path"),
                             ({"auth_method": "key", "key_path": "-oFoo"}, "key_path"),
                             ({"auth_method": "key", "key_path": "/a/b;c|d"}, "key_path"),
                             ({"auth_method": "key", "key_path": "-----BEGIN OPENSSH PRIVATE KEY-----\nxx"}, "key_path"),
                             ({"os": "plan9"}, "os"), ({"gpu_count": -1}, "gpu_count")]:
            s, j, _ = self.call("POST", "/api/machines", {**GOOD, **patch})
            self.assertEqual(s, 400, patch)
            self.assertIn(field, j["fields"], patch)
        self.assertEqual(self.call("GET", "/api/machines")[1]["count"], 0)

    def test_no_secrets_accepted(self):
        for k in ("password", "passphrase", "private_key"):
            s, j, _ = self.call("POST", "/api/machines", {**GOOD, k: "x"})
            self.assertEqual(s, 400)
            self.assertIn(k, j["fields"])
        self.call("POST", "/api/machines", {**GOOD, "auth_method": "key", "key_path": "C:\\Users\\x\\.ssh\\id_ed25519"})
        txt = self.file.read_text(encoding="utf-8").lower()
        self.assertNotIn("password", txt)
        self.assertNotIn("begin", txt)

    def test_duplicates(self):
        self.assertEqual(self.call("POST", "/api/machines", GOOD)[0], 201)
        s, j, _ = self.call("POST", "/api/machines", {**GOOD, "host": "10.0.0.5"})
        self.assertEqual((s, "name" in j["fields"]), (400, True))
        s, j, _ = self.call("POST", "/api/machines", {**GOOD, "name": "LAB-2"})
        self.assertEqual((s, "host" in j["fields"]), (400, True))
        s, j, _ = self.call("POST", "/api/machines", {**GOOD, "name": "lab-2", "port": 2222})
        self.assertEqual(s, 201)

    def test_origin_and_body(self):
        s, _, _ = self.call("POST", "/api/machines", GOOD, headers={"Origin": "http://evil.example"})
        self.assertEqual(s, 403)
        req = urllib.request.Request(self.base + "/api/machines", data=b"non json", method="POST")
        with self.assertRaises(urllib.error.HTTPError) as c:
            urllib.request.urlopen(req)
        self.assertEqual(c.exception.code, 400)

    def test_corrupt_file_not_overwritten(self):
        self.file.write_text("{rotto", encoding="utf-8")
        s, j, _ = self.call("POST", "/api/machines", GOOD)
        self.assertEqual(s, 500)
        self.assertEqual(self.file.read_text(encoding="utf-8"), "{rotto")

    # --- import/export ---
    def test_export_import(self):
        self.call("POST", "/api/machines", GOOD)
        self.call("POST", "/api/machines", {**GOOD, "name": "lab-2", "host": "10.1.1.1"})
        s, exp, h = self.call("GET", "/api/machines/export")
        self.assertEqual((s, len(exp["machines"])), (200, 2))
        self.assertIn("attachment", h["Content-Disposition"])
        self.file.unlink()
        s, j, _ = self.call("POST", "/api/machines/import", exp)
        self.assertEqual((s, j["imported"], j["count"]), (201, 2, 2))
        # reimportare: tutti duplicati -> rifiutato, nulla cambia
        s, j, _ = self.call("POST", "/api/machines/import", exp)
        self.assertEqual(s, 400)
        self.assertEqual(self.call("GET", "/api/machines")[1]["count"], 2)
        # voce non valida -> tutto o niente
        s, j, _ = self.call("POST", "/api/machines/import", {"machines": [{**GOOD, "name": "ok-1", "host": "10.9.9.9"}, {**GOOD, "name": "bad", "host": "!!"}]})
        self.assertEqual(s, 400)
        self.assertEqual(self.call("GET", "/api/machines")[1]["count"], 2)
        # segreti nell'import rifiutati
        s, _, _ = self.call("POST", "/api/machines/import", [{**GOOD, "name": "s", "host": "10.8.8.8", "password": "x"}])
        self.assertEqual(s, 400)

    # --- check/profile con rete locale e mock ---
    def test_check_offline_local(self):
        with socket.socket() as s0:          # porta locale libera e chiusa
            s0.bind(("127.0.0.1", 0))
            closed = s0.getsockname()[1]
        mid = self.call("POST", "/api/machines", {**GOOD, "host": "127.0.0.1", "port": closed})[1]["machine"]["id"]
        s, j, _ = self.call("POST", f"/api/machines/{mid}/check")
        self.assertEqual(s, 200)
        self.assertEqual(j["machine"]["status"]["state"], "offline")
        self.assertIsNotNone(j["machine"]["status"]["checked_at"])

    def test_check_tcp_open_without_ssh_client(self):
        with socket.socket() as srv:
            srv.bind(("127.0.0.1", 0))
            srv.listen(5)
            mid = self.call("POST", "/api/machines", {**GOOD, "host": "127.0.0.1", "port": srv.getsockname()[1]})[1]["machine"]["id"]
            with mock.patch.object(remote, "ssh_binary", return_value=None):
                s, j, _ = self.call("POST", f"/api/machines/{mid}/check")
        self.assertEqual(j["machine"]["status"]["state"], "errore")
        self.assertIn("NOT verified", j["machine"]["status"]["message"])

    def test_check_online_and_auth_error_mocked(self):
        with socket.socket() as srv:
            srv.bind(("127.0.0.1", 0))
            srv.listen(5)
            mid = self.call("POST", "/api/machines", {**GOOD, "host": "127.0.0.1", "port": srv.getsockname()[1]})[1]["machine"]["id"]
            with mock.patch.object(remote, "ssh_binary", return_value="ssh"), \
                    mock.patch.object(remote, "run_remote", return_value=(0, "ok\n", "")):
                self.assertEqual(self.call("POST", f"/api/machines/{mid}/check")[1]["machine"]["status"]["state"], "online")
            with mock.patch.object(remote, "ssh_binary", return_value="ssh"), \
                    mock.patch.object(remote, "run_remote", return_value=(255, "", "Permission denied (publickey).")):
                st = self.call("POST", f"/api/machines/{mid}/check")[1]["machine"]["status"]
        self.assertEqual(st["state"], "errore")
        self.assertIn("Permission denied", st["message"])

    def test_profile_mocked(self):
        mid = self.call("POST", "/api/machines", GOOD)[1]["machine"]["id"]
        outs = {"system": (0, "Linux 6.1.0 x86_64\n", ""), "python": (0, "Python 3.10.12\n", ""),
                "gpu": (0, "NVIDIA GeForce RTX 4090, 24564, 23000\nNVIDIA GeForce RTX 4090, 24564, 1000\n", ""),
                "ram": (0, "              total        used        free      shared  buff/cache   available\nMem:          64000       10000       40000         100       14000       53000\nSwap:  0 0 0\n", ""),
                "disk": (0, "Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/sda1 1048576000 100000000 948576000 10% /home\n", "")}
        with mock.patch.object(remote, "ssh_binary", return_value="ssh"), \
                mock.patch.object(remote, "run_remote", side_effect=lambda m, k, timeout=20: outs[k]):
            s, j, _ = self.call("POST", f"/api/machines/{mid}/profile")
        p = j["machine"]["profile"]
        self.assertEqual(s, 200)
        self.assertEqual(len(p["gpus"]), 2)
        self.assertEqual((p["ram_total_mb"], p["ram_available_mb"]), (64000, 53000))
        self.assertEqual(p["python"], "Python 3.10.12")
        self.assertAlmostEqual(p["disk_total_gb"], 1000.0, delta=0.1)
        self.assertEqual(self.call("GET", "/api/machines")[1]["machines"][0]["profile"]["gpus"][1]["mem_free_mb"], 1000)

    def test_profile_without_ssh_is_501_and_failed_connection_502(self):
        mid = self.call("POST", "/api/machines", GOOD)[1]["machine"]["id"]
        with mock.patch.object(remote, "ssh_binary", return_value=None):
            self.assertEqual(self.call("POST", f"/api/machines/{mid}/profile")[0], 501)
        with mock.patch.object(remote, "ssh_binary", return_value="ssh"), \
                mock.patch.object(remote, "run_remote", return_value=(255, "", "Connection refused")):
            self.assertEqual(self.call("POST", f"/api/machines/{mid}/profile")[0], 502)
        self.assertEqual(self.call("GET", "/api/machines")[1]["machines"][0]["status"]["state"], "errore")


class RemoteUnitTest(unittest.TestCase):
    M = {"host": "10.0.0.1", "port": 22, "user": "u", "auth_method": "key", "key_path": "/k/id", "host_key_policy": "strict"}

    def test_argv_is_list_whitelist_and_strict(self):
        a = remote.build_ssh_args(self.M, "gpu")
        self.assertIsInstance(a, list)
        self.assertIn("BatchMode=yes", a)
        self.assertIn("ConnectTimeout=5", a)
        self.assertIn("StrictHostKeyChecking=yes", a)
        self.assertNotIn("StrictHostKeyChecking=accept-new", a)
        self.assertEqual(a[-3:-1], ["--", "10.0.0.1"])
        self.assertEqual(a[-1], remote.REMOTE_COMMANDS["gpu"])

    def test_accept_new_only_when_chosen(self):
        a = remote.build_ssh_args({**self.M, "host_key_policy": "accept-new"}, "ping")
        self.assertIn("StrictHostKeyChecking=accept-new", a)

    def test_arbitrary_command_and_bad_args_refused(self):
        with self.assertRaises(ValueError):
            remote.build_ssh_args(self.M, "rm -rf /")
        for bad in ({"host": "-oProxyCommand=x"}, {"user": "a b"}, {"key_path": "-i"}, {"port": "22; ls"}):
            with self.assertRaises(ValueError, msg=bad):
                remote.build_ssh_args({**self.M, **bad}, "ping")

    def test_no_shell_true_in_source(self):
        src = (ROOT / "distributed" / "remote.py").read_text(encoding="utf-8")
        self.assertNotIn("shell=True", src)

    def test_valid_host(self):
        for ok in ("10.0.0.1", "lab-pc.dei.unipd.it", "::1", "localhost"):
            self.assertTrue(valid_host(ok), ok)
        for bad in ("", "-x", "a_b", "a..b", "a b", "x;y", "$(id)", "1.2.3.4/24"):
            self.assertFalse(valid_host(bad), bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
