"""Sezioni dell'interfaccia (/copilot, /guided, /control, /catalog), home, navigazione comune e /api/catalog_info."""
import json

import pytest

# L'import di web_ui.app avviene dentro le fixture, non in collezione: registra i blueprint reali e
# altererebbe lo stato globale visto dai test dell'assistente che girano prima.


@pytest.fixture()
def client():
    from web_ui import app as webapp
    webapp.app.config["TESTING"] = True
    return webapp.app.test_client()


@pytest.fixture()
def pages_bp():
    from web_ui import pages_bp as mod
    return mod


@pytest.mark.parametrize("path", ["/", "/copilot", "/guided", "/control", "/catalog", "/classic", "/machines", "/assistant"])
def test_pagine_rispondono_200_con_navigazione(client, path):
    r = client.get(path)
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert 'class="sitenav"' in html
    for href in ("/copilot", "/guided", "/catalog", "/control", "/machines"):
        assert f'href="{href}"' in html


@pytest.mark.parametrize("path", ["/copilot", "/guided", "/control", "/catalog"])
def test_pagina_corrente_evidenziata(client, path):
    html = client.get(path).get_data(as_text=True)
    assert f'href="{path}" aria-current="page"' in html


def test_home_ha_le_quattro_scelte(client):
    html = client.get("/").get_data(as_text=True)
    for testo in ("I don't know what to choose", "I already know what to do", "I just want to explore", "Overview"):
        assert testo in html


def test_vecchi_indirizzi_rimandano(client):
    for n, dest in {1: "/copilot", 2: "/guided", 4: "/control", 5: "/catalog"}.items():
        r = client.get(f"/design/{n}")
        assert r.status_code == 302 and r.headers["Location"].endswith(dest)
    assert client.get("/design/3").status_code == 404


def test_catalog_info_modalita_sconosciuta(client):
    assert client.get("/api/catalog_info/xyz").status_code == 404


def test_catalog_info_file_mancanti_non_da_errore(client, pages_bp, tmp_path, monkeypatch):
    monkeypatch.setattr(pages_bp, "CATALOG_INFO_DIR", tmp_path / "non_esiste")
    for mod in ("tabular", "ts", "graph"):
        r = client.get(f"/api/catalog_info/{mod}")
        assert r.status_code == 200
        assert r.get_json() == {"algorithms": {}, "datasets": {}}


def test_catalog_info_legge_e_tollera_incompleti(client, pages_bp, tmp_path, monkeypatch):
    monkeypatch.setattr(pages_bp, "CATALOG_INFO_DIR", tmp_path)
    (tmp_path / "algorithms_ts.json").write_text(json.dumps({"timegan": {"what": "x", "verified": False}, "rotto": 5}), encoding="utf-8")
    (tmp_path / "datasets_ts.json").write_text("{non json", encoding="utf-8")
    j = client.get("/api/catalog_info/ts").get_json()
    assert j["algorithms"] == {"timegan": {"what": "x", "verified": False}}
    assert j["datasets"] == {}
    assert j["warnings"]
