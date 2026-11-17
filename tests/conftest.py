import pytest


@pytest.fixture(autouse=True)
def _backend_claude_di_default(monkeypatch):
    """I test non devono usare l'Ollama cloud reale (chiavi in ChaviOllama.json): backend fissato a claude."""
    monkeypatch.setenv("ASSISTANT_BACKEND", "claude")


@pytest.fixture(autouse=True)
def _kb_assente_di_default(monkeypatch, tmp_path_factory):
    """I test storici del consigliere (CSV reali) non devono dipendere dal KB reale in assistant/kb/: cartella inesistente.
    I test del KB (test_recommender_kb.py) costruiscono un KB finto e lo indicano con kb_dir o AUGPILOT_KB_DIR."""
    monkeypatch.setenv("AUGPILOT_KB_DIR", str(tmp_path_factory.getbasetemp() / "nessun_kb"))


# Test che leggono i dataset completi e i CSV storici del benchmark (cartelle datasets/, datasets_graph/,
# evaluation_metrics/, non ridistribuite): in un clone pulito vengono saltati con un motivo esplicito.
_NEED_FULL_DATA = {
    "test_recommend_tabular_real_data", "test_recommend_fairness_filter",
    "test_recommend_time_budget_and_unmeasured_warning", "test_compare_and_explain_tabular_real",
    "test_analyze_tabular", "test_analyze_ts_and_graph", "test_list_datasets",
    "test_agent_loop_tool_use_and_result_pairing", "test_run_requires_confirmation[kw2]",
    "test_bp_offline_chat_declares_offline_and_tools_still_work", "test_fallback_without_kb",
    "test_run_modality_dall_endpoint_usa_cartelle_dedicate",
}


def pytest_collection_modifyitems(config, items):
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    if all((root / p).exists() for p in ("datasets/adult_min.csv", "datasets_graph/cora.pt", "evaluation_metrics")):
        return
    skip = pytest.mark.skip(reason="needs the full benchmark datasets and historical CSVs (not redistributed, see README)")
    for it in items:
        if it.name in _NEED_FULL_DATA:
            it.add_marker(skip)
