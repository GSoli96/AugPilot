"""App Flask minimale per provare l'assistente (NON usa web_ui/app.py). Porta 5057.

    python tests/test_assistant_server.py            # modalita' reale (online solo se c'e' la chiave)
    python tests/test_assistant_server.py --mock     # client simulato: nessuna chiamata all'API
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask  # noqa: E402


def make_app() -> Flask:
    from web_ui.assistant_bp import bp
    app = Flask(__name__, template_folder=str(ROOT / "web_ui" / "templates"), static_folder=str(ROOT / "web_ui" / "static"))
    app.register_blueprint(bp)

    @app.route("/")
    def index():
        return '<a href="/assistant">assistente</a>'
    return app


if __name__ == "__main__":
    if "--mock" in sys.argv:
        from assistant.agent import Agent
        from assistant.client import MockClient
        from web_ui import assistant_bp
        script = [[{"type": "text", "text": "[MOCK] Elenco gli algoritmi."},
                   {"type": "tool_use", "id": "t1", "name": "list_algorithms", "input": {"modality": "ts"}}],
                  [{"type": "text", "text": "[MOCK] Fatto (risposta simulata, nessuna API)."}]] * 20
        assistant_bp.set_agent(Agent(MockClient([r for r in script])))
    make_app().run(host="127.0.0.1", port=5057, debug=False)
