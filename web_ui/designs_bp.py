"""Compatibilita' con i vecchi indirizzi dei prototipi: i quattro prototipi scelti sono ora sezioni dell'applicazione
(web_ui/pages_bp.py). /design/<N> e /designs rimandano alle sezioni; nessuna logica di prodotto qui.
"""
from flask import Blueprint, abort, redirect

bp = Blueprint("designs", __name__)

# vecchio numero del prototipo -> sezione definitiva (il 3 e' stato scartato)
MOVED = {1: "/copilot", 2: "/guided", 4: "/control", 5: "/catalog"}


@bp.route("/designs")
def designs_index():
    return redirect("/", code=302)


@bp.route("/design/<int:n>")
def design_page(n):
    if n not in MOVED:
        abort(404)
    return redirect(MOVED[n], code=302)
