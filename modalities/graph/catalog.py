"""Catalogo grafi: carica i sotto-cataloghi se presenti."""
import importlib


def load():
    for name in ("catalog_node", "catalog_graph"):
        try:
            importlib.import_module(f"modalities.graph.{name}").load()
        except ModuleNotFoundError as e:
            if e.name != f"modalities.graph.{name}":
                raise
