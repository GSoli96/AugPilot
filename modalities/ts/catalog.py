"""Catalogo serie temporali: carica i sotto-cataloghi se presenti."""
import importlib


def load():
    for name in ("catalog_classic", "catalog_gen", "catalog_ext"):
        try:
            importlib.import_module(f"modalities.ts.{name}").load()
        except ModuleNotFoundError as e:
            if e.name != f"modalities.ts.{name}":
                raise
