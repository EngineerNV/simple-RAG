"""Utilities that make the numbered pipeline scripts easier to import.

The ingest helper lives in ``00_ingest.py`` so that the execution order is obvious when
running scripts directly. To keep later milestones simple we re-export ``ingest`` and
``preview`` from this package, allowing ``from scripts import ingest`` without awkward
``importlib`` gymnastics.
"""

from importlib import import_module  # Dynamically load modules whose filenames begin with digits

_ingest_module = import_module("scripts.00_ingest")

# Re-export the public helpers so downstream scripts can import them cleanly.
ingest = _ingest_module.ingest  # type: ignore[attr-defined]
preview = _ingest_module.preview  # type: ignore[attr-defined]

__all__ = ["ingest", "preview"]
