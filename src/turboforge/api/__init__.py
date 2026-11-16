"""Local web GUI for TurboForge.

A clean JSON API (Flask blueprint) plus a built single-page frontend (web/dist).
The API is the single source of truth for every choice the UI offers: dropdowns,
required/optional markers and recipe details all come from /api/meta, which is
derived from the same registry and processors the CLI uses, so the frontend can
never drift from the backend.
"""

from __future__ import annotations

from .server import create_app, main

__all__ = ["create_app", "main"]