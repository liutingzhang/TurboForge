"""Local web GUI entry point.

The implementation lives in turboforge.api: a data-driven JSON API plus the
built single-page frontend (web/dist) served on one origin. This module is kept
as the stable entry the CLI calls (``turboforge webui``).

Requires flask:  pip install flask
Build the frontend first:  cd web && npm install && npm run build
"""

from __future__ import annotations

from .api.server import create_app, main

__all__ = ["create_app", "main"]


if __name__ == "__main__":
    raise SystemExit(main())