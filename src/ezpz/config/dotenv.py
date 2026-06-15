"""Minimal .env loader (no external dependency).

Loads `KEY=VALUE` lines from a `.env` file into ``os.environ``. Real environment variables always
win — a `.env` value is applied only when the variable is unset (unless ``override=True``), so an
exported key takes precedence. Supports comments (`#`), blank lines, an optional `export ` prefix,
and single/double-quoted values.

Secrets are referenced by env-var NAME from adapter configs (``api_key_env``); the `.env` file is
gitignored — never commit real keys.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: "str | os.PathLike" = ".env", override: bool = False) -> int:
    """Load `.env` at ``path`` into os.environ. Returns the number of variables set. No-op if absent."""
    p = Path(path)
    if not p.exists():
        return 0
    loaded = 0
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]  # strip matching surrounding quotes
        if key and (override or key not in os.environ):
            os.environ[key] = value
            loaded += 1
    return loaded
