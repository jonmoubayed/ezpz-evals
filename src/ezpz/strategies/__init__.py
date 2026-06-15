"""Composite extraction strategies — meta-adapters that orchestrate inner pipelines.

Importing this package registers them, so `adapter: cascade` / `ensemble` / `verify` work like any
other adapter. They wrap ANY registered inner adapters (fake for no-key demos, or real ones).
"""
from ezpz.strategies import cascade, ensemble, verify  # noqa: F401  (import = register)
