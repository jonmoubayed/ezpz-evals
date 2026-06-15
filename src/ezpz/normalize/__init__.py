"""Central, type-aware value normalization. Runs on BOTH predictions and ground truth before
scoring, so every tool is canonicalized identically. Deliberately not per-adapter."""
from ezpz.normalize.canonical import normalize_value, normalize_fields  # noqa: F401
