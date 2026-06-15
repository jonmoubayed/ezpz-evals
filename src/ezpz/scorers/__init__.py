"""Scorers operate ONLY on (normalized prediction, normalized ground truth). Import a module
to register its scorer. A task references scorers by name; per-field overrides win.

Importing this package registers the built-in scorers so the registry is populated.
Scorers with network backends (embedding_similarity, llm_judge) import their SDKs lazily, so
registration here never requires an optional extra."""
from ezpz.scorers import (  # noqa: F401  (import = register)
    date_match,
    embedding_similarity,
    exact,
    list_table,
    llm_judge,
    numeric_tolerance,
    presence,
    string_similarity,
)
