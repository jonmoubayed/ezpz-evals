"""Adapters: the swappable layer. Import a tool's module to register its pipeline.
Provider SDK calls live ONLY inside these modules so SDK churn stays contained.

All adapter classes register on import. Their provider SDKs are imported LAZILY (inside the
call path), so the registry is populated and `validate`/`list` work even when an optional extra
isn't installed — only an actual run needs the SDK + API key."""
from ezpz.adapters import (  # noqa: F401  (import = register)
    anthropic,
    extend,
    fake,
    gemini,
    llamaindex,
    openai,
)
from ezpz import strategies  # noqa: F401,E402  (composite strategies are adapters too)
