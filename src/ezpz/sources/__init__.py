"""Document sources: the input-side swappable layer — *where documents come from*.

A source is the sibling of an adapter: an adapter is the tool under test (output side), a source is
the cohort's origin (input side). Both register by name and both keep their SDK calls lazy, so the
registry lists every source even when an optional extra isn't installed — only an actual load needs
the SDK + credentials.

Built-ins: ``local`` (manifest + files on disk; the default), ``s3``, ``langfuse``, ``extend``.
Third parties add more via the ``ezpz.sources`` entry-point group (see ``ezpz.plugins``).
"""
from ezpz.sources import (  # noqa: F401  (import = register)
    extend,
    langfuse,
    local,
    s3,
)
