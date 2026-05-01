"""Generate unique run identifiers.

This module produces run IDs of the form ``YYYYMMDD-HHMMSS-<slug>``,
where the slug names what the run does. The IDs sort lexically by
time, so they double as directory names under ``runs/``.

Produces:
    String identifiers like ``20260315-141522-fm1-train-baseline``.

Depends on:
    Standard library only.
"""

from __future__ import annotations

import re
from datetime import datetime

_SLUG_RE = re.compile(r"[^a-z0-9-]+")


def generate_run_id(slug: str, *, now: datetime | None = None) -> str:
    """Return a run identifier of the form ``YYYYMMDD-HHMMSS-<slug>``.

    The function lowercases the slug, replaces any run of non-alphanumeric
    characters with a single hyphen, and strips leading and trailing
    hyphens. The slug must contain at least one alphanumeric character.

    Args:
        slug: A short label describing what the run does.
        now: Override the current time. Useful for deterministic tests.

    Returns:
        The run identifier string.

    Raises:
        ValueError: If the slug reduces to an empty string after cleaning.
    """
    if now is None:
        now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    cleaned = _SLUG_RE.sub("-", slug.lower()).strip("-")
    if not cleaned:
        raise ValueError(f"slug {slug!r} produced an empty identifier")
    return f"{timestamp}-{cleaned}"
