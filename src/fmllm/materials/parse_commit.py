"""Lenient parser for the Final-commit JSON in materials CoT outputs.

The strict parser used originally searched for the literal string
``"Final commit:"`` and balanced-brace-extracted the JSON after it.
In practice the SFT-tuned LoRA emits several variants:

  * ``Final commit: {...}``        (canonical)
  * ``Final commit : {...}``       (space before colon)
  * ``Final answer: {...}``        (synonym occasionally produced)
  * just ``{...}`` at end of CoT   (no tag)

Plus minor JSON quirks (Python-style booleans, trailing commas).

This module implements a layered parser:

  1. Try each tag variant via case-insensitive regex; extract the
     first balanced JSON object after the matched tag.
  2. If no tag is found, scan the entire text for any balanced
     JSON object whose keys look like a materials claim.
  3. Apply small JSON repair (Python booleans, trailing commas)
     before json.loads as a final fallback.

A parsed object is accepted only if it has at least 3 of the 5
expected claim keys, so we don't accidentally pick up a probe
output dict that happens to be balanced.

Used by:
  * scripts/materials/09_run_singleshot.py   (live parsing)
  * scripts/materials/repair_parse_failures.py (post-hoc rescue)
  * scripts/materials/inspect_parse_failures.py (categorization)

Depends on:
    Stdlib only.
"""

from __future__ import annotations

import json
import re
from typing import Any


_EXPECTED_KEYS = frozenset({
    "formation_energy", "e_above_hull", "is_stable",
    "band_gap_class", "space_group",
})


# Order matters: prefer "Final commit", then "Final answer".
_TAG_PATTERNS = (
    re.compile(r"final\s+commit\s*:?", re.IGNORECASE),
    re.compile(r"final\s+answer\s*:?", re.IGNORECASE),
)


def _looks_like_claim(d: object) -> bool:
    """Accept dict only if it has >=3 of the 5 expected claim keys."""
    if not isinstance(d, dict):
        return False
    return len(set(d.keys()) & _EXPECTED_KEYS) >= 3


def _balanced_object_from(text: str, start: int) -> str | None:
    """Return the substring of the first balanced ``{...}`` object
    starting at or after ``start``, or None if unbalanced/missing.
    """
    brace_start = text.find("{", start)
    if brace_start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(brace_start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start : i + 1]
    return None


def _all_balanced_objects(text: str) -> list[str]:
    """Find every balanced ``{...}`` object in the text in order."""
    out: list[str] = []
    cursor = 0
    while True:
        obj = _balanced_object_from(text, cursor)
        if obj is None:
            break
        # Advance past this object before searching for the next.
        idx = text.index(obj, cursor) + len(obj)
        out.append(obj)
        cursor = idx
    return out


_PYTHON_BOOL_RE = re.compile(r"\b(True|False|None)\b")
_PY_TO_JSON = {"True": "true", "False": "false", "None": "null"}
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _repair_json_payload(payload: str) -> str:
    """Fix common JSON-format drifts emitted by SFT-tuned LLMs.

    * Python booleans/None -> JSON literals.
    * Trailing commas in objects/arrays -> remove.
    """
    repaired = _PYTHON_BOOL_RE.sub(
        lambda m: _PY_TO_JSON[m.group(1)], payload,
    )
    repaired = _TRAILING_COMMA_RE.sub(r"\1", repaired)
    return repaired


def _try_load(payload: str) -> dict | None:
    """Try strict json.loads, then fall back to repaired payload."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        try:
            data = json.loads(_repair_json_payload(payload))
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def parse_final_commit(text: str) -> dict[str, Any] | None:
    """Lenient Final-commit JSON extraction.

    Returns the parsed claim dict, or None if no recoverable claim
    is present in ``text``.
    """
    if not text:
        return None

    # 1. Tag-anchored search.
    for pat in _TAG_PATTERNS:
        for m in pat.finditer(text):
            obj_str = _balanced_object_from(text, m.end())
            if obj_str is None:
                continue
            data = _try_load(obj_str)
            if _looks_like_claim(data):
                return data

    # 2. Untagged fallback: scan all balanced objects, return the
    #    LAST one that looks like a materials claim. The last is
    #    chosen because the user message contains an example claim
    #    schema that could otherwise match.
    candidates = _all_balanced_objects(text)
    for obj_str in reversed(candidates):
        data = _try_load(obj_str)
        if _looks_like_claim(data):
            return data

    return None


__all__ = ["parse_final_commit"]
