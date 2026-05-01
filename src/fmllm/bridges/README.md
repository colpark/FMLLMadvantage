# fmllm.bridges

Bridges that transport FM outputs into LLM-consumable artifacts.

Phase 3 will add:
- `language_anchored.py` - converts FM outputs into natural-language
  captions with numerical values.
- `structure_preserving.py` - packages FM outputs as typed JSON objects
  with explicit units, uncertainty bounds, equivariance markers, and
  references to the physics constraints the FM respects.
- A shared base class that enforces the common contract.

Currently empty.
