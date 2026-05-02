# fmllm.fms.fm2_rdf.probes

Behavioral probes for FM2 (RDF -> per-atom energy).

## Files

- `permutation_invariance.py` - confirms model output is identical
  for repeated forward passes and verifies that g(r) computed from
  different atom orderings is identical (the upstream invariant).
- `extensive_scaling.py` - groups predictions by atom count and
  measures the spread of per-atom energy within each group.
- `non_negativity.py` - confirms predicted per-atom energy stays at
  or above the LJ pair-energy floor.
