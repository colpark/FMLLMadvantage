# fmllm.fms.fm1_image.probes

Behavioral probes for FM1 (image -> atom positions). Each probe
exposes ``run_probe(model, items, device, config) -> ProbeResult`` and
tests one constraint declared in `metadata.yaml`.

## Files

- `translation_equivariance.py` - shifts each test image by
  `patch_size` pixels and measures whether predicted positions shift
  by the same amount in LJ units.
- `atom_count_consistency.py` - confirms the count head's argmax
  agrees with the number of confidence-thresholded query slots.
- `positions_in_box.py` - confirms confident predicted positions lie
  inside the imaging box.

The probe runner in `fmllm.fms.probe_runner` imports each module by
the dotted path declared in `metadata.yaml`.
