# fmllm.fms.fm2_rdf

FM2: a 1D Transformer that maps the radial distribution function
``g(r)`` to the coarse-grained per-atom potential energy.

## Files

- `model.py` - `FM2RDFTransformer` model and `build_fm2_model(cfg)`
  constructor. A linear bin embedding plus learned positional
  embeddings feed a Transformer encoder with a CLS token. The CLS
  output passes through an MLP energy head.
- `train.py` - the training script. Huber loss against per-atom
  potential energy plus a soft non-negativity penalty against the
  cluster's LJ energy floor.
- `conformal.py` - split-conformal calibration on absolute residuals.

## Symmetries

- **Permutation invariance** is automatic. ``g(r)`` depends only on
  the multi-set of pairwise distances, so the model's input does not
  see atom order.
- **Extensive scaling** holds by output design. The model predicts
  per-atom energy. Total energy = ``N * per_atom_energy`` with no
  additional loss term needed.
