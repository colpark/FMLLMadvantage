# fmllm.fms.fm1_image

FM1: a small Vision Transformer with a DETR-style set-prediction head
that maps a 64x64 grayscale image to an atom count and a set of
``(x, y)`` atom positions in LJ units.

## Files

- `model.py` - `FM1ImageViT` model and `build_fm1_model(cfg)`
  constructor. Conv-based patch embedding feeds a Transformer encoder
  on patch tokens, then learned object queries cross-attend to the
  encoded patches via a Transformer decoder. The CLS token feeds the
  count head, each query feeds position and confidence heads.
- `train.py` - the training script. Combines a count cross-entropy,
  Hungarian-matched L2 position loss, BCE objectness loss, and a soft
  box-constraint loss. Uses AdamW + linear warmup + cosine decay,
  optional mixed precision, and validates every epoch.
- `conformal.py` - split-conformal calibration. Computes per-pair L2
  position errors on the calibration subset and fits thresholds for
  every alpha in `conformal_alpha_levels`.

## Translation equivariance

The conv-based patch embedding gives exact translation equivariance
when the input shifts by integer multiples of `patch_size` pixels.
Learned absolute positional embeddings break exact equivariance for
non-multiples; the patch-grid prior plus convolutional feature
aggregation supplies the inductive bias that the Hungarian-matched
position loss exploits during training. The model emits absolute LJ
coordinates, so equivariance manifests as: a translation of the input
image by `(dx_px, dy_px)` shifts the predicted positions by
`(dx_px, dy_px) * pixel_size_lj` in expectation.
