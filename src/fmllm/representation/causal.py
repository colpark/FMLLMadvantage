"""Counterfactual causal interventions on SAE features.

Phase 14. The Phase 13 Top-K SAE produces a sparse latent ``z`` of FM2's
CLS embedding plus a (lossy) decoder back to the embedding. The
correlation labels Stage 1 emits are *descriptive* -- they say which
attributes co-occur with each feature on the labelling set. They do not
say whether the feature has a *causal* effect on FM2's downstream
prediction (per-atom energy).

This module supplies the causal handle. For one feature index ``i``,

  1. forward the specimen through FM2 to get the CLS embedding,
  2. encode the CLS through the SAE to get ``z``,
  3. apply an intervention to ``z`` (zero feature ``i``, clamp it high,
     or set a fixed value),
  4. decode the intervened ``z`` back to a reconstructed CLS,
  5. apply FM2's energy head to the reconstructed CLS,

and compare the predicted energy against the same pipeline with no
intervention. The energy delta is the local causal effect of feature
``i`` on FM2's prediction at that specimen.

Two baselines:

* ``original_energy``  -- ``energy_head(cls_original)``. Mixes the SAE
  reconstruction loss with any intervention effect.
* ``recon_energy`` -- ``energy_head(decode(encode(cls_original)))``.
  Removes the reconstruction-loss component; the difference against
  the intervened prediction isolates the causal effect of the
  intervention itself.

We always report both, with ``causal_effect = intervened - recon``
being the cleanest signal.

Depends on:
    numpy, torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch
from torch import Tensor, nn


class InterventionKind(str, Enum):
    """The supported intervention modes."""

    KNOCK_OUT = "knock_out"          # set z[i] = 0
    KNOCK_IN = "knock_in"            # set z[i] = `value` even if it was 0
    CLAMP = "clamp"                  # set z[i] = `value` regardless of prior


@dataclass
class Intervention:
    """One unit of intervention specification."""

    kind: InterventionKind
    feature_idx: int
    value: float = 0.0       # only used for KNOCK_IN / CLAMP

    def apply(self, z: Tensor) -> Tensor:
        """Return a new tensor with the intervention applied.

        ``z`` is shape ``(B, hidden_dim)``. The returned tensor has the
        same shape; only column ``feature_idx`` is modified.
        """
        if z.dim() != 2:
            raise ValueError(f"expected (B, hidden_dim) z, got {tuple(z.shape)}")
        z_new = z.clone()
        if self.kind is InterventionKind.KNOCK_OUT:
            z_new[:, self.feature_idx] = 0.0
        elif self.kind is InterventionKind.KNOCK_IN:
            z_new[:, self.feature_idx] = float(self.value)
        elif self.kind is InterventionKind.CLAMP:
            z_new[:, self.feature_idx] = float(self.value)
        else:                                    # pragma: no cover, exhaustive
            raise ValueError(f"unknown intervention {self.kind}")
        return z_new


@dataclass
class CausalEffect:
    """Per-feature summary of intervention -> energy effect."""

    feature_idx: int
    label: str
    n_specimens: int
    activation_rate: float                   # fraction with z[:,i] > 0

    # Mean predicted energies under each pipeline path. All on the
    # same set of N specimens, so paired comparisons are valid.
    energy_original_mean: float              # E[energy_head(cls)]
    energy_recon_mean: float                 # E[energy_head(decode(encode(cls)))]
    energy_knock_out_mean: float             # E[energy_head(decode(z with z[:,i]=0))]
    energy_knock_in_mean: float              # E[energy_head(decode(z with z[:,i]=v_high))]

    # Causal effect = intervened - recon. Signed: negative means
    # knocking the feature out lowers the predicted energy.
    knock_out_effect: float
    knock_in_effect: float

    # |knock_out_effect| / std(energy_recon). Per-feature signal-to-
    # noise; > 1.0 is a strong handle, < 0.1 is essentially decorative.
    knock_out_effect_norm: float
    knock_in_effect_norm: float

    # Standard deviation of intervened energy across specimens; used
    # by callers to know whether the average effect hides per-specimen
    # heterogeneity.
    knock_out_effect_std: float = 0.0
    knock_in_effect_std: float = 0.0

    extra: dict = field(default_factory=dict)


def normalize_cls(
    cls: Tensor, mean: Tensor, std: Tensor,
) -> Tensor:
    """Apply the SAE's per-feature CLS normalization."""
    return (cls - mean) / std.clamp_min(1.0e-6)


def denormalize_cls(
    cls_norm: Tensor, mean: Tensor, std: Tensor,
) -> Tensor:
    """Invert :func:`normalize_cls`."""
    return cls_norm * std + mean


def cls_through_sae(
    *,
    sae: nn.Module,
    cls_norm: Tensor,
    intervention: Intervention | None = None,
) -> Tensor:
    """Encode CLS, optionally intervene, decode back to a CLS-shape tensor.

    Args:
        sae: A :class:`fmllm.representation.sae.TopKSAE`.
        cls_norm: Already-normalized CLS, shape ``(B, in_dim)``.
        intervention: Optional intervention applied to ``z`` before
            decoding.

    Returns:
        Reconstructed CLS in normalized space, shape ``(B, in_dim)``.
    """
    z = sae.encode(cls_norm)                 # (B, hidden_dim)
    if intervention is not None:
        z = intervention.apply(z)
    return sae.decode(z)


def predict_energy(
    *,
    energy_head: nn.Module,
    cls: Tensor,
) -> Tensor:
    """Apply FM2's energy head to a CLS-shape tensor.

    The head expects the un-normalized CLS. Callers that intervened in
    SAE-space should denormalize before calling this.

    Returns:
        ``(B,)`` tensor of per-atom energies.
    """
    out = energy_head(cls)
    if out.dim() == 2 and out.shape[-1] == 1:
        out = out.squeeze(-1)
    return out


def audit_feature(
    *,
    sae: nn.Module,
    energy_head: nn.Module,
    cls_original: Tensor,             # (B, in_dim) un-normalized
    cls_mean: Tensor,                 # (in_dim,)
    cls_std: Tensor,                  # (in_dim,)
    feature_idx: int,
    label: str,
    knock_in_value: float | None = None,
) -> CausalEffect:
    """Run knock-out and knock-in interventions and summarize the effect.

    Args:
        sae: trained Top-K SAE.
        energy_head: FM2's energy head ``nn.Module`` (acts on CLS).
        cls_original: un-normalized CLS embeddings on the audit set.
        cls_mean, cls_std: SAE's saved normalization statistics.
        feature_idx: which SAE feature to intervene on.
        label: human-readable label for the feature (informational).
        knock_in_value: what value to clamp the feature to in the
            knock-in test. Default: 99th percentile of nonzero
            activations on the audit set, so we test "this feature is
            *strongly* active even when it normally would not be."

    Returns:
        A :class:`CausalEffect` record.
    """
    if cls_original.dim() != 2:
        raise ValueError(f"expected (B, in_dim) cls, got {tuple(cls_original.shape)}")
    n = int(cls_original.shape[0])
    if n == 0:
        raise ValueError("cls_original is empty")

    cls_norm = normalize_cls(cls_original, cls_mean, cls_std)

    with torch.no_grad():
        # Baseline activations to compute the activation rate and a
        # sensible knock-in value if not supplied.
        z_baseline = sae.encode(cls_norm)
        col = z_baseline[:, feature_idx]
        nonzero_mask = col > 0
        activation_rate = float(nonzero_mask.float().mean().item())
        if knock_in_value is None:
            nz = col[nonzero_mask]
            knock_in_value = (
                float(torch.quantile(nz, 0.99).item()) if nz.numel() > 0 else 1.0
            )

        # Three forward paths.
        cls_recon_norm = cls_through_sae(sae=sae, cls_norm=cls_norm)
        cls_ko_norm = cls_through_sae(
            sae=sae, cls_norm=cls_norm,
            intervention=Intervention(InterventionKind.KNOCK_OUT, feature_idx),
        )
        cls_ki_norm = cls_through_sae(
            sae=sae, cls_norm=cls_norm,
            intervention=Intervention(
                InterventionKind.KNOCK_IN, feature_idx, knock_in_value,
            ),
        )

        cls_recon = denormalize_cls(cls_recon_norm, cls_mean, cls_std)
        cls_ko = denormalize_cls(cls_ko_norm, cls_mean, cls_std)
        cls_ki = denormalize_cls(cls_ki_norm, cls_mean, cls_std)

        e_orig = predict_energy(energy_head=energy_head, cls=cls_original)
        e_recon = predict_energy(energy_head=energy_head, cls=cls_recon)
        e_ko = predict_energy(energy_head=energy_head, cls=cls_ko)
        e_ki = predict_energy(energy_head=energy_head, cls=cls_ki)

    e_orig_np = e_orig.detach().cpu().numpy().astype(np.float64)
    e_recon_np = e_recon.detach().cpu().numpy().astype(np.float64)
    e_ko_np = e_ko.detach().cpu().numpy().astype(np.float64)
    e_ki_np = e_ki.detach().cpu().numpy().astype(np.float64)

    # Per-specimen effects, then mean and std.
    ko_eff = e_ko_np - e_recon_np
    ki_eff = e_ki_np - e_recon_np
    ko_eff_mean = float(ko_eff.mean())
    ki_eff_mean = float(ki_eff.mean())
    ko_eff_std = float(ko_eff.std())
    ki_eff_std = float(ki_eff.std())

    # Normalize against the inter-specimen energy spread so the score
    # is comparable across features.
    ref_std = float(e_recon_np.std())
    ref_std = ref_std if ref_std > 1.0e-9 else 1.0
    ko_eff_norm = abs(ko_eff_mean) / ref_std
    ki_eff_norm = abs(ki_eff_mean) / ref_std

    return CausalEffect(
        feature_idx=feature_idx,
        label=label,
        n_specimens=n,
        activation_rate=activation_rate,
        energy_original_mean=float(e_orig_np.mean()),
        energy_recon_mean=float(e_recon_np.mean()),
        energy_knock_out_mean=float(e_ko_np.mean()),
        energy_knock_in_mean=float(e_ki_np.mean()),
        knock_out_effect=ko_eff_mean,
        knock_in_effect=ki_eff_mean,
        knock_out_effect_norm=ko_eff_norm,
        knock_in_effect_norm=ki_eff_norm,
        knock_out_effect_std=ko_eff_std,
        knock_in_effect_std=ki_eff_std,
        extra={
            "knock_in_value": float(knock_in_value),
            "ref_std_recon_energy": ref_std,
        },
    )


def filter_features_by_causal_effect(
    *,
    effects: list[CausalEffect],
    min_norm_effect: float = 0.10,
    require_activation: bool = True,
    min_activation_rate: float = 0.005,
) -> list[int]:
    """Return the subset of feature indices that pass the causal gate.

    A feature passes if either its knock-out or its knock-in effect,
    normalized by the inter-specimen energy spread, exceeds
    ``min_norm_effect``. Optionally also require the feature to be
    active on at least ``min_activation_rate`` of the audit set, so we
    drop dead features that the SAE never uses.
    """
    out: list[int] = []
    for eff in effects:
        if require_activation and eff.activation_rate < min_activation_rate:
            continue
        if max(eff.knock_out_effect_norm, eff.knock_in_effect_norm) >= min_norm_effect:
            out.append(eff.feature_idx)
    return out


__all__ = [
    "CausalEffect",
    "Intervention",
    "InterventionKind",
    "audit_feature",
    "cls_through_sae",
    "denormalize_cls",
    "filter_features_by_causal_effect",
    "normalize_cls",
    "predict_energy",
]
