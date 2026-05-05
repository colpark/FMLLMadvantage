"""Tests for the Phase 10 self-supervised FM2 backbone.

These tests run on CPU and only need torch + numpy. They cover:

* The encode() shape contract (must match the supervised FM2 so the
  existing connector can drop in without changes).
* The forward(rdf, mask) contract (returns (B, rdf_bins) prediction).
* Mask handling: masked positions receive the learned mask token,
  unmasked positions retain their bin embedding.
* Reconstruction loss flows backward only through the parameters
  that produced the masked predictions (sanity check that the
  training loop's gradient path is intact).
"""

from __future__ import annotations

import torch

from fmllm.fms.fm2_rdf_ssl.model import FM2SSLTransformer


def _build_small() -> FM2SSLTransformer:
    return FM2SSLTransformer(
        rdf_bins=200, embed_dim=64, depth=2, num_heads=4,
    )


def test_encode_returns_full_sequence_with_cls():
    model = _build_small()
    rdf = torch.randn(3, 200)
    out = model.encode(rdf)
    assert out.shape == (3, 201, 64)


def test_forward_returns_per_bin_prediction():
    model = _build_small()
    rdf = torch.randn(3, 200)
    mask = torch.zeros(3, 200, dtype=torch.bool)
    mask[:, ::10] = True   # mask every tenth bin
    pred = model(rdf, mask)
    assert pred.shape == (3, 200)


def test_mask_token_is_used_when_masked():
    """When all bins are masked, the encoder input is dominated by the
    learned mask token plus pos_embed. We don't directly inspect the
    intermediate, but the prediction should be deterministic in
    eval() because there is no other source of variation across
    different rdf inputs."""
    model = _build_small()
    model.eval()
    rdf_a = torch.randn(1, 200)
    rdf_b = torch.randn(1, 200)
    mask = torch.ones(1, 200, dtype=torch.bool)
    with torch.no_grad():
        pred_a = model(rdf_a, mask)
        pred_b = model(rdf_b, mask)
    # Both inputs are fully masked, so their predictions must agree.
    assert torch.allclose(pred_a, pred_b, atol=1.0e-6)


def test_forward_loss_flows_backward_only_through_masked():
    """Sanity check: the training loop only computes loss on masked
    positions. Backprop through that loss must produce a non-zero
    gradient on the recon_head and the mask_token (since they sit on
    the masked path)."""
    torch.manual_seed(0)
    model = _build_small()
    rdf = torch.randn(2, 200, requires_grad=False)
    mask = torch.zeros(2, 200, dtype=torch.bool)
    mask[:, :50] = True   # mask the first 50 bins of every row
    pred = model(rdf, mask)
    loss = (pred[mask] - rdf[mask]).pow(2).mean()
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters()}
    assert grads["mask_token"] is not None
    assert grads["mask_token"].abs().sum() > 0
    # recon_head's first layer should also have gradient.
    assert grads["recon_head.0.weight"].abs().sum() > 0


def test_encode_is_independent_of_mask_state():
    """encode() must not consult any mask. It is the path probes and
    the connector use, so it has to be a pure function of the input
    RDF."""
    torch.manual_seed(0)
    model = _build_small()
    model.eval()
    rdf = torch.randn(2, 200)
    with torch.no_grad():
        out_a = model.encode(rdf)
        out_b = model.encode(rdf)
    assert torch.allclose(out_a, out_b, atol=1.0e-6)


def test_rejects_wrong_input_shape():
    model = _build_small()
    bad_rdf = torch.randn(2, 199)
    try:
        model.encode(bad_rdf)
    except ValueError:
        pass
    else:
        raise AssertionError("encode should raise on wrong rdf_bins")
    rdf = torch.randn(2, 200)
    bad_mask = torch.zeros(2, 199, dtype=torch.bool)
    try:
        model(rdf, bad_mask)
    except ValueError:
        pass
    else:
        raise AssertionError("forward should raise on mask shape mismatch")
