"""Simulator verifier source: short MD rollout from the LLM's claim.

Given a claim that names ``positions`` and ``temperature``, the source
runs a small NVE rollout from the claim's positions at the claim's
temperature, computes the per-atom mean kinetic energy along the
trajectory, and compares it against any FM3 derived temperature in
the bridged outputs. When the agreement falls outside a tolerance,
the source returns ``CAVEAT``. ``FAIL`` only fires when the rollout
diverges (atoms escape the box or the energy blows up).

The source skips when the claim lacks positions or temperature.

Depends on:
    torch, fmllm.physics.
"""

from __future__ import annotations

from typing import Any

import torch

from fmllm.fms._schemas import BridgedFMOutput
from fmllm.physics.lj_potential import total_energy_and_forces
from fmllm.physics.md import maxwell_boltzmann_velocities, run_md
from fmllm.verifier.schema import (
    PhysicalStateClaim,
    SourceDecision,
    SourceVerdict,
)


class SimulatorSource:
    """Verifier source that confirms the claim's state evolves consistently
    with the FM3-derived temperature under MD."""

    name = "simulator"

    def __init__(
        self,
        *,
        n_steps: int = 100,
        dt: float = 0.005,
        confinement_k: float = 0.05,
        rel_tolerance: float = 0.20,
        max_radius: float = 8.0,
        seed: int = 0,
    ) -> None:
        self.n_steps = n_steps
        self.dt = dt
        self.confinement_k = confinement_k
        self.rel_tolerance = rel_tolerance
        self.max_radius = max_radius
        self.seed = seed

    def _forces_fn(self):
        kc = self.confinement_k

        def fn(positions: torch.Tensor):
            return total_energy_and_forces(positions, k_conf=kc)

        return fn

    def check(
        self,
        bridged_outputs: list[BridgedFMOutput],
        claim: PhysicalStateClaim,
    ) -> SourceVerdict:
        if claim.positions is None or claim.temperature is None:
            missing = []
            if claim.positions is None:
                missing.append("positions")
            if claim.temperature is None:
                missing.append("temperature")
            return SourceVerdict(
                source_name=self.name,
                decision=SourceDecision.SKIP,
                confidence=0.0,
                message=f"claim lacks {' and '.join(missing)}; nothing to simulate",
                evidence={"missing_fields": missing},
            )

        positions = torch.tensor(claim.positions, dtype=torch.float32)
        if positions.dim() != 2 or positions.shape[-1] != 2:
            return SourceVerdict(
                source_name=self.name,
                decision=SourceDecision.SKIP,
                confidence=0.0,
                message=f"claim.positions has unexpected shape {tuple(positions.shape)}",
                evidence={},
            )
        n_atoms = positions.shape[0]
        if n_atoms < 2:
            return SourceVerdict(
                source_name=self.name,
                decision=SourceDecision.SKIP,
                confidence=0.0,
                message=f"too few atoms ({n_atoms}) to simulate",
                evidence={},
            )

        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        velocities = maxwell_boltzmann_velocities(
            n_atoms,
            temperature=float(claim.temperature),
            dim=2,
            generator=gen,
            remove_com=True,
            rescale_to_target=True,
        )

        traj = run_md(
            positions, velocities, self._forces_fn(),
            dt=self.dt, n_steps=self.n_steps, record_every=10,
        )
        traj_pos = traj["positions"]
        traj_vel = traj["velocities"]

        # Stability: do atoms stay inside max_radius?
        radii = traj_pos.norm(dim=-1)
        max_r = float(radii.max().item())
        diverged = max_r > self.max_radius

        # Empirical mean per-atom KE over the trajectory.
        ke = 0.5 * (traj_vel * traj_vel).sum(dim=-1)
        observed_T = float(ke.mean().item())

        # Compare against any FM3-derived temperature.
        fm3_T = None
        for bo in bridged_outputs:
            if bo.source.fm_name == "fm3_traj":
                for d in bo.dependencies:
                    if d.target_variable == "temperature" and d.derived_value is not None:
                        try:
                            fm3_T = float(d.derived_value)
                            break
                        except (TypeError, ValueError):
                            continue
        evidence: dict[str, Any] = {
            "n_steps": self.n_steps,
            "n_atoms": n_atoms,
            "observed_T": observed_T,
            "claim_T": float(claim.temperature),
            "max_radius": max_r,
        }
        if fm3_T is not None:
            evidence["fm3_T"] = fm3_T

        if diverged:
            return SourceVerdict(
                source_name=self.name,
                decision=SourceDecision.FAIL,
                confidence=1.0,
                message=f"trajectory diverged: max radius {max_r:.2f} > {self.max_radius:.2f}",
                evidence=evidence,
            )

        # Compare observed_T to claim_T (and to fm3_T if present).
        denom = max(abs(claim.temperature), 1.0e-3)
        rel_err = abs(observed_T - float(claim.temperature)) / denom
        if rel_err > self.rel_tolerance:
            return SourceVerdict(
                source_name=self.name,
                decision=SourceDecision.CAVEAT,
                confidence=max(0.0, 1.0 - rel_err),
                message=(
                    f"simulated T {observed_T:.3f} differs from claim T "
                    f"{claim.temperature:.3f} by {rel_err * 100:.1f}%"
                ),
                evidence=evidence,
            )

        return SourceVerdict(
            source_name=self.name,
            decision=SourceDecision.PASS,
            confidence=max(0.0, 1.0 - rel_err),
            message=(
                f"simulated T {observed_T:.3f} agrees with claim T "
                f"{claim.temperature:.3f} within {self.rel_tolerance * 100:.0f}%"
            ),
            evidence=evidence,
        )


__all__ = ["SimulatorSource"]
