"""Literature verifier source: lookup against a curated cluster database.

The database lives at ``data/literature/clusters.json`` and is a list
of canonical 2D Lennard-Jones clusters with their reference per-atom
potential energies and first-peak g(r) positions. The source matches
the claim and the bridged outputs against the closest entry by atom
count and motif, then checks whether the FM-derived energy or
temperature aligns with the literature value.

Database entry schema:

    {
        "n_atoms": int,
        "motif": str,                    # canonical motif name
        "per_atom_potential_energy_lj": float,
        "rdf_first_peak_lj": float,
        "diameter_lj": float,
        "reference": str
    }

The source returns:

    - ``PASS`` when the claim's atom count + motif match an entry and
      no comparable FM value disagrees beyond the tolerance.
    - ``CAVEAT`` when an FM value disagrees with the literature value.
    - ``SKIP`` when the database has no matching entry.
    - ``FAIL`` is reserved for future strict-match modes; the current
      source never escalates to FAIL.

Depends on:
    json (stdlib).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fmllm.fms._schemas import BridgedFMOutput
from fmllm.verifier.schema import (
    PhysicalStateClaim,
    SourceDecision,
    SourceVerdict,
)


class LiteratureSource:
    """Verifier source backed by a curated cluster database."""

    name = "literature"

    def __init__(
        self,
        db_path: Path | str,
        *,
        energy_tolerance: float = 0.20,
    ) -> None:
        self.db_path = Path(db_path)
        self.energy_tolerance = energy_tolerance
        self._db: list[dict[str, Any]] = []
        if self.db_path.exists():
            with self.db_path.open("r") as f:
                self._db = json.load(f)

    @property
    def database(self) -> list[dict[str, Any]]:
        return self._db

    def _match_entry(self, n_atoms: int, motif: str | None) -> dict[str, Any] | None:
        if not self._db:
            return None
        # Prefer exact (N, motif) match; fall back to (N, any motif).
        if motif is not None:
            for e in self._db:
                if int(e["n_atoms"]) == n_atoms and e["motif"] == motif:
                    return e
        candidates = [e for e in self._db if int(e["n_atoms"]) == n_atoms]
        if candidates:
            # Pick the lowest-energy reference for the size.
            return min(candidates, key=lambda e: float(e["per_atom_potential_energy_lj"]))
        return None

    def check(
        self,
        bridged_outputs: list[BridgedFMOutput],
        claim: PhysicalStateClaim,
    ) -> SourceVerdict:
        # Decide which atom count to look up: the claim, or FM1's derived value.
        n_atoms = claim.n_atoms
        if n_atoms is None:
            for bo in bridged_outputs:
                if bo.source.fm_name == "fm1_image":
                    for d in bo.dependencies:
                        if d.target_variable == "atom_count" and d.derived_value is not None:
                            try:
                                n_atoms = int(d.derived_value)
                                break
                            except (TypeError, ValueError):
                                continue
                    break

        if n_atoms is None:
            return SourceVerdict(
                source_name=self.name,
                decision=SourceDecision.SKIP,
                confidence=0.0,
                message="no atom count available from claim or FM1",
                evidence={"db_size": len(self._db)},
            )

        entry = self._match_entry(int(n_atoms), claim.motif)
        if entry is None:
            return SourceVerdict(
                source_name=self.name,
                decision=SourceDecision.SKIP,
                confidence=0.0,
                message=f"no literature entry for N={n_atoms}, motif={claim.motif}",
                evidence={"db_size": len(self._db)},
            )

        # Compare FM2 / claim energy against the reference.
        ref_e = float(entry["per_atom_potential_energy_lj"])
        candidate_energy: float | None = None
        candidate_source: str | None = None
        for bo in bridged_outputs:
            if bo.source.fm_name == "fm2_rdf":
                v = (bo.prediction.value or {}).get("value_lj")
                if v is not None:
                    candidate_energy = float(v)
                    candidate_source = "fm2_rdf"
                    break
        if candidate_energy is None and claim.per_atom_potential_energy is not None:
            candidate_energy = float(claim.per_atom_potential_energy)
            candidate_source = "claim"

        evidence: dict[str, Any] = {
            "n_atoms": int(n_atoms),
            "motif": entry["motif"],
            "reference": entry,
            "candidate_energy": candidate_energy,
            "candidate_source": candidate_source,
        }

        if candidate_energy is None:
            return SourceVerdict(
                source_name=self.name,
                decision=SourceDecision.PASS,
                confidence=0.7,
                message=(
                    f"matched literature entry N={n_atoms} motif={entry['motif']}; "
                    "no candidate energy to compare"
                ),
                evidence=evidence,
            )

        diff = abs(candidate_energy - ref_e)
        if diff <= self.energy_tolerance:
            return SourceVerdict(
                source_name=self.name,
                decision=SourceDecision.PASS,
                confidence=max(0.5, 1.0 - diff / max(self.energy_tolerance, 1.0e-6)),
                message=(
                    f"energy {candidate_energy:.3f} agrees with literature "
                    f"{ref_e:.3f} (diff {diff:.3f} <= {self.energy_tolerance:.2f})"
                ),
                evidence=evidence,
            )

        return SourceVerdict(
            source_name=self.name,
            decision=SourceDecision.CAVEAT,
            confidence=max(0.0, 1.0 - diff / max(self.energy_tolerance, 1.0e-6)),
            message=(
                f"energy {candidate_energy:.3f} differs from literature "
                f"{ref_e:.3f} by {diff:.3f} (tolerance {self.energy_tolerance:.2f})"
            ),
            evidence=evidence,
        )


__all__ = ["LiteratureSource"]
