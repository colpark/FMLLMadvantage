"""Literature verifier source: lookup against a curated cluster database.

The database lives at ``data/literature/clusters.json`` and is a list
of canonical 2D Lennard-Jones clusters with their reference per-atom
potential energies and first-peak g(r) positions. The source matches
the claim and the bridged outputs against the closest entry by atom
count and motif.

Database entry schema:

    {
        "n_atoms": int,
        "motif": str,                    # canonical motif name
        "per_atom_potential_energy_lj": float,
        "rdf_first_peak_lj": float,
        "diameter_lj": float,
        "reference": str
    }

Decisions:

    - ``PASS`` when the claim's atom count matches an entry in the DB.
      The matched entry is returned in ``evidence`` for downstream
      use (e.g. the LLM can read motif and reference structural
      properties).
    - ``CAVEAT`` when ``compare_energy=True`` AND a candidate FM-derived
      energy disagrees with the literature reference beyond
      ``energy_tolerance``.
    - ``SKIP`` when no atom count is available or no entry matches.

The energy comparison is disabled by default (``compare_energy=False``).
The reason: the literature DB carries ground-state cluster energies
(T → 0 limit), but the testbed's specimens sit at finite temperature
(typically T ∈ [0.1, 2.0]). FM2 reports the actual finite-T per-atom
potential energy, which is systematically higher than the ground-state
reference by an amount that scales with T. Comparing the two and
flagging the disagreement produces a near-constant CAVEAT signal that
is uninformative as a confidence ranking — observed in the Phase 8a
baseline run (165/200 commits flagged CAVEAT, 87% calibrated
abstention but no useful filtering of correct vs incorrect commits).

Re-enable ``compare_energy=True`` when:
  - the literature DB grows a temperature-resolved energy axis, OR
  - the verifier subtracts a kT correction to the ground-state value
    before comparing.

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
    """Verifier source backed by a curated cluster database.

    Args:
        db_path: Path to ``clusters.json``.
        compare_energy: When True, raise ``CAVEAT`` if a candidate
            energy disagrees with the literature reference beyond
            ``energy_tolerance``. Default False because the reference
            energies are ground-state (T → 0) and the testbed runs
            at finite temperature. Re-enable only after the DB
            grows a T-resolved axis.
        energy_tolerance: Absolute LJ-units tolerance for the
            energy CAVEAT (only consulted when ``compare_energy`` is True).
    """

    name = "literature"

    def __init__(
        self,
        db_path: Path | str,
        *,
        compare_energy: bool = False,
        energy_tolerance: float = 0.20,
    ) -> None:
        self.db_path = Path(db_path)
        self.compare_energy = compare_energy
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
        if motif is not None:
            for e in self._db:
                if int(e["n_atoms"]) == n_atoms and e["motif"] == motif:
                    return e
        candidates = [e for e in self._db if int(e["n_atoms"]) == n_atoms]
        if candidates:
            return min(candidates, key=lambda e: float(e["per_atom_potential_energy_lj"]))
        return None

    def check(
        self,
        bridged_outputs: list[BridgedFMOutput],
        claim: PhysicalStateClaim,
    ) -> SourceVerdict:
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

        # Pull a candidate energy if any FM or the claim provides one;
        # used for evidence in every branch and (optionally) for the
        # CAVEAT comparison when compare_energy is enabled.
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
            "compare_energy": self.compare_energy,
        }

        if not self.compare_energy:
            return SourceVerdict(
                source_name=self.name,
                decision=SourceDecision.PASS,
                confidence=0.8,
                message=(
                    f"matched literature entry N={n_atoms} "
                    f"motif={entry['motif']}; energy comparison disabled "
                    "(reference is ground-state, data is finite-T)"
                ),
                evidence=evidence,
            )

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

        ref_e = float(entry["per_atom_potential_energy_lj"])
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
