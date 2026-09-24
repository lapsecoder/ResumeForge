"""Baseline model definitions for the candidate-matching problem.

Track C will eventually train/evaluate these; Track B only fixes their
contracts so the design stays honest and deterministic-first. None of them run
here, none touch user data, all are offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BASELINES: dict[str, str] = {
    "A: uniform random": "pareto-blind; establishes chance-level precision@30 "
    "and recall ceiling under the 30-cap",
    "B: role-match exact": "requires exact job_title == resume role canonical form; "
    "no embedding, fully deterministic, 33% of labels covered by role match",
    "C: must-have coverage": "count of must-have skills present / len(must-have); "
    "the generative rule, with random tie-break as the notebook does",
    "D: term-frequency overlap": "token-weighting (TF/IDF from the audit-time "
    "vocabulary counts) over skills+bullets; deterministic, no embeddings",
    "E: sentence embedding cosine": "local all-MiniLM-L6-v2 cosine between resume "
    "and job embeddings; the only non-deterministic baseline and the upper "
    "benchmark track",
}


@dataclass(frozen=True)
class BaselineContract:
    """Converged answer format for a candidate-matching baseline."""

    name: str
    score_kind: str
    unit: str
    output_columns: tuple[str, ...] = field(
        default_factory=lambda: ("job_id", "resume_id", "score", "rank")
    )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score_kind": self.score_kind,
            "unit": self.unit,
            "output_columns": list(self.output_columns),
        }


CONTRACTS: list[BaselineContract] = [
    BaselineContract("A", "probability", "uniform"),
    BaselineContract("B", "binary", "match/no-match"),
    BaselineContract("C", "normalised", "0-1 coverage"),
    BaselineContract("D", "normalised", "0-1 overlap"),
    BaselineContract("E", "cosine", "-1..1"),
]
