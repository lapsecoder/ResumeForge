"""Run the full dataset audit and write artefacts.

Usage (from ``backend/``):

    python -m ml.scripts.run_dataset_audit [--json-out PATH]

This script is offline-only by default and will fail loudly if the pinned
snapshot is not already in the local HF hub cache. Use ``--allow-download``
to permit a pinned network fetch.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from ml.audit.leakage import leak_jobs, leak_resumes, vocab_overlap
from ml.audit.quality import quality_diagnose
from ml.audit.report import render_plots, write_report
from ml.audit.splits import split_strategy_recommendation
from ml.audit.statistics import cross_section, summarize_jobs, summarize_resumes
from ml.audit.target import derivability, resume_exposure, summarize_matches
from ml.config import DATASET_ID, DATASET_REVISION
from ml.loader import DatasetUnavailableError, load_bundle
from ml.models.baselines import BASELINES


def _run(*, allow_download: bool) -> dict[str, Any]:
    bundle = load_bundle(allow_download=allow_download)

    payload: dict[str, Any] = {
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "frame_counts": bundle.frame_counts,
        },
        "statistics": {
            "resumes": summarize_resumes(bundle.resumes),
            "jobs": summarize_jobs(bundle.jobs),
            "cross_section": cross_section(bundle.resumes, bundle.jobs),
        },
        "target": summarize_matches(bundle.matches),
        "exposure": resume_exposure(bundle.matches),
        "derivability": derivability(
            bundle.resumes, bundle.jobs, bundle.matches
        ),
        "leakage": {
            "resumes": leak_resumes(bundle.resumes),
            "jobs": leak_jobs(bundle.jobs),
            "vocab": vocab_overlap(bundle.resumes, bundle.jobs),
        },
        "quality": quality_diagnose(bundle.resumes, bundle.jobs, bundle.matches),
        "split_strategy": split_strategy_recommendation(
            bundle.resumes, bundle.jobs, bundle.matches
        ),
        "baselines": BASELINES,
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-download", action="store_true", default=False)
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="skip matplotlib render (faster; CI-safe)",
    )
    args = parser.parse_args(argv)

    try:
        payload = _run(allow_download=args.allow_download)
    except DatasetUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    artefacts = write_report(payload)
    if not args.skip_plots:
        plots = render_plots(payload)
        print("plots written:")
        for path in plots:
            print(f"  {path}")
    print("artefacts written:")
    for kind, path in artefacts.items():
        print(f"  {kind}: {path}")

    echoes = payload["derivability"]
    cap_cut = echoes["fraction_candidate_pairs_discarded_by_cap"]
    print(
        f"summary: {echoes['published_relevant_pairs']} pairs; "
        f"{echoes['pct_relevant_published_within_candidate']:.1f}% labels "
        f"reachable by rule; {cap_cut:.1f}% "
        "candidate mass discarded by cap"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
