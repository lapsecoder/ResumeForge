"""Assemble audit artefacts (JSON + Markdown) and EDA plots.

All outputs contain aggregate numbers only. No resume or job text, no PII,
and no per-record data leaves this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.config import ARTIFACTS_DIR, PLOTS_DIR


def write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    return path


def _bullets(items: list[tuple[str, int]]) -> list[str]:
    return [f"  - `{name}`: {count}" for name, count in items]


def markdown_report(payload: dict[str, Any]) -> str:
    stats = payload["statistics"]
    target = payload["target"]
    deriv = payload["derivability"]
    leaks = payload["leakage"]
    quality = payload["quality"]
    split = payload["split_strategy"]

    lines: list[str] = []
    lines.append("# Candidate-Matching Synthetic Dataset — Audit Report")
    lines.append("")
    lines.append(
        f"- Dataset: `{payload['dataset']['id']}` @ "
        f"`{payload['dataset']['revision']}`"
    )
    lines.append(f"- Frames: {payload['dataset']['frame_counts']}")
    lines.append(
        "- Loader: pinned snapshot via `hf_hub_download` + pandas parquet "
        "(`datasets.load_dataset` cannot expose jobs/matches)"
    )
    lines.append("")
    lines.append("## 1. Structure")
    lines.append("")
    rs, js = stats["resumes"], stats["jobs"]
    lines.append(
        f"- Resumes: {rs['count']} rows, {len(rs['columns'])} columns; "
        f"skills/resume mean {rs['skills_per_resume']['mean']:.1f}"
    )
    lines.append(
        f"- Jobs: {js['count']} rows, {len(js['columns'])} columns; "
        f"must-have/job mean {js['must_have_per_job']['mean']:.1f}"
    )
    lines.append(
        f"- Seniority (resumes): {rs['seniority_counts']}"
    )
    lines.append(
        f"- Seniority (jobs): {js['seniority_counts']}"
    )
    lines.append(
        f"- Years experience: mean {rs['years_experience']['mean']:.1f}, "
        f"range {rs['years_experience']['min']}-{rs['years_experience']['max']}"
    )
    lines.append("")
    lines.append("## 2. Target construction")
    lines.append("")
    lines.append(
        f"- `matches` rows: {target['rows']}, unique jobs {target['unique_jobs']}, "
        f"relevant/job {target['per_job_relevant_count']['distinct_values']}"
    )
    lines.append(
        f"- Reconstructed rule: must-have coverage >= "
        f"{deriv['threshold_must_have']:.0%}, random sample capped at "
        f"{deriv['cap']}"
    )
    lines.append(
        f"- Published pairs: {deriv['published_relevant_pairs']}; "
        f"rule candidates: {deriv['rule_candidate_pairs']}"
    )
    lines.append(
        f"- Relevant labels that are rule-candidates: "
        f"{deriv['pct_relevant_published_within_candidate']:.1f}%"
    )
    lines.append(
        f"- Candidate mass discarded by the 30 cap: "
        f"{deriv['fraction_candidate_pairs_discarded_by_cap']:.1f}%"
    )
    lines.append(
        f"- Jobs with more candidates than the cap: "
        f"{deriv['jobs_with_more_candidates_than_cap']}"
    )
    exposure = payload["exposure"]
    lines.append(
        f"- Distinct resumes ever labeled relevant: "
        f"{exposure['distinct_resumes_used']} / {rs['count']}"
    )
    lines.append("")
    lines.append("## 3. Leakage / syntheticness")
    lines.append("")
    lines.append(
        f"- Summary template match: "
        f"{leaks['resumes']['pct_summary_follows_template']:.1f}%"
    )
    bullet_stats = leaks["resumes"]["experience_bullets"]
    lines.append(
        f"- Bullet strings shared across resumes: "
        f"{bullet_stats['pct_occurrences_of_shared_strings']:.1f}% "
        f"(distinct strings: {bullet_stats['distinct_strings']})"
    )
    lines.append(
        f"- Shared narrative-word Jaccard: "
        f"{leaks['vocab']['shared_narrative_words_fraction']:.1f}%"
    )
    lines.append(
        f"- Job must-have vocabulary covered by resume skills: "
        f"{leaks['vocab']['skills_shared_fraction']:.1f}%"
    )
    lines.append("")
    lines.append("## 4. Quality")
    lines.append("")
    lines.append(f"- Resume PII-like detections: {quality['resumes']}")
    lines.append(f"- Job PII-like detections: {quality['jobs']}")
    lines.append(f"- Matches integrity: {quality['matches']}")
    lines.append("")
    lines.append("## 5. Split strategy")
    lines.append("")
    lines.append(
        "- Repository ships a test split: "
        f"{split['has_repository_test_split']}"
    )
    for rec in split["recommendations"]:
        lines.append(f"  - {rec}")
    lines.append("")
    lines.append("## 6. Top skills")
    lines.append("")
    lines.append("- Top resume skills:")
    lines.extend(_bullets(rs["top_20_skills"][:20]))
    lines.append("- Top job must-have skills:")
    lines.extend(_bullets(js["top_20_must_have"][:20]))
    lines.append("")
    lines.append("## 7. Baselines (defined, not trained)")
    lines.append("")
    for name, description in payload["baselines"].items():
        lines.append(f"- **{name}**: {description}")
    lines.append("")
    return "\n".join(lines)


def write_report(payload: dict[str, Any]) -> dict[str, str]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = write_json(payload, ARTIFACTS_DIR / "audit_report.json")
    md_path = ARTIFACTS_DIR / "audit_report.md"
    md_path.write_text(markdown_report(payload), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def _lazy_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def render_plots(payload: dict[str, Any]) -> list[str]:
    plt = _lazy_pyplot()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    rs, js = payload["statistics"]["resumes"], payload["statistics"]["jobs"]

    top_resume = list(reversed(rs["top_20_skills"][:15]))
    top_job = list(reversed(js["top_20_must_have"][:15]))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([n for n, _ in top_resume], [c for _, c in top_resume], color="#2563eb")
    ax.set_title("Top 15 resume skills")
    ax.set_xlabel("count")
    fig.tight_layout()
    path = PLOTS_DIR / "top_resume_skills.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([n for n, _ in top_job], [c for _, c in top_job], color="#16a34a")
    ax.set_title("Top 15 job must-have skills")
    ax.set_xlabel("count")
    fig.tight_layout()
    path = PLOTS_DIR / "top_job_must_have.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    resume_sen = rs["seniority_counts"]
    job_sen = js["seniority_counts"]
    labels = sorted(set(resume_sen) | set(job_sen))
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.38
    ax.bar([i - width / 2 for i in x],
           [resume_sen.get(label, 0) for label in labels],
           width, label="resumes", color="#2563eb")
    ax.bar([i + width / 2 for i in x],
           [job_sen.get(label, 0) for label in labels],
           width, label="jobs", color="#16a34a")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title("Seniority supply vs demand")
    ax.legend()
    fig.tight_layout()
    path = PLOTS_DIR / "seniority_supply_demand.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    hist = payload["exposure"]["histogram_buckets"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(list(hist.keys()), list(hist.values()), color="#7c3aed")
    ax.set_title("Resumes by number of relevant jobs")
    ax.set_ylabel("resume count")
    fig.tight_layout()
    path = PLOTS_DIR / "resume_exposure.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    return written
