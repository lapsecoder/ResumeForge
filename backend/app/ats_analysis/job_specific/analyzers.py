"""Deterministic analyzer for Job-Specific ATS Coverage.

Produces:
- per-category scores (``None`` = not applicable → weight redistribution),
- the coverage report,
- the per-term match log,
- findings with the documented rule IDs.

Every score deduction is attributable to an aggregate coverage finding whose
``impact`` equals ``weight * (1 - score/100)``. Individual missing-term
findings are advisory (impact 0). No LLM, no randomness, no persistence, no
external services.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.ats_analysis.job_specific.heuristics import (
    collect_resume_locations,
    extract_phrases,
    has_strong_evidence,
    is_skills_only,
    resolve_term,
)
from app.ats_analysis.job_specific.rules import (
    MAX_PREFERRED_TERMS,
    MAX_REQUIRED_TERMS,
    MAX_TERM_MATCHES,
    WEIGHTS,
)
from app.ats_analysis.job_specific.schemas import (
    CoverageReport,
    CoverageTotals,
    MatchType,
    TermMatch,
    TermOrigin,
)
from app.ats_analysis.schemas import Finding, Severity

if TYPE_CHECKING:
    from app.job_parsing.schemas import JobDescription
    from app.parsing.schemas import Resume


def _f(
    *,
    severity: Severity,
    rule_id: str,
    title: str,
    explanation: str,
    evidence: str = "",
    recommendation: str,
    impact: float = 0.0,
) -> Finding:
    return Finding(
        category="job_specific",
        severity=severity,
        rule_id=rule_id,
        title=title,
        explanation=explanation,
        evidence=evidence[:120],
        recommendation=recommendation,
        impact=max(0.0, round(impact, 1)),
    )


def _dedupe_terms(terms: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        key = term.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(term.strip())
    return out[:limit]


def _severity_for(impact: float, default: Severity) -> Severity:
    if impact <= 0:
        return Severity.INFO
    if impact >= 20:
        return Severity.HIGH
    if impact >= 10:
        return Severity.MEDIUM
    return Severity.LOW if default is Severity.MEDIUM else default


def _coverage_finding(
    rule_id: str,
    label: str,
    matched: int,
    total: int,
    weight: float,
) -> Finding:
    coverage = (matched / total) if total else 0.0
    impact = round(weight * (1.0 - coverage), 1)
    missing = total - matched

    if impact > 0:
        title = f"{label}: {missing} of {total} term(s) missing"
        explanation = (
            f"{matched} of {total} {label.lower()} term(s) were found in the "
            "resume. Missing terminology reduces ATS keyword coverage for "
            "this specific job."
        )
    else:
        title = f"{label}: fully represented"
        explanation = (
            f"All {total} {label.lower()} term(s) from the job description "
            "were found in the resume."
        )

    finding = _f(
        severity=_severity_for(impact, Severity.MEDIUM),
        rule_id=rule_id,
        title=title,
        explanation=explanation,
        evidence=f"{matched}/{total} covered",
        recommendation=(
            "Add the missing terms to the skills section and, where possible, "
            "show them in use within experience bullets or projects."
            if missing
            else "Maintain this level of alignment for the target role."
        ),
        impact=impact,
    )
    return finding


def analyze_job_specific(
    resume: Resume, job: JobDescription
) -> tuple[
    dict[str, float | None], CoverageReport, list[TermMatch], list[Finding]
]:
    """Run the full deterministic coverage analysis for one resume+job pair."""
    locations = collect_resume_locations(resume)

    required_terms = _dedupe_terms(job.required_skills, MAX_REQUIRED_TERMS)
    preferred_terms = _dedupe_terms(job.preferred_skills, MAX_PREFERRED_TERMS)
    phrases = extract_phrases(job)

    term_matches: list[TermMatch] = []
    matched_required = 0
    matched_preferred = 0
    matched_phrases = 0

    strong_matched = 0
    matched_total = 0
    skills_only_count = 0
    skills_only_terms: list[str] = []

    for term in required_terms:
        match_type, locations_hit, evidence = resolve_term(term, locations)
        matched = match_type is not MatchType.ABSENT
        if matched:
            matched_required += 1
            matched_total += 1
            if has_strong_evidence(locations_hit):
                strong_matched += 1
            if is_skills_only(locations_hit):
                skills_only_count += 1
                skills_only_terms.append(term)
        term_matches.append(
            TermMatch(
                term=term,
                origin=TermOrigin.REQUIRED,
                match_type=match_type,
                evidence_locations=locations_hit,
                evidence=evidence,
                explanation=_explanation_for(match_type, term),
            )
        )

    for term in preferred_terms:
        match_type, locations_hit, evidence = resolve_term(term, locations)
        matched = match_type is not MatchType.ABSENT
        if matched:
            matched_preferred += 1
            matched_total += 1
            if has_strong_evidence(locations_hit):
                strong_matched += 1
            if is_skills_only(locations_hit):
                skills_only_count += 1
                skills_only_terms.append(term)
        term_matches.append(
            TermMatch(
                term=term,
                origin=TermOrigin.PREFERRED,
                match_type=match_type,
                evidence_locations=locations_hit,
                evidence=evidence,
                explanation=_explanation_for(match_type, term),
            )
        )

    for phrase in phrases:
        match_type, locations_hit, evidence = resolve_term(phrase, locations)
        matched = match_type is not MatchType.ABSENT
        if matched:
            matched_phrases += 1
            matched_total += 1
            if has_strong_evidence(locations_hit):
                strong_matched += 1
            if is_skills_only(locations_hit):
                skills_only_count += 1
                skills_only_terms.append(phrase)
        term_matches.append(
            TermMatch(
                term=phrase,
                origin=TermOrigin.PHRASE,
                match_type=match_type,
                evidence_locations=locations_hit,
                evidence=evidence,
                explanation=_explanation_for(match_type, phrase),
            )
        )

    term_matches = term_matches[:MAX_TERM_MATCHES]

    total_required = len(required_terms)
    total_preferred = len(preferred_terms)
    total_phrases = len(phrases)

    required_coverage = (
        (matched_required / total_required) if total_required else None
    )
    preferred_coverage = (
        (matched_preferred / total_preferred) if total_preferred else None
    )
    phrase_coverage = (
        (matched_phrases / total_phrases) if total_phrases else None
    )

    denom = total_required + total_preferred
    overall_coverage = (
        ((matched_required + matched_preferred) / denom) if denom else None
    )
    evidence_supported_required = (
        _strong_fraction(required_terms, term_matches)
        if total_required
        else None
    )
    evidence_supported_preferred = (
        _strong_fraction(preferred_terms, term_matches)
        if total_preferred
        else None
    )

    quality_evidence = (
        (strong_matched / matched_total) if matched_total else None
    )

    scores: dict[str, float | None] = {}
    if required_coverage is not None:
        scores["required_coverage"] = round(required_coverage * 100, 1)
    if preferred_coverage is not None:
        scores["preferred_coverage"] = round(preferred_coverage * 100, 1)
    if phrase_coverage is not None:
        scores["phrase_coverage"] = round(phrase_coverage * 100, 1)
    if quality_evidence is not None:
        scores["evidence_representation"] = round(quality_evidence * 100, 1)

    findings: list[Finding] = _build_findings(
        required_terms=required_terms,
        preferred_terms=preferred_terms,
        phrases=phrases,
        term_matches=term_matches,
        matched_required=matched_required,
        total_required=total_required,
        matched_preferred=matched_preferred,
        total_preferred=total_preferred,
        matched_phrases=matched_phrases,
        total_phrases=total_phrases,
        matched_total=matched_total,
        strong_matched=strong_matched,
        skills_only_count=skills_only_count,
        skills_only_terms=skills_only_terms,
    )

    totals = CoverageTotals(
        required=total_required,
        required_matched=matched_required,
        preferred=total_preferred,
        preferred_matched=matched_preferred,
        phrases=total_phrases,
        phrases_matched=matched_phrases,
    )

    coverage = CoverageReport(
        required_coverage=(
            round(required_coverage, 4) if required_coverage is not None else None
        ),
        preferred_coverage=(
            round(preferred_coverage, 4) if preferred_coverage is not None else None
        ),
        overall_coverage=(
            round(overall_coverage, 4) if overall_coverage is not None else None
        ),
        evidence_supported_required=(
            round(evidence_supported_required, 4)
            if evidence_supported_required is not None
            else None
        ),
        evidence_supported_preferred=(
            round(evidence_supported_preferred, 4)
            if evidence_supported_preferred is not None
            else None
        ),
        totals=totals,
    )

    return scores, coverage, term_matches, findings


def _strong_fraction(terms: list[str], term_matches: list[TermMatch]) -> float:
    by_term = {m.term: m for m in term_matches}
    matched = [
        by_term[t]
        for t in terms
        if by_term.get(t) and by_term[t].match_type is not MatchType.ABSENT
    ]
    if not matched:
        return 0.0
    return (
        sum(1 for m in matched if has_strong_evidence(m.evidence_locations))
        / len(matched)
    )


def _explanation_for(match_type: MatchType, term: str) -> str:
    if match_type is MatchType.EXACT:
        return "Found verbatim in a skills or skill-bearing section."
    if match_type is MatchType.NORMALIZED:
        return "Found after canonical normalisation of the resume skills."
    if match_type is MatchType.ALIAS:
        return "Found via a documented abbreviation expansion."
    if match_type is MatchType.PHRASE:
        return "Found as a matching token sequence in resume text."
    return "Not found in any resume section."


def _build_findings(
    *,
    required_terms: list[str],
    preferred_terms: list[str],
    phrases: list[str],
    term_matches: list[TermMatch],
    matched_required: int,
    total_required: int,
    matched_preferred: int,
    total_preferred: int,
    matched_phrases: int,
    total_phrases: int,
    matched_total: int,
    strong_matched: int,
    skills_only_count: int,
    skills_only_terms: list[str],
) -> list[Finding]:
    findings: list[Finding] = []

    by_term = {m.term: m for m in term_matches}

    if total_required == 0 and total_preferred == 0 and total_phrases == 0:
        findings.append(_f(
            severity=Severity.INFO,
            rule_id="ats.job.limited_terminology",
            title="Limited terminology available in the job description",
            explanation=(
                "The job description exposed no required skills, preferred "
                "skills, or candidate phrases. The coverage signal is "
                "'limited evidence' rather than a judgement of fit."
            ),
            recommendation=(
                "Re-run with a more structured job description for a "
                "meaningful coverage signal."
            ),
        ))
        return findings

    for term in required_terms:
        match = by_term[term]
        if match.match_type is MatchType.ABSENT:
            findings.append(_f(
                severity=Severity.MEDIUM,
                rule_id="ats.job.required_term_missing",
                title=f"Required terminology not found: {term}",
                explanation=(
                    "A required term from the job description is not "
                    "explicitly represented anywhere in the resume."
                ),
                evidence=term,
                recommendation=(
                    f"Add '{term}' to the skills section and, if genuine, "
                    "mention it in an experience bullet or project."
                ),
            ))

    for term in preferred_terms:
        match = by_term[term]
        if match.match_type is MatchType.ABSENT:
            findings.append(_f(
                severity=Severity.LOW,
                rule_id="ats.job.preferred_term_missing",
                title=f"Preferred terminology not found: {term}",
                explanation=(
                    "A preferred term from the job description is not "
                    "explicitly represented in the resume."
                ),
                evidence=term,
                recommendation=(
                    "Add the term if applicable; preferred terms carry "
                    "lower weight than required ones."
                ),
            ))

    for phrase in phrases:
        match = by_term[phrase]
        if match.match_type is MatchType.ABSENT:
            findings.append(_f(
                severity=Severity.LOW,
                rule_id="ats.job.phrase_missing",
                title=f"Job phrase not found: {phrase}",
                explanation=(
                    "A meaningful phrase from the job description does not "
                    "appear as a matching token sequence in the resume."
                ),
                evidence=phrase,
                recommendation=(
                    "Mirror the job's own phrasing in bullets or the summary "
                    "where it is genuine."
                ),
            ))

    if total_required:
        findings.append(_coverage_finding(
            "ats.job.required_coverage", "Required terminology",
            matched_required, total_required, WEIGHTS["required_coverage"],
        ))
    if total_preferred:
        findings.append(_coverage_finding(
            "ats.job.preferred_coverage", "Preferred terminology",
            matched_preferred, total_preferred, WEIGHTS["preferred_coverage"],
        ))

    if total_phrases and matched_phrases < total_phrases:
        missing = total_phrases - matched_phrases
        findings.append(_f(
            severity=_severity_for(1.0, Severity.LOW),
            rule_id="ats.job.phrase_coverage",
            title=f"Job-specific phrases: {missing} of {total_phrases} missing",
            explanation=(
                f"{matched_phrases} of {total_phrases} job-specific phrases "
                "were found in the resume. Matching the job's own language "
                "increases keyword alignment."
            ),
            evidence=f"{matched_phrases}/{total_phrases} phrases covered",
            recommendation=(
                "Use the job description's wording in the summary and "
                "achievement bullets where accurate."
            ),
            impact=WEIGHTS["phrase_coverage"] * (1 - (matched_phrases / total_phrases)),
        ))

    if matched_total and strong_matched < matched_total:
        findings.append(_f(
            severity=_severity_for(1.0, Severity.MEDIUM),
            rule_id="ats.job.evidence_representation",
            title="Match evidence is not evenly distributed",
            explanation=(
                f"Only {strong_matched} of {matched_total} matched term(s) "
                "are backed by experience or project sections."
            ),
            evidence=f"{strong_matched}/{matched_total} matched with strong evidence",
            recommendation=(
                "Show key skills inside experience bullets and project "
                "technology lists, not only in a skills section."
            ),
            impact=WEIGHTS["evidence_representation"]
            * (1 - (strong_matched / matched_total) if matched_total else 0),
        ))

    if skills_only_count:
        samples = ", ".join(skills_only_terms[:3])
        findings.append(_f(
            severity=Severity.INFO,
            rule_id="ats.job.skills_only",
            title="Some matched terms appear only in the skills section",
            explanation=(
                f"{skills_only_count} matched term(s) are evidenced only by "
                "the skills list. Skills-only representation is weaker "
                "evidence than experience or project usage."
            ),
            evidence=f"skills-only terms: {samples}",
            recommendation=(
                "For the most job-critical terms, demonstrate them in "
                "experience bullets or projects."
            ),
        ))

    if matched_required and any(
        has_strong_evidence(by_term[t].evidence_locations)
        for t in required_terms
        if by_term.get(t) and by_term[t].match_type is not MatchType.ABSENT
    ):
        findings.append(_f(
            severity=Severity.INFO,
            rule_id="ats.job.evidence_present",
            title="Required terminology has strong supporting evidence",
            explanation=(
                "At least one required term is demonstrated through "
                "experience or project content, which is the strongest "
                "representation signal."
            ),
            recommendation=(
                "Continue demonstrating key skills in context rather than "
                "relying on skills lists alone."
            ),
        ))

    return findings
