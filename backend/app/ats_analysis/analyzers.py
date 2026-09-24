"""Per-category deterministic analyzers for ATS readiness.

Each analyzer returns a tuple ``(score_0_to_100_or_None, list[Finding])``.
A ``None`` score means the category is *not applicable* to the resume; its
weight is redistributed proportionally instead of penalising the resume.

Every score deduction is traceable to a ``Finding`` with a deterministic
``rule_id``. No LLM, no randomness, no persistence, no external services.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.ats_analysis.heuristics import (
    ACTION_VERBS,
    DateVal,
    _date_val_sortable,
    collect_lines,
    computed_word_count,
    first_person_words,
    has_quantitative,
    is_action_start,
    is_vague,
    parse_date_value,
)
from app.ats_analysis.schemas import Finding, Severity

if TYPE_CHECKING:
    from app.parsing.schemas import Resume


# ---------------------------------------------------------------------------
# Internal deduction tracker
# ---------------------------------------------------------------------------


def _f(
    *,
    category: str,
    severity: Severity,
    rule_id: str,
    title: str,
    explanation: str,
    evidence: str = "",
    recommendation: str,
    impact: float = 0.0,
) -> Finding:
    return Finding(
        category=category,
        severity=severity,
        rule_id=rule_id,
        title=title,
        explanation=explanation,
        evidence=evidence[:120],
        recommendation=recommendation,
        impact=max(0.0, impact),
    )


_EVIDENCE_TRUNC = 120


def _trunc(text: str) -> str:
    return text[:_EVIDENCE_TRUNC]


# ---------------------------------------------------------------------------
# 1. Contact completeness
# ---------------------------------------------------------------------------


def analyze_contact(resume: Resume) -> tuple[float | None, list[Finding]]:
    score = 100.0
    findings: list[Finding] = []
    c = resume.contact

    if not c.name:
        findings.append(_f(
            category="contact", severity=Severity.HIGH,
            rule_id="ats.contact.name",
            title="No name detected",
            explanation=(
                "A candidate name was not found in the header region. "
                "Most ATS and recruiter workflows expect a name."
            ),
            recommendation=(
                "Add a full name at the top of the resume."
            ),
            impact=30.0,
        ))
        score -= 30.0
    if not c.email:
        findings.append(_f(
            category="contact", severity=Severity.HIGH,
            rule_id="ats.contact.email",
            title="No email address detected",
            explanation=(
                "An email address is the most common professional contact "
                "method and is expected on virtually all resumes."
            ),
            recommendation="Add a professional email address.",
            impact=30.0,
        ))
        score -= 30.0
    if not c.phone:
        findings.append(_f(
            category="contact", severity=Severity.LOW,
            rule_id="ats.contact.phone",
            title="No phone number detected",
            explanation=(
                "A phone number is not strictly required but is "
                "commonly expected by recruiters."
            ),
            recommendation="Consider adding a phone number.",
            impact=8.0,
        ))
        score -= 8.0

    links = sum(1 for v in (c.linkedin, c.github, c.website) if v)
    if links == 0:
        findings.append(_f(
            category="contact", severity=Severity.MEDIUM,
            rule_id="ats.contact.no_links",
            title="No profile links detected",
            explanation=(
                "No LinkedIn, GitHub, or portfolio/website link was found. "
                "At least one professional link is recommended."
            ),
            recommendation=(
                "Add a LinkedIn profile URL and/or a GitHub/portfolio link."
            ),
            impact=15.0,
        ))
        score -= 15.0
    elif links == 1:
        findings.append(_f(
            category="contact", severity=Severity.LOW,
            rule_id="ats.contact.one_link",
            title="Only one profile link detected",
            explanation=(
                "Only one of LinkedIn / GitHub / portfolio was found. "
                "More links help recruiters validate your profile."
            ),
            recommendation=(
                "Consider adding at least one more profile link."
            ),
            impact=8.0,
        ))
        score -= 8.0

    # informational findings for each missing optional link
    for name, value in [
        ("LinkedIn", c.linkedin),
        ("GitHub", c.github),
        ("Portfolio/website", c.website),
    ]:
        if not value:
            findings.append(_f(
                category="contact", severity=Severity.INFO,
                rule_id=(
                    "ats.contact."
                    + name.lower().replace("/", "_").replace(" ", "_")
                ),
                title=f"No {name} link detected",
                explanation=f"{name} is optional but recommended.",
                recommendation=f"Add a {name} link if available.",
            ))

    return max(0.0, score), findings


# ---------------------------------------------------------------------------
# 2. Section completeness & structure
# ---------------------------------------------------------------------------

StructureKeys = frozenset(
    {"summary", "experience", "education", "skills", "projects", "certifications"}
)


def analyze_structure(resume: Resume) -> tuple[float | None, list[Finding]]:
    presence = 100.0
    findings: list[Finding] = []

    has_summary = bool(resume.summary and resume.summary.strip())
    has_experience = len(resume.experience) > 0
    has_education = len(resume.education) > 0
    has_skills = bool(
        resume.skills.technical
        or resume.skills.soft
        or resume.skills.tools
        or resume.skills.languages
        or resume.skills.all
    )
    has_projects = len(resume.projects) > 0
    has_certs = len(resume.certifications) > 0

    if not has_summary:
        findings.append(_f(
            category="structure", severity=Severity.LOW,
            rule_id="ats.structure.summary",
            title="No summary or objective",
            explanation=(
                "A summary helps ATS and recruiters quickly assess fit. "
                "Its absence is not heavily penalised but is noted."
            ),
            recommendation=(
                "Consider adding a brief 2–3 sentence professional summary."
            ),
            impact=8.0,
        ))
        presence -= 8.0

    if not has_experience:
        mitigation = "Projects are present and provide partial evidence of work."
        severity = Severity.MEDIUM
        deduction = 12.0
        if has_projects:
            mitigation = (
                "Projects section is present, providing partial evidence of "
                "substantive work."
            )
            severity = Severity.MEDIUM
            deduction = 12.0
        else:
            severity = Severity.HIGH
            deduction = 25.0
            mitigation = (
                "Neither work experience nor projects were detected. This "
                "significantly limits the ATS content assessment."
            )
        findings.append(_f(
            category="structure", severity=severity,
            rule_id="ats.structure.experience",
            title="No work experience detected",
            explanation=mitigation,
            recommendation=(
                "Add work experience entries with clear roles, dates, and "
                "achievement bullets."
            ),
            impact=deduction,
        ))
        presence -= deduction

    if not has_education:
        findings.append(_f(
            category="structure", severity=Severity.MEDIUM,
            rule_id="ats.structure.education",
            title="No education section detected",
            explanation=(
                "An education section is commonly expected. Its absence "
                "is noted but not heavily penalised for experienced profiles."
            ),
            recommendation=(
                "Add education with institution, degree, and dates."
            ),
            impact=10.0,
        ))
        presence -= 10.0

    if not has_skills:
        findings.append(_f(
            category="structure", severity=Severity.HIGH,
            rule_id="ats.structure.skills",
            title="No skills section detected",
            explanation=(
                "A skills section is a key ATS signal. Its absence "
                "significantly limits how well an ATS can identify relevant "
                "technology and capability matches."
            ),
            recommendation=(
                "Add a structured skills section with technical tools, "
                "languages, and relevant competencies."
            ),
            impact=20.0,
        ))
        presence -= 20.0

    if not has_projects:
        if has_experience:
            findings.append(_f(
                category="structure", severity=Severity.INFO,
                rule_id="ats.structure.projects_optional",
                title="No projects section",
                explanation=(
                    "Projects are optional for experienced profiles "
                    "with a strong experience section."
                ),
                recommendation=(
                    "Optional: add a projects section if relevant to the "
                    "target role."
                ),
            ))
        else:
            findings.append(_f(
                category="structure", severity=Severity.INFO,
                rule_id="ats.structure.projects_recommended",
                title="No projects section",
                explanation=(
                    "Early-career profiles often benefit from a projects "
                    "section to demonstrate substantive work."
                ),
                recommendation=(
                    "Consider adding a projects section to demonstrate "
                    "applied skills."
                ),
            ))

    if not has_certs:
        findings.append(_f(
            category="structure", severity=Severity.INFO,
            rule_id="ats.structure.certs_optional",
            title="No certifications section",
            explanation="Certifications are optional and not penalised.",
            recommendation=(
                "Optional: add relevant certifications if available."
            ),
        ))

    presence = max(0.0, presence)

    # --- Section ordering subscore ---
    order_score: float | None = None
    sec_order = resume.metadata.section_order
    known = [k for k in sec_order if k in StructureKeys]
    if len(known) >= 2:
        pos = {k: i for i, k in enumerate(known)}
        o = 100.0
        if (
            "education" in pos
            and "experience" in pos
            and pos["education"] < pos["experience"]
        ):
            findings.append(_f(
                category="structure", severity=Severity.LOW,
                rule_id="ats.structure.order.education_first",
                title="Education section appears before experience",
                explanation=(
                    "Section ordering suggests education precedes experience. "
                    "Reverse-chronological work-first ordering is more common "
                    "in ATS-ready resumes."
                ),
                recommendation=(
                    "Consider placing experience before education unless "
                    "there is a strong reason for the current order."
                ),
                impact=8.0,
            ))
            o -= 8.0
        if (
            "summary" in pos
            and "experience" in pos
            and pos["summary"] > pos["experience"]
        ):
            findings.append(_f(
                category="structure", severity=Severity.LOW,
                rule_id="ats.structure.order.summary_late",
                title="Summary appears after experience",
                explanation=(
                    "A professional summary is conventionally placed "
                    "before experience."
                ),
                recommendation=(
                    "Consider moving the summary to the top of the resume "
                    "for a more standard order."
                ),
                impact=6.0,
            ))
            o -= 6.0
        order_score = max(0.0, o)
    else:
        if has_experience or has_education or has_summary:
            findings.append(_f(
                category="structure", severity=Severity.INFO,
                rule_id="ats.structure.order.unavailable",
                title="Section ordering could not be assessed",
                explanation=(
                    "Ordering metadata is unavailable; section order "
                    "could not be evaluated."
                ),
                recommendation=(
                    "No action required; ordering analysis is informational only."
                ),
            ))

    if order_score is not None:
        final = round(0.75 * presence + 0.25 * order_score, 1)
    else:
        final = presence

    return max(0.0, final), findings


# ---------------------------------------------------------------------------
# 3. Experience quality
# ---------------------------------------------------------------------------


def analyze_experience(resume: Resume) -> tuple[float | None, list[Finding]]:
    exp = resume.experience
    if not exp:
        return None, []
    findings: list[Finding] = []
    n = len(exp)

    entry_scores: list[float] = []
    for idx, e in enumerate(exp):
        s = 100.0
        label = e.company or e.title or f"entry {idx + 1}"
        bullets = e.achievements or []

        if not e.title.strip():
            findings.append(_f(
                category="experience", severity=Severity.MEDIUM,
                rule_id="ats.experience.title",
                title=f"Missing job title in {label}",
                explanation=(
                    "Role title helps ATS identify the seniority and function "
                    "of the position."
                ),
                recommendation="Add a clear job title.",
                impact=20.0,
            ))
            s -= 20.0

        if not e.company.strip():
            findings.append(_f(
                category="experience", severity=Severity.MEDIUM,
                rule_id="ats.experience.company",
                title=f"Missing company name in {label}",
                explanation=(
                    "Company name provides important context and is a common "
                    "ATS field."
                ),
                recommendation="Add the company or organisation name.",
                impact=20.0,
            ))
            s -= 20.0

        if not e.start_date:
            findings.append(_f(
                category="experience", severity=Severity.LOW,
                rule_id="ats.experience.start_date",
                title=f"Missing start date in {label}",
                explanation=(
                    "Start date is common metadata for ATS experience parsing."
                ),
                recommendation="Add start date (e.g., Jan 2022).",
                impact=10.0,
            ))
            s -= 10.0

        if not e.end_date:
            findings.append(_f(
                category="experience", severity=Severity.LOW,
                rule_id="ats.experience.end_date",
                title=f"Missing end date in {label}",
                explanation=(
                    "End date (or 'Present' for current roles) helps "
                    "establish timeline context."
                ),
                recommendation="Add end date or mark as 'Present'.",
                impact=5.0,
            ))
            s -= 5.0

        detail = e.description or ""
        detail_words = len(detail.split())
        bullet_words_total = sum(len(b.split()) for b in bullets)
        total_words = detail_words + bullet_words_total

        if total_words == 0:
            findings.append(_f(
                category="experience", severity=Severity.HIGH,
                rule_id="ats.experience.detail",
                title=f"No evidence content in {label}",
                explanation=(
                    "The role entry contains no description or achievement "
                    "bullets; an ATS will find no substantive content."
                ),
                recommendation=(
                    "Add 2–5 achievement bullets describing impact and scope."
                ),
                impact=25.0,
            ))
            s -= 25.0
        else:
            if len(bullets) == 0:
                findings.append(_f(
                    category="experience", severity=Severity.MEDIUM,
                    rule_id="ats.experience.bullets",
                    title=f"No bullet points in {label}",
                    explanation=(
                        "Achievement bullets are more readable and parseable "
                        "than a paragraph description alone."
                    ),
                    recommendation=(
                        "Break descriptions into 2–5 bullet points starting "
                        "with action verbs."
                    ),
                    impact=10.0,
                ))
                s -= 10.0
            elif len(bullets) < 3:
                findings.append(_f(
                    category="experience", severity=Severity.LOW,
                    rule_id="ats.experience.few_bullets",
                    title=f"Very few bullets in {label}",
                    explanation=(
                        "Only "
                        f"{len(bullets)} bullet(s) found; 3 or more is typical."
                    ),
                    recommendation="Consider adding more achievement bullets.",
                    impact=5.0,
                ))
                s -= 5.0

        if total_words > 0 and total_words < 10:
            findings.append(_f(
                category="experience", severity=Severity.LOW,
                rule_id="ats.experience.short_entry",
                title=f"Sparse content in {label}",
                explanation=(
                    f"The entry contains only about {total_words} words, "
                    "which may signal incomplete or stub content."
                ),
                recommendation=(
                    "Expand with quantified achievements and context."
                ),
                impact=10.0,
            ))
            s -= 10.0
        elif total_words > 150:
            findings.append(_f(
                category="experience", severity=Severity.LOW,
                rule_id="ats.experience.long_entry",
                title=f"Unusually dense entry in {label}",
                explanation=(
                    f"The entry contains {total_words} words, which is "
                    "notably dense. ATS extraction may lose information."
                ),
                recommendation=(
                    "Consider trimming to the most relevant, quantified "
                    "achievements."
                ),
                impact=10.0,
            ))
            s -= 10.0

        entry_scores.append(max(0.0, min(100.0, s)))

    avg = sum(entry_scores) / len(entry_scores) if entry_scores else 0.0

    count_score = 100.0
    if n == 1:
        findings.append(_f(
            category="experience", severity=Severity.INFO,
            rule_id="ats.experience.single_entry",
            title="Single work entry detected",
            explanation=(
                "Only one work experience entry was found. This is "
                "normal for early-career candidates but may leave "
                "senior profiles under-evidenced."
            ),
            recommendation=(
                "For experienced candidates, aim for 2 or more "
                "role entries."
            ),
            impact=0.0,
        ))
        count_score = 95.0
    elif n == 0:
        count_score = 0.0

    score = round(0.8 * avg + 0.2 * count_score, 1)
    return max(0.0, min(100.0, score)), findings


# ---------------------------------------------------------------------------
# 4. Bullet / action quality
# ---------------------------------------------------------------------------


def analyze_bullets(resume: Resume) -> tuple[float | None, list[Finding]]:
    lines = collect_lines(resume)
    if not lines:
        return None, []

    findings: list[Finding] = []
    total = len(lines)
    texts = [al.text for al in lines]

    # Action-verb coverage
    verb_hits = sum(is_action_start(t) for t in texts)
    coverage = verb_hits / total if total else 0.0
    if coverage == 0:
        findings.append(_f(
            category="bullets", severity=Severity.MEDIUM,
            rule_id="ats.bullets.no_action",
            title="No action-verb beginnings detected",
            explanation=(
                "None of the achievement lines begin with a recognised "
                "action verb. Action-verb openings improve ATS readability."
            ),
            recommendation=(
                "Start each bullet with an action verb such as 'Led', "
                "'Built', 'Reduced', or 'Designed'."
            ),
            impact=25.0,
        ))
    elif coverage < 0.2:
        findings.append(_f(
            category="bullets", severity=Severity.MEDIUM,
            rule_id="ats.bullets.low_action_coverage",
            title="Very low action-verb coverage",
            explanation=(
                f"Only {verb_hits} of {total} achievement lines begin with "
                "a recognised action verb."
            ),
            recommendation=(
                "Rewrite most bullets to begin with action verbs."
            ),
            impact=25.0,
        ))
    elif coverage < 0.4:
        findings.append(_f(
            category="bullets", severity=Severity.MEDIUM,
            rule_id="ats.bullets.moderate_action_coverage",
            title="Moderate action-verb coverage",
            explanation=(
                f"{verb_hits} of {total} achievement lines use action verbs."
            ),
            recommendation=(
                "Aim for at least 60% of bullets to start with action verbs."
            ),
            impact=15.0,
        ))
    elif coverage < 0.6:
        findings.append(_f(
            category="bullets", severity=Severity.LOW,
            rule_id="ats.bullets.improving_action_coverage",
            title="Action-verb coverage is approaching target",
            explanation=(
                f"{verb_hits} of {total} achievement lines use action verbs."
            ),
            recommendation=(
                "Consider improving a few more bullets to improve coverage."
            ),
            impact=5.0,
        ))

    # First-person pronouns
    fp_hits = [t for t in texts if first_person_words(t)]
    if fp_hits:
        findings.append(_f(
            category="bullets", severity=Severity.MEDIUM,
            rule_id="ats.bullets.first_person",
            title="First-person pronouns detected",
            explanation=(
                f"{len(fp_hits)} achievement line(s) contain first-person "
                "pronouns (I, my, we, our). Resumes conventionally omit "
                "first-person subjects."
            ),
            evidence=_trunc(fp_hits[0]),
            recommendation="Rewrite without first-person pronouns.",
            impact=10.0,
        ))

    # Short bullets (< 3 words)
    short_lines = [al for al in lines if len(al.text.split()) < 3]
    if short_lines:
        frac = len(short_lines) / total
        if frac > 0.33:
            findings.append(_f(
                category="bullets", severity=Severity.MEDIUM,
                rule_id="ats.bullets.too_short",
                title="Many achievement lines are very short",
                explanation=(
                    f"{len(short_lines)} of {total} lines are under 3 words "
                    "and may not carry meaningful substance."
                ),
                evidence=_trunc(short_lines[0].text),
                recommendation=(
                    "Expand very short bullets with quantified context."
                ),
                impact=10.0,
            ))

    # Long bullets (> 40 words)
    long_lines = [al for al in lines if len(al.text.split()) > 40]
    if long_lines:
        frac = len(long_lines) / total
        if frac > 0.2:
            findings.append(_f(
                category="bullets", severity=Severity.LOW,
                rule_id="ats.bullets.too_long",
                title="Some achievement lines are very long",
                explanation=(
                    f"{len(long_lines)} of {total} lines exceed 40 words. "
                    "Long lines may lose ATS parseability."
                ),
                evidence=_trunc(long_lines[0].text),
                recommendation=(
                    "Trim long bullets to the key quantified result."
                ),
                impact=10.0,
            ))

    # Vague / non-informative
    vague_hits = [al for al in lines if is_vague(al.text)]
    if vague_hits:
        frac = len(vague_hits) / total
        if frac > 0.5:
            findings.append(_f(
                category="bullets", severity=Severity.MEDIUM,
                rule_id="ats.bullets.vague",
                title="Many achievement lines are vague",
                explanation=(
                    f"{len(vague_hits)} of {total} lines contain vague or "
                    "non-informative markers."
                ),
                evidence=_trunc(vague_hits[0].text),
                recommendation=(
                    "Replace vague phrasing with specific actions and outcomes."
                ),
                impact=15.0,
            ))

    # Repetitive first-verb pattern
    verb_stems: list[str] = []
    for t in texts:
        tokens = t.split()
        if tokens:
            stem = tokens[0].lower().rstrip(".")
            if stem:
                verb_stems.append(stem)
    if verb_stems:
        from collections import Counter

        counts = Counter(verb_stems)
        most_common_count = counts.most_common(1)[0][1]
        verb_hit_count = sum(1 for s in verb_stems if s in ACTION_VERBS)
        if verb_hit_count >= 4 and most_common_count >= verb_hit_count * 0.5:
            findings.append(_f(
                category="bullets", severity=Severity.LOW,
                rule_id="ats.bullets.repetitive",
                title="Repetitive first-verb pattern",
                explanation=(
                    f"A single verb stem accounts for {most_common_count} of "
                    f"{verb_hit_count} action-verb beginnings. Variety "
                    "improves readability."
                ),
                recommendation=(
                    "Diversify the opening verbs across achievement lines."
                ),
                impact=8.0,
            ))

    # Compute score from deductions
    total_impact = sum(f.impact for f in findings if f.category == "bullets")
    score = max(0.0, min(100.0, 100.0 - total_impact))
    return round(score, 1), findings


# ---------------------------------------------------------------------------
# 5. Quantified achievements
# ---------------------------------------------------------------------------


def analyze_quantified(resume: Resume) -> tuple[float | None, list[Finding]]:
    lines = collect_lines(resume)
    if not lines:
        return None, []

    findings: list[Finding] = []
    texts = [al.text for al in lines]
    total = len(texts)
    quant_hits = [al for al in lines if has_quantitative(al.text)]
    n_quant = len(quant_hits)
    frac = n_quant / total if total else 0.0

    impact = 0.0
    if frac == 0:
        findings.append(_f(
            category="quantified", severity=Severity.MEDIUM,
            rule_id="ats.quantified.none",
            title="No quantified achievements detected",
            explanation=(
                "No percentages, currency, counts, or other quantitative "
                "tokens were found in the achievement lines."
            ),
            recommendation=(
                "Add quantified outcomes where possible "
                "(e.g., 'Reduced processing time by 30%')."
            ),
            impact=25.0,
        ))
        impact = 25.0
    elif frac < 0.15:
        findings.append(_f(
            category="quantified", severity=Severity.LOW,
            rule_id="ats.quantified.low",
            title="Low quantitative evidence",
            explanation=(
                f"Quantitative tokens were found in {n_quant} of {total} "
                "achievement lines; more is generally stronger."
            ),
            evidence=_trunc(quant_hits[0].text),
            recommendation="Add more quantified metrics and scope indicators.",
            impact=15.0,
        ))
        impact = 15.0
    elif frac < 0.3:
        findings.append(_f(
            category="quantified", severity=Severity.LOW,
            rule_id="ats.quantified.moderate",
            title="Moderate quantitative evidence",
            explanation=(
                f"Quantitative tokens were found in {n_quant} of {total} "
                "achievement lines."
            ),
            recommendation=(
                "Add a few more quantified outcomes to strengthen the "
                "achievement lines."
            ),
            impact=8.0,
        ))
        impact = 8.0
    else:
        findings.append(_f(
            category="quantified", severity=Severity.INFO,
            rule_id="ats.quantified.good",
            title="Good quantitative evidence",
            explanation=(
                f"Quantitative tokens were found in {n_quant} of {total} "
                "achievement lines."
            ),
            evidence=_trunc(quant_hits[0].text),
            recommendation=(
                "Maintain quantified outcomes across the most relevant "
                "achievement lines."
            ),
        ))

    return max(0.0, min(100.0, 100.0 - impact)), findings


# ---------------------------------------------------------------------------
# 6. Skills presentation
# ---------------------------------------------------------------------------


def analyze_skills(resume: Resume) -> tuple[float | None, list[Finding]]:
    technical = resume.skills.technical
    soft = resume.skills.soft
    tools = resume.skills.tools
    languages = resume.skills.languages
    all_skills = resume.skills.all

    primary = technical + soft + tools + languages
    if not primary and all_skills:
        primary = list(all_skills)
    if not primary:
        return None, []

    findings: list[Finding] = []
    score = 100.0

    # Duplicate detection (case-insensitive, whitespace-stripped)
    seen: dict[str, str] = {}
    dups: list[str] = []
    for item in primary:
        key = item.strip().lower()
        if not key:
            continue
        if key in seen:
            dups.append(item)
        else:
            seen[key] = item

    n_unique = len(seen)
    if dups:
        if len(dups) <= 3:
            findings.append(_f(
                category="skills", severity=Severity.LOW,
                rule_id="ats.skills.duplicates",
                title="Duplicate skill entries detected",
                explanation=(
                    f"{len(dups)} duplicate skill(s) were found in the "
                    "skills list."
                ),
                evidence=_trunc(", ".join(dups[:3])),
                recommendation="Remove duplicate entries.",
                impact=5.0,
            ))
            score -= 5.0
        else:
            findings.append(_f(
                category="skills", severity=Severity.MEDIUM,
                rule_id="ats.skills.duplicates",
                title="Multiple duplicate skill entries",
                explanation=(
                    f"{len(dups)} duplicate skill(s) were found in the "
                    "skills list."
                ),
                evidence=_trunc(", ".join(dups[:3])),
                recommendation="Remove duplicate entries.",
                impact=12.0,
            ))
            score -= 12.0

    # Extremely large skill list
    if n_unique > 60:
        findings.append(_f(
            category="skills", severity=Severity.MEDIUM,
            rule_id="ats.skills.large",
            title="Very large skill list",
            explanation=(
                f"{n_unique} unique skills were detected; very large lists "
                "may overwhelm ATS keyword extraction."
            ),
            recommendation=(
                "Focus on the most relevant 20–40 skills."
            ),
            impact=25.0,
        ))
        score -= 25.0
    elif n_unique > 40:
        findings.append(_f(
            category="skills", severity=Severity.LOW,
            rule_id="ats.skills.large",
            title="Large skill list",
            explanation=(
                f"{n_unique} unique skills were detected; consider "
                "narrowing to the most relevant."
            ),
            recommendation=(
                "Consider trimming the list to the most relevant skills."
            ),
            impact=15.0,
        ))
        score -= 15.0

    # Structured categories
    nonempty_cats = sum(
        1 for lst in (technical, soft, tools, languages) if lst
    )
    if nonempty_cats <= 1 and n_unique >= 5:
        findings.append(_f(
            category="skills", severity=Severity.LOW,
            rule_id="ats.skills.unstructured",
            title="Skills are minimally structured",
            explanation=(
                f"Only {nonempty_cats} skill category is populated out of "
                "technical/soft/tools/languages. Structured categories help "
                "ATS parse skills."
            ),
            recommendation=(
                "Separate skills into categories: technical, tools, "
                "soft skills, and languages."
            ),
            impact=8.0,
        ))
        score -= 8.0

    if nonempty_cats == 0 and (all_skills):
        findings.append(_f(
            category="skills", severity=Severity.LOW,
            rule_id="ats.skills.unusual_categories",
            title="Skills found only in flat list",
            explanation=(
                "Skills are present but not placed in any structured "
                "category. All skills are in the flat 'all' list only."
            ),
            recommendation=(
                "Use the structured skill categories "
                "(technical/tools/soft/languages)."
            ),
            impact=5.0,
        ))
        score -= 5.0

    # Phrase entries (> 3 words)
    phrase_skills = [item for item in primary if len(item.split()) > 3]
    if phrase_skills:
        frac = len(phrase_skills) / len(primary) if primary else 0.0
        if frac >= 0.2:
            findings.append(_f(
                category="skills", severity=Severity.LOW,
                rule_id="ats.skills.phrase_entries",
                title="Many skill entries look like phrases",
                explanation=(
                    f"{len(phrase_skills)} of {len(primary)} skill entries "
                    "exceed 3 words and may not parse as distinct skills."
                ),
                evidence=_trunc(", ".join(phrase_skills[:2])),
                recommendation=(
                    "Use short skill names where possible; save phrases "
                    "for summary bullets."
                ),
                impact=8.0,
            ))
            score -= 8.0

    # Empty category info
    empty_cats = [name for name, lst in [
        ("technical", technical), ("soft", soft),
        ("tools", tools), ("languages", languages),
    ] if not lst and nonempty_cats > 0]
    if empty_cats:
        findings.append(_f(
            category="skills", severity=Severity.INFO,
            rule_id="ats.skills.empty_categories",
            title="Some skill categories are empty",
            explanation=(
                f"The following categories are empty: {', '.join(empty_cats)}."
            ),
            recommendation=(
                "Populating these categories may improve ATS readability."
            ),
        ))

    return max(0.0, min(100.0, score)), findings


# ---------------------------------------------------------------------------
# 7. Education quality
# ---------------------------------------------------------------------------


def analyze_education(resume: Resume) -> tuple[float | None, list[Finding]]:
    edu = resume.education
    if not edu:
        return None, []

    findings: list[Finding] = []
    scores: list[float] = []

    for idx, e in enumerate(edu):
        s = 100.0

        if not e.institution.strip():
            findings.append(_f(
                category="education", severity=Severity.MEDIUM,
                rule_id="ats.education.institution",
                title=f"Missing institution in education entry {idx + 1}",
                explanation=(
                    "Institution name provides context for ATS and recruiter "
                    "evaluation."
                ),
                recommendation="Add the institution name.",
                impact=25.0,
            ))
            s -= 25.0

        if not e.degree:
            findings.append(_f(
                category="education", severity=Severity.MEDIUM,
                rule_id="ats.education.degree",
                title=f"Missing degree in education entry {idx + 1}",
                explanation=(
                    "Degree level (e.g., B.Tech, B.Sc, MCA) is a common "
                    "ATS field."
                ),
                recommendation="Add the degree name.",
                impact=20.0,
            ))
            s -= 20.0

        if not e.field:
            findings.append(_f(
                category="education", severity=Severity.LOW,
                rule_id="ats.education.field",
                title=f"Missing field of study in education entry {idx + 1}",
                explanation=(
                    "Field of study helps ATS match specialised roles."
                ),
                recommendation="Add the field of study if applicable.",
                impact=5.0,
            ))
            s -= 5.0

        if not e.start_date and not e.end_date:
            findings.append(_f(
                category="education", severity=Severity.LOW,
                rule_id="ats.education.dates",
                title=f"No dates in education entry {idx + 1}",
                explanation=(
                    "Dates of study are not strictly required but are "
                    "commonly present."
                ),
                recommendation="Add study dates if applicable.",
                impact=10.0,
            ))
            s -= 10.0

        scores.append(max(0.0, min(100.0, s)))

    overall = round(sum(scores) / len(scores), 1) if scores else 0.0
    return max(0.0, min(100.0, overall)), findings


# ---------------------------------------------------------------------------
# 8. Projects & certifications
# ---------------------------------------------------------------------------


def analyze_projects_certs(resume: Resume) -> tuple[float | None, list[Finding]]:
    projects = resume.projects
    certs = resume.certifications
    if not projects and not certs:
        return None, []

    findings: list[Finding] = []
    scores: list[float] = []

    for idx, p in enumerate(projects):
        s = 100.0
        label = p.name or f"project {idx + 1}"

        if not p.name.strip():
            findings.append(_f(
                category="projects_certs", severity=Severity.MEDIUM,
                rule_id="ats.projects.name",
                title=f"Missing project name in entry {idx + 1}",
                explanation="Project name provides essential context.",
                recommendation="Add a clear project name.",
                impact=30.0,
            ))
            s -= 30.0

        if not p.description.strip():
            findings.append(_f(
                category="projects_certs", severity=Severity.MEDIUM,
                rule_id="ats.projects.description",
                title=f"Missing project description for {label}",
                explanation=(
                    "A description helps ATS and recruiters assess project "
                    "relevance."
                ),
                recommendation="Add a concise project description.",
                impact=25.0,
            ))
            s -= 25.0
        elif len(p.description.split()) < 6:
            findings.append(_f(
                category="projects_certs", severity=Severity.LOW,
                rule_id="ats.projects.thin",
                title=f"Thin project description for {label}",
                explanation=(
                    f"The description contains only {len(p.description.split())} "
                    "words."
                ),
                recommendation="Expand with scope and quantified outcomes.",
                impact=10.0,
            ))
            s -= 10.0

        if not p.technologies:
            findings.append(_f(
                category="projects_certs", severity=Severity.LOW,
                rule_id="ats.projects.tech",
                title=f"No technologies listed for {label}",
                explanation=(
                    "Technologies help ATS identify relevant technical "
                    "experience."
                ),
                recommendation="List the key technologies used.",
                impact=10.0,
            ))
            s -= 10.0

        scores.append(max(0.0, min(100.0, s)))

    for idx, c in enumerate(certs):
        s = 100.0

        if not c.name.strip():
            findings.append(_f(
                category="projects_certs", severity=Severity.MEDIUM,
                rule_id="ats.certs.name",
                title=f"Missing certification name in entry {idx + 1}",
                explanation="A certification requires a name to be meaningful.",
                recommendation="Add the certification name.",
                impact=40.0,
            ))
            s -= 40.0

        if not c.issuer.strip():
            findings.append(_f(
                category="projects_certs", severity=Severity.LOW,
                rule_id="ats.certs.issuer",
                title=f"No issuer for certification '{c.name or f'entry {idx + 1}'}'",
                explanation=(
                    "Issuing body adds credibility and ATS context."
                ),
                recommendation="Add the issuing organisation.",
                impact=15.0,
            ))
            s -= 15.0

        if not c.date:
            findings.append(_f(
                category="projects_certs", severity=Severity.LOW,
                rule_id="ats.certs.date",
                title=f"No date for certification '{c.name or f'entry {idx + 1}'}'",
                explanation=(
                    "Date of certification helps establish currency."
                ),
                recommendation="Add the certification date if available.",
                impact=10.0,
            ))
            s -= 10.0

        scores.append(max(0.0, min(100.0, s)))

    overall = round(sum(scores) / len(scores), 1) if scores else 0.0
    return max(0.0, min(100.0, overall)), findings


# ---------------------------------------------------------------------------
# 9. Date consistency
# ---------------------------------------------------------------------------


def analyze_dates(resume: Resume) -> tuple[float | None, list[Finding]]:
    entries: list[tuple[str, str | None, str | None, str]] = []
    for idx, exp_entry in enumerate(resume.experience):
        label = exp_entry.company or exp_entry.title or f"experience {idx + 1}"
        entries.append(("experience", exp_entry.start_date, exp_entry.end_date, label))
    for idx, edu_entry in enumerate(resume.education):
        label = edu_entry.institution or f"education {idx + 1}"
        entries.append(("education", edu_entry.start_date, edu_entry.end_date, label))

    findings: list[Finding] = []
    score = 100.0
    total_impact = 0.0

    parsed: list[tuple[str, DateVal, DateVal, str]] = []
    for kind, start_s, end_s, label in entries:
        sv = parse_date_value(start_s)
        ev = parse_date_value(end_s)
        if sv and ev and not sv.is_current and not ev.is_current:
            parsed.append((kind, sv, ev, label))

    if not parsed:
        if entries:
            findings.append(_f(
                category="dates", severity=Severity.INFO,
                rule_id="ats.dates.unverified",
                title="Date consistency could not be assessed",
                explanation=(
                    "No parseable start/end date pairs were found. Date "
                    "consistency checks require at least one fully-dated "
                    "entry."
                ),
                recommendation="Add dates to experience/education entries.",
            ))
        return None, findings

    # End-before-start
    for _kind, sv, ev, label in parsed:
        sv_t = _date_val_sortable(sv)
        ev_t = _date_val_sortable(ev)
        if sv_t and ev_t and ev_t < sv_t:
            findings.append(_f(
                category="dates", severity=Severity.MEDIUM,
                rule_id="ats.dates.end_before_start",
                title=f"End date precedes start date in {label}",
                explanation=(
                    f"End date '{_fmt_dv(ev)}' appears before start date "
                    f"'{_fmt_dv(sv)}'."
                ),
                recommendation=(
                    "Review and correct the dates; ensure end date is on "
                    "or after start date."
                ),
                impact=25.0,
            ))
            score -= 25.0
            total_impact += 25.0

    # Overlap detection (experience only, non-current)
    exp_parsed = [
        (sv, ev, label)
        for kind, sv, ev, label in parsed
        if kind == "experience"
    ]
    overlaps_found = False
    for i in range(len(exp_parsed)):
        for j in range(i + 1, len(exp_parsed)):
            sv_i, ev_i, lab_i = exp_parsed[i]
            sv_j, ev_j, lab_j = exp_parsed[j]
            svt_i = _date_val_sortable(sv_i)
            evt_i = _date_val_sortable(ev_i)
            svt_j = _date_val_sortable(sv_j)
            evt_j = _date_val_sortable(ev_j)
            if svt_i and evt_i and svt_j and evt_j:
                # Strict overlap only: adjacent year-only ranges such as
                # "2018-2020" and "2020-2024" are not flagged.
                if svt_i < evt_j and svt_j < evt_i:
                    if not overlaps_found:
                        findings.append(_f(
                            category="dates", severity=Severity.LOW,
                            rule_id="ats.dates.overlap",
                            title="Overlapping date ranges detected",
                            explanation=(
                                f"'{lab_i}' and '{lab_j}' have overlapping "
                                "date ranges. Overlaps are not inherently "
                                "invalid but may be confusing to ATS."
                            ),
                            recommendation=(
                                "Review overlapping entries; this may be "
                                "legitimate (e.g., concurrent roles)."
                            ),
                            impact=8.0,
                        ))
                        score -= 8.0
                        total_impact += 8.0
                        overlaps_found = True

    # Reverse-chronological order check (experience only)
    exp_dates = [
        (sv, ev, lab)
        for kind, sv, ev, lab in parsed
        if kind == "experience" and not sv.is_current and not ev.is_current
    ]
    if len(exp_dates) >= 2:
        starts = [_date_val_sortable(sv) for sv, _, _ in exp_dates]
        starts_clean = [s for s in starts if s is not None]
        if len(starts_clean) == len(exp_dates):
            # Check if listed in ascending (not reverse-chronological) order
            for k in range(len(starts_clean) - 1):
                if starts_clean[k] < starts_clean[k + 1]:
                    findings.append(_f(
                        category="dates", severity=Severity.LOW,
                        rule_id="ats.dates.order",
                        title="Experience entries not in reverse-chronological order",
                        explanation=(
                            "The resume lists earlier roles after later ones. "
                            "Reverse-chronological is the standard ATS "
                            "convention."
                        ),
                        recommendation=(
                            "Reverse the listing order so the most recent "
                            "role is first."
                        ),
                        impact=5.0,
                    ))
                    score -= 5.0
                    total_impact += 5.0
                    break

    return max(0.0, min(100.0, score)), findings


def _fmt_dv(dv: DateVal) -> str:
    if dv.is_current:
        return "Present"
    parts: list[str] = []
    if dv.month is not None:
        parts.append(str(dv.month).zfill(2))
    if dv.year is not None:
        parts.append(str(dv.year))
    return "/".join(parts) or "unknown"


# ---------------------------------------------------------------------------
# 10. Parsing / readability
# ---------------------------------------------------------------------------


def analyze_parsing(resume: Resume) -> tuple[float | None, list[Finding]]:
    findings: list[Finding] = []
    score = 100.0

    wc = resume.metadata.word_count
    if wc == 0:
        wc = computed_word_count(resume)

    if wc < 30:
        findings.append(_f(
            category="parsing", severity=Severity.MEDIUM,
            rule_id="ats.parsing.sparse",
            title="Resume content is very sparse",
            explanation=(
                f"The resume contains approximately {wc} words; very sparse "
                "content may limit ATS extractability."
            ),
            recommendation=(
                "Expand content with concrete achievements and context."
            ),
            impact=20.0,
        ))
        score -= 20.0
    elif wc < 80:
        findings.append(_f(
            category="parsing", severity=Severity.LOW,
            rule_id="ats.parsing.sparse",
            title="Resume content is sparse",
            explanation=(
                f"The resume contains approximately {wc} words; a minimum "
                "of 80–100 words is typical for ATS-ready resumes."
            ),
            recommendation="Consider expanding content.",
            impact=10.0,
        ))
        score -= 10.0
    elif wc > 1800:
        findings.append(_f(
            category="parsing", severity=Severity.LOW,
            rule_id="ats.parsing.dense",
            title="Resume content is unusually dense",
            explanation=(
                f"The resume contains approximately {wc} words; very long "
                "resumes may lose ATS parseability."
            ),
            recommendation=(
                "Consider trimming to the most relevant content."
            ),
            impact=8.0,
        ))
        score -= 8.0

    # Custom sections
    n_custom = len(resume.custom_sections)
    if n_custom > 2:
        findings.append(_f(
            category="parsing", severity=Severity.LOW,
            rule_id="ats.parsing.custom_sections",
            title=f"{n_custom} unrecognised section(s) detected",
            explanation=(
                "Unrecognised sections may not be parsed as expected by "
                "ATS software. Standard section headings are preferred."
            ),
            recommendation=(
                "Review custom sections; consider renaming them to "
                "recognised headings if applicable."
            ),
            impact=min(12.0, 6.0 * (n_custom - 2)),
        ))
        score -= min(12.0, 6.0 * (n_custom - 2))

    # Low parse confidence (informational only)
    for sc in resume.metadata.section_confidence:
        if sc.level.value == "low" and sc.section != "certifications":
            findings.append(_f(
                category="parsing", severity=Severity.INFO,
                rule_id=f"ats.parsing.low_confidence.{sc.section}",
                title=f"Section '{sc.section}' has low parse confidence",
                explanation=(
                    f"The parser could not clearly structure the "
                    f"'{sc.section}' section."
                ),
                recommendation=(
                    f"Ensure the '{sc.section}' section uses standard "
                    "formatting."
                ),
            ))

    # Missing standard section labels (informational)
    known_sections = {"summary", "experience", "education", "skills"}
    parsed_sections = {
        sc.section for sc in resume.metadata.section_confidence
        if sc.level.value != "low"
    }
    missing_labels = known_sections - parsed_sections
    content_present = bool(resume.experience or resume.education or resume.summary)
    if missing_labels and content_present and n_custom >= 1:
        findings.append(_f(
            category="parsing", severity=Severity.INFO,
            rule_id="ats.parsing.labels",
            title="Some standard sections were not detected",
            explanation=(
                "The following sections could not be clearly identified: "
                "{', '.join(sorted(missing_labels))}."
            ),
            recommendation=(
                "Use standard section headings (Experience, Education, Skills) "
                "for better ATS compatibility."
            ),
        ))

    # Formatting not assessable — always informational
    findings.append(_f(
        category="parsing", severity=Severity.INFO,
        rule_id="ats.parsing.format_not_assessed",
        title="Visual formatting risks not assessed",
        explanation=(
            "Visual formatting risks (multi-column layouts, tables, images, "
            "graphics, font choices) cannot be detected from parsed structured "
            "text. This analysis covers structural risks only."
        ),
        recommendation=(
            "Review the resume visually for ATS-unfriendly formatting "
            "elements."
        ),
    ))

    return max(0.0, min(100.0, score)), findings
