"""
Redrob Candidate Ranking — feature layer (Stage A preview build).

Pure-stdlib (datetime, re) so this module is safe to import from a future CPU-only
`rank.py`. No pandas/numpy here; no I/O; no model calls. Three public functions:

    build_evidence_text(candidate)  -> str   # what a recruiter actually reads (NO skills[])
    structured_features(candidate)  -> dict  # title family, services flag, yoe fit, geo, signals
    honeypot_flags(candidate)       -> dict  # consistency checks + reason string

Design notes baked in from DECISIONS.md:
- The skills[] array is engineered noise -> evidence text DELIBERATELY excludes it.
- Sentinels github_activity_score == -1 and offer_acceptance_rate == -1 mean MISSING/neutral.
- Negative signals (services-only career, title-chasing) matter as much as positive ones.
"""

from __future__ import annotations
from datetime import date, datetime
import re

# Reference "today" for recency math. Data's last_active_date tops out ~2026-05;
# session/current date is 2026-06-30. Centralized so it's easy to change for backtests.
REFERENCE_DATE = date(2026, 6, 30)

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

# Indian IT-services / consulting firms the JD explicitly down-weights ("People who have
# only worked at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini,
# etc.) in their entire career"). Matched case-insensitively as substrings on company name.
SERVICES_COMPANIES = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mindtree", "ltimindtree", "mphasis",
    "dxc", "igate", "syntel", "hexaware", "birlasoft", "persistent", "coforge",
]

# JD-preferred locations: "Candidates in Hyderabad, Pune, Mumbai, Delhi NCR welcome" +
# Pune/Noida offices. Delhi NCR := Delhi / Noida / Gurgaon / Gurugram / Ghaziabad / Faridabad.
PREFERRED_LOCATION_TOKENS = [
    "pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram",
    "ghaziabad", "faridabad", "new delhi",
]

# Title-family classification. Order matters: we test ai_ml first, then data, then swe,
# then fall through to nontech. Discipline-engineering titles (mechanical/civil/etc.) must
# NOT be caught by the generic "engineer" token, so swe matching is keyword-scoped and the
# non-software engineering disciplines are listed explicitly as nontech guards.
_AI_ML_TOKENS = [
    "machine learning", " ml ", "ml engineer", "ml ", "a.i", "ai engineer",
    "ai research", "ai specialist", "applied ml", "applied scientist", "deep learning",
    "nlp", "natural language", "data scientist", "computer vision", "recommendation",
    "search engineer", "research engineer", "ai/ml", "llm", "genai", "generative ai",
]
_DATA_TOKENS = [
    "data engineer", "data analyst", "analytics engineer", "business intelligence",
    "bi engineer", "data warehouse", "etl", "data platform",
]
_SWE_TOKENS = [
    "software", "full stack", "fullstack", "backend", "back end", "frontend",
    "front end", "devops", "cloud engineer", "qa engineer", "sdet", "mobile developer",
    "java developer", ".net", "python developer", "web developer", "platform engineer",
    "site reliability", "sre", "android", "ios developer", "application developer",
]
# Non-software engineering disciplines -> force nontech even though title says "engineer".
_NONTECH_ENGINEER_GUARDS = [
    "mechanical", "civil", "electrical", "chemical", "industrial", "production engineer",
    "structural", "automobile", "manufacturing engineer", "hardware",
]

# Parenthetical/suffix specialty qualifier that signals an ML/NLP/AI specialization sitting behind
# a generic base title, e.g. "Software Engineer (ML)", "Senior Engineer - NLP", "Lead / GenAI".
_AI_ML_QUALIFIER_RE = re.compile(
    r"[(\[\-–:/,]\s*(?:ml|nlp|ai|gen\s?ai|llm|data\s?science|machine\s?learning)\s*[)\]]?\s*$"
    r"|\(\s*(?:ml|nlp|ai|gen\s?ai|llm|data\s?science|machine\s?learning)\s*\)",
    re.IGNORECASE)

# First "<N> years" figure stated in profile.summary (informational corroboration for the
# inflated-yoe honeypot check — the templated summaries state the true career span).
_SUMMARY_YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(s):
    """Parse 'YYYY-MM-DD' (or 'YYYY-MM' / 'YYYY') -> date, else None."""
    if not s or not isinstance(s, str):
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _sorted_history(candidate):
    """career_history sorted most-recent-first by start_date (current roles float to top)."""
    ch = candidate.get("career_history") or []

    def key(role):
        d = _parse_date(role.get("start_date"))
        # missing start -> very old so it sinks; current role nudged to the very top.
        base = d.toordinal() if d else 0
        return (1 if role.get("is_current") else 0, base)

    return sorted(ch, key=key, reverse=True)


def _norm(s):
    return (s or "").lower()


# ---------------------------------------------------------------------------
# 1. Evidence text  (what a recruiter actually reads — NO skills[] array)
# ---------------------------------------------------------------------------

def build_evidence_text(candidate, repeat_recent=True):
    """
    Build the free-text 'evidence' a recruiter would actually read to judge fit.

    PRIMARY signal = career_history[].description (what they actually built/did), ordered
    MOST-RECENT-FIRST, with the current role's description repeated once (repeat_recent=True)
    to bias downstream vectorizers toward present work.

    SECONDARY context = profile.summary, appended at the END behind an explicit marker and
    DOWN-WEIGHTED. Rationale (Stage A finding): summaries are templated boilerplate that can
    contradict the actual title/roles (e.g. an "Accountant" whose summary claims a marketing
    career). It is context, not primary evidence.

    The skills[] array is DELIBERATELY EXCLUDED (engineered noise).

    Returns a single readable string.
    """
    profile = candidate.get("profile") or {}
    parts = []

    headline = (profile.get("headline") or "").strip()
    summary = (profile.get("summary") or "").strip()
    title = (profile.get("current_title") or "").strip()
    yoe = profile.get("years_of_experience")

    # --- header: title + yoe + one-line headline (lightweight context) ---
    head_bits = []
    if title:
        head_bits.append(title)
    if isinstance(yoe, (int, float)):
        head_bits.append(f"{yoe:g} yrs experience")
    if head_bits:
        parts.append(" | ".join(head_bits))
    if headline:
        parts.append(headline)

    # --- PRIMARY: career-history role descriptions, most-recent-first ---
    history = _sorted_history(candidate)
    for i, role in enumerate(history):
        rtitle = (role.get("title") or "").strip()
        company = (role.get("company") or "").strip()
        dur = role.get("duration_months")
        cur = role.get("is_current")
        desc = (role.get("description") or "").strip()

        tag = "CURRENT ROLE" if cur else "PAST ROLE"
        hdr_bits = [b for b in [rtitle, ("@ " + company) if company else ""] if b]
        meta = []
        if isinstance(dur, (int, float)):
            meta.append(f"{dur} mo")
        hdr = f"[{tag}] " + " ".join(hdr_bits)
        if meta:
            hdr += " (" + ", ".join(meta) + ")"
        block = hdr + (("\n" + desc) if desc else "")
        parts.append(block)

        # Upweight the single most-recent role by repeating its description once.
        if repeat_recent and i == 0 and desc:
            parts.append(desc)

    # --- SECONDARY: self-described summary, marked + down-weighted, at the very end ---
    if summary:
        parts.append(f"[SECONDARY — self-described summary] {summary}")

    return "\n\n".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------
# 2. Structured features
# ---------------------------------------------------------------------------

def classify_title_family(title):
    """Classify a job title into {ai_ml, data, swe, nontech}."""
    t = _norm(title)
    if not t:
        return "nontech"
    padded = f" {t} "
    # Non-software engineering disciplines first (don't let 'engineer' -> swe).
    if any(g in t for g in _NONTECH_ENGINEER_GUARDS):
        return "nontech"
    # Parenthetical / suffix specialty qualifier -> ai_ml, e.g. "Software Engineer (ML)",
    # "Engineer - NLP", "Engineer / GenAI". Catches ML titles hidden behind a generic base title.
    if _AI_ML_QUALIFIER_RE.search(title):
        return "ai_ml"
    if any(tok in padded for tok in _AI_ML_TOKENS):
        return "ai_ml"
    if any(tok in t for tok in _DATA_TOKENS):
        return "data"
    if any(tok in t for tok in _SWE_TOKENS):
        return "swe"
    # generic 'engineer'/'developer' with no discipline guard -> lean swe; else nontech.
    if "developer" in t or "engineer" in t or "programmer" in t:
        return "swe"
    return "nontech"


def _yoe_fit(yoe):
    """Score fit against the JD band 5-9 yrs, sweet spot 6-8. Returns 0..1."""
    if not isinstance(yoe, (int, float)):
        return None
    if 6 <= yoe <= 8:
        return 1.0
    if 5 <= yoe <= 9:
        return 0.85
    if 4 <= yoe < 5 or 9 < yoe <= 10:
        return 0.6
    if 3 <= yoe < 4 or 10 < yoe <= 12:
        return 0.4
    if 2 <= yoe < 3 or 12 < yoe <= 14:
        return 0.25
    return 0.1


def _is_services(company):
    c = _norm(company)
    return any(s in c for s in SERVICES_COMPANIES)


def structured_features(candidate):
    """
    Structured, JD-aligned features. Returns a flat dict. Sentinel -1 values for
    github_activity_score and offer_acceptance_rate are mapped to None (missing/neutral),
    NEVER treated as a low score.
    """
    profile = candidate.get("profile") or {}
    signals = candidate.get("redrob_signals") or {}
    history = _sorted_history(candidate)  # most-recent-first

    title = profile.get("current_title")
    title_family = classify_title_family(title)

    # --- services-company exposure across full career ---
    fam_companies = [(r.get("company"), _is_services(r.get("company"))) for r in history]
    services_roles = [c for c, flag in fam_companies if flag]
    n_roles = len(history)
    is_services_company = bool(services_roles) or _is_services(profile.get("current_company"))
    services_ratio = (len(services_roles) / n_roles) if n_roles else 0.0
    # JD's actual disqualifier is career-LONG services with no product exposure.
    services_only_career = n_roles > 0 and len(services_roles) == n_roles

    # --- yoe fit ---
    yoe = profile.get("years_of_experience")
    yoe_fit = _yoe_fit(yoe)

    # --- career coherence: fraction of roles in tech families (ai_ml/data/swe) ---
    role_fams = [classify_title_family(r.get("title")) for r in history]
    tech_fams = [f for f in role_fams if f in ("ai_ml", "data", "swe")]
    career_coherence = (len(tech_fams) / n_roles) if n_roles else 0.0
    # family-switch churn: distinct families / roles (lower = more coherent)
    distinct_fams = len(set(role_fams)) if role_fams else 0

    # --- product_vs_services trajectory in [-1, 1], recent roles weighted higher ---
    # +1 = consistently product (non-services); -1 = consistently services. Recency-weighted.
    if n_roles:
        weights = [1.0 / (i + 1) for i in range(n_roles)]  # recent roles weigh more
        wsum = sum(weights)
        score = sum(w * (-1.0 if flag else 1.0)
                    for w, (_, flag) in zip(weights, fam_companies))
        product_vs_services = round(score / wsum, 3)
    else:
        product_vs_services = None

    # --- behavioral / availability signals (sentinels -> None) ---
    last_active = _parse_date(signals.get("last_active_date"))
    days_since_last_active = (REFERENCE_DATE - last_active).days if last_active else None

    gh = signals.get("github_activity_score")
    github_activity = None if (gh is None or gh == -1) else gh
    oar = signals.get("offer_acceptance_rate")
    offer_acceptance_rate = None if (oar is None or oar == -1) else oar

    # --- geography ---
    location = _norm(profile.get("location"))
    country = profile.get("country")
    country_in_india = (_norm(country) == "india")
    location_preferred = any(tok in location for tok in PREFERRED_LOCATION_TOKENS)

    return {
        "title": title,
        "title_family": title_family,
        "is_services_company": is_services_company,
        "services_ratio": round(services_ratio, 3),
        "services_only_career": services_only_career,
        "years_of_experience": yoe,
        "yoe_fit": yoe_fit,
        "career_coherence": round(career_coherence, 3),
        "n_distinct_title_families": distinct_fams,
        "product_vs_services": product_vs_services,
        "days_since_last_active": days_since_last_active,
        "recruiter_response_rate": signals.get("recruiter_response_rate"),
        "notice_period_days": signals.get("notice_period_days"),
        "open_to_work": signals.get("open_to_work_flag"),
        "willing_to_relocate": signals.get("willing_to_relocate"),
        "location_preferred": location_preferred,
        "country_in_india": country_in_india,
        "github_activity": github_activity,          # None == no GitHub linked (neutral)
        "offer_acceptance_rate": offer_acceptance_rate,  # None == no offer history (neutral)
    }


# ---------------------------------------------------------------------------
# 3. Honeypot detection  (conservative-but-broad; spec says ~80 exist)
# ---------------------------------------------------------------------------

def honeypot_flags(candidate):
    """
    Run profile-consistency checks for subtly-impossible (honeypot) profiles. Returns a dict
    of individual boolean checks, an aggregate `is_honeypot`, a `n_flags` count, and a human
    `reason` string. Designed conservative-but-broad: the spec says ~80 exist and a naive
    detector found only 8.

    GATING checks (gate `is_honeypot`; each fires on ~0.0-0.03% of the pool, summing to the
    spec's ~80/100k):
      - expert_zero_duration      : a skill at 'expert' proficiency with duration_months == 0
      - expert_low_experience     : any 'expert' skill while years_of_experience < 3
      - role_tenure_gt_career     : one role's tenure exceeds total experience (proxy for
                                     "8 yrs at a company founded 3 yrs ago"; no founding data)
      - career_months_gt_experience: summed role months far exceed stated years_of_experience
      - career_date_error         : end<start, future start, or duration vs date mismatch,
                                     or >1 concurrent current role / heavy overlaps
      - yoe_gt_career_span        : claimed years_of_experience exceeds the observed span from
                                     the earliest career start_date to REFERENCE_DATE by >12 mo
                                     (the INVERSE inflated-experience trap: e.g. yoe field 16.9
                                     while the whole listed career spans 6.7 yrs; F1 audit found
                                     25/100k, 21 corroborated by the candidate's own summary
                                     stating the true span)

    SOFT / informational only (NOT gating — Stage-A measurement showed this fires on ~13% of
    the pool because the synthetic generator assigns skill durations independently of
    experience, so it is generator noise, not a designed trap):
      - skill_duration_gt_career  : a skill used longer than the candidate's whole career
    """
    profile = candidate.get("profile") or {}
    skills = candidate.get("skills") or []
    history = candidate.get("career_history") or []
    yoe = profile.get("years_of_experience")
    yoe_months = (yoe * 12) if isinstance(yoe, (int, float)) else None

    reasons = []        # strong / gating reasons
    soft_reasons = []   # informational only (generator noise)

    # --- skill-based checks ---
    expert_zero_duration = False
    expert_low_experience = False
    skill_duration_gt_career = False  # SOFT — does not gate is_honeypot
    for s in skills:
        prof = _norm(s.get("proficiency"))
        dur = s.get("duration_months")
        name = s.get("name") or "?"
        if prof == "expert" and (dur == 0 or dur is None):
            expert_zero_duration = True
            reasons.append(f"'expert' in {name} with 0 months used")
        if prof == "expert" and isinstance(yoe, (int, float)) and yoe < 3:
            expert_low_experience = True
            reasons.append(f"'expert' in {name} but only {yoe:g} yrs total experience")
        if (yoe_months is not None and isinstance(dur, (int, float))
                and dur > yoe_months + 6):
            skill_duration_gt_career = True
            soft_reasons.append(
                f"used {name} for {dur} mo but total career is ~{yoe_months:.0f} mo")

    # --- career-history checks ---
    role_tenure_gt_career = False
    career_date_error = False
    total_role_months = 0
    n_current = 0
    intervals = []  # (start_ord, end_ord) for overlap detection

    for r in history:
        dur = r.get("duration_months")
        if isinstance(dur, (int, float)):
            total_role_months += dur
            if yoe_months is not None and dur > yoe_months + 6:
                role_tenure_gt_career = True
                reasons.append(
                    f"single role tenure {dur} mo exceeds total career ~{yoe_months:.0f} mo")
        if r.get("is_current"):
            n_current += 1

        sd = _parse_date(r.get("start_date"))
        ed = _parse_date(r.get("end_date"))
        if sd and ed and ed < sd:
            career_date_error = True
            reasons.append(f"role end_date {ed} precedes start_date {sd}")
        if sd and sd > REFERENCE_DATE:
            career_date_error = True
            reasons.append(f"role start_date {sd} is in the future")
        # duration vs date span mismatch (> 6 months off)
        if sd and ed and isinstance(dur, (int, float)):
            span_months = (ed.year - sd.year) * 12 + (ed.month - sd.month)
            if abs(span_months - dur) > 6:
                career_date_error = True
                reasons.append(
                    f"duration_months {dur} disagrees with date span ~{span_months} mo")
        if sd:
            end_ord = (ed or REFERENCE_DATE).toordinal()
            intervals.append((sd.toordinal(), end_ord))

    if n_current > 1:
        career_date_error = True
        reasons.append(f"{n_current} concurrent 'current' roles")

    # heavy overlap between any two non-identical roles (> ~12 months overlap)
    intervals.sort()
    for i in range(len(intervals) - 1):
        a_s, a_e = intervals[i]
        b_s, b_e = intervals[i + 1]
        overlap_days = min(a_e, b_e) - b_s
        if overlap_days > 365:
            career_date_error = True
            reasons.append("two roles overlap by more than a year")
            break

    career_months_gt_experience = False
    if yoe_months is not None and total_role_months > yoe_months + 24:
        # claims substantially more job-months than total experience (gaps are fine; this is excess)
        career_months_gt_experience = True
        reasons.append(
            f"career months sum to {total_role_months} but total experience is "
            f"~{yoe_months:.0f} mo")

    # --- inflated-experience check (F1): claimed yoe exceeds the OBSERVED career span ---
    # Span = earliest role start_date -> REFERENCE_DATE (same pinned constant as every other
    # date computation here; deterministic run-to-run). +12 mo tolerance for rounding/gaps.
    yoe_gt_career_span = False
    career_span_months = None
    starts = [_parse_date(r.get("start_date")) for r in history]
    starts = [s for s in starts if s]
    if starts:
        first = min(starts)
        career_span_months = ((REFERENCE_DATE.year - first.year) * 12
                              + (REFERENCE_DATE.month - first.month))
        if yoe_months is not None and yoe_months > career_span_months + 12:
            yoe_gt_career_span = True
            reasons.append(
                f"claims {yoe:g} yrs experience but entire listed career spans only "
                f"~{career_span_months} mo (from {first})")

    # Informational corroboration (NOT gating): years figure stated in the candidate's own
    # summary. In the F1 honeypot family the summary states the TRUE span while the yoe field
    # is inflated — a self-contradiction that corroborates the deterministic date math.
    summary_years_stated = None
    summary_contradicts_yoe = False
    m = _SUMMARY_YEARS_RE.search((profile.get("summary") or ""))
    if m:
        try:
            summary_years_stated = float(m.group(1))
        except ValueError:
            summary_years_stated = None
    if (summary_years_stated is not None and isinstance(yoe, (int, float))
            and abs(yoe - summary_years_stated) > 4):
        summary_contradicts_yoe = True
        soft_reasons.append(
            f"summary states {summary_years_stated:g} yrs but profile field claims {yoe:g} yrs")

    # Gating checks (reliable, ~designed traps). skill_duration_gt_career is SOFT and excluded.
    gating = {
        "expert_zero_duration": expert_zero_duration,
        "expert_low_experience": expert_low_experience,
        "role_tenure_gt_career": role_tenure_gt_career,
        "career_months_gt_experience": career_months_gt_experience,
        "career_date_error": career_date_error,
        "yoe_gt_career_span": yoe_gt_career_span,
    }
    n_flags = sum(1 for v in gating.values() if v)
    checks = dict(gating)
    checks["skill_duration_gt_career"] = skill_duration_gt_career  # soft/informational
    checks["career_span_months"] = career_span_months              # informational
    checks["summary_years_stated"] = summary_years_stated          # informational
    checks["summary_contradicts_yoe"] = summary_contradicts_yoe    # soft/informational
    checks["is_honeypot"] = n_flags > 0
    checks["n_flags"] = n_flags
    checks["reason"] = "; ".join(dict.fromkeys(reasons)) if reasons else ""
    checks["soft_reason"] = "; ".join(dict.fromkeys(soft_reasons)) if soft_reasons else ""
    return checks
