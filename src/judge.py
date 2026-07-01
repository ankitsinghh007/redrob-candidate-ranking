"""
Stage C / Part 2 — LLM-as-recruiter judge. JUDGE-TIME ONLY (offline, local GPU via Ollama).

Model: Qwen2.5-7B-Instruct q5_K_M via local Ollama, temperature 0.15, format=json.
Per candidate: system = the JD rubric (recruiter contract); user = evidence_text + a compact
PRECOMPUTED FACTS block (already verified — the model must NOT recompute, only reason about the
build narrative + overall fit). Output is strict JSON validated by pydantic; one stricter retry
on parse failure, then logged.

NOT imported by rank.py. rank.py consumes the judge's precomputed scores only.
"""
from __future__ import annotations
import os, json, time
from typing import List, Optional

import ollama
from pydantic import BaseModel, Field, ValidationError, conint, confloat

MODEL = "qwen2.5:7b-instruct-q5_K_M"
TEMPERATURE = 0.15
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUBRIC_PATH = os.path.join(ROOT, "artifacts", "jd_rubric.txt")


def load_rubric() -> str:
    with open(RUBRIC_PATH, encoding="utf-8") as f:
        return f.read().strip()


# --------------------------------------------------------------------------- schema
class JudgeVerdict(BaseModel):
    fit_tier: conint(ge=0, le=4)
    fit_score: confloat(ge=0.0, le=1.0)
    key_evidence: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    availability_note: str = ""
    honeypot_suspicion: bool = False
    honeypot_reason: str = ""
    reasoning: str = ""


# --------------------------------------------------------------------------- few-shots
# Calibration anchors (compact). Shown as prior assistant turns so the tier scale is pinned.
FEWSHOTS = [
    # 1) TRUE FIT — product-company ML engineer who shipped retrieval at scale -> tier 4
    {
        "facts": "years_of_experience=7.5; yoe_fit=1.0; title_family=ai_ml; "
                 "is_services_company=False (career-level); availability: days_since_last_active=12, "
                 "recruiter_response_rate=0.78, notice_period_days=30, open_to_work=True; "
                 "location=Pune, India; TIMELINE CONSISTENCY check: PASSED — tenure, experience, and "
                 "proficiency are mutually consistent",
        "evidence": "[CURRENT ROLE] Senior ML Engineer @ a product marketplace. Built and shipped the "
                    "candidate-search ranking system: hybrid BM25 + dense retrieval over Qdrant, a "
                    "LightGBM learning-to-rank re-ranker, and an offline NDCG/MRR eval harness wired to "
                    "online A/B tests. Owned index refresh and retrieval-quality regression alerts.",
        "verdict": {
            "fit_tier": 4, "fit_score": 0.93,
            "key_evidence": ["Shipped hybrid BM25+dense retrieval over Qdrant to production",
                             "Built LightGBM learning-to-rank re-ranker with NDCG/MRR offline eval + online A/B",
                             "Owned index refresh & retrieval-quality regression"],
            "concerns": [], "availability_note": "Active (12d), responsive (0.78), 30-day notice, Pune.",
            "honeypot_suspicion": False, "honeypot_reason": "",
            "reasoning": "Product-company ML engineer who actually shipped a hybrid retrieval + LTR ranking "
                         "system with rigorous eval — squarely the must-haves, and reachable."},
    },
    # 2) KEYWORD-STUFFER — nontech title, AI skills, nothing built -> tier 0/1
    {
        "facts": "years_of_experience=9.0; yoe_fit=0.4; title_family=nontech; "
                 "is_services_company=True (career-level); availability: days_since_last_active=140, "
                 "recruiter_response_rate=0.15, notice_period_days=90, open_to_work=False; location=Jaipur, India; "
                 "TIMELINE CONSISTENCY check: PASSED — tenure, experience, and proficiency are mutually consistent",
        "evidence": "[CURRENT ROLE] Marketing Manager @ a services firm. Ran campaign calendars, managed "
                    "vendor relationships and brand collateral, coordinated event logistics and social media.",
        "verdict": {
            "fit_tier": 0, "fit_score": 0.05,
            "key_evidence": ["Career is marketing/brand management at a services firm"],
            "concerns": ["No engineering, retrieval, or ML systems built anywhere in the history",
                         "Any AI skills listed are keyword-level, not reflected in the work"],
            "availability_note": "Stale (140d) and unresponsive (0.15); not really reachable.",
            "honeypot_suspicion": False, "honeypot_reason": "",
            "reasoning": "A marketing manager whose actual work has nothing to do with ranking/retrieval; "
                         "AI keywords without any built systems — not a fit."},
    },
    # 3) HONEYPOT — impossible timeline / expert@0 -> tier 0, honeypot_suspicion true
    {
        "facts": "years_of_experience=2.7; yoe_fit=0.25; title_family=ai_ml; "
                 "is_services_company=False (career-level); availability: days_since_last_active=20, "
                 "recruiter_response_rate=0.6, notice_period_days=60, open_to_work=True; location=Bangalore, India; "
                 "TIMELINE CONSISTENCY check: FAILED — a single role's tenure exceeds the candidate's total "
                 "years of experience; 'expert' proficiency is claimed despite very low total experience",
        "evidence": "[CURRENT ROLE] Senior ML Engineer (43 mo). Led RAG and vector-search platform; "
                    "expert in LLMs, RAG, PEFT, Milvus, Weaviate, FAISS, Information Retrieval.",
        "verdict": {
            "fit_tier": 0, "fit_score": 0.02,
            "key_evidence": ["Claims a 43-month senior role and broad expert depth"],
            "concerns": ["Timeline is impossible: 43-month senior tenure but only 2.7 years total experience",
                         "Breadth of 'expert' claims is implausible at this experience level"],
            "availability_note": "Active and responsive, but the profile itself is not credible.",
            "honeypot_suspicion": True,
            "honeypot_reason": "Tenure exceeds total experience and expert breadth is impossible at 2.7 yrs.",
            "reasoning": "The profile is internally impossible (senior tenure > total experience), a classic "
                         "fabricated honeypot — tier 0 regardless of keyword fit."},
    },
    # 4) PLAIN-LANGUAGE BUILDER — non-ML title, shipped a real recsys -> tier 3/4
    {
        "facts": "years_of_experience=6.0; yoe_fit=1.0; title_family=swe; "
                 "is_services_company=False (career-level); availability: days_since_last_active=30, "
                 "recruiter_response_rate=0.7, notice_period_days=45, open_to_work=True; location=Noida, India; "
                 "TIMELINE CONSISTENCY check: PASSED — tenure, experience, and proficiency are mutually consistent",
        "evidence": "[CURRENT ROLE] Backend Engineer @ an e-commerce product company. Designed and shipped "
                    "the product recommendation system: built the candidate-retrieval service over FAISS, an "
                    "online feature store, and a re-ranking model; measured uplift with A/B tests and offline "
                    "MAP. Scaled it to tens of millions of users.",
        "verdict": {
            "fit_tier": 4, "fit_score": 0.88,
            "key_evidence": ["Shipped a production recommendation system over FAISS at tens-of-millions scale",
                             "Built retrieval service + re-ranker, validated with A/B tests and offline MAP"],
            "concerns": ["Title is 'Backend Engineer' but the work is clearly applied ML/recsys"],
            "availability_note": "Active, responsive, Noida, 45-day notice.",
            "honeypot_suspicion": False, "honeypot_reason": "",
            "reasoning": "Non-ML title but the history shows a real, scaled recommendation system with retrieval "
                         "and re-ranking — trust what they built; strong fit."},
    },
]


def timeline_status(row) -> str:
    """Human-readable TIMELINE CONSISTENCY status from the verified honeypot consistency flags
    (role_tenure_gt_career, career_months_gt_experience, expert_low_experience). PASSED means the
    profile's dates/experience/proficiency are mutually consistent; FAILED names the contradiction."""
    reasons = []
    if row.get("role_tenure_gt_career"):
        reasons.append("a single role's tenure exceeds the candidate's total years of experience")
    if row.get("career_months_gt_experience"):
        reasons.append("summed role months far exceed the stated total years of experience")
    if row.get("expert_low_experience"):
        reasons.append("'expert' proficiency is claimed despite very low total experience")
    if reasons:
        return "FAILED — " + "; ".join(reasons)
    return "PASSED — tenure, experience, and proficiency are mutually consistent"


def build_facts_block(row) -> str:
    """Compact PRECOMPUTED FACTS block from structured features (already verified)."""
    def g(k, default="?"):
        v = row.get(k)
        return default if v is None else v
    yoe = g("years_of_experience")
    return (
        f"years_of_experience={yoe}; yoe_fit={g('yoe_fit')}; title_family={g('title_family')}; "
        f"current_title={g('title')!r}; "
        f"is_services_company={g('is_services_company')} (career-level), "
        f"services_only_career={g('services_only_career')}; "
        f"availability: days_since_last_active={g('days_since_last_active')}, "
        f"recruiter_response_rate={g('recruiter_response_rate')}, "
        f"notice_period_days={g('notice_period_days')}, open_to_work={g('open_to_work')}; "
        f"location_preferred={g('location_preferred')}, country_in_india={g('country_in_india')}; "
        f"TIMELINE CONSISTENCY check: {timeline_status(row)}"
    )


SYSTEM_SUFFIX = (
    "\n\nOUTPUT: respond with ONE JSON object only, matching exactly these keys: "
    "fit_tier (int 0-4), fit_score (float 0-1), key_evidence (list of strings — quote/paraphrase "
    "REAL specifics from the candidate's evidence; never invent skills or employers), concerns "
    "(list of strings), availability_note (string), honeypot_suspicion (bool), honeypot_reason "
    "(string), reasoning (string, 1-2 sentences in a recruiter voice citing concrete facts). "
    "The PRECOMPUTED FACTS are already verified — do NOT recompute them; reason about the BUILD "
    "NARRATIVE in the evidence and the overall fit. No markdown, no commentary outside the JSON."
    "\n\nTIMELINE CONSISTENCY: the facts include a verified timeline check. CONSIDER it for "
    "plausibility — it is NOT an automatic disqualifier (a separate filter handles removal). If it "
    "reads FAILED, the profile contains an internal impossibility (e.g. claimed tenure or expertise "
    "that the candidate's total experience cannot support); a polished, keyword-perfect narrative does "
    "NOT override an impossible timeline — treat such a profile as likely fabricated, set "
    "honeypot_suspicion=true, name the contradiction in honeypot_reason, and lower fit accordingly. If "
    "it reads PASSED, you may cite the consistent timeline as supporting evidence for a genuine fit."
)


def _messages(rubric, facts, evidence):
    msgs = [{"role": "system", "content": rubric + SYSTEM_SUFFIX}]
    for fs in FEWSHOTS:
        msgs.append({"role": "user",
                     "content": f"PRECOMPUTED FACTS (verified):\n{fs['facts']}\n\n"
                                f"CANDIDATE EVIDENCE:\n{fs['evidence']}"})
        msgs.append({"role": "assistant", "content": json.dumps(fs["verdict"])})
    msgs.append({"role": "user",
                 "content": f"PRECOMPUTED FACTS (verified):\n{facts}\n\nCANDIDATE EVIDENCE:\n{evidence}"})
    return msgs


def _call(messages, stricter=False):
    if stricter:
        messages = messages + [{
            "role": "user",
            "content": "Your previous reply was not valid JSON for the schema. Reply again with "
                       "ONLY the JSON object, all required keys, correct types. No other text."}]
    resp = ollama.chat(
        model=MODEL, messages=messages, format="json",
        options={"temperature": TEMPERATURE, "num_ctx": 8192})
    return resp["message"]["content"]


def judge_candidate(row, rubric=None):
    """Judge one candidate row (dict-like with feature fields + evidence_text).
    Returns (JudgeVerdict | None, meta dict with latency/parse info)."""
    rubric = rubric or load_rubric()
    facts = build_facts_block(row)
    evidence = (row.get("evidence_text") or "")[:6000]
    messages = _messages(rubric, facts, evidence)

    t0 = time.time()
    raw = _call(messages)
    parse_err = None
    for attempt in (1, 2):
        try:
            verdict = JudgeVerdict.model_validate_json(raw)
            return verdict, {"latency_s": time.time() - t0, "attempts": attempt, "ok": True}
        except (ValidationError, ValueError) as e:
            parse_err = str(e)[:200]
            if attempt == 1:
                raw = _call(messages, stricter=True)
    return None, {"latency_s": time.time() - t0, "attempts": 2, "ok": False,
                  "error": parse_err, "raw": raw[:400]}


def ensure_model():
    have = {m.model for m in ollama.list().models}
    if MODEL not in have:
        print(f"pulling {MODEL} ...")
        ollama.pull(MODEL)
    print(f"judge model ready: {MODEL}  (present={MODEL in {m.model for m in ollama.list().models}})")


if __name__ == "__main__":
    ensure_model()
