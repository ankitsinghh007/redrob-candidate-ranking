# Stage C — LLM-as-recruiter judge: design + preview calibration

Status: **judge built and previewed on 17 curated candidates. NOT run on the full shortlist.**
Model: **Qwen2.5-7B-Instruct q5_K_M** via local Ollama (temp 0.15, `format=json`), no hosted API.

---

## Part 0 — carryover fixes + deps
- **title_family fix:** titles with a parenthetical/suffix specialty qualifier (ML|NLP|AI|GenAI|LLM|
  Data Science|Machine Learning) now route to `ai_ml`, e.g. "Software Engineer (ML)". Rebuilt
  `features.parquet` (~8.5s). **142 candidates changed `swe → ai_ml`** (all "Senior Software
  Engineer (ML)"). Guards intact: Mechanical Engineer→nontech, Backend Engineer→swe, Data
  Scientist→ai_ml. (Added `open_to_work` to features for the judge's availability block.)
- **Widened shortlist → `artifacts/shortlist.parquet`, 985 candidates.** Composition:
  `retrieval_rank` 501, `both` 299, `build_signal` 185; family mix ai_ml 807 / swe 97 / data 78 /
  nontech 3. **4 honeypots KEPT IN and flagged** (`is_honeypot_gated`) so the judge can be validated
  independently. Bundles the structured features + evidence_text the judge reads.
- **requirements.txt:** added a commented **"JUDGE-TIME ONLY (offline, GPU)"** block — `ollama`,
  `pydantic`, `tenacity`. Rank-time section untouched.

## Part 1 — JD rubric
`artifacts/jd_rubric.txt` distills the JD into MUST-HAVES / NICE-TO-HAVE / EXPLICIT DISQUALIFIERS /
IDEAL / a 0–4 TIER SCALE / and the two standing instructions ("trust what they BUILT over what they
CLAIM"; "down-weight unavailable candidates"). Recorded verbatim in DECISIONS.md. It is pasted as the
judge's system role.

## Part 2 — judge (`src/judge.py`)
- Per candidate: **system** = rubric + strict-JSON output contract; **4 few-shot anchors** (true fit→4,
  keyword-stuffer→0/1, honeypot→0+suspicion, plain-language builder→4) to pin the tier scale;
  **user** = a compact **PRECOMPUTED FACTS** block (yoe, yoe_fit, title_family, services flags,
  availability, geography — "already verified, do NOT recompute") + the candidate's evidence_text.
- Output validated by a **pydantic** schema (`fit_tier 0-4`, `fit_score 0-1`, `key_evidence[]`,
  `concerns[]`, `availability_note`, `honeypot_suspicion`, `honeypot_reason`, `reasoning`). One
  stricter retry on parse failure, then logged.

## Part 3 — preview (17 curated candidates spanning the spectrum)

| candidate | bucket | title | gated_hp | fit_tier | score | hp_susp |
|---|---|---|---|---|---|---|
| CAND_0007411 | top_fused | Senior ML Engineer | – | 4 | 0.95 | F |
| CAND_0055905 | top_fused | Senior ML Engineer | – | 4 | 0.95 | F |
| CAND_0008425 | top_fused | Senior NLP Engineer | – | 4 | 0.95 | F |
| CAND_0071974 | top_fused | Senior AI Engineer | – | 4 | 0.95 | F |
| CAND_0085706 | plainlang_nonml | Senior Software Engineer | – | 3 | 0.75 | F |
| CAND_0013392 | plainlang_nonml | Senior Software Engineer | – | 2 | 0.56 | F |
| CAND_0093587 | plainlang_nonml | Senior Software Engineer | – | 2 | 0.55 | F |
| CAND_0091914 | plainlang_nonml | Senior Software Engineer | – | 2 | 0.55 | F |
| CAND_0000031 | rescued_recsys | Recommendation Systems Engineer | – | 3 | 0.75 | F |
| **CAND_0093547** | **honeypot** | Senior ML Engineer | **True** | **3** | 0.75 | **F** |
| **CAND_0001610** | **honeypot** | ML Engineer | **True** | **3** | 0.75 | **F** |
| **CAND_0019480** | **honeypot** | NLP Engineer | **True** | **3** | 0.75 | **F** |
| CAND_0016000 | honeypot_forced | Full Stack Developer | True | 1 | 0.35 | F |
| CAND_0000903 | services_only | DevOps Engineer | – | 1 | 0.35 | F |
| CAND_0000952 | services_only | Full Stack Developer | – | 1 | 0.35 | F |
| CAND_0000083 | keyword_stuffer | Graphic Designer | – | 1 | 0.25 | T |
| CAND_0000399 | keyword_stuffer | Business Analyst | – | 1 | 0.25 | F |

### Answers to the validation questions
- **Do real builders (incl. surfaced non-ML titles) get HIGH tier?** ✅ The 4 top AI/ML engineers all
  tier 4 (0.95). The rescued recsys engineer tier 3. The `plainlang_nonml` swe candidates are correctly
  tier 2 (one tier 3) — on inspection they are **data-infra engineers transitioning to ML** (Kafka/Spark
  pipelines, "model-serving integration", "interested in ML"), i.e. genuine stretch candidates, NOT
  shipped-a-ranking-system builders. **Tier 2–3 is the right call, not a miss** — the judge discriminates
  "built it" from "near it." (So the 67% "high-tier among builders" number reflects correct nuance, not error.)
- **Do keyword-stuffers get LOW tier despite rich AI skills?** ✅ Both tier 1 (0.25). Concerns explicitly:
  "no production ML/AI systems built anywhere", "keyword-stuffed self-description". One flagged hp_susp=True.
- **Do services-only seniors get disqualifier pressure?** ✅ Both tier 1; concerns cite "services-only
  career with no product-company ML/AI experience" — the JD disqualifier applied from the career narrative.
- **JSON parse success?** ✅ **17/17 (100%)**, zero malformed, zero retries needed.
- **Does reasoning cite real evidence without hallucinating?** ✅ Spot-checked: cited specifics
  ("PhonePe", "Pinecone", "NDCG@10 by 18%", "LoRA/QLoRA") are literally present in the evidence text.
  No invented employers/skills observed. Availability is reasoned (e.g. CAND_0007411 flagged "last active
  6 months ago but open to work").
- **Latency / full-run extrapolation:** mean **8.1s/candidate** (median 7.3s) → **~133 min (2.2 h)** for
  the 985-candidate shortlist on this box.

### ⚠️ KEY FINDING — the judge does NOT independently catch honeypots
**All 3 "clever" honeypots scored tier 3 with `honeypot_suspicion=False`.** They have keyword-perfect,
fabricated-but-plausible narratives ("Shipped a ranking pipeline at PhonePe: embedding generation →
Pinecone retrieval → XGBoost re-scoring…"), and their **impossibility lives only in the date/tenure math**
(e.g. 2.9 yrs total experience but career tenures summing to ~74 months), which is **not in the prose the
judge reads** — so the LLM takes the narrative at face value. It even *noticed* the symptom ("title-chaser,
roles 31+21+9 mo") but did not connect it to a fabricated timeline. The cruder honeypot (CAND_0016000,
no ML narrative) landed tier 1 but still `hp_susp=False`.

**Why this is OK for the pipeline, and what it means:**
1. The **deterministic Stage-A honeypot gate is the primary, non-negotiable defense** — it already flags
   all 4 of these (`is_honeypot_gated=True`). The final top-100 will **hard-drop gated honeypots regardless
   of judge output**, so the >10% DQ rule is protected.
2. This empirically confirms the spec's premise: a keyword-perfect profile fools **both** embedding retrieval
   (Stage B put one at rank 33) **and** a strong 7B LLM judge. Neither semantic method is a honeypot defense
   on its own — only the explicit consistency checks are.
3. **Recommended for the full run (defense-in-depth, needs your OK):** surface the verified consistency
   signals (`role_tenure_gt_career`, `career_months_gt_experience`, `expert_low_experience`) into the judge's
   PRECOMPUTED FACTS block as a one-line "TIMELINE CONSISTENCY" fact. That lets the judge corroborate tier-0 +
   honeypot_suspicion. It stays faithful to the "pass verified facts, don't recompute" design. The hard gate
   remains the real filter; this is belt-and-suspenders.

### Minor note — temperature variance
At temp 0.15 there is small run-to-run movement at tier boundaries (services-only seniors were tier 2/1 in a
first pass, tier 1/1 here; one keyword-stuffer's hp_susp flipped). Acceptable for a single full pass; if we
want determinism on borderline cases we can drop temp to 0 or take the median of 2 samples for tier∈{2,3}.

---

## Decision for the full pass (pending your call)
- Shortlist = **985** candidates; projected judge runtime **~2.2 h** offline.
- **Honeypots:** keep the deterministic gate as the hard filter (decided). **Open:** add the timeline-
  consistency fact to the judge prompt for defense-in-depth? (recommended, ~no cost).
- Then: combine `fit_tier`/`fit_score` (primary) with structured features + availability into the listwise
  re-rank, hard-drop gated honeypots, emit top-100. Open items unchanged in DECISIONS.md.

---

# Stage C-final — timeline fact wired in + full judge run

## Part 1 — TIMELINE CONSISTENCY fact (defense-in-depth)
Added one verified line to the judge's PRECOMPUTED FACTS block, derived from the honeypot consistency
flags (`role_tenure_gt_career` / `career_months_gt_experience` / `expert_low_experience`):
`TIMELINE CONSISTENCY check: PASSED — …` or `FAILED — <specific contradiction>`. The system prompt
frames it as a plausibility signal to **consider** (a polished narrative does NOT override an impossible
timeline → set honeypot_suspicion, lower fit) and explicitly **NOT an auto-disqualifier** (the
deterministic gate does the filtering). PASSED may be cited as positive evidence for a genuine fit. The
4 few-shot anchors were updated to carry the line (honeypot anchor shows FAILED→suspicion).

## Part 2 — micro-validation (4 honeypots + 4 genuine fits), before/after

| candidate | kind | timeline | tier before→after | score before→after | suspicion before→after |
|---|---|---|---|---|---|
| CAND_0093547 | honeypot | FAILED | 3 → **0** | 0.75 → 0.05 | False → **True** |
| CAND_0001610 | honeypot | FAILED | 3 → 3 | 0.75 → 0.75 | False → **True** |
| CAND_0019480 | honeypot | FAILED | 3 → 3 | 0.75 → 0.75 | False → **True** |
| CAND_0016000 | honeypot | FAILED | 1 → 1 | 0.35 → 0.25 | False → False |
| CAND_0007411 | genuine | PASSED | 4 → 4 | 0.95 → 0.95 | False → False |
| CAND_0055905 | genuine | PASSED | 4 → 4 | 0.95 → 0.95 | False → False |
| CAND_0008425 | genuine | PASSED | 4 → 4 | 0.95 → 0.95 | False → False |
| CAND_0071974 | genuine | PASSED | 4 → 4 | 0.95 → 0.95 | False → False |

- **(a) Honeypots moved: 3/4 now `honeypot_suspicion=True`** (one also dropped tier 3→0). The 4th
  (CAND_0016000) is already tier 1 and deterministically gated.
- **(b) Genuine fits preserved: 4/4 tier unchanged, max |score Δ| = 0.000.** The new fact did not perturb
  calibration → no prompt softening needed; proceed.

## Part 3 — resumable full run (`src/run_judge.py`)
- **Checkpoint:** each verdict appended to `artifacts/judgments.jsonl` (keyed by candidate_id),
  consolidated to `artifacts/judgments.parquet` at the end.
- **Resumable:** loads done candidate_ids on startup and SKIPs them — safe to re-run after interruption
  (smoke-tested: 3 judged → re-run skipped them).
- **Robust:** each call wrapped in `tenacity` retry (4 attempts, exp backoff) for transient Ollama
  errors; permanent failures (and parse failures after the judge's own stricter retry) logged to
  `artifacts/judge_failures.jsonl` and the run continues. Progress printed every 50.
- Launched over all **985** candidates (~2.2 h offline). `judgments.parquet` is the core Stage-D input.

### Final run results
- **985 / 985 judged, 0 failures, 100% JSON parse success.** Coverage verified: `judgments.parquet`
  has a row for every shortlisted candidate (`set(judgments)==set(shortlist)`, unaccounted = 0).
- Runtime: 609 done in session 1, remaining 376 in **41.9 min** after a server restart + resume
  (resume worked: skipped the 609, finished the rest). End-to-end ≈ the projected ~2.2 h.
- **Tier distribution (0–4):** tier4 = **47**, tier3 = **548**, tier2 = 382, tier1 = 8, tier0 = 0.
  → ample headroom for the top-100 (47 tier-4 + 548 tier-3). Top fits are the expected Senior
  AI/ML/NLP engineers at fit_score 0.92–0.95.
- **Honeypots:** all 4 gated honeypots in the shortlist have `timeline_failed=True`; 3/4 also raised
  `honeypot_suspicion=True` (CAND_0093547 scored tier 2 / suspicion False this pass — temp-0.15
  boundary variance). **All 4 will be HARD-DROPPED by the deterministic gate in Stage D regardless**,
  so they cannot reach the top-100; the judge signal is corroboration only. The judge also
  independently suspected 2 *non-gated* candidates (5 suspicions total) — a bonus signal for Stage D.
- Artifacts: `artifacts/judgments.parquet` (0.18 MB; the core Stage-D input) + `judgments.jsonl`
  (checkpoint) + `judge_failures.jsonl` (empty).

**Stage C done. Ready for Stage D (listwise re-rank + honeypot hard-gate → top-100).**
