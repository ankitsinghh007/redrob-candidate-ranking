# Stage A — Spec verification + feature-layer preview

Status: **feature logic validated on a small sample. No full run, no embeddings, no LLM, no models.**
Scope of this step: verify the submission spec verbatim, build `src/features.py` (3 functions),
preview on 15 candidates incl. the two diagnostic IDs, and calibrate the honeypot detector.

---

## Part 1 — Spec verification (VERBATIM)

### (a) Compute budget — and what it actually applies to
From `submission_spec.docx` §3 "Compute constraints" (verbatim table):
> Total runtime | **≤ 5 minutes wall-clock**
> Memory | **≤ 16 GB RAM**
> Compute | **CPU only — no GPU during ranking**
> Network | **Off — your ranking code must not make external API calls (no OpenAI, Anthropic, Cohere, Gemini, or any hosted LLM service)**
> Disk | **≤ 5 GB intermediate state**

> "You CANNOT, **during the ranking step**: Call hosted LLM APIs. Use GPUs. Exceed the runtime/memory limits."

> "Plan for a small ranker over **precomputed features, indexes, or compact local models**."

**The decisive line on offline preprocessing** — `submission_spec.docx` §10.3 (verbatim):
> "If your system requires pre-computation (e.g., generating embeddings), document this clearly —
> **pre-computation may exceed the 5-minute window, but the ranking step that produces the CSV
> must complete within it.**"

Reproducibility corroboration — `submission_metadata_template.yaml`:
> `reproduce_command: "...The single command that produces submission.csv from candidates.jsonl.
> Should run end-to-end within 5 minutes on CPU with 16GB RAM and no network."`
> plus declarable fields `pre_computation_required` / `pre_computation_time_minutes`.

**BUDGET VERDICT (recorded in DECISIONS.md → Open Questions → RESOLVED):**
The 5-min / 16 GB / CPU-only / network-off budget governs **only the ranking step** (`rank.py` →
CSV). **Offline preprocessing — embeddings, indexes, and the LLM-judge — is permitted and may
exceed 5 minutes.** We precompute heavy artifacts offline, bake them into files, and keep `rank.py`
a fast CPU-only loader + combiner. The repo must ship those artifacts or a script that regenerates
them (§10.3). At Stage 3 the **ranking step** is reproduced in a sandboxed Docker matching the limits.

### (b) Scoring formula + relevance-tier definition
`submission_spec.docx` §4 (verbatim):
> NDCG@10 | 0.50 | Quality of your top-10 picks
> NDCG@50 | 0.30 | Quality of your top-50 picks
> MAP (Mean Avg Precision) | 0.15 | Precision across all relevance levels
> P@10 | 0.05 | Fraction of top-10 that are "relevant" **(tier 3+)**
> "Final composite = **0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10**"

Tiebreaks (verbatim): "Higher P@5 wins. Higher P@10 wins. Earlier submission timestamp wins."
→ Relevance is **tiered**; tier 3+ = "relevant", and (per §7) honeypots = tier 0. **Top-10 quality
is 50% of the score** — the head of the list matters most; a perfect top-10 + P@5 tiebreak focus pays.

### (c) Honeypot policy + disqualification threshold
`submission_spec.docx` §7 "Honeypot warning" (verbatim):
> "The dataset contains a small number (**~80**) of honeypot candidates with **subtly impossible
> profiles** (e.g., 8 years of experience at a company founded 3 years ago; "expert" proficiency in
> 10 skills with 0 years used). These are **forced to relevance tier 0** in the ground truth."
> "We use the honeypot rate as a Stage 3 filter: **submissions with honeypot rate > 10% in top 100
> are disqualified.**"
> "You can identify honeypots through careful profile inspection. We expect a good ranking system to
> naturally avoid them; you don't need to special-case them."

### (d) Model size / external models / reproducibility
- No hard model-size cap stated; the operative limits are **≤ 5 GB intermediate disk**, **≤ 16 GB
  RAM**, **CPU-only**, **no network** — all scoped to the ranking step.
- External models are allowed **only if local + within budget** ("compact local models"); **no hosted
  LLM APIs at ranking time** (OpenAI/Anthropic/Cohere/Gemini named explicitly).
- Reproducibility (§10.3): repo must include README with a **single reproduce command**, full source
  ("no hidden steps, no manual edits"), **pre-computed artifacts or a script that produces them**,
  pinned `requirements.txt`, and `submission_metadata.yaml`. Stage 4 also checks **git-history
  authenticity** (real iteration vs single dump) — so commit the work incrementally.

### Honeypot / trap construction patterns documented in the bundle
> Note: `redrob_signals_doc.docx` documents **only the 23 behavioral signals** — it does NOT contain
> trap construction patterns. The trap descriptions live in the **spec §7**, the **README**, and the
> **JD's participant note**. Consolidated list:
1. **Subtly-impossible honeypots (~80, tier 0):** experience > company age ("8 yrs at a company
   founded 3 yrs ago"); "expert" proficiency in many skills with **0 months used**. (spec §7)
2. **Keyword stuffers:** profiles loaded with AI skill keywords but no real backing — the skills[]
   array is engineered noise; "find candidates whose skills section contains the most AI keywords…
   is a trap we've explicitly built into the dataset." (JD note)
3. **Plain-language "Tier-5" fits:** genuinely strong candidates who describe real ML/recsys/search
   work in plain prose **without buzzwords** and often under a non-ML title. (JD note + README)
4. **Behavioral twins:** near-identical-on-paper candidates separated only by behavioral signals —
   the available one (logs in, responds) beats the perfect-on-paper-but-unreachable one. (README; JD)

---

## Part 2 — Feature layer (`src/features.py`, stdlib-only, no I/O, no models)

Three functions, importable from a future CPU-only `rank.py`:

- **`build_evidence_text(candidate)`** → the string a recruiter actually reads: `headline` +
  `summary` + each `career_history[].description`, ordered **most-recent-first**, with the current
  role's description repeated once to **upweight recency** for downstream vectorizers.
  **The `skills[]` array is deliberately excluded** (engineered noise).
- **`structured_features(candidate)`** → flat dict: `title_family` ∈ {ai_ml, data, swe, nontech}
  (discipline-engineer guard so "Mechanical/Civil Engineer" → nontech); `is_services_company` +
  `services_ratio` + `services_only_career` (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini/HCL/
  Tech Mahindra + more); `yoe_fit` (band 5–9, sweet 6–8); `career_coherence`; `product_vs_services`
  (recency-weighted, −1…+1); `days_since_last_active`; `recruiter_response_rate`; `notice_period_days`;
  `willing_to_relocate`; `location_preferred` (Pune/Noida/Hyderabad/Mumbai/Delhi-NCR); `country_in_india`.
  **Sentinels `github_activity_score == -1` and `offer_acceptance_rate == -1` → `None` (neutral),
  never treated as low.**
- **`honeypot_flags(candidate)`** → per-check booleans + `is_honeypot` + `n_flags` + `reason`.

`REFERENCE_DATE = 2026-06-30` centralizes recency math.

---

## Part 3 — Preview on a sample + honeypot calibration

Ran all three functions over `sample_candidates.json` (50) + first 2,000 of `candidates.jsonl`,
plus the two diagnostic IDs pulled by direct lookup.

### Diagnostic targets — both behave exactly as required
| ID | title | family | yoe | is_honeypot | why |
|---|---|---|---|---|---|
| **CAND_0000001** | Backend Engineer | swe | 6.9 | **False ✓** | plain-language ML-adjacent fit (Spark/Airflow + "transitioning to ML", uses Milvus/LoRA in career text). Not a trap. |
| **CAND_0016000** | Full Stack Developer | swe | 2.0 | **True ✓ (n=2)** | `expert_zero_duration` + `expert_low_experience`: "'expert' in TypeScript/Go/Docker/Hadoop/Photoshop with 0 months used" at only 2 yrs total. Canonical honeypot. |

### 15-candidate preview (abridged)
`title_family` classification is correct across the spread: Backend/Frontend/Software/QA → swe;
Operations Manager/Customer Support/Marketing Manager/Accountant → nontext; Data Engineer → data;
Recommendation Systems Engineer → ai_ml. Services flag correctly fires on Mindtree (0000001) and
Wipro (0000002). `yoe_fit`, `days_since_last_active`, `location_preferred` all populate sensibly.

### Honeypot detector calibration — the key Stage-A finding
First pass flagged **13.55%** of the sample — implausible vs the spec's ~80/100k (≈0.08%). Per-check
rates over 10k isolated the cause:

| check | rate | verdict |
|---|---|---|
| expert_zero_duration | 0.01% | reliable (designed trap) |
| expert_low_experience | 0.00% | reliable |
| role_tenure_gt_career | 0.02% | reliable (company-age proxy) |
| career_months_gt_experience | 0.03% | reliable |
| career_date_error | 0.00% | reliable |
| **skill_duration_gt_career** | **13.35%** | **GENERATOR NOISE — demoted to soft/non-gating** |

The synthetic generator assigns each skill's `duration_months` **independently of experience**, so
"skill used longer than career" fires on ~1 in 8 normal candidates. **Fix:** `skill_duration_gt_career`
is now **soft/informational** and excluded from `is_honeypot`; the gate uses only the five reliable
checks.

**After the fix:**
- Sample-50 + first-2000 sweep: **1 / 2,000 flagged (0.05%)**.
- First **50,000** of the pool: **22 flagged → 0.044% → ~44 per 100k**. Same order as the spec's ~80,
  with **near-zero false positives**.
- **Side benefit:** the fix rescued `CAND_0000031` (a genuine *Recommendation Systems Engineer*,
  yoe 6, response-rate 0.91, Hyderabad) that the noisy check had wrongly flagged — exactly the kind
  of strong top-of-list candidate we must not demote.

**Precision/recall stance:** the detector currently catches **~44/80 (~55% recall) at very high
precision**. We deliberately favor precision (no false honeypot on a real fit) over recall — the
ranker itself will naturally bury most undetected honeypots (they're nontech/incoherent), and the
gate is a safety net for the top-100 DQ rule, not the primary defense. Recall can rise later with a
real company-age heuristic (the "founded 3 yrs ago" pattern needs company-founding data we don't have).

### Data-quality observations surfaced during the preview
- **Templated, sometimes self-contradicting `summary` text:** e.g., CAND_0000003 (*Customer Support*)
  and CAND_0000005 (*Accountant*) both have summaries claiming "I've spent my career in marketing
  manager roles." The `summary` field is boilerplate and can contradict `current_title`. → **Weight
  `career_history[].description` above `summary`** in the evidence layer (already ordered after it).
- Joke employers (Initech, Acme Corp, Dunder Mifflin) and a near-uniform skill distribution confirm
  the synthetic origin — don't build company-prestige features from raw company names.

---

## What's NOT done yet (next stages)
- No embeddings, no index, no LLM-judge, no model — by design.
- Open items remain in DECISIONS.md: tier definition for optimization, behavioral weighting policy,
  geography hard-vs-soft, honeypot hard-drop-vs-demote + recall tuning.
