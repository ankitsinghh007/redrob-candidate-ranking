# Redrob Candidate Ranking — Decision Log

Single source of truth for design decisions. Quotable for the deck. Append to the
**Changelog** at the end of every step. Keep entries terse and evidence-backed.

---

## Problem & Scoring

- **Task:** Rank the **top 100** candidates from a **single global pool of 100,000**
  (`candidates.jsonl`) against **one free-text job description** (`job_description.docx`,
  "Senior AI Engineer — Founding Team", Pune/Noida, 5–9 yrs). Output CSV:
  `candidate_id,rank,score,reasoning` (exactly 100 rows, ranks 1–100 once, score
  non-increasing, tie-break candidate_id ascending).
- **Unsupervised.** No labels in the data; the JD itself is the rubric. Scored once,
  after close, against a **hidden ground-truth relevance tier** (no leaderboard).
- **Scoring formula (verbatim):**
  `Final composite = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10`.
  → **Top-10 quality dominates (50%).** Get the head of the list right above all else.
- **Relevance tiers:** P@10 measures "Fraction of top-10 that are 'relevant' **(tier 3+)**";
  honeypots are "**forced to relevance tier 0**". So tiers run 0..N, 3+ = relevant, 0 = trap.

## Key Data Facts

- **`skills[]` array is engineered noise.** 133 skills each appear ~12k× regardless of role;
  a Marketing Manager can list "Fine-tuning LLMs". Keyword/skill matching is a deliberate trap.
  → **Evidence of fit = free text** (`summary` + `career_history[].description`) + title
  trajectory, NOT the skills array.
- **~80 honeypots**, subtly impossible profiles, forced to tier 0. **DQ rule: honeypot rate
  > 10% in top-100 → disqualified at Stage 3.** Naive detector (≥5 expert@0-dur) found only 8;
  need conservative-but-broad consistency checks.
- **True target pool is tiny:** AI Engineer 21, Senior AI Engineer 4, ML Engineer 167 / 100k.
  JD: "not expecting to find many matches… rather 10 great matches than 1000 maybes."
- **Sentinels:** `github_activity_score = -1` (64.6%) and `offer_acceptance_rate = -1` (59.6%)
  mean **no data → neutral/missing, never "low"**.
- **`education.tier` (tier_1..4) = institution prestige, NOT the relevance tier.** Do not conflate.
- **Negative signals matter:** JD explicitly down-ranks title-chasers, career-long services-only
  (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini/HCL/Tech Mahindra), CV/speech/robotics-only,
  and unavailable candidates (stale `last_active_date`, low `recruiter_response_rate`).

## Architecture

Target pipeline (all heavy compute **offline**; `rank.py` stays CPU-only / ≤5 min / no network):

1. **Ensemble retrieval** (offline) — shortlist candidates for the JD via hybrid signals
   (dense embeddings over evidence text + BM25/lexical over descriptions + structured
   pre-filters: title_family, services flag, yoe, geography). Cuts 100k → a few thousand.
2. **LLM-judge (offline only)** — score the shortlist for genuine JD fit by reading evidence
   text (career history, not skills). Runs OFFLINE during preprocessing; its outputs are baked
   into artifacts. **Never called from rank.py** (no hosted LLM / no network at ranking time).
3. **Listwise re-rank** — combine judge scores + structured features into a final ordering of
   the head of the list (top-10 quality is 50% of score).
4. **Honeypot gate** — `honeypot_flags()` demotes/removes trap profiles before emitting the
   top 100, protecting against the >10% DQ rule.

### Honeypot defense — deliberate design stance (layered, deterministic-first)
Empirically (Stage B + C), keyword-perfect honeypots fool BOTH dense retrieval (one reached rank
33) AND a 7B LLM judge reading the narrative (all 3 clever ones scored tier 3, suspicion=False). The
impossibility lives in date/tenure math, not prose. So:
1. **PRIMARY (non-negotiable): the deterministic Stage-A honeypot gate.** Gated honeypots are
   HARD-DROPPED from the final top-100 regardless of any model score. This is what protects the
   ">10% honeypots in top-100 → DQ" rule. We never rely on the LLM to filter them.
2. **SECONDARY (corroboration): a verified TIMELINE CONSISTENCY fact in the judge prompt**
   (role_tenure_gt_career / career_months_gt_experience / expert_low_experience → PASSED/FAILED).
   Framed as a plausibility signal, NOT an auto-disqualifier. Micro-validation: it flipped 3/4
   honeypots to honeypot_suspicion=True (one to tier 0) while leaving genuine fits UNCHANGED
   (4/4 same tier, max |score Δ| = 0.000). It also lets the judge cite a consistent timeline as
   positive evidence for genuine fits.

**Compute split (resolved — see Open Questions):** offline preprocessing/embedding/index
building/LLM-judging is **permitted and may exceed 5 min**; only the **ranking step that emits
the CSV** must obey ≤5 min, ≤16 GB, CPU-only, network-off. Artifacts (embeddings, indexes,
judge scores, features) are precomputed and loaded by `rank.py`.

## JD Query Strings (core design artifact — Stage B/Part 4)

Two query strings distilled from `job_description.docx`, embedded with both models (E5 with
`query:` prefix; BGE with the instruction `Represent this sentence for searching relevant
passages: `). Stored verbatim in `artifacts/jd_queries.json`. Retrieval fit uses
`pos_sim − 0.5·neg_sim` per model, RRF-fused across models.

**POSITIVE** (from "what you'd actually be doing" + "absolutely need" + "ideal candidate"):
> Senior AI Engineer who owns the intelligence layer of a product: the ranking, retrieval, and
> matching systems that decide what recruiters and candidates see. Production experience with
> embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5) deployed
> to real users, including handling embedding drift, index refresh, and retrieval-quality
> regression. Production experience with vector databases or hybrid search infrastructure
> (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS). Strong Python and code
> quality. Designs rigorous evaluation frameworks for ranking systems using NDCG, MRR, MAP,
> offline-to-online correlation, and A/B testing. Has shipped at least one end-to-end ranking,
> search, or recommendation system to real users at meaningful scale, at a product company rather
> than pure services. Roughly 6 to 8 years total experience with 4 to 5 years in applied ML/AI at
> product companies. Nice to have: LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank models
> (XGBoost-based or neural), HR-tech or recruiting or marketplace products, distributed systems and
> large-scale inference, open-source ML contributions. Scrappy product engineer who ships working
> systems fast. Based in or willing to relocate to Pune or Noida.

**NEGATIVE** (from the "explicitly do NOT want" list):
> Not a fit. Title-chasers who switch companies every 1.5 years to climb from Senior to Staff to
> Principal. Framework enthusiasts whose GitHub is full of LangChain tutorials and hot-framework
> demos, whose AI experience is only recent LangChain calls to OpenAI with no pre-LLM-era ML
> production experience. People who have only ever worked at IT services and consulting firms (TCS,
> Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL, Tech Mahindra) for their entire career with
> no product-company experience. Pure-research candidates from academic or research-only labs with
> no production deployment. People whose primary expertise is computer vision, speech, or robotics
> without NLP or information-retrieval exposure. Senior engineers who have not written production
> code in the last 18 months because they moved into architecture or tech-lead roles. Keyword-stuffed
> profiles that list many AI skills but show no real systems built. Candidates who are unavailable,
> inactive, or unresponsive to recruiters.

## JD Rubric (judge contract — core deck artifact, Stage C/Part 1)

Verbatim copy of `artifacts/jd_rubric.txt`, pasted as the judge's system role. Distilled
faithfully from `job_description.docx`.

```
ROLE: Senior AI Engineer — Founding Team @ Redrob AI (Series A, AI-native talent
intelligence). Pune/Noida, hybrid. You own the intelligence layer: the ranking,
retrieval, and matching systems behind the product.

You are a recruiter executing THIS rubric to rank candidates for THIS role. Score a
candidate on genuine fit, not keyword overlap.

MUST-HAVES (the bar — strong candidates show most of these in what they actually BUILT):
- Production experience with embeddings-based retrieval deployed to real users
  (sentence-transformers / OpenAI embeddings / BGE / E5 / similar; embedding drift,
  index refresh, retrieval-quality regression).
- Production experience with vector databases or hybrid search infrastructure
  (Pinecone / Weaviate / Qdrant / Milvus / OpenSearch / Elasticsearch / FAISS / similar).
- Strong Python and real code quality.
- Has designed evaluation frameworks for ranking systems (NDCG / MRR / MAP,
  offline-to-online correlation, A/B-test interpretation).
- Has shipped AT LEAST ONE end-to-end ranking, search, or recommendation system to real
  users at meaningful scale — at a PRODUCT company, not pure services.

NICE-TO-HAVE (raise, do not gate): LLM fine-tuning (LoRA/QLoRA/PEFT); learning-to-rank
(XGBoost or neural); HR-tech / recruiting / marketplace; distributed systems / large-scale
inference; open-source ML contributions.

EXPLICIT DISQUALIFIERS (push toward low tier; the JD is emphatic about these):
- Services-only career: only ever worked at IT-services/consulting firms (TCS, Infosys,
  Wipro, Accenture, Cognizant, Capgemini, HCL, Tech Mahindra, ...) with no product-company
  experience.
- Title-chasers: hopping companies ~every 1.5 years to climb Senior→Staff→Principal.
- Framework-enthusiasts: "AI experience" is mostly recent LangChain-calling-OpenAI / tutorial
  demos, with no pre-LLM-era ML production depth.
- Research-only: pure academic/research-lab roles with no production deployment.
- CV/speech/robotics-only with no NLP / information-retrieval exposure.
- 5+ years entirely on closed-source proprietary systems with no external validation
  (papers, talks, open-source).

IDEAL CANDIDATE: ~6-8 years total, of which ~4-5 in applied ML/AI at PRODUCT companies; has
shipped an end-to-end ranking/search/recsys system at real scale; strong opinions on
retrieval, evaluation, and LLM integration backed by systems they actually built; in or
willing to relocate to Pune/Noida; reachable / active in the market.

TIER SCALE (assign fit_tier 0-4):
- 4 = excellent fit ... Top-10 material.
- 3 = strong/relevant fit ... ("relevant" = tier 3+.)
- 2 = partial/adjacent ...
- 1 = weak ...
- 0 = not a fit / trap (incl. honeypot / impossible profile).

TWO STANDING INSTRUCTIONS:
1. TRUST WHAT THEY BUILT OVER WHAT THEY CLAIM (career descriptions > summary/title).
2. DOWN-WEIGHT CANDIDATES WHO AREN'T ACTUALLY AVAILABLE (stale last_active, low response rate).
```

## Final ranking fusion (Stage D — recorded weights)

Top-100 assembly, honeypot-gated first, then tiered + blended + reranked:
- **PART 1 hard gate (first):** drop every `is_honeypot_gated==True` (4 dropped) and any tier-0
  (0) BEFORE ranking, so a honeypot can never occupy a rank slot. Pool 985 → 981.
- **PART 2 within-tier score:** primary key = `fit_tier` (4>3>2…). Within tier:
  `within_tier_score = (0.70·judge_fit_score + 0.30·structured_fit) · availability_mult`.
  - `structured_fit` (NOT skills): `0.30·yoe_fit + 0.25·((product_vs_services+1)/2) +
    0.20·career_coherence + 0.15·family_score + 0.06·location_preferred + 0.04·country_in_india`,
    minus **0.15 if services_only_career**, clamped [0,1]. `family_score`: ai_ml 1.0 / data 0.5 /
    swe 0.3 / nontech 0.0.
  - `availability_mult` ∈ **[0.85, 1.0]** = `0.85 + 0.15·avail`, where `avail` = mean of recency
    (`last_active`: 30d→1, 200d→0), `recruiter_response_rate` (None→0.5), `open_to_work` (1.0/0.5),
    notice (30d→1, 180d→0). Sentinels treated neutral.
- **PART 3 listwise rerank (top zone):** the top **40** (all tier-4) are reranked by Qwen2.5-7B in a
  **windowed** scheme (window 10, step 5), **3 passes**, aggregated by **mean rank** to damp temp-0.1
  nondeterminism. Guards (enforced in code): the rerank may REORDER but may not add/remove a
  candidate, and may not place a tier-3 above a tier-4 (no tier inversion). Ranks 41–100 keep the
  `within_tier_score` order.
- **PART 4 score:** smooth strictly-decreasing curve mapped to final rank, **0.99 → 0.50** (rank 1→0.99,
  100→0.50). Guarantees monotonic-non-increasing; unique per rank so the candidate_id tie-break is
  satisfied by construction (we also sort by candidate_id asc as the final total-order tiebreaker).
- **PART 4 reasoning:** built from the judge's extracted `key_evidence` (concrete company / built
  system / metric — the JD must-haves) + a per-candidate anchor (title + yoe) + a rotated sentence
  frame + an honest `concern`. The anchor + frame rotation deliberately breaks up the dataset's
  **behavioral twins** (many candidates carry an identical fabricated evidence sentence): first-60-char
  prefix collisions dropped 44→11, first-120 9→2, all 100 distinct, every row cites a concrete fact,
  ≤240 chars, grounded only in judge-extracted facts (no hallucination).

**Honeypot/temp-noise stance (reaffirmed):** honeypots are removed **deterministically and first**
(never trusted to the LLM). The rerank's run-to-run nondeterminism (temp 0.1) is damped by 3-pass
mean-rank aggregation rather than relying on a single sample.

## Open Questions

- **[RESOLVED — Step Stage-A/Part-1] Does the 5-min/CPU/no-network budget cover the whole
  pipeline or only ranking? Is offline preprocessing allowed?**
  → **Only the ranking step.** Spec §10.3 (verbatim): *"If your system requires pre-computation
  (e.g., generating embeddings), document this clearly — **pre-computation may exceed the
  5-minute window, but the ranking step that produces the CSV must complete within it.**"*
  Compute table applies to "code that produces the submission"; the CANNOT-list is scoped
  "during the ranking step". **Verdict:** precompute embeddings/indexes/LLM-judge scores offline,
  bake into artifacts, keep `rank.py` a fast CPU-only loader+combiner. Repo must ship the
  artifacts or a script that regenerates them (§10.3).
- How exactly to define the relevance tiers we optimize toward (rubric-derived pseudo-tiers vs
  judge-derived), and how to validate without labels.
- Behavioral signals: multiplier vs additive penalty vs tie-breaker, and exact thresholds for
  "unavailable" (days_since_last_active, response_rate).
- Geography/notice/visa: hard filter vs soft down-weight (JD says "case-by-case").
- Honeypot handling: hard-drop flagged vs soft-demote, and final precision/recall of the detector.

## Changelog

- **Stage A — Data understanding (prior step):** Mapped dataset; produced `DATA_REPORT.md`.
  Confirmed single global pool + one JD, output format, scoring, ~80 honeypots, skills-as-noise.
- **Stage A — Part 0/1 (this step):** Created `requirements.txt` (living) and this log.
  Verified spec verbatim (compute budget, scoring, honeypot policy, reproducibility) and
  **resolved the offline-preprocessing question** (above). Confirmed `redrob_signals_doc.docx`
  documents only the 23 signals — trap/honeypot construction patterns live in the spec §7,
  README, and the JD's participant note (not the signals doc).
- **Stage A — Part 2/3 (this step):** Built `src/features.py` (stdlib-only; `build_evidence_text`
  excludes skills[] and upweights recent roles; `structured_features` with title_family /
  services flags / yoe_fit / coherence / geography / sentinel-safe signals; `honeypot_flags`).
  Previewed on 50 + first 2,000 candidates. **Diagnostics pass:** CAND_0000001 NOT flagged,
  CAND_0016000 flagged (expert@0-duration + expert-low-experience).
  **Honeypot calibration finding:** `skill_duration_gt_career` fired on 13.35% (synthetic
  generator assigns skill durations independently of experience) → **demoted to soft/non-gating**.
  Gate now uses 5 reliable checks → **0.044% (~44/100k)**, same order as spec's ~80, near-zero
  false positives; fix also rescued a genuine ai_ml top candidate (CAND_0000031) from a false flag.
  Stance: favor precision over recall (currently ~55% recall) — ranker buries undetected traps,
  gate protects the top-100 DQ rule. Logged DQ observation: `profile.summary` is templated and can
  contradict `current_title` → weight `career_history[].description` over `summary`. No new deps
  (features.py stdlib-only; preview used already-listed pandas). Stopped before any full run.
- **Stage B — full build + retrieval dry-run (this step):** Added BUILD-TIME deps (torch cu124,
  sentence-transformers, scikit-learn, pyarrow, tqdm). **Verified torch.cuda.is_available()==True**
  on RTX 4060 Ti before embedding. Down-weighted summary in `build_evidence_text` (career desc
  primary, summary secondary marked). Built `features/honeypot_flags/evidence_text.parquet` (100k
  each, 8.8s; honeypots=44/0.044%). Embedded all 100k with **bge-large-en-v1.5** + **e5-large-v2**
  (fp16, 512 tok, L2-norm, 0 NaN, ~35 min) → `emb_bge.npy`/`emb_e5.npy` (204.8 MB each) + query
  vecs. Recorded POSITIVE/NEGATIVE JD queries (above).
  **Retrieval dry-run (pos−0.5·neg, RRF-fused, no title filter):**
  top-50 = 100% ai_ml; must-not-miss titles (Senior AI Eng 4/4, AI Eng 21/21) fully inside top-600;
  Data Scientists correctly down-ranked (median ~3918) by the negative query.
  **Honeypot gate validated as ESSENTIAL** — pure retrieval put 4 keyword-perfect honeypots in
  top-300 (one at rank 33), all caught by the gate.
  **CAND_0000001 re-assessed:** ranks 80th pct (not shortlist) — correct, he's a transitioning
  data-infra eng (low pos-sim, services-elevated neg-sim, no build-signal), NOT a strong fit; the
  Stage-A "should rank high" prior was overstated → defer transitioner calls to the LLM judge.
  Build-signal union rescues 484 "built-a-system" fits (212 beyond top-600; 42 with non-ML titles).
  **DECISION — shortlist for judge = top-600 fused (gated) ∪ build-signal (gated) ≈ 808 candidates.**
  Logged refinement: title_family should detect "(ML)/(NLP)" suffixes. Stopped before judge stage.
- **Stage C — judge design + preview (this step):** Carryover fixes: title_family now routes
  "(ML)/(NLP)/(AI)/…" qualifiers to ai_ml (142 swe→ai_ml); added open_to_work to features.
  Widened shortlist → `shortlist.parquet` (**985**; honeypots kept in + flagged). Added JUDGE-TIME
  deps (ollama/pydantic/tenacity). Distilled `jd_rubric.txt` (recorded verbatim above). Built
  `src/judge.py` (Qwen2.5-7B-Instruct **q5_K_M** via local Ollama, temp 0.15, format=json, pydantic
  schema + 1 stricter retry, 4 few-shot anchors). **Preview on 17 curated candidates: 100% JSON
  parse, mean 8.1s/cand → ~2.2h for full 985.** Calibration ✅: top AI/ML eng→tier 4; keyword-stuffers
  & services-only→tier 1; non-ML data-eng transitioners→tier 2-3 (correct nuance); reasoning cites
  real evidence, no hallucination.
  **KEY FINDING:** the LLM judge does NOT independently catch honeypots — all 3 clever ones got
  tier 3 / honeypot_suspicion=False (keyword-perfect fabricated narratives; impossibility is in
  date math, not prose). **→ The deterministic Stage-A honeypot gate is the primary, non-negotiable
  defense (hard-drop gated honeypots from the final top-100 regardless of judge).** Confirms spec
  premise: keyword-perfect profiles fool both embeddings (rank 33 in Stage B) AND a 7B judge.
  RECOMMENDATION (pending sign-off): add a verified "TIMELINE CONSISTENCY" fact (role_tenure_gt_career
  / career_months_gt_experience / expert_low_experience) to the judge's facts block for defense-in-depth.
  Noted temp-0.15 boundary variance. Stopped before the full judge pass.
- **Stage C-final — timeline fact + full judge run (this step):** Added a verified TIMELINE
  CONSISTENCY line to the judge facts block (PASSED/FAILED from honeypot consistency flags), framed
  as a plausibility signal (not auto-DQ); updated the 4 few-shot anchors to carry it.
  **Micro-validation (4 honeypots + 4 fits): honeypots 3/4 → honeypot_suspicion=True (one tier 3→0);
  genuine fits 4/4 tier unchanged, max |score Δ| 0.000** → no softening needed. Built resumable
  `src/run_judge.py` (per-candidate append to `judgments.jsonl`, skip-done resume, tenacity retry on
  transient errors, failures logged to `judge_failures.jsonl`, consolidated to `judgments.parquet`;
  progress every 50). Smoke-tested checkpoint/resume on 3. Launched the full **985**-candidate run
  (~2.2h offline). `judgments.parquet` is the core input artifact for Stage D.
- **Stage D — final top-100 submission (this step):** `src/rank_submission.py`. Hard honeypot gate
  first (4 dropped, 0 tier-0, 0 survivors); tiered + blended `within_tier_score` (weights above);
  listwise windowed rerank of the top-40 tier-4 (3 passes, mean-rank, guards: no add/remove, no tier
  inversion); smooth 0.99→0.50 score; fact-led recruiter-note reasoning from judge `key_evidence` +
  anchor + rotated frames (twin-collision 44→11). Wrote `submission/ankitsingh058622_1300.csv` (named `team_h2s_redrob.csv` until the
  registered-ID rename) —
  **`validate_submission.py` → "Submission is valid." (exit 0).** Top-100 = 47 tier-4 + 53 tier-3,
  0 honeypots, 100/100 distinct reasonings ≤240 chars. Audit in `artifacts/submission_audit.parquet`.
  Stopped for review before treating as final.
- **Stage D-final — audit + freeze (this step):** Verification + locking only, no logic change.
  **PART 1 honeypot audit** (independently recomputed from raw `candidates.jsonl`): **0 of the top-100
  trip any hard timeline/date flag, none in top-10** — no DQ risk. **PART 2 boundary:** tier-consistent
  cut; exactly one strong-fit (judge 0.85) relegated below 100 — CAND_0092278 at rank 106 — correctly,
  because it is unavailable (235d idle, 7% response, not open), i.e. the JD's "down-weight the unreachable"
  case; surfaced, not changed. **PART 3 freeze:** cached the temp-0 single-pass rerank order to
  `artifacts/frozen_rerank_order.json`; `rank_submission.py` reads it if present (no LLM on regen);
  CSV written with LF newlines; **byte-identical across 3 runs (A==B==C, md5 0673da50…).**
  **PART 4:** regenerated from frozen order → `validate_submission.py` "Submission is valid." (exit 0).
  Filename placeholder retained pending registered participant ID. Stopped for final review.
- **Eval harness — gold set + metrics + ablations + traces (this step):** Built an INDEPENDENT
  evaluation harness (NOT using the judge's own scores as labels — that's circular). `eval/export_gold.py`
  → `eval/gold_set.jsonl` (47 stratified candidates: top-10 / tier-boundary 44-52 / cut-line 90-110
  incl CAND_0092278 / deep 300-800 / 4 gated honeypots / 3 keyword-stuffers / 3 plain-language builders;
  each with facts + FULL evidence + blank `label_tier`) + `gold_labeling_sheet.md`. `eval/metrics.py`
  (NDCG@10/50, MAP, P@10, composite 0.50/0.30/0.15/0.05; tier>=3 relevant) — built, NOT run (awaits
  labels). `eval/ablations.py` → `ablation_scores.csv` reconstructing full / judge_only / embed_only /
  pos_only / no_availability from existing artifacts (keyword-stuffers have no judge score → those 3
  cols NaN, noted; pos/neg query vectors were saved so embed/pos are fully reconstructable).
  `eval/traces.md` — full reasoning chain (evidence→facts→judge verdict→final rank) for 5 records.
  Labels come from an independent source next; then run metrics + ablations.
- **Eval run — independent labels merged + ablations scored (this step):** Merged 47 independent
  `label_tier` values (dist 0:7 1:13 2:11 3:5 4:11; 16 relevant) into `gold_set.jsonl`; `eval/run_eval.py`
  → `eval/EVAL_RESULTS.md`. **Composite: full 0.985, no_availability 0.981, pos_only 0.823, judge_only
  0.819, embed_only 0.806.** Deltas (common 44-univ): full−judge_only +0.166, full−embed_only +0.179
  (judge+structure add real signal over retrieval); full−no_availability +0.004 (availability small on
  composite, +0.024 MAP, correctly demotes the unavailable); **embed_only−pos_only −0.017 → the
  negative-query subtraction slightly HURTS on this set (tuning candidate).** Sanity: (a) honeypots
  score +0.063 HIGHER than lowest non-fits on raw embeddings — retrieval is fooled, gate is essential;
  (b) CAND_0092278 drops 12 places under availability (labeler agreed: tier 2); (c) 1 recall miss —
  CAND_0094056 (tier-3, deep shortlist). Labels are a different model than the judge → deltas are the
  trustworthy signal; absolute `full` inflated by gold-set construction (includes the submitted top-10).
- **F1 controlled re-freeze — second honeypot family gated (this step):** Full-project audit
  (AUDIT_FINDINGS.md) found a **second, inverse honeypot family invisible to all five gate checks**:
  profiles with an **inflated `years_of_experience` field** (every existing check catches
  UNDER-claimed experience — tenure/months exceeding yoe — none catches the reverse). New 6th
  deterministic gating check `yoe_gt_career_span` in `features.py`: fires when
  `yoe×12 > months(earliest career start_date → REFERENCE_DATE) + 12` (same pinned
  REFERENCE_DATE=2026-06-30 as all other date math — reproducible, no live clock). Fires on
  **exactly 25/100k, zero overlap with the existing 44** → gate total **69 ≈ spec's ~80; recall
  ~55% → ~86%** at unchanged precision. Evidence it's a designed trap, not noise: **22/25
  corroborated by the candidate's own summary** stating the true span while the field is inflated
  (e.g. CAND_0055992: field 16.9y, summary "6.8 years", observed span 6.7y); the 3 uncorroborated
  cases were manually reviewed — all fabricated (joke employers Hooli/Acme, single 8–16-mo role vs
  12–15y claims, Ph.D-before-Masters dates), none a genuine long-tenure person; one previously
  gated honeypot (CAND_0019480) shows the same summary-contradiction signature in reverse.
  **Impact: 3 of the 25 sat in the submitted top-100 at ranks 44/46/47** (CAND_0039754,
  CAND_0091534, CAND_0055992 — inside the NDCG@50 window; independent gold labeler had rated them
  tier 2/2/1). **Controlled re-freeze protocol (all passed):** features/evidence parquets
  byte-equal after rebuild; shortlist membership identical (985); none of the newly gated in
  `frozen_rerank_order.json` → regeneration deterministic, **no LLM call**; ranks 1–43 unchanged;
  the 3 dropped, ranks below shift up, judged-tier-3 high-availability entrants at 98–100
  (CAND_0033445, CAND_0027691, CAND_0073504); 97 shared reasonings byte-identical;
  `validate_submission.py` → "Submission is valid." (exit 0); independent fresh-logic audit from
  raw `candidates.jsonl`: **0/100 trip any hard check incl. the new one**; byte-identical across
  3 regenerations — **new md5 `d21e098092d73a4a51aaac7910a4c160`** (supersedes `0673da50…`).
  Top-100 now 44 tier-4 + 56 tier-3. Audit companions: AUDIT_FINDINGS.md (full audit, incl. the
  negative-query KEEP verdict and the recall-miss diagnosis) + AUDIT_F1_NOTES.md (flag table,
  manual-review cases, before/after boundary).
