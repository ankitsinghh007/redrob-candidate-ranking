# AUDIT_FINDINGS — full-project fresh-eyes audit (2026-07-02)

Scope: read-only audit of the dataset docs, DECISIONS.md, all stage notes, eval harness, and
`src/` code, plus four empirical investigations run against the existing artifacts (nothing was
modified; frozen CSV re-validated at the end: **"Submission is valid.", md5 `0673da50…` unchanged**).

---

## Headline findings (evidence first, recommendations after)

### F1 — A second honeypot family exists, our gate is structurally blind to it, and 3 of them sit at submitted ranks 44/46/47

All five gating checks in `features.py:431-438` catch **under-claimed** experience (role tenure or
summed months EXCEEDING `years_of_experience`, expert-at-low-yoe). None catches the inverse:
**inflated `years_of_experience`**. Testing one new deterministic check on the full 100k pool —
`yoe×12 > (months from earliest career start_date to REFERENCE_DATE) + 12` — fires on exactly
**25 candidates**, with **zero overlap** with the existing 44 gated honeypots. 44 + 25 = **69 ≈ the
spec's ~80** (§7). Recall would go ~55% → ~86%.

Precision evidence (this is not generator noise):
- All 25 show `years_of_experience` ≈ 2–10× their observed career span.
- **21 of 25 are independently corroborated by their own summary text**: the summary states a
  years figure that matches the career span almost exactly while contradicting the yoe field
  (e.g. CAND_0055992: yoe field **16.9**, summary "**6.8 years**", career span **6.7y**;
  CAND_0039754: yoe **16.2**, summary "**8.3 years**", span **8.1y**). The generator derives summary
  years from career history; the yoe field was deliberately inflated — the signature of a designed
  "subtly impossible profile".
- One *already-gated* honeypot (CAND_0019480) shows the same summary-vs-yoe contradiction in
  reverse (yoe 2.8, summary 7.4) — this contradiction pattern is honeypot construction.
- The 25 include disqualifier-shaped decoys with AI titles: 2× Recommendation Systems Engineer,
  3× NLP Engineer, 2× AI Engineer, Senior Applied Scientist, Applied ML Engineer.

Impact on the submission: **CAND_0039754 (rank 44), CAND_0091534 (rank 46), CAND_0055992 (rank 47)**
trip the check — all three inside the NDCG@50 window (30% of composite). The independent gold
labeler already rated them **tier 2, 2, and 1** (`eval/gold_set.jsonl`, tier_boundary stratum), i.e.
non-relevant even under the charitable branch. If they are honeypots (the evidence says so), they
are forced tier-0 at ranks 44–47 and also consume 3 points of the 10-honeypot DQ budget. None are
in the top-10, so NDCG@10/P@10 are unaffected either way.

Fix path is surgical and stays deterministic: none of the three is inside the frozen top-40
(`artifacts/frozen_rerank_order.json` untouched → regeneration needs **no LLM call**). Add the check
as a 6th gating flag in `honeypot_flags()`, rebuild `honeypot_flags.parquet` (~9s) and
`shortlist.parquet` (~1 min; membership unchanged — gating only flags), regenerate via the frozen
path, then re-run the full Stage-D-final audit protocol (validator, 0-honeypot assert, byte-repro
×3, boundary check). Ranks 48–100 shift up 3; pipeline ranks 101–103 (ML Engineer / NLP Engineer /
Junior ML Engineer, all judged tier-3, availability ≥0.97) enter at 98–100.

**Recommendation: DO-NOW.** This is the one finding that justifies touching the frozen CSV: the
downside branch (they're genuine) swaps three label-tier-1/2 candidates for three judged-tier-3
candidates — roughly neutral; the upside branch (honeypots) removes three tier-0s from the top-50.
Asymmetric in our favor, ~1–2h including the re-audit, and reproducibility is preserved.

### F2 — The negative query is NOT net-harmful; keep λ = 0.5 (Part 3 answered)

Recomputed the fused shortlist at λ ∈ {0.5, 0.25, 0} from the saved embeddings/query vectors:

| λ=0 vs λ=0.5 (top-800) | result |
|---|---|
| membership churn | 140 in / 140 out (660 stable) |
| entrants that are disqualifier-class | **12 nontech** (Sales Executives, Mechanical Engineers…), **9 services-only careers**, 77 services-exposed, **6 Computer Vision Engineers** |
| leavers | mostly Senior Data Engineers (49), Junior ML Engineers (22) — the profiles the JD is lukewarm on, which the judge would prune anyway |
| best gated-honeypot rank | 33 → **27** (worse without the neg query) |
| CAND_0094056 (the recall miss) | fused rank 234 → **358** (worse without the neg query) |

The eval's −0.017 was measured on `embed_only`, a scorer that is **not deployed anywhere**: in the
real pipeline, retrieval only decides *shortlist membership*; final ordering is judge+structured.
On the deployed decision (membership), removing the negative query admits disqualifier-class
profiles and helps nothing — the miss it was blamed for gets worse. λ tuning also cannot change
the frozen CSV (all top-100 candidates sit at fused ranks ≪ 800 at any tested λ).
**Recommendation: KEEP λ=0.5, change nothing. Record this in DECISIONS.md as the resolution of the
eval's "tuning candidate" flag** — it's a strong deck slide (ablation looked negative; deployment
analysis shows the ablation measured the wrong surface).

### F3 — The one recall miss is a within-tier-3 discrimination limit, not a retrieval class-miss

CAND_0094056 was retrieved fine (fused rank 234, `inclusion_reason=retrieval_rank`) and the judge
gave it **tier 3 — the same tier as the independent labeler**. It lands at pipeline rank 388 because
the judge's fit_score is heavily quantized: **500 of 548 tier-3s share the identical 0.75**
(`judgments.parquet`; tier-4 is 29×0.95 + 18×0.92). Within tier 3, ordering is therefore decided
almost entirely by `structured_fit × availability`, which cannot see that this candidate's build
narrative (sentence-transformers + FAISS semantic search, +35% relevance vs BM25) is stronger than
a churn-model ML Engineer's. There is no under-retrieved *class*: the band at ranks 101–130 is
uniformly judged-tier-3 AI/ML titles, i.e. the cut through tier 3 is a coin-flip zone by design.
The proper fix (finer judge scale, or extending the listwise rerank across the ranks-40–110
boundary zone) requires an LLM re-run and reopens frozen-CSV risk for an unvalidatable gain on a
44-label set. **Recommendation: SKIP the fix; document as a known limitation** — it's an honest,
well-understood deck point ("our 1 measured recall miss is a within-tier ordering tie, judge and
labeler agree on its tier").

---

## Ranked improvement table

| # | Improvement | Expected payoff (metric / deliverable) | Effort | Regret-risk | Re-run? | Recommendation |
|---|---|---|---|---|---|---|
| 1 | **Add inflated-yoe gate check** (F1): 6th gating flag `yoe_gt_career_span` (+ optional summary-years corroboration), rebuild flags+shortlist, regenerate CSV via frozen path, full re-audit | Removes 3 near-certain tier-0s from ranks 44/46/47 → NDCG@50 (0.30 wt) + MAP; honeypot recall 55%→86%; strong deck story | ~1–2h | Low: frozen top-40 untouched, no LLM, deterministic; downside branch ≈ neutral (labeler rated them 2/2/1) | No LLM re-run; CSV changes (controlled re-freeze) | **DO-NOW** |
| 2 | **Repo scaffolding** (see checklist below): git init + staged commits, README with reproduce command, thin CPU-only `rank.py`, `submission_metadata.yaml`, `.gitignore`, pinned deps | Stage 1/3/4 survival — these are pass/fail gates, worth more than any score delta | ~3–4h | None | No | **DO-NOW** |
| 3 | **Sandbox link** (spec §10.5 — **mandatory, flagged at Stage 1 if missing**; currently unplanned): HF Space/Colab that runs `rank.py` on a pre-loaded ≤100-candidate sample from the pool (sample includes shortlisted IDs so precomputed judge scores exercise the true rank path) | Stage 1 compliance — submission is flagged without it | ~2–3h | Low; "pre-loaded sample" is explicitly allowed | No | **DO-NOW** |
| 4 | **Measure + document the ranking-step runtime/memory** (one timed run of the frozen path on CPU; record in README + metadata `pre_computation_*` fields) | Stage 3 reproduction confidence; metadata honesty | ~15 min | None | No | **DO-NOW** |
| 5 | **DECISIONS.md deck-prep pass**: mark the 4 stale "Open Questions" resolved (they were, in Stage C/D); annotate the superseded "top-600 ∪ build ≈ 808" line (implemented: top-800 ∪ build = 985); append F1/F2/F3 resolutions | Deck + Stage-5 interview consistency (interviewers read the log against the code) | ~30 min | None | No | **DO-NOW** |
| 6 | Record the F2 negative-query analysis as an eval addendum (EVAL_RESULTS.md caveat: `embed_only−pos_only` measured a non-deployed surface) | Deck honesty; pre-empts an interview gotcha | ~15 min | None | No | DO-IF-TIME |
| 7 | Reasoning touch-up for the 3 rows that replace F1's drops (auto-generated by the existing `make_reasoning`; just eyeball tone/rank consistency at ranks 98–100) | Stage 4 reasoning checks | ~10 min (inside #1) | None | No | DO-IF-TIME (bundled with #1) |
| 8 | Extend listwise rerank across the tier-3 boundary zone (ranks ~40–110) to pick *which* tier-3s make the cut (F3) | Possible NDCG@50/MAP; unvalidatable on 44 labels | ~2–3h + LLM run | **High**: reopens frozen CSV + nondeterminism for an unmeasurable gain | Yes (LLM) | **SKIP** |
| 9 | Finer-grained judge score scale (fix the 0.75 quantization) | Within-tier ordering quality | ~2.2h re-judge + full re-fuse | **High**: full re-run, invalidates frozen order + audits | Yes (full) | **SKIP** |
| 10 | Use `github_activity_score` (JD nice-to-have "OSS contributions"; currently extracted in features but unused in `structured_fit`) | Marginal within-tier nudge; 64.6% missing | ~30 min | Medium: changes frozen CSV for a weak, sentinel-heavy signal | CSV regen | **SKIP** |
| 11 | Add skills proficiency/duration back as evidence or consistency features | None demonstrated — exclusion was right (see Part-2 verdicts) | — | High (re-embed everything) | Yes (full) | **SKIP** |
| 12 | Lower/drop negative-query λ per the eval delta | Negative (F2: admits disqualifier-class profiles, worsens the actual miss) | — | High | Yes | **SKIP — resolved KEEP** |

## Part-2 audit verdicts (pipeline correctness — no action needed beyond the table)

- **Evidence text**: excluding `skills[]` was right (uniform ~12k×133 noise; JD calls it a trap);
  down-weighting the summary was right (templated, self-contradicting — Stage A examples). The one
  place skills data DOES carry signal is *consistency* (expert-with-real-durations vs expert@0mo),
  and the pipeline already uses exactly that in the gate — nothing recoverable was discarded.
  Certifications (75% empty) and `skill_assessment_scores` (75.8% empty) are too sparse to matter.
- **Retrieval**: recall is genuinely safe. Must-not-miss titles fully inside top-600 at λ=0.5;
  build-signal union rescues 484 plain-language builders; the single labeled miss was retrieved
  (rank 234). No fit-shaped class is systematically outside the 985 shortlist.
- **Judge**: precomputed-facts design held up (100% parse, 0 failures, micro-validated timeline
  fact moved honeypots without perturbing genuine fits: max |Δscore| 0.000). The quantization
  limitation (F3) is real but not worth a re-run. No prompt change is worth re-judging 985 with
  ~1 day left.
- **Fusion double-counting (flag for the deck, not for change)**: availability appears in the judge's
  facts block (rubric standing instruction #2) *and* as the ×[0.85,1.0] multiplier; yoe/services/
  location likewise appear in both judge facts and `structured_fit`. This is deliberate stacking of
  correlated views, and the ablations say the stack adds signal (full − judge_only = +0.166), but be
  ready to defend "isn't that double counting?" in the interview: the answer is the multiplier is
  bounded (max −15%) and the ablation shows the components are complementary, not redundant.
  CAND_0092278 (judge 0.85, avail 0.894 → rank 106) is the worked example, and the independent
  labeler agreed (tier 2).
- **Honeypot gate**: precision-first stance remains correct; F1 closes most of the recall gap with
  a check of the same deterministic character as the existing five.

## Deliverable-readiness checklist (Part 4 — what the repo/deck still need)

**Blocking (spec-mandated):**
1. **Git repository with real history** — the project directory is *not a git repo*. Stage 4
   explicitly eliminates "flat git history with no iteration". Initialize now and commit the
   existing work as a sequence of true-to-history commits (stage-by-stage, mirroring the notes
   files' timestamps/content); do not dump one commit. This is the single biggest submission risk
   after the CSV itself.
2. **README.md** with the single reproduce command (§10.3) + precompute documentation (embedding
   build, judge run) + artifact map.
3. **Thin `rank.py`** honoring the spec's command shape (`python rank.py --candidates … --out …`):
   the frozen-path logic of `rank_submission.py` without the module-level `import ollama`
   (rank-time deps must not include judge-time libs; keep the import lazy inside the rerank
   function). Time it (expect seconds).
4. **`submission_metadata.yaml`** at repo root from the template (participant ID, sandbox link,
   `pre_computation_required: true` + honest minutes, AI-tools declaration).
5. **Sandbox link** (table row 3) — currently missing from all plans; it is a required portal field.
6. **Artifacts vs GitHub limits**: `emb_bge.npy`/`emb_e5.npy` are 196 MB each (>100 MB GitHub cap).
   Ship the small artifacts (`judgments.parquet`, `features.parquet`, `honeypot_flags.parquet`,
   `shortlist.parquet`, `frozen_rerank_order.json`, query vecs, rubric, queries) and rely on
   `src/build_embeddings.py` as the §10.3 "script that produces them" for the big two — or use LFS.
7. **`.gitignore`**: `.h2s/` venv, `__pycache__/`, `dataset/` (465 MB + `__MACOSX` junk), scratch.
8. **Pin exact dependency versions** (spec: "dependencies and versions"; current file is `>=` floors
   — snapshot `pip freeze` into a lock section).
9. **Rename the CSV to the registered participant ID** before upload (known placeholder).

**Hygiene (verified clean):** no secrets, no hard-coded absolute paths (all `__file__`-relative),
no `__MACOSX` outside `dataset/`, `judge_failures.jsonl` empty, tail reasonings (ranks 96–100) are
measured-with-concerns — rank-consistency check should pass.

**Deck:** DECISIONS.md is quotable and complete on rationale; the quantitative spine is all present
(honeypot calibration 13.35%→0.044%, retrieval dry-run table, judge micro-validation, ablation
deltas, byte-repro md5). Fix the two staleness items (table row 5) so the log can't be quoted
against the code in the interview.

## DO NOT TOUCH

- **The deterministic honeypot gate as primary defense** and its precision-first stance. F1 *adds*
  a check of the same character; do not soften or LLM-ify the gate.
- **The frozen-rerank reproducibility mechanism** (`frozen_rerank_order.json` + LF newlines +
  byte-identical regeneration). Any accepted change must go through the same freeze → validate →
  3×-byte-repro → independent honeypot re-audit protocol before the CSV is re-frozen.
- **The submitted CSV except via that protocol** — no other table row (8–12) is worth reopening it.
- **The judge artifacts** (`judgments.parquet`) — no re-judging with ~1 day left.
- **λ=0.5 negative query, RRF ensemble, shortlist-985 membership** — audited safe (F2); churn here
  buys nothing and risks recall.
- **The eval harness and gold labels** — its independence (different model than the judge) is the
  credibility of every delta quoted in the deck.
