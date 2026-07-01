# AUDIT F1 — inflated-yoe honeypot family: gate extension + controlled re-freeze

Status: **complete; all protocol steps passed.** Submission regenerated deterministically
(no LLM), validator-clean, byte-reproducible.
**New md5 `d21e098092d73a4a51aaac7910a4c160`** (supersedes `0673da5035a1b627c078971d7b89768b`).

## Pre-flight — date determinism

Every date computation in `features.py` (recency, future-start check, open-role intervals, and
the new check) uses the single pinned constant **`REFERENCE_DATE = date(2026, 6, 30)`**
(`features.py:23`). Grep over `src/` and `eval/` confirms **no `datetime.now()` / `date.today()`
anywhere** — the flag is reproducible run-to-run. Nothing to fix.

## The check

`yoe_gt_career_span` (6th GATING check in `honeypot_flags()`):
`years_of_experience × 12 > months(earliest career_history start_date → REFERENCE_DATE) + 12`.

Informational (non-gating) corroboration also computed and persisted:
`summary_years_stated` (first "N years" figure parsed from `profile.summary`) and
`summary_contradicts_yoe` (|yoe − summary_years| > 4).

## Full-pool scan (100,000 candidates)

- Old gate (5 checks): **44** — unchanged. New check: **25**. Overlap: **0**.
- New gate total: **69 ≈ spec's ~80 → recall ~55% → ~86%** at unchanged precision.
- **22 of 25 corroborated** by the candidate's own summary stating the true span while the yoe
  field is inflated.

| candidate_id | title | yoe field | span (mo) | summary yrs | corroborated |
|---|---|---|---|---|---|
| CAND_0003430 | Business Analyst | 13.7 | 15 | 1.3 | YES |
| CAND_0005291 | Business Analyst | 12.8 | 33 | 4.7 | YES |
| CAND_0007413 | Business Analyst | 13.3 | 43 | 10.8 | NO — see manual review |
| CAND_0010770 | Recommendation Systems Engineer | 15.2 | 86 | 7.2 | YES |
| CAND_0013536 | Applied ML Engineer | 14.1 | 56 | 4.8 | YES |
| CAND_0024752 | Civil Engineer | 14.9 | 48 | 14.2 | NO — see manual review |
| CAND_0025579 | HR Manager | 12.9 | 26 | 14.6 | NO — see manual review |
| CAND_0033131 | Operations Manager | 12.7 | 33 | 8.5 | YES |
| CAND_0036299 | Mobile Developer | 12.2 | 44 | 6.0 | YES |
| CAND_0038431 | Mobile Developer | 15.0 | 44 | 7.9 | YES |
| **CAND_0039754** | **Senior Applied Scientist** | **16.2** | **97** | **8.3** | **YES — was rank 44** |
| CAND_0052478 | Marketing Manager | 12.4 | 16 | 3.2 | YES |
| **CAND_0055992** | **AI Engineer** | **16.9** | **80** | **6.8** | **YES — was rank 47** |
| CAND_0065787 | Java Developer | 10.9 | 32 | 2.7 | YES |
| CAND_0066405 | Cloud Engineer | 12.3 | 31 | 2.6 | YES |
| CAND_0071115 | Recommendation Systems Engineer | 16.5 | 69 | 5.8 | YES |
| CAND_0074119 | Content Writer | 11.4 | 25 | 4.7 | YES |
| CAND_0077250 | Project Manager | 13.1 | 32 | 4.4 | YES |
| CAND_0086808 | Graphic Designer | 11.4 | 19 | 1.6 | YES |
| CAND_0090900 | Senior Data Engineer | 11.7 | 38 | 3.2 | YES |
| CAND_0091068 | HR Manager | 12.7 | 27 | 4.1 | YES |
| **CAND_0091534** | **AI Engineer** | **16.6** | **87** | **7.2** | **YES — was rank 46** |
| CAND_0093331 | NLP Engineer | 16.1 | 86 | 7.2 | YES |
| CAND_0095619 | NLP Engineer | 15.6 | 50 | 4.2 | YES |
| CAND_0096150 | Accountant | 14.7 | 25 | 2.1 | YES |

## Manual review — the 3 non-corroborated cases (all judged fabricated, none gated in error)

1. **CAND_0007413** (Business Analyst): ONE role, 16 mo at Globex Inc; yoe field 13.3, summary
   "10.8+ years" — *three* mutually contradicting experience numbers; summary claims a
   "marketing manager" background under a Business Analyst title. Fabricated.
2. **CAND_0024752** (Civil Engineer @ **Hooli** — joke employer): ONE role, 8 mo; yoe field 14.9,
   summary 14.2 claims "marketing manager roles"; education has a **Ph.D (2001–2005) BEFORE an
   M.E. (2007–2011)**. Fabricated.
3. **CAND_0025579** (HR Manager @ **Acme Corp** — joke employer): ONE role, 12 mo; yoe field 12.9,
   summary 14.6 claims marketing career; **B.E. in "Artificial Intelligence" dated 2004**
   (anachronistic). Fabricated.

In these three the summary *agrees* with the inflated field but both contradict the actual
8–16-month single-role history — a different corruption of the same trap family. No genuine
long-tenure profile is among the 25; all gated.

## Regeneration through the frozen path

- `features.parquet` and `evidence_text.parquet` **byte-content identical** after rebuild
  (only `honeypot_flags.parquet` gained flags/columns).
- `shortlist.parquet` membership **identical (985/985 asserted)**; gated-within-shortlist 4 → 12
  (8 new; the 9th shortlisted family member CAND_0019480 was already gated by the old checks).
- **None of the newly gated appear in `frozen_rerank_order.json`** → regeneration used the frozen
  top-40 order, **no LLM call**. Ranks 1–43 byte-identical to the previous CSV.
- Note for reproduce docs: run with `PYTHONIOENCODING=utf-8` on Windows — the final cosmetic
  console table contains '→' and dies on cp1252 *after* the CSV is written (harmless, but exit
  code matters for scripts).

## Re-audit protocol results (all PASS)

| step | result |
|---|---|
| `validate_submission.py` | **"Submission is valid."** exit 0 |
| gated honeypots in top-100 | **0** (asserted) |
| independent fresh-logic audit from raw `candidates.jsonl` (all 6 hard checks + date sanity) | **0/100 trip anything**, incl. the new check |
| byte-reproducibility | 3 regenerations, **identical md5 `d21e098092d73a4a51aaac7910a4c160`** |
| tier monotonicity | non-increasing, top-100 = **44 tier-4 + 56 tier-3** (was 47+53; the 3 removed were tier-4) |
| reasonings | 100/100 distinct, max 240 chars, 44 rows carry explicit `Concern:` (old CSV: 43); **97 shared candidates byte-identical reasoning** |

## Before / after boundary

**Dropped** (were ranks 44 / 46 / 47): CAND_0039754, CAND_0091534, CAND_0055992.
**Entered** at ranks 98–100: CAND_0033445 (ML Engineer, tier 3, avail 0.997),
CAND_0027691 (NLP Engineer, tier 3, avail 0.975), CAND_0073504 (Junior ML Engineer, tier 3,
avail 0.974) — exactly the pre-identified next-best pipeline candidates.

New ranks 44–52 (tier-4→3 cut now at 44/45; no inversion — within_tier_score is per-tier by design):

```
rank  candidate_id  title                            tier  judge  struct  avail   score
 44   CAND_0041611  Staff Machine Learning Engineer   4    0.95   0.900   0.909   0.7772
 45   CAND_0000031  Recommendation Systems Engineer   3    0.85   1.000   0.988   0.7722
 46   CAND_0036184  Recommendation Systems Engineer   3    0.85   0.940   0.995   0.7673
 47   CAND_0083879  Machine Learning Engineer         3    0.85   1.000   0.970   0.7623
 48   CAND_0007009  Recommendation Systems Engineer   3    0.85   1.000   0.970   0.7574
 49   CAND_0045250  Applied ML Engineer               3    0.85   1.000   0.964   0.7524
 50   CAND_0010257  Senior Data Scientist             3    0.85   1.000   0.963   0.7475
 51   CAND_0076163  NLP Engineer                      3    0.85   0.940   0.982   0.7425
 52   CAND_0006418  Machine Learning Engineer         3    0.85   0.955   0.976   0.7376
```

New ranks 96–100:

```
rank  candidate_id  title                          tier  judge  struct  avail   score
 96   CAND_0070525  Senior Software Engineer (ML)   3    0.75   0.955   0.992   0.5198
 97   CAND_0068932  ML Engineer                     3    0.75   0.955   0.992   0.5148
 98   CAND_0033445  ML Engineer                     3    0.75   0.940   0.997   0.5099
 99   CAND_0027691  NLP Engineer                    3    0.75   1.000   0.975   0.5049
100   CAND_0073504  Junior ML Engineer              3    0.75   1.000   0.974   0.5000
```

Entrant reasonings (auto-generated, fact-grounded, rank-consistent tone):
- 98 CAND_0033445: "ML Engineer, 6.8y: Built computer vision models using PyTorch at Niramai and
  upGrad. Concern: Limited experience in production ML systems beyond modeling."
- 99 CAND_0027691: "NLP Engineer, 6.5y: Built and operated production ML pipelines using MLflow
  and Kubeflow; Owned the ranking layer for an e-commerce search product, improving
  revenue-per-search by 12%."
- 100 CAND_0073504: "Junior ML Engineer, 6.6y: Built recommendation-style features using
  collaborative filtering and gradient-boosted re-ranking. Concern: Title-chaser: 'Junior ML
  Engineer' despite 6.6 years experience."
