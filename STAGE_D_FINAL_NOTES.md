# Stage D-final — top-100 audit + frozen reproducible artifact

Status: **audited and frozen. Submission valid. Stopped for your review before we call it final.**
No ranking logic changed in this stage — verification + locking only.

Artifacts: `submission/ankitsingh058622_1300.csv` (frozen), `artifacts/frozen_rerank_order.json`,
`artifacts/submission_audit.parquet`.

---

## PART 1 — Honeypot audit of the ACTUAL top-100 (the real DQ risk)

Recomputed the timeline/consistency checks **independently from the raw `candidates.jsonl`** (fresh
logic, not the cached parquet) over all 100 submitted candidates:

- **CLEAN: 0 of the top-100 trips ANY hard flag** — role_tenure_gt_career, career_months_gt_experience,
  expert_low_experience, expert_zero_duration, or date-inconsistency (end<start / future / duration-vs-span
  / >1 current role). **Nothing in the top-10. Nothing anywhere in the 100.**
- The 4 gated honeypots were removed pre-ranking (PART 1 of the pipeline), and no *other* top-100
  candidate has an impossible profile.
- 47/100 trip only the **noisy soft skill-duration signal** (skill `duration_months` > yoe·12). This
  flag fires on ~13% of the *entire* pool — it is synthetic-generator noise (durations assigned
  independently of experience), NOT a fabrication signal. Informational only; not actionable.
- Hand-eyeball of the top-5 raw records confirms genuine profiles: real product companies (LinkedIn,
  Yellow.ai, Freshworks, Meesho, Salesforce, Apple, Paytm, Razorpay, Glance), coherent multi-role
  histories with tenures consistent with total experience, and "expert" skills backed by real durations
  (e.g. Machine Learning expert 61mo, Search Infrastructure expert 83mo, Vector Search expert 79mo).

**Verdict: no honeypot/DQ risk in the submission.**

## PART 2 — Boundary check at the tier cut and just below the list

Tier cut sits between **rank 47** (last tier-4) and **rank 48** (first tier-3). Ranks 48–100 are the top
tier-3 by `within_tier_score`.

- **Tier-consistent across the cut:** min tier in ranks 90–100 = 3; max tier in ranks 101–115 = 3. No
  higher-tier candidate was left below a lower-tier one. (By design `fit_tier` is the dominant sort key,
  so a tier-3 with very high structured_fit/availability can show a higher `within_tier_score` than a
  tier-4 yet still rank below it — intentional, not an inversion.)
- **Exactly ONE strong-fit candidate missed the cut, and it is correct:**
  **CAND_0092278 — rank 106, Senior NLP Engineer, judge_fit 0.85, structured_fit 1.0** — relegated by
  the availability multiplier (0.894) because **last active 235 days ago, recruiter_response_rate 0.07,
  open_to_work False.** This is exactly the JD's "perfect-on-paper but hasn't logged in for months / ~5%
  response rate → not actually available, down-weight" case. It sits below judge-0.75-but-highly-available
  candidates at ranks 97–100 (avail 0.98–0.99). **Defensible per the JD — surfaced for your call, not
  auto-changed.** If you consider the availability signal too aggressive here, this is the one candidate
  to reconsider promoting.

No other promotion/relegation issue surfaced.

## PART 3 — Freeze for reproducibility

- Re-ran the listwise rerank at **temperature 0, single deterministic pass** (seed 0) and cached the
  resulting top-40 order to **`artifacts/frozen_rerank_order.json`**.
- `rank_submission.py` now **reads the cached order if present** (no LLM call on regeneration); `--freeze`
  regenerates the cache. CSV is written with LF newlines for byte-stability.
- **Byte-identical across runs confirmed:** freeze-run A, then two frozen regenerations B and C →
  `diff` reports **identical (A==B==C)**, identical md5 `0673da5035a1b627c078971d7b89768b`.

## PART 4 — Regenerate + validate

Final CSV regenerated from the frozen order:
```
$ python validate_submission.py submission/ankitsingh058622_1300.csv
Submission is valid.
EXIT: 0
```
Filename was the placeholder `team_h2s_redrob.csv`, since renamed to the registered participant ID
`ankitsingh058622_1300.csv` (spec §2; validator only requires a non-empty `.csv` stem — original note: swap before
upload — validator only requires a non-empty `.csv` stem).

### Final TOP-15
| # | candidate_id | title | tier | score |
|--|--|--|--|--|
| 1 | CAND_0061257 | Staff ML Engineer | 4 | 0.9900 |
| 2 | CAND_0046525 | Senior ML Engineer | 4 | 0.9851 |
| 3 | CAND_0064326 | Search Engineer | 4 | 0.9801 |
| 4 | CAND_0068811 | Applied ML Engineer | 4 | 0.9752 |
| 5 | CAND_0077337 | Staff ML Engineer | 4 | 0.9702 |
| 6 | CAND_0071974 | Senior AI Engineer | 4 | 0.9653 |
| 7 | CAND_0011687 | Senior NLP Engineer | 4 | 0.9603 |
| 8 | CAND_0018499 | Senior ML Engineer | 4 | 0.9554 |
| 9 | CAND_0006567 | Senior AI Engineer | 4 | 0.9504 |
| 10 | CAND_0081846 | Lead AI Engineer | 4 | 0.9455 |
| 11 | CAND_0068351 | Lead AI Engineer | 4 | 0.9405 |
| 12 | CAND_0007412 | Applied ML Engineer | 4 | 0.9356 |
| 13 | CAND_0050454 | AI Engineer | 4 | 0.9306 |
| 14 | CAND_0079387 | AI Engineer | 4 | 0.9257 |
| 15 | CAND_0010685 | NLP Engineer | 4 | 0.9207 |

(Frozen temp-0 top-15 is identical to the earlier 3-pass mean-rank order — stable.)

---

## For your review before we call it final
1. **CAND_0092278** (rank 106): strong on paper (judge 0.85) but unavailable (235d idle, 7% response,
   not open). Down-weighted correctly per the JD — promote it or leave it? (Recommend: leave.)
2. **Participant ID:** confirmed as `ankitsingh058622_1300` — file renamed accordingly (was `team_h2s_redrob.csv`).
3. Otherwise: 0 honeypot risk, tier-consistent boundary, validator-clean, byte-reproducible.
