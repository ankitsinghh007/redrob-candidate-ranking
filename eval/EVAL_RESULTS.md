# EVAL_RESULTS — ranking vs. independent hand-labeled gold set

Labels are an **independent second opinion** (a different model than the Qwen judge), so the **deltas between configs are the trustworthy signal**, not the absolute values.

Gold set: 47 candidates, 16 relevant (tier>=3). Label distribution: {0: 7, 1: 13, 2: 11, 3: 5, 4: 11}

## 1. Metrics per config

```
                  n  n_relevant(tier>=3)  NDCG@10  NDCG@50     MAP  P@10  composite
score                                                                              
full             44                   16   1.0000   0.9930  0.9161   1.0     0.9853
judge_only       44                   16   0.7876   0.9373  0.7279   0.7     0.8192
embed_only       47                   16   0.7847   0.9232  0.6803   0.7     0.8064
pos_only         47                   16   0.8069   0.9350  0.6957   0.7     0.8233
no_availability  44                   16   1.0000   0.9904  0.8923   1.0     0.9810
```
_full/judge_only/no_availability exclude the 3 keyword-stuffers (no judge score); embed_only/pos_only include all 47._

## 2. `full` minus each config (common 44-candidate universe)

```
                 NDCG@10  NDCG@50     MAP  P@10  composite
full              0.0000   0.0000  0.0000   0.0     0.0000
judge_only        0.2124   0.0557  0.1882   0.3     0.1662
embed_only        0.2153   0.0697  0.2358   0.3     0.1789
pos_only          0.1931   0.0580  0.2205   0.3     0.1620
no_availability   0.0000   0.0026  0.0238   0.0     0.0043
```

- `full - judge_only` = value added by structured features + availability over the judge
- `full - no_availability` = contribution of the availability multiplier alone
- `full - embed_only` = value the judge (+structure) adds over raw ensemble embeddings
- `embed_only - pos_only` = contribution of the negative-query subtraction (composite -0.0169)

## 3. Findings

**(a) Honeypots are HIGH on raw embeddings.** mean embed_only: 4 honeypots 0.4514 vs 4 lowest non-fits 0.3880 (+0.0634). Pure retrieval is fooled by keyword-perfect fabrications; the deterministic gate removes them.

**(b) Availability multiplier.**
- CAND_0041611 (label 2, submitted_rank 45.0): full #17/44 vs no_availability #11/44 (+6 places)
- CAND_0092278 (label 2, submitted_rank nan): full #28/44 vs no_availability #16/44 (+12 places)

**(c) Recall misses (tier>=3 not in submitted top-100):**
- CAND_0094056 | NLP Engineer | tier 3 | deep_shortlist

## Takeaways & caveats

**What the deltas say (the trustworthy signal):**
- **Judge + structured features do real work over retrieval.** `full − embed_only` = **+0.179 composite**;
  `full − judge_only` = **+0.166 composite**. Raw embeddings alone (0.806) and the judge alone (0.819)
  land in the same ballpark; stacking judge + structured features + availability lifts composite to
  **0.985**. The components are complementary, not redundant.
- **The negative-query subtraction slightly HURTS on this set:** `embed_only − pos_only` = **−0.017
  composite** (pos_only 0.823 > embed_only 0.806). Subtracting `0.5·cos(neg)` cost a little ranking
  quality here. Small and on 44 labels, but a concrete candidate for tuning (lower λ, or drop the
  negative query) if it holds up on more labels.
- **Availability contributes little to composite (+0.004) but is directionally correct:** it lifts MAP
  (+0.024) and demotes the genuinely unavailable — CAND_0092278 (235d idle, 7% response) drops **12
  places** under `full` vs `no_availability`, and the independent labeler rated it only **tier 2**,
  agreeing it is not a top fit.

**Honeypots:** on pure embeddings the 4 honeypots (all labeled tier 0) score **+0.063 higher** than the
lowest genuine non-fits — retrieval is fooled by keyword-perfect fabrications. They are removed by the
**deterministic gate**, not by any learned score. (In this eval the `full` universe still *contains* the
honeypots as ranking targets; the deployed pipeline hard-drops them first, so real top-of-list quality
is at least as good as the numbers here.)

**Recall:** exactly **1** labeled-relevant (tier≥3) candidate missed the submitted top-100 —
**CAND_0094056** (NLP Engineer, tier 3), sitting deep in the shortlist. A single borderline miss.

**Absolute-value caveat:** `full` shows NDCG@10 = P@10 = 1.0, but the gold set *by construction*
includes the submitted top-10 (which `full` ranks highest), so the absolute top-of-list numbers are
optimistic. The labels are an **independent second opinion (a different model than the Qwen judge)**, so
treat the **between-config deltas** as the finding, not the absolute magnitudes.
