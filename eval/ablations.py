"""
Eval / PART 3 — reconstruct ablation ranking scores for the gold-set candidates from EXISTING
artifacts (no re-embedding, no re-judging). Each column induces a ranking; metrics.py scores them
once labels exist.

  full            : the submitted within-tier blended score  (0.70*judge + 0.30*struct)*avail_mult
  judge_only      : judge fit_score alone
  embed_only      : fused embedding fit = mean over BGE,E5 of cos(pos) - 0.5*cos(neg)
  pos_only        : positive-query similarity only = mean over BGE,E5 of cos(pos)   (no neg subtraction)
  no_availability : full without the availability multiplier = 0.70*judge + 0.30*struct

Inputs availability note:
  - pos/neg query vectors WERE saved (emb_bge_q.npy, emb_e5_q.npy) and passage embeddings saved, so
    embed_only and pos_only are fully reconstructable from embeddings.
  - full / judge_only / no_availability need the judge fit_score, which exists only for SHORTLIST
    candidates (985). Gold candidates outside the shortlist (keyword-stuffers) have no judge score →
    those three columns are NaN for them (noted in the output).
"""
from __future__ import annotations
import os, sys, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, "artifacts")
sys.path.insert(0, os.path.join(ROOT, "src"))
from rank_submission import structured_fit, availability_multiplier, W_JUDGE, W_STRUCT
LAMBDA = 0.5


def main():
    gold = pd.DataFrame([json.loads(l) for l in open(os.path.join(HERE, "gold_set.jsonl"), encoding="utf-8")])
    ids = list(gold.candidate_id)

    feats = pd.read_parquet(os.path.join(ART, "features.parquet")).set_index("candidate_id")
    judg = pd.read_parquet(os.path.join(ART, "judgments.parquet")).set_index("candidate_id")

    # embeddings + query vectors (all 100k; index by candidate_id)
    bge = np.load(os.path.join(ART, "emb_bge.npy")).astype(np.float32)
    e5 = np.load(os.path.join(ART, "emb_e5.npy")).astype(np.float32)
    emb_ids = np.load(os.path.join(ART, "emb_ids.npy"), allow_pickle=True)
    bge_q = np.load(os.path.join(ART, "emb_bge_q.npy")).astype(np.float32)   # [pos, neg]
    e5_q = np.load(os.path.join(ART, "emb_e5_q.npy")).astype(np.float32)
    emb_idx = {c: i for i, c in enumerate(emb_ids)}

    rows = []
    missing_judge = []
    for cid in ids:
        f = feats.loc[cid]
        struct = structured_fit(f)
        avail = availability_multiplier(f)
        judge = float(judg.loc[cid, "fit_score"]) if cid in judg.index else np.nan

        # embedding sims (normalized vectors -> dot = cosine)
        i = emb_idx[cid]
        bpos, bneg = float(bge[i] @ bge_q[0]), float(bge[i] @ bge_q[1])
        epos, eneg = float(e5[i] @ e5_q[0]), float(e5[i] @ e5_q[1])
        pos_only = (bpos + epos) / 2.0
        embed_only = ((bpos - LAMBDA * bneg) + (epos - LAMBDA * eneg)) / 2.0

        if np.isnan(judge):
            missing_judge.append(cid)
            full = judge_only = no_availability = np.nan
        else:
            no_availability = W_JUDGE * judge + W_STRUCT * struct
            full = no_availability * avail
            judge_only = judge

        rows.append({
            "candidate_id": cid,
            "stratum": gold.set_index("candidate_id").loc[cid, "stratum"],
            "submitted_rank": gold.set_index("candidate_id").loc[cid, "submitted_rank"],
            "full": full, "judge_only": judge_only, "embed_only": embed_only,
            "pos_only": pos_only, "no_availability": no_availability,
            "structured_fit": round(struct, 4), "avail_mult": round(avail, 4),
        })

    out = pd.DataFrame(rows)
    path = os.path.join(HERE, "ablation_scores.csv")
    out.to_csv(path, index=False, encoding="utf-8")

    print(f"wrote {path}  ({len(out)} gold candidates)")
    print(f"ablation columns: full, judge_only, embed_only, pos_only, no_availability")
    print(f"candidates with NO judge score (full/judge_only/no_availability = NaN): "
          f"{len(missing_judge)} -> {missing_judge}")
    print("\n--- score preview (sorted by 'embed_only' desc) ---")
    with pd.option_context("display.width", 200, "display.max_rows", 60):
        show = out.sort_values("embed_only", ascending=False)[
            ["candidate_id", "stratum", "submitted_rank", "full", "judge_only",
             "embed_only", "pos_only", "no_availability"]]
        print(show.round(4).to_string(index=False))
    print("\nNOTE: no metrics computed — awaiting independent label_tier. "
          "Feed this + labeled gold_set into eval/metrics.py:evaluate_many().")


if __name__ == "__main__":
    main()
