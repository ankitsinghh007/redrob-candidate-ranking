"""
Eval / run — score the 5 ablation configs against the INDEPENDENT hand labels and write
eval/EVAL_RESULTS.md. Labels are a second opinion from a DIFFERENT model than the Qwen judge, so
the DELTAS between configs are the trustworthy signal, not the absolute numbers.
"""
from __future__ import annotations
import os, sys, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metrics import evaluate, evaluate_many, RELEVANT_TIER

CONFIGS = ["full", "judge_only", "embed_only", "pos_only", "no_availability"]
JUDGE_DEP = {"full", "judge_only", "no_availability"}
METRICS = ["NDCG@10", "NDCG@50", "MAP", "P@10", "composite"]


def load():
    gold = pd.DataFrame([json.loads(l) for l in open(os.path.join(HERE, "gold_set.jsonl"), encoding="utf-8")])
    scores = pd.read_csv(os.path.join(HERE, "ablation_scores.csv"))
    g = gold[["candidate_id", "stratum", "current_title", "years_of_experience",
              "submitted_rank", "label_tier"]].merge(
        scores.drop(columns=["stratum", "submitted_rank"]), on="candidate_id")
    return g


def rank_in(df, col, cid):
    d = df.dropna(subset=[col]).sort_values([col, "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    pos = d.index[d.candidate_id == cid]
    return (int(pos[0]) + 1, len(d)) if len(pos) else (None, len(d))


def main():
    g = load()
    n_rel = int((g.label_tier >= RELEVANT_TIER).sum())
    lines = []
    def P(s=""):
        print(s); lines.append(s)

    P(f"# Gold set: {len(g)} candidates | relevant (tier>=3): {n_rel} | "
      f"label dist: {g.label_tier.value_counts().sort_index().to_dict()}")
    P("Labels are an INDEPENDENT second opinion (different model than the Qwen judge) -> read the "
      "DELTAS between configs, not the absolute values.")

    # ---------- 1. main table (native universe per config) ----------
    tbl = evaluate_many(g, CONFIGS).round(4)
    P("\n## 1. Metrics per config (native universe; embed/pos keep all 47, judge-dep drop 3 stuffers)")
    P(tbl[["n", "n_relevant(tier>=3)"] + METRICS].to_string())
    P("note: full/judge_only/no_availability exclude the 3 keyword-stuffers (no judge score); "
      "embed_only/pos_only include them.")

    # ---------- 2. deltas on the COMMON 44-candidate universe ----------
    g44 = g.dropna(subset=["full"]).copy()   # the 44 with judge scores (drops the 3 stuffers)
    tbl44 = evaluate_many(g44, CONFIGS)
    delta = pd.DataFrame({m: {c: tbl44.loc["full", m] - tbl44.loc[c, m] for c in CONFIGS} for m in METRICS})
    P("\n## 2. `full` minus each config, on the common 44-candidate universe (apples-to-apples)")
    P("(positive = full is better; isolates each component's contribution)")
    P(delta.round(4).to_string())
    P(f"\n  full - judge_only     -> what the STRUCTURED features + availability add on top of the judge")
    P(f"  full - no_availability -> what the AVAILABILITY multiplier alone contributes")
    P(f"  full - embed_only      -> what the JUDGE (+structure) adds over raw ensemble embeddings")
    P(f"  embed_only - pos_only  -> what the NEGATIVE-query subtraction contributes "
      f"(NDCG@10 {tbl44.loc['embed_only','NDCG@10']-tbl44.loc['pos_only','NDCG@10']:+.4f}, "
      f"composite {tbl44.loc['embed_only','composite']-tbl44.loc['pos_only','composite']:+.4f})")

    # ---------- 3. sanity checks ----------
    P("\n## 3. Sanity checks")
    hp = g[g.stratum == "gated_honeypot"]
    hp_mean = hp.embed_only.mean()
    nonfit = g[(g.label_tier <= 1) & (g.stratum != "gated_honeypot")]
    low4 = nonfit.nsmallest(4, "embed_only")
    P(f"(a) HONEYPOTS score HIGH on raw embeddings (the trap):")
    P(f"    mean embed_only — 4 honeypots         = {hp_mean:.4f}  ({', '.join(f'{c}:{v:.3f}' for c,v in zip(hp.candidate_id,hp.embed_only))})")
    P(f"    mean embed_only — 4 lowest non-fits   = {low4.embed_only.mean():.4f}  ({', '.join(f'{c}:{v:.3f}' for c,v in zip(low4.candidate_id,low4.embed_only))})")
    P(f"    => honeypots outrank the low non-fits on pure embeddings by "
      f"{hp_mean-low4.embed_only.mean():+.4f}. Raw retrieval is fooled; the DETERMINISTIC GATE (not the "
      f"score) removes them, and all 4 honeypots are labeled tier 0.")

    P(f"\n(b) availability effect on two candidates (rank within the 44-candidate judged universe):")
    for cid in ["CAND_0041611", "CAND_0092278"]:
        rf, nf = rank_in(g44, "full", cid)
        rn, nn = rank_in(g44, "no_availability", cid)
        sub = g[g.candidate_id == cid].iloc[0]
        P(f"    {cid} (label {int(sub.label_tier)}, submitted_rank {sub.submitted_rank}): "
          f"full=#{rf}/{nf}  no_availability=#{rn}/{nn}  -> availability moves it {rf-rn:+d} places "
          f"({'penalized by' if rf>rn else 'helped by' if rf<rn else 'unchanged under'} availability)")

    P(f"\n(c) recall misses — labeled relevant (tier>=3) but NOT in submitted top-100:")
    miss = g[(g.label_tier >= RELEVANT_TIER) & (g.submitted_rank.isna())]
    if len(miss):
        for _, r in miss.iterrows():
            P(f"    {r.candidate_id} | {r.current_title} | label tier {int(r.label_tier)} | "
              f"stratum {r.stratum} | submitted_rank=None")
    else:
        P("    none.")
    P(f"    ({len(miss)} recall miss{'es' if len(miss)!=1 else ''} in the gold set.)")

    # ---------- write EVAL_RESULTS.md ----------
    with open(os.path.join(HERE, "EVAL_RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("# EVAL_RESULTS — ranking vs. independent hand-labeled gold set\n\n")
        f.write("Labels are an **independent second opinion** (a different model than the Qwen judge), "
                "so the **deltas between configs are the trustworthy signal**, not the absolute values.\n\n")
        f.write(f"Gold set: {len(g)} candidates, {n_rel} relevant (tier>=3). "
                f"Label distribution: {g.label_tier.value_counts().sort_index().to_dict()}\n\n")
        f.write("## 1. Metrics per config\n\n```\n" + tbl[["n","n_relevant(tier>=3)"]+METRICS].to_string() + "\n```\n")
        f.write("_full/judge_only/no_availability exclude the 3 keyword-stuffers (no judge score); "
                "embed_only/pos_only include all 47._\n\n")
        f.write("## 2. `full` minus each config (common 44-candidate universe)\n\n```\n"
                + delta.round(4).to_string() + "\n```\n\n")
        f.write("- `full - judge_only` = value added by structured features + availability over the judge\n")
        f.write("- `full - no_availability` = contribution of the availability multiplier alone\n")
        f.write("- `full - embed_only` = value the judge (+structure) adds over raw ensemble embeddings\n")
        f.write(f"- `embed_only - pos_only` = contribution of the negative-query subtraction "
                f"(composite {tbl44.loc['embed_only','composite']-tbl44.loc['pos_only','composite']:+.4f})\n\n")
        f.write("## 3. Findings\n\n")
        f.write(f"**(a) Honeypots are HIGH on raw embeddings.** mean embed_only: 4 honeypots {hp_mean:.4f} "
                f"vs 4 lowest non-fits {low4.embed_only.mean():.4f} ({hp_mean-low4.embed_only.mean():+.4f}). "
                f"Pure retrieval is fooled by keyword-perfect fabrications; the deterministic gate removes them.\n\n")
        f.write("**(b) Availability multiplier.**\n")
        for cid in ["CAND_0041611", "CAND_0092278"]:
            rf, nf = rank_in(g44, "full", cid); rn, nn = rank_in(g44, "no_availability", cid)
            sub = g[g.candidate_id == cid].iloc[0]
            f.write(f"- {cid} (label {int(sub.label_tier)}, submitted_rank {sub.submitted_rank}): "
                    f"full #{rf}/{nf} vs no_availability #{rn}/{nn} ({rf-rn:+d} places)\n")
        f.write("\n**(c) Recall misses (tier>=3 not in submitted top-100):**\n")
        for _, r in miss.iterrows():
            f.write(f"- {r.candidate_id} | {r.current_title} | tier {int(r.label_tier)} | {r.stratum}\n")
        if not len(miss):
            f.write("- none\n")
    P(f"\nwrote {os.path.join(HERE,'EVAL_RESULTS.md')}")


if __name__ == "__main__":
    main()
