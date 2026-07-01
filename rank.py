"""
rank.py — the RANKING STEP (spec §3 compute budget: <=5 min, <=16 GB, CPU-only, no network).

Implements ONLY the frozen deterministic path: loads precomputed artifacts, applies the hard
honeypot gate, tiered fusion (judge + structured features, availability multiplier), reads the
frozen top-40 listwise order, assembles the top-100, and writes the submission CSV.

NO Ollama, NO embedding models, NO network. All heavy compute (embeddings, the 985-candidate
LLM-judge pass, the listwise rerank) runs OFFLINE beforehand and is baked into the artifacts —
see README.md ("Full offline reproduction") and DECISIONS.md for the design log.

Usage:
    python rank.py --artifacts artifacts/ --out submission/ranking.csv

Requires only the rank-time deps (numpy, pandas, pyarrow). Deterministic: output is
byte-identical across runs (LF newlines, frozen rerank order, fixed tie-breaks).
"""
from __future__ import annotations
import argparse, json, os, sys, time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from rank_submission import (  # pure functions + recorded weights; no LLM import at module level
    structured_fit, availability_multiplier, make_reasoning,
    W_JUDGE, W_STRUCT, TOP_ZONE,
)


def main():
    ap = argparse.ArgumentParser(description="Frozen deterministic ranking step (CPU-only).")
    ap.add_argument("--artifacts", default=os.path.join(HERE, "artifacts"),
                    help="directory with the precomputed artifacts")
    ap.add_argument("--out", default=os.path.join(HERE, "submission", "ranking.csv"),
                    help="output CSV path")
    ap.add_argument("--candidates", default=None,
                    help="path to candidates.jsonl (accepted for spec-shape compatibility; the "
                         "frozen path ranks from precomputed artifacts and does not re-scan it)")
    args = ap.parse_args()
    art = args.artifacts
    t0 = time.time()

    # ---- load precomputed artifacts (small; the embeddings are NOT needed at rank time) ----
    short = pd.read_parquet(os.path.join(art, "shortlist.parquet"))
    judg = pd.read_parquet(os.path.join(art, "judgments.parquet"))
    feats = pd.read_parquet(os.path.join(art, "features.parquet"))[
        ["candidate_id", "product_vs_services", "career_coherence"]]
    frozen_path = os.path.join(art, "frozen_rerank_order.json")
    if not os.path.exists(frozen_path):
        sys.exit("ERROR: frozen_rerank_order.json missing — the ranking step is only the "
                 "deterministic frozen path. Regenerate it offline with "
                 "`python src/rank_submission.py --freeze` (judge-time environment).")
    frozen_order = json.load(open(frozen_path, encoding="utf-8"))["order"]

    df = short.merge(judg[["candidate_id", "fit_tier", "fit_score", "key_evidence", "concerns",
                           "availability_note", "reasoning", "honeypot_suspicion"]],
                     on="candidate_id", how="left")
    df = df.merge(feats, on="candidate_id", how="left")
    n0 = len(df)

    # ---- PART 1: hard honeypot gate FIRST (deterministic; never delegated to a model) ----
    df = df[df.is_honeypot_gated != True].copy()
    df = df[df.fit_tier != 0].copy()
    assert (df.is_honeypot_gated == True).sum() == 0
    print(f"honeypot gate: pool {n0} -> {len(df)}")

    # ---- PART 2: tiered fusion ----
    df["structured_fit"] = df.apply(structured_fit, axis=1)
    df["avail_mult"] = df.apply(availability_multiplier, axis=1)
    df["fit_score"] = df["fit_score"].fillna(0.0)
    df["within_tier_score"] = ((W_JUDGE * df.fit_score + W_STRUCT * df.structured_fit)
                               * df.avail_mult)
    df = df.sort_values(["fit_tier", "within_tier_score", "candidate_id"],
                        ascending=[False, False, True]).reset_index(drop=True)

    # ---- PART 3: frozen listwise order for the top zone (no LLM call) ----
    top_ids = list(df.head(TOP_ZONE).candidate_id)
    assert set(frozen_order) == set(top_ids), \
        "frozen order set mismatch with current top-40 (upstream artifacts changed)"
    final_ids = frozen_order + [c for c in df.candidate_id if c not in set(frozen_order)]

    # GUARDS: membership + no tier inversion
    assert len(final_ids) == len(set(final_ids)) == len(df)
    tier = dict(zip(df.candidate_id, df.fit_tier))
    for a, b in zip(final_ids, final_ids[1:]):
        assert tier[a] >= tier[b], f"tier inversion: {a} before {b}"

    # ---- PART 4: assemble top-100 ----
    out = df.set_index("candidate_id").loc[final_ids[:100]].reset_index()
    out["rank"] = np.arange(1, 101)
    out["score"] = (0.99 - (out["rank"] - 1) * (0.49 / 99.0)).round(4)
    assert (out["score"].diff().dropna() < 0).all()
    used = set()
    out["reasoning"] = [make_reasoning(r, used) for r in out.itertuples()]
    assert out["reasoning"].nunique() == 100
    assert out["reasoning"].str.len().max() <= 240

    # ---- PART 5: write CSV (LF newlines for byte-stable output) ----
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out[["candidate_id", "rank", "score", "reasoning"]].to_csv(
        args.out, index=False, encoding="utf-8", lineterminator="\n")
    print(f"wrote {args.out}  ({time.time() - t0:.2f}s wall-clock)")


if __name__ == "__main__":
    main()
