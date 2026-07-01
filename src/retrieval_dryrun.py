"""
Stage B / Part 5 — retrieval DRY-RUN (NO LLM judge). Validates recall of pure-ensemble
retrieval before committing to the judge stage.

Per model: cosine(candidate, positive) and cosine(candidate, negative);
score = pos - lambda*neg (lambda=0.5). Fuse the two models with Reciprocal Rank Fusion (RRF).
Shortlist = top-N by fused score (NO title pre-filter = the recall rescue) UNION any candidate
whose evidence text shows a concrete build-signal phrase. Then validate.
"""
from __future__ import annotations
import os, re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")

LAMBDA = 0.5
RRF_K = 60
RARE_TRUE_TITLES = {
    "AI Engineer", "Senior AI Engineer", "ML Engineer", "AI Research Engineer", "Data Scientist",
}
BUILD_SIGNAL = re.compile(
    r"\b(?:built|build|building|shipped|ship|designed|design|developed|develop|launched|owned)\b"
    r".{0,45}\b(?:recommendation|recommender|ranking|rank|search|retrieval|recsys|relevance|"
    r"matching|personali[sz]ation)\b.{0,30}\b(?:system|systems|engine|pipeline|model|models|"
    r"platform|infrastructure|stack)\b",
    re.IGNORECASE | re.DOTALL)


def load():
    bge = np.load(os.path.join(ART, "emb_bge.npy")).astype(np.float32)
    e5 = np.load(os.path.join(ART, "emb_e5.npy")).astype(np.float32)
    ids = np.load(os.path.join(ART, "emb_ids.npy"), allow_pickle=True)
    bge_q = np.load(os.path.join(ART, "emb_bge_q.npy")).astype(np.float32)
    e5_q = np.load(os.path.join(ART, "emb_e5_q.npy")).astype(np.float32)
    feat = pd.read_parquet(os.path.join(ART, "features.parquet")).set_index("candidate_id")
    hp = pd.read_parquet(os.path.join(ART, "honeypot_flags.parquet")).set_index("candidate_id")
    ev = pd.read_parquet(os.path.join(ART, "evidence_text.parquet")).set_index("candidate_id")
    return bge, e5, ids, bge_q, e5_q, feat, hp, ev


def model_score(emb, q):
    pos = emb @ q[0]
    neg = emb @ q[1]
    return pos - LAMBDA * neg, pos, neg


def rrf_from_scores(*score_arrays):
    """RRF over multiple per-model score arrays (higher score = better)."""
    n = len(score_arrays[0])
    rrf = np.zeros(n, dtype=np.float64)
    for s in score_arrays:
        order = np.argsort(-s)            # indices best->worst
        ranks = np.empty(n, dtype=np.int64)
        ranks[order] = np.arange(1, n + 1)  # rank 1 = best
        rrf += 1.0 / (RRF_K + ranks)
    return rrf


def main():
    bge, e5, ids, bge_q, e5_q, feat, hp, ev = load()
    n = len(ids)
    print(f"loaded embeddings: bge{bge.shape} e5{e5.shape} ids={n}")

    s_bge, pos_bge, _ = model_score(bge, bge_q)
    s_e5, pos_e5, _ = model_score(e5, e5_q)
    fused = rrf_from_scores(s_bge, s_e5)

    df = pd.DataFrame({
        "candidate_id": ids,
        "fused_fit": fused,
        "pos_bge": pos_bge, "pos_e5": pos_e5,
        "s_bge": s_bge, "s_e5": s_e5,
    })
    df = df.join(feat[["title", "title_family", "years_of_experience"]], on="candidate_id")
    df = df.join(hp[["is_honeypot", "reason"]], on="candidate_id")
    df = df.sort_values("fused_fit", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    rank_of = dict(zip(df.candidate_id, df["rank"]))

    # ---- 1. top-50 by fused_fit ----
    print("\n=== TOP-50 by fused_fit (no title pre-filter) ===")
    top = df.head(50)
    with pd.option_context("display.max_colwidth", 30, "display.width", 200):
        print(top[["rank", "candidate_id", "title", "title_family",
                   "years_of_experience", "fused_fit", "is_honeypot"]]
              .to_string(index=False,
                         formatters={"fused_fit": lambda x: f"{x:.5f}"}))
    print(f"\n  title_family mix in top-50: "
          f"{top.title_family.value_counts().to_dict()}")

    # ---- 2. where do rare true-title candidates land? ----
    print("\n=== Rank distribution of rare true-title candidates ===")
    rare = df[df.title.isin(RARE_TRUE_TITLES)]
    for t in sorted(RARE_TRUE_TITLES):
        sub = rare[rare.title == t]
        if len(sub) == 0:
            print(f"  {t:24s} : (none in pool)"); continue
        rk = sub["rank"].to_numpy()
        print(f"  {t:24s} : n={len(sub):4d}  best={rk.min():5d}  median={int(np.median(rk)):6d}"
              f"  in_top300={int((rk<=300).sum())}  in_top600={int((rk<=600).sum())}"
              f"  in_top1000={int((rk<=1000).sum())}")
    allrk = rare["rank"].to_numpy()
    print(f"  ALL rare-title    : n={len(rare)}  in_top300={int((allrk<=300).sum())}"
          f"  in_top600={int((allrk<=600).sum())}  in_top1000={int((allrk<=1000).sum())}")

    # ---- 3. CAND_0000001 headline recall test ----
    c1 = "CAND_0000001"
    print(f"\n=== CAND_0000001 (Backend Engineer, plain-language ML fit) ===")
    print(f"  fused_fit rank = {rank_of.get(c1)} / {n}   "
          f"(percentile {100*(1-rank_of[c1]/n):.2f})")

    # ---- 4. honeypot DQ-threat in top 300 ----
    top300 = df.head(300)
    hp_in_300 = top300[top300.is_honeypot]
    print(f"\n=== Honeypot DQ-threat ===")
    print(f"  gated honeypots appearing in top-300 (pre-gate): {len(hp_in_300)}")
    if len(hp_in_300):
        print(hp_in_300[["rank", "candidate_id", "title", "reason"]].to_string(index=False))
    print(f"  -> after gating (drop is_honeypot) they are removed from the shortlist.")

    # ---- 5. build-signal UNION + recommend N ----
    evtxt = ev.loc[df.candidate_id, "evidence_text"].fillna("")
    df["build_signal"] = evtxt.str.contains(BUILD_SIGNAL).to_numpy()
    bs = df[df.build_signal]
    print(f"\n=== Build-signal UNION (plain-language fit guarantee) ===")
    print(f"  candidates with concrete build-signal phrase: {int(df.build_signal.sum())}")
    print(f"  of those, rank>600 (would be MISSED by top-600 alone): "
          f"{int((bs['rank']>600).sum())}  -> rescued by UNION")

    for N in (300, 500, 600, 800, 1000):
        gated = df[(df["rank"] <= N) & (~df.is_honeypot)]
        union = df[((df["rank"] <= N) & (~df.is_honeypot)) | (df.build_signal & ~df.is_honeypot)]
        rare_in = int((rare["rank"] <= N).sum())
        print(f"  N={N:5d}: shortlist(top-N, gated)={len(gated):4d}  "
              f"+build-union={len(union):4d}  rare-true-titles<=N={rare_in}/{len(rare)}")

    print("\nDONE (dry-run only — no LLM, no models trained).")


if __name__ == "__main__":
    main()
