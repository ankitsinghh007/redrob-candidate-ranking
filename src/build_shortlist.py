"""
Stage C / Part 0 — build the judge shortlist: top-800 by fused_fit UNION build-signal
candidates. Honeypots are KEPT IN (flagged is_honeypot_gated) so the judge preview can be
validated to flag them independently. Bundles the structured features + evidence_text the
judge will read.

Output: artifacts/shortlist.parquet
"""
from __future__ import annotations
import os, re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")

TOP_N = 800
LAMBDA = 0.5
RRF_K = 60
BUILD_SIGNAL = re.compile(
    r"\b(?:built|build|building|shipped|ship|designed|design|developed|develop|launched|owned)\b"
    r".{0,45}\b(?:recommendation|recommender|ranking|rank|search|retrieval|recsys|relevance|"
    r"matching|personali[sz]ation)\b.{0,30}\b(?:system|systems|engine|pipeline|model|models|"
    r"platform|infrastructure|stack)\b",
    re.IGNORECASE | re.DOTALL)

JUDGE_FEATURE_COLS = [
    "title", "title_family", "years_of_experience", "yoe_fit",
    "is_services_company", "services_only_career", "services_ratio",
    "days_since_last_active", "recruiter_response_rate", "notice_period_days",
    "open_to_work", "willing_to_relocate", "location_preferred", "country_in_india",
]


def main():
    bge = np.load(os.path.join(ART, "emb_bge.npy")).astype(np.float32)
    e5 = np.load(os.path.join(ART, "emb_e5.npy")).astype(np.float32)
    ids = np.load(os.path.join(ART, "emb_ids.npy"), allow_pickle=True)
    bge_q = np.load(os.path.join(ART, "emb_bge_q.npy")).astype(np.float32)
    e5_q = np.load(os.path.join(ART, "emb_e5_q.npy")).astype(np.float32)
    feat = pd.read_parquet(os.path.join(ART, "features.parquet")).set_index("candidate_id")
    hp = pd.read_parquet(os.path.join(ART, "honeypot_flags.parquet")).set_index("candidate_id")
    ev = pd.read_parquet(os.path.join(ART, "evidence_text.parquet")).set_index("candidate_id")

    n = len(ids)
    s_bge = bge @ bge_q[0] - LAMBDA * (bge @ bge_q[1])
    s_e5 = e5 @ e5_q[0] - LAMBDA * (e5 @ e5_q[1])

    def ranks_of(s):
        order = np.argsort(-s)
        r = np.empty(n, dtype=np.int64)
        r[order] = np.arange(1, n + 1)
        return r
    fused = 1.0 / (RRF_K + ranks_of(s_bge)) + 1.0 / (RRF_K + ranks_of(s_e5))

    df = pd.DataFrame({"candidate_id": ids, "fused_fit": fused})
    df = df.sort_values("fused_fit", ascending=False).reset_index(drop=True)
    df["fused_rank"] = np.arange(1, n + 1)

    evtxt = ev.loc[df.candidate_id, "evidence_text"].fillna("").to_numpy()
    df["build_signal"] = pd.Series(evtxt).str.contains(BUILD_SIGNAL).to_numpy()

    in_top = df["fused_rank"] <= TOP_N
    in_build = df["build_signal"].to_numpy()
    keep = in_top | in_build
    short = df[keep].copy()

    def reason(row):
        if (row.fused_rank <= TOP_N) and row.build_signal:
            return "both"
        return "retrieval_rank" if row.fused_rank <= TOP_N else "build_signal"
    short["inclusion_reason"] = short.apply(reason, axis=1)

    # honeypots KEPT IN, flagged
    short = short.join(hp[["is_honeypot", "reason"]], on="candidate_id")
    short = short.rename(columns={"is_honeypot": "is_honeypot_gated",
                                  "reason": "honeypot_reason"})
    short = short.join(feat[JUDGE_FEATURE_COLS], on="candidate_id")
    short = short.join(ev[["evidence_text"]], on="candidate_id")

    short = short.sort_values("fused_rank").reset_index(drop=True)
    short.to_parquet(os.path.join(ART, "shortlist.parquet"), index=False)

    print(f"shortlist size: {len(short)}")
    print(f"  inclusion_reason: {short.inclusion_reason.value_counts().to_dict()}")
    print(f"  is_honeypot_gated (kept in, flagged): {int(short.is_honeypot_gated.sum())}")
    print(f"  title_family mix: {short.title_family.value_counts().to_dict()}")
    print(f"  top-800 by fused: {int(in_top.sum())}  | build_signal total: {int(in_build.sum())}"
          f"  | build-only added beyond top-800: {int((in_build & ~in_top).sum())}")
    print(f"  columns: {list(short.columns)}")
    print(f"  saved artifacts/shortlist.parquet "
          f"({os.path.getsize(os.path.join(ART,'shortlist.parquet'))/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
