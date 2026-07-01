"""
Stage B / Part 2 — full structured-feature + honeypot-flag + evidence-text build
over all 100,000 candidates. BUILD-TIME ONLY (offline). Streams candidates.jsonl with
orjson (never loads the whole file), writes three parquet artifacts.

Outputs:
  artifacts/features.parquet        (candidate_id + all structured_features)
  artifacts/honeypot_flags.parquet  (candidate_id + gate flags + soft flag + reason strings)
  artifacts/evidence_text.parquet   (candidate_id + evidence_text)
"""
from __future__ import annotations
import os, sys, time
import orjson
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from features import structured_features, honeypot_flags, build_evidence_text

DATA = os.path.join(
    ROOT, "dataset", "[PUB] India_runs_data_and_ai_challenge",
    "India_runs_data_and_ai_challenge", "candidates.jsonl")
ART = os.path.join(ROOT, "artifacts")
os.makedirs(ART, exist_ok=True)


def main():
    t0 = time.time()
    feat_rows, hp_rows, ev_rows = [], [], []
    n = 0
    with open(DATA, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            d = orjson.loads(line)
            cid = d["candidate_id"]

            sf = structured_features(d)
            sf_row = {"candidate_id": cid, **sf}
            feat_rows.append(sf_row)

            hp = honeypot_flags(d)
            hp_rows.append({
                "candidate_id": cid,
                "is_honeypot": hp["is_honeypot"],
                "n_flags": hp["n_flags"],
                "expert_zero_duration": hp["expert_zero_duration"],
                "expert_low_experience": hp["expert_low_experience"],
                "role_tenure_gt_career": hp["role_tenure_gt_career"],
                "career_months_gt_experience": hp["career_months_gt_experience"],
                "career_date_error": hp["career_date_error"],
                "yoe_gt_career_span": hp["yoe_gt_career_span"],
                "soft_skill_duration_gt_career": hp["skill_duration_gt_career"],
                "career_span_months": hp["career_span_months"],
                "summary_years_stated": hp["summary_years_stated"],
                "summary_contradicts_yoe": hp["summary_contradicts_yoe"],
                "reason": hp["reason"],
                "soft_reason": hp["soft_reason"],
            })

            ev_rows.append({"candidate_id": cid,
                            "evidence_text": build_evidence_text(d)})
            n += 1
            if n % 20000 == 0:
                print(f"  ...{n:,} processed ({time.time()-t0:.1f}s)")

    df_feat = pd.DataFrame(feat_rows)
    df_hp = pd.DataFrame(hp_rows)
    df_ev = pd.DataFrame(ev_rows)

    df_feat.to_parquet(os.path.join(ART, "features.parquet"), index=False)
    df_hp.to_parquet(os.path.join(ART, "honeypot_flags.parquet"), index=False)
    df_ev.to_parquet(os.path.join(ART, "evidence_text.parquet"), index=False)

    dt = time.time() - t0
    print(f"\nDONE in {dt:.1f}s  ({n/dt:,.0f} rec/s)")
    print(f"  features.parquet      rows={len(df_feat):,}  cols={list(df_feat.columns)}")
    print(f"  honeypot_flags.parquet rows={len(df_hp):,}  honeypots={int(df_hp.is_honeypot.sum())} "
          f"({100*df_hp.is_honeypot.mean():.3f}%)")
    print(f"  evidence_text.parquet rows={len(df_ev):,}  "
          f"avg_len={int(df_ev.evidence_text.str.len().mean())} chars")
    assert len(df_feat) == len(df_hp) == len(df_ev) == n
    for name in ("features.parquet", "honeypot_flags.parquet", "evidence_text.parquet"):
        mb = os.path.getsize(os.path.join(ART, name)) / 1e6
        print(f"  {name}: {mb:.1f} MB on disk")


if __name__ == "__main__":
    main()
