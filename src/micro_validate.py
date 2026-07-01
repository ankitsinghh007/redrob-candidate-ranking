"""
Stage C-final / Part 2 — micro-validation of the TIMELINE CONSISTENCY fact on 8 candidates:
the 4 gated honeypots + 4 genuine fits. Confirms (a) honeypots move toward lower tier and/or
honeypot_suspicion=True, and (b) genuine fits keep essentially the same tier/score (no perturbation).
"""
from __future__ import annotations
import os, sys, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from judge import judge_candidate, load_rubric, timeline_status

ART = os.path.join(ROOT, "artifacts")

HONEYPOTS = ["CAND_0093547", "CAND_0001610", "CAND_0019480", "CAND_0016000"]
GENUINE = ["CAND_0007411", "CAND_0055905", "CAND_0008425", "CAND_0071974"]
FLAG_COLS = ["role_tenure_gt_career", "career_months_gt_experience", "expert_low_experience"]


def main():
    feat = pd.read_parquet(os.path.join(ART, "features.parquet")).set_index("candidate_id")
    ev = pd.read_parquet(os.path.join(ART, "evidence_text.parquet")).set_index("candidate_id")
    hp = pd.read_parquet(os.path.join(ART, "honeypot_flags.parquet")).set_index("candidate_id")

    # "before" baseline from the Stage-C preview (judged WITHOUT the timeline fact)
    before = {}
    pv = os.path.join(ART, "judge_preview.jsonl")
    if os.path.exists(pv):
        for line in open(pv, encoding="utf-8"):
            r = json.loads(line)
            before[r["candidate_id"]] = (r.get("fit_tier"), r.get("fit_score"), r.get("hp_susp"))

    rubric = load_rubric()
    rows = []
    for cid in HONEYPOTS + GENUINE:
        r = feat.loc[cid].to_dict()
        r["candidate_id"] = cid
        r["evidence_text"] = ev.loc[cid, "evidence_text"] if cid in ev.index else ""
        for c in FLAG_COLS:
            r[c] = bool(hp.loc[cid, c]) if cid in hp.index else False
        kind = "honeypot" if cid in HONEYPOTS else "genuine_fit"
        v, meta = judge_candidate(r, rubric=rubric)
        b_tier, b_score, b_susp = before.get(cid, (None, None, None))
        rows.append({
            "candidate_id": cid, "kind": kind, "title": r.get("title"),
            "timeline": timeline_status(r).split(" — ")[0],
            "tier_before": b_tier, "tier_after": (v.fit_tier if v else None),
            "score_before": b_score, "score_after": (round(v.fit_score, 3) if v else None),
            "susp_before": b_susp, "susp_after": (v.honeypot_suspicion if v else None),
            "hp_reason_after": (v.honeypot_reason if v else "")[:80],
            "ok": meta["ok"],
        })
        print(f"  judged {cid} ({kind}) tier {b_tier}->{rows[-1]['tier_after']} "
              f"susp {b_susp}->{rows[-1]['susp_after']}")

    df = pd.DataFrame(rows)
    print("\n=== BEFORE / AFTER (timeline fact) ===")
    with pd.option_context("display.width", 240, "display.max_colwidth", 40):
        print(df[["candidate_id", "kind", "title", "timeline", "tier_before", "tier_after",
                  "score_before", "score_after", "susp_before", "susp_after"]].to_string(index=False))

    hpd = df[df.kind == "honeypot"]
    gd = df[df.kind == "genuine_fit"]
    moved = hpd[(hpd.susp_after == True) | (hpd.tier_after < hpd.tier_before.fillna(99))]
    print(f"\nHoneypots that moved (lower tier and/or suspicion=True): {len(moved)}/{len(hpd)}")
    print(df[df.kind == "honeypot"][["candidate_id", "tier_after", "susp_after",
                                     "hp_reason_after"]].to_string(index=False))

    # genuine-fit perturbation check
    gd2 = gd.dropna(subset=["tier_before"])
    tier_drop = gd2[gd2.tier_after < gd2.tier_before]
    print(f"\nGenuine fits unchanged tier: {(gd2.tier_after == gd2.tier_before).sum()}/{len(gd2)}; "
          f"dropped: {len(tier_drop)}")
    max_score_delta = (gd2.score_after - gd2.score_before).abs().max() if len(gd2) else 0
    print(f"  max |score delta| among genuine fits: {max_score_delta:.3f}")
    if len(tier_drop):
        print("  WARNING — a genuine fit dropped tier; soften prompt before proceeding:")
        print(tier_drop[["candidate_id", "tier_before", "tier_after"]].to_string(index=False))
    else:
        print("  PASS — no genuine fit dropped tier; calibration preserved.")


if __name__ == "__main__":
    main()
