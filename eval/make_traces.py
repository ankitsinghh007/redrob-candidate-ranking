"""
Eval / PART 4 — write eval/traces.md: the full reasoning chain for 5 representative candidates
(evidence_text -> structured facts block -> judge verdict -> final rank), so we can watch the
system reason on real records.
"""
from __future__ import annotations
import os, sys, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, "artifacts")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from judge import build_facts_block, timeline_status
from export_gold import master_order

PICKS = [
    ("top-10 fit", "CAND_0061257"),
    ("tier boundary (~rank 48)", "CAND_0000031"),
    ("gated honeypot", "CAND_0093547"),
    ("plain-language builder (non-ML title)", "CAND_0085706"),
    ("clear non-fit from deep shortlist", "CAND_0041696"),
]


def _loads(x):
    try:
        return json.loads(x) if isinstance(x, str) else (x or [])
    except Exception:
        return []


def main():
    feats = pd.read_parquet(os.path.join(ART, "features.parquet")).set_index("candidate_id")
    ev = pd.read_parquet(os.path.join(ART, "evidence_text.parquet")).set_index("candidate_id")
    hp = pd.read_parquet(os.path.join(ART, "honeypot_flags.parquet")).set_index("candidate_id")
    judg = pd.read_parquet(os.path.join(ART, "judgments.parquet")).set_index("candidate_id")
    _, rank = master_order()

    out = ["# eval/traces.md — full reasoning chain on 5 real records",
           "",
           "For each: **evidence_text** (what the recruiter reads) -> **precomputed facts** (fed to the "
           "judge) -> **judge verdict** -> **final submitted rank**. The judge's tier is shown here for "
           "inspection only; it is NOT used as a gold label (that would be circular).", ""]

    for role, cid in PICKS:
        f = feats.loc[cid]
        row = f.to_dict()
        row["candidate_id"] = cid
        row["title"] = f["title"]
        for c in ["role_tenure_gt_career", "career_months_gt_experience", "expert_low_experience"]:
            row[c] = bool(hp.loc[cid, c]) if cid in hp.index else False
        facts = build_facts_block(row)
        evidence = str(ev.loc[cid, "evidence_text"]) if cid in ev.index else "(none)"

        mr = rank.get(cid)
        gated = bool(hp.loc[cid, "is_honeypot"]) if cid in hp.index else False
        if gated:
            final = "**DROPPED before ranking** (is_honeypot_gated=True — hard honeypot gate)"
        elif mr is None:
            final = "not in the shortlist pool (never ranked)"
        elif mr <= 100:
            final = f"**rank {mr} / 100** (in the submission)"
        else:
            final = f"ranked {mr} of 981 — **outside the top-100** (not submitted)"

        out.append(f"\n---\n\n## {role} — `{cid}`\n")
        out.append(f"**Final outcome:** {final}\n")

        if cid in judg.index:
            j = judg.loc[cid]
            ke = _loads(j["key_evidence"]); co = _loads(j["concerns"])
            out.append("**Judge verdict** (for inspection only, not a label):")
            out.append(f"- fit_tier: **{int(j['fit_tier'])}**   fit_score: **{float(j['fit_score']):.2f}**   "
                       f"honeypot_suspicion: **{bool(j['honeypot_suspicion'])}**   "
                       f"timeline_failed: **{bool(j['timeline_failed'])}**")
            out.append(f"- reasoning: {j['reasoning']}")
            out.append("- key_evidence:")
            out += [f"    - {x}" for x in ke]
            out.append("- concerns:" + (" (none)" if not co else ""))
            out += [f"    - {x}" for x in co]
            if str(j.get("honeypot_reason", "")).strip():
                out.append(f"- honeypot_reason: {j['honeypot_reason']}")
        else:
            out.append("**Judge verdict:** (candidate not in the judged shortlist — no verdict)")

        out.append("\n**Precomputed facts block (fed to the judge):**\n```")
        out.append(facts)
        out.append(f"timeline_status(): {timeline_status(row)}")
        out.append("```\n")
        out.append("**evidence_text (full — the primary signal, skills[] excluded):**\n```")
        out.append(evidence)
        out.append("```")

    path = os.path.join(HERE, "traces.md")
    with open(path, "w", encoding="utf-8") as fo:
        fo.write("\n".join(out))
    print(f"wrote {path}  ({len(PICKS)} candidates)")
    for role, cid in PICKS:
        mr = rank.get(cid)
        gated = bool(hp.loc[cid, "is_honeypot"]) if cid in hp.index else False
        tier = int(judg.loc[cid, "fit_tier"]) if cid in judg.index else "-"
        where = "DROPPED(honeypot)" if gated else (f"rank {mr}" if mr else "unranked")
        print(f"  {role:38s} {cid}  judge_tier={tier}  {where}")


if __name__ == "__main__":
    main()
