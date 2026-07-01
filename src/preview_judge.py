"""
Stage C / Part 3 — PREVIEW the judge on ~18 curated candidates spanning the spectrum.
NOT the full run. Prints a verdict table + the calibration report.
"""
from __future__ import annotations
import os, sys, json, io, time
import numpy as np
import pandas as pd
import orjson

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from judge import judge_candidate, load_rubric, ensure_model

ART = os.path.join(ROOT, "artifacts")
JSONL = os.path.join(ROOT, "dataset", "[PUB] India_runs_data_and_ai_challenge",
                     "India_runs_data_and_ai_challenge", "candidates.jsonl")

AI_SKILLS = {s.lower() for s in [
    "NLP", "RAG", "LLM", "LLMs", "Fine-tuning LLMs", "Transformers", "PyTorch", "TensorFlow",
    "Pinecone", "Milvus", "FAISS", "Weaviate", "Qdrant", "Embeddings", "Information Retrieval",
    "Recommendation Systems", "LangChain", "PEFT", "LoRA", "BERT", "Vector Search",
    "Semantic Search", "Deep Learning", "MLOps", "Feature Engineering", "XGBoost", "CNN", "GANs"]}


def curate():
    feat = pd.read_parquet(os.path.join(ART, "features.parquet")).set_index("candidate_id")
    ev = pd.read_parquet(os.path.join(ART, "evidence_text.parquet")).set_index("candidate_id")
    hp = pd.read_parquet(os.path.join(ART, "honeypot_flags.parquet")).set_index("candidate_id")
    short = pd.read_parquet(os.path.join(ART, "shortlist.parquet")).set_index("candidate_id")

    picks = {}  # cid -> bucket label

    # 1) top-fused AI/ML engineers (4)
    top = short.sort_values("fused_rank").head(60)
    for cid in top[top.title_family == "ai_ml"].index[:4]:
        picks[cid] = "top_fused_ai_ml"

    # 2) plain-language fits with NON-ml title: candidates that surfaced into the shortlist by
    #    TEXT (high fused_rank) despite a swe/data title — the recall-rescue cases. (After the
    #    title fix, the old "(ML)"-suffixed builders are correctly ai_ml, so non-ml builders are
    #    now the text-surfaced swe/data engineers.) Prefer ones with an explicit build_signal.
    nonml = short[short.title_family.isin(["swe", "data"])].sort_values(
        ["build_signal", "fused_rank"], ascending=[False, True])
    for cid in nonml.index[:4]:
        picks.setdefault(cid, "plainlang_nonml")

    # 3) gated honeypots (3) + force CAND_0016000
    for cid in short[short.is_honeypot_gated].index[:3]:
        picks.setdefault(cid, "honeypot")
    picks.setdefault("CAND_0016000", "honeypot_forced")

    # 4) CAND_0000031 rescued recsys engineer
    picks.setdefault("CAND_0000031", "rescued_recsys")

    # 5) services-only seniors (2) — look superficially relevant (ai_ml/swe), all-services career
    svc = feat[(feat.services_only_career) & (feat.years_of_experience >= 8) &
               (feat.title_family.isin(["ai_ml", "swe", "data"]))]
    for cid in svc.index[:2]:
        picks.setdefault(cid, "services_only_senior")

    # 6) keyword-stuffers (2): nontech title + many AI skills (scan raw jsonl)
    stuffers = []
    with open(JSONL, "rb") as f:
        for line in f:
            d = orjson.loads(line)
            cid = d["candidate_id"]
            if feat.loc[cid, "title_family"] != "nontech":
                continue
            n_ai = sum(1 for s in (d.get("skills") or [])
                       if (s.get("name") or "").lower() in AI_SKILLS)
            if n_ai >= 6:
                stuffers.append((cid, n_ai))
            if len(stuffers) >= 40:
                break
    stuffers.sort(key=lambda x: -x[1])
    for cid, _ in stuffers[:2]:
        picks.setdefault(cid, "keyword_stuffer")

    # assemble rows (works for any cid via features + evidence + honeypot)
    rows = []
    for cid, bucket in picks.items():
        if cid not in feat.index:
            continue
        r = feat.loc[cid].to_dict()
        r["candidate_id"] = cid
        r["evidence_text"] = ev.loc[cid, "evidence_text"] if cid in ev.index else ""
        r["bucket"] = bucket
        r["is_honeypot_gated"] = bool(hp.loc[cid, "is_honeypot"]) if cid in hp.index else False
        rows.append(r)
    return rows


def main():
    ensure_model()
    rubric = load_rubric()
    rows = curate()
    print(f"\ncurated {len(rows)} candidates across buckets: "
          f"{pd.Series([r['bucket'] for r in rows]).value_counts().to_dict()}\n")

    results, lat, ok = [], [], 0
    for r in rows:
        v, meta = judge_candidate(r, rubric=rubric)
        lat.append(meta["latency_s"])
        if meta["ok"]:
            ok += 1
            results.append({
                "candidate_id": r["candidate_id"], "bucket": r["bucket"],
                "title": r["title"], "fam": r["title_family"],
                "gated_hp": r["is_honeypot_gated"],
                "fit_tier": v.fit_tier, "fit_score": round(v.fit_score, 3),
                "hp_susp": v.honeypot_suspicion,
                "reasoning": v.reasoning,
                "key_evidence": v.key_evidence, "concerns": v.concerns,
                "avail": v.availability_note,
                "latency_s": round(meta["latency_s"], 1)})
        else:
            results.append({"candidate_id": r["candidate_id"], "bucket": r["bucket"],
                            "title": r["title"], "fam": r["title_family"],
                            "gated_hp": r["is_honeypot_gated"], "fit_tier": None,
                            "fit_score": None, "hp_susp": None,
                            "reasoning": f"[PARSE FAIL] {meta.get('error','')}",
                            "key_evidence": [], "concerns": [], "avail": "",
                            "latency_s": round(meta["latency_s"], 1)})
        print(f"  judged {r['candidate_id']} ({r['bucket']}) tier="
              f"{results[-1]['fit_tier']} {meta['latency_s']:.1f}s")

    df = pd.DataFrame(results)
    df.to_json(os.path.join(ART, "judge_preview.jsonl"), orient="records", lines=True)

    print("\n=== JUDGE PREVIEW TABLE ===")
    with pd.option_context("display.max_colwidth", 60, "display.width", 240):
        print(df[["candidate_id", "bucket", "title", "fam", "gated_hp",
                  "fit_tier", "fit_score", "hp_susp", "latency_s"]].to_string(index=False))

    print("\n=== one-line reasoning per candidate ===")
    for _, r in df.iterrows():
        print(f"\n[{r.candidate_id} | {r.bucket} | tier={r.fit_tier} score={r.fit_score} "
              f"hp_susp={r.hp_susp}]\n  {r.title}: {r.reasoning}")
        if isinstance(r.key_evidence, list) and r.key_evidence:
            print(f"  evidence_cited: {r.key_evidence[:3]}")
        if isinstance(r.concerns, list) and r.concerns:
            print(f"  concerns: {r.concerns[:3]}")

    # ---------- report ----------
    print("\n\n========== CALIBRATION REPORT ==========")
    good = df[df.fit_tier.notna()]
    print(f"JSON parse success: {ok}/{len(df)} ({100*ok/len(df):.0f}%)")
    print(f"mean latency/candidate: {np.mean(lat):.1f}s  median: {np.median(lat):.1f}s")
    print(f"  -> extrapolated 985-candidate full run: "
          f"{np.mean(lat)*985/60:.0f} min ({np.mean(lat)*985/3600:.1f} h)")

    def bucket_view(b):
        sub = df[df.bucket.str.startswith(b)]
        return sub[["candidate_id", "fit_tier", "fit_score", "hp_susp"]]

    print("\n[Honeypots] expect tier 0 + hp_susp=True:")
    print(df[df.bucket.str.startswith("honeypot")][
        ["candidate_id", "title", "fit_tier", "hp_susp"]].to_string(index=False))
    print("\n[Keyword-stuffers] expect LOW tier despite rich AI skills:")
    print(bucket_view("keyword_stuffer").to_string(index=False))
    print("\n[Real builders: top_fused + plainlang_nonml + rescued] expect HIGH tier:")
    builders = df[df.bucket.isin(["top_fused_ai_ml", "plainlang_nonml", "rescued_recsys"])]
    print(builders[["candidate_id", "bucket", "title", "fit_tier", "fit_score"]].to_string(index=False))
    print(f"  high-tier(>=3) rate among builders: "
          f"{100*(builders.fit_tier>=3).mean():.0f}%")
    print("\n[Services-only seniors] expect disqualifier pressure (lower tier):")
    print(bucket_view("services_only").to_string(index=False))


if __name__ == "__main__":
    main()
