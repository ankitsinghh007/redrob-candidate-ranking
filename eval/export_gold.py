"""
Eval / PART 1 — export a stratified gold set for INDEPENDENT hand-labeling.

Writes eval/gold_set.jsonl (~50 candidates), each with the facts a human needs to assign a
label_tier (0-4) WITHOUT looking at the Qwen judge's tier (that would be circular). The judge's
scores are NOT included here. Prints the whole set to stdout (evidence truncated) for copy-out.

Strata: top-10 / tier-boundary (44-52) / cut-line (90-110, incl CAND_0092278) / deep shortlist
(300-800) / 4 gated honeypots / 3 keyword-stuffers / 3 plain-language builders.
"""
from __future__ import annotations
import os, sys, json, re
import pandas as pd, numpy as np, orjson

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, "artifacts")
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
from rank_submission import structured_fit, availability_multiplier, W_JUDGE, W_STRUCT, TOP_ZONE
JSONL = os.path.join(ROOT, "dataset", "[PUB] India_runs_data_and_ai_challenge",
                     "India_runs_data_and_ai_challenge", "candidates.jsonl")
OUT = os.path.join(HERE, "gold_set.jsonl")

AI_SKILLS = {s.lower() for s in [
    "NLP", "RAG", "LLM", "LLMs", "Fine-tuning LLMs", "Transformers", "PyTorch", "TensorFlow",
    "Pinecone", "Milvus", "FAISS", "Weaviate", "Qdrant", "Embeddings", "Information Retrieval",
    "Recommendation Systems", "LangChain", "PEFT", "LoRA", "BERT", "Vector Search",
    "Semantic Search", "Deep Learning", "MLOps", "Feature Engineering", "XGBoost", "CNN", "GANs"]}
BUILD = re.compile(
    r"\b(?:built|build|building|shipped|ship|designed|design|developed|develop|launched|owned)\b"
    r".{0,45}\b(?:recommendation|recommender|ranking|rank|search|retrieval|recsys|relevance|"
    r"matching|personali[sz]ation)\b.{0,30}\b(?:system|systems|engine|pipeline|model|models|"
    r"platform|infrastructure|stack)\b", re.IGNORECASE | re.DOTALL)


def master_order():
    """Reconstruct the exact final 981-candidate ordering (matches the submission for 1-100)."""
    short = pd.read_parquet(os.path.join(ART, "shortlist.parquet"))
    judg = pd.read_parquet(os.path.join(ART, "judgments.parquet"))[["candidate_id", "fit_tier", "fit_score"]]
    feats = pd.read_parquet(os.path.join(ART, "features.parquet"))[
        ["candidate_id", "product_vs_services", "career_coherence"]]
    df = short.merge(judg, on="candidate_id", how="left").merge(feats, on="candidate_id", how="left")
    df = df[(df.is_honeypot_gated != True) & (df.fit_tier != 0)].copy()
    df["structured_fit"] = df.apply(structured_fit, axis=1)
    df["avail_mult"] = df.apply(availability_multiplier, axis=1)
    df["fit_score"] = df.fit_score.fillna(0.0)
    df["within_tier_score"] = (W_JUDGE * df.fit_score + W_STRUCT * df.structured_fit) * df.avail_mult
    df = df.sort_values(["fit_tier", "within_tier_score", "candidate_id"],
                        ascending=[False, False, True]).reset_index(drop=True)
    frozen = json.load(open(os.path.join(ART, "frozen_rerank_order.json"), encoding="utf-8"))["order"]
    top40 = list(df.candidate_id[:TOP_ZONE])
    assert set(frozen) == set(top40), "frozen set != current top-40"
    rest = [c for c in df.candidate_id[TOP_ZONE:]]
    final = frozen + rest
    rank = {c: i + 1 for i, c in enumerate(final)}
    return df.set_index("candidate_id"), rank


def main():
    feats = pd.read_parquet(os.path.join(ART, "features.parquet")).set_index("candidate_id")
    ev = pd.read_parquet(os.path.join(ART, "evidence_text.parquet")).set_index("candidate_id")
    hp = pd.read_parquet(os.path.join(ART, "honeypot_flags.parquet")).set_index("candidate_id")
    short = pd.read_parquet(os.path.join(ART, "shortlist.parquet")).set_index("candidate_id")
    scored, rank = master_order()
    inv = {r: c for c, r in rank.items()}

    picks = []  # (stratum, cid) in priority order; first-wins on dedupe

    # the 4 gated honeypots that were IN the shortlist (not the 44 global ones)
    for cid in short[short.is_honeypot_gated == True].index:
        picks.append(("gated_honeypot", cid))

    # plain-language builders: non-ai_ml shortlist w/ build-signal (surfaced by text despite title)
    nonml = short[short.title_family.isin(["swe", "data", "nontech"])]
    builder_ids = [cid for cid in nonml.index
                   if BUILD.search(str(ev.loc[cid, "evidence_text"]) if cid in ev.index else "")]
    if len(builder_ids) < 3:  # fallback: top non-ml by fused_rank
        builder_ids = list(nonml.sort_values("fused_rank").index)
    for cid in builder_ids[:3]:
        picks.append(("plainlang_builder", cid))

    # rank-range strata (from the master order)
    for r in range(1, 11):
        picks.append(("top10", inv[r]))
    for r in range(44, 53):
        picks.append(("tier_boundary", inv[r]))
    for r in [90, 94, 97, 99, 100, 101, 103, 105, 106, 109]:
        picks.append(("cut_line", inv[r]))
    for r in [312, 388, 455, 530, 610, 680, 745, 795]:
        picks.append(("deep_shortlist", inv[r]))

    # keyword-stuffers: scan raw jsonl (nontech title + >=6 AI skills). Also cache raw location
    # for every candidate we need.
    need = {cid for _, cid in picks}
    stuffers, raw_loc, raw_cache = [], {}, {}
    with open(JSONL, "rb") as f:
        for line in f:
            d = orjson.loads(line)
            cid = d["candidate_id"]
            if cid in need:
                raw_loc[cid] = d["profile"].get("location", "")
                raw_cache[cid] = d
            fam = feats.loc[cid, "title_family"] if cid in feats.index else None
            if fam == "nontech":
                nai = sum(1 for s in (d.get("skills") or []) if (s.get("name") or "").lower() in AI_SKILLS)
                if nai >= 6 and cid not in need:
                    stuffers.append((cid, nai)); raw_loc[cid] = d["profile"].get("location", "")
    stuffers.sort(key=lambda x: -x[1])
    for cid, _ in stuffers[:3]:
        picks.append(("keyword_stuffer", cid))

    # first-wins dedupe
    seen, gold = set(), []
    for stratum, cid in picks:
        if cid in seen or cid not in feats.index:
            continue
        seen.add(cid)
        f = feats.loc[cid]
        mr = rank.get(cid)
        gold.append({
            "candidate_id": cid,
            "stratum": stratum,
            "current_title": f["title"],
            "title_family": f["title_family"],
            "years_of_experience": _n(f["years_of_experience"]),
            "is_services": bool(f["is_services_company"]),
            "submitted_rank": (int(mr) if (mr is not None and mr <= 100) else None),
            "structured_facts": {
                "yoe_fit": _n(f["yoe_fit"]),
                "availability": {
                    "days_since_last_active": _n(f["days_since_last_active"]),
                    "recruiter_response_rate": _n(f["recruiter_response_rate"]),
                    "open_to_work": bool(f["open_to_work"]) if pd.notna(f["open_to_work"]) else None,
                    "notice_period_days": _n(f["notice_period_days"]),
                },
                "location": raw_loc.get(cid, ""),
                "location_preferred": bool(f["location_preferred"]),
                "country_in_india": bool(f["country_in_india"]),
            },
            "evidence_text": str(ev.loc[cid, "evidence_text"]) if cid in ev.index else "",
            "label_tier": None,   # <-- INDEPENDENT human label goes here (0-4)
        })

    with open(OUT, "w", encoding="utf-8") as fo:
        for row in gold:
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- print to stdout (evidence truncated) ----
    print(f"# gold_set.jsonl — {len(gold)} candidates. label_tier is BLANK for independent labeling.")
    print(f"# strata: {pd.Series([g['stratum'] for g in gold]).value_counts().to_dict()}\n")
    for g in gold:
        gp = dict(g)
        gp["evidence_text"] = (g["evidence_text"][:400] + " …[trunc]") if len(g["evidence_text"]) > 400 else g["evidence_text"]
        print(json.dumps(gp, ensure_ascii=False))
    print(f"\n# wrote {OUT} (FULL evidence_text inside; stdout above is truncated to ~400 chars).")


def _n(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 4)
    return v


if __name__ == "__main__":
    main()
