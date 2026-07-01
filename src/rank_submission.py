"""
Stage D — produce the final top-100 submission.

Pipeline:
  PART 1  hard honeypot gate (drop is_honeypot_gated==True and any tier-0) BEFORE ranking
  PART 2  primary order by fit_tier, then within_tier_score = (0.70*judge + 0.30*structured)*avail_mult
  PART 3  listwise rerank of the top zone (tier-4) via Qwen2.5-7B, windowed, 3 passes, mean-rank
  PART 4  assemble top-100, smooth non-increasing score, recruiter-note reasoning from judge fields
  PART 5  write submission CSV (validated separately against validate_submission.py)

The rerank calls a LOCAL Ollama model (offline, GPU) — this is a build-time step. The eventual
rank.py reproduction would load the precomputed order; it stays CPU-only / no-network.
"""
from __future__ import annotations
import os, sys, json, re, time
import numpy as np
import pandas as pd
# NOTE: ollama is imported LAZILY inside rerank_window() only. The frozen deterministic
# path (frozen_rerank_order.json present) never touches it, so the rank-time environment
# does not need the judge-time ollama dependency.

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, "artifacts")
SUBDIR = os.path.join(ROOT, "submission")
PARTICIPANT_ID = "ankitsingh058622_1300"   # registered participant ID (spec §2: filename = ID.csv)
MODEL = "qwen2.5:7b-instruct-q5_K_M"

# ---- explicit fusion weights (recorded in DECISIONS.md) ----
W_JUDGE = 0.70
W_STRUCT = 0.30
FAM_SCORE = {"ai_ml": 1.0, "data": 0.5, "swe": 0.3, "nontech": 0.0}
STRUCT_W = {"yoe_fit": 0.30, "product": 0.25, "coherence": 0.20,
            "family": 0.15, "location": 0.06, "india": 0.04}
SERVICES_ONLY_PENALTY = 0.15
AVAIL_FLOOR = 0.85           # availability multiplier in [0.85, 1.0]
TOP_ZONE = 40
RERANK_PASSES = 3
WIN = 10
STEP = 5


# ---------------------------------------------------------------- PART 2 scoring
def structured_fit(r) -> float:
    yoe_fit = r.get("yoe_fit");  yoe_fit = 0.0 if yoe_fit is None else float(yoe_fit)
    pvs = r.get("product_vs_services")
    pvs = 0.0 if pvs is None else float(pvs)          # -1..1 ; 0 = neutral/no-history
    pvs01 = (pvs + 1.0) / 2.0
    coh = r.get("career_coherence");  coh = 0.0 if coh is None else float(coh)
    fam = FAM_SCORE.get(r.get("title_family"), 0.0)
    loc = 1.0 if r.get("location_preferred") else 0.0
    india = 1.0 if r.get("country_in_india") else 0.0
    s = (STRUCT_W["yoe_fit"] * yoe_fit + STRUCT_W["product"] * pvs01 +
         STRUCT_W["coherence"] * coh + STRUCT_W["family"] * fam +
         STRUCT_W["location"] * loc + STRUCT_W["india"] * india)
    if r.get("services_only_career"):
        s -= SERVICES_ONLY_PENALTY
    return float(max(0.0, min(1.0, s)))


def availability_multiplier(r) -> float:
    comps = []
    days = r.get("days_since_last_active")
    if days is not None:
        comps.append(max(0.0, min(1.0, 1.0 - max(0.0, float(days) - 30) / 170.0)))  # 30d->1, 200d->0
    rr = r.get("recruiter_response_rate")
    comps.append(0.5 if rr is None else max(0.0, min(1.0, float(rr))))
    comps.append(1.0 if r.get("open_to_work") else 0.5)
    notice = r.get("notice_period_days")
    if notice is not None:
        comps.append(max(0.0, min(1.0, 1.0 - max(0.0, float(notice) - 30) / 150.0)))  # 30d->1, 180d->0
    a = float(np.mean(comps)) if comps else 0.5
    return AVAIL_FLOOR + (1.0 - AVAIL_FLOOR) * a


# ---------------------------------------------------------------- PART 3 rerank
def _ke_short(ke_json, n=3, cap=320):
    try:
        ke = json.loads(ke_json) if isinstance(ke_json, str) else (ke_json or [])
    except Exception:
        ke = []
    s = " | ".join(str(x) for x in ke[:n])
    return s[:cap]


def rerank_window(cands, temperature=0.1):
    """cands: list of dicts {label,title,yoe,fit_score,tier,ke}. Returns ordered list of labels."""
    import ollama  # judge-time-only dep; never reached on the frozen deterministic path
    lines = []
    for c in cands:
        lines.append(f"[{c['label']}] title={c['title']} | yoe={c['yoe']} | "
                     f"judge_tier={c['tier']} judge_fit={c['fit_score']} | built: {c['ke']}")
    sys_msg = (
        "You are a senior technical recruiter ranking candidates for a Senior AI Engineer role that "
        "OWNS the ranking, retrieval, and matching systems of an AI product. Order the candidates "
        "BEST to WORST fit. Strongly favor those who actually BUILT and shipped production "
        "ranking/search/recommendation/retrieval systems at PRODUCT companies, with rigorous "
        "evaluation (NDCG/MRR/MAP, A/B), in the ~6-8 year sweet spot, and who are reachable. "
        "Penalize keyword-only depth, services-only careers, and unavailability. "
        "Return ONLY JSON: {\"order\": [labels best..worst], \"why\": {label: one short reason}}. "
        "The 'order' list must contain EXACTLY the given labels, each once, nothing else.")
    user = "Candidates:\n" + "\n".join(lines)
    labels = [c["label"] for c in cands]
    for attempt in (1, 2):
        try:
            resp = ollama.chat(model=MODEL, format="json",
                               messages=[{"role": "system", "content": sys_msg},
                                         {"role": "user", "content": user}],
                               options={"temperature": temperature, "num_ctx": 8192, "seed": 0})
            data = json.loads(resp["message"]["content"])
            order = [str(x).strip().strip("[]") for x in data.get("order", [])]
            order = [o for o in order if o in set(labels)]
            if sorted(order) == sorted(labels):
                return order
        except Exception:
            pass
        user += "\n\nReturn valid JSON with 'order' containing exactly these labels: " + ",".join(labels)
    return labels  # guard: invalid -> keep input order


def listwise_rerank(top_df, passes=RERANK_PASSES, temperature=0.1):
    seed = list(top_df.candidate_id)              # already in (tier, within_tier) order
    meta = {r.candidate_id: {"title": r.title, "yoe": r.years_of_experience,
                             "fit_score": r.fit_score, "tier": int(r.fit_tier),
                             "ke": _ke_short(r.key_evidence)}
            for r in top_df.itertuples()}
    positions = {cid: [] for cid in seed}
    for p in range(passes):
        order = list(seed)
        starts = list(range(0, max(1, len(order) - WIN + 1), STEP))
        if starts[-1] != len(order) - WIN:
            starts.append(len(order) - WIN)
        for st in starts:
            window = order[st:st + WIN]
            cands = [{"label": f"C{i}", **meta[cid]} for i, cid in enumerate(window)]
            lab2cid = {f"C{i}": cid for i, cid in enumerate(window)}
            new_labels = rerank_window(cands, temperature=temperature)
            order[st:st + WIN] = [lab2cid[l] for l in new_labels]
        for idx, cid in enumerate(order):
            positions[cid].append(idx)
        print(f"  rerank pass {p+1}/{passes} (temp={temperature}) done")
    mean_pos = {cid: float(np.mean(v)) for cid, v in positions.items()}
    seed_pos = {cid: i for i, cid in enumerate(seed)}
    final = sorted(seed, key=lambda c: (mean_pos[c], seed_pos[c], c))
    # GUARD: same set, no membership change
    assert set(final) == set(seed), "rerank changed membership"
    return final, mean_pos


FROZEN_PATH = os.path.join(ART, "frozen_rerank_order.json")


def get_rerank_order(top_df, freeze=False):
    """Reproducibility: if a frozen top-40 order exists, use it (deterministic — no LLM call).
    With freeze=True, (re)compute a single deterministic temp-0 pass and cache it. This is what
    makes the final CSV regenerate byte-identically every run."""
    seed_set = set(top_df.candidate_id)
    if freeze:
        order, _ = listwise_rerank(top_df, passes=1, temperature=0.0)
        with open(FROZEN_PATH, "w", encoding="utf-8") as f:
            json.dump({"order": order}, f, indent=2)
        print(f"         FROZE rerank order (temp 0, single pass) -> {FROZEN_PATH}")
        return order
    if os.path.exists(FROZEN_PATH):
        order = json.load(open(FROZEN_PATH, encoding="utf-8"))["order"]
        assert set(order) == seed_set, "frozen order set mismatch with current top-40 (upstream changed)"
        print(f"         using FROZEN rerank order from {FROZEN_PATH} (deterministic, no LLM)")
        return order
    order, _ = listwise_rerank(top_df, passes=RERANK_PASSES, temperature=0.1)  # legacy non-frozen path
    return order


# ---------------------------------------------------------------- PART 4 reasoning
def _loads(x):
    try:
        return json.loads(x) if isinstance(x, str) else (x or [])
    except Exception:
        return []


def _clean(s):
    return " ".join(str(s or "").split()).strip()


def make_reasoning(row, used_set):
    """Build a sharp, grounded recruiter note from the judge's extracted facts.

    Leads with the most concrete key_evidence (named company / built system / metric — these ARE
    the JD must-haves) plus a per-candidate anchor (title + yoe) and a rotated sentence frame, then
    an honest concern. The anchor + frame rotation breaks up the dataset's "behavioral twins"
    (many candidates share an identical fabricated evidence sentence) so the rows read distinct and
    human, not templated. Grounded only in the judge's extracted facts — no hallucinated detail."""
    facts = [_clean(x) for x in _loads(row.key_evidence) if _clean(x)]
    concerns = [_clean(x) for x in _loads(row.concerns) if _clean(x)]
    title = _clean(row.title)
    yoe = row.years_of_experience
    anchor = f"{title}, {yoe:g}y" if title and yoe is not None else (title or "")

    if facts:
        core = facts[0].rstrip(". ")
        for f in facts[1:]:                          # add a metric/eval-bearing 2nd clause
            f2 = f.rstrip(". ")
            if (re.search(r"\d|NDCG|MRR|MAP|A/B", f2) and f2.lower() not in core.lower()
                    and len(core) + len(f2) < 170):
                core = core + "; " + f2
                break
    else:
        core = _clean(row.reasoning).rstrip(". ")

    # rotate among 3 frames (stable per candidate) so twinned evidence yields distinct prose
    h = sum(ord(c) for c in str(row.candidate_id)) % 3
    if not anchor:
        text = core + "."
    elif h == 0:
        text = f"{anchor} — {core}."
    elif h == 1:
        text = f"{core} — {anchor}."
    else:
        text = f"{anchor}: {core}."

    if concerns and "concern" not in text.lower():
        c = concerns[0].rstrip(". ")
        cand = text.rstrip(". ") + ". Concern: " + c + "."
        if len(cand) <= 240:
            text = cand

    if len(text) > 240:
        cut = text[:240].rfind(" ")
        text = (text[:cut] if cut > 60 else text[:240]).rstrip(",;:· —") + "…"

    base = text                                      # exact-duplicate guard (rare)
    k = 2
    while text in used_set:
        suffix = f" [{row.candidate_id[-4:]}]"
        text = base[:240 - len(suffix)].rstrip(",;:· —") + suffix
        k += 1
    used_set.add(text)
    return text


# ---------------------------------------------------------------- main
def main(freeze=False):
    short = pd.read_parquet(os.path.join(ART, "shortlist.parquet"))
    judg = pd.read_parquet(os.path.join(ART, "judgments.parquet"))
    feats = pd.read_parquet(os.path.join(ART, "features.parquet"))[
        ["candidate_id", "product_vs_services", "career_coherence"]]
    df = short.merge(judg[["candidate_id", "fit_tier", "fit_score", "key_evidence", "concerns",
                           "availability_note", "reasoning", "honeypot_suspicion"]],
                     on="candidate_id", how="left")
    df = df.merge(feats, on="candidate_id", how="left")
    n0 = len(df)

    # ---- PART 1: hard honeypot gate FIRST ----
    gated = df[df.is_honeypot_gated == True]
    df = df[df.is_honeypot_gated != True].copy()
    print(f"PART 1 — honeypot gate: dropped {len(gated)} gated honeypots "
          f"{list(gated.candidate_id)}; pool {n0} -> {len(df)}")
    t0 = df[df.fit_tier == 0]
    df = df[df.fit_tier != 0].copy()
    print(f"         dropped tier-0: {len(t0)} (expected 0). "
          f"survivors with is_honeypot_gated: {int((df.is_honeypot_gated==True).sum())} (must be 0)")
    assert (df.is_honeypot_gated == True).sum() == 0

    # ---- PART 2: structured fit + availability + within-tier score ----
    df["structured_fit"] = df.apply(structured_fit, axis=1)
    df["avail_mult"] = df.apply(availability_multiplier, axis=1)
    df["fit_score"] = df["fit_score"].fillna(0.0)
    df["within_tier_score"] = ((W_JUDGE * df.fit_score + W_STRUCT * df.structured_fit)
                               * df.avail_mult)
    df = df.sort_values(["fit_tier", "within_tier_score", "candidate_id"],
                        ascending=[False, False, True]).reset_index(drop=True)
    print(f"PART 2 — ordered {len(df)} candidates by (tier, within_tier_score). "
          f"tier mix top-100: {df.head(100).fit_tier.value_counts().sort_index().to_dict()}")

    # ---- PART 3: listwise rerank of the top zone (all tier-4) ----
    top_df = df.head(TOP_ZONE).copy()
    assert top_df.fit_tier.min() == top_df.fit_tier.max() == 4, \
        f"top zone not all tier-4: {top_df.fit_tier.value_counts().to_dict()}"
    print(f"PART 3 — listwise rerank of top {TOP_ZONE} (all tier {int(top_df.fit_tier.iloc[0])}), "
          f"window {WIN}/step {STEP} {'[FREEZE: temp 0, 1 pass]' if freeze else ''} ...")
    reranked_ids = get_rerank_order(top_df, freeze=freeze)

    # ---- PART 4: assemble final order ----
    rest = df.iloc[TOP_ZONE:]                       # already in (tier, within_tier) order
    final_ids = reranked_ids + list(rest.candidate_id)
    # GUARDS
    assert len(final_ids) == len(set(final_ids)) == len(df), "final order broke membership"
    pos = {cid: i for i, cid in enumerate(final_ids)}
    tier = dict(zip(df.candidate_id, df.fit_tier))
    for a, b in zip(final_ids, final_ids[1:]):      # no lower tier above higher tier
        assert tier[a] >= tier[b], f"tier inversion: {a}(t{tier[a]}) before {b}(t{tier[b]})"
    print("         GUARDS ok: membership preserved, no tier inversion (no tier-3 above tier-4).")

    top100_ids = final_ids[:100]
    out = df.set_index("candidate_id").loc[top100_ids].reset_index()
    out["rank"] = np.arange(1, 101)
    out["score"] = (0.99 - (out["rank"] - 1) * (0.49 / 99.0)).round(4)
    # monotonic check (strictly decreasing by construction)
    assert (out["score"].diff().dropna() < 0).all(), "score not strictly decreasing"

    used = set()
    out["reasoning"] = [make_reasoning(r, used) for r in out.itertuples()]
    assert out["reasoning"].nunique() == 100, "reasoning not all-distinct"
    assert out["reasoning"].str.len().max() <= 240, "a reasoning exceeds 240 chars"

    # ---- PART 5: write CSV (LF newlines for byte-stable output across runs) ----
    os.makedirs(SUBDIR, exist_ok=True)
    path = os.path.join(SUBDIR, f"{PARTICIPANT_ID}.csv")
    out[["candidate_id", "rank", "score", "reasoning"]].to_csv(
        path, index=False, encoding="utf-8", lineterminator="\n")
    print(f"PART 5 — wrote {path}")

    # save an audit table for eyeballing
    out[["rank", "candidate_id", "title", "fit_tier", "fit_score", "structured_fit",
         "avail_mult", "within_tier_score", "score", "reasoning"]].to_parquet(
        os.path.join(ART, "submission_audit.parquet"), index=False)
    print("\n=== TOP-15 ===")
    with pd.option_context("display.max_colwidth", 90, "display.width", 260):
        print(out.head(15)[["rank", "candidate_id", "title", "fit_tier", "score",
                            "reasoning"]].to_string(index=False))
    return path


if __name__ == "__main__":
    main(freeze=("--freeze" in sys.argv))
