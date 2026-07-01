# Stage B — Full feature + embedding build, and retrieval dry-run

Status: **full-scale build done; pure-ensemble retrieval validated. No LLM judge yet, no models trained.**
Hardware: Windows + RTX 4060 Ti (8 GB), driver 581.57. **torch 2.6.0+cu124, `torch.cuda.is_available()==True`**
(verified before any embedding — we did NOT repeat the CPU-only-torch mistake).

---

## Part 0 — deps
`requirements.txt` now has a clearly-marked **"BUILD-TIME ONLY (not needed at rank time)"** section:
`torch` (CUDA cu124 wheel — install note included), `sentence-transformers`, `scikit-learn`,
`pyarrow`, `tqdm`. Rank-time deps (numpy/pandas/orjson/pyyaml) stay in the top section.

## Part 1 — summary down-weighting
`build_evidence_text` now puts **career_history[].description first** (recent-first, current role
repeated once) and appends `profile.summary` **last, behind a `[SECONDARY — self-described summary]`
marker**. Rationale: Stage-A found summaries are templated boilerplate that can contradict the title.
Re-confirmed on CAND_0000001: evidence still reads as a plausible data/ML-adjacent profile (Kafka/Spark
streaming pipelines, working with a DS team on feature pipelines, "transitioning toward AI/ML").

## Part 2 — full feature build (100k, streamed)
Streamed `candidates.jsonl` with orjson (no full-file load). **8.8 s, 11.4k rec/s.** Three artifacts,
**100,000 rows each**:
- `artifacts/features.parquet` (1.5 MB) — 18 structured features + candidate_id.
- `artifacts/honeypot_flags.parquet` (1.0 MB) — gate flags + soft flag + reason strings.
  **is_honeypot = 44 (0.044%)** over the full pool — consistent with Stage-A's 50k estimate and the spec's ~80.
- `artifacts/evidence_text.parquet` (42.3 MB) — avg 2,384 chars/candidate.

## Part 3 — ensemble embeddings
Two encoders, fp16 on CUDA, max_seq_length=512, L2-normalized, saved float16:
| artifact | shape | dtype | size | NaN rows | wall-clock |
|---|---|---|---|---|---|
| `emb_bge.npy` (BAAI/bge-large-en-v1.5) | (100000, 1024) | float16 | 204.8 MB | 0 | 1051.8 s |
| `emb_e5.npy` (intfloat/e5-large-v2) | (100000, 1024) | float16 | 204.8 MB | 0 | 1071.3 s |
| `emb_ids.npy` | (100000,) | <U13 | 1.5 MB | — | — |

Prefix conventions honored: **BGE** passages raw / query instruction `Represent this sentence for
searching relevant passages: `; **E5** passages `passage: ` / query `query: `. Total embed time ~35 min.
Both models were already in the HF cache (no download wait). GPU peaked ~4.1 GB / 8 GB at batch 64.

## Part 4 — JD queries
Two strings (POSITIVE from "what you'd actually be doing" + "absolutely need" + "ideal candidate";
NEGATIVE from "explicitly do NOT want"), saved verbatim to `artifacts/jd_queries.json` and recorded in
DECISIONS.md. Query vectors saved: `emb_bge_q.npy`, `emb_e5_q.npy` (each (2,1024) = [positive, negative]).

---

## Part 5 — Retrieval DRY-RUN (no LLM)
Per model: `score = cos(cand, positive) − 0.5·cos(cand, negative)`; the two models fused with RRF
(k=60). **No title pre-filter** (this is the recall rescue — we retrieve on text, not title).

### 1. Top-50 by fused_fit — clean
**All 50 are title_family `ai_ml`**, with no title filter applied: Senior ML/NLP/AI Engineers, Search
Engineers, Recommendation Systems Engineers, Applied/Staff ML, Senior Data Scientists. The
positive−negative formulation surfaces genuine AI-engineering profiles purely from text. Top-5:
Senior ML Engineer (8.0y), Senior ML Engineer (8.1y), Senior NLP Engineer (7.8y), Senior AI Engineer
(7.8y), Senior AI Engineer (5.9y).

### 2. Where do the rare true-title candidates land?
| current_title | n | best rank | median | ≤300 | ≤600 | ≤1000 |
|---|---|---|---|---|---|---|
| Senior AI Engineer | 4 | 4 | 18 | 4 | 4 | 4 |
| AI Engineer | 21 | 7 | 56 | 20 | **21** | 21 |
| ML Engineer | 167 | 183 | 863 | 17 | 60 | 94 |
| AI Research Engineer | 153 | 154 | 604 | 25 | 75 | 118 |
| Data Scientist | 145 | 327 | 3918 | 0 | 6 | 19 |
| **ALL rare-title** | 490 | — | — | 66 | 166 | 256 |

The "must-not-miss" tier (Senior AI Engineer, AI Engineer) is **fully inside top-600**. **Data
Scientists rank deliberately low** (median ~3918) — the negative query correctly down-weights generic
DS/analytics/CV-leaning profiles that aren't ranking/retrieval engineers; the genuine recsys-building
ones are rescued by the build-signal union. AI Research Engineers are mid-ranked (JD is explicitly
cautious about pure research). **This is desired behavior, not a recall gap.**

### 3. CAND_0000001 headline recall test — an honest, corrected finding
**Rank 19,649 / 100,000 (80th percentile) — NOT in the shortlist.** Diagnosis (actual sims):
BGE pos=0.667 / neg=0.596; E5 pos=0.801 / neg=0.777 — his positive similarity is far below true fits
(~0.83+) and his negative similarity is **elevated** because his evidence is data-pipeline/streaming
work **at a services firm (Mindtree)**, not building ranking/retrieval systems. He has **no build-signal
phrase** (he aligned feature pipelines *with* a DS team; he didn't ship an ML system).
→ **Re-assessment:** CAND_0000001 is a *data-infra engineer transitioning to ML*, i.e. a borderline
stretch candidate — NOT the canonical "plain-language strong fit." The Stage-A "should rank high" prior
was overstated. Pure retrieval's mid-pack placement is **correct and defensible**. We will let the **LLM
judge** adjudicate transitioners rather than force them up with heuristics.
→ The **true** plain-language fits (built a system, non-ML title) ARE surfaced: e.g. 42 "Senior Software
Engineer (ML)"-type candidates, captured by the build-signal union (below).

### 4. Honeypot DQ-threat — gate validated as ESSENTIAL
Pure retrieval put **4 honeypots in the top-300** (ranks 33, 51, 59, 94) — including **rank 33**, which
would land in a top-100 submission and risk the >10% DQ rule. These are the *clever* honeypots:
"Senior ML Engineer / NLP Engineer / Search Engineer" with keyword-perfect text ("expert" in
LLMs/RAG/PEFT/Milvus/Weaviate/FAISS) but only **2.7–2.9 yrs** experience and impossible career math.
**All 4 are caught by `is_honeypot`** and removed by gating. **Takeaway: semantic retrieval walks
straight into the keyword trap; the honeypot gate is non-optional.**

### 5. Build-signal UNION + shortlist size
- **484** candidates contain a concrete build-signal phrase ("built/shipped/designed a
  recommendation|ranking|search|retrieval|recsys system"); **212 of them rank > 600** and would be
  missed by a top-600-only cut — **rescued by the union.** 42 have a non-`ai_ml` title (the canonical
  plain-language fits).
- Shortlist sizing (gated = honeypots dropped):

| N (top fused) | gated top-N | + build-union | rare-true-titles ≤ N |
|---|---|---|---|
| 300 | 296 | 596 | 66 / 490 |
| 500 | 496 | 735 | 139 / 490 |
| **600** | **596** | **808** | **166 / 490** |
| 800 | 796 | 981 | 213 / 490 |
| 1000 | 996 | 1152 | 256 / 490 |

**RECOMMENDED shortlist for the judge stage: top-600 by fused_fit (honeypot-gated) ∪ build-signal
(gated) ≈ 808 candidates.** Confidence: the must-not-miss titles are fully inside top-600; the union
guarantees all 484 "built-a-system" plain-language fits (incl. wrong-title ones); honeypots excluded.
The candidates left outside are overwhelmingly generic Data Scientists / lower research profiles that
the judge would prune anyway. If judge throughput is cheap, widen to top-800 ∪ build (~981) for margin.

---

## Small refinements logged for later (not blocking)
- **title_family misses parenthetical specialties:** "Senior Software Engineer (ML)" classifies as
  `swe` (the `(ml)` suffix isn't matched). Doesn't hurt retrieval (text-based) but worth fixing for
  feature cleanliness — add `(ml)`/`(nlp)`/`(ai)` suffix detection to `_AI_ML_TOKENS`.
- `fused_fit` magnitudes are small (~0.03) because RRF sums two `1/(60+rank)` terms; that's expected —
  it's an ordinal score, used only for ranking, not as a calibrated probability.

## What's NOT done (next: judge stage)
- No LLM judge yet (this was the gate: validate retrieval recall first → **passed**).
- Next: run the offline LLM judge over the ~808-candidate shortlist (reads evidence_text, scores JD
  fit), then listwise re-rank + final honeypot gate → top-100. Open items (tier definition, behavioral
  weighting, geography hard-vs-soft, honeypot hard-drop-vs-demote) still pending in DECISIONS.md.
