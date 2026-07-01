# Redrob Challenge — Intelligent Candidate Discovery & Ranking

Rank the top-100 candidates for one "Senior AI Engineer — Founding Team" job description out of
a 100,000-candidate pool, unsupervised, against a hidden tiered ground truth
(composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10; ~80 honeypots forced to tier 0).
Our approach: all heavy compute runs **offline** (GPU ensemble embeddings, a local
LLM-as-recruiter judge over a 985-candidate shortlist, a frozen listwise rerank of the top-40);
the **ranking step itself is a deterministic CPU-only combiner of precomputed artifacts** that
finishes in under a second. Honeypots are removed by a six-check deterministic consistency gate —
never delegated to a model, because we measured that keyword-perfect honeypots fool both dense
retrieval and a 7B judge. The full design log with evidence for every decision is
[DECISIONS.md](DECISIONS.md).

## Reproduce the submission (the ranking step — CPU, no network, seconds)

```bash
pip install numpy pandas pyarrow          # rank-time deps only (see requirements.txt)
python rank.py --candidates ./dataset/.../candidates.jsonl --out submission/ranking.csv
# (--candidates is accepted for spec-shape compatibility; the frozen path reads
#  precomputed artifacts/ and does not re-scan the pool)
python rank.py --artifacts artifacts/ --out submission/ranking.csv   # equivalent
```

Measured on this machine (Windows 11, Python 3.12, CPU only):
**wall-clock ≈ 0.5 s (0.08 s past interpreter+imports), peak working set ≈ 166 MB** — far inside
the ≤5 min / ≤16 GB budget. Output is **byte-identical** to the submitted CSV
(md5 `d21e098092d73a4a51aaac7910a4c160`); LF newlines, frozen top-40 order, fixed tie-breaks.
Validate with the bundle's checker:

```bash
python <dataset>/validate_submission.py submission/ranking.csv   # -> "Submission is valid."
```

## Full offline reproduction (the heavy precompute — GPU, hours)

Only needed to rebuild the artifacts from scratch. Place the challenge bundle at `./dataset/`.

| step | command | runtime (RTX 4060 Ti 8 GB) |
|---|---|---|
| 1. features + honeypot flags + evidence text (100k) | `python src/build_features.py` | ~10 s (CPU) |
| 2. ensemble embeddings (BGE-large + E5-large, 100k passages + JD queries) | `python src/build_embeddings.py` | ~35 min (GPU) |
| 3. shortlist (top-800 fused retrieval ∪ build-signal, 985) | `python src/build_shortlist.py` | ~1 min |
| 4. LLM judge over the shortlist (Qwen2.5-7B-Instruct q5_K_M via **local** Ollama) | `python src/run_judge.py` | ~2.2 h (GPU) |
| 5. fusion + freeze the top-40 listwise order | `python src/rank_submission.py --freeze` | ~1 min |

Steps 2/4/5 need the build/judge-time dependency sections in `requirements.txt` and a local
Ollama server — **no hosted APIs anywhere in the pipeline**. On Windows, run the src/ scripts with
`PYTHONIOENCODING=utf-8` (console tables contain non-cp1252 characters).

## How it works

1. **Ensemble retrieval** — evidence text (career descriptions first, templated summary
   down-weighted, `skills[]` excluded as engineered noise) embedded with BGE-large + E5-large;
   per-model score `cos(positive JD query) − 0.5·cos(negative JD query)`, RRF-fused. Shortlist =
   top-800 ∪ every candidate whose text contains a concrete "built/shipped a
   ranking/search/recsys system" signal (rescues plain-language builders with non-ML titles).
2. **LLM-as-recruiter judge (offline)** — Qwen2.5-7B scores all 985 shortlisted candidates
   against a rubric distilled verbatim from the JD, reading the evidence text plus a verified
   PRECOMPUTED FACTS block (yoe, services flags, availability, timeline-consistency). Strict
   JSON via pydantic; 985/985 parsed.
3. **Tiered fusion + frozen listwise rerank** — primary key = judge tier; within tier
   `(0.70·judge + 0.30·structured_fit) × availability_mult∈[0.85,1]`; the top-40 order comes
   from a windowed listwise rerank frozen at temperature 0 into
   `artifacts/frozen_rerank_order.json` (rank time never calls an LLM).
4. **Deterministic honeypot gate (first, always)** — six consistency checks (expert@0-duration,
   expert-at-low-yoe, tenure>career, career-months>experience, date errors, and inflated-yoe vs
   observed career span) hard-drop 69 gated profiles ≈ the spec's ~80. Measured: retrieval put a
   honeypot at rank 33 and the judge tier-3'd three of them — only the deterministic gate is a
   real defense. See [AUDIT_FINDINGS.md](AUDIT_FINDINGS.md) / [AUDIT_F1_NOTES.md](AUDIT_F1_NOTES.md).
5. **Independent eval** — 47-candidate hand-labeled gold set (labels from a different model than
   the judge): full pipeline composite 0.985 vs 0.806 embeddings-only / 0.819 judge-only
   ([eval/EVAL_RESULTS.md](eval/EVAL_RESULTS.md)).

## Repo structure

```
rank.py                      # THE ranking step: CPU-only frozen deterministic path (~0.5 s)
requirements.txt             # pinned deps, sectioned rank-time / build / judge / eval / packaging
DECISIONS.md                 # full design log — every decision with evidence (deck source)
DATA_REPORT.md               # dataset mapping: schema, distributions, traps
STAGE_{A,B,C,D...}_NOTES.md  # per-stage working notes (what was validated before proceeding)
AUDIT_FINDINGS.md            # end-to-end fresh-eyes audit (ranked findings)
AUDIT_F1_NOTES.md            # the second honeypot family: evidence + controlled re-freeze
src/
  features.py                # stdlib-only: evidence text, structured features, 6-check honeypot gate
  build_features.py          # 100k feature/flag/evidence build (~10 s)
  build_embeddings.py        # BGE+E5 passage & JD-query embeddings (GPU, ~35 min) — regenerates emb_*.npy
  retrieval_dryrun.py        # Stage-B retrieval recall validation
  build_shortlist.py         # top-800 fused ∪ build-signal -> 985-candidate shortlist
  judge.py / run_judge.py    # LLM-as-recruiter (local Ollama), resumable 985-candidate pass
  preview_judge.py / micro_validate.py  # judge calibration + timeline-fact micro-validation
  rank_submission.py         # full Stage-D pipeline incl. --freeze (writes frozen_rerank_order.json)
artifacts/
  features.parquet           # 18 structured features x 100k
  honeypot_flags.parquet     # 6 gating checks + soft flags x 100k (69 gated)
  evidence_text.parquet      # the text a recruiter reads (42 MB)
  shortlist.parquet          # the 985 judged candidates (features + evidence bundled)
  judgments.parquet/.jsonl   # judge verdicts (tier, score, evidence, concerns) — core rank input
  frozen_rerank_order.json   # deterministic top-40 order (temp-0), read at rank time
  jd_queries.json/jd_rubric.txt  # the JD distillations (retrieval queries + judge contract)
  emb_ids.npy, emb_*_q.npy   # embedding row ids + JD query vectors (small)
  emb_bge.npy, emb_e5.npy    # NOT in git (196 MB each) — regenerate via src/build_embeddings.py
eval/                        # independent gold set, metrics, ablations, traces (EVAL_RESULTS.md)
submission/
  ankitsingh058622_1300.csv  # the frozen submission, named per spec §2 (md5 d21e0980…)
  ankitsingh058622_1300_ranking.xlsx  # identical content, XLSX deliverable
```

## Compute note

Per spec §3/§10.3, only the **ranking step** is budget-bound (≤5 min, ≤16 GB, CPU-only, network
off) — `rank.py` satisfies it with ~3 orders of magnitude of headroom because everything
expensive (embeddings ~35 min GPU; 985 LLM judgments ~2.2 h local GPU) is precomputed offline,
which the spec explicitly permits ("pre-computation may exceed the 5-minute window"). The two
large embedding matrices are excluded from git and regenerated by `src/build_embeddings.py`;
every other artifact `rank.py` needs is shipped in `artifacts/`.
