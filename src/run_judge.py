"""
Stage C-final / Part 3 — resumable full judge run over the 985-candidate shortlist.

- Incremental checkpoint: each verdict appended to artifacts/judgments.jsonl (one JSON/line,
  keyed by candidate_id). Consolidated to artifacts/judgments.parquet at the end.
- Resumable: on startup, load already-done candidate_ids from judgments.jsonl and SKIP them.
  Safe to re-run after any interruption.
- tenacity retry wraps each call for transient Ollama errors; permanent failures are logged to
  artifacts/judge_failures.jsonl and the run continues (one bad candidate never kills the run).
- Progress every 50 candidates (done/total, elapsed, ETA). Final summary: counts, parse-success,
  failures, tier distribution, runtime.

This produces the core artifact for Stage D.
"""
from __future__ import annotations
import os, sys, json, time
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from judge import judge_candidate, load_rubric, ensure_model, timeline_status

ART = os.path.join(ROOT, "artifacts")
SHORTLIST = os.path.join(ART, "shortlist.parquet")
HP_FLAGS = os.path.join(ART, "honeypot_flags.parquet")
JUDGMENTS_JSONL = os.path.join(ART, "judgments.jsonl")
JUDGMENTS_PARQUET = os.path.join(ART, "judgments.parquet")
FAILURES_JSONL = os.path.join(ART, "judge_failures.jsonl")

FLAG_COLS = ["role_tenure_gt_career", "career_months_gt_experience", "expert_low_experience"]

# transient ollama/network errors -> retry; ValueError/ValidationError are handled inside
# judge_candidate (it returns ok=False rather than raising), so they don't trigger tenacity.
TRANSIENT = (ConnectionError, TimeoutError, OSError, Exception)


@retry(stop=stop_after_attempt(4),
       wait=wait_exponential(multiplier=1, min=2, max=20),
       retry=retry_if_exception_type(TRANSIENT), reraise=True)
def judge_with_retry(row, rubric):
    return judge_candidate(row, rubric=rubric)


def load_done():
    done = set()
    if os.path.exists(JUDGMENTS_JSONL):
        for line in open(JUDGMENTS_JSONL, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["candidate_id"])
            except Exception:
                pass
    return done


def main():
    ensure_model()
    rubric = load_rubric()

    short = pd.read_parquet(SHORTLIST)
    hp = pd.read_parquet(HP_FLAGS).set_index("candidate_id")[FLAG_COLS]
    short = short.merge(hp, left_on="candidate_id", right_index=True, how="left")
    total = len(short)

    done = load_done()
    todo = short[~short.candidate_id.isin(done)].reset_index(drop=True)
    limit = int(os.environ.get("LIMIT", "0"))
    if limit > 0:
        todo = todo.head(limit)
        print(f"[LIMIT={limit} — smoke test subset]")
    print(f"shortlist={total}  already_done={len(done)}  to_judge={len(todo)}")

    jf = open(JUDGMENTS_JSONL, "a", encoding="utf-8")
    ff = open(FAILURES_JSONL, "a", encoding="utf-8")

    t0 = time.time()
    ok_parse = 0
    failures = []
    n = 0
    for _, r in todo.iterrows():
        row = r.to_dict()
        cid = row["candidate_id"]
        try:
            verdict, meta = judge_with_retry(row, rubric)
        except Exception as e:  # permanent transient-error failure after retries
            failures.append(cid)
            ff.write(json.dumps({"candidate_id": cid, "stage": "call",
                                 "error": str(e)[:300]}) + "\n"); ff.flush()
            n += 1
            continue

        if meta["ok"]:
            ok_parse += 1
            rec = {
                "candidate_id": cid,
                "fit_tier": verdict.fit_tier,
                "fit_score": verdict.fit_score,
                "key_evidence": json.dumps(verdict.key_evidence, ensure_ascii=False),
                "concerns": json.dumps(verdict.concerns, ensure_ascii=False),
                "availability_note": verdict.availability_note,
                "honeypot_suspicion": verdict.honeypot_suspicion,
                "honeypot_reason": verdict.honeypot_reason,
                "reasoning": verdict.reasoning,
                "is_honeypot_gated": bool(row.get("is_honeypot_gated", False)),
                "timeline_failed": timeline_status(row).startswith("FAILED"),
                "inclusion_reason": row.get("inclusion_reason"),
                "latency_s": round(meta["latency_s"], 2),
                "attempts": meta["attempts"],
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n"); jf.flush()
        else:  # parse failure after judge's own stricter retry -> logged failure, continue
            failures.append(cid)
            ff.write(json.dumps({"candidate_id": cid, "stage": "parse",
                                 "error": meta.get("error", ""),
                                 "raw": meta.get("raw", "")[:300]}) + "\n"); ff.flush()

        n += 1
        if n % 50 == 0:
            el = time.time() - t0
            rate = n / el
            eta = (len(todo) - n) / rate / 60
            print(f"  {len(done)+n}/{total} done | elapsed {el/60:.1f}m | "
                  f"{rate:.2f} cand/s | ETA {eta:.1f}m | parse_ok {ok_parse}/{n} | fails {len(failures)}")

    jf.close(); ff.close()

    # ---- consolidate -> parquet ----
    recs = [json.loads(l) for l in open(JUDGMENTS_JSONL, encoding="utf-8") if l.strip()]
    dfj = pd.DataFrame(recs).drop_duplicates("candidate_id", keep="last")
    dfj.to_parquet(JUDGMENTS_PARQUET, index=False)

    el = time.time() - t0
    judged_ids = set(dfj.candidate_id)
    all_ids = set(short.candidate_id)
    missing = all_ids - judged_ids - set(failures)
    print("\n========== FULL JUDGE RUN COMPLETE ==========")
    print(f"runtime this session: {el/60:.1f} min")
    print(f"total shortlist: {total}")
    print(f"judged (in judgments.parquet): {len(dfj)}")
    print(f"parse-success this session: {ok_parse}/{n if n else 1} "
          f"({100*ok_parse/max(n,1):.1f}%)")
    print(f"failures this session: {len(failures)} -> {failures[:10]}{'...' if len(failures)>10 else ''}")
    print(f"coverage: {len(dfj)}/{total} judged + {len(set(failures))} failed; "
          f"unaccounted: {len(missing)} {sorted(missing)[:10]}")
    tier_dist = dfj.fit_tier.value_counts().sort_index().to_dict()
    print(f"tier distribution (0..4): {tier_dist}")
    print(f"honeypot_suspicion=True: {int(dfj.honeypot_suspicion.sum())} "
          f"(of which gated honeypots: {int((dfj.honeypot_suspicion & dfj.is_honeypot_gated).sum())})")
    print(f"gated honeypots in shortlist: {int(dfj.is_honeypot_gated.sum())}")
    if missing:
        print("\nWARNING: some candidates are neither judged nor failed — re-run to complete.")
    else:
        print("\nOK: every shortlisted candidate has a judgment row or a logged failure.")


if __name__ == "__main__":
    main()
