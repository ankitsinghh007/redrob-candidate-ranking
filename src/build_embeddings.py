"""
Stage B / Parts 3-4 — ensemble embeddings over evidence_text + JD query embeddings.
BUILD-TIME ONLY (GPU). Two encoders with their correct prefix conventions:

  BAAI/bge-large-en-v1.5 : passages = raw text;  query = instruction-prefixed.
  intfloat/e5-large-v2   : passages = "passage: ..."; query = "query: ...".

Truncate to 512 tokens, encode fp16 on CUDA, L2-normalize, save float16 .npy.

Outputs:
  artifacts/emb_bge.npy   (N, 1024) float16   passage embeddings, BGE
  artifacts/emb_e5.npy    (N, 1024) float16   passage embeddings, E5
  artifacts/emb_ids.npy   (N,)      <U13      candidate_id order (shared by both)
  artifacts/emb_bge_q.npy (2, 1024) float16   [positive, negative] JD query, BGE
  artifacts/emb_e5_q.npy  (2, 1024) float16   [positive, negative] JD query, E5
  artifacts/jd_queries.json  the two query strings
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# ---------------------------------------------------------------------------
# JD queries (Part 4) — distilled from job_description.docx.
# POSITIVE := "what you'd actually be doing" + "absolutely need" + "ideal candidate".
# NEGATIVE := the "explicitly do NOT want" list.
# ---------------------------------------------------------------------------
POSITIVE_QUERY = (
    "Senior AI Engineer who owns the intelligence layer of a product: the ranking, retrieval, "
    "and matching systems that decide what recruiters and candidates see. Production experience "
    "with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5) "
    "deployed to real users, including handling embedding drift, index refresh, and "
    "retrieval-quality regression. Production experience with vector databases or hybrid search "
    "infrastructure (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS). Strong "
    "Python and code quality. Designs rigorous evaluation frameworks for ranking systems using "
    "NDCG, MRR, MAP, offline-to-online correlation, and A/B testing. Has shipped at least one "
    "end-to-end ranking, search, or recommendation system to real users at meaningful scale, at a "
    "product company rather than pure services. Roughly 6 to 8 years total experience with 4 to 5 "
    "years in applied ML/AI at product companies. Nice to have: LLM fine-tuning (LoRA, QLoRA, "
    "PEFT), learning-to-rank models (XGBoost-based or neural), HR-tech or recruiting or marketplace "
    "products, distributed systems and large-scale inference, open-source ML contributions. Scrappy "
    "product engineer who ships working systems fast. Based in or willing to relocate to Pune or Noida."
)
NEGATIVE_QUERY = (
    "Not a fit. Title-chasers who switch companies every 1.5 years to climb from Senior to Staff to "
    "Principal. Framework enthusiasts whose GitHub is full of LangChain tutorials and hot-framework "
    "demos, whose AI experience is only recent LangChain calls to OpenAI with no pre-LLM-era ML "
    "production experience. People who have only ever worked at IT services and consulting firms "
    "(TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL, Tech Mahindra) for their entire "
    "career with no product-company experience. Pure-research candidates from academic or "
    "research-only labs with no production deployment. People whose primary expertise is computer "
    "vision, speech, or robotics without NLP or information-retrieval exposure. Senior engineers who "
    "have not written production code in the last 18 months because they moved into architecture or "
    "tech-lead roles. Keyword-stuffed profiles that list many AI skills but show no real systems "
    "built. Candidates who are unavailable, inactive, or unresponsive to recruiters."
)


def encode(model, texts, batch_size):
    emb = model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True)
    return emb.astype(np.float16)


def run_model(name, hf_id, passages, pos_q, neg_q, batch_size):
    print(f"\n=== {name} ({hf_id}) ===")
    t0 = time.time()
    model = SentenceTransformer(hf_id, device="cuda")
    model.max_seq_length = 512
    model.half()  # fp16 weights
    print(f"  loaded, max_seq_length={model.max_seq_length}, "
          f"device={model.device}, dtype=fp16")
    p_emb = encode(model, passages, batch_size)
    q_emb = encode(model, [pos_q, neg_q], batch_size=2)
    nan = int(np.isnan(p_emb.astype(np.float32)).any(axis=1).sum())
    print(f"  passages={p_emb.shape} dtype={p_emb.dtype} "
          f"({p_emb.nbytes/1e6:.1f} MB)  queries={q_emb.shape}  NaN_rows={nan}")
    print(f"  wall-clock: {time.time()-t0:.1f}s")
    del model
    torch.cuda.empty_cache()
    return p_emb, q_emb


def main():
    assert torch.cuda.is_available(), "CUDA not available — STOP (see Stage-B note)."
    print("device:", torch.cuda.get_device_name(0))

    df = pd.read_parquet(os.path.join(ART, "evidence_text.parquet"))
    ids = df["candidate_id"].to_numpy()
    raw = df["evidence_text"].fillna("").tolist()
    print(f"loaded {len(raw):,} evidence texts")

    BATCH = 64

    # BGE: passages raw, query instruction-prefixed.
    bge_pass = raw
    bge_pos = BGE_QUERY_INSTRUCTION + POSITIVE_QUERY
    bge_neg = BGE_QUERY_INSTRUCTION + NEGATIVE_QUERY
    bge_p, bge_q = run_model("BGE-large", "BAAI/bge-large-en-v1.5",
                             bge_pass, bge_pos, bge_neg, BATCH)

    # E5: passages "passage: ", queries "query: ".
    e5_pass = ["passage: " + t for t in raw]
    e5_pos = "query: " + POSITIVE_QUERY
    e5_neg = "query: " + NEGATIVE_QUERY
    e5_p, e5_q = run_model("E5-large-v2", "intfloat/e5-large-v2",
                           e5_pass, e5_pos, e5_neg, BATCH)

    np.save(os.path.join(ART, "emb_bge.npy"), bge_p)
    np.save(os.path.join(ART, "emb_e5.npy"), e5_p)
    np.save(os.path.join(ART, "emb_ids.npy"), ids)
    np.save(os.path.join(ART, "emb_bge_q.npy"), bge_q)
    np.save(os.path.join(ART, "emb_e5_q.npy"), e5_q)
    with open(os.path.join(ART, "jd_queries.json"), "w", encoding="utf-8") as f:
        json.dump({"positive": POSITIVE_QUERY, "negative": NEGATIVE_QUERY,
                   "bge_query_instruction": BGE_QUERY_INSTRUCTION}, f, indent=2)

    print("\nSAVED artifacts:")
    for nm in ("emb_bge.npy", "emb_e5.npy", "emb_ids.npy",
               "emb_bge_q.npy", "emb_e5_q.npy", "jd_queries.json"):
        mb = os.path.getsize(os.path.join(ART, nm)) / 1e6
        print(f"  {nm}: {mb:.1f} MB")


if __name__ == "__main__":
    main()
