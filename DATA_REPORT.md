# DATA_REPORT — Redrob "Intelligent Candidate Discovery & Ranking" Challenge

**Data-understanding pass only.** No pipeline, no models. This document maps the dataset
so we can design the architecture from facts, not assumptions.

Dataset root analyzed: `./dataset` (the brief said `./AB_H2S_Redrob/dataset`, but the working
directory **is** `AB_H2S_Redrob`, so the real path is `./dataset`).

---

## 0. Full directory tree (all levels, with sizes)

```
dataset/
├── [PUB] India_runs_data_and_ai_challenge/
│   ├── .DS_Store                                              6,148 B   (macOS junk)
│   └── India_runs_data_and_ai_challenge/                      <-- ALL REAL DATA IS HERE
│       ├── README.docx                                       10,166 B
│       ├── candidate_schema.json                              8,820 B
│       ├── candidates.jsonl                            487,259,903 B   (~465 MB, 100,000 records)
│       ├── job_description.docx                              40,225 B
│       ├── redrob_signals_doc.docx                           37,170 B
│       ├── sample_candidates.json                           300,099 B   (first 50 candidates, pretty JSON)
│       ├── sample_submission.csv                             9,247 B    (100 rows, format demo)
│       ├── submission_metadata_template.yaml                 5,211 B
│       ├── submission_spec.docx                             42,707 B
│       └── validate_submission.py                            5,036 B
└── __MACOSX/                                                            (macOS resource-fork junk — IGNORE)
    └── ... ._* sidecar files (120–212 B each)
```

**The two "top-level folders" are not two data partitions.** They are:
1. `[PUB] India_runs_data_and_ai_challenge/` — the actual challenge bundle (one nested folder of
   the same name holds every real file).
2. `__MACOSX/` — a macOS zip artifact containing only `._*` AppleDouble sidecar files. **Junk; ignore entirely.**

There is **no train/test split, no per-candidate folders, no by-role split, no jobs folder.**
Everything is a single flat directory. The "structure" is one big candidate file plus a set of
spec/doc/template files.

---

## 1. File inventory & relationships

| File | Format | Size | Records | Role |
|---|---|---|---|---|
| `candidates.jsonl` | JSON Lines (1 obj/line) | 465 MB | **100,000** | **THE dataset** — one candidate per line, fully self-contained nested JSON |
| `sample_candidates.json` | Pretty JSON array | 300 KB | 50 | Human-readable first-50 slice of the above (same schema). Convenience only. |
| `candidate_schema.json` | JSON Schema (draft-07) | 8.8 KB | — | Authoritative field contract for a candidate record |
| `job_description.docx` | Word (free text) | 40 KB | **1 JD** | The single role to rank against (see §4) |
| `redrob_signals_doc.docx` | Word (table) | 37 KB | — | Reference: the 23 behavioral signals + how traps are built |
| `submission_spec.docx` | Word | 43 KB | — | Submission rules, scoring metrics, honeypot policy, compute limits |
| `sample_submission.csv` | CSV | 9 KB | 100 | **Output template** (see §5). Format demo only — *not* a good ranking |
| `submission_metadata_template.yaml` | YAML | 5 KB | — | Team/repro metadata to submit alongside the CSV |
| `validate_submission.py` | Python | 5 KB | — | Local format validator for the submission CSV |
| `README.docx` | Word | 10 KB | — | Bundle orientation / reading order |

**How files relate:** This is **not** a relational/multi-table dataset. There are **no joins**.
`candidates.jsonl` is **one nested JSON document per entity (candidate)** — career history,
education, skills, certifications, languages, and platform signals are all **embedded arrays/objects
inside each candidate record**. The JD is a separate single document. The task is to score all
100,000 self-contained candidate documents against that one JD and emit the top 100.

> Note: README references `candidates.jsonl.gz`, `job_description.md`, `submission_spec.md`, etc.
> Our bundle ships the **uncompressed** `.jsonl` and the docs as **`.docx`** instead of `.md` —
> content is identical, only the container differs.

---

## 2. Schema per file type

There is effectively **one record schema** (the candidate), used by all 100,000 lines of
`candidates.jsonl` and the 50 records in `sample_candidates.json`. Below is the fully-expanded
structure with dtype, example, and measured % missing (None/empty) across all 100,000 records.

### 2a. Candidate — top level
| Field | Type | Example | % missing |
|---|---|---|---|
| `candidate_id` | string `^CAND_[0-9]{7}$` | `CAND_0000001` | 0% |
| `profile` | object | (see 2b) | 0% |
| `career_history` | array[obj] (1–10) | (see 2c) | 0% |
| `education` | array[obj] (0–5) | (see 2d) | 0% |
| `skills` | array[obj] | (see 2e) | 0% |
| `certifications` | array[obj] | `[]` | **75.02% empty** |
| `languages` | array[obj] | (see 2f) | 0% |
| `redrob_signals` | object (23 fields) | (see 2g) | 0% |

### 2b. `profile` — "career snapshot" (all 0% missing)
| Field | Type | Example |
|---|---|---|
| `anonymized_name` | string | `Ira Vora` |
| `headline` | string | `Backend Engineer \| SQL, Spark, Cloud` |
| `summary` | string (multi-sentence, **free text**) | `Software / data professional with 6.9 years…` |
| `location` | string `City, Region` | `Toronto` / `Pune, Maharashtra` |
| `country` | string | `Canada` |
| `years_of_experience` | number (1.0–16.9) | `6.9` |
| `current_title` | string (enum-like, 47 values) | `Backend Engineer` |
| `current_company` | string | `Mindtree` |
| `current_company_size` | enum (8 bands) | `10001+` |
| `current_industry` | string (24 values) | `IT Services` |

### 2c. `career_history[]` — **"career history" signal** (1–10 roles; median 3)
| Field | Type | Example |
|---|---|---|
| `company` | string | `Mindtree` |
| `title` | string | `Backend Engineer` |
| `start_date` | date | `2024-03-08` |
| `end_date` | date or **null** (null = current) | `null` |
| `duration_months` | int | `27` |
| `is_current` | bool | `true` |
| `industry` | string | `IT Services` |
| `company_size` | enum (8 bands) | `10001+` |
| `description` | string (**free text**, rich — the gold signal) | `Implemented streaming data pipelines on Kafka and Spark…` |

> `description` is where genuine fit lives (what they *actually built*), per the JD's explicit
> instruction. This is the field a real ranker must read, not just the skills array.

### 2d. `education[]` (0–5 items; median 1, max observed 2)
| Field | Type | Example |
|---|---|---|
| `institution` | string | `Lovely Professional University` |
| `degree` | string (8 values) | `B.E.` |
| `field_of_study` | string | `Computer Science` |
| `start_year` / `end_year` | int | `2017` / `2020` |
| `grade` | string or null | `8.24 CGPA` |
| `tier` | enum `tier_1..tier_4, unknown` | `tier_3` | **← institution prestige, NOT relevance label** |

### 2e. `skills[]` — **"skills" signal** (5–23 items; median 9)
| Field | Type | Example |
|---|---|---|
| `name` | string (133 distinct) | `Fine-tuning LLMs` |
| `proficiency` | enum `beginner/intermediate/advanced/expert` | `advanced` |
| `endorsements` | int ≥0 | `21` |
| `duration_months` | int ≥0 (months used) | `36` |

### 2f. `languages[]` (always present)
`language` (string) + `proficiency` (`basic/conversational/professional/native`).

### 2g. `certifications[]` (present in only ~25% of candidates)
`name`, `issuer`, `year`.

### 2h. `redrob_signals` — **"behavioral signals" + "platform activity"** (23 fields, all 0% missing)
| # | Field | Type / Range | Example | Notes |
|---|---|---|---|---|
| 1 | `profile_completeness_score` | 0–100 | 86.9 | |
| 2 | `signup_date` | date | 2025-10-16 | |
| 3 | `last_active_date` | date | 2026-05-20 | **recency / availability** |
| 4 | `open_to_work_flag` | bool | true | 35% true |
| 5 | `profile_views_received_30d` | int | 23 | platform activity |
| 6 | `applications_submitted_30d` | int | 2 | |
| 7 | `recruiter_response_rate` | 0–1 | 0.34 | **availability — JD says down-weight low** |
| 8 | `avg_response_time_hours` | num ≥0 | 177.8 | |
| 9 | `skill_assessment_scores` | dict[skill→0–100] | `{"NLP":38.8,…}` | **empty for 75.8%** |
| 10 | `connection_count` | int | 356 | |
| 11 | `endorsements_received` | int | 35 | |
| 12 | `notice_period_days` | 0–180 | 60 | JD prefers <30 |
| 13 | `expected_salary_range_inr_lpa` | {min,max} | {18.7, 36.1} | |
| 14 | `preferred_work_mode` | remote/hybrid/onsite/flexible | onsite | ~uniform 25% each |
| 15 | `willing_to_relocate` | bool | false | 29% true |
| 16 | `github_activity_score` | **-1**–100 | 9.2 | **-1 = no GitHub (64.6%)** |
| 17 | `search_appearance_30d` | int | 249 | platform activity |
| 18 | `saved_by_recruiters_30d` | int | 4 | |
| 19 | `interview_completion_rate` | 0–1 | 0.71 | |
| 20 | `offer_acceptance_rate` | **-1**–1 | 0.58 | **-1 = no history (59.6%)** |
| 21 | `verified_email` | bool | true | 72% true |
| 22 | `verified_phone` | bool | true | |
| 23 | `linkedin_connected` | bool | false | 36% true |

---

## 3. Keys & linkage

- **Candidate primary key:** `candidate_id` (`CAND_XXXXXXX`). **Confirmed unique** — 100,000
  distinct IDs, 0 duplicates.
- **`anonymized_name` is NOT a key** — only 3,312 distinct names across 100k rows (e.g. "Pooja Nair"
  appears 55×). Names are decorative; never use them to identify or dedupe.
- **Job description key:** there is **one JD only** (no ID field; it's a free-text `.docx`).

**Candidate↔JD mapping = ONE GLOBAL POOL ranked against ONE JD.**
Evidence:
- `submission_spec` §1: *"A CSV ranking the top 100 candidates from candidates.jsonl for the
  released job description."*
- `validate_submission.py`: requires **exactly 100 data rows**, ranks **1–100 each once**,
  `candidate_id` unique, `score` non-increasing by rank, tie-break by `candidate_id` ascending.
- `sample_submission.csv`: 100 rows of `candidate_id,rank,score,reasoning`.

There are **no pre-assigned candidate↔JD pairs, no mapping table, no folder grouping.** Every one of
the 100,000 candidates is a ranking candidate for the single JD; we emit our best 100.

---

## 4. The Job Description

- **Count:** 1. **Location in tree:** `…/job_description.docx`.
- **Format:** **Free-form prose** (not structured fields) — ~76 paragraphs, deliberately written
  as an opinionated narrative, not a checklist. It even contains an explicit "Final note for
  hackathon participants" describing the traps in the data.

**Verbatim, the full JD reads:**

> **Job Description: Senior AI Engineer — Founding Team**
> Company: Redrob AI (Series A AI-native talent intelligence platform)
> Location: Pune/Noida, India (Hybrid — flexible cadence) | Open to relocation candidates from Tier-1 Indian cities
> Employment Type: Full-time
> Experience Required: 5–9 years (see "what we mean by this" below)
>
> **Let's be honest about this role** — We're a Series A company building a new AI Engineering org
> from scratch… We need someone comfortable with two things that sound contradictory: deep technical
> depth in modern ML systems (embeddings, retrieval, ranking, LLMs, fine-tuning) **and** a scrappy
> product-engineering attitude (ship a working ranker in a week)… we'd rather you tilt toward shipper.
>
> **What you'd actually be doing** — own the intelligence layer (ranking/retrieval/matching). First
> 90 days: audit BM25+rules → ship v2 ranking (embeddings, hybrid retrieval, LLM re-ranking) → build
> eval infra (NDCG/MRR/MAP, A/B). Grow team 4→12.
>
> **"5–9 years" disqualifiers:** pure-research-only (no production) → no; "AI experience" = <12 months
> of LangChain-calling-OpenAI without prior ML production → probably no; senior who hasn't shipped
> code in 18 months ("architecture/tech-lead" only) → probably no.
>
> **Absolutely need:** production embeddings-retrieval (sentence-transformers / OpenAI emb / BGE / E5),
> vector DB / hybrid search (Pinecone/Weaviate/Qdrant/Milvus/OpenSearch/Elasticsearch/FAISS), strong
> Python, evaluation frameworks for ranking (NDCG/MRR/MAP, offline↔online).
> **Nice to have:** LoRA/QLoRA/PEFT, learning-to-rank (XGBoost/neural), HR-tech, distributed systems,
> OSS.
> **Explicitly do NOT want:** title-chasers (job-hop every 1.5y for title), framework enthusiasts
> (LangChain-tutorial GitHub), career-long pure-services people (TCS/Infosys/Wipro/Accenture/
> Cognizant/Capgemini), CV/speech/robotics-only without NLP/IR, 5+ years closed-source with no
> external validation.
>
> **Logistics:** Pune/Noida preferred (Hyderabad/Mumbai/Delhi NCR welcome; outside India case-by-case,
> no visa sponsorship). Notice: prefer <30 days, can buy out 30, 30+ raises the bar.
>
> **Ideal candidate:** ~6–8 yrs total, 4–5 in applied ML/AI at **product** companies (not services);
> shipped ≥1 end-to-end ranking/search/recommendation system at scale; strong opinions on retrieval/
> eval/LLM-integration; in or willing to relocate to Noida/Pune; active on Redrob (reachable).
> *"We're not expecting to find many matches in a 100K candidate pool… we'd rather see 10 great
> matches than 1000 maybes."*
>
> **Note for hackathon participants:** The right answer is **not** "most AI keywords in the skills
> section — that's a trap we built in." The right answer reasons about the **gap between what the JD
> says and what it means**: a candidate who built a recommendation system at a product company is a
> fit even without the buzzwords; a "Marketing Manager" with a perfect AI skill list is **not**.
> Also weigh behavioral signals — a perfect-on-paper candidate who hasn't logged in for 6 months with
> a 5% response rate is not actually available; down-weight them.

**Implication:** fit is multi-factor and partly *negative* (disqualifiers). Title/career-trajectory
and free-text role descriptions matter more than the `skills[]` array. The JD itself is the rubric.

---

## 5. Required output format

**A sample-submission / output template EXISTS:** `sample_submission.csv` (+ the spec in
`submission_spec.docx` and the enforcer `validate_submission.py`).

**Exact columns (in this order):** `candidate_id,rank,score,reasoning`

**One example row (from `sample_submission.csv`):**
```
CAND_0004989,1,0.9920,"HR Manager with 6.1 yrs; 9 AI core skills; response rate 0.76."
```

**Hard format rules (enforced by `validate_submission.py`):**
- `.csv`, UTF-8, filename = registered participant ID.
- Exactly **100 data rows** + 1 header. Header must be exactly `candidate_id,rank,score,reasoning`.
- `rank` uses each integer **1–100 exactly once**; `candidate_id` unique and matching the regex;
  every ID must exist in `candidates.jsonl`.
- `score` is **monotonically non-increasing** with rank; ties allowed but must then tie-break by
  `candidate_id` ascending.
- `reasoning` optional but **heavily weighted at manual review (Stage 4)** — must cite specific
  profile facts, connect to JD requirements, acknowledge concerns, not hallucinate, and vary per row.

> ⚠️ The shipped `sample_submission.csv` is a **deliberately naive baseline**: scores are a synthetic
> linear ramp (0.9920, 0.9840, …, −0.008/rank) and it ranks HR Managers / Graphic Designers above
> AI/ML Engineers by "AI core skill count." It is a **format demo and an anti-pattern**, not a target.

**Scoring (hidden ground truth, revealed only after close):**
`composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`, where relevance = ground-truth
**tiers** (tier 3+ = "relevant"; honeypots forced to **tier 0**). No public leaderboard, 3 submissions max.

---

## 6. Signal distributions (measured over all 100,000)

**Experience & structure**
- `years_of_experience`: min 1.0, p25 3.9, **median 6.8**, p75 9.9, p95 14.0, max 16.9 (mean 7.17).
- `career_history` length: median 3 (1–9). `skills` length: median 9 (5–23). `education`: median 1 (1–2).
- `certifications`: 75% have none.

**Titles (47 distinct) — class imbalance is the whole story:**
- 12 "generalist/non-AI" roles dominate, each ~5,500–5,800: Business Analyst (5,833), HR Manager
  (5,830), Mechanical Engineer, Accountant, Project Manager, Customer Support, Operations Manager,
  Content Writer, Sales Executive, Civil Engineer, Graphic Designer, Marketing Manager.
- Software roles each ~2,700–3,450: Software Engineer (3,450), Full Stack Developer, Cloud Engineer,
  Java/.NET Developer, DevOps, Mobile, Frontend, QA.
- Data roles in the hundreds: Data Engineer (744), Data Analyst (728), Senior Data Engineer (687).
- **True target titles are RARE:** ML Engineer 167, AI Research Engineer 153, Data Scientist 145,
  Junior ML Engineer 131, AI Specialist 130, **AI Engineer 21, Senior AI Engineer 4.** The genuine
  pool for *this* JD is a few hundred at most — exactly as the JD warns ("not many matches in 100K").

**Industry (24):** IT Services 29.9k, Software 22.4k, Manufacturing 22.3k, Conglomerate 7.6k, Paper
Products 7.5k (note: includes the "Dunder Mifflin"/"Acme" joke companies). Genuine AI-flavored
industries are tiny (AI/ML 278, Conversational AI 62, Voice AI 31, AI Services 42).

**Company size:** skews large — `10001+` 40.5%, `1001-5000` 18.2%.

**Geography:** India 75.1%, USA 10.0%, then Australia/Canada/UK/Germany/Singapore/UAE ~2.5% each.
Within India, ~18 cities each ~4,000 (Bangalore, Pune, Noida, Hyderabad, Delhi, Mumbai…). **Pune &
Noida (the JD's preferred cities) are present and well-populated.**

**Education:** tier_3 (53.2k) and tier_4 (51.9k) dominate, tier_2 27.8k, tier_1 only 6.9k. Degrees
spread evenly across 8 values incl. Ph.D (note many Ph.D.s — research signal, which the JD treats
cautiously).

**Skills — suspiciously uniform (a red flag, see §7):** 133 distinct names, each appearing **~12,000
times (±300)**. proficiency is heavily skewed: intermediate 470k, beginner 379k, advanced 110k,
**expert only 1,311 (0.14%)** — "expert" is rare and therefore meaningful (and weaponized in honeypots).

**Behavioral / platform signals:**
- `recruiter_response_rate`: median 0.44 (0.02–0.95). `interview_completion_rate`: median 0.62.
- `profile_completeness_score`: median 56.8. `open_to_work`: 35% true. `willing_to_relocate`: 29%.
- `github_activity_score`: **64.6% are −1 (no GitHub)**; among the rest, p95 ≈ 48, max 96.9.
- `offer_acceptance_rate`: **59.6% are −1 (no history)**.
- `connection_count` median 335; `profile_views_30d` median 45; `search_appearance_30d` median 105;
  `saved_by_recruiters_30d` median 7; `notice_period_days` median 90 (most candidates are 60–120,
  i.e. *above* the JD's preferred <30).
- `preferred_work_mode`: ~uniform (~25% each) — likely random, low signal.
- `expected_salary_inr_lpa`: min median 11.9, max median 19.4 LPA.

---

## 7. Data-quality flags (things that will bite us)

1. **`sample_submission.csv` is a trap-shaped baseline.** Its logic ("count AI core skills, sort by
   response rate") is precisely what the JD says is wrong. Do **not** anchor on it.
2. **Skill arrays are near-uniformly random.** Every one of 133 skills shows up ~12k times regardless
   of role — a Marketing Manager can carry "Fine-tuning LLMs". This is intentional: **keyword/skill
   matching is engineered to be noise.** Genuine signal is in `career_history[].description`, `title`
   trajectory, and `summary`.
3. **~80 honeypots, forced to tier 0; ranking them in top-10 is a strong negative; >10% honeypot rate
   in top-100 = disqualification (Stage 3).** They have *subtly impossible* profiles. Confirmed example
   `CAND_0016000`: 2.0 yrs experience but 5 skills at **"expert" with `duration_months=0`** (Photoshop,
   Hadoop, Go, Docker, TypeScript). Other honeypot patterns per spec: tenure > company age. **Our rough
   detector (≥5 expert@0-duration) found only 8** — the real ~80 are subtler and need multiple
   consistency checks (date math, tenure vs company founding, expertise vs duration).
4. **"Behavioral twins" and "plain-language Tier-5s" exist (README).** Some strong candidates describe
   real ML/recsys work in plain prose **without buzzwords** (e.g. `CAND_0000001`: title "Backend
   Engineer" but built streaming pipelines + has Milvus/LoRA/Fine-tuning-LLMs and an explicit
   "transitioning to ML" summary). A keyword filter misses them; a JD-aware reader catches them.
5. **`education.tier` ≠ relevance tier.** `tier_1..4/unknown` is *institution prestige*. The hidden
   *ground-truth relevance tier* (0–?) is a different thing and is **not in the data.** Don't conflate;
   don't treat `education.tier` as a label.
6. **Sentinel −1 values** in `github_activity_score` (64.6%) and `offer_acceptance_rate` (59.6%) mean
   "no data," not "worst." Treating −1 as a low score will systematically punish no-GitHub candidates.
7. **Heavy emptiness in two fields:** `certifications` (75% empty) and `skill_assessment_scores`
   (75.8% empty). Low coverage → weak features; don't over-rely.
8. **Duplicate/placeholder text:** anonymized names repeat (3,312 unique / 100k); joke employers
   ("Dunder Mifflin", "Acme Corp", "Initech") and joke industries ("Paper Products") appear — synthetic
   data tells, not real companies. Don't build company-prestige features off company names naively.
9. **Self-referential career rows:** same company appears as consecutive roles (e.g. CAND_0000002 has
   two "Wipro / Operations Manager" stints) — internal promotions vs data artifacts; date math needed.
10. **Encoding:** `candidates.jsonl` is clean UTF-8; the docs are `.docx` (not the `.md` the README
    names). On Windows, console is cp1252 — must force UTF-8 when printing (we did). `__MACOSX/` and
    `.DS_Store` are pure junk.
11. **NO leakage label present.** I searched every candidate field: there is **no pre-computed fit
    score, relevance tier, or match label** embedded in the records. Good — but it also means
    **this is unsupervised**: we have no training labels, only the JD-as-rubric and the hidden GT.

---

## 8. Open questions to settle before designing the pipeline

1. **Relevance definition / pseudo-labels.** With no labels and an opinionated free-text JD, how do we
   define "fit"? Do we hand-encode the JD's explicit must-haves/disqualifiers into a rubric scorer, or
   bootstrap weak labels (e.g., title∈{AI/ML Eng, applied DS} + product-company history + recsys/search
   in descriptions) and learn-to-rank? This decision drives everything.
2. **How much weight to behavioral signals vs fit?** The JD says down-weight unavailable candidates
   (stale `last_active_date`, low `recruiter_response_rate`), but a great-but-quiet candidate shouldn't
   be buried. Is it a multiplier on a fit score, a soft penalty, or a tie-breaker? Need an explicit
   policy (and how to treat −1 sentinels and notice_period).
3. **Honeypot strategy.** Special-case detector (date/tenure/expertise consistency checks) vs trust the
   ranker to naturally avoid them? Given the >10%-in-top-100 disqualifier, do we want an explicit
   tier-0 filter on the final 100 regardless of approach? What exact impossibility rules do we encode?
4. **Text understanding under the compute budget (CPU-only, ≤5 min, no network, no GPU).** `summary` +
   `career_history[].description` carry the real signal, but we can't call an LLM per candidate at
   ranking time. Precompute embeddings offline (sentence-transformers) + cheap CPU re-rank? Or
   rules/TF-IDF over descriptions? This trade-off must be decided before architecture.
5. **Geography/logistics as hard vs soft constraints.** JD prefers Pune/Noida and <30-day notice but
   says "case-by-case." Are these filters, penalties, or just rank nudges — and do we exclude non-India
   candidates (no visa sponsorship) or merely down-weight? Affects who can even reach the top 100.

---

## Appendix — three representative raw records (real texture)

**A. `CAND_0000001` — the "plain-language Tier-5" (real ML-adjacent work, non-ML title).** Title
*Backend Engineer*, 6.9 yrs, Toronto. Career: streaming pipelines on Kafka/Spark + Airflow/dbt/
Snowflake; summary explicitly says "transitioning toward AI/ML." Skills include Milvus (vector DB),
LoRA, Fine-tuning LLMs, NLP. *A keyword-only ranker under-rates this person; a JD-aware one sees a
plausible fit-with-ramp.* (Full JSON in `scratchpad/rec1.json`.)

**B. `CAND_0000002` — generalist, low fit.** *Operations Manager*, 12.5 yrs, ex-Wipro/Marketing.
Skills are a random grab-bag (React, Photoshop, Kafka, Feature Engineering, GCP) at beginner/
intermediate. No coherent ML/IR career thread. *Should rank low despite a couple of tech keywords.*

**C. `CAND_0016000` — HONEYPOT (subtly impossible).** *Full Stack Developer*, **2.0 yrs**, one job
(Initech, 24 months). Carries **five "expert" skills with `duration_months=0`** (TypeScript, Go,
Docker, Hadoop, Photoshop) — "expert at things used for zero months." Forced to tier 0; ranking it
high = disqualification signal. *This is the canonical trap shape to detect.*
```
career: [Initech / Full Stack Developer / 2024-06 → present / 24 mo / 51-200]
skills: Flask(beg,16mo), Spring Boot(beg,15mo), TypeScript(EXPERT,0mo), Go(EXPERT,1mo),
        REST APIs(beg,10mo), Docker(EXPERT,3mo), Terraform(int,12mo), Hadoop(EXPERT,2mo),
        AWS(beg,8mo), Photoshop(EXPERT,1mo)
```
