# eval/traces.md — full reasoning chain on 5 real records

For each: **evidence_text** (what the recruiter reads) -> **precomputed facts** (fed to the judge) -> **judge verdict** -> **final submitted rank**. The judge's tier is shown here for inspection only; it is NOT used as a gold label (that would be circular).


---

## top-10 fit — `CAND_0061257`

**Final outcome:** **rank 1 / 100** (in the submission)

**Judge verdict** (for inspection only, not a label):
- fit_tier: **4**   fit_score: **0.95**   honeypot_suspicion: **False**   timeline_failed: **False**
- reasoning: Clear evidence of end-to-end ML experience at product companies; strong fit with the role requirements.
- key_evidence:
    - Designed and shipped the ranking layer for LinkedIn's flagship product
    - Owned data pipeline, evaluation framework, and operational discipline across 14 months
    - End-to-end experience in search and discovery at Yellow.ai
    - Built offline metrics that correlated with online engagement
- concerns: (none)

**Precomputed facts block (fed to the judge):**
```
years_of_experience=8.0; yoe_fit=1.0; title_family=ai_ml; current_title='Staff Machine Learning Engineer'; is_services_company=False (career-level), services_only_career=False; availability: days_since_last_active=75, recruiter_response_rate=0.87, notice_period_days=30, open_to_work=True; location_preferred=True, country_in_india=True; TIMELINE CONSISTENCY check: PASSED — tenure, experience, and proficiency are mutually consistent
timeline_status(): PASSED — tenure, experience, and proficiency are mutually consistent
```

**evidence_text (full — the primary signal, skills[] excluded):**
```
Staff Machine Learning Engineer | 8 yrs experience

Senior Engineer | 8.0+ yrs in production systems

[CURRENT ROLE] Staff Machine Learning Engineer @ LinkedIn (43 mo)
Designed the ranking layer for the company's flagship product: how do we surface the right thing at the right time, across millions of items, for millions of users. The hard problem was rarely the modeling — it was the data pipeline that fed the models, the evaluation framework that told us whether they worked, and the operational discipline of keeping all of it healthy in production. I owned all three across roughly 14 months.

Designed the ranking layer for the company's flagship product: how do we surface the right thing at the right time, across millions of items, for millions of users. The hard problem was rarely the modeling — it was the data pipeline that fed the models, the evaluation framework that told us whether they worked, and the operational discipline of keeping all of it healthy in production. I owned all three across roughly 14 months.

[PAST ROLE] Senior Applied Scientist @ Yellow.ai (52 mo)
Owned the search and discovery experience end-to-end at a consumer product, from how content is represented internally through to how the most relevant results appear for each user's intent. The work spanned data infrastructure, ranking algorithms, evaluation methodology, and direct collaboration with product/PM on what 'relevance' actually means for our users. Spent a fair amount of time on the eval side — building offline metrics that actually correlated with online engagement, which turned out to be the hardest part.

[SECONDARY — self-described summary] Senior engineer who has spent the last several years building systems that connect users with relevant information at scale. Comfortable across the full stack from infrastructure to algorithms to product experience, though most of my time has been in the middle layer — the ranking and retrieval systems that decide what to show. Strong preference for shipping real systems over research-only work; I'd rather have a working v1 in 6 weeks than a perfect v2 in 6 months. I've made the standard mistakes — over-engineering early, optimizing offline metrics that didn't move online numbers, building beautiful infrastructure for features that users didn't actually want — so I notice them faster now. Looking for senior IC or tech-lead roles where I can own the intelligence layer end-to-end at a product company.
```

---

## tier boundary (~rank 48) — `CAND_0000031`

**Final outcome:** **rank 48 / 100** (in the submission)

**Judge verdict** (for inspection only, not a label):
- fit_tier: **3**   fit_score: **0.85**   honeypot_suspicion: **False**   timeline_failed: **False**
- reasoning: Strong background in ranking systems and feature engineering across multiple roles; lacks explicit mention of embeddings or vector databases but otherwise meets the must-haves.
- key_evidence:
    - Trained and shipped multiple ranking models using XGBoost and LightGBM
    - Designed features across three families: content metadata, user behavior signals, and item engagement history
    - Owned offline-online correlation analysis to determine which metrics predicted A/B test outcomes
    - Worked closely with PMs to define optimization targets (click-through vs. dwell time vs. downstream conversion)
    - Led the migration of a keyword-search-based product to embedding-based retrieval
- concerns:
    - Repetitive description across multiple roles, suggesting potential title-chasing
    - Lack of explicit mention of production experience with embeddings or vector databases

**Precomputed facts block (fed to the judge):**
```
years_of_experience=6.0; yoe_fit=1.0; title_family=ai_ml; current_title='Recommendation Systems Engineer'; is_services_company=False (career-level), services_only_career=False; availability: days_since_last_active=37, recruiter_response_rate=0.91, notice_period_days=60, open_to_work=True; location_preferred=True, country_in_india=True; TIMELINE CONSISTENCY check: PASSED — tenure, experience, and proficiency are mutually consistent
timeline_status(): PASSED — tenure, experience, and proficiency are mutually consistent
```

**evidence_text (full — the primary signal, skills[] excluded):**
```
Recommendation Systems Engineer | 6 yrs experience

Recommendation Systems Engineer | Search, Ranking & Retrieval

[CURRENT ROLE] Recommendation Systems Engineer @ Swiggy (14 mo)
Trained and shipped multiple ranking models for our product's discovery feed using XGBoost and LightGBM. Designed features across three families: content metadata, user behavior signals, and item engagement history. Owned the offline-online correlation analysis that determined which offline metrics actually predicted A/B test outcomes. Worked closely with PMs to define the optimization target (click-through vs. dwell time vs. downstream conversion) — that work was as important as the modeling itself.

Trained and shipped multiple ranking models for our product's discovery feed using XGBoost and LightGBM. Designed features across three families: content metadata, user behavior signals, and item engagement history. Owned the offline-online correlation analysis that determined which offline metrics actually predicted A/B test outcomes. Worked closely with PMs to define the optimization target (click-through vs. dwell time vs. downstream conversion) — that work was as important as the modeling itself.

[PAST ROLE] Search Engineer @ Mad Street Den (16 mo)
Trained and shipped multiple ranking models for our product's discovery feed using XGBoost and LightGBM. Designed features across three families: content metadata, user behavior signals, and item engagement history. Owned the offline-online correlation analysis that determined which offline metrics actually predicted A/B test outcomes. Worked closely with PMs to define the optimization target (click-through vs. dwell time vs. downstream conversion) — that work was as important as the modeling itself.

[PAST ROLE] NLP Engineer @ Uber (27 mo)
Trained and shipped multiple ranking models for our product's discovery feed using XGBoost and LightGBM. Designed features across three families: content metadata, user behavior signals, and item engagement history. Owned the offline-online correlation analysis that determined which offline metrics actually predicted A/B test outcomes. Worked closely with PMs to define the optimization target (click-through vs. dwell time vs. downstream conversion) — that work was as important as the modeling itself.

[PAST ROLE] Applied ML Engineer @ Zomato (13 mo)
Owned the ranking layer for an e-commerce search product, evolving it from a hand-tuned scoring function to a learning-to-rank model over 9 months. Designed the relevance labeling pipeline (mix of click-through data and explicit human judgments), the feature pipeline, and the training/eval workflow. Most of the work was infrastructure and data quality — the modeling part was almost the easy bit. Final model improved revenue-per-search by 12%.

[SECONDARY — self-described summary] Machine learning engineer with 6.0 years of experience building ML-powered features in production. Strong background in NLP, recommendation systems, and applied AI; comfortable across the ML stack from feature engineering through deployment. Recently, I led the team that migrated our keyword-search-based product to embedding-based retrieval. I've learned that most retrieval problems are actually evaluation problems in disguise. My academic background is in CS/ML but my main learning has come from shipping real systems and seeing what holds up under production load. Open to senior IC roles in applied ML or AI engineering, ideally at product companies where I'd own a meaningful piece of the ML stack.
```

---

## gated honeypot — `CAND_0093547`

**Final outcome:** **DROPPED before ranking** (is_honeypot_gated=True — hard honeypot gate)

**Judge verdict** (for inspection only, not a label):
- fit_tier: **2**   fit_score: **0.55**   honeypot_suspicion: **False**   timeline_failed: **True**
- reasoning: Frequent job hopping and title-chasing; summed role months exceed total experience but the systems built are relevant.
- key_evidence:
    - Fine-tuned LLaMA-2-7B and Mistral-7B using LoRA/QLoRA for candidate-JD matching
    - Built data curation pipeline with eval harness using ranking metrics and human-quality scores
    - Deployed models via BentoML on Kubernetes with sub-200ms p95 latency
- concerns:
    - Summed role months far exceed the stated total years of experience (6.3 vs 2.9)
    - Title-chaser, frequent job hopping (~1 year per role)

**Precomputed facts block (fed to the judge):**
```
years_of_experience=2.9; yoe_fit=0.25; title_family=ai_ml; current_title='Senior Machine Learning Engineer'; is_services_company=False (career-level), services_only_career=False; availability: days_since_last_active=81, recruiter_response_rate=0.75, notice_period_days=60, open_to_work=False; location_preferred=False, country_in_india=True; TIMELINE CONSISTENCY check: FAILED — a single role's tenure exceeds the candidate's total years of experience; summed role months far exceed the stated total years of experience; 'expert' proficiency is claimed despite very low total experience
timeline_status(): FAILED — a single role's tenure exceeds the candidate's total years of experience; summed role months far exceed the stated total years of experience; 'expert' proficiency is claimed despite very low total experience
```

**evidence_text (full — the primary signal, skills[] excluded):**
```
Senior Machine Learning Engineer | 2.9 yrs experience

Senior Machine Learning Engineer | Building AI-native search & ranking systems

[CURRENT ROLE] Senior Machine Learning Engineer @ PhonePe (22 mo)
Owned the end-to-end ranking pipeline at a recommendations-heavy consumer product: candidate sourcing → embedding generation (using a fine-tuned BGE-large) → Pinecone retrieval → learning-to-rank re-scoring (XGBoost) → behavioral-signal integration. The hardest part wasn't the ML — it was the evaluation: building offline metrics that actually predicted what the recommendation would do to live engagement. After three iterations we landed on a calibration approach using simulated A/B tests that has held up over the last 18 months.

Owned the end-to-end ranking pipeline at a recommendations-heavy consumer product: candidate sourcing → embedding generation (using a fine-tuned BGE-large) → Pinecone retrieval → learning-to-rank re-scoring (XGBoost) → behavioral-signal integration. The hardest part wasn't the ML — it was the evaluation: building offline metrics that actually predicted what the recommendation would do to live engagement. After three iterations we landed on a calibration approach using simulated A/B tests that has held up over the last 18 months.

[PAST ROLE] Senior Machine Learning Engineer @ Sarvam AI (43 mo)
Fine-tuned LLaMA-2-7B and Mistral-7B variants using LoRA and QLoRA for domain-specific candidate-JD matching. Built the data curation pipeline that generated 200K high-quality preference pairs from recruiter labels, plus the eval harness using both ranking metrics and human-quality scores. Deployed the model via BentoML on Kubernetes with sub-200ms p95 latency by quantizing to INT8 and batching at the request level. Cost per inference dropped from $0.04 with GPT-3.5-fallback to under $0.001.

[PAST ROLE] Lead AI Engineer @ Aganitha (9 mo)
Fine-tuned LLaMA-2-7B and Mistral-7B variants using LoRA and QLoRA for domain-specific candidate-JD matching. Built the data curation pipeline that generated 200K high-quality preference pairs from recruiter labels, plus the eval harness using both ranking metrics and human-quality scores. Deployed the model via BentoML on Kubernetes with sub-200ms p95 latency by quantizing to INT8 and batching at the request level. Cost per inference dropped from $0.04 with GPT-3.5-fallback to under $0.001.

[SECONDARY — self-described summary] Senior AI engineer with 6.3 years of hands-on experience building production ML systems, with a focus on search, retrieval, and ranking. Most recently, I rebuilt the candidate-JD matching pipeline from scratch, taking it from 0.72 to 0.91 NDCG@10, handling peak QPS of 8K with sub-200ms p95. My day-to-day work spans embedding model selection and fine-tuning, hybrid retrieval architecture, learning-to-rank, behavioral-signal integration, and the offline/online evaluation that ties it all together. I've shipped systems in both early-stage product companies and at larger scale, and I've spent enough time on both that I know which tradeoffs apply where. I believe most ranking problems are solved by careful feature engineering and rigorous eval, not by bigger models. Currently exploring my next move — looking for senior IC or tech-lead roles where I can own the intelligence layer end-to-end.
```

---

## plain-language builder (non-ML title) — `CAND_0085706`

**Final outcome:** ranked 267 of 981 — **outside the top-100** (not submitted)

**Judge verdict** (for inspection only, not a label):
- fit_tier: **3**   fit_score: **0.75**   honeypot_suspicion: **False**   timeline_failed: **False**
- reasoning: Significant data engineering experience with some ML exposure; strong on infrastructure and integration but lacks a clear ML project or system.
- key_evidence:
    - Owned analytics-and-reporting service for ~3K customers
    - Integrated model-serving service into API layer at Wipro and BYJU'S
    - Built data warehouse and orchestration layer at BYJU'S
- concerns:
    - Most experience is in data engineering, with limited direct ML work
    - Self-described summary emphasizes transition to ML but lacks concrete ML projects or systems

**Precomputed facts block (fed to the judge):**
```
years_of_experience=7.7; yoe_fit=1.0; title_family=swe; current_title='Senior Software Engineer'; is_services_company=True (career-level), services_only_career=False; availability: days_since_last_active=38, recruiter_response_rate=0.68, notice_period_days=90, open_to_work=True; location_preferred=True, country_in_india=True; TIMELINE CONSISTENCY check: PASSED — tenure, experience, and proficiency are mutually consistent
timeline_status(): PASSED — tenure, experience, and proficiency are mutually consistent
```

**evidence_text (full — the primary signal, skills[] excluded):**
```
Senior Software Engineer | 7.7 yrs experience

Senior Software Engineer | 7.7+ yrs in data engineering

[CURRENT ROLE] Senior Software Engineer @ Flipkart (14 mo)
Implemented streaming data pipelines on Kafka and Spark Streaming for a real-time user-activity processing platform. Designed the schema-registry integration, the watermark/state management approach, and the deduplication logic for late-arriving events. Worked closely with the data science team to make sure feature pipelines aligned with what their models needed. Most of my career has been data engineering, with some adjacent ML exposure.

Implemented streaming data pipelines on Kafka and Spark Streaming for a real-time user-activity processing platform. Designed the schema-registry integration, the watermark/state management approach, and the deduplication logic for late-arriving events. Worked closely with the data science team to make sure feature pipelines aligned with what their models needed. Most of my career has been data engineering, with some adjacent ML exposure.

[PAST ROLE] Senior Data Engineer @ Wayne Enterprises (26 mo)
Backend development with Python (FastAPI), PostgreSQL, and Redis at a B2B SaaS product. Owned the analytics-and-reporting service which serves dashboards to ~3K paying customers. Recent work includes integrating a model-serving service (built by another team) into our API layer; my work was the integration and observability, not the model itself. Strong on API design, database performance, and reliability engineering.

[PAST ROLE] Backend Engineer @ BYJU'S (30 mo)
Backend + data hybrid role at a growth-stage startup. Built the company's first proper data warehouse (migrating from a tangled set of Postgres replicas to a clean Snowflake setup with dbt), the orchestration layer (Airflow), and the BI integration (Looker). Shipped a couple of small predictive features but the bulk of the role was data infrastructure.

[PAST ROLE] Senior Data Engineer @ Wipro (21 mo)
Backend development with Python (FastAPI), PostgreSQL, and Redis at a B2B SaaS product. Owned the analytics-and-reporting service which serves dashboards to ~3K paying customers. Recent work includes integrating a model-serving service (built by another team) into our API layer; my work was the integration and observability, not the model itself. Strong on API design, database performance, and reliability engineering.

[SECONDARY — self-described summary] Software / data professional with 7.7 years of experience building data pipelines, backend systems, and analytics infrastructure. I've been the engineer who makes ML possible by getting the data pipelines right; now I want to do more of the ML itself. My toolkit is solid on the data engineering side — Python, SQL, Spark, Airflow, warehouse design — and I've completed a couple of self-directed ML projects (Kaggle competitions, side projects fine-tuning small models). Interested in transitioning toward more AI/ML-focused work, ideally at a company where I can leverage my existing data-infra skills while learning modern ML practice.
```

---

## clear non-fit from deep shortlist — `CAND_0041696`

**Final outcome:** ranked 680 of 981 — **outside the top-100** (not submitted)

**Judge verdict** (for inspection only, not a label):
- fit_tier: **2**   fit_score: **0.55**   honeypot_suspicion: **False**   timeline_failed: **False**
- reasoning: Has applied ML experience but lacks production-scale retrieval/ranking systems; services company career without product experience.
- key_evidence:
    - Worked on churn prediction, conversion likelihood, and lifetime value estimation using scikit-learn and XGBoost
    - Built NLP pipelines for sentiment analysis and document classification using sklearn-based models and DistilBERT
    - Comfortable with Python, scikit-learn, pandas, but seeking more production ML experience
- concerns:
    - Services company career without product-company experience
    - Lightweight deployment workflows, not deep in retrieval or ranking systems

**Precomputed facts block (fed to the judge):**
```
years_of_experience=6.2; yoe_fit=1.0; title_family=ai_ml; current_title='Senior Software Engineer (ML)'; is_services_company=True (career-level), services_only_career=False; availability: days_since_last_active=128, recruiter_response_rate=0.26, notice_period_days=60, open_to_work=True; location_preferred=True, country_in_india=True; TIMELINE CONSISTENCY check: PASSED — tenure, experience, and proficiency are mutually consistent
timeline_status(): PASSED — tenure, experience, and proficiency are mutually consistent
```

**evidence_text (full — the primary signal, skills[] excluded):**
```
Senior Software Engineer (ML) | 6.2 yrs experience

Senior Software Engineer (ML) | 6.2 yrs in analytics & ML

[CURRENT ROLE] Senior Software Engineer (ML) @ Saarthi.ai (30 mo)
Worked on customer-facing predictive modeling for an e-commerce platform — churn prediction, conversion likelihood, lifetime value estimation. Used scikit-learn and XGBoost; main models were gradient-boosted trees with ~80 hand-engineered features. The work was split roughly 60/40 between modeling and data prep / SQL. The churn model is now used by the retention team, though my role was more on the modeling side than the productionization.

Worked on customer-facing predictive modeling for an e-commerce platform — churn prediction, conversion likelihood, lifetime value estimation. Used scikit-learn and XGBoost; main models were gradient-boosted trees with ~80 hand-engineered features. The work was split roughly 60/40 between modeling and data prep / SQL. The churn model is now used by the retention team, though my role was more on the modeling side than the productionization.

[PAST ROLE] AI Research Engineer @ HCL (44 mo)
Built NLP pipelines for sentiment analysis and document classification — primarily for an internal feedback-analytics dashboard. Started with sklearn-based bag-of-words models, then moved to transformer-based classifiers (DistilBERT) for the harder classes. Comfortable with PyTorch and Hugging Face but most of my training experience has been on small datasets and pre-trained model fine-tuning, not from-scratch model design.

[SECONDARY — self-described summary] Data scientist / ML engineer with 6.2 years of experience in applied machine learning. Worked across predictive modeling, NLP, analytics, and lightweight deployment workflows. I've been working on recommendation-style features but lighter on the deep-learning side — mostly classical methods like collaborative filtering and gradient-boosted models. I'm strongest at the modeling and analysis side; comfortable with Python, scikit-learn, pandas, and standard MLOps tooling, but I'm still building depth on the engineering and infra side of production ML. Looking for a role where I can step up to more end-to-end ownership of ML systems, not just modeling.
```