# Architecture Document: Blinkit Discovery Engine

## 1. System Overview

The Blinkit Discovery Engine is an end-to-end data pipeline that ingests multi-channel consumer feedback, applies deterministic cleaning filters, classifies surviving rows against strategic friction pillars, and surfaces actionable insights through a semantic search layer and interactive dashboard.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          BLINKIT DISCOVERY ENGINE                       │
│                                                                         │
│   Data Sources ──► Ingestion ──► Cleaning ──► Classification            │
│                                                   │                     │
│                                    Vectorisation ◄─┘                    │
│                                         │                               │
│                              Insight Generation                         │
│                                         │                               │
│                              Streamlit Dashboard                        │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key design principles:**

- **Dual-pathway ingestion** — live scrapers + historical/curated dumps converge into a single normalised DataFrame
- **Deterministic cleaning** — three sequential filter stages with no probabilistic heuristics; every drop is auditable
- **Pluggable classification** — offline regex rules (zero cost) or Groq Llama 3.3 70B (Groq) (higher accuracy), selectable at runtime
- **Semantic retrieval** — FAISS vector index enables natural-language querying over the cleaned corpus
- **Offline-first** — the entire pipeline runs without an API key; AI features activate only when `GROQ_API_KEY` is present

---

## 2. High-Level Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       DATA SOURCES (Dual Pathway)                        │
├──────────────────────────────────┬───────────────────────────────────────┤
│  Pathway A: Live Scrapers        │  Pathway B: Historical & Curated      │
│                                  │                                       │
│  ┌────────────────────────────┐  │  ┌─────────────────────────────────┐  │
│  │ Google Play Store          │  │  │ reviews_raw.csv       (28,255) │  │
│  │ com.grofers.customerapp    │  │  │ Ankesh-reviews_raw.csv(28,255) │  │
│  │ country=in, lang=en        │  │  ├─────────────────────────────────┤  │
│  ├────────────────────────────┤  │  │ Pre-classified Reddit CSV  (19)│  │
│  │ Apple App Store            │  │  │ Reddit Data URLs           (32)│  │
│  │ ID: 960335206              │  │  │ links-data.docx                │  │
│  ├────────────────────────────┤  │  └─────────────────────────────────┘  │
│  │ Reddit (PRAW) — manual     │  │                                       │
│  └────────────────────────────┘  │                                       │
└──────────────────────────────────┴───────────────────────────────────────┘
                                   │
                          ┌────────▼────────┐
                          │   INGESTION     │
                          │  Normalise,     │
                          │  Deduplicate,   │
                          │  Unify schema   │
                          │  56,530→28,274  │
                          └────────┬────────┘
                                   │
                ┌──────────────────▼──────────────────┐
                │      3-STAGE CLEANING PIPELINE       │
                │                                      │
                │  S1  Transactional Noise    −685     │
                │  S2  Rating / Praise        −909     │
                │  S3  Category Isolation   −24,872    │
                │                                      │
                │  28,274 ──► 1,808  (6.4% retained)   │
                └──────────────────┬──────────────────┘
                                   │
                ┌──────────────────▼──────────────────┐
                │   CLASSIFICATION & ENRICHMENT        │
                │                                      │
                │  ┌──────────────┐ ┌───────────────┐  │
                │  │ Friction     │ │ Theme Pattern │  │
                │  │ Pillar       │ │ Matching      │  │
                │  │ (4 pillars)  │ │ (10 themes)   │  │
                │  └──────────────┘ └───────────────┘  │
                └──────────────────┬──────────────────┘
                                   │
                ┌──────────────────▼──────────────────┐
                │         DISCOVERY ENGINE             │
                │                                      │
                │  ┌──────────────┐ ┌───────────────┐  │
                │  │ FAISS Vector │ │ AI Insight    │  │
                │  │ Index        │ │ Generator     │  │
                │  │ (cosine sim) │ │ (Llama 3.3 70B (Groq)) │  │
                │  └──────────────┘ └───────────────┘  │
                └──────────────────┬──────────────────┘
                                   │
                ┌──────────────────▼──────────────────┐
                │       STREAMLIT DASHBOARD            │
                │                                      │
                │  Overview │ Themes │ Questions │      │
                │  Search   │ Validation               │
                └─────────────────────────────────────┘
```

---

## 3. Component Architecture

### 3.1 Module Dependency Graph

```
config.py  ◄──────────────────────────── single source of truth
    │
    ├──► scrapers/
    │       ├── playstore.py             google-play-scraper
    │       ├── appstore.py              app-store-scraper
    │       └── reddit.py                praw
    │
    ├──► ingestion/__init__.py           pandas CSV loader
    │
    ├──► pipeline/
    │       ├── filters.py               3-stage deterministic filters
    │       └── classifier.py            offline regex │ Llama 3.3 70B (Groq)
    │
    ├──► engine/
    │       ├── __init__.py              FAISS index (faiss-cpu)
    │       ├── themes.py                regex theme detection
    │       └── insights.py              Llama 3.3 70B (Groq) structured insights
    │
    ├──► main.py                         CLI orchestrator
    ├──► scrape_reviews.py               standalone weekly scraper
    └──► app.py                          Streamlit UI
```

### 3.2 Data Flow Sequence

```
1.  main.py receives CLI args (--live, --classify, --reddit, etc.)
        │
2.  ingestion.load_all_historical()
        │   loads both CSVs + pre-classified Reddit
        │   normalises columns, deduplicates on text
        │
3.  [optional] scrapers.playstore / appstore / reddit
        │   appends live rows to the unified DataFrame
        │
4.  pipeline.filters.run_pipeline(df)
        │   Stage 1 → Stage 2 → Stage 3
        │   returns cleaned DataFrame + stage-level drop counts
        │
5.  pipeline.classifier.classify(df, mode)
        │   assigns Friction Pillar + Opportunity to each row
        │
6.  Output/blinkit_clean_data.csv  ◄── written to disk
        │
7.  [dashboard] app.py reads clean CSV
        │   engine/ builds FAISS index if API key present
        │   engine/themes.py tags themes
        │   engine/insights.py answers strategic questions
        │
8.  Streamlit serves 5 interactive tabs on localhost:8501
```

---

## 4. Data Architecture

### 4.1 Source Channel Coverage

| Channel | Pathway | Method | Volume |
|---------|---------|--------|--------|
| Google Play Store | A (live) | `google-play-scraper` | ~60/batch |
| Apple App Store | A (live) | `app-store-scraper` | ~200/run |
| Reddit | A (live, manual) | PRAW | variable |
| Play Store (historical) | B | CSV dump | 28,255 × 2 |
| App Store (historical) | B | CSV dump | included above |
| Reddit (historical) | B | CSV dump + 32 URLs | 19 pre-classified |
| YouTube | B | CSV dump | included in historical |
| HackerNews | B | CSV dump | included in historical |
| PissedConsumer | B | CSV dump | included in historical |

### 4.2 Category Taxonomy

```
                    ┌─────────────────────────┐
                    │    10 DATA CATEGORIES    │
                    └────────────┬────────────┘
            ┌────────────────────┼────────────────────┐
            │                    │                    │
   ┌────────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐
   │  PRIMARY (keep)  │  │ SECONDARY (keep)│  │  DROP (noise)  │
   │                  │  │                 │  │                │
   │  electronics     │  │  home_cleaning  │  │ groceries_fresh│
   │  personal_care_  │  │  intimate_      │  │ snacks_        │
   │    beauty        │  │    personal     │  │   beverages    │
   │  pet             │  │  pharmacy_      │  │                │
   │  baby            │  │    health       │  │                │
   └──────────────────┘  └─────────────────┘  └────────────────┘

   general — contextual inspection: keep only if text references
             expansion-category products (120+ keyword regex)
```

### 4.3 Internal DataFrame Schema (post-ingestion)

| Column | Type | Origin |
|--------|------|--------|
| `source_raw` | str | Unmapped channel name from CSV |
| `date` | str | Review timestamp |
| `rating` | float/NaN | 1–5 stars; NaN for Reddit/forums |
| `text` | str | Raw review body |
| `category` | str | Pre-assigned or inferred label |
| `url` | str | Source URL |

### 4.4 Output DataFrame Schema (post-classification)

| Column | Type | Rule |
|--------|------|------|
| `Source` | str | Mapped via `config.SOURCE_LABELS` |
| `Platform` | str | Always `"Blinkit"` |
| `Target Category` | str | Canonical category from taxonomy |
| `Friction Pillar` | str | One of 4 pillars |
| `Raw Content` | str | Cleaned verbatim text |
| `Opportunity` | str | PM hypothesis (AI) or template (offline) |

---

## 5. Cleaning Pipeline Architecture

The pipeline is implemented as three composable, stateless filter functions. Each receives a DataFrame and returns a smaller DataFrame plus a drop count.

```
                 28,274 rows
                     │
        ┌────────────▼────────────┐
        │   STAGE 1: NOISE        │
        │                         │
        │   15 blacklisted token  │
        │   phrases (regex OR)    │
        │                         │
        │   ─685 rows (2.4%)      │
        └────────────┬────────────┘
                     │
                 27,589
                     │
        ┌────────────▼────────────┐
        │   STAGE 2: RATING       │
        │                         │
        │   Drop 5★ + praise      │
        │   pattern (11 phrases)  │
        │   Keep ≤4★ or friction  │
        │   keywords in text      │
        │                         │
        │   ─909 rows (3.3%)      │
        └────────────┬────────────┘
                     │
                 26,680
                     │
        ┌────────────▼────────────┐
        │   STAGE 3: CATEGORY     │
        │                         │
        │   Drop: groceries_fresh │
        │         snacks_beverages│
        │   Keep: 7 expansion cats│
        │   general: keyword scan │
        │   (120+ product terms)  │
        │                         │
        │   ─24,872 rows (93.2%)  │
        └────────────┬────────────┘
                     │
                  1,808 rows
                  (6.4% retention)
```

**Design rationale for stage ordering:**

1. **Noise first** — cheapest filter; removes irrelevant rows before any content analysis
2. **Rating second** — pure metadata check (star value + short string match); no heavy regex
3. **Category last** — most expensive (120+ keyword regex on every `general` row); applied to the smallest surviving set

---

## 6. Classification Architecture

### 6.1 Friction Pillar Model

```
               ┌─────────────────────────┐
               │    INCOMING CLEAN ROW    │
               └────────────┬────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   --classify offline       │
              │   (default, zero-cost)     │
              │                            │
              │   Sequential regex match:  │
              │   1. Habitual Tunnel Vision │
              │   2. Quality & Authenticity │
              │   3. Discovery Blind Spots │
              │   4. Value Disconnection   │
              │   Fallback → Quality       │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   --classify ai            │
              │   (requires GROQ_API_KEY)│
              │                            │
              │   Llama 3.3 70B (Groq) structured   │
              │   JSON, temperature=0.1    │
              │   + Opportunity field      │
              │   Rate limit: 1s / 20 rows │
              └─────────────┬─────────────┘
                            │
               ┌────────────▼────────────┐
               │   Pillar + Opportunity   │
               │   appended to DataFrame  │
               └─────────────────────────┘
```

### 6.2 Theme Detection Model

10 themes detected via compiled regex patterns, independent of pillar classification:

| # | Theme | Pillar Mapping |
|---|-------|----------------|
| 1 | Counterfeit / Fake Products | Quality & Authenticity Risk |
| 2 | Warranty & Return Anxiety | Quality & Authenticity Risk |
| 3 | Dark Store Storage Concerns | Quality & Authenticity Risk |
| 4 | Price Premium vs Alternatives | Immediate Value Disconnection |
| 5 | Category Unawareness | Discovery Blind Spots |
| 6 | Search & Navigation Friction | Discovery Blind Spots |
| 7 | Grocery-Only Habit Lock | Habitual Tunnel Vision |
| 8 | Customer Support Failures | Quality & Authenticity Risk |
| 9 | Product Quality Issues | Quality & Authenticity Risk |
| 10 | Trust & Brand Verification | Quality & Authenticity Risk |

Each review is scored against all 10 patterns; primary theme = highest match count. A `Theme × Category` cross-pattern matrix exposes which themes cluster in which product categories.

---

## 7. Discovery Engine Architecture

### 7.1 Vector Search Layer

```
┌──────────────────────────────────────────────────────────┐
│                   FAISS INDEX                             │
│                                                          │
│  Embedding model:  all-MiniLM-L6-v2 (local) (384 dims)   │
│  Index type:       IndexFlatIP (cosine via L2-norm)      │
│  Corpus size:      1,808 vectors                         │
│  Persistence:      Output/faiss_index.bin                │
│                    Output/faiss_meta.pkl                  │
│                                                          │
│  Query flow:                                             │
│    "Why don't users buy electronics on Blinkit?"         │
│         │                                                │
│         ▼                                                │
│    embed(query) → 384-dim vector                       │
│         │                                                │
│         ▼                                                │
│    IndexFlatIP.search(vec, k=15)                         │
│         │                                                │
│         ▼                                                │
│    top-K reviews ranked by cosine similarity             │
└──────────────────────────────────────────────────────────┘
```

### 7.2 Insight Generation Layer

```
┌─────────────────────────────────────────────────────────────┐
│              AI INSIGHT GENERATOR                            │
│                                                             │
│  Input:   one of 8 strategic questions                      │
│           + 15–20 relevant reviews (keyword or FAISS)       │
│                                                             │
│  Model:   Llama 3.3 70B (Groq), temperature=0.3                      │
│  Format:  structured JSON                                   │
│                                                             │
│  Output:                                                    │
│    ├── executive_summary        (1 paragraph)               │
│    ├── themes[]                 (3–6, each with quotes)     │
│    ├── key_insight              (non-obvious finding)       │
│    ├── recommendations[]        (3 actionable items)        │
│    ├── confidence               (high | medium | low)       │
│    └── evidence_gaps            (what data is missing)      │
│                                                             │
│  Fallback: offline rule-based aggregation when no API key   │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 The 8 Strategic Questions

| # | Question |
|---|----------|
| 1 | Why do users repeatedly buy from the same categories? |
| 2 | What prevents users from exploring new categories? |
| 3 | How do users discover products today? |
| 4 | What role do habits play in shopping behavior? |
| 5 | What information do users need before trying a new category? |
| 6 | What frustrations emerge repeatedly? |
| 7 | Which user segments are more likely to experiment? |
| 8 | What unmet needs emerge consistently across discussions? |

---

## 8. Presentation Layer Architecture

### 8.1 Streamlit Dashboard

```
┌─────────────────────────────────────────────────────────┐
│                 app.py — Streamlit                       │
│                 localhost:8501                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Tab 1: 📊 Dashboard Overview                           │
│    • Key metrics (row count, categories, pillars)       │
│    • Category distribution bar chart                    │
│    • Pillar distribution pie chart                      │
│    • Theme frequency chart                              │
│    • Pipeline funnel table                              │
│                                                         │
│  Tab 2: 🔬 Theme Explorer                               │
│    • Expandable theme cards with verbatim quotes        │
│    • Affected categories per theme                      │
│    • Theme → Pillar mapping                             │
│    • Theme × Category cross-pattern matrix              │
│                                                         │
│  Tab 3: ❓ Strategic Questions                           │
│    • Dropdown selector (8 questions)                    │
│    • AI-generated insight with grounded themes          │
│    • Verbatim quotes as evidence                        │
│    • Actionable recommendations                         │
│                                                         │
│  Tab 4: 🔎 Search & Explore                             │
│    • Multi-filter: category, pillar, theme              │
│    • Keyword search                                     │
│    • Browse + download filtered data                    │
│                                                         │
│  Tab 5: 💬 RAG Chatbot                                  │
│    • Free-form conversational Q&A interface             │
│    • FAISS semantic retrieval + Llama 3.3 70B (Groq) generation  │
│    • Multi-turn conversation memory (last 3 turns)      │
│    • 8 suggested evaluator questions                    │
│    • Expandable evidence panels per answer              │
│    • Graceful degradation: 4 operating modes            │
│                                                         │
│  Tab 6: ✅ Validation & Methodology                     │
│    • Data sourcing documentation                        │
│    • Theme identification methodology                   │
│    • Insight generation process                         │
│    • Quality validation framework                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 8.2 RAG Chatbot Architecture

```
┌───────────────────────────────────────────────────────────┐
│                  RAG CHATBOT (engine/chatbot.py)           │
│                                                           │
│  ┌─────────────────────────────────────────────┐          │
│  │  RETRIEVAL LAYER                             │          │
│  │  ┌─────────────┐    ┌────────────────────┐   │          │
│  │  │ FAISS Search │ OR │ Keyword Overlap    │   │          │
│  │  │ (cosine sim) │    │ (offline fallback) │   │          │
│  │  └─────────────┘    └────────────────────┘   │          │
│  └───────────────────┬─────────────────────────┘          │
│                      │ top-K reviews                      │
│  ┌───────────────────▼─────────────────────────┐          │
│  │  CONTEXT ASSEMBLY                            │          │
│  │  Evidence block + corpus stats               │          │
│  │  + conversation history (last 3 turns)       │          │
│  └───────────────────┬─────────────────────────┘          │
│                      │                                    │
│  ┌───────────────────▼─────────────────────────┐          │
│  │  GENERATION LAYER                            │          │
│  │  ┌──────────────┐    ┌───────────────────┐   │          │
│  │  │ Llama 3.3 70B (Groq)  │ OR │ Offline Template  │   │          │
│  │  │ (temp=0.3)   │    │ (rule-based)      │   │          │
│  │  └──────────────┘    └───────────────────┘   │          │
│  └───────────────────┬─────────────────────────┘          │
│                      │                                    │
│                      ▼                                    │
│  Evidence-backed answer + source citations                │
│  + metadata (mode, sources, categories)                   │
└───────────────────────────────────────────────────────────┘

4 Operating Modes:
  🟢 Full RAG        = FAISS + Llama 3.3 70B (Groq)
  🟡 AI + Keyword    = keyword retrieval + Llama 3.3 70B (Groq)
  🟡 FAISS + Offline = FAISS + rule-based generation
  🟠 Offline         = keyword retrieval + rule-based
```

### 8.3 Dashboard Data Dependencies

```
app.py
  │
  ├── reads Output/blinkit_clean_data.csv
  │
  ├── engine/themes.py
  │     └── assigns themes to every row (regex, no API)
  │
  ├── engine/__init__.py
  │     └── builds/loads FAISS index (requires GROQ_API_KEY)
  │
  ├── engine/insights.py
  │     └── generates strategic answers (requires GROQ_API_KEY)
  │     └── offline fallback: rule-based pillar/category aggregation
  │
  └── engine/chatbot.py
        └── RAGChatbot class: retrieval + generation + conversation memory
        └── graceful degradation across 4 operating modes
```

---

## 9. Automation Architecture

### 9.1 GitHub Actions Weekly Scrape

```
┌─────────────────────────────────────────────────────────────┐
│            .github/workflows/weekly-scrape.yml               │
│                                                             │
│  Trigger:   cron '30 0 * * 1' (Monday 6:00 AM IST)         │
│             + manual workflow_dispatch                       │
│                                                             │
│  Steps:                                                     │
│    1. Checkout repo                                         │
│    2. pip install google-play-scraper app-store-scraper      │
│    3. python scrape_reviews.py --max-reviews 200            │
│    4. Deduplicate against Output/scrapes/all_scraped_reviews │
│    5. git add + commit + push (if new rows exist)           │
│                                                             │
│  Outputs:                                                   │
│    • Output/scrapes/all_scraped_reviews.csv (cumulative)    │
│    • GitHub Actions run summary with scrape stats           │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 CLI Orchestrator

```bash
# Historical data only, offline classifier (default)
python main.py

# Live scraping + AI classification
python main.py --live --classify ai

# App Store only
python main.py --appstore

# Full pipeline with Reddit
python main.py --live --reddit --classify ai

# Standalone scraper (for GitHub Actions)
python scrape_reviews.py --max-reviews 200
python scrape_reviews.py --playstore-only
python scrape_reviews.py --appstore-only
```

---

## 10. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3 | All components |
| **Data processing** | pandas | DataFrame manipulation, CSV I/O |
| **Play Store scraping** | google-play-scraper | Live review extraction |
| **App Store scraping** | app-store-scraper | Live review extraction |
| **Reddit scraping** | PRAW | Thread/comment extraction |
| **Embeddings** | Local all-MiniLM-L6-v2 (local) | 384-dim vectors for FAISS |
| **LLM** | Groq Llama 3.3 70B (Groq) | Classification + insight generation |
| **Vector search** | faiss-cpu | Cosine similarity search |
| **Dashboard** | Streamlit | Interactive 5-tab UI |
| **CI/CD** | GitHub Actions | Weekly automated scraping |
| **Version control** | Git + GitHub | HimanshiBhambhani/Grad-Project |

---

## 11. Security & Configuration

| Concern | Approach |
|---------|----------|
| **API keys** | Stored in `.env` (gitignored); GitHub Actions uses repo secrets |
| **Reddit credentials** | `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` in `.env` |
| **Groq key** | `GROQ_API_KEY` in `.env`; all AI features degrade gracefully without it |
| **Output data** | `Output/` CSVs are gitignored (except `scrapes/` for cumulative tracking) |
| **Rate limiting** | Classifier pauses 1s every 20 rows; scrapers respect API limits |
| **Region lock** | All scraping constrained to `country='in'`, `lang='en'` |

---

## 12. Scalability Considerations

| Dimension | Current | Path to Scale |
|-----------|---------|---------------|
| **Corpus size** | 1,808 clean rows | FAISS `IndexIVFFlat` for >100K vectors |
| **Embedding model** | all-MiniLM-L6-v2 (local) | Upgrade to all-MiniLM-L6-v2 (local)-large for higher recall |
| **Classification** | Sequential Llama 3.3 70B (Groq) calls | Batch API or fine-tuned classifier |
| **Storage** | Local CSV files | PostgreSQL / BigQuery for production |
| **Dashboard** | Single-user Streamlit | Streamlit Cloud or containerised deployment |
| **Scraping frequency** | Weekly cron | Daily with backoff; add more store regions |
| **Source coverage** | 7 channels | Add Twitter/X, Trustpilot, LinkedIn posts |
