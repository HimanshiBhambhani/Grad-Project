# Implementation Document: Blinkit Discovery Engine

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Project Structure](#2-project-structure)
3. [Data Ingestion Layer](#3-data-ingestion-layer)
4. [Cleaning Pipeline](#4-cleaning-pipeline)
5. [Classification Engine](#5-classification-engine)
6. [Theme Identification](#6-theme-identification)
7. [Discovery Engine (Semantic Search + Insights)](#7-discovery-engine)
8. [RAG Chatbot](#8-rag-chatbot)
9. [Streamlit Dashboard](#9-streamlit-dashboard)
10. [Automated Scraping (GitHub Actions)](#10-automated-scraping)
11. [Pipeline Results & Statistics](#11-pipeline-results--statistics)
12. [How to Run](#12-how-to-run)
13. [Validation Framework](#13-validation-framework)

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                     DATA SOURCES (Dual Pathway)                      │
├──────────────────────────────┬───────────────────────────────────────┤
│  Pathway A: Live Scrapers    │  Pathway B: Historical + Curated      │
│  ┌─────────────────────┐     │  ┌────────────────────────────────┐   │
│  │ Play Store Scraper   │     │  │ reviews_raw.csv (28,255 rows)  │   │
│  │ (google-play-scraper)│     │  │ Ankesh-reviews_raw.csv (28,255)│   │
│  ├─────────────────────┤     │  ├────────────────────────────────┤   │
│  │ App Store Scraper    │     │  │ Pre-classified Reddit CSV (19) │   │
│  │ (app-store-scraper)  │     │  │ Reddit Data URLs (32 URLs)     │   │
│  ├─────────────────────┤     │  │ links-data.docx                │   │
│  │ Reddit Scraper       │     │  └────────────────────────────────┘   │
│  │ (praw) — manual only │     │                                       │
│  └─────────────────────┘     │                                       │
└──────────────────────────────┴───────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    3-STAGE CLEANING PIPELINE                         │
│  Stage 1: Transactional Noise Filter (blacklisted logistics tokens)  │
│  Stage 2: Rating Filter (drop 5-star praise, keep ≤4 + friction)    │
│  Stage 3: Category Isolation (drop grocery, keep expansion cats)     │
│  Result: 28,274 raw → 1,808 clean (6.4% retention)                  │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    CLASSIFICATION & ENRICHMENT                       │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐    │
│  │ Friction Pillar       │    │ Theme Pattern Matching            │    │
│  │ (4 pillars — offline  │    │ (10 themes — regex keyword        │    │
│  │  or GPT-4o-mini)      │    │  detection with confidence)       │    │
│  └──────────────────────┘    └──────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    DISCOVERY ENGINE                                   │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐    │
│  │ FAISS Vector Index    │    │ AI Insight Generator              │    │
│  │ (text-embedding-3-    │    │ (GPT-4o-mini structured JSON      │    │
│  │  small, cosine sim)   │    │  + offline rule-based fallback)   │    │
│  └──────────────────────┘    └──────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD (app.py)                       │
│  Tab 1: Dashboard Overview     │  Tab 4: Search & Explore            │
│  Tab 2: Theme Explorer         │  Tab 5: Validation & Methodology    │
│  Tab 3: Strategic Questions    │                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Project Structure

```
Data-Extraction/
├── .github/
│   └── workflows/
│       └── weekly-scrape.yml       # GitHub Actions: weekly Play Store + App Store scrape
├── Data/
│   ├── Ankesh-reviews_raw.csv      # 28,255 rows — Claude-driven multi-channel dump
│   ├── reviews_raw.csv             # 28,255 rows — Perplexity-driven multi-channel dump
│   ├── Reddit Data URLs            # 32 curated Reddit thread URLs (4 categories)
│   ├── links-data.docx             # Additional Reddit URLs in DOCX format
│   └── SourceThreadTitleSite-...csv # 19 pre-classified Reddit threads
├── Docs/
│   ├── context.md                  # Project context & specification
│   ├── problemstatement.txt        # Original problem statement
│   └── implementation.md           # This file
├── Output/
│   ├── blinkit_clean_data.csv      # Final cleaned output (1,808 rows)
│   └── scrapes/                    # Weekly scrape accumulation directory
├── scrapers/
│   ├── playstore.py                # Pathway A: Google Play Store scraper
│   ├── appstore.py                 # Pathway A: Apple App Store scraper
│   └── reddit.py                   # Pathway B: Reddit thread scraper (PRAW)
├── ingestion/
│   └── __init__.py                 # Historical CSV loader + normalizer
├── pipeline/
│   ├── filters.py                  # 3-stage cleaning pipeline
│   └── classifier.py              # Friction pillar classifier (AI + offline)
├── engine/
│   ├── __init__.py                 # FAISS vector index (build, load, search)
│   ├── insights.py                 # AI insight generator for strategic questions
│  ├── themes.py                   # Theme identification via keyword patterns
│  └── chatbot.py                  # RAG chatbot for evaluator Q&A
├── config.py                       # Central configuration (paths, rules, taxonomy)
├── main.py                         # Full pipeline orchestrator (CLI)
├── scrape_reviews.py               # Standalone scraper for GitHub Actions
├── app.py                          # Streamlit discovery dashboard
└── requirements.txt                # Python dependencies
```

---

## 3. Data Ingestion Layer

### 3.1 Data Sources Inventory

| Source | File / Method | Rows | Channels Covered |
|--------|---------------|------|------------------|
| Historical Dump 1 | `reviews_raw.csv` | 28,255 | Play Store, App Store, Reddit, YouTube, HackerNews, PissedConsumer |
| Historical Dump 2 | `Ankesh-reviews_raw.csv` | 28,255 | Same as above (Claude-driven extraction) |
| Pre-classified Reddit | `SourceThread...csv` | 19 | Reddit threads with expert labels |
| Reddit URLs | `Reddit Data URLs` | 32 URLs | r/FuckBlinkit, r/IndianBeautyTalks, r/headphonesindia, etc. |
| Live Play Store | `scrapers/playstore.py` | ~60/batch | Google Play Store API |
| Live App Store | `scrapers/appstore.py` | ~200/run | Apple App Store (ID: 960335206) |

### 3.2 Raw CSV Schema

Both `reviews_raw.csv` and `Ankesh-reviews_raw.csv` share this schema:

| Column | Description |
|--------|-------------|
| (index) | Row number |
| (unnamed) | Source channel: `Play Store`, `App Store`, `Reddit (post)`, `Reddit (comment)`, `HackerNews`, `YouTube`, `PissedConsumer` |
| `date` | Review date |
| `rating` | Star rating (1-5, or empty for Reddit/forums) |
| `text` | Review text content |
| `category` | Pre-assigned category label |
| `url` | Source URL |

### 3.3 Category Distribution in Raw Data

| Category | Count | Kept/Dropped |
|----------|-------|-------------|
| `general` | 26,202 | Contextual (keep if non-grocery content) |
| `groceries_fresh` | 779 | **Dropped** |
| `snacks_beverages` | 517 | **Dropped** |
| `electronics` | 378 | **Kept** |
| `personal_care_beauty` | 175 | **Kept** |
| `pharmacy_health` | 135 | **Kept** |
| `intimate_personal` | 20 | **Kept** |
| `home_cleaning` | 20 | **Kept** |
| `pet` | 15 | **Kept** |
| `baby` | 14 | **Kept** |

### 3.4 Ingestion Implementation

**File:** `ingestion/__init__.py`

- `load_reviews_csv(filepath)` — Loads a raw CSV, normalizes column names (the unnamed `''` column → `source_raw`), drops the numeric index column
- `load_preclassified_csv()` — Loads the Reddit pre-classified CSV, maps alias labels to canonical taxonomy (e.g., `Electronics/Accessories` → `electronics`), expands multi-category rows
- `load_all_historical()` — Combines all sources, deduplicates on text content (exact match), producing a unified DataFrame
- Deduplication result: 56,530 → 28,274 rows (the two CSV dumps overlap almost entirely)

---

## 4. Cleaning Pipeline

**File:** `pipeline/filters.py`

### 4.1 Stage 1: Transactional Noise Filter

**Purpose:** Remove reviews about delivery logistics, payment failures, driver behavior — content irrelevant to shopping habits.

**Method:** Regex pattern matching against 15 blacklisted token phrases:
```
"late delivery", "delivery boy rude", "refund pending", "money deducted",
"payment failed", "upi failed", "gpay", "app crash", "coupon code invalid",
"driver behavior", "otp not received", "delivery partner", "delivery boy",
"wrong item delivered", "order cancelled automatically"
```

**Result:** 28,274 → 27,589 (dropped 685 rows, 2.4%)

### 4.2 Stage 2: Low-Star / High-Friction Focus

**Purpose:** Isolate negative/friction-laden feedback. Praise reviews offer no actionable insight for cross-shopping barriers.

**Method:**
- Drop all 5-star reviews that match praise patterns (`"good app"`, `"best app"`, `"amazing app"`, etc. — 11 patterns)
- Exception: Keep 5-star reviews that contain friction language (`"but"`, `"however"`, `"issue"`, `"fake"`, `"expensive"`, `"didn't know"`, etc.)
- Keep all reviews rated ≤ 4 stars
- For unrated reviews (Reddit, forums): drop if pure praise, keep otherwise

**Result:** 27,589 → 26,680 (dropped 909 rows, 3.3%)

### 4.3 Stage 3: Vertical Category Isolation

**Purpose:** Remove grocery staple complaints, keep only expansion-category content.

**Method:**
- **Drop** rows with `category` = `groceries_fresh` or `snacks_beverages`
- **Keep** rows with `category` in the 7 expansion categories
- **Contextual inspection** for `general` / unknown categories: keep only if text matches expansion-category keyword patterns (120+ product keywords compiled into a regex)
- Additional grocery noise check: drop rows mentioning grocery-specific items even if categorized differently

**Expansion keyword regex covers:** electronics terms (earbuds, charger, bluetooth, smartwatch...), beauty terms (serum, moisturizer, makeup, shampoo...), pet terms (dog, cat, kibble...), baby terms (diaper, formula, stroller...), health terms (medicine, vitamin, supplement...), cleaning terms (detergent, dishwash...), intimate terms (sanitary, tampon...)

**Result:** 26,680 → 1,808 (dropped 24,872 rows — the largest filter, since `general` = 26,202 rows and most don't reference expansion products)

### 4.4 Pipeline Funnel Summary

| Stage | Input | Output | Dropped | Drop % |
|-------|-------|--------|---------|--------|
| Raw input | 56,530 | — | — | — |
| Deduplication | 56,530 | 28,274 | 28,256 | 50.0% |
| Empty text removal | 28,274 | 28,274 | 0 | 0% |
| Stage 1: Noise | 28,274 | 27,589 | 685 | 2.4% |
| Stage 2: Rating | 27,589 | 26,680 | 909 | 3.3% |
| Stage 3: Category | 26,680 | **1,808** | 24,872 | 93.2% |

**Overall retention rate:** 6.4% (1,808 / 28,274)

---

## 5. Classification Engine

**File:** `pipeline/classifier.py`

### 5.1 Friction Pillar Assignment

Each surviving row is assigned to exactly one of **4 friction pillars**:

| Pillar | Offline Detection Keywords |
|--------|---------------------------|
| **Habitual Tunnel Vision** | "only use for grocery", "never knew", "routine order", "habit", "stuck with grocery" |
| **Quality & Authenticity Risk** | "fake", "counterfeit", "damaged", "warranty", "defective", "tampered", "not genuine" |
| **Discovery Blind Spots** | "didn't know", "had no idea", "can't find", "not visible", "hidden", "not aware" |
| **Immediate Value Disconnection** | "expensive", "overpriced", "cheaper on", "amazon has", "nykaa", "price", "pack size" |

### 5.2 Two Classification Modes

**Offline (default — `--classify offline`):**
- Rule-based regex pattern matching
- Priority ordering: checks pillars sequentially, assigns first match
- Falls back to "Quality & Authenticity Risk" if no pattern matches
- Runs instantly, no API cost

**AI-powered (`--classify ai`):**
- OpenAI GPT-4o-mini with structured JSON output
- System prompt defines all 4 pillars with detailed descriptions
- Also generates `Target Category` validation and `Opportunity` hypothesis
- Rate-limited: 1-second pause every 20 rows
- Requires `OPENAI_API_KEY` in `.env`

### 5.3 Output Schema

| Column | Source |
|--------|--------|
| **Source** | Mapped from raw source labels via `config.SOURCE_LABELS` |
| **Platform** | Always `"Blinkit"` |
| **Target Category** | From data `category` column (canonical label) |
| **Friction Pillar** | Classifier output (exactly one of 4) |
| **Raw Content** | Cleaned review text |
| **Opportunity** | PM hypothesis (AI mode) or template (offline mode) |

---

## 6. Theme Identification

**File:** `engine/themes.py`

### 6.1 Theme Definitions

10 themes detected via compiled regex keyword patterns:

| Theme | Example Keywords | Maps To Pillar |
|-------|-----------------|----------------|
| Counterfeit / Fake Products | fake, counterfeit, duplicate, not original | Quality & Authenticity Risk |
| Warranty & Return Anxiety | warranty, return, non-returnable, refund denied | Quality & Authenticity Risk |
| Dark Store Storage Concerns | storage, dark store, heat, expired, tampered | Quality & Authenticity Risk |
| Price Premium vs Alternatives | expensive, overpriced, Amazon, Nykaa, Flipkart | Immediate Value Disconnection |
| Category Unawareness | didn't know, had no idea, never knew, not aware | Discovery Blind Spots |
| Search & Navigation Friction | can't find, search, hidden, not visible | Discovery Blind Spots |
| Grocery-Only Habit Lock | only use for grocery, routine, habit, always buy | Habitual Tunnel Vision |
| Customer Support Failures | support, customer care, bot, no response | Quality & Authenticity Risk |
| Product Quality Issues | quality, defective, broken, not working, faulty | Quality & Authenticity Risk |
| Trust & Brand Verification | trust, authentic, verify, legit, serial number | Quality & Authenticity Risk |

### 6.2 Implementation

- Each review scored against all 10 theme patterns
- Primary theme = highest match count
- `Theme Confidence` = number of keyword matches (higher = stronger signal)
- Cross-pattern analysis: `Theme × Category` matrix shows which themes cluster in which product categories

---

## 7. Discovery Engine

### 7.1 FAISS Vector Index

**File:** `engine/__init__.py`

- Embeds all 1,808 reviews using `text-embedding-3-small` (1,536 dimensions)
- Builds `IndexFlatIP` (inner product after L2 normalization = cosine similarity)
- Persists to `Output/faiss_index.bin` + `Output/faiss_meta.pkl`
- Semantic search: natural language query → top-K most relevant reviews

### 7.2 AI Insight Generator

**File:** `engine/insights.py`

Answers the 8 strategic questions:

1. Why do users repeatedly buy from the same categories?
2. What prevents users from exploring new categories?
3. How do users discover products today?
4. What role do habits play in shopping behavior?
5. What information do users need before trying a new category?
6. What frustrations emerge repeatedly?
7. Which user segments are more likely to experiment?
8. What unmet needs emerge consistently across discussions?

**Process per question:**
1. Retrieve 15-20 most relevant reviews (keyword relevance or FAISS semantic search)
2. Feed evidence to GPT-4o-mini with structured system prompt
3. Output: executive summary, 3-6 grounded themes with verbatim quotes, key non-obvious insight, 3 actionable recommendations, confidence level, evidence gaps

**Offline fallback:** Rule-based aggregation of pillar/category distributions with template-based recommendations.

---

## 8. RAG Chatbot

**File:** `engine/chatbot.py`

### 8.1 Purpose

An interactive conversational interface designed for evaluator Q&A sessions. The chatbot answers free-form questions about Blinkit's cross-shopping inertia problem using evidence retrieved from the cleaned review corpus.

### 8.2 Architecture

```
    Evaluator Question
          │
          ▼
    ┌──────────────┐       ┌──────────────────┐
    │  Retrieval    │──────►│ FAISS Semantic   │  (if index built)
    │  Layer        │       │ Search (top-K)   │
    │               │──────►│ Keyword Overlap  │  (offline fallback)
    └──────┬───────┘       └──────────────────┘
           │
           ▼
    ┌──────────────┐
    │  Context      │  Evidence block + corpus stats
    │  Assembly     │  + conversation history (last 3 turns)
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐       ┌──────────────────┐
    │  Generation   │──────►│ GPT-4o-mini      │  (if API key present)
    │  Layer        │──────►│ Offline Template │  (rule-based fallback)
    └──────┬───────┘       └──────────────────┘
           │
           ▼
    Evidence-Backed Answer
    + Source citations
    + Metadata (mode, sources, categories)
```

### 8.3 Operating Modes

The chatbot degrades gracefully depending on available infrastructure:

| Mode | FAISS Index | OpenAI Key | Retrieval | Generation |
|------|-------------|------------|-----------|------------|
| **Full RAG** | ✅ | ✅ | Semantic (cosine similarity) | GPT-4o-mini |
| **AI + Keyword** | ❌ | ✅ | Keyword overlap scoring | GPT-4o-mini |
| **FAISS + Offline** | ✅ | ❌ | Semantic search | Rule-based template |
| **Offline** | ❌ | ❌ | Keyword overlap scoring | Rule-based template |

### 8.4 Key Features

- **Conversational memory:** Maintains last 3 Q&A pairs for multi-turn context
- **Evidence grounding:** Every answer cites verbatim quotes with source, category, and pillar attribution
- **Suggested questions:** 8 pre-built evaluator-style questions for quick access
- **Expandable evidence panel:** Each answer shows retrieved review count, mode, sources used, and categories covered
- **System prompt engineering:** Instructs the LLM to reference the 4 friction pillars, acknowledge evidence gaps, and stay concise (200–400 words)

### 8.5 System Prompt Design

The chatbot system prompt:
- Identifies the LLM as a senior product analyst on Blinkit's Growth Team
- Injects corpus-level statistics (total reviews, categories, top pillar, source channels)
- Enforces citation format: `*"<quote>"* — <Source>, <Category>`
- Requires gap acknowledgment when evidence is insufficient
- Caps response length at 200–400 words for evaluator-friendly density

---

## 9. Streamlit Dashboard

**File:** `app.py`

**Launch:** `streamlit run app.py`

### 6 Interactive Tabs

| Tab | Purpose |
|-----|---------|
| **📊 Dashboard Overview** | Key metrics, category/pillar/theme distribution charts, pipeline funnel table |
| **🔬 Theme Explorer** | Expandable theme cards with quotes, affected categories, pillar mapping; Theme × Category cross-pattern matrix |
| **❓ Strategic Questions** | Dropdown to select any of the 8 strategic questions; generates evidence-backed insight with themes, quotes, and recommendations |
| **🔎 Search & Explore** | Multi-filter (category, pillar, theme) + keyword search; browse and download filtered review data |
| **💬 RAG Chatbot** | Conversational Q&A interface — ask any question, get evidence-backed answers with source citations; supports multi-turn conversation, suggested evaluator questions, and expandable evidence panels |
| **✅ Validation & Methodology** | Documents how data is gathered, themes identified, insights generated, and quality validated |

---

## 10. Automated Scraping (GitHub Actions)

**File:** `.github/workflows/weekly-scrape.yml`

| Setting | Value |
|---------|-------|
| **Schedule** | Every Monday at 6:00 AM IST (`cron: '30 0 * * 1'`) |
| **Manual trigger** | Yes (with store selection + max review count inputs) |
| **Stores scraped** | Google Play Store + Apple App Store |
| **Batch size** | 200 reviews per store (configurable) |
| **Deduplication** | Against cumulative `Output/scrapes/all_scraped_reviews.csv` |
| **Auto-commit** | Commits new data back to repo with count in commit message |
| **Summary** | Posts scrape stats to GitHub Actions run summary |

**Standalone script:** `scrape_reviews.py`
```bash
python scrape_reviews.py                    # Both stores
python scrape_reviews.py --playstore-only   # Play Store only
python scrape_reviews.py --appstore-only    # App Store only
python scrape_reviews.py --max-reviews 500  # Custom limit
```

---

## 11. Pipeline Results & Statistics

### Final Output Distribution

**By Category (1,808 rows):**

| Category | Count | % |
|----------|-------|---|
| general (non-grocery relevant) | 1,084 | 59.9% |
| electronics | 364 | 20.1% |
| personal_care_beauty | 172 | 9.5% |
| pharmacy_health | 120 | 6.6% |
| intimate_personal | 19 | 1.1% |
| home_cleaning | 18 | 1.0% |
| pet | 16 | 0.9% |
| baby | 15 | 0.8% |

**By Friction Pillar (offline classifier):**

| Pillar | Count | % |
|--------|-------|---|
| Quality & Authenticity Risk | 1,722 | 95.2% |
| Immediate Value Disconnection | 74 | 4.1% |
| Discovery Blind Spots | 7 | 0.4% |
| Habitual Tunnel Vision | 5 | 0.3% |

> **Note:** The heavy skew toward "Quality & Authenticity Risk" is an artifact of the offline rule-based classifier — keywords like "fake", "quality", "damaged" are the most common in filtered expansion-category reviews. Running with `--classify ai` distributes pillars more accurately.

---

## 12. How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Full Pipeline (historical data, offline classification)

```bash
python main.py
```

### With Live Scraping

```bash
python main.py --live                    # Play Store + App Store
python main.py --live --reddit           # + Reddit (needs PRAW creds in .env)
python main.py --appstore                # App Store only
```

### With AI Classification

```bash
# Set API key
echo "OPENAI_API_KEY=sk-..." > .env

# Run with AI classifier
python main.py --classify ai
```

### Launch Dashboard

```bash
streamlit run app.py
```

### Weekly Scraper (standalone)

```bash
python scrape_reviews.py --max-reviews 200
```

---

## 13. Validation Framework

### How data quality is validated

| Layer | Method | Implementation |
|-------|--------|----------------|
| **Ingestion** | Deduplication on text content; column normalization; category alias mapping | `ingestion/__init__.py` |
| **Cleaning** | 3-stage deterministic filters with logged drop counts at each stage | `pipeline/filters.py` |
| **Classification** | Pre-classified Reddit rows preserved as ground truth; AI classifier uses structured JSON with temperature=0.1 | `pipeline/classifier.py` |
| **Themes** | Pattern match confidence scores; cross-category matrix for theme coherence | `engine/themes.py` |
| **Insights** | Quote grounding (every theme must cite verbatim text); confidence levels (high/medium/low); evidence gap disclosure | `engine/insights.py` |
| **Source Triangulation** | Same patterns verified across 7+ independent channels (Play Store, App Store, Reddit, YouTube, HackerNews, PissedConsumer) | Dashboard "Validation" tab |
| **Pipeline Transparency** | Full funnel metrics logged and displayed; retention rate tracked | Dashboard + CLI logs |

### What could improve accuracy

- Run `--classify ai` for more nuanced pillar distribution
- Build FAISS index (`OPENAI_API_KEY` required) for semantic search in the Strategic Questions tab
- Add more Reddit thread URLs to `Data/Reddit Data URLs` for broader coverage
- Collect more non-`general` category reviews to reduce the `general` dominance (59.9%)
