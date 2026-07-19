# Context Document: Blinkit Multi-Channel Data Ingestion & Analytics Engine

## Project Overview

This project builds an **AI-powered data ingestion and analytics engine** for **Blinkit** (Quick-Commerce Platform). It is designed to support the **Growth Team's Catalog Cross-Shopping Initiative** by collecting, cleaning, classifying, and structuring consumer feedback at scale — before any feature design begins.

---

## Business Problem

Blinkit users exhibit **"surgical intent"** — they open the app, rapidly purchase recurring grocery staples (milk, eggs, snacks) in under 60 seconds, and leave. This creates **Cross-Shopping Inertia**: users rarely explore or buy from high-margin lifestyle expansion categories.

### Full Category Taxonomy (from data)

The data spans **10 distinct categories** across all source files. These are grouped by strategic relevance:

**Primary Expansion Targets (high-margin lifestyle categories):**

| # | Category Label (in data) | Description |
|---|--------------------------|-------------|
| 1 | `electronics` | Electronics & Accessories |
| 2 | `personal_care_beauty` | Premium Beauty / Skincare / Cosmetics |
| 3 | `pet` | Pet Care |
| 4 | `baby` | Baby Products |

**Secondary Expansion Categories (non-grocery, relevant for cross-shopping):**

| # | Category Label (in data) | Description |
|---|--------------------------|-------------|
| 5 | `home_cleaning` | Home & Cleaning Products |
| 6 | `intimate_personal` | Intimate & Personal Care |
| 7 | `pharmacy_health` | Pharmacy & Health / Wellness |

**Grocery / Baseline Categories (filtered out as noise in Stage 3):**

| # | Category Label (in data) | Description |
|---|--------------------------|-------------|
| 8 | `groceries_fresh` | Fresh Groceries & Staples |
| 9 | `snacks_beverages` | Snacks & Beverages |
| 10 | `general` | General / Cross-category / Unclassified |

> **Note:** The pre-classified Reddit CSV also uses labels like `Electronics/Accessories`, `Beauty/Skincare/Cosmetics`, `Pet Care`, `Baby Products`, and various `General (...)` sub-types. These map onto the same taxonomy above.

### North Star Metric

Increase the percentage of **Monthly Active Customers (MAC)** who purchase from at least **one new category** every month (e.g., converting a grocery buyer into a premium beauty or pet care consumer).

---

## Data Architecture: Dual-Pathway Ingestion

The engine collects data through two complementary channels:

### Pathway A — Live Programmatic App Store Stream

- **Source:** Google Play Store public API
- **Package:** `com.grofers.customerapp` (Blinkit)
- **Region:** India (`country='in'`, `lang='en'`)
- **Sort:** `Sort.NEWEST`; batch size of 50–60 reviews per invocation

### Pathway B — Curated Social Threads & Historical Text

- **Reddit Threads:** 30–40 high-signal public URLs from subreddits like `r/india`, `r/bangalore`, `r/mumbai`, `r/IndianSkincareAddicts`, `r/FuckBlinkit`, etc.
- **Historical Dumps:** Raw, uncleaned text files from external AI agent extractions across multi-channel forums
- **Data Folder:** All raw input files reside in the `Data/` directory:
  - `Ankesh-reviews_raw.csv` — Play Store, AppStore, Hackernews, Reddit, Youtube, Pinned Consumer big Claude driven reviews 
  dumps
  - `reviews_raw.csv` — Additional Play Store, AppStore, Hackernews, Reddit, Youtube, Pinned Consumer big Perplexity driven reviews 
  dumps
  - `Reddit Data URLs` — Curated Reddit thread URL list
  - `SourceThreadTitleSite-PlatformBlinkit-TargetCatego.csv` — Pre-classified thread metadata

Both pathways feed into a **unified cleaning, filtering, and classification pipeline**.

---

## Three-Stage Cleaning & Noise Filter ("Trash Filter")

Every text row — regardless of source — must pass through these sequential gates:

### Stage 1: Transactional Noise Filter

Drop rows that focus on backend logistics/driver operations rather than shopping habits. Blacklisted tokens include:

> `"late delivery"`, `"delivery boy rude"`, `"refund pending"`, `"money deducted"`, `"payment failed"`, `"upi failed"`, `"gpay"`, `"app crash"`, `"coupon code invalid"`, `"driver behavior"`, `"otp not received"`

### Stage 2: Low-Star / High-Friction Focus

- **Drop** all 5-star praise strings (e.g., "Good app", "Very fast")
- **Keep** only rows rated **4 stars or below**, or text conveying hesitation, anxiety, skepticism, or UX frustration

### Stage 3: Vertical Category Isolation

- **Drop** complaints about standard grocery staples (e.g., "potatoes were rotten", "milk packet leaked") — categories `groceries_fresh`, `snacks_beverages`
- **Keep** text tied to **all non-grocery expansion categories**: `electronics`, `personal_care_beauty`, `pet`, `baby`, `home_cleaning`, `intimate_personal`, `pharmacy_health`
- Rows tagged `general` require contextual inspection — keep only if the content references a non-grocery product or shopping behavior

---

## Semantic Classification: 4 Core Friction Pillars

Every surviving row is assigned to exactly **one** of these pillars:

| Pillar | Description |
|--------|-------------|
| **Habitual Tunnel Vision** | Entrenched path-dependency — users behave as automated surgical buyers, purchasing routine grocery baskets and checking out immediately without exploring other categories. |
| **Quality & Authenticity Risk** | Trust deficits — users worry about dark-store storage conditions degrading products (e.g., overheated skincare serums), counterfeit items, or lack of brand warranty validation on expensive gadgets. |
| **Discovery Blind Spots** | Catalog visibility failure — users say they "had no idea" Blinkit sold these products, citing poor search parsing, hidden navigation, or banner blindness from ad-like promotions. |
| **Immediate Value Disconnection** | Unit economics gap — users compare price-per-gram or pack sizes unfavorably against bulk networks (Amazon) or vertical specialists (Nykaa, Supertails). |

---

## Output Schema

The engine exports a structured table / Pandas DataFrame with this schema:

| Column | Assignment Rule |
|--------|-----------------|
| **Source** | Origin channel: `"Play Store Live Stream"`, `"Reddit Link"`, `"Play Store Historical Dump"`, or `"Reddit Historical Dump"` |
| **Platform** | Always `"Blinkit"` |
| **Target Category** | One of the non-grocery categories: `electronics`, `personal_care_beauty`, `pet`, `baby`, `home_cleaning`, `intimate_personal`, `pharmacy_health`, or `general` (when contextually relevant) |
| **Friction Pillar** | Exactly one of the 4 defined pillars |
| **Raw Content** | Cleaned verbatim text snippet / user comment |
| **Opportunity** | Strategic growth PM hypothesis on how to refactor the product to clear the user's complaint |

---

## Downstream Integration

The output data is designed to integrate with:

- **FAISS vector index** using `text-embedding-3-small` for local-memory semantic search
- **Streamlit runtime** for interactive dashboard tabs and exploration

No rows that clear the ingestion filters should be truncated or dropped.

---

## Key Constraints & Directives

1. All data must be region-locked to **India** and **English** language
2. The cleaning pipeline must be **deterministic** — no ambiguous heuristics
3. Every row must map to exactly **one** friction pillar and **one** target category
4. The system must handle both structured (CSV) and unstructured (raw text dump) inputs through a unified funnel
5. Token efficiency matters — aggressive noise filtering reduces downstream AI processing costs
