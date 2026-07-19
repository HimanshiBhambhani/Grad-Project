# Edge Cases, Corner Scenarios & Phase-Wise Implementation Plan

## Table of Contents

1. [Data Ingestion Edge Cases](#1-data-ingestion-edge-cases)
2. [Cleaning Pipeline Edge Cases](#2-cleaning-pipeline-edge-cases)
3. [Classification Edge Cases](#3-classification-edge-cases)
4. [Discovery Engine Edge Cases](#4-discovery-engine-edge-cases)
5. [RAG Chatbot Edge Cases](#5-rag-chatbot-edge-cases)
6. [Streamlit Dashboard Edge Cases](#6-streamlit-dashboard-edge-cases)
7. [Scraping & Automation Edge Cases](#7-scraping--automation-edge-cases)
8. [API & Infrastructure Edge Cases](#8-api--infrastructure-edge-cases)
9. [Data Quality & Semantic Edge Cases](#9-data-quality--semantic-edge-cases)
10. [Phase-Wise Implementation Plan](#10-phase-wise-implementation-plan)

---

## 1. Data Ingestion Edge Cases

### 1.1 CSV Schema Drift

| Scenario | Risk | Mitigation |
|----------|------|------------|
| Column names change between CSV dumps | `KeyError` on `text`, `rating`, `category` | Use positional fallback: if expected column missing, try index-based mapping; log warning |
| Unnamed column (`''`) not present | `source_raw` column missing → downstream `SOURCE_LABELS` mapping fails | Default to `"Unknown"` source if no unnamed column found |
| New source channels appear (e.g., `"Trustpilot"`, `"Twitter"`) | Unmapped sources get no `SOURCE_LABELS` entry → `KeyError` or empty Source | Add `"Unknown Channel"` as default fallback in `config.SOURCE_LABELS` |
| CSV has extra/unexpected columns | Columns leak into output schema | Use explicit column selection after loading; ignore unknown columns |

### 1.2 Encoding & Format Issues

| Scenario | Risk | Mitigation |
|----------|------|------------|
| Non-UTF-8 characters in review text (Hindi, emoji, special chars) | `UnicodeDecodeError` during CSV read | Use `encoding='utf-8'` with `errors='replace'`; log replacement count |
| CSV cells contain embedded newlines or commas | Row parsing breaks; one review spans multiple rows | Use `pandas.read_csv()` with `quoting=csv.QUOTE_ALL`; verify row count post-load |
| BOM (Byte Order Mark) in CSV header | First column name has invisible `\ufeff` prefix | Strip BOM with `encoding='utf-8-sig'` |
| Empty CSV file (0 bytes) | `EmptyDataError` from pandas | Check `os.path.getsize()` before loading; skip with warning |

### 1.3 Deduplication Corner Cases

| Scenario | Risk | Mitigation |
|----------|------|------------|
| Near-duplicate reviews (differ by whitespace, punctuation, or casing) | Not caught by exact-match dedup → inflated counts | Normalise text (lowercase, strip whitespace, remove punctuation) before dedup hash |
| Same text but different source channels | Legitimate cross-posting dropped as duplicate | Dedup on `(text, source_raw)` pair instead of text alone; or keep first occurrence with source priority |
| Very short reviews (`"good"`, `"ok"`, `"nice"`) duplicated thousands of times | One-word reviews dominate corpus after dedup keeps one | Stage 2 praise filter handles most; add minimum text length check (e.g., ≥10 chars) |
| Pre-classified Reddit CSV has categories not in `CATEGORY_ALIAS_MAP` | Unmapped label → `None` or empty category | Log unmapped labels; default to `"general"` |

### 1.4 Missing or Corrupt Data Files

| Scenario | Risk | Mitigation |
|----------|------|------------|
| `reviews_raw.csv` missing from `Data/` | `FileNotFoundError` crash | Check `Path.exists()` before loading; log error and proceed with remaining sources |
| `links-data.docx` missing | Reddit URL loading fails | Wrap in try/except; skip with warning |
| `Reddit Data URLs` file empty or malformed | Zero URLs scraped | Validate URL count after loading; warn if 0 |
| CSV file partially downloaded / truncated | Pandas loads partial data with corrupted last row | Use `on_bad_lines='warn'` in `pd.read_csv()` |

---

## 2. Cleaning Pipeline Edge Cases

### 2.1 Stage 1: Transactional Noise Filter

| Scenario | Risk | Mitigation |
|----------|------|------------|
| Review mentions logistics AND product quality (e.g., "late delivery but earbuds were also fake") | Entire row dropped — loses the product quality signal | Two-pass approach: check if text ALSO matches expansion keywords; if yes, keep but flag |
| Blacklisted token appears inside a legitimate word (e.g., "app crash" inside "app crashed while browsing electronics") | Over-aggressive filtering drops relevant content | Use word-boundary regex `\b` anchors; review dropped rows periodically |
| Non-English text passes through (Hindi reviews transliterated in English script) | Blacklisted tokens don't match Hindi equivalents | Accept this limitation for v1; future: add Hindi transliteration patterns |
| Review text is `NaN` / `None` / empty string | Regex match on `NaN` throws `TypeError` | Convert to `str` and check `len() > 0` before pattern matching |

### 2.2 Stage 2: Rating Filter

| Scenario | Risk | Mitigation |
|----------|------|------------|
| Rating column has non-numeric values (`"N/A"`, `"five"`, `""`) | `float()` conversion fails | Use `pd.to_numeric(errors='coerce')` → NaN; treat NaN as unrated |
| Reddit/forum posts have no star rating (NaN) | All Reddit content treated as unrated → different filter logic path | Unrated path: drop if pure praise pattern, keep otherwise (already implemented) |
| 5-star review with genuine friction buried deep in text ("Great app overall BUT the earbuds I got were clearly fake") | Dropped by praise filter if initial text matches "great app" | Friction keyword exception scan (`but`, `however`, `issue`, `fake`, `expensive`) — already implemented but needs thorough keyword list |
| Rating = 0 or rating > 5 | Out-of-range values confuse the ≤4 logic | Clamp ratings to [1, 5]; treat 0 as missing |

### 2.3 Stage 3: Category Isolation

| Scenario | Risk | Mitigation |
|----------|------|------------|
| `general` review mentions BOTH grocery AND expansion product ("ordered milk and the earbuds were fake") | Keyword regex matches "earbuds" → kept, but also contains grocery noise | Accept: the expansion signal is more valuable than the grocery noise. Log as mixed-content |
| Expansion keyword appears in a negative context ("I would never buy electronics from Blinkit") | Keyword match keeps the row — but the intent is refusal, not purchase | Valid: refusal-to-buy IS a friction signal. This is correct behavior |
| Product keyword regex has false positives (e.g., "baby" matches "baby corn", "pet" matches "Peter") | Grocery items or names trigger expansion keyword match | Use multi-word patterns where possible (`baby product`, `pet food`); add exclusion list for known false positives |
| Category label has trailing whitespace or mixed casing | `"Electronics "` ≠ `"electronics"` → row treated as unknown | `.str.strip().str.lower()` on category column during ingestion |
| New product category appears in future data (e.g., `"books"`, `"stationery"`) | Not in taxonomy → treated as `general` → likely dropped by keyword filter | Periodically review dropped `general` rows; update `KEEP_CATEGORIES` in config |
| 93.2% of rows dropped at Stage 3 — all `general` with no expansion keywords | Extremely aggressive filter leaves very thin corpus (1,808 rows) | Expected behavior for this use case; log retention rate; future: loosen keyword regex or use semantic classifier |

---

## 3. Classification Edge Cases

### 3.1 Offline Classifier

| Scenario | Risk | Mitigation |
|----------|------|------------|
| Review matches keywords from multiple pillars ("didn't know Blinkit had electronics, and they're overpriced") | Sequential priority assigns first match only | Known limitation of rule-based approach; first-match priority order is deliberate. Run `--classify ai` for nuanced multi-signal analysis |
| No keywords match any pillar | Default fallback to "Quality & Authenticity Risk" | Causes 95.2% skew toward this pillar. Acceptable for offline mode; AI mode redistributes accurately |
| Review is entirely emoji or non-text content | Regex finds no matches → fallback pillar assigned | Pre-filter: drop rows with <5 alpha characters |
| Pillar keywords appear in negation ("NOT fake", "no quality issues") | False positive: assigned to Quality & Authenticity Risk | Rule-based approach cannot handle negation. Only AI mode can. Document this limitation |

### 3.2 AI Classifier (Groq / Llama 3.3 70B)

| Scenario | Risk | Mitigation |
|----------|------|------------|
| API returns non-JSON response | `json.loads()` raises `JSONDecodeError` | Wrap in try/except; fall back to offline classification for that row |
| API returns valid JSON but with wrong keys | `result.get("friction_pillar")` returns `None` | Validate presence of expected keys; use offline fallback if missing |
| API returns pillar name with slight spelling variation ("Quality and Authenticity Risk" vs "Quality & Authenticity Risk") | Not in `config.FRICTION_PILLARS` list → validation fails | Fuzzy matching: check `pillar.lower() in fp.lower()` (already implemented) |
| Rate limit exceeded (Groq: 30 req/min free tier) | `429 Too Many Requests` error | Exponential backoff with retry; current: 1s pause every 20 rows. May need 2–3s for free tier |
| API timeout on a single row | Row classification hangs or fails | Set `timeout=30` on API call; catch timeout exception; fall back to offline for that row |
| Batch of 1,808 rows at 20/s = ~90 seconds total | Acceptable for one-time runs; expensive at scale | For >10K rows, consider batch API or pre-filtering to reduce AI-classified count |
| API key invalid or expired | All AI classification fails silently with fallback | Validate API key at startup with a test call; fail fast with clear error message |

### 3.3 Pre-Classified Data Handling

| Scenario | Risk | Mitigation |
|----------|------|------------|
| Pre-classified CSV has pillar names not in the 4-pillar taxonomy | Row preserved with invalid pillar → downstream charts break | Map or validate against `config.FRICTION_PILLARS`; log and correct mismatches |
| Multi-category row (`"Electronics; Beauty"` with `;` separator) | Single-category schema can't hold two values | Expand into two rows (already implemented); verify neither is `NaN` |
| Pre-classified row contradicts AI classifier on same text | Inconsistency in ground truth vs prediction | Pre-classified rows always take priority (already implemented); log disagreements for review |

---

## 4. Discovery Engine Edge Cases

### 4.1 FAISS Index

| Scenario | Risk | Mitigation |
|----------|------|------------|
| `GOOGLE_API_KEY` missing or invalid | Embedding call fails → no FAISS index built | Graceful degradation: dashboard falls back to keyword search; show user warning |
| Embedding quota exhausted (N/A with local model) | `429 RESOURCE_EXHAUSTED` during batch embedding | Implement retry with exponential backoff; save partial index; resume from checkpoint |
| Embedding API returns different dimensions than expected | FAISS index dimension mismatch → `RuntimeError` | Detect dimension from first batch; build index dynamically rather than hardcoding |
| Review text is empty string after cleaning | Zero-length text produces meaningless embedding | Filter out empty/whitespace-only texts before embedding |
| Very long review text (>8,000 chars) | Exceeds embedding model token limit | Truncate to 8,000 chars (already implemented) |
| FAISS index file corrupted on disk | `faiss.read_index()` fails | Catch exception; rebuild index from scratch; delete corrupt files |
| Metadata pickle file out of sync with FAISS index | Index has N vectors but metadata has M ≠ N entries | Validate `index.ntotal == len(metadata)` on load; rebuild if mismatch |
| Search query returns 0 results (top-K scores all below threshold) | Empty evidence for RAG chatbot | Return all K results regardless of score; let the LLM decide relevance |

### 4.2 Semantic Search Query

| Scenario | Risk | Mitigation |
|----------|------|------------|
| Query in Hindi or mixed-language | English embedding model struggles with non-English queries | Accept limitation for v1; future: use multilingual embedding model |
| Very short query ("fake") | Low-dimensional signal → poor retrieval quality | Pad short queries with context: "Blinkit customer reviews about: fake" |
| Very long query (>500 words) | Exceeds model token limit or dilutes signal | Truncate query to 500 chars |
| Query about a topic not in the corpus ("What about Blinkit's grocery delivery speed?") | Returns irrelevant results (best-match among bad options) | LLM should acknowledge evidence gap; offline mode explicitly says "no relevant evidence" |

---

## 5. RAG Chatbot Edge Cases

### 5.1 Conversation Management

| Scenario | Risk | Mitigation |
|----------|------|------------|
| User asks 50+ questions in one session | Conversation history grows unboundedly → token limit exceeded | Window limited to last 3 Q&A pairs (6 messages) — already implemented |
| User asks a question unrelated to Blinkit ("What's the weather?") | LLM may hallucinate an answer using review evidence | System prompt constrains to Blinkit context; LLM should say "I can only answer questions about Blinkit's customer feedback" |
| User inputs adversarial/prompt-injection text | LLM behavior manipulated | System prompt is robust; review text is treated as evidence (not instructions); Groq has safety filters |
| User asks about specific customer by name or PII in review | Privacy concern if reviews contain names | Reviews are public (Play Store, Reddit); but redact if any PII found in preprocessing |
| User clicks suggested question while previous answer is still generating | Streamlit may duplicate the request or show stale state | Use `st.spinner()` to block interaction during generation (already implemented) |

### 5.2 Generation Quality

| Scenario | Risk | Mitigation |
|----------|------|------------|
| Retrieved evidence is all from one category (e.g., all electronics) | Answer is biased toward one vertical | Include category distribution in context template; LLM should acknowledge skew |
| Retrieved evidence contradicts itself | LLM may cherry-pick one side | System prompt instructs to present multiple perspectives with quotes |
| LLM hallucinates quotes not in the evidence | Fabricated citations undermine credibility | System prompt enforces verbatim citation format; offline mode uses actual text excerpts |
| LLM response exceeds 400-word limit | Long responses overwhelm evaluator | `max_tokens=1500` caps output; system prompt requests 200–400 words |
| API failure mid-conversation | Answer generation fails; conversation breaks | Catch exception; return offline fallback answer; preserve conversation history |

### 5.3 Mode Transitions

| Scenario | Risk | Mitigation |
|----------|------|------------|
| FAISS index exists but API key removed between sessions | Mode changes from Full RAG to FAISS+Offline mid-session | Re-evaluate mode on each `ask()` call; display current mode in sidebar |
| User builds FAISS index while dashboard is running | Streamlit cache holds stale "no index" state | Use `st.cache_data` with TTL or clear cache on tab switch |
| Offline mode produces noticeably worse answers than AI mode | User experience inconsistency | Clearly label mode in UI; offline answers include disclaimer |

---

## 6. Streamlit Dashboard Edge Cases

### 6.1 Data Loading

| Scenario | Risk | Mitigation |
|----------|------|------------|
| `Output/blinkit_clean_data.csv` doesn't exist | Dashboard crashes on startup | Show error message with instructions: "Run `python main.py` first" (already implemented) |
| CSV has 0 rows after pipeline (all filtered out) | Division by zero in percentage calculations; empty charts | Guard all calculations with `if len(df) > 0`; show "No data" message |
| CSV has columns missing (e.g., `Friction Pillar` missing because pipeline was interrupted) | `KeyError` in chart rendering | Check for required columns at load time; show specific missing column error |
| CSV is very large (>100K rows after future scaling) | Streamlit becomes sluggish; `st.dataframe` slow | Use pagination; limit displayed rows to 500; use `@st.cache_data` aggressively |

### 6.2 Chart Rendering

| Scenario | Risk | Mitigation |
|----------|------|------------|
| Category has only 1 review (e.g., `baby` = 15 rows) | Bar chart segment too small to see or label | Use horizontal bar charts with explicit value labels (already implemented) |
| All reviews fall into one pillar (offline classifier: 95.2% Quality) | Pie chart is essentially one color | Acknowledge in UI: "Note: offline classifier skews toward this pillar" (already implemented) |
| Theme × Category matrix is sparse (many zeros) | Hard to read; no clear patterns | Highlight non-zero cells; sort by row/column totals |
| Theme Confidence = 0 for all themes (no keywords matched) | "Primary Theme" column is empty/None | Default to "Unclassified" theme; exclude from theme charts |

### 6.3 Interactive Filters

| Scenario | Risk | Mitigation |
|----------|------|------------|
| User selects all filters simultaneously (category + pillar + theme + keyword) | Filter intersection = 0 rows | Show "No reviews match all filters" with suggestion to relax criteria |
| User enters regex-special characters in keyword search (e.g., `(fake)`, `price+`) | `re.error` in `str.contains()` | Use `regex=False` in `str.contains()` for keyword search (already using `case=False, na=False`) |
| User downloads CSV with 0 matching rows | Empty CSV downloaded | Disable download button when filtered count = 0 |

---

## 7. Scraping & Automation Edge Cases

### 7.1 Play Store Scraper

| Scenario | Risk | Mitigation |
|----------|------|------------|
| Google Play Store changes its API / blocks scraping | `google-play-scraper` returns empty or errors | Pin package version; add error handling; alert on 0 reviews scraped |
| Package ID `com.grofers.customerapp` is deprecated / renamed | Scraper returns no results | Monitor package ID validity; add configurable fallback |
| Reviews returned in non-English language despite `lang='en'` filter | Non-English text passes into pipeline | Secondary language detection filter; or accept and let cleaning pipeline handle |
| API returns duplicate reviews across batches | Inflated scrape counts | Deduplication against cumulative CSV (already implemented) |
| Rate limiting / IP blocking by Google | Scraper fails with connection error | Add retry logic with exponential backoff; respect `SCRAPE_DELAY_SECONDS` |

### 7.2 App Store Scraper

| Scenario | Risk | Mitigation |
|----------|------|------------|
| App ID `960335206` changes (app re-listed) | Scraper returns wrong app or fails | Verify app name in response matches "Blinkit"; alert on mismatch |
| `app-store-scraper` library breaks with Apple API changes | Import error or empty results | Pin version; handle gracefully; fall back to Play Store only |
| App Store returns reviews in multiple languages | Mixed-language corpus | Filter to English-only; use `country='in'` (already set) |

### 7.3 Reddit Scraper (PRAW)

| Scenario | Risk | Mitigation |
|----------|------|------------|
| Reddit thread URL is deleted or private | PRAW returns `403 Forbidden` or empty response | Catch `prawcore.exceptions.Forbidden`; skip URL and log |
| Reddit API credentials expired | All Reddit scraping fails | Validate credentials at startup; clear error message |
| Thread has 0 comments | Empty data from that URL | Skip threads with 0 comments; log warning |
| Comment text contains only links / images (no text content) | Empty `text` field after extraction | Filter comments with `len(text.strip()) < 10` |
| Rate limit: Reddit allows ~60 requests/minute | Scraper exceeds rate → temporary ban | PRAW handles rate limiting internally; add `SCRAPE_DELAY_SECONDS` between threads |

### 7.4 GitHub Actions Automation

| Scenario | Risk | Mitigation |
|----------|------|------------|
| GitHub Actions secrets not set (`GOOGLE_API_KEY`, etc.) | Scraper runs but AI features fail | Scraper doesn't need AI keys; only needs `google-play-scraper` / `app-store-scraper` — no secret required for scraping |
| Cron job doesn't trigger (GitHub disables cron on inactive repos) | No new data collected for weeks | Add manual `workflow_dispatch` trigger (already implemented); periodic activity commits |
| Git push fails (merge conflict on `all_scraped_reviews.csv`) | Workflow fails silently | Use `git pull --rebase` before push; or force push scrape data |
| Scrape returns 0 new reviews (all duplicates) | Empty commit or unnecessary commit | Check new row count before committing; skip commit if 0 new rows (already implemented) |
| Workflow runs longer than GitHub Actions timeout (6h default) | Job killed mid-scrape | `--max-reviews 200` keeps runtime under 2 minutes; safe margin |

---

## 8. API & Infrastructure Edge Cases

### 8.1 Groq API (LLM)

| Scenario | Risk | Mitigation |
|----------|------|------------|
| Groq free tier: 30 requests/minute, 14,400/day | Classification of 1,808 rows needs ~1,808 calls | Batch where possible; add 2–3s delay between calls; offline fallback for bulk runs |
| `llama-3.3-70b-versatile` model deprecated | API returns model-not-found error | Pin model in config; update when new version available |
| Groq response includes markdown formatting in JSON | `json.loads()` fails on ```json wrapped output | Strip markdown code fences before parsing; regex: `re.sub(r'^```json\n|```$', '', text)` |
| Network timeout / intermittent connectivity | Random API failures across batch | Retry up to 3 times with exponential backoff; fall back to offline for persistent failures |
| API key revoked / billing issues | All AI features fail | Validate key at startup; clear error message; automatic offline fallback |

### 8.2 Local Embeddings (sentence-transformers)

| Scenario | Risk | Mitigation |
|----------|------|------------|
| Free tier quota exhausted (as observed with LLM endpoint) | `429 RESOURCE_EXHAUSTED` on embedding calls | Different quota bucket for embeddings (confirmed working); if exhausted, skip FAISS build |
| Embedding dimension changes between model versions | FAISS index incompatible with new embeddings | Store model version in metadata; rebuild index if version changes |
| API returns partial batch results | Some texts embedded, others not | Validate batch output length matches input length; retry failed items |
| Large corpus (>10K reviews) exhausts embedding quota | Cannot embed all reviews in one session | Checkpoint progress: save partial index; resume from last embedded position |

### 8.3 Environment & Dependencies

| Scenario | Risk | Mitigation |
|----------|------|------------|
| Python version incompatibility (project uses 3.14, user has 3.10) | Syntax errors or missing features | Specify `python_requires >= 3.10` in setup; test on 3.10+ |
| `faiss-cpu` installation fails on ARM Mac (M-series) | ImportError at runtime | Use `conda install faiss-cpu` or `pip install faiss-cpu` (ARM wheels available since faiss 1.7.4) |
| `.env` file has Windows line endings (`\r\n`) | `dotenv` may include `\r` in values | Use `load_dotenv()` with default strip behavior; explicitly strip values |
| Multiple `.env` files in different locations | Wrong config loaded | Use explicit path: `load_dotenv(PROJECT_ROOT / '.env')` |

---

## 9. Data Quality & Semantic Edge Cases

### 9.1 Content Quality

| Scenario | Risk | Mitigation |
|----------|------|------------|
| Reviews in Hindi/Hinglish transliteration ("ye earbuds bilkul fake hai") | English keyword regex misses Hindi friction signals | Accept for v1; future: add transliterated Hindi keyword list |
| Sarcastic reviews ("Oh great, another fake product from Blinkit 🙄") | Sentiment analysis may misread as positive praise | Stage 2 friction exception catches "fake"; sarcasm handling is an AI-mode benefit |
| Very long reviews (>2000 chars) with mixed signals | Dominant keywords may not reflect overall intent | AI classifier processes full context; offline classifier takes first match only |
| Spam / bot reviews (repeated text, nonsensical content) | Inflate corpus with noise | Deduplication catches exact duplicates; length filter catches very short spam |
| Reviews about a different "Blinkit" product/company | Wrong-context data in corpus | All data is region-locked to India + specific app package IDs |

### 9.2 Temporal Edge Cases

| Scenario | Risk | Mitigation |
|----------|------|------------|
| Historical data is 2+ years old; Blinkit's catalog has changed | Insights based on stale data may not reflect current state | Weekly live scraping adds fresh data; note data recency in Validation tab |
| Seasonal spikes (Diwali electronics, summer skincare) | Temporal bias in theme distribution | Future: add date-based filtering in dashboard; note seasonality in insights |
| Review dates in inconsistent formats across sources | Date parsing fails or produces wrong values | Use `pd.to_datetime(errors='coerce')`; accept NaT for unparseable dates |

### 9.3 Corpus Composition

| Scenario | Risk | Mitigation |
|----------|------|------------|
| 59.9% of clean data is `general` category | Expansion categories underrepresented → thin evidence per category | Acknowledged in docs; future: targeted scraping for underrepresented categories |
| Only 5 rows for "Habitual Tunnel Vision" pillar (0.3%) | Insufficient evidence for insights on this pillar | AI classifier distributes better; offline skew is documented; note low-confidence for this pillar |
| `pet` (16) and `baby` (15) categories have <20 rows each | Statistical analysis unreliable at this sample size | Flag low-N categories in dashboard; set confidence=low for insights on these |
| Source distribution heavily skewed to Play Store historical | Cross-channel triangulation limited for minority sources | Log source distribution; future: balance with more Reddit and direct scraping |

---

## 10. Phase-Wise Implementation Plan

### Phase 0: Environment Setup (Day 1)

| Step | Action | Verification |
|------|--------|--------------|
| 0.1 | Clone repo: `git clone git@github-personal:HimanshiBhambhani/Grad-Project.git` | `ls` shows project structure |
| 0.2 | Install Python dependencies: `pip install -r requirements.txt` | `python -c "import pandas, streamlit, groq, faiss"` succeeds |
| 0.3 | Create `.env` with `GROQ_API_KEY` and `GOOGLE_API_KEY` | `python -c "from dotenv import load_dotenv; load_dotenv(); import config; print(config.GROQ_API_KEY[:8])"` prints key prefix |
| 0.4 | Verify API keys work | Run test: Groq responds, local embedding returns 384 dims |
| 0.5 | Verify data files exist in `Data/` | `ls Data/` shows both CSVs, Reddit URLs, pre-classified CSV |

**Exit criteria:** All imports succeed, both API keys validated, data files present.

---

### Phase 1: Data Ingestion & Validation (Day 1–2)

| Step | Action | Verification |
|------|--------|--------------|
| 1.1 | Run ingestion: `python -c "from ingestion import load_all_historical; df = load_all_historical(); print(len(df))"` | Output: `28274` |
| 1.2 | Verify schema: check columns `source_raw`, `date`, `rating`, `text`, `category`, `url` exist | `print(df.columns.tolist())` shows all 6 |
| 1.3 | Verify deduplication: confirm 56,530 → 28,274 | Log message shows "Deduplicated: removed X rows" |
| 1.4 | Verify category distribution matches expected counts | `df['category'].value_counts()` matches Section 3.3 of implementation.md |
| 1.5 | Check for NaN/empty text rows | `df['text'].isna().sum()` should be 0 |
| 1.6 | Check pre-classified CSV loads and alias mapping works | Pre-classified rows have canonical category labels |

**Edge cases to test:**
- [ ] Load with one CSV missing — verify graceful handling
- [ ] Check for encoding issues in review text
- [ ] Verify dedup handles near-duplicates (trailing whitespace)

**Exit criteria:** Unified DataFrame of 28,274 rows with correct schema and no NaN text values.

---

### Phase 2: Cleaning Pipeline (Day 2–3)

| Step | Action | Verification |
|------|--------|--------------|
| 2.1 | Run full pipeline: `python main.py` | Output CSV created at `Output/blinkit_clean_data.csv` |
| 2.2 | Verify Stage 1 drops: ~685 rows removed (transactional noise) | CLI logs show "Stage 1: 28274 → 27589" |
| 2.3 | Verify Stage 2 drops: ~909 rows removed (5-star praise) | CLI logs show "Stage 2: 27589 → 26680" |
| 2.4 | Verify Stage 3 drops: ~24,872 rows removed (grocery/general) | CLI logs show "Stage 3: 26680 → 1808" |
| 2.5 | Spot-check 20 dropped rows from each stage | Confirm drops are justified; no false positives in expansion content |
| 2.6 | Spot-check 20 surviving rows | Confirm all reference expansion categories or non-grocery content |
| 2.7 | Verify no `groceries_fresh` or `snacks_beverages` rows survive | `df[df['category'].isin(['groceries_fresh','snacks_beverages'])]` should be empty |

**Edge cases to test:**
- [ ] Review with logistics AND product keywords — verify handling
- [ ] 5-star review with friction language ("great but fake") — verify it's kept
- [ ] `general` review mentioning expansion product keyword — verify it's kept
- [ ] `general` review with NO expansion keywords — verify it's dropped
- [ ] Empty text rows — verify they don't cause crashes

**Exit criteria:** 1,808 clean rows in output CSV; pipeline funnel matches documented numbers.

---

### Phase 3: Classification (Day 3–4)

| Step | Action | Verification |
|------|--------|--------------|
| 3.1 | Run offline classification: `python main.py --classify offline` | All 1,808 rows have `Friction Pillar` assigned |
| 3.2 | Check pillar distribution | Dominated by "Quality & Authenticity Risk" (expected for offline) |
| 3.3 | Run AI classification on a sample: `python main.py --classify ai` | Groq API responds; pillars more evenly distributed |
| 3.4 | Verify output schema: `Source`, `Platform`, `Target Category`, `Friction Pillar`, `Raw Content`, `Opportunity` | `df.columns.tolist()` matches expected 6 columns |
| 3.5 | Verify pre-classified rows preserved | Pre-classified Reddit rows retain their expert labels |
| 3.6 | Check `Opportunity` field populated (AI mode) | Non-empty string for AI-classified rows |

**Edge cases to test:**
- [ ] Review matching zero pillar keywords — verify fallback
- [ ] Review matching multiple pillars — verify priority ordering
- [ ] API failure mid-batch — verify offline fallback activates
- [ ] Groq rate limiting — verify delay logic works
- [ ] Invalid JSON from LLM — verify error handling

**Exit criteria:** All rows have valid pillar assignment; AI mode produces better distribution than offline.

---

### Phase 4: Theme Identification (Day 4)

| Step | Action | Verification |
|------|--------|--------------|
| 4.1 | Run theme identification: `python -c "from engine.themes import identify_themes; import pandas as pd; df = pd.read_csv('Output/blinkit_clean_data.csv'); df = identify_themes(df); print(df['Primary Theme'].value_counts())"` | All 10 themes have counts |
| 4.2 | Verify Theme × Category matrix | Cross-pattern shows non-trivial clustering |
| 4.3 | Spot-check theme assignments | Review 10 rows per theme; confirm keyword matches are correct |
| 4.4 | Check for "Unclassified" themes | Rows with no keyword matches → verify handling |

**Edge cases to test:**
- [ ] Review with 0 theme keyword matches — verify default behavior
- [ ] Review matching all 10 themes equally — verify tiebreaker
- [ ] Very short review (<10 words) — verify theme assignment quality

**Exit criteria:** All rows have `Primary Theme` and `Theme Confidence` values.

---

### Phase 5: Discovery Engine — FAISS Index (Day 5)

| Step | Action | Verification |
|------|--------|--------------|
| 5.1 | Build FAISS index: `python -c "from engine import build_index; import pandas as pd; build_index(pd.read_csv('Output/blinkit_clean_data.csv'))"` | Files created: `Output/faiss_index.bin`, `Output/faiss_meta.pkl` |
| 5.2 | Verify index dimensions | Should be 384 (all-MiniLM-L6-v2 dimension) |
| 5.3 | Test search: query "fake electronics" | Returns relevant reviews about counterfeit products |
| 5.4 | Test search: query "beauty product storage" | Returns relevant reviews about skincare/dark store |
| 5.5 | Verify metadata count matches index vectors | `index.ntotal == len(metadata)` |

**Edge cases to test:**
- [ ] Embedding API fails mid-batch — verify partial recovery
- [ ] Empty text review in corpus — verify no crash
- [ ] Search with 0-length query — verify handling
- [ ] Load index after file deleted — verify graceful failure

**Exit criteria:** FAISS index built with 1,808 vectors; semantic search returns relevant results.

---

### Phase 6: RAG Chatbot (Day 5–6)

| Step | Action | Verification |
|------|--------|--------------|
| 6.1 | Test chatbot in Full RAG mode (FAISS + Groq) | Evidence-backed answers with source citations |
| 6.2 | Test chatbot in offline mode (remove GROQ_API_KEY temporarily) | Rule-based answers with category/pillar breakdown |
| 6.3 | Test multi-turn conversation | Bot remembers context from previous questions |
| 6.4 | Test all 8 suggested questions | Each produces a structured, relevant answer |
| 6.5 | Test clear history | Conversation resets cleanly |

**Edge cases to test:**
- [ ] Off-topic question ("What's the weather?") — verify polite refusal
- [ ] Very long question (>500 words) — verify handling
- [ ] Rapid-fire questions (spam clicking) — verify no duplicate responses
- [ ] API failure during generation — verify offline fallback
- [ ] Question about a category with <15 rows (pet, baby) — verify low-confidence acknowledgment

**Exit criteria:** Chatbot answers evaluator questions with grounded evidence across all 4 operating modes.

---

### Phase 7: Streamlit Dashboard (Day 6–7)

| Step | Action | Verification |
|------|--------|--------------|
| 7.1 | Launch dashboard: `streamlit run app.py` | Opens on `localhost:8501` |
| 7.2 | Tab 1 (Overview): verify metrics, charts, funnel table | All numbers match pipeline output |
| 7.3 | Tab 2 (Themes): expand all theme cards; verify quotes and categories | Quotes are real review text |
| 7.4 | Tab 3 (Strategic Questions): test each of the 8 questions | Insights generated with themes and recommendations |
| 7.5 | Tab 4 (Search): apply category + pillar + keyword filters | Filtered results correct; download works |
| 7.6 | Tab 5 (Chatbot): test conversation flow | Messages display correctly; evidence panel expands |
| 7.7 | Tab 6 (Validation): verify methodology documentation | Text is accurate and up-to-date |

**Edge cases to test:**
- [ ] Dashboard with no data file — verify error message
- [ ] All filters selected → 0 results — verify UX
- [ ] Browser refresh — verify session state persists for chatbot
- [ ] Multiple browser tabs — verify independent sessions
- [ ] Regex-special characters in keyword search — verify no crash

**Exit criteria:** All 6 tabs functional; dashboard is presentation-ready.

---

### Phase 8: Automated Scraping (Day 7)

| Step | Action | Verification |
|------|--------|--------------|
| 8.1 | Run standalone scraper: `python scrape_reviews.py --max-reviews 50` | New reviews written to `Output/scrapes/all_scraped_reviews.csv` |
| 8.2 | Run again — verify deduplication | No new rows added if all reviews already exist |
| 8.3 | Verify GitHub Actions workflow YAML is valid | `act` or manual trigger on GitHub |
| 8.4 | Test manual workflow dispatch on GitHub | Workflow runs successfully; commit with new data |

**Edge cases to test:**
- [ ] Scraper with `--max-reviews 0` — verify no crash
- [ ] Network failure during scrape — verify partial results saved
- [ ] App Store scraper when library breaks — verify Play Store still works independently

**Exit criteria:** Weekly scrape pipeline verified end-to-end.

---

### Phase 9: Documentation & Final Validation (Day 8)

| Step | Action | Verification |
|------|--------|--------------|
| 9.1 | Update all doc files to reflect current LLM provider (Groq + local sentence-transformers) | Grep for "OpenAI" / "Llama 3.3 70B (Groq)" → replace with actual providers |
| 9.2 | Run full pipeline end-to-end one final time | Clean output matches expected counts |
| 9.3 | Verify all edge case mitigations from this document are implemented | Cross-reference each section |
| 9.4 | Push all changes to GitHub | `git push` succeeds |
| 9.5 | Record a 2-minute demo walkthrough of the dashboard | Video showing all tabs + chatbot interaction |

**Exit criteria:** All documentation current; pipeline runs cleanly; dashboard demo-ready.

---

## Implementation Timeline Summary

```
Day 1     ████░░░░░░░░░░░░  Phase 0 + Phase 1 (Setup + Ingestion)
Day 2     ░░██████░░░░░░░░  Phase 1 + Phase 2 (Ingestion + Cleaning)
Day 3     ░░░░░░████░░░░░░  Phase 2 + Phase 3 (Cleaning + Classification)
Day 4     ░░░░░░░░░░██░░░░  Phase 4 (Themes)
Day 5     ░░░░░░░░░░░░██░░  Phase 5 + Phase 6 (FAISS + Chatbot)
Day 6     ░░░░░░░░░░░░░░██  Phase 6 + Phase 7 (Chatbot + Dashboard)
Day 7     ░░░░░░░░░░░░░░░█  Phase 7 + Phase 8 (Dashboard + Scraping)
Day 8     ░░░░░░░░░░░░░░░█  Phase 9 (Docs + Final Validation)
```

**Total estimated time:** 8 working days (can compress to 5 days with focused execution)

**Critical path:** Phase 2 (Cleaning) → Phase 3 (Classification) → Phase 5 (FAISS) → Phase 6 (Chatbot)

**Parallelizable:** Phase 4 (Themes) can run alongside Phase 3; Phase 8 (Scraping) can run alongside Phase 7.
