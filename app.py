"""
app.py — Streamlit Discovery Engine Dashboard.

Interactive dashboard for exploring Blinkit customer feedback insights.
Answers strategic questions using semantic search + AI insight generation.

Usage:
    streamlit run app.py
"""

import os
import sys
from pathlib import Path

# ── SSL & HF fixes (must run before any HuggingFace / model imports) ──
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from engine.themes import identify_themes, get_theme_summary, get_cross_theme_patterns
from engine.insights import STRATEGIC_QUESTIONS, generate_insight_offline
from engine.chatbot import RAGChatbot

# ─────────────────── Page Config ───────────────────
st.set_page_config(
    page_title="Blinkit Discovery Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────── Data Loading ───────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(str(config.CLEAN_OUTPUT_CSV))
    df = identify_themes(df)
    return df

@st.cache_data
def get_themes(df_hash):
    df = load_data()
    return get_theme_summary(df)

@st.cache_data
def get_patterns(df_hash):
    df = load_data()
    return get_cross_theme_patterns(df)


# ─────────────────── Sidebar ───────────────────
st.sidebar.title("🔍 Blinkit Discovery Engine")
st.sidebar.markdown("*Cross-Shopping Inertia Analysis*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Dashboard Overview",
        "🔬 Theme Explorer",
        "❓ Strategic Questions",
        "🔎 Search & Explore",
        "💬 RAG Chatbot",
        "✅ Validation & Methodology",
    ],
)

# Load data
try:
    df = load_data()
except FileNotFoundError:
    st.error("No cleaned data found. Run `python main.py` first to generate `Output/blinkit_clean_data.csv`.")
    st.stop()


# ═══════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD OVERVIEW
# ═══════════════════════════════════════════════════════════
if page == "📊 Dashboard Overview":
    st.title("📊 Dashboard Overview")
    st.markdown(f"**{len(df):,} cleaned reviews** from multi-channel sources, filtered through a 3-stage pipeline.")

    # ── Key Metrics Row ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", f"{len(df):,}")
    col2.metric("Categories", df["Target Category"].nunique())
    col3.metric("Friction Pillars", df["Friction Pillar"].nunique())
    col4.metric("Themes Identified", df["Primary Theme"].nunique())

    st.divider()

    # ── Distribution Charts ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Category Distribution")
        cat_counts = df["Target Category"].value_counts()
        st.bar_chart(cat_counts, horizontal=True)

    with col_right:
        st.subheader("Friction Pillar Distribution")
        pillar_counts = df["Friction Pillar"].value_counts()
        st.bar_chart(pillar_counts, horizontal=True)

    st.divider()

    # ── Theme Distribution ──
    st.subheader("Theme Distribution")
    theme_counts = df["Primary Theme"].value_counts()
    st.bar_chart(theme_counts, horizontal=True)

    st.divider()

    # ── Data Pipeline Funnel ──
    st.subheader("Data Pipeline Funnel")
    funnel_data = {
        "Stage": [
            "Raw Input (both CSVs)",
            "After Deduplication",
            "Stage 1: Noise Filter",
            "Stage 2: Rating Filter",
            "Stage 3: Category Isolation",
            "Final Output",
        ],
        "Rows": [56530, 28274, 27589, 26680, 1808, len(df)],
    }
    st.dataframe(pd.DataFrame(funnel_data), width='stretch', hide_index=True)


# ═══════════════════════════════════════════════════════════
# PAGE 2: THEME EXPLORER
# ═══════════════════════════════════════════════════════════
elif page == "🔬 Theme Explorer":
    st.title("🔬 Theme Explorer")
    st.markdown("Emergent themes identified through keyword pattern matching across all reviews.")

    themes = get_theme_summary(df)

    for i, theme in enumerate(themes):
        with st.expander(f"**{theme['theme']}** — {theme['count']} reviews ({theme['percentage']}%)", expanded=(i < 3)):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Categories affected:**")
                for cat, count in sorted(theme["categories"].items(), key=lambda x: -x[1]):
                    st.markdown(f"- `{cat}`: {count}")

            with col2:
                st.markdown("**Friction pillars:**")
                for pillar, count in sorted(theme["pillars"].items(), key=lambda x: -x[1]):
                    st.markdown(f"- {pillar}: {count}")

            st.markdown("**Representative quotes:**")
            for q in theme["representative_quotes"]:
                st.markdown(f"> _{q}_")

    # ── Cross-Pattern Analysis ──
    st.divider()
    st.subheader("Cross-Pattern: Theme × Category Matrix")

    patterns = get_cross_theme_patterns(df)

    # Build matrix
    all_themes = sorted(df["Primary Theme"].unique())
    all_cats = sorted(df["Target Category"].unique())

    matrix_data = []
    for cat in all_cats:
        row = {"Category": cat}
        cat_themes = patterns["theme_by_category"].get(cat, {})
        for theme in all_themes:
            row[theme] = cat_themes.get(theme, 0)
        matrix_data.append(row)

    matrix_df = pd.DataFrame(matrix_data).set_index("Category")
    st.dataframe(matrix_df, width='stretch')


# ═══════════════════════════════════════════════════════════
# PAGE 3: STRATEGIC QUESTIONS
# ═══════════════════════════════════════════════════════════
elif page == "❓ Strategic Questions":
    st.title("❓ Strategic Questions")
    st.markdown("Answer key growth questions using evidence from customer reviews.")

    selected_q = st.selectbox("Select a strategic question:", STRATEGIC_QUESTIONS)

    # Filter method
    filter_method = st.radio("Analysis method:", ["Keyword Relevance (offline)", "Semantic Search (requires FAISS + API keys)"], horizontal=True)

    if st.button("Generate Insight", type="primary"):
        with st.spinner("Analyzing evidence..."):

            if filter_method == "Keyword Relevance (offline)":
                # Simple keyword relevance filtering
                q_lower = selected_q.lower()
                keywords = [w for w in q_lower.split() if len(w) > 3 and w not in {"what", "does", "from", "that", "they", "this", "which", "more", "users", "user"}]

                # Score each review by keyword overlap
                def score_review(text):
                    text_lower = str(text).lower()
                    return sum(1 for kw in keywords if kw in text_lower)

                df_copy = df.copy()
                df_copy["_relevance"] = df_copy["Raw Content"].apply(score_review)
                relevant = df_copy[df_copy["_relevance"] > 0].nlargest(20, "_relevance")

                if relevant.empty:
                    # Fallback: use all data
                    relevant = df.sample(min(20, len(df)))

                reviews_list = relevant.to_dict("records")
                insight = generate_insight_offline(selected_q, reviews_list)

            else:
                # FAISS semantic search
                try:
                    from engine import load_index, search
                    index, metadata = load_index()
                    if index is None:
                        st.error("FAISS index not built. Run: `python -c \"from engine import build_index; import pandas as pd; build_index(pd.read_csv('Output/blinkit_clean_data.csv'))\"`")
                        st.stop()
                    reviews_list = search(selected_q, index, metadata, top_k=20)

                    if config.GROQ_API_KEY:
                        from engine.insights import generate_insight
                        insight = generate_insight(selected_q, reviews_list)
                    else:
                        insight = generate_insight_offline(selected_q, reviews_list)
                except Exception as e:
                    st.error(f"Semantic search failed: {e}")
                    st.stop()

        # ── Display Insight ──
        st.divider()

        st.subheader("Executive Summary")
        st.info(insight.get("executive_summary", ""))

        st.subheader("Key Insight")
        st.success(insight.get("key_insight", ""))

        # Themes
        st.subheader(f"Identified Themes ({len(insight.get('themes', []))})")
        for theme in insight.get("themes", []):
            with st.expander(f"**{theme.get('theme_name', 'Theme')}** — {theme.get('evidence_count', 0)} reviews"):
                st.markdown(theme.get("description", ""))
                st.markdown(f"**Friction Pillar:** {theme.get('friction_pillar', 'N/A')}")
                st.markdown(f"**Categories:** {', '.join(theme.get('categories_affected', []))}")
                if theme.get("representative_quotes"):
                    st.markdown("**Evidence:**")
                    for q in theme["representative_quotes"]:
                        st.markdown(f"> _{q[:300]}_")

        # Recommendations
        st.subheader("Actionable Recommendations")
        for rec in insight.get("actionable_recommendations", []):
            st.markdown(f"- {rec}")

        # Metadata
        col1, col2 = st.columns(2)
        col1.metric("Confidence Level", insight.get("confidence_level", "N/A").upper())
        col2.markdown(f"**Evidence Gap:** {insight.get('evidence_gap', 'None identified')}")


# ═══════════════════════════════════════════════════════════
# PAGE 4: SEARCH & EXPLORE
# ═══════════════════════════════════════════════════════════
elif page == "🔎 Search & Explore":
    st.title("🔎 Search & Explore Reviews")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        cat_filter = st.multiselect("Category", options=sorted(df["Target Category"].unique()), default=[])
    with col2:
        pillar_filter = st.multiselect("Friction Pillar", options=sorted(df["Friction Pillar"].unique()), default=[])
    with col3:
        theme_filter = st.multiselect("Theme", options=sorted(df["Primary Theme"].unique()), default=[])

    search_text = st.text_input("Search text (keyword):", placeholder="e.g., fake earbuds warranty")

    # Apply filters
    filtered = df.copy()
    if cat_filter:
        filtered = filtered[filtered["Target Category"].isin(cat_filter)]
    if pillar_filter:
        filtered = filtered[filtered["Friction Pillar"].isin(pillar_filter)]
    if theme_filter:
        filtered = filtered[filtered["Primary Theme"].isin(theme_filter)]
    if search_text:
        filtered = filtered[filtered["Raw Content"].str.contains(search_text, case=False, na=False)]

    st.markdown(f"**{len(filtered):,} reviews** matching filters")

    # Display
    display_cols = ["Target Category", "Friction Pillar", "Primary Theme", "Raw Content", "Source"]
    st.dataframe(
        filtered[display_cols].head(100),
        width='stretch',
        hide_index=True,
        column_config={
            "Raw Content": st.column_config.TextColumn("Review", width="large"),
        },
    )

    # Download
    csv_data = filtered[display_cols].to_csv(index=False)
    st.download_button("Download filtered data", csv_data, "filtered_reviews.csv", "text/csv")


# ═══════════════════════════════════════════════════════════
# PAGE 5: RAG CHATBOT
# ═══════════════════════════════════════════════════════════
elif page == "💬 RAG Chatbot":
    st.title("💬 RAG Chatbot")
    st.markdown(
        "Ask any question about Blinkit's cross-shopping inertia. "
        "The chatbot retrieves relevant reviews and generates evidence-backed answers."
    )

    # ── Initialise chatbot in session state ──
    if "chatbot" not in st.session_state:
        # Try loading FAISS index
        faiss_index, faiss_meta = None, None
        try:
            from engine import load_index
            faiss_index, faiss_meta = load_index()
        except Exception:
            pass
        st.session_state.chatbot = RAGChatbot(df, faiss_index, faiss_meta)

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    bot = st.session_state.chatbot

    # ── Mode indicator ──
    mode_colors = {
        "Full RAG": "🟢", "AI + Keyword": "🟡", "FAISS Retrieval": "🟡", "Offline": "🟠"
    }
    mode_key = next((k for k in mode_colors if k in bot.mode_label), "Offline")
    st.sidebar.markdown(f"{mode_colors[mode_key]} **Mode:** {bot.mode_label}")

    # ── Suggested questions ──
    with st.expander("💡 Suggested questions for evaluators", expanded=False):
        suggestions = [
            "What are the top reasons users don't buy electronics on Blinkit?",
            "How does trust in product authenticity vary across categories?",
            "What evidence exists that users are unaware of non-grocery categories?",
            "How do users compare Blinkit's prices to Amazon and Nykaa?",
            "What are the biggest quality concerns for beauty/skincare products?",
            "Which friction pillar has the highest impact on cross-shopping?",
            "What do pet care buyers complain about most?",
            "How does the grocery habit lock-in manifest in user reviews?",
        ]
        for s in suggestions:
            if st.button(s, key=f"suggest_{hash(s)}", width='stretch'):
                st.session_state._pending_question = s

    # ── Chat history display ──
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                meta = msg["meta"]
                with st.expander(f"📎 Evidence details — {meta['evidence_count']} reviews retrieved", expanded=False):
                    st.markdown(f"**Mode:** {meta['mode']}")
                    st.markdown(f"**Sources:** {', '.join(meta['sources_used'])}")
                    st.markdown(f"**Categories:** {', '.join(meta['categories_in_evidence'])}")

    # ── Chat input ──
    pending = st.session_state.pop("_pending_question", None)
    user_input = st.chat_input("Ask a question about Blinkit's cross-shopping data...")
    question = pending or user_input

    if question:
        # Display user message
        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence & generating answer..."):
                result = bot.ask(question, top_k=15)

            st.markdown(result["answer"])

            with st.expander(f"📎 Evidence details — {result['evidence_count']} reviews retrieved", expanded=False):
                st.markdown(f"**Mode:** {result['mode']}")
                st.markdown(f"**Sources:** {', '.join(result['sources_used'])}")
                st.markdown(f"**Categories:** {', '.join(result['categories_in_evidence'])}")

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": result["answer"],
            "meta": {
                "evidence_count": result["evidence_count"],
                "mode": result["mode"],
                "sources_used": result["sources_used"],
                "categories_in_evidence": result["categories_in_evidence"],
            },
        })

    # ── Clear chat button ──
    if st.session_state.chat_messages:
        if st.sidebar.button("🗑️ Clear Chat History"):
            st.session_state.chat_messages.clear()
            bot.clear_history()
            st.rerun()


# ═══════════════════════════════════════════════════════════
# PAGE 6: VALIDATION & METHODOLOGY
# ═══════════════════════════════════════════════════════════
elif page == "✅ Validation & Methodology":
    st.title("✅ Validation & Methodology")

    st.subheader("1. How the Workflow Gathers & Analyzes Data")
    st.markdown("""
    **Dual-Pathway Ingestion Architecture:**

    | Pathway | Source | Method |
    |---------|--------|--------|
    | **A: Live Stream** | Google Play Store + Apple App Store | Programmatic API scraping (`google-play-scraper`, `app-store-scraper`) |
    | **B: Historical** | CSV dumps from Play Store, App Store, Reddit, YouTube, HackerNews, PissedConsumer | Pre-collected by AI agents (Claude, Perplexity) |
    | **B: Curated** | 32 Reddit thread URLs | Manually curated high-signal threads |
    | **B: Pre-classified** | 19 Reddit threads with expert labels | Already mapped to categories + friction pillars |

    **Pipeline:** Raw data → 3-stage filter → Theme identification → Pillar classification → Output schema
    """)

    st.subheader("2. How Themes Are Identified")
    st.markdown("""
    **Two complementary approaches:**

    1. **Rule-Based Theme Detection** (offline, deterministic):
       - 10 predefined theme patterns with regex keyword lists
       - Each review matched against all patterns; highest-match theme assigned
       - Themes: Counterfeit/Fake Products, Warranty Anxiety, Dark Store Concerns, Price Premium, Category Unawareness, Search Friction, Grocery Habit Lock, Support Failures, Quality Issues, Trust/Verification

    2. **AI-Powered Semantic Clustering** (with Groq + local embeddings):
       - Reviews embedded via `all-MiniLM-L6-v2` sentence-transformers model (384 dims, local)
       - FAISS index enables semantic search for any natural language query
       - Groq Llama 3.3 70B generates structured theme analysis grounded in retrieved evidence
    """)

    st.subheader("3. How Insights Are Generated")
    st.markdown("""
    **For each strategic question:**

    1. **Evidence retrieval**: Keyword relevance scoring OR semantic FAISS search to find the 15-20 most relevant reviews
    2. **Theme extraction**: Identify 3-6 distinct patterns in the evidence
    3. **Grounded analysis**: Every theme backed by verbatim customer quotes
    4. **Actionable output**: Specific PM recommendations tied to each friction pattern

    **Output structure:** Executive summary → Identified themes (with quotes) → Key non-obvious insight → Actionable recommendations → Confidence level → Evidence gaps
    """)

    st.subheader("4. How Insights Are Validated")
    st.markdown("""
    **Multi-layer validation framework:**

    | Validation Method | Description |
    |-------------------|-------------|
    | **Data Quality Gate** | 3-stage filter removes logistics noise, 5-star praise, and grocery complaints before any analysis |
    | **Source Triangulation** | Reviews from 7+ independent channels (Play Store, App Store, Reddit, YouTube, HackerNews, PissedConsumer) — same patterns across sources = higher confidence |
    | **Quote Grounding** | Every theme must cite verbatim user quotes — no unsubstantiated claims |
    | **Confidence Scoring** | Each insight rated high/medium/low based on evidence density |
    | **Evidence Gap Disclosure** | Explicitly states what data is missing to strengthen each finding |
    | **Cross-Pattern Validation** | Theme × Category matrix reveals whether patterns are category-specific or systemic |
    | **Pillar Consistency** | Themes mapped to the 4 friction pillars to ensure strategic coherence |
    """)

    st.divider()

    st.subheader("Pipeline Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        - **Raw input rows:** 56,530 (2 CSV files)
        - **After dedup:** 28,274
        - **Stage 1 (noise):** 27,589 (-685)
        - **Stage 2 (rating):** 26,680 (-909)
        - **Stage 3 (category):** 1,808 (-24,872)
        """)
    with col2:
        st.markdown(f"""
        - **Source channels:** {df['Source'].nunique()}
        - **Categories:** {df['Target Category'].nunique()}
        - **Friction pillars:** {df['Friction Pillar'].nunique()}
        - **Themes identified:** {df['Primary Theme'].nunique()}
        - **Retention rate:** {len(df)/28274*100:.1f}%
        """)
