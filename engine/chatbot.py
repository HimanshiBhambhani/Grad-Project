"""
engine/chatbot.py — RAG Chatbot for answering evaluator questions.

Uses the FAISS vector index to retrieve relevant reviews, then feeds them as
grounded context to GPT-4o-mini for evidence-backed conversational answers.
Falls back to offline keyword-based retrieval when no FAISS index is available.
"""

import json
import logging
from collections import Counter

import pandas as pd

import config

logger = logging.getLogger(__name__)

# ─────────────────── System Prompt ───────────────────

CHATBOT_SYSTEM_PROMPT = """You are a senior product analyst embedded in Blinkit's Growth Team.
You have access to a curated corpus of {num_reviews} cleaned customer reviews spanning 7+ channels
(Play Store, App Store, Reddit, YouTube, HackerNews, PissedConsumer).

Your job is to answer questions from an evaluator about Blinkit's cross-shopping inertia problem
— why users stick to grocery staples and rarely purchase from expansion categories like Electronics,
Beauty/Skincare, Pet Care, Baby, Pharmacy, Home Cleaning, and Intimate/Personal Care.

## Rules
1. **Ground every claim in evidence.** Cite verbatim quotes from the provided reviews.
   Use the format: *"<quote>"* — <Source>, <Category>
2. **Be specific.** Replace vague phrases with concrete examples from the data.
3. **Organise answers** with clear structure: summary → key themes (with evidence) → implications.
4. **Acknowledge gaps.** If the retrieved evidence doesn't fully answer the question, say so.
5. **Reference the 4 Friction Pillars** when relevant:
   - Habitual Tunnel Vision (grocery-only habit lock)
   - Quality & Authenticity Risk (fake/counterfeit/storage fears)
   - Discovery Blind Spots (users don't know Blinkit sells X)
   - Immediate Value Disconnection (overpriced vs Amazon/Nykaa)
6. **Stay concise.** Evaluators value density over length. Aim for 200-400 words unless the question demands more.
7. **Never fabricate data.** If no evidence exists for a claim, don't make one.

## Corpus Statistics
- Total cleaned reviews: {num_reviews}
- Categories: {categories}
- Top friction pillar: {top_pillar}
- Source channels: {sources}
"""

CONTEXT_TEMPLATE = """## Retrieved Evidence ({num_results} most relevant reviews)

{evidence_block}

## Corpus-Level Statistics
- Category distribution: {category_dist}
- Pillar distribution: {pillar_dist}

## User Question
{question}

Answer the question above using ONLY the evidence provided. Cite specific quotes."""


# ─────────────────── Chatbot Class ───────────────────


class RAGChatbot:
    """
    Retrieval-Augmented Generation chatbot backed by the FAISS vector index
    over cleaned Blinkit reviews.
    """

    def __init__(self, df: pd.DataFrame, faiss_index=None, faiss_metadata=None):
        """
        Args:
            df: The full cleaned DataFrame (for stats and offline fallback).
            faiss_index: A loaded FAISS index (or None for offline mode).
            faiss_metadata: Corresponding metadata list (or None).
        """
        self.df = df
        self.index = faiss_index
        self.metadata = faiss_metadata
        self.has_faiss = faiss_index is not None
        self.has_openai = bool(config.OPENAI_API_KEY)
        self.conversation_history: list[dict] = []

        # Precompute corpus stats
        self._num_reviews = len(df)
        self._categories = ", ".join(sorted(df["Target Category"].unique()))
        self._top_pillar = df["Friction Pillar"].value_counts().index[0] if "Friction Pillar" in df.columns else "N/A"
        self._sources = ", ".join(sorted(df["Source"].unique())) if "Source" in df.columns else "N/A"
        self._category_dist = dict(df["Target Category"].value_counts())
        self._pillar_dist = dict(df["Friction Pillar"].value_counts()) if "Friction Pillar" in df.columns else {}

        self._system_prompt = CHATBOT_SYSTEM_PROMPT.format(
            num_reviews=self._num_reviews,
            categories=self._categories,
            top_pillar=self._top_pillar,
            sources=self._sources,
        )

    # ─────────────────── Retrieval ───────────────────

    def _retrieve_faiss(self, query: str, top_k: int = 15) -> list[dict]:
        """Semantic retrieval via FAISS."""
        from engine import search
        return search(query, self.index, self.metadata, top_k=top_k)

    def _retrieve_keyword(self, query: str, top_k: int = 15) -> list[dict]:
        """Offline keyword-overlap retrieval."""
        q_lower = query.lower()
        stop_words = {
            "what", "does", "from", "that", "they", "this", "which", "more",
            "users", "user", "about", "with", "have", "their", "there", "been",
            "into", "will", "would", "could", "should", "than", "then", "when",
            "where", "your", "them", "some", "only", "also", "most", "very",
            "just", "like", "blinkit", "how", "why", "are", "the",
        }
        keywords = [w for w in q_lower.split() if len(w) > 2 and w not in stop_words]

        if not keywords:
            # Fallback: random sample
            sample = self.df.sample(min(top_k, len(self.df)))
            return sample.to_dict("records")

        def score(text: str) -> int:
            text_lower = str(text).lower()
            return sum(1 for kw in keywords if kw in text_lower)

        scored = self.df.copy()
        scored["_score"] = scored["Raw Content"].apply(score)
        relevant = scored[scored["_score"] > 0].nlargest(top_k, "_score")

        if relevant.empty:
            relevant = self.df.sample(min(top_k, len(self.df)))

        return relevant.drop(columns=["_score"], errors="ignore").to_dict("records")

    def retrieve(self, query: str, top_k: int = 15) -> list[dict]:
        """Retrieve the most relevant reviews for a query."""
        if self.has_faiss:
            return self._retrieve_faiss(query, top_k)
        return self._retrieve_keyword(query, top_k)

    # ─────────────────── Evidence Formatting ───────────────────

    @staticmethod
    def _format_evidence(reviews: list[dict]) -> str:
        """Format retrieved reviews into a numbered evidence block."""
        lines = []
        for i, r in enumerate(reviews, 1):
            cat = r.get("Target Category", "N/A")
            pillar = r.get("Friction Pillar", "N/A")
            source = r.get("Source", "N/A")
            text = str(r.get("Raw Content", ""))[:500]
            lines.append(f"[{i}] Source: {source} | Category: {cat} | Pillar: {pillar}")
            lines.append(f'    "{text}"')
            lines.append("")
        return "\n".join(lines)

    # ─────────────────── Answer Generation ───────────────────

    def ask(self, question: str, top_k: int = 15) -> dict:
        """
        Answer an evaluator question using RAG.

        Returns:
            dict with keys: answer, evidence_count, mode, sources_used
        """
        # 1. Retrieve
        evidence = self.retrieve(question, top_k=top_k)
        evidence_block = self._format_evidence(evidence)

        # 2. Build stats
        cat_dist_str = ", ".join(f"{k}: {v}" for k, v in sorted(self._category_dist.items(), key=lambda x: -x[1])[:5])
        pillar_dist_str = ", ".join(f"{k}: {v}" for k, v in sorted(self._pillar_dist.items(), key=lambda x: -x[1]))

        # 3. Generate answer
        if self.has_openai:
            answer = self._generate_ai(question, evidence_block, cat_dist_str, pillar_dist_str, len(evidence))
        else:
            answer = self._generate_offline(question, evidence)

        # 4. Track conversation
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "evidence_count": len(evidence),
            "mode": "AI (GPT-4o-mini + FAISS)" if (self.has_openai and self.has_faiss)
                    else "AI (GPT-4o-mini + keyword)" if self.has_openai
                    else "Offline (keyword retrieval)",
            "sources_used": list(set(r.get("Source", "") for r in evidence)),
            "categories_in_evidence": list(set(r.get("Target Category", "") for r in evidence)),
            "evidence": evidence,
        }

    def _generate_ai(
        self,
        question: str,
        evidence_block: str,
        cat_dist_str: str,
        pillar_dist_str: str,
        num_results: int,
    ) -> str:
        """Generate answer using GPT-4o-mini with RAG context."""
        from openai import OpenAI

        client = OpenAI(api_key=config.OPENAI_API_KEY)

        user_msg = CONTEXT_TEMPLATE.format(
            num_results=num_results,
            evidence_block=evidence_block,
            category_dist=cat_dist_str,
            pillar_dist=pillar_dist_str,
            question=question,
        )

        # Build messages: system + last 6 conversation turns for continuity + current
        messages = [{"role": "system", "content": self._system_prompt}]

        # Add recent conversation history (last 3 Q&A pairs = 6 messages)
        history_window = self.conversation_history[-6:]
        messages.extend(history_window)

        messages.append({"role": "user", "content": user_msg})

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=1500,
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error("Chatbot AI generation failed: %s", e)
            return self._generate_offline(question, [])

    def _generate_offline(self, question: str, evidence: list[dict]) -> str:
        """Generate a structured offline answer from retrieved evidence."""
        if not evidence:
            return (
                "I couldn't find relevant evidence in the corpus for this question. "
                "Try rephrasing, or ensure the FAISS index is built and OPENAI_API_KEY is set "
                "for semantic search and AI-powered answers."
            )

        # Aggregate stats from evidence
        cats = Counter(r.get("Target Category", "") for r in evidence)
        pillars = Counter(r.get("Friction Pillar", "") for r in evidence)
        top_cat = cats.most_common(1)[0][0] if cats else "N/A"
        top_pillar = pillars.most_common(1)[0][0] if pillars else "N/A"

        # Pick top quotes
        quotes = []
        for r in evidence[:5]:
            text = str(r.get("Raw Content", ""))[:200]
            source = r.get("Source", "")
            cat = r.get("Target Category", "")
            if text.strip():
                quotes.append(f'*"{text}"* — {source}, {cat}')

        quotes_block = "\n".join(f"- {q}" for q in quotes) if quotes else "No quotes available."

        return (
            f"**Based on {len(evidence)} relevant reviews:**\n\n"
            f"The dominant pattern relates to **{top_pillar}**, primarily in the "
            f"**{top_cat}** category.\n\n"
            f"**Category breakdown:** {', '.join(f'{k} ({v})' for k, v in cats.most_common(5))}\n\n"
            f"**Pillar breakdown:** {', '.join(f'{k} ({v})' for k, v in pillars.most_common())}\n\n"
            f"**Representative quotes:**\n{quotes_block}\n\n"
            f"*Note: This is an offline analysis. Set OPENAI_API_KEY for AI-powered, "
            f"nuanced answers with deeper theme extraction.*"
        )

    def clear_history(self):
        """Reset conversation history."""
        self.conversation_history.clear()

    @property
    def mode_label(self) -> str:
        """Human-readable label for the current operating mode."""
        if self.has_openai and self.has_faiss:
            return "Full RAG (FAISS + GPT-4o-mini)"
        elif self.has_openai:
            return "AI + Keyword Retrieval (no FAISS index)"
        elif self.has_faiss:
            return "FAISS Retrieval + Offline Generation (no API key)"
        else:
            return "Offline Mode (keyword retrieval + rule-based)"
