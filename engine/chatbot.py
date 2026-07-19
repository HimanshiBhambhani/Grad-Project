"""
engine/chatbot.py - RAG Chatbot with full safety guardrails.

Safety: No buying advice, brand neutrality, prompt injection defense,
PII filtering, hallucination guard, confidence scoring, thin category warnings,
contradiction surfacing, suggested follow-ups.
"""

import logging
import re
from collections import Counter

import pandas as pd

import config
from engine.pii_filter import strip_pii

logger = logging.getLogger(__name__)

# --------------- Safety Patterns ---------------

_BUYING_ADVICE_RE = re.compile(
    r"\b("
    r"should\s+i\s+buy|would\s+you\s+recommend|suggest\s+(?:me|a|an|the)|"
    r"which\s+(?:is|one\s+is)\s+better|worth\s+buying|best\s+(?:platform|app|site)|"
    r"which\s+app|where\s+should\s+i|recommend\s+(?:me|a)|"
    r"should\s+i\s+(?:use|switch|order|shop|get|download|install)|"
    r"is\s+it\s+worth|advise\s+(?:me|on)|which\s+(?:should|would)|"
    r"better\s+to\s+(?:buy|use|order|shop)|good\s+(?:to\s+buy|for\s+buying)"
    r")\b",
    re.IGNORECASE,
)

_COMPARISON_RE = re.compile(
    r"\b("
    r"blinkit\s+vs|vs\s+blinkit|blinkit\s+or\s+zepto|zepto\s+or\s+blinkit|"
    r"blinkit\s+or\s+instamart|instamart\s+or\s+blinkit|"
    r"blinkit\s+or\s+amazon|amazon\s+or\s+blinkit|"
    r"blinkit\s+or\s+flipkart|blinkit\s+or\s+bigbasket|blinkit\s+or\s+jiomart|"
    r"zepto\s+vs|instamart\s+vs|which\s+platform|"
    r"better\s+than\s+(?:blinkit|zepto|instamart|amazon|swiggy)|"
    r"switch\s+to\s+(?:zepto|instamart|amazon|swiggy|flipkart|bigbasket)|"
    r"compared\s+to\s+(?:zepto|instamart|amazon|swiggy|flipkart)|"
    r"(?:zepto|instamart|swiggy|flipkart|bigbasket|jiomart)\s+(?:is\s+)?better|"
    r"which\s+(?:delivery|grocery|shopping)\s+app"
    r")\b",
    re.IGNORECASE,
)

_INJECTION_RE = re.compile(
    r"("
    r"ignore\s+(?:previous|all|your|above)\s+(?:instructions|rules|prompts|guidelines)|"
    r"pretend\s+(?:you\s+are|to\s+be)|"
    r"act\s+as\s+(?:a|an|if)|"
    r"new\s+instructions|forget\s+(?:your|all|previous)\s+(?:rules|instructions)|"
    r"system\s+prompt|jailbreak|DAN\s+mode|"
    r"bypass\s+(?:your|the|all)|override\s+(?:your|the|all)|"
    r"disregard\s+(?:your|the|all|previous)|you\s+are\s+now"
    r")",
    re.IGNORECASE,
)

THIN_CATEGORIES = {"pet": 16, "baby": 15, "intimate_personal": 19, "home_cleaning": 18}
MIN_EVIDENCE = 2

# --------------- Refusal Templates ---------------

_REFUSAL_BUY = (
    "\U0001f6ab I'm designed to analyze user feedback patterns, not provide "
    "purchasing recommendations. I can share what users report about {topic}.\n\n"
    "**Try asking:** \"{suggestion}\""
)

_REFUSAL_CMP = (
    "\U0001f6ab I can't provide platform comparisons or recommendations. "
    "Our dataset is primarily composed of Blinkit user reviews (~95%), making "
    "any cross-platform comparison statistically biased and misleading.\n\n"
    "I can help you explore what Blinkit users report about specific categories "
    "or friction patterns.\n\n"
    "**Try:** \"{suggestion}\""
)

_REFUSAL_INJ = (
    "\U0001f6ab I'm designed to analyze Blinkit user feedback only. "
    "I can't modify my operating parameters.\n\n"
    "**Try asking about:** user friction patterns, category insights, or recurring themes."
)

_REFUSAL_DATA = (
    "\U0001f534 I don't have sufficient data to answer this question reliably. "
    "Our dataset contains {n:,} reviews focused on non-grocery categories: "
    "electronics, beauty, pet care, baby products, pharmacy, home cleaning, "
    "and intimate care.\n\n"
    "**Try asking about one of these categories specifically.**"
)

# --------------- System Prompt ---------------

_SYS_PROMPT = """You are a senior product analyst embedded in Blinkit's Growth Team.
You have access to a curated corpus of {num_reviews} cleaned customer reviews spanning 7+ channels
(Play Store, App Store, Reddit, YouTube, HackerNews, PissedConsumer).

Your job is to answer questions about Blinkit's cross-shopping inertia problem.

## STRICT RULES
1. Ground every claim in evidence. Cite verbatim quotes: *"<quote>"* - <Source>, <Category>
2. Be specific. Use concrete examples from the data.
3. Organise: summary -> key themes (with evidence) -> implications.
4. Acknowledge gaps. If evidence is insufficient, say so.
5. Reference the 4 Friction Pillars when relevant:
   - Habitual Tunnel Vision (grocery-only habit lock)
   - Quality & Authenticity Risk (fake/counterfeit/storage fears)
   - Discovery Blind Spots (users don't know Blinkit sells X)
   - Immediate Value Disconnection (overpriced vs Amazon/Nykaa)
6. Stay concise. Aim for 200-400 words.
7. Never fabricate data.
8. NEVER give buying advice or recommend any platform.
9. NEVER compare Blinkit with Zepto, Instamart, Amazon, Flipkart, or any competitor.
   The dataset is Blinkit-centric; any comparison would be statistically biased.
   If asked to compare, REFUSE and explain the dataset bias.
10. NEVER suggest one platform is better than another.
11. If a user review mentions a competitor, you may quote it but NEVER draw
    comparative conclusions. Frame as: "Some users mentioned [competitor]"
    not as "[competitor] is better/worse."

## Corpus Statistics
- Total cleaned reviews: {num_reviews}
- Categories: {categories}
- Top friction pillar: {top_pillar}
- Source channels: {sources}
"""

_CTX_TEMPLATE = """## Retrieved Evidence ({n} most relevant reviews)

{evidence}

## Corpus-Level Statistics
- Category distribution: {cat_dist}
- Pillar distribution: {pil_dist}

## User Question
{question}

Answer using ONLY the evidence provided. Cite specific quotes.
Do NOT give buying advice or compare platforms."""


# --------------- Chatbot Class ---------------


class RAGChatbot:
    """RAG chatbot with full safety guardrails."""

    def __init__(self, df, faiss_index=None, faiss_metadata=None):
        self.df = df
        self.index = faiss_index
        self.metadata = faiss_metadata
        self.has_faiss = faiss_index is not None
        self.has_api_key = bool(config.GROQ_API_KEY)
        self.conversation_history = []

        self._n = len(df)
        self._cats = ", ".join(sorted(df["Target Category"].unique()))
        self._top_pil = (
            df["Friction Pillar"].value_counts().index[0]
            if "Friction Pillar" in df.columns else "N/A"
        )
        self._sources = (
            ", ".join(sorted(df["Source"].unique()))
            if "Source" in df.columns else "N/A"
        )
        self._cat_dist = dict(df["Target Category"].value_counts())
        self._pil_dist = (
            dict(df["Friction Pillar"].value_counts())
            if "Friction Pillar" in df.columns else {}
        )
        self._sys = _SYS_PROMPT.format(
            num_reviews=self._n,
            categories=self._cats,
            top_pillar=self._top_pil,
            sources=self._sources,
        )

    # --- Safety ---

    def _check_safety(self, q):
        if _INJECTION_RE.search(q):
            logger.warning("Injection blocked: %s", q[:80])
            return _REFUSAL_INJ
        if _COMPARISON_RE.search(q):
            logger.warning("Comparison blocked: %s", q[:80])
            return _REFUSAL_CMP.format(suggestion=self._alt(q, "cmp"))
        if _BUYING_ADVICE_RE.search(q):
            logger.warning("Buying advice blocked: %s", q[:80])
            return _REFUSAL_BUY.format(
                topic=self._topic(q), suggestion=self._alt(q, "buy")
            )
        return None

    @staticmethod
    def _topic(q):
        q = q.lower()
        mapping = {
            "electronics": "electronics quality and delivery",
            "beauty": "beauty product authenticity",
            "skincare": "skincare product quality",
            "pet": "pet care product availability",
            "baby": "baby product experiences",
            "pharmacy": "pharmacy product delivery",
            "cleaning": "home cleaning products",
            "intimate": "intimate care products",
        }
        for kw, t in mapping.items():
            if kw in q:
                return t
        return "product quality and user experiences on Blinkit"

    @staticmethod
    def _alt(q, kind):
        q = q.lower()
        if kind == "cmp":
            if "electronics" in q:
                return "What are the main quality concerns Blinkit users report about electronics?"
            if "beauty" in q or "skincare" in q:
                return "What do Blinkit users say about beauty product authenticity?"
            return "What are the most common frustrations Blinkit users report?"
        if "electronics" in q:
            return "What do users say about electronics quality on Blinkit?"
        if "beauty" in q or "skincare" in q:
            return "What are the top concerns about beauty products on Blinkit?"
        if "pet" in q:
            return "What do pet care buyers complain about most on Blinkit?"
        return "What are the top user frustrations with non-grocery categories on Blinkit?"

    # --- Confidence & Warnings ---

    @staticmethod
    def _confidence(evidence):
        n = len(evidence)
        if n >= 8:
            return {"level": "high", "emoji": "\U0001f7e2",
                    "description": f"High confidence - based on {n} supporting reviews"}
        if n >= 3:
            return {"level": "medium", "emoji": "\U0001f7e1",
                    "description": f"Medium confidence - based on {n} supporting reviews"}
        return {"level": "low", "emoji": "\U0001f534",
                "description": f"Low confidence - only {n} supporting review(s)"}

    @staticmethod
    def _thin_warning(evidence):
        cats = {r.get("Target Category", "") for r in evidence}
        w = []
        for c in cats:
            if c in THIN_CATEGORIES:
                w.append(
                    f"\u26a0\ufe0f **{c}** has limited corpus data "
                    f"({THIN_CATEGORIES[c]} total reviews). "
                    f"Insights may not be fully representative."
                )
        return "\n".join(w) if w else None

    @staticmethod
    def _contradictions(evidence):
        if len(evidence) < 4:
            return None
        pos_kw = {"good", "great", "excellent", "love", "best", "amazing",
                  "happy", "satisfied", "perfect", "awesome"}
        neg_kw = {"bad", "worst", "terrible", "horrible", "fake", "fraud",
                  "scam", "poor", "awful", "useless", "waste", "pathetic"}
        pos = sum(1 for r in evidence
                  if any(w in str(r.get("Raw Content", "")).lower() for w in pos_kw))
        neg = sum(1 for r in evidence
                  if any(w in str(r.get("Raw Content", "")).lower() for w in neg_kw))
        if pos >= 2 and neg >= 2:
            return (
                f"\u26a1 **Mixed signals detected:** {pos} reviews express "
                f"satisfaction while {neg} report frustration. "
                f"Both perspectives are reflected."
            )
        return None

    # --- Follow-ups ---

    def follow_ups(self, evidence):
        cats = Counter(r.get("Target Category", "") for r in evidence)
        pils = Counter(r.get("Friction Pillar", "") for r in evidence)
        tc = cats.most_common(1)[0][0] if cats else None
        tp = pils.most_common(1)[0][0] if pils else None
        s = []
        if tc and tc != "general":
            s.append(f"What are the specific quality concerns in the {tc} category?")
        pq = {
            "Quality & Authenticity Risk":
                "What evidence exists that users doubt product authenticity on Blinkit?",
            "Habitual Tunnel Vision":
                "How does the grocery habit lock-in manifest in user reviews?",
            "Discovery Blind Spots":
                "What evidence shows users are unaware of non-grocery categories?",
            "Immediate Value Disconnection":
                "How do users perceive Blinkit's pricing for non-grocery items?",
        }
        if tp and tp in pq:
            s.append(pq[tp])
        others = [c for c in cats if c != tc and c != "general"]
        if others:
            s.append(f"How does the {others[0]} category compare in user satisfaction?")
        if not s:
            s = [
                "Which friction pillar has the highest impact on cross-shopping?",
                "What are the top reasons users don't explore non-grocery categories?",
            ]
        return s[:3]

    # --- Retrieval ---

    def _faiss_retrieve(self, q, top_k=15):
        from engine import search
        return search(q, self.index, self.metadata, top_k=top_k)

    def _keyword_retrieve(self, q, top_k=15):
        stops = {
            "what", "does", "from", "that", "they", "this", "which", "more",
            "users", "user", "about", "with", "have", "their", "there", "been",
            "into", "will", "would", "could", "should", "than", "then", "when",
            "where", "your", "them", "some", "only", "also", "most", "very",
            "just", "like", "blinkit", "how", "why", "are", "the",
        }
        kws = [w for w in q.lower().split() if len(w) > 2 and w not in stops]
        if not kws:
            return self.df.sample(min(top_k, len(self.df))).to_dict("records")
        tmp = self.df.copy()
        tmp["_s"] = tmp["Raw Content"].apply(
            lambda t: sum(1 for k in kws if k in str(t).lower())
        )
        rel = tmp[tmp["_s"] > 0].nlargest(top_k, "_s")
        if rel.empty:
            rel = self.df.sample(min(top_k, len(self.df)))
        return rel.drop(columns=["_s"], errors="ignore").to_dict("records")

    def retrieve(self, q, top_k=15):
        if self.has_faiss:
            return self._faiss_retrieve(q, top_k)
        return self._keyword_retrieve(q, top_k)

    # --- Evidence Formatting ---

    @staticmethod
    def _fmt_evidence(reviews):
        lines = []
        for i, r in enumerate(reviews, 1):
            cat = r.get("Target Category", "N/A")
            pil = r.get("Friction Pillar", "N/A")
            src = r.get("Source", "N/A")
            txt = strip_pii(str(r.get("Raw Content", ""))[:500])
            lines.append(f"[{i}] Source: {src} | Category: {cat} | Pillar: {pil}")
            lines.append(f'    "{txt}"')
            lines.append("")
        return "\n".join(lines)

    # --- Main Ask ---

    def ask(self, question, top_k=15, brief=False):
        """Answer with full guardrails."""

        # Safety
        refusal = self._check_safety(question)
        if refusal:
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": refusal})
            return {
                "answer": refusal,
                "evidence_count": 0,
                "mode": "Safety refusal",
                "sources_used": [],
                "categories_in_evidence": [],
                "confidence": {
                    "level": "n/a", "emoji": "\U0001f6ab",
                    "description": "Question refused by safety filter",
                },
                "thin_warning": None,
                "contradiction_note": None,
                "follow_ups": [
                    "What are the most common frustrations Blinkit users report?",
                    "Which non-grocery categories have the most quality complaints?",
                    "What do users say about electronics on Blinkit?",
                ],
                "evidence": [],
                "was_refused": True,
            }

        # Retrieve
        ev = self.retrieve(question, top_k=top_k)

        # Hallucination guard
        if len(ev) < MIN_EVIDENCE:
            ans = _REFUSAL_DATA.format(n=self._n)
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": ans})
            return {
                "answer": ans,
                "evidence_count": len(ev),
                "mode": "Insufficient data",
                "sources_used": [],
                "categories_in_evidence": [],
                "confidence": {
                    "level": "low", "emoji": "\U0001f534",
                    "description": "Insufficient evidence",
                },
                "thin_warning": None,
                "contradiction_note": None,
                "follow_ups": [
                    "What are the top user frustrations with non-grocery categories?"
                ],
                "evidence": ev,
                "was_refused": True,
            }

        # Metadata
        ev_block = self._fmt_evidence(ev)
        conf = self._confidence(ev)
        tw = self._thin_warning(ev)
        cn = self._contradictions(ev)
        fu = self.follow_ups(ev)

        cd = ", ".join(
            f"{k}: {v}"
            for k, v in sorted(self._cat_dist.items(), key=lambda x: -x[1])[:5]
        )
        pd_str = ", ".join(
            f"{k}: {v}"
            for k, v in sorted(self._pil_dist.items(), key=lambda x: -x[1])
        )

        # Generate
        if self.has_api_key:
            ans = self._gen_ai(question, ev_block, cd, pd_str, len(ev), brief)
        else:
            ans = self._gen_offline(question, ev)

        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": ans})

        if self.has_api_key and self.has_faiss:
            mode = "AI (Groq Llama 3.3 + FAISS)"
        elif self.has_api_key:
            mode = "AI (Groq Llama 3.3 + keyword)"
        else:
            mode = "Offline (keyword retrieval)"

        return {
            "answer": ans,
            "evidence_count": len(ev),
            "mode": mode,
            "sources_used": list({r.get("Source", "") for r in ev}),
            "categories_in_evidence": list({r.get("Target Category", "") for r in ev}),
            "confidence": conf,
            "thin_warning": tw,
            "contradiction_note": cn,
            "follow_ups": fu,
            "evidence": ev,
            "was_refused": False,
        }

    def _gen_ai(self, question, ev_block, cd, pd_str, n, brief=False):
        from groq import Groq

        client = Groq(api_key=config.GROQ_API_KEY)
        extra = (
            "\n\nIMPORTANT: Keep the answer brief - 2-3 sentences with key evidence only."
            if brief else ""
        )
        user_msg = _CTX_TEMPLATE.format(
            n=n, evidence=ev_block, cat_dist=cd, pil_dist=pd_str, question=question,
        ) + extra

        msgs = [{"role": "system", "content": self._sys}]
        msgs.extend(self.conversation_history[-6:])
        msgs.append({"role": "user", "content": user_msg})

        try:
            resp = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=msgs,
                temperature=0.3,
                max_tokens=800 if brief else 1500,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error("AI generation failed: %s", e)
            return self._gen_offline(question, [])

    def _gen_offline(self, question, evidence):
        if not evidence:
            return (
                "I couldn't find relevant evidence. Try rephrasing, or ensure "
                "FAISS index is built and GROQ_API_KEY is set."
            )
        cats = Counter(r.get("Target Category", "") for r in evidence)
        pils = Counter(r.get("Friction Pillar", "") for r in evidence)
        tc = cats.most_common(1)[0][0] if cats else "N/A"
        tp = pils.most_common(1)[0][0] if pils else "N/A"
        quotes = []
        for r in evidence[:5]:
            txt = strip_pii(str(r.get("Raw Content", ""))[:200])
            src = r.get("Source", "")
            cat = r.get("Target Category", "")
            if txt.strip():
                quotes.append(f'*"{txt}"* - {src}, {cat}')
        qb = "\n".join(f"- {q}" for q in quotes) if quotes else "No quotes."
        return (
            f"**Based on {len(evidence)} relevant reviews:**\n\n"
            f"The dominant pattern relates to **{tp}**, primarily in **{tc}**.\n\n"
            f"**Category breakdown:** "
            f"{', '.join(f'{k} ({v})' for k, v in cats.most_common(5))}\n\n"
            f"**Pillar breakdown:** "
            f"{', '.join(f'{k} ({v})' for k, v in pils.most_common())}\n\n"
            f"**Representative quotes:**\n{qb}\n\n"
            f"*Note: Offline analysis. Set GROQ_API_KEY for AI-powered answers.*"
        )

    def clear_history(self):
        self.conversation_history.clear()

    def export_history(self):
        if not self.conversation_history:
            return "No conversation history to export."
        lines = ["=" * 60, "BLINKIT DISCOVERY ENGINE - CHAT EXPORT", "=" * 60, ""]
        for i in range(0, len(self.conversation_history), 2):
            q = self.conversation_history[i] if i < len(self.conversation_history) else None
            a = (
                self.conversation_history[i + 1]
                if i + 1 < len(self.conversation_history) else None
            )
            if q:
                lines.extend([f"USER: {q['content']}", ""])
            if a:
                lines.extend([f"ASSISTANT: {a['content']}", "", "-" * 40, ""])
        lines.extend(["=" * 60, "END OF EXPORT"])
        return "\n".join(lines)

    @property
    def mode_label(self):
        if self.has_api_key and self.has_faiss:
            return "Full RAG (FAISS + Groq Llama 3.3 70B)"
        if self.has_api_key:
            return "AI + Keyword Retrieval (Groq, no FAISS index)"
        if self.has_faiss:
            return "FAISS Retrieval + Offline Generation (no API key)"
        return "Offline Mode (keyword retrieval + rule-based)"
