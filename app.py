"""
=============================================================
  FashionIQ – AI-Powered Business Intelligence Agent
  Course : MGNM521 – Disruptive Technologies for Business
  CA1    : Chatbot + Sentiment Engine
  Domain : Retail / Fashion (Myntra + Customer Sentiment)
=============================================================
  Tech Stack:
    • Python       – Pandas, TextBlob, NLTK
    • Streamlit    – Multi-page App / Dashboard
    • LangChain    – Agentic AI layer
    • Groq LLaMA-3 – Primary LLM  (free tier)
    • Gemini 1.5   – Fallback LLM (free tier)
    • Plotly       – Interactive visualisations
=============================================================
  Run:  streamlit run app.py
=============================================================
"""

# ── Standard library ──────────────────────────────────────
import os
import re
import textwrap
import json
import urllib.request
from datetime import datetime
from collections import Counter

# ── Data / NLP ────────────────────────────────────────────
import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk

# ── Visualisation ─────────────────────────────────────────
import plotly.express as px
import plotly.graph_objects as go

# ── Streamlit ─────────────────────────────────────────────
import streamlit as st

# ── LangChain + LLMs ──────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_classic.memory import ConversationBufferWindowMemory

# ─────────────────────────────────────────────────────────
# ★  API KEYS  — edit ONLY these two lines
#    Groq  : https://console.groq.com       (free)
#    Gemini: https://aistudio.google.com    (free)
# ─────────────────────────────────────────────────────────
GROQ_API_KEY   = "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"
GEMINI_API_KEY = "AIzaSyDiodDh7LUrwITTqetyHNqGKbraOAJ8eAM"
# ─────────────────────────────────────────────────────────

SENTIMENT_PATH = "Customer_Sentiment.csv"
CATALOG_PATH   = "myntra_products_catalog.csv"

# ─────────────────────────────────────────────────────────
# 0.  PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FashionIQ – BI Agent",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# 1.  NLTK DOWNLOAD (silent)
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def download_nltk():
    for pkg in ["punkt", "stopwords", "wordnet"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

download_nltk()

# ─────────────────────────────────────────────────────────
# 2.  CATEGORY INFERENCE  (module-level — fixes UnboundLocalError)
# ─────────────────────────────────────────────────────────
def infer_cat(name: str) -> str:
    n = str(name).lower()
    if re.search(r"shirt|t-shirt|top|blouse|polo|tee", n):               return "Tops & Shirts"
    if re.search(r"jean|trouser|pant|chino|legging", n):                  return "Jeans & Trousers"
    if re.search(r"dress|kurta|kurti|gown|saree|salwar|anarkali", n):     return "Dresses & Kurtas"
    if re.search(r"jacket|coat|blazer|suit|shrug|cardigan", n):           return "Jackets & Blazers"
    if re.search(r"shoe|sandal|boot|sneaker|loafer|heel|flat|slipper", n):return "Footwear"
    if re.search(r"bag|handbag|backpack|purse|tote|clutch", n):           return "Bags"
    if re.search(r"\bshort", n):                                          return "Shorts"
    if re.search(r"belt|watch|wallet|sunglass|cap|hat|scarf|jewel|ring|bracelet|necklace", n):
        return "Accessories"
    return "Other"

# ─────────────────────────────────────────────────────────
# 3.  LOAD & CLEAN DATA
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    # ── Sentiment ─────────────────────────────────────────
    df_sent = pd.read_csv(SENTIMENT_PATH)
    df_sent.columns = [c.strip().lower().replace(" ", "_") for c in df_sent.columns]
    df_sent.dropna(subset=["review_text"], inplace=True)
    df_sent.drop_duplicates(subset=["customer_id"], inplace=True)

    def clean_text(txt):
        txt = str(txt).lower()
        txt = re.sub(r"[^a-z\s]", " ", txt)
        return re.sub(r"\s+", " ", txt).strip()

    df_sent["clean_review"] = df_sent["review_text"].apply(clean_text)
    df_sent["tb_polarity"]  = df_sent["clean_review"].apply(
        lambda t: TextBlob(t).sentiment.polarity
    )
    df_sent["tb_sentiment"] = df_sent["tb_polarity"].apply(
        lambda p: "positive" if p > 0.05 else ("negative" if p < -0.05 else "neutral")
    )
    df_sent["customer_rating"] = pd.to_numeric(df_sent["customer_rating"], errors="coerce")

    # ── Catalog ───────────────────────────────────────────
    df_cat = pd.read_csv(CATALOG_PATH)
    df_cat.columns   = [c.strip() for c in df_cat.columns]
    df_cat["PrimaryColor"] = df_cat["PrimaryColor"].str.strip()
    df_cat["Price (INR)"]  = pd.to_numeric(df_cat["Price (INR)"], errors="coerce")
    df_cat.dropna(subset=["ProductName"], inplace=True)
    df_cat["Category"] = df_cat["ProductName"].apply(infer_cat)   # uses module-level fn

    return df_sent, df_cat

# ─────────────────────────────────────────────────────────
# 4.  INSIGHT HELPERS
# ─────────────────────────────────────────────────────────
def build_context_summary(df_sent: pd.DataFrame, df_cat: pd.DataFrame) -> str:
    fash = df_sent[df_sent["product_category"] == "fashion"]
    sent_dist   = fash["sentiment"].value_counts(normalize=True).mul(100).round(1).to_dict()
    avg_rating  = round(df_sent["customer_rating"].mean(), 2)
    fash_rating = round(fash["customer_rating"].mean(), 2)
    region_avg  = fash.groupby("region")["customer_rating"].mean().round(2).to_dict()
    age_avg     = fash.groupby("age_group")["customer_rating"].mean().round(2).to_dict()
    gender_avg  = fash.groupby("gender")["customer_rating"].mean().round(2).to_dict()
    top_colors  = df_cat["PrimaryColor"].value_counts().head(10).to_dict()
    top_brands  = df_cat["ProductBrand"].value_counts().head(10).to_dict()
    top_cats    = df_cat["Category"].value_counts().head(8).to_dict()
    price_stats = df_cat["Price (INR)"].describe().round(0)

    return textwrap.dedent(f"""
    ═══ LIVE PLATFORM DATA ═══
    Total customer reviews       : {len(df_sent):,}
    Fashion reviews              : {len(fash):,}
    Overall avg rating           : {avg_rating}/5
    Fashion avg rating           : {fash_rating}/5

    Fashion Sentiment (%)        : {sent_dist}
    Fashion by Region (avg ★)   : {region_avg}
    Fashion by Age group (avg ★) : {age_avg}
    Fashion by Gender (avg ★)   : {gender_avg}

    Catalog products             : {len(df_cat):,}
    Top categories by count      : {top_cats}
    Trending colors (count)      : {top_colors}
    Top brands (count)           : {top_brands}
    Price – min/median/max       : ₹{price_stats['min']:.0f} / ₹{price_stats['50%']:.0f} / ₹{price_stats['max']:.0f}
    """)


def top_complaints(df_sent: pd.DataFrame, n: int = 15) -> list:
    fash_neg = df_sent[
        (df_sent["product_category"] == "fashion") &
        (df_sent["sentiment"] == "negative")
    ]["clean_review"]
    stop = {
        "the","a","an","is","was","were","are","i","it","of","to","and","in","with",
        "for","my","not","very","this","that","but","have","had","has","be","on","at",
        "me","they","we","so","no","just","also","really","would","could","get","got",
        "like","more","from","than","then","when","where","what","which","will","been",
    }
    words = Counter()
    for txt in fash_neg:
        words.update([w for w in txt.split() if w not in stop and len(w) > 3])
    return [w for w, _ in words.most_common(n)]

# ─────────────────────────────────────────────────────────
# 5.  LLM SETUP
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_groq_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.65,
        max_tokens=900,
    )

def call_gemini(user_prompt: str, system_prompt: str) -> str:
    """Call Gemini via REST API."""
    url = (
        "https://generativelanguage.googleapis.com/v1/models/"
        f"gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    ),
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"{system_prompt}\n\nUser Question:\n{user_prompt}"
                    }
                ]
            }
        ]
    }
    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
 
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode())

    return result["candidates"][0]["content"]["parts"][0]["text"]

# ─────────────────────────────────────────────────────────
# 6.  AGENT — intent routing + dual-LLM
# ─────────────────────────────────────────────────────────
INTENT_MAP = {
    "complaints": [
        "complain","unhappy","negative","bad","problem","issue","dissatisf","worst",
        "hate","poor","return","refund","damage","wrong","delay","late","disappoint",
    ],
    "trends": [
        "trend","popular","season","latest","new","hot","2025","2026","cargo",
        "sneaker","quiet luxury","old money","streetwear","y2k","minimalist","what's in",
    ],
    "styling": [
        "wear","outfit","look","style","suggest","combination","match","pair",
        "formal","casual","office","party","wedding","date","diwali","beach",
        "airport","winter","summer","vacation","interview","color","colour",
        "skin tone","wide-leg","blazer","jeans","kurta","saree","smart casual",
        "first date","job interview","beach party","airport look","office look",
    ],
    "education": [
        "what is","difference between","how do","how to","explain","define",
        "meaning","slim fit","regular fit","leather shoes","fabric","cotton",
        "linen","smart casual","old money","take care",
    ],
    "sentiment": [
        "sentiment","rating","satisfaction","review","score","overall","analysis",
        "how are","customer feel","feedback","data","statistics",
    ],
    "inventory": [
        "stock","inventory","store","buy","order","restock","supplier","quantity",
        "which product","what to keep","sell","demand","priorit",
    ],
    "region": [
        "region","north","south","east","west","central","city","state","mumbai",
        "delhi","bangalore","kolkata","chennai","hyderabad","local",
    ],
}

def detect_intent(q: str) -> str:
    q_low = q.lower()
    scores = {intent: sum(kw in q_low for kw in kws) for intent, kws in INTENT_MAP.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def build_extra_context(intent: str, df_sent: pd.DataFrame, df_cat: pd.DataFrame) -> str:
    fash = df_sent[df_sent["product_category"] == "fashion"]

    if intent == "complaints":
        kws     = top_complaints(df_sent, 12)
        neg     = fash[fash["sentiment"] == "negative"]
        samples = neg["review_text"].dropna().sample(
            min(4, len(neg)), random_state=42
        ).tolist()
        return (
            f"Top negative keywords from fashion reviews: {kws}.\n"
            f"Sample real negative reviews: {samples}\n"
            f"Negative fashion reviews: {len(neg):,} ({len(neg)/len(fash)*100:.1f}%)"
        )

    elif intent == "trends":
        colors  = df_cat["PrimaryColor"].value_counts().head(8).to_dict()
        brands  = df_cat["ProductBrand"].value_counts().head(8).index.tolist()
        cats    = df_cat["Category"].value_counts().head(6).to_dict()
        pos_rev = fash[fash["sentiment"] == "positive"]["review_text"].dropna().sample(
            min(3, len(fash[fash["sentiment"] == "positive"])), random_state=7
        ).tolist()
        return (
            f"Trending colors in catalog: {colors}\n"
            f"Top brands: {brands}\nCategory volume: {cats}\n"
            f"Sample positive customer voices: {pos_rev}"
        )

    elif intent == "styling":
        color_by_gender = df_cat.groupby("Gender")["PrimaryColor"].apply(
            lambda s: s.value_counts().head(3).index.tolist()
        ).to_dict()
        price_by_cat = (
            df_cat[df_cat["Category"] != "Other"]
            .groupby("Category")["Price (INR)"].median().round(0).to_dict()
        )
        top_brands = df_cat["ProductBrand"].value_counts().head(10).index.tolist()
        return (
            f"Top colors by gender: {color_by_gender}\n"
            f"Median price by category: {price_by_cat}\n"
            f"Top catalog brands to recommend: {top_brands}"
        )

    elif intent == "education":
        return (
            "You are explaining fashion to a young Indian consumer. "
            "Use relatable Indian examples (Myntra brands, kurta vs western, etc.). "
            "Be warm, clear, and practical."
        )

    elif intent == "sentiment":
        by_age    = fash.groupby("age_group")["customer_rating"].mean().round(2).to_dict()
        by_gender = fash.groupby("gender")["customer_rating"].mean().round(2).to_dict()
        by_region = fash.groupby("region")["customer_rating"].mean().round(2).to_dict()
        tb_pos    = round(len(fash[fash["tb_sentiment"] == "positive"]) / len(fash) * 100, 1)
        tb_neg    = round(len(fash[fash["tb_sentiment"] == "negative"]) / len(fash) * 100, 1)
        return (
            f"Fashion sentiment by age group: {by_age}\n"
            f"By gender: {by_gender}\nBy region: {by_region}\n"
            f"TextBlob polarity: {tb_pos}% positive, {tb_neg}% negative"
        )

    elif intent == "inventory":
        by_cat   = (
            df_cat[df_cat["Category"] != "Other"]
            .groupby("Category")["Price (INR)"]
            .agg(count="count", median="median").round(0).to_dict("index")
        )
        by_color = df_cat["PrimaryColor"].value_counts().head(8).to_dict()
        by_brand = df_cat["ProductBrand"].value_counts().head(10).to_dict()
        return (
            f"Category stats (count + median price): {by_cat}\n"
            f"Color demand: {by_color}\nBrand stock count: {by_brand}"
        )

    elif intent == "region":
        by_region_full = (
            fash.groupby("region")
            .agg(avg_rating=("customer_rating", "mean"), total=("customer_id", "count"))
            .round(2).to_dict("index")
        )
        return f"Fashion performance by region: {by_region_full}"

    return ""


SYSTEM_TEMPLATE = """You are FashionIQ — a brilliant AI fashion advisor and retail business intelligence agent, built for an Indian fashion platform (like Myntra).

You play two roles depending on the question:
1. **Personal Stylist** — Give warm, confident, specific outfit advice like a friend who works at a top fashion brand. Cover any occasion: weddings, office, dates, beaches, Diwali, airport, winter, summer, etc.
2. **BI Analyst** — Give sharp, data-backed insights using the actual numbers from our platform.

═══ LIVE DATA FROM OUR PLATFORM ═══
{context}

═══ INTENT-SPECIFIC LIVE DATA ═══
{extra}

═══ CONVERSATION HISTORY ═══
{history}

═══ RESPONSE RULES ═══

For STYLING / OUTFIT questions:
- Sound like a stylish friend, not a robot. Be enthusiastic and specific.
- Give 2–3 concrete outfit combinations with specific item names.
- Mention actual colors from our data (Blue is #1, then Black, Red, Green).
- Recommend real brands from our catalog: Puma, AURELIA, Indian Terrain, Pepe Jeans, WROGN, Roadster, W, SPYKAR, U.S. Polo Assn., Flying Machine.
- Include price ranges (e.g., "a kurta from AURELIA at around ₹1,200").
- Add a styling tip or accessory suggestion at the end.

For TREND questions:
- Quote actual numbers from the data (e.g., "Blue dominates with 3,443 products in our catalog").
- Name real current 2025–26 trends: quiet luxury, Y2K revival, earth tones, wide-leg trousers, barrel jeans, sheer fabrics, ethnic fusion.
- Connect trends to what's available in our inventory.

For BUSINESS / BI questions:
- Use precise numbers. Be direct and confident.
- Give a clear, actionable recommendation at the end.

For EDUCATION questions:
- Explain with clarity and use Indian context (Myntra, Zara India, ethnic vs western).
- Keep it friendly and useful for a real shopper.

For COMPLAINT / SENTIMENT questions:
- Reference specific keywords or review samples from the data.
- Follow with a business recommendation.

ALWAYS:
- Sound human, warm, and expert — never robotic or template-like.
- Vary your sentence structure and tone.
- Use ₹ for Indian prices.
- Keep responses 4–8 sentences for styling, 5–10 for BI.
- Never use generic filler like "Great question!" or "Certainly!"
"""


def agent_respond(
    user_query: str,
    memory: ConversationBufferWindowMemory,
    df_sent: pd.DataFrame,
    df_cat: pd.DataFrame,
    context: str,
) -> tuple:
    """Returns (answer_text, llm_name_used)."""
    intent  = detect_intent(user_query)
    extra   = build_extra_context(intent, df_sent, df_cat)
    history = memory.load_memory_variables({}).get("history", "No prior conversation.")

    system_prompt = SYSTEM_TEMPLATE.format(
        context=context, extra=extra, history=history
    )

    # ── Try Groq first ────────────────────────────────────
    answer   = None
    llm_used = "Groq LLaMA-3-70B"
    try:
        llm      = get_groq_llm()
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_query)]
        response = llm.invoke(messages)
        answer   = response.content.strip()
    except Exception as groq_err:
        # ── Fallback: Gemini 1.5 Flash ────────────────────
        llm_used = "Gemini 1.5 Flash"
        try:
            answer = call_gemini(user_query, system_prompt).strip()
        except Exception as gem_err:
            answer = (
                f"⚠️ Both LLMs are unavailable right now.\n\n"
                f"**Groq error:** {groq_err}\n"
                f"**Gemini error:** {gem_err}\n\n"
                "Please verify your API keys in `app.py` (lines 30–31) and try again."
            )
            llm_used = "None"

    memory.save_context({"input": user_query}, {"output": answer})
    return answer, llm_used


# ─────────────────────────────────────────────────────────
# 7.  STREAMLIT UI
# ─────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <style>
    .main-title  { font-size:2rem; font-weight:700; }
    .sub-title   { font-size:.9rem; color:#666; margin-top:-8px; margin-bottom:4px; }
    .kpi-box     { background:linear-gradient(135deg,#fef7f0,#f8f4f0);
                   border-left:4px solid #e91e63; padding:14px 16px;
                   border-radius:10px; margin-bottom:6px; box-shadow: 0 2px 8px rgba(233,30,99,0.1); }
    .kpi-label   { font-size:.72rem; color:#888; text-transform:uppercase; letter-spacing:.6px; }
    .kpi-value   { font-size:1.6rem; font-weight:700; color:#2c1810; }
    .kpi-sub     { font-size:.7rem; color:#666; margin-top:2px; }
    .chat-user   { background:#f0f8ff; border-radius:14px 14px 4px 14px;
                   padding:11px 15px; margin:6px 0 6px 80px;
                   color:#1a365d; font-size:.92rem; line-height:1.6; border: 1px solid #e0f2fe; }
    .chat-bot    { background:#fef7f0; border-radius:4px 14px 14px 14px;
                   padding:13px 17px; margin:6px 80px 6px 0;
                   border-left:3px solid #e91e63; color:#2c1810;
                   font-size:.93rem; line-height:1.65; box-shadow: 0 2px 8px rgba(233,30,99,0.08); }
    .llm-badge   { font-size:.64rem; color:#e91e63; margin-top:6px; opacity:.8; }
    .section-hdr { font-size:1.05rem; font-weight:600; margin-bottom:6px; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 👗 FashionIQ BI Agent")
        st.caption("MGNM521 – CA1 · LPU · Dr. Arun Khatri")
        st.divider()
 
        
        page = st.radio(
            "Navigate",
            ["💬 AI Chatbot", "📊 Sentiment Dashboard", "📦 Inventory Advisor", "🧹 Data Preview"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("📦 25,000 reviews · 12,491 products\nMyntra Catalog + Sentiment Data")

    # ── Load data ─────────────────────────────────────────
    with st.spinner("🔄 Loading & cleaning datasets…"):
        try:
            df_sent, df_cat = load_data()
        except FileNotFoundError as e:
            st.error(
                f"❌ CSV not found: {e}\n\n"
                "Place **Customer_Sentiment.csv** and **myntra_products_catalog.csv** "
                "in the same folder as app.py, then re-run."
            )
            st.stop()

    context  = build_context_summary(df_sent, df_cat)
    df_fash  = df_sent[df_sent["product_category"] == "fashion"].copy()

    # ── Session state ─────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=8, return_messages=False)

    # ═════════════════════════════════════════════════════
    # PAGE 1 – CHATBOT
    # ═════════════════════════════════════════════════════
    if page == "💬 AI Chatbot":
        st.markdown('<p class="main-title">💬 FashionIQ AI Chatbot</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sub-title">Groq LLaMA-3-70B · Gemini 1.5 Flash fallback · '
            'LangChain Agent · Live dataset insights</p>',
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Quick prompts ─────────────────────────────────
        st.markdown("**✨ Quick questions — click to ask:**")
        rows = [
            ["What should I wear to a wedding?",
             "Suggest an outfit for a first date",
             "What colors suit wheatish skin tone?",
             "How can I style wide-leg jeans?"],
            ["What's trending this season?",
             "What is quiet luxury fashion?",
             "Explain old money fashion style",
             "Are cargo pants still in fashion?"],
            ["What should I wear to a job interview?",
             "Suggest outfits for a beach party",
             "Help me style for Diwali",
             "Recommend a winter wedding outfit"],
            ["What drives customer dissatisfaction?",
             "Which region needs most attention?",
             "What stock should I prioritize?",
             "What is the overall customer sentiment?"],
        ]
        for row_idx, row in enumerate(rows):
            cols = st.columns(len(row))
            for col_idx, (col, prompt) in enumerate(zip(cols, row)):
                unique_key = f"chip_{row_idx}_{col_idx}_{prompt[:20]}"
                if col.button(
            prompt,
            use_container_width=True,
            key=unique_key
                ):
                    st.session_state.pending_prompt = prompt
                

        st.divider()

        # ── Chat display ──────────────────────────────────
        chat_box = st.container(height=460)
        with chat_box:
            if not st.session_state.messages:
                st.markdown("""
                <div class="chat-bot">
                👗 <b>Hi! I'm FashionIQ</b> — your personal AI stylist and retail intelligence agent.<br><br>
                I'm trained on <b>25,000 real customer reviews</b> and <b>12,491 Myntra products</b>, so every answer I give is backed by actual data.<br><br>
                Ask me about: outfit ideas 👔 · color combos 🎨 · fashion trends 🔥 · occasion dressing 💒 · customer sentiment 📊 · inventory advice 📦 · or anything fashion!
                </div>
                """, unsafe_allow_html=True)

            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="chat-user">🧑 {msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    txt   = msg["content"].replace("\n", "<br>")
                    badge = msg.get("llm", "")
                    st.markdown(
                        f'<div class="chat-bot">🤖 {txt}'
                        f'<div class="llm-badge">⚡ Answered by {badge}</div></div>',
                        unsafe_allow_html=True,
                    )

        # ── Input bar ─────────────────────────────────────
        with st.form("chat_form", clear_on_submit=True):
            c1, c2 = st.columns([6, 1])
            user_input = c1.text_input(
                "msg", label_visibility="collapsed",
                placeholder="e.g.  What should I wear for a beach party in Goa?"
            )
            submitted = c2.form_submit_button("Send 🚀", use_container_width=True)

        if "pending_prompt" in st.session_state:
            user_input = st.session_state.pop("pending_prompt")
            submitted  = True

        if submitted and user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("✨ FashionIQ is thinking…"):
                answer, llm_used = agent_respond(
                    user_input, st.session_state.memory,
                    df_sent, df_cat, context,
                )
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "llm": llm_used}
            )
            st.rerun()

        if st.session_state.messages:
            if st.button("🗑️ Clear conversation"):
                st.session_state.messages = []
                st.session_state.memory   = ConversationBufferWindowMemory(k=8, return_messages=False)
                st.rerun()

    # ═════════════════════════════════════════════════════
    # PAGE 2 – SENTIMENT DASHBOARD  (FASHION ONLY)
    # ═════════════════════════════════════════════════════
    elif page == "📊 Sentiment Dashboard":
        st.markdown('<p class="main-title">📊 Sentiment Dashboard</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="sub-title">👗 Fashion category only · TextBlob NLP · '
            f'{len(df_fash):,} fashion reviews analysed</p>',
            unsafe_allow_html=True,
        )
        st.divider()

        # ── KPIs ──────────────────────────────────────────
        pos_n   = len(df_fash[df_fash["sentiment"] == "positive"])
        neg_n   = len(df_fash[df_fash["sentiment"] == "negative"])
        neu_n   = len(df_fash[df_fash["sentiment"] == "neutral"])
        avg_rat = round(df_fash["customer_rating"].mean(), 2)
        avg_pol = round(df_fash["tb_polarity"].mean(), 3)
        total_f = len(df_fash)

        k1, k2, k3, k4, k5 = st.columns(5)
        for col, label, val, sub in [
            (k1, "Fashion Reviews",  f"{total_f:,}",        "of 25K total reviews"),
            (k2, "✅ Positive",      f"{pos_n:,}",          f"{pos_n/total_f*100:.1f}%"),
            (k3, "❌ Negative",      f"{neg_n:,}",          f"{neg_n/total_f*100:.1f}%"),
            (k4, "😐 Neutral",       f"{neu_n:,}",          f"{neu_n/total_f*100:.1f}%"),
            (k5, "Avg Rating ⭐",    f"{avg_rat}",          f"Polarity score: {avg_pol}"),
        ]:
            col.markdown(f"""
            <div class="kpi-box">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── Row 1: Pie + Rating bar ───────────────────────
        c1, c2 = st.columns(2)
        with c1:
            sc = df_fash["sentiment"].value_counts().reset_index()
            sc.columns = ["Sentiment", "Count"]
            fig1 = px.pie(
                sc, names="Sentiment", values="Count",
                title="Fashion Sentiment Distribution",
                color="Sentiment",
                color_discrete_map={"positive":"#2ecc71","negative":"#e74c3c","neutral":"#f39c12"},
                hole=0.42,
            )
            fig1.update_layout(margin=dict(t=45,b=5,l=5,r=5))
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            rc = df_fash["customer_rating"].value_counts().sort_index().reset_index()
            rc.columns = ["Rating", "Count"]
            fig2 = px.bar(
                rc, x="Rating", y="Count",
                title="Fashion Rating Distribution (1–5 Stars)",
                color="Rating", color_continuous_scale="RdYlGn",
                text="Count",
            )
            fig2.update_layout(coloraxis_showscale=False, margin=dict(t=45,b=5))
            st.plotly_chart(fig2, use_container_width=True)

        # ── Row 2: Polarity histogram + Age group ─────────
        c3, c4 = st.columns(2)
        with c3:
            fig3 = px.histogram(
                df_fash, x="tb_polarity", nbins=40,
                title="TextBlob Polarity Distribution – Fashion Reviews",
                color_discrete_sequence=["#7048e8"],
            )
            fig3.add_vline(x=0.05,  line_dash="dot", line_color="#2ecc71",
                           annotation_text="Positive →")
            fig3.add_vline(x=-0.05, line_dash="dot", line_color="#e74c3c",
                           annotation_text="← Negative")
            fig3.update_layout(margin=dict(t=45,b=5))
            st.plotly_chart(fig3, use_container_width=True)

        with c4:
            age_df = (
                df_fash.groupby("age_group")["customer_rating"]
                .mean().reset_index()
            )
            age_df.columns = ["Age Group", "Avg Rating"]
            age_df = age_df.sort_values("Avg Rating", ascending=False)
            fig4 = px.bar(
                age_df, x="Age Group", y="Avg Rating",
                title="Avg Fashion Rating by Age Group",
                color="Avg Rating", color_continuous_scale="RdYlGn",
                text=age_df["Avg Rating"].round(2),
            )
            fig4.update_layout(coloraxis_showscale=False,
                               yaxis_range=[2.5, 3.6], margin=dict(t=45,b=5))
            st.plotly_chart(fig4, use_container_width=True)

        # ── Row 3: Region stacked bar ─────────────────────
        reg_sent = (
            df_fash.groupby(["region", "sentiment"])
            .size().reset_index(name="count")
        )
        fig5 = px.bar(
            reg_sent, x="region", y="count", color="sentiment",
            barmode="stack",
            title="Fashion Sentiment by Region",
            color_discrete_map={"positive":"#2ecc71","negative":"#e74c3c","neutral":"#f39c12"},
        )
        fig5.update_layout(xaxis_title="Region", yaxis_title="Review Count",
                           margin=dict(t=45,b=5))
        st.plotly_chart(fig5, use_container_width=True)

        # ── Row 4: Gender + Purchase Channel ─────────────
        c5, c6 = st.columns(2)
        with c5:
            gen_s = df_fash.groupby(["gender","sentiment"]).size().reset_index(name="count")
            fig6 = px.bar(
                gen_s, x="gender", y="count", color="sentiment",
                barmode="group",
                title="Fashion Sentiment by Gender",
                color_discrete_map={"positive":"#2ecc71","negative":"#e74c3c","neutral":"#f39c12"},
            )
            st.plotly_chart(fig6, use_container_width=True)

        with c6:
            if "purchase_channel" in df_fash.columns:
                ch_s = df_fash.groupby(["purchase_channel","sentiment"]).size().reset_index(name="count")
                fig7 = px.bar(
                    ch_s, x="purchase_channel", y="count", color="sentiment",
                    barmode="stack",
                    title="Fashion Sentiment by Purchase Channel",
                    color_discrete_map={"positive":"#2ecc71","negative":"#e74c3c","neutral":"#f39c12"},
                )
                st.plotly_chart(fig7, use_container_width=True)

        # ── Complaint keywords ────────────────────────────
        st.divider()
        st.markdown('<p class="section-hdr">🔍 Top Complaint Keywords – Fashion Negative Reviews</p>',
                    unsafe_allow_html=True)
        complaints = top_complaints(df_sent, 20)
        st.markdown("  ".join([f"`{w}`" for w in complaints]))

    # ═════════════════════════════════════════════════════
    # PAGE 3 – INVENTORY ADVISOR
    # ═════════════════════════════════════════════════════
    elif page == "📦 Inventory Advisor":
        st.markdown('<p class="main-title">📦 Inventory Advisor</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-title">AI-driven stock recommendations from catalog + sentiment data</p>',
                    unsafe_allow_html=True)
        st.divider()

        k1, k2, k3, k4 = st.columns(4)
        for col, label, val, sub in [
            (k1, "Total Products",  f"{len(df_cat):,}",                            "Myntra catalog"),
            (k2, "Avg Price",       f"₹{df_cat['Price (INR)'].mean():,.0f}",        "across all items"),
            (k3, "Median Price",    f"₹{df_cat['Price (INR)'].median():,.0f}",      "50th percentile"),
            (k4, "Price Range",
             f"₹{df_cat['Price (INR)'].min():.0f} – ₹{df_cat['Price (INR)'].max():,.0f}",
             "min to max"),
        ]:
            col.markdown(f"""
            <div class="kpi-box">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        c1, c2 = st.columns(2)
        with c1:
            tc = df_cat["PrimaryColor"].value_counts().head(12).reset_index()
            tc.columns = ["Color", "Count"]
            fig1 = px.bar(
                tc, x="Count", y="Color", orientation="h",
                title="Top 12 Colors in Catalog",
                color="Count", color_continuous_scale="Plasma",
            )
            fig1.update_layout(coloraxis_showscale=False,
                               yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            gd = df_cat["Gender"].value_counts().reset_index()
            gd.columns = ["Gender", "Count"]
            fig2 = px.pie(gd, names="Gender", values="Count",
                          title="Products by Gender",
                          hole=0.38,
                          color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig2, use_container_width=True)

        # Category is already in df_cat (computed at load time)
        price_cat = (
            df_cat[df_cat["Category"] != "Other"]
            .groupby("Category")["Price (INR)"].median()
            .reset_index()
        )
        price_cat.columns = ["Category", "Median Price (₹)"]
        price_cat = price_cat.sort_values("Median Price (₹)", ascending=False)

        fig3 = px.bar(
            price_cat, x="Category", y="Median Price (₹)",
            title="Median Price by Fashion Category",
            color="Median Price (₹)", color_continuous_scale="Blues",
            text=price_cat["Median Price (₹)"].apply(lambda x: f"₹{x:,.0f}"),
        )
        fig3.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

        tb = df_cat["ProductBrand"].value_counts().head(15).reset_index()
        tb.columns = ["Brand", "Products"]
        fig4 = px.treemap(tb, path=["Brand"], values="Products",
                          title="Top 15 Brands by Product Count",
                          color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown('<p class="section-hdr">🎯 Stocking Priority Recommendations</p>',
                    unsafe_allow_html=True)
        recs = pd.DataFrame([
            {"Item": "Blue Casual Shirts (Men)",    "Brand": "Indian Terrain / Parx",    "Avg Price": "₹955",   "Priority": "🔴 High",   "Reason": "Blue = #1 color (3,443 items); shirts = largest category"},
            {"Item": "Ethnic Kurta Sets (Women)",   "Brand": "AURELIA / W / EthnoVogue", "Avg Price": "₹1,271", "Priority": "🔴 High",   "Reason": "Highest positively-reviewed women's segment"},
            {"Item": "Skinny Jeans (Women)",        "Brand": "SPYKAR / Flying Machine",  "Avg Price": "₹1,325", "Priority": "🔴 High",   "Reason": "Consistent high demand, youth-driven"},
            {"Item": "Slim Fit Trousers (Men)",     "Brand": "Parx / Pepe Jeans",        "Avg Price": "₹1,200", "Priority": "🟡 Medium", "Reason": "Steady demand, formal segment growing"},
            {"Item": "Sneakers & Casual Shoes",     "Brand": "Puma / Roadster",          "Avg Price": "₹2,002", "Priority": "🟡 Medium", "Reason": "Highest avg price = best margin category"},
            {"Item": "Backpacks & Tote Bags",       "Brand": "GAP / Kenneth Cole",       "Avg Price": "₹3,128", "Priority": "🟡 Medium", "Reason": "Accessories are a rising trend"},
            {"Item": "Activewear / Shorts",         "Brand": "Puma / WROGN",             "Avg Price": "₹824",   "Priority": "🟢 Low",    "Reason": "Seasonal, smaller catalog share"},
            {"Item": "Formal Blazers (Men)",        "Brand": "Raymond / Indian Terrain", "Avg Price": "₹2,219", "Priority": "🟢 Low",    "Reason": "Niche segment, stock selectively"},
        ])
        st.dataframe(recs, use_container_width=True, hide_index=True)

    # ═════════════════════════════════════════════════════
    # PAGE 4 – DATA PREVIEW
    # ═════════════════════════════════════════════════════
    elif page == "🧹 Data Preview":
        st.markdown('<p class="main-title">🧹 Data Preview & Cleaning Log</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-title">Cleaned datasets powering all analysis</p>',
                    unsafe_allow_html=True)
        st.divider()

        tab1, tab2 = st.tabs(["📋 Customer Sentiment", "🛍️ Myntra Catalog"])

        with tab1:
            st.info(
                f"**Rows:** {df_sent.shape[0]:,} | **Columns:** {df_sent.shape[1]} | "
                f"**Fashion rows:** {len(df_fash):,}"
            )
            st.caption(
                "Cleaning applied: lowercase, special-char removal, null drop, "
                "duplicate drop on customer_id, TextBlob polarity score added."
            )
            cols_show = [
                "customer_id","gender","age_group","region","product_category",
                "customer_rating","review_text","sentiment","tb_polarity","tb_sentiment",
            ]
            st.dataframe(df_sent[cols_show].head(200), use_container_width=True, height=420)
            st.download_button(
                "⬇️ Download Cleaned Sentiment CSV",
                data=df_sent.to_csv(index=False).encode(),
                file_name="cleaned_sentiment.csv", mime="text/csv",
            )

        with tab2:
            st.info(f"**Rows:** {df_cat.shape[0]:,} | **Columns:** {df_cat.shape[1]}")
            st.caption(
                "Cleaning applied: column strip, color strip, price coerce to numeric, "
                "null ProductName drop, Category inferred from product name."
            )
            cols_cat = [
                "ProductID","ProductName","ProductBrand","Gender",
                "Price (INR)","PrimaryColor","Category",
            ]
            st.dataframe(df_cat[cols_cat].head(200), use_container_width=True, height=420)
            st.download_button(
                "⬇️ Download Cleaned Catalog CSV",
                data=df_cat.to_csv(index=False).encode(),
                file_name="cleaned_catalog.csv", mime="text/csv",
            )

    # ── Footer ─────────────────────────────────────────
    st.divider()
    st.caption(
        f"👗 FashionIQ · MGNM521 CA1 · LPU · Dr. Arun Khatri · "
        f"Groq LLaMA-3-70B + Gemini 1.5 Flash · "
        f"25,000 reviews + 12,491 products · {datetime.now().strftime('%b %Y')}"
    )


if __name__ == "__main__":
    main()
