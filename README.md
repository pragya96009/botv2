## 📌 Project Overview
An AI-powered BI chatbot that:
- Analyses 25,000 customer reviews (TextBlob sentiment)
- Answers business queries via LangChain + Groq LLaMA-3 (free)
- Visualises sentiment, trends, and inventory insights (Plotly)
- Built with Streamlit (no heavy setup needed)

## 🗂️ Folder Structure
```
fashion_bi_agent/
├── app.py                        ← Main Streamlit app (run this)
├── requirements.txt              ← Python dependencies
├── README.md                     ← This file
├── Customer_Sentiment.csv        ← Place here (25K reviews)
└── myntra_products_catalog.csv   ← Place here (12K products)
```

---

## ⚙️ Setup & Run

### Step 1 – Place CSVs
Copy both CSV files into the `fashion_bi_agent/` folder:
- `Customer_Sentiment.csv`
- `myntra_products_catalog.csv`

### Step 2 – Install dependencies
```bash
pip install -r requirements.txt
python -m textblob.download_corpora   # one-time NLTK data
```

### Step 3 – Get a FREE Groq API key
1. Visit https://console.groq.com
2. Sign up (free) → Create API Key
3. Copy the key (starts with `gsk_...`)

### Step 4 – Run the app
```bash
streamlit run app.py
```
The app opens at **http://localhost:8501** in your browser.

### Step 5 – Paste API key in sidebar
Paste your Groq key in the sidebar field → chatbot is ready!

---

## 🧠 Tech Stack (as per CA1 criteria)

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| App framework | **Streamlit** |
| NLP / Sentiment | **TextBlob + NLTK** |
| LLM | **Groq LLaMA-3-70B** (free tier) |
| Agent layer | **LangChain** |
| Visualisation | **Plotly** |
| Data handling | **Pandas + NumPy** |

---

## 📋 Features

### 💬 AI Chatbot (Page 1)
- Natural language Q&A powered by LLaMA-3 via Groq
- LangChain ConversationMemory (last 6 turns)
- Intent detection: complaints / trends / styling / region / sentiment
- Quick-prompt chips for instant queries

### 📊 Sentiment Dashboard (Page 2)
- Pie chart: overall sentiment distribution
- Bar chart: rating distribution (1–5)
- Grouped bar: sentiment by product category
- Stacked bar: regional sentiment
- Histogram: TextBlob polarity distribution
- Age-group avg ratings

### 📦 Inventory Advisor (Page 3)
- Color trend bar chart
- Gender-wise product pie chart
- Median price by category
- Brand treemap
- Priority stocking table (High / Medium / Low)

### 🧹 Data Preview (Page 4)
- Shows cleaned DataFrames (first 100 rows)
- Download cleaned CSVs

---

## 💡 Sample Questions for Chatbot
- "What are customers most unhappy about?"
- "Which region has the worst satisfaction?"
- "Suggest a formal office look for men"
- "What colors should I stock this season?"
- "What stock should I prioritize?"
- "How has fashion sentiment performed?"

---

## 📊 Evaluation Criteria Covered

| Criteria | Marks | Status |
|----------|-------|--------|
| Project execution / UI/UX | 10 | ✅ Streamlit multi-page app |
| Data Handling | 10 | ✅ Clean, preprocess, TextBlob |
| Presentation | 5 | ✅ Plotly charts + KPI cards |
| Platform (advanced features) | 5 | ✅ LangChain agent + Groq LLM |
| **Total** | **30** | |
