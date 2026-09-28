"""Generates InsightForge.ipynb (kept as a script so the notebook is reproducible)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# InsightForge: AI-Powered Business Intelligence Assistant
**UCSB PaCE: Advanced Generative AI Capstone**

InsightForge helps small and medium-sized businesses turn raw sales data into insight. It uses
**LangChain**, **Retrieval-Augmented Generation (RAG)** and an **LLM** to:

1. **Analyze business data**: identify key trends and patterns
2. **Generate insights and recommendations**: answer questions in natural language
3. **Visualize data insights**: charts that are easy to interpret

**Design decision:** the LLM is never asked to do arithmetic. Pandas computes every statistic and a
custom retriever passes only the relevant figures to the LLM, which explains them. This keeps the
numbers accurate and makes the answers traceable.

| Part | Step | Section |
|---|---|---|
| 1 | Data preparation | 1 |
| 1 | Knowledge base creation | 2 |
| 1 | LLM application development (advanced summary, custom retriever, prompt engineering) | 3 |
| 1 | Chain prompts | 4 |
| 1 | RAG system setup | 5 |
| 1 | Memory integration | 6 |
| 2 | Model evaluation (QAEvalChain) | 7 |
| 2 | Monitoring (latency, tokens, cost) | 7.1 |
| 2 | Data visualization | 8 |
| 2 | Streamlit UI | 9 (`app.py`) |
""")

md("## 0. Setup")
code("""
# !pip install -r requirements.txt   # uncomment on first run

%matplotlib inline
import os, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from IPython.display import display, Markdown

import insightforge as ifg

# The OpenAI key is read from the environment or a local .env file (OPENAI_API_KEY=sk-...),
# loaded by insightforge on import. Never hard-code it in the notebook.
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
print("LLM backend:", "OpenAI gpt-4o-mini" if USE_OPENAI else "DEMO MODE (no API key found - placeholder responses)")
""")

md("""
## 1. Data preparation
The dataset is pre-prepared (no cleaning required), so this step only loads it and adds analysis
helper columns: `Year`, `Quarter`, `Month`, `Month_Name` and `Age_Group`.
""")
code("""
df = ifg.load_data()
print(df.shape)
display(df.head())
df.info()
""")
code("""
print("Missing values:\\n", df.isna().sum().to_string())
print("\\nDate range:", df["Date"].min().date(), "->", df["Date"].max().date())
display(df.describe(include="all").T)
""")

md("""
## 2. Knowledge base creation
The data is organized into a structured, retrievable form: each business topic (time, product,
region, demographics, statistics) becomes one LangChain `Document` with `topic` metadata.
""")
code("""
knowledge_base = ifg.build_knowledge_base(df)
print(f"{len(knowledge_base)} knowledge-base documents:")
for d in knowledge_base:
    print(" -", d.metadata["topic"])
""")

md("""
## 3. LLM application development
### 3.1 Advanced data summary
Key metrics and trends computed with pandas.
""")
md("**Overview**")
code("summaries = ifg.compute_summaries(df)\nprint(summaries['overview'])")
md("**Sales performance by time period**")
code("print(summaries['time_yearly']); print(); print(summaries['time_monthly'])")
code("print(summaries['time_quarterly'])")
md("**Product and regional analysis**")
code("print(summaries['product']); print(); print(summaries['region']); print(); print(summaries['product_region'])")
md("**Customer segmentation by demographics**")
code("print(summaries['gender']); print(); print(summaries['age']); print(); print(summaries['segment'])")
md("**Statistical measures (median, standard deviation, correlation)**")
code("print(summaries['statistics']); print(); print(summaries['product_satisfaction'])")

md("""
### 3.2 Custom retriever
`SalesStatsRetriever` extends LangChain's `BaseRetriever`. It routes each question to the most
relevant statistics documents by keyword and always includes the overview. Because the documents
are exact pandas outputs, retrieval is deterministic and needs no embeddings.
""")
code("""
stats_retriever = ifg.SalesStatsRetriever(docs=knowledge_base)
for q in ["What is the median and standard deviation of sales?",
          "Which region sells the most Widget A?",
          "How do sales differ by age and gender?",
          "How did sales change over the years?"]:
    print(f"{q}\\n   -> {[d.metadata['topic'] for d in stats_retriever.invoke(q)]}")
""")

md("""
### 3.3 Prompt engineering
The system prompt gives the LLM a role and grounding rules: use only the retrieved statistics, never
invent numbers, admit when data is missing, and lead with the answer followed by supporting figures.
""")
code("print(ifg.SYSTEM_PROMPT)")

md("""
## 4. Chain prompts
Two prompts are chained with the LangChain Expression Language (LCEL):

1. **Analysis prompt**: retrieved statistics + PDF knowledge + chat history + question -> grounded answer
2. **Recommendation prompt**: takes the analysis and adds 2-3 actionable business recommendations

```
question -> [stats retriever + PDF retriever] -> analysis prompt -> LLM -> recommendation prompt -> LLM -> answer
```
""")
code("print(ifg.RECOMMENDATION_PROMPT.messages[0].prompt.template)")

md("""
## 5. RAG system setup
Beyond the sales statistics, the four reference PDFs (BI approaches, AI business model innovation,
Walmart sales analysis, time-series prediction) are split into chunks, embedded with OpenAI
embeddings and indexed in **FAISS**. The top matching chunks give the LLM domain best practices to
draw on when making recommendations.
""")
code("""
if USE_OPENAI:
    llm = ifg.get_llm()
    embeddings = ifg.get_embeddings()
else:  # demo mode so the notebook still runs end to end without a key
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_community.embeddings import FakeEmbeddings
    llm = FakeListChatModel(responses=["[demo mode - set OPENAI_API_KEY for real answers]"])
    embeddings = FakeEmbeddings(size=256)

pdf_store = ifg.build_pdf_vectorstore(embeddings)
pdf_retriever = pdf_store.as_retriever(search_kwargs={"k": 3})
print("PDF chunks indexed:", pdf_store.index.ntotal)
""")
code("""
for d in pdf_retriever.invoke("How can business intelligence improve decision making?"):
    print(f"[{os.path.basename(d.metadata['source'])}, p.{d.metadata.get('page')}] {d.page_content[:200]}...\\n")
""")

md("""
## 6. Memory integration
`InsightForgeAssistant` wraps the chain in `RunnableWithMessageHistory`, so each conversation
(`session_id`) keeps its own history. Follow-up questions such as *"What about the West?"* are
resolved using earlier turns. History is capped at the last 10 messages to keep prompts short.
""")
code("""
assistant = ifg.InsightForgeAssistant(llm, df, pdf_retriever=pdf_retriever)

def ask(q, session="demo"):
    display(Markdown(f"**Q:** {q}"))
    display(Markdown(assistant.ask(q, session_id=session)))

ask("Which product generates the most revenue, and how does it perform by region?")
""")
code("""
ask("What about customer satisfaction for that product?")   # follow-up relies on memory
""")
code("""
ask("Which customer segments should we target to grow sales?")
""")
code("""
print("Messages stored in memory:", len(assistant.history("demo")))
for m in assistant.history("demo"):
    print(f"{m.type:>5}: {m.content[:90]}...")
""")

md("""
## 7. Model evaluation with QAEvalChain
The ground-truth answers are **computed with pandas**, so they are exactly correct. Each question
is answered by the assistant (in a fresh session), then `QAEvalChain` uses an LLM as the grader to
mark each prediction CORRECT or INCORRECT against the ground truth.
""")
code("""
eval_set = ifg.build_eval_set(df)
display(pd.DataFrame(eval_set))
""")
code("""
if USE_OPENAI:
    eval_assistant = ifg.InsightForgeAssistant(llm, df, with_recommendations=False)  # grade the core answer
    results = ifg.evaluate_with_qaevalchain(eval_assistant, eval_llm=ifg.get_llm())
    accuracy = (results["grade"] == "CORRECT").mean()
    display(results)
    print(f"Accuracy: {accuracy:.0%} ({(results['grade']=='CORRECT').sum()}/{len(results)})")
else:
    print("Set OPENAI_API_KEY and re-run this cell to evaluate the model.")
""")

md("""
### 7.1 Monitoring
Every call to `InsightForgeAssistant.ask()` is logged to `logs/monitoring.csv`: timestamp, session,
question, latency, number of LLM calls, prompt/completion tokens, estimated OpenAI cost and status
(ok or error). The same log powers the **Monitoring** tab in the Streamlit app, so response time,
token usage and spend can be tracked over time and regressions spotted early.
""")
code("""
log = ifg.load_monitoring_log()
if log.empty:
    print("No questions logged yet.")
else:
    summary = ifg.monitoring_summary(log)
    display(pd.Series(summary).round(4).to_frame("value"))
    display(log.tail(10))
    ifg.plot_monitoring(log);
""")

md("""
## 8. Data visualization
""")
md("### 8.1 Sales trends over time")
code("ifg.plot_sales_trend(df);")
code("ifg.plot_quarterly_sales(df);")
md("### 8.2 Product performance comparisons")
code("ifg.plot_product_performance(df);")
code("ifg.plot_product_trend(df);")
md("### 8.3 Regional analysis")
code("ifg.plot_regional_analysis(df);")
md("### 8.4 Customer demographics and segmentation")
code("ifg.plot_demographics(df);")
code("ifg.plot_satisfaction(df);")

md("""
## 9. Streamlit UI
The interactive app is in `app.py`. Run it from this folder with:

```bash
streamlit run app.py
```
The OpenAI key is read from `.env` (or can be pasted into the sidebar).

It has five tabs: **Chat** (the AI assistant with memory), **Dashboard** (KPIs and all charts),
**Data Insights** (the computed summaries and raw data), **Evaluation** (runs QAEvalChain) and
**Monitoring** (latency, token usage, cost and errors for every question asked).
""")

md("""
## 10. Key findings and conclusion
*(Numbers below come from the pandas summaries above.)*

- **Revenue:** 1.38M in total sales across 2,500 transactions (Jan 2022 to Nov 2028). The average
  transaction is about 553, with a high spread (std about 260).
- **Time:** yearly sales are broadly flat (roughly 195K to 206K per full year). The best full year
  is 2026 and there is no strong upward trend, so growth has to come from mix and targeting.
- **Products:** Widget A leads total sales and is the top product in the East, North and South.
  Widget C leads in the West.
- **Regions:** sales are spread fairly evenly across regions. East is the smallest, which makes it
  an expansion opportunity.
- **Customers:** customers aged 25 to 64 drive most of the revenue. Sales by gender are nearly equal.
- **Satisfaction:** the average is 3.0 out of 5, with almost no correlation to sales or age. This
  points to a service-quality opportunity across the board.

**Conclusion:** InsightForge combines pandas analytics, a custom retriever, PDF-based RAG, chained
prompts and conversational memory, with every answer monitored for latency, tokens and cost. The result is an assistant that gives accurate, explainable
answers, evaluated with QAEvalChain and delivered through a Streamlit interface. Possible next
steps: forecasting (for example Prophet or ARIMA), connecting live data sources, and a SQL agent
for ad-hoc queries.
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "InsightForge.ipynb")
print("written", len(C), "cells")
