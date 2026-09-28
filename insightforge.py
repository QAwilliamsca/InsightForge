"""
InsightForge - AI-Powered Business Intelligence Assistant
=========================================================
Core module shared by the notebook (InsightForge.ipynb) and the Streamlit app (app.py).

Pipeline
--------
1. Data preparation     -> load_data()
2. Knowledge base       -> build_knowledge_base()  (pandas statistics as Documents)
3. Advanced summary     -> compute_summaries()     (time, product, region, demographics, stats)
4. Custom retriever     -> SalesStatsRetriever     (routes a question to the relevant stats)
5. RAG + PDFs           -> build_pdf_vectorstore() (FAISS over the reference PDFs, optional)
6. Prompt chaining      -> build_chain()           (analysis prompt -> recommendation prompt)
7. Memory               -> InsightForgeAssistant   (per-session chat history)
8. Evaluation           -> evaluate_with_qaevalchain()
   Monitoring           -> InsightForgeAssistant.ask() logs latency/tokens/cost -> load_monitoring_log()
9. Visualisations       -> plot_* functions
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sales_data.csv"
PDF_DIR = BASE_DIR / "data" / "pdfs"
MONITORING_LOG = BASE_DIR / "logs" / "monitoring.csv"

try:  # pick up OPENAI_API_KEY from a local .env file (never committed or shared)
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

AGE_BINS = [17, 24, 34, 44, 54, 64, 120]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]


# ---------------------------------------------------------------------------
# 1. Data preparation
# ---------------------------------------------------------------------------
def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the sales data and add helper columns used for analysis."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Year"] = df["Date"].dt.year
    df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Month_Name"] = df["Date"].dt.month_name()
    df["Age_Group"] = pd.cut(df["Customer_Age"], bins=AGE_BINS, labels=AGE_LABELS)
    return df


# ---------------------------------------------------------------------------
# 2-3. Advanced data summary -> knowledge base
# ---------------------------------------------------------------------------
def _describe(series: pd.Series) -> str:
    return (
        f"mean={series.mean():.2f}, median={series.median():.2f}, "
        f"std={series.std():.2f}, min={series.min():.2f}, max={series.max():.2f}"
    )


def _group_table(df: pd.DataFrame, by, value: str = "Sales") -> str:
    g = df.groupby(by, observed=True)[value].agg(["sum", "mean", "median", "std", "count"])
    g = g.round(2).rename(
        columns={"sum": "total", "mean": "average", "median": "median", "std": "std_dev", "count": "transactions"}
    )
    if value == "Customer_Satisfaction":  # a sum of 1-5 ratings is meaningless
        g = g.drop(columns="total")
    return g.to_string()


def _complete_months(df: pd.DataFrame) -> pd.Series:
    """Monthly total sales, excluding months the data only partially covers (e.g. the last one)."""
    monthly = df.set_index("Date")["Sales"].resample("MS").sum()
    month_end = monthly.index + pd.offsets.MonthEnd(0)
    return monthly[(monthly.index >= df["Date"].min().normalize()) & (month_end <= df["Date"].max())]


def compute_summaries(df: pd.DataFrame) -> Dict[str, str]:
    """Compute the statistics the LLM will be grounded on. Each entry becomes one Document."""
    s: Dict[str, str] = {}

    total = df["Sales"].sum()
    s["overview"] = (
        f"Dataset covers {len(df)} transactions from {df['Date'].min().date()} to {df['Date'].max().date()}.\n"
        f"Products: {', '.join(sorted(df['Product'].unique()))}. "
        f"Regions: {', '.join(sorted(df['Region'].unique()))}.\n"
        f"Total sales: {total:,.0f}.\n"
        f"Sales per transaction: {_describe(df['Sales'])}.\n"
        f"Customer age: {_describe(df['Customer_Age'])}.\n"
        f"Customer satisfaction (1-5): {_describe(df['Customer_Satisfaction'])}."
    )

    # --- Sales performance by time period
    yearly = df.groupby("Year")["Sales"].sum()
    growth = yearly.pct_change().mul(100).round(1)
    s["time_yearly"] = (
        "Sales by year:\n" + _group_table(df, "Year")
        + "\n\nYear-over-year growth of total sales (%):\n" + growth.to_string()
        + "\nNote: the first and last years may be partial years."
    )
    s["time_quarterly"] = "Sales by quarter:\n" + _group_table(df, "Quarter")
    monthly = _complete_months(df)
    monthly.index = monthly.index.strftime("%Y-%m")
    s["time_monthly"] = (
        f"Monthly total sales (complete months only; partial months at the edges of the data are excluded): "
        f"best month {monthly.idxmax()} ({monthly.max():,.0f}), "
        f"worst month {monthly.idxmin()} ({monthly.min():,.0f}), "
        f"average month {monthly.mean():,.0f}.\n"
        "Sales by calendar month (all years combined, seasonality):\n"
        + _group_table(df, "Month_Name")
    )

    # --- Product and regional analysis
    s["product"] = "Sales by product:\n" + _group_table(df, "Product")
    s["region"] = "Sales by region:\n" + _group_table(df, "Region")
    pivot = df.pivot_table(index="Product", columns="Region", values="Sales", aggfunc="sum")
    # Pre-ranked so the LLM reads the max/min instead of comparing numbers itself
    pr = df.groupby(["Product", "Region"])["Sales"].sum().sort_values(ascending=False)
    ranked = "\n".join(f"{i}. {p} in {r}: {v:,.0f}" for i, ((p, r), v) in enumerate(pr.items(), 1))
    best_region = "\n".join(
        f"{p}: strongest in {g.idxmax()[1]} ({g.max():,.0f}), weakest in {g.idxmin()[1]} ({g.min():,.0f})"
        for p, g in pr.groupby(level="Product")
    )
    best_product = "\n".join(
        f"{r}: top product {g.idxmax()[0]} ({g.max():,.0f}), bottom product {g.idxmin()[0]} ({g.min():,.0f})"
        for r, g in pr.groupby(level="Region")
    )
    (top_p, top_r), top_v = pr.index[0], pr.iloc[0]
    prod_tot = df.groupby("Product")["Sales"].sum()
    best_p = prod_tot.idxmax()
    best_p_regions = pr.xs(best_p, level="Product")  # already sorted high -> low
    s["product_region"] = (
        f"ANSWER TO 'which product sells best and where': {best_p} is the best-selling product overall "
        f"({prod_tot.max():,.0f}); it sells best in the {best_p_regions.index[0]} ({best_p_regions.iloc[0]:,.0f}). "
        f"{best_p} by region, high to low: "
        + ", ".join(f"{r} {v:,.0f}" for r, v in best_p_regions.items()) + ".\n"
        f"Highest-selling product-region combination: {top_p} in {top_r} ({top_v:,.0f}).\n\n"
        "WHERE EACH PRODUCT SELLS BEST (compares regions for one product; use this for 'where does X sell best'):\n"
        + best_region
        + "\n\nWHICH PRODUCT LEADS EACH REGION (compares products within one region; do NOT use this to say "
        "where a product sells best):\n" + best_product
        + "\n\nAll product-region combinations ranked by total sales:\n" + ranked
        + "\n\nTotal sales, product x region:\n" + pivot.round(0).to_string()
    )
    s["product_satisfaction"] = (
        "Customer satisfaction by product:\n" + _group_table(df, "Product", "Customer_Satisfaction")
        + "\n\nCustomer satisfaction by region:\n" + _group_table(df, "Region", "Customer_Satisfaction")
    )

    # --- Customer segmentation by demographics
    s["gender"] = (
        "Sales by customer gender:\n" + _group_table(df, "Customer_Gender")
        + "\n\nSatisfaction by gender:\n" + _group_table(df, "Customer_Gender", "Customer_Satisfaction")
    )
    s["age"] = (
        "Sales by customer age group:\n" + _group_table(df, "Age_Group")
        + "\n\nSatisfaction by age group:\n" + _group_table(df, "Age_Group", "Customer_Satisfaction")
    )
    seg = df.pivot_table(index="Age_Group", columns="Customer_Gender", values="Sales", aggfunc="sum", observed=True)
    s["segment"] = (
        "Total sales by age group x gender:\n" + seg.round(0).to_string()
        + "\n\nMost popular product per region (by total sales):\n"
        + df.groupby(["Region", "Product"])["Sales"].sum().groupby(level=0).idxmax().map(lambda t: t[1]).to_string()
    )

    # --- Statistical measures
    corr = df[["Sales", "Customer_Age", "Customer_Satisfaction"]].corr().round(3)
    max_r = corr.where(~(corr == 1.0)).abs().max().max()
    corr_note = (
        f"Interpretation: all correlations are negligible (largest |r| = {max_r:.3f}). Sales, customer age "
        "and satisfaction are essentially unrelated in this data; none of them explains the others."
        if max_r < 0.1 else f"Interpretation: the largest |r| is {max_r:.3f}."
    )
    s["statistics"] = (
        "Statistical measures:\n"
        + df[["Sales", "Customer_Age", "Customer_Satisfaction"]].describe().round(2).to_string()
        + "\n\nCorrelation matrix:\n" + corr.to_string()
        + "\n" + corr_note
    )
    return s


def build_knowledge_base(df: pd.DataFrame) -> List[Document]:
    """Organise the computed statistics into LangChain Documents with metadata."""
    return [
        Document(page_content=text, metadata={"source": "sales_data.csv", "topic": topic})
        for topic, text in compute_summaries(df).items()
    ]


# ---------------------------------------------------------------------------
# 4. Custom retriever over the statistics
# ---------------------------------------------------------------------------
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "time_yearly": ["year", "annual", "growth", "trend", "over time", "2022", "2023", "2024", "2025", "2026", "2027", "2028"],
    "time_quarterly": ["quarter", "q1", "q2", "q3", "q4", "trend", "period"],
    "time_monthly": ["month", "season", "january", "february", "march", "april", "may", "june", "july",
                     "august", "september", "october", "november", "december", "best month", "worst month"],
    "product": ["product", "widget", "best-selling", "best selling", "top seller", "item"],
    "region": ["region", "north", "south", "east", "west", "area", "location", "market"],
    "product_region": ["product", "widget", "region", "north", "south", "east", "west", "where",
                       "combination", "strongest", "weakest"],
    "product_satisfaction": ["satisfaction", "satisfied", "rating", "happy", "experience", "driver", "drivers", "drives"],
    "gender": ["gender", "male", "female", "men", "women", "demographic"],
    "age": ["age", "young", "old", "older", "younger", "demographic", "generation"],
    "segment": ["segment", "demographic", "customer", "popular", "target"],
    "statistics": ["median", "standard deviation", "std", "variance", "average", "mean", "statistic",
                   "correlation", "distribution", "min", "max", "driver", "drivers", "drive", "drives", "affect", "affects",
                   "relationship", "related", "influence", "impact"],
}


class SalesStatsRetriever(BaseRetriever):
    """Keyword-routed retriever: returns the pre-computed statistics that match the question.

    Pandas does the arithmetic, so the LLM never has to calculate numbers itself - it only
    reads and explains them. This keeps answers accurate.
    """

    docs: List[Document]
    k: int = 4

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        q = query.lower()
        scores = {
            topic: sum(1 for kw in kws if re.search(r"\b" + re.escape(kw) + r"\b", q))
            for topic, kws in TOPIC_KEYWORDS.items()
        }
        ranked = [t for t, sc in sorted(scores.items(), key=lambda x: -x[1]) if sc > 0][: self.k]
        topics = ["overview"] + ranked  # overview always gives the LLM the big picture
        if not ranked:  # general question -> give the main business views
            topics += ["time_yearly", "product", "region"]
        by_topic = {d.metadata["topic"]: d for d in self.docs}
        return [by_topic[t] for t in topics if t in by_topic]


# ---------------------------------------------------------------------------
# 5. RAG over the reference PDFs (domain knowledge)
# ---------------------------------------------------------------------------
def build_pdf_vectorstore(embeddings, pdf_dir: str | Path = PDF_DIR, chunk_size: int = 1000):
    """Load the reference PDFs, split them into chunks and index them in FAISS."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    pages = []
    for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
        pages.extend(PyPDFLoader(str(pdf)).load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    return FAISS.from_documents(chunks, embeddings)


def format_docs(docs: List[Document]) -> str:
    out = []
    for d in docs:
        label = d.metadata.get("topic") or Path(d.metadata.get("source", "")).name
        out.append(f"[{label}]\n{d.page_content}")
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# 6. Prompt engineering + prompt chaining
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are InsightForge, a business intelligence analyst for a small company that sells
four products (Widget A-D) in four regions.

Rules:
- Base every number on the SALES STATISTICS below. Never invent or recalculate figures;
  if the statistics do not contain the answer, say so.
- Only state numbers that appear in the SALES STATISTICS. When asked for a best, worst, highest,
  lowest, max or min, use the pre-ranked lists and "strongest/weakest" lines rather than comparing
  numbers yourself. Never compare values in the product x region grid yourself.
- "Where does a product sell best" means the region where THAT product's sales are highest (compare
  regions for that product), not the region where it beats other products.
- Do not claim that one variable drives or affects another unless the correlation matrix shows it;
  if the correlations are negligible, say there is no meaningful relationship.
- Use the BUSINESS KNOWLEDGE excerpts only for general context and best practices.
- Use the conversation history to resolve follow-up questions ("what about the North?").
- Be concise: lead with the direct answer, then 2-4 supporting facts with numbers.

SALES STATISTICS:
{stats_context}

BUSINESS KNOWLEDGE:
{pdf_context}"""

ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), MessagesPlaceholder("history"), ("human", "{question}")]
)

RECOMMENDATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You turn data analysis into action for a business owner. Given a question and an analyst's "
         "answer, return the answer text unchanged (without any 'Analyst answer' label), then add a "
         "section 'Recommendations' with 2-3 short, specific, actionable recommendations that follow "
         "from the numbers. Do not add new figures."),
        ("human", "Question: {question}\n\nAnalyst answer:\n{analysis}"),
    ]
)


def build_chain(llm, stats_retriever: BaseRetriever, pdf_retriever: Optional[BaseRetriever] = None,
                with_recommendations: bool = True):
    """Chain: retrieve -> analysis prompt -> LLM -> (recommendation prompt -> LLM)."""

    def get_pdf_context(x):
        if pdf_retriever is None:
            return "(no reference documents loaded)"
        return format_docs(pdf_retriever.invoke(x["question"]))

    analysis = (
        RunnablePassthrough.assign(
            stats_context=lambda x: format_docs(stats_retriever.invoke(x["question"])),
            pdf_context=RunnableLambda(get_pdf_context),
        )
        | ANALYSIS_PROMPT
        | llm
        | StrOutputParser()
    )
    if not with_recommendations:
        return analysis
    return (
        RunnablePassthrough.assign(analysis=analysis)
        | RECOMMENDATION_PROMPT
        | llm
        | StrOutputParser()
    )


# ---------------------------------------------------------------------------
# 7. Memory
# ---------------------------------------------------------------------------
class InsightForgeAssistant:
    """Conversational assistant: RAG chain + per-session memory."""

    def __init__(self, llm, df: pd.DataFrame, pdf_retriever: Optional[BaseRetriever] = None,
                 with_recommendations: bool = True, max_history_messages: int = 10,
                 log_path: Optional[Path] = MONITORING_LOG):
        self.df = df
        self.log_path = Path(log_path) if log_path else None
        self.docs = build_knowledge_base(df)
        self.stats_retriever = SalesStatsRetriever(docs=self.docs)
        self.max_history_messages = max_history_messages
        self._store: Dict[str, InMemoryChatMessageHistory] = {}
        chain = build_chain(llm, self.stats_retriever, pdf_retriever, with_recommendations)
        self.chain = RunnableWithMessageHistory(
            chain,
            self._get_history,
            input_messages_key="question",
            history_messages_key="history",
        )

    def _get_history(self, session_id: str) -> InMemoryChatMessageHistory:
        hist = self._store.setdefault(session_id, InMemoryChatMessageHistory())
        # keep memory bounded so prompts stay small
        if len(hist.messages) > self.max_history_messages:
            hist.messages = hist.messages[-self.max_history_messages:]
        return hist

    def ask(self, question: str, session_id: str = "default") -> str:
        """Answer a question and log latency, token usage and cost for monitoring."""
        from langchain_community.callbacks import get_openai_callback

        start, status, answer = time.perf_counter(), "ok", ""
        with get_openai_callback() as cb:
            try:
                answer = self.chain.invoke({"question": question},
                                           config={"configurable": {"session_id": session_id}})
            except Exception as e:
                status = f"error: {type(e).__name__}"
                raise
            finally:
                self._log({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "session_id": session_id,
                    "question": question,
                    "latency_s": round(time.perf_counter() - start, 2),
                    "llm_calls": cb.successful_requests,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_tokens": cb.total_tokens,
                    "cost_usd": round(cb.total_cost, 6),
                    "answer_chars": len(answer),
                    "status": status,
                })
        return answer

    def _log(self, row: Dict) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(self.log_path, mode="a", index=False,
                                   header=not self.log_path.exists())

    def history(self, session_id: str = "default"):
        return self._get_history(session_id).messages

    def clear(self, session_id: str = "default"):
        self._store.pop(session_id, None)


# ---------------------------------------------------------------------------
# 8. Evaluation with QAEvalChain
# ---------------------------------------------------------------------------
def build_eval_set(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Question/answer pairs whose ground truth is computed directly with pandas."""
    prod = df.groupby("Product")["Sales"].sum()
    reg = df.groupby("Region")["Sales"].sum()
    gender_sat = df.groupby("Customer_Gender")["Customer_Satisfaction"].mean()
    yearly = df.groupby("Year")["Sales"].sum()
    full_years = yearly.loc[df.groupby("Year")["Date"].nunique() >= 360]
    prod_reg = df.groupby(["Product", "Region"])["Sales"].sum()
    (top_p, top_r), top_v = prod_reg.idxmax(), prod_reg.max()
    return [
        {"query": "Which product-region combination has the highest sales?",
         "answer": f"{top_p} in the {top_r}, with total sales of {top_v:,.0f}."},
        {"query": "Which product has the highest total sales?",
         "answer": f"{prod.idxmax()} with total sales of {prod.max():,.0f}."},
        {"query": "Which region has the lowest total sales?",
         "answer": f"{reg.idxmin()} with total sales of {reg.min():,.0f}."},
        {"query": "What is the median sales value per transaction?",
         "answer": f"{df['Sales'].median():.2f}"},
        {"query": "What is the standard deviation of sales per transaction?",
         "answer": f"{df['Sales'].std():.2f}"},
        {"query": "What is the average customer satisfaction score?",
         "answer": f"{df['Customer_Satisfaction'].mean():.2f} out of 5"},
        {"query": "Which gender has a higher average customer satisfaction?",
         "answer": f"{gender_sat.idxmax()} ({gender_sat.max():.2f} vs {gender_sat.min():.2f})."},
        {"query": "Among full calendar years, which year had the highest total sales?",
         "answer": f"{full_years.idxmax()} with total sales of {full_years.max():,.0f}."},
        {"query": "What is the average age of customers?",
         "answer": f"{df['Customer_Age'].mean():.1f} years"},
    ]


def evaluate_with_qaevalchain(assistant: InsightForgeAssistant, eval_llm,
                              eval_set: Optional[List[Dict[str, str]]] = None) -> pd.DataFrame:
    """Answer every eval question (fresh memory each time) and grade with QAEvalChain."""
    from langchain_classic.evaluation.qa import QAEvalChain

    eval_set = eval_set or build_eval_set(assistant.df)
    predictions = []
    for i, ex in enumerate(eval_set):
        sid = f"eval-{i}"
        predictions.append({"result": assistant.ask(ex["query"], session_id=sid)})
        assistant.clear(sid)

    grader = QAEvalChain.from_llm(eval_llm)
    graded = grader.evaluate(eval_set, predictions, question_key="query",
                             answer_key="answer", prediction_key="result")
    rows = []
    for ex, pred, g in zip(eval_set, predictions, graded):
        text = g.get("results", "")
        grade = "CORRECT" if "CORRECT" in text.upper() and "INCORRECT" not in text.upper() else "INCORRECT"
        rows.append({"question": ex["query"], "expected": ex["answer"],
                     "prediction": pred["result"], "grade": grade})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8b. Monitoring
# ---------------------------------------------------------------------------
def load_monitoring_log(path: str | Path = MONITORING_LOG) -> pd.DataFrame:
    """Every question the assistant has answered, with latency, tokens, cost and status."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["timestamp"])


def monitoring_summary(log: pd.DataFrame) -> Dict[str, float]:
    if log.empty:
        return {}
    ok = log["status"].eq("ok")
    return {
        "questions": len(log),
        "error_rate": 1 - ok.mean(),
        "avg_latency_s": log["latency_s"].mean(),
        "p95_latency_s": log["latency_s"].quantile(0.95),
        "avg_tokens": log["total_tokens"].mean(),
        "total_tokens": log["total_tokens"].sum(),
        "total_cost_usd": log["cost_usd"].sum(),
    }


def plot_monitoring(log: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    axes[0].plot(range(1, len(log) + 1), log["latency_s"], marker="o", color="#4C72B0")
    axes[0].axhline(log["latency_s"].mean(), ls="--", color="#C44E52", label="average")
    axes[0].set(title="Response latency per question", xlabel="Question #", ylabel="Seconds")
    axes[0].legend()
    axes[1].bar(range(1, len(log) + 1), log["prompt_tokens"], color="#4C72B0", label="prompt")
    axes[1].bar(range(1, len(log) + 1), log["completion_tokens"], bottom=log["prompt_tokens"],
                color="#DD8452", label="completion")
    axes[1].set(title="Tokens per question", xlabel="Question #", ylabel="Tokens")
    axes[1].legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 9. Visualisations
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid")


def plot_sales_trend(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly = _complete_months(df)
    ax.plot(monthly.index, monthly.values, color="#4C72B0", alpha=0.5, label="Monthly sales")
    ax.plot(monthly.index, monthly.rolling(6).mean(), color="#C44E52", lw=2, label="6-month moving avg")
    ax.set(title="Sales trend over time (complete months only)", xlabel="Month", ylabel="Total sales")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_quarterly_sales(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 4))
    q = df.groupby("Quarter")["Sales"].sum()
    ax.bar(q.index, q.values, color="#4C72B0")
    ax.set(title="Total sales by quarter", xlabel="Quarter", ylabel="Total sales")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    return fig


def plot_product_performance(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    prod = df.groupby("Product")["Sales"].sum().sort_values(ascending=False)
    sns.barplot(x=prod.index, y=prod.values, ax=axes[0], color="#4C72B0")
    axes[0].set(title="Total sales by product", xlabel="Product", ylabel="Total sales")
    sns.boxplot(data=df, x="Product", y="Sales", order=prod.index, ax=axes[1], color="#DD8452")
    axes[1].set(title="Sales distribution per transaction", xlabel="Product", ylabel="Sales")
    fig.tight_layout()
    return fig


def plot_product_trend(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    yearly = df.pivot_table(index="Year", columns="Product", values="Sales", aggfunc="sum")
    yearly.plot(marker="o", ax=ax)
    ax.set(title="Yearly sales by product", xlabel="Year", ylabel="Total sales")
    fig.tight_layout()
    return fig


def plot_regional_analysis(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    reg = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    sns.barplot(x=reg.index, y=reg.values, ax=axes[0], color="#55A868")
    axes[0].set(title="Total sales by region", xlabel="Region", ylabel="Total sales")
    pivot = df.pivot_table(index="Region", columns="Product", values="Sales", aggfunc="sum")
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues", ax=axes[1])
    axes[1].set(title="Total sales: region x product")
    fig.tight_layout()
    return fig


def plot_demographics(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(df["Customer_Age"], bins=20, ax=axes[0, 0], color="#8172B3")
    axes[0, 0].set(title="Customer age distribution", xlabel="Age")
    age = df.groupby("Age_Group", observed=True)["Sales"].sum()
    sns.barplot(x=age.index.astype(str), y=age.values, ax=axes[0, 1], color="#8172B3")
    axes[0, 1].set(title="Total sales by age group", xlabel="Age group", ylabel="Total sales")
    g = df.groupby("Customer_Gender")["Sales"].sum()
    axes[1, 0].pie(g.values, labels=g.index, autopct="%1.1f%%", colors=["#C44E52", "#4C72B0"])
    axes[1, 0].set(title="Sales share by gender")
    sns.boxplot(data=df, x="Age_Group", y="Customer_Satisfaction", hue="Customer_Gender", ax=axes[1, 1])
    axes[1, 1].set(title="Satisfaction by age group and gender", xlabel="Age group", ylabel="Satisfaction")
    fig.tight_layout()
    return fig


def plot_satisfaction(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(data=df, x="Product", y="Customer_Satisfaction", hue="Region", ax=axes[0], errorbar=None)
    axes[0].set(title="Average satisfaction by product and region", ylabel="Satisfaction (1-5)")
    axes[0].legend(fontsize=8)
    sns.heatmap(df[["Sales", "Customer_Age", "Customer_Satisfaction"]].corr(), annot=True,
                cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1])
    axes[1].set(title="Correlation matrix")
    fig.tight_layout()
    return fig


ALL_PLOTS = {
    "Sales trend over time": plot_sales_trend,
    "Quarterly sales": plot_quarterly_sales,
    "Product performance": plot_product_performance,
    "Yearly sales by product": plot_product_trend,
    "Regional analysis": plot_regional_analysis,
    "Customer demographics": plot_demographics,
    "Satisfaction and correlations": plot_satisfaction,
}


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.0):
    """OpenAI chat model. Requires the OPENAI_API_KEY environment variable."""
    from langchain_openai import ChatOpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Set the OPENAI_API_KEY environment variable first.")
    return ChatOpenAI(model=model, temperature=temperature)


def get_embeddings(model: str = "text-embedding-3-small"):
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=model)
