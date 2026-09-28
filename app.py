"""InsightForge - Streamlit UI.   Run with:  streamlit run app.py"""
import os
import uuid
import warnings

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import insightforge as ifg

st.set_page_config(page_title="InsightForge", page_icon="📊", layout="wide")


# ---------------------------------------------------------------- cached resources
@st.cache_data
def load_df():
    return ifg.load_data()


@st.cache_resource(show_spinner="Indexing reference PDFs...")
def load_pdf_retriever(_api_key: str):
    return ifg.build_pdf_vectorstore(ifg.get_embeddings()).as_retriever(search_kwargs={"k": 3})


@st.cache_resource
def load_assistant(_api_key: str, model: str, use_pdfs: bool, recommendations: bool):
    retriever = load_pdf_retriever(_api_key) if use_pdfs else None
    return ifg.InsightForgeAssistant(ifg.get_llm(model), load_df(), retriever,
                                     with_recommendations=recommendations)


df = load_df()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📊 InsightForge")
    st.caption("AI-powered business intelligence assistant")
    key = st.text_input("OpenAI API key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    if key:
        os.environ["OPENAI_API_KEY"] = key
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])
    use_pdfs = st.checkbox("Use reference PDFs (RAG)", value=True)
    recs = st.checkbox("Add recommendations", value=True)
    st.divider()
    st.subheader("Filters (dashboard)")
    regions = st.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    products = st.multiselect("Product", sorted(df["Product"].unique()), default=sorted(df["Product"].unique()))
    years = st.slider("Year", int(df["Year"].min()), int(df["Year"].max()),
                      (int(df["Year"].min()), int(df["Year"].max())))
    if st.button("Clear chat memory"):
        st.session_state.pop("messages", None)
        st.session_state["session_id"] = str(uuid.uuid4())

st.session_state.setdefault("session_id", str(uuid.uuid4()))
st.session_state.setdefault("messages", [])

fdf = df[df["Region"].isin(regions) & df["Product"].isin(products) & df["Year"].between(*years)]

tab_chat, tab_dash, tab_data, tab_eval, tab_mon = st.tabs(
    ["💬 Chat", "📈 Dashboard", "🔎 Data Insights", "✅ Evaluation", "🩺 Monitoring"])

# ---------------------------------------------------------------- chat
with tab_chat:
    st.header("Ask InsightForge")
    if not key:
        st.info("Enter your OpenAI API key in the sidebar to chat.")
    examples = ["Which product sells best and where?",
                "How have sales trended over the years?",
                "Which customer segment should we target?",
                "What is the median and standard deviation of sales?"]
    cols = st.columns(len(examples))
    clicked = next((q for c, q in zip(cols, examples) if c.button(q, use_container_width=True)), None)

    for m in st.session_state["messages"]:
        st.chat_message(m["role"]).markdown(m["content"])

    question = st.chat_input("Ask about sales, products, regions, customers...") or clicked
    if question and key:
        st.session_state["messages"].append({"role": "user", "content": question})
        st.chat_message("user").markdown(question)
        with st.chat_message("assistant"), st.spinner("Analyzing..."):
            try:
                answer = load_assistant(key, model, use_pdfs, recs).ask(
                    question, session_id=st.session_state["session_id"])
            except Exception as e:  # show API errors instead of crashing
                answer = f"⚠️ Error: {e}"
            st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})

# ---------------------------------------------------------------- dashboard
with tab_dash:
    st.header("Sales dashboard")
    if fdf.empty:
        st.warning("No data for the selected filters.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total sales", f"{fdf['Sales'].sum():,.0f}")
        k2.metric("Transactions", f"{len(fdf):,}")
        k3.metric("Median sale", f"{fdf['Sales'].median():,.0f}")
        k4.metric("Avg satisfaction", f"{fdf['Customer_Satisfaction'].mean():.2f} / 5")
        for name, fn in ifg.ALL_PLOTS.items():
            st.subheader(name)
            fig = fn(fdf)
            st.pyplot(fig)
            plt.close(fig)

# ---------------------------------------------------------------- data insights
with tab_data:
    st.header("Computed statistics (the assistant's knowledge base)")
    for topic, text in ifg.compute_summaries(fdf if not fdf.empty else df).items():
        with st.expander(topic.replace("_", " ").title()):
            st.code(text)
    with st.expander("Raw data"):
        st.dataframe(fdf, use_container_width=True)

# ---------------------------------------------------------------- evaluation
with tab_eval:
    st.header("Model evaluation (QAEvalChain)")
    st.write("Ground-truth answers are computed with pandas; an LLM grader compares each "
             "assistant answer against them.")
    st.dataframe(ifg.build_eval_set(df), use_container_width=True)
    if st.button("Run evaluation", disabled=not key):
        with st.spinner("Evaluating..."):
            asst = ifg.InsightForgeAssistant(ifg.get_llm(model), df, with_recommendations=False)
            res = ifg.evaluate_with_qaevalchain(asst, ifg.get_llm(model))
        acc = (res["grade"] == "CORRECT").mean()
        st.metric("Accuracy", f"{acc:.0%}")
        st.dataframe(res, use_container_width=True)

# ---------------------------------------------------------------- monitoring
with tab_mon:
    st.header("Monitoring")
    st.write("Every question answered by the assistant (chat, evaluation and notebook runs) is logged "
             "with its latency, token usage, estimated OpenAI cost and status.")
    log = ifg.load_monitoring_log()
    if log.empty:
        st.info("No questions logged yet. Ask something in the Chat tab.")
    else:
        m = ifg.monitoring_summary(log)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Questions", f"{m['questions']:,}")
        c2.metric("Avg latency", f"{m['avg_latency_s']:.1f} s")
        c3.metric("P95 latency", f"{m['p95_latency_s']:.1f} s")
        c4.metric("Avg tokens / question", f"{m['avg_tokens']:,.0f}")
        c5.metric("Total cost", f"${m['total_cost_usd']:.4f}")
        if m["error_rate"] > 0:
            st.warning(f"Error rate: {m['error_rate']:.0%}")
        fig = ifg.plot_monitoring(log)
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(log.sort_values("timestamp", ascending=False), use_container_width=True)
