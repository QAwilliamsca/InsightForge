# InsightForge: AI-Powered Business Intelligence Assistant

UCSB PaCE Advanced Generative AI capstone. LangChain + RAG + LLM over `sales_data.csv` and the four reference PDFs.

## Files
| File | Purpose |
|---|---|
| `InsightForge.ipynb` | Full walkthrough: Part 1 (data, knowledge base, retriever, prompt chaining, RAG, memory) and Part 2 (QAEvalChain, monitoring, visualizations) |
| `app.py` | Streamlit UI: chat, dashboard, data insights, evaluation, monitoring |
| `insightforge.py` | Shared logic used by both |
| `data/` | `sales_data.csv` and the reference PDFs (the PDFs are not in the GitHub repo; download them from the course's Reference Materials into `data/pdfs/`) |
| `logs/monitoring.csv` | Per-question latency, tokens, cost and status (written automatically) |

## Run
```bash
pip install -r requirements.txt
echo 'OPENAI_API_KEY=sk-...' > .env # or: export OPENAI_API_KEY="sk-..." (PowerShell: $env:OPENAI_API_KEY="sk-...")
jupyter notebook InsightForge.ipynb # run all cells
streamlit run app.py                # opens the UI in the browser
```
Without a key, the notebook runs in demo mode: the analysis and charts are real, but the LLM responses are placeholders.
