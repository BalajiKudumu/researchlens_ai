# ResearchLens AI — Hybrid GraphRAG + Multi-Agent Research Assistant

A Streamlit starter implementation for an M.Tech research project on scientific literature analysis and research-gap discovery.

## Features
- Multi-PDF upload and text extraction using PyMuPDF
- Complementary table extraction using pdfplumber
- Chunking, Sentence Transformer embeddings and persistent ChromaDB
- LLM-assisted metadata extraction (Gemini)
- NetworkX knowledge graph for papers, authors, methods, datasets, keywords and findings
- Vector RAG baseline and hybrid semantic + lexical + graph retrieval
- Five sequential research agents: analyzer, literature review, methodology comparison, gap discovery and evidence audit
- Markdown research report export

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your Gemini key to `.env` or enter it in the app sidebar.

Run:

```powershell
streamlit run app.py
```

## Project structure

```text
hybrid_graphrag_research/
├── app.py
├── src/
│   ├── config.py
│   ├── ingestion.py
│   ├── store.py
│   ├── graph.py
│   ├── retrieval.py
│   ├── agents.py
│   └── reporting.py
├── data/                 # created automatically
├── requirements.txt
└── .env.example
```

## Important research limitations and next steps
1. The initial graph is built from LLM-extracted paper metadata. Validate extracted authors, methods, datasets and findings before treating the graph as ground truth.
2. The starter keyword retriever uses token overlap and reciprocal-rank fusion. For a stronger baseline, add BM25 with a curated corpus and tune retrieval weights.
3. PDF tables are extracted and stored in paper metadata, but are not yet independently indexed as table-aware chunks.
4. Scanned PDFs need OCR (for example, Tesseract) before text chunking.
5. The current agent workflow is sequential prompting, not yet a fully stateful LangGraph implementation. This makes the baseline easy to inspect and ablate.
6. Citation verification currently checks whether agents cite retrieved evidence IDs. A proper verifier should also test entailment between each claim and cited passage, ideally with human adjudication.
7. Research-gap discovery cannot guarantee novelty. Validate proposed gaps with systematic database searches and a human-reviewed literature matrix.

## Suggested experimental design
Compare these configurations on the same held-out question set:
- A: Vector RAG
- B: Hybrid semantic + keyword retrieval
- C: Hybrid GraphRAG
- D: Hybrid GraphRAG + multi-agent synthesis + evidence audit

Measure:
- Retrieval: Recall@K, MRR, nDCG@K
- Answer quality: expert-rated correctness, completeness and relevance
- Grounding: faithfulness, citation precision and citation recall
- Efficiency: end-to-end latency, number of LLM calls and token cost

Build a gold dataset with question, relevant paper IDs, relevant passage/page IDs and expert reference answer. Do not compare systems using only LLM self-scores.

## Citation format
The app labels retrieved passages as `[E1]`, `[E2]`, etc. These labels refer to paper title and page shown in the evidence panel. For a dissertation, preserve stable paper IDs and page-level source information in exported reports.

## Security
Do not commit `.env`, API keys, research PDFs or the local `data/` directory to Git.


## Evaluation tab (expanded)
Upload a benchmark CSV with columns:
- `qid`: stable question ID
- `question`: the exact same question sent to each system
- `relevant_filenames`: exact indexed PDF filename(s), separated by semicolons
- `reference_answer`: human-validated reference answer

The app runs Vector RAG, Hybrid GraphRAG, and a two-step Multi-Agent GraphRAG answer (draft + evidence verifier). It calculates Recall@K, Precision@K, MRR, nDCG@K, reference-answer token F1, strict exact match, citation-ID validity, elapsed latency, and rough token estimates. Results are saved locally in `data/evaluation_results.sqlite3` and can be exported as CSV.

Correctness, faithfulness and citation-support ratings are provided as 1–5 human-review fields. Citation-ID validity is not the same as citation entailment; human review is needed to determine whether the cited passage actually supports the claim. Token estimates are character-based approximations, not exact Gemini usage metadata. Use a sufficiently large, human-annotated benchmark and repeat runs before drawing dissertation conclusions.

## Evaluation workflow (updated)

The Evaluation tab now reads saved Research Q&A interactions from `data/research_qa_history.sqlite3`; it no longer requires a separate benchmark CSV upload. Each successful Q&A interaction is saved with its question, generated answer, retrieved evidence, PDF filenames and page numbers. In Evaluation, review/edit the draft reference answer and relevant PDF filenames, then tick **Approved for benchmark** and run the selected questions against Vector RAG, Hybrid GraphRAG and Multi-Agent GraphRAG.

Important: the earlier AI-generated answer is only a draft reference answer, not ground truth. Validate/correct it against the paper before approving. Retrieval relevance filenames should also be reviewed. Existing PDFs and ChromaDB index are reused; do not upload them again. Q&A interactions from an older version are not retroactively available unless they were saved by that version; new interactions are saved automatically.
