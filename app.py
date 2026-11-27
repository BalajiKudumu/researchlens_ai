import streamlit as st
from src.config import settings
from src.ingestion import ingest_pdfs
from src.store import ResearchStore
from src.graph import ResearchGraph
from src.retrieval import HybridRetriever
from src.agents import ResearchAgentTeam
from src.reporting import build_markdown_report
from src.evaluation import (retrieval_metrics, token_f1, exact_match, citation_validity,
    save_results, load_results, save_qa_record, load_qa_history, update_qa_benchmark)
import pandas as pd
import json, time, uuid
from datetime import datetime

st.set_page_config(page_title="ResearchLens AI", page_icon="🔬", layout="wide")
st.title("🔬 ResearchLens AI")
st.caption("Hybrid GraphRAG and Multi-Agent AI for Scientific Research and Research Gap Discovery")

@st.cache_resource
def services():
    store = ResearchStore()
    graph = ResearchGraph()
    return store, graph

store, graph = services()
retriever = HybridRetriever(store, graph)
agents = ResearchAgentTeam(retriever)

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API key", type="password", value=settings.gemini_api_key)
    model = st.selectbox("Gemini model", ["gemini-3.5-flash-lite", "gemini-3.8-flash"], index=0)
    st.caption("Your key is used locally and is not saved by this application.")
    st.divider()
    st.metric("Indexed chunks", store.count())
    st.metric("Graph nodes", graph.node_count())
    st.metric("Graph edges", graph.edge_count())

tab_upload, tab_library, tab_chat, tab_agents, tab_graph, tab_eval = st.tabs([
    "📄 Upload papers", "📚 Paper library", "💬 Research Q&A",
    "🤖 Research agents", "🕸️ Knowledge graph", "📊 Evaluation"
])

with tab_upload:
    st.subheader("Upload scientific papers")
    uploads = st.file_uploader("Select one or more PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("Process and index papers", type="primary", disabled=not uploads):
        with st.spinner("Extracting text, chunking, embedding and building graph..."):
            result = ingest_pdfs(uploads, store, graph, api_key=api_key, model=model)
        st.success(f"Processed {result['papers']} paper(s), created {result['chunks']} chunks and {result['entities']} graph entities.")
        if result["warnings"]:
            for warning in result["warnings"]:
                st.warning(warning)

with tab_library:
    st.subheader("Indexed research papers")
    papers = store.list_papers()
    if not papers:
        st.info("No papers indexed yet. Upload PDFs to get started.")
    else:
        st.dataframe(papers, width=True, hide_index=True)

with tab_chat:
    st.subheader("Ask questions across your research collection")
    mode = st.radio("Retrieval strategy", ["Vector RAG", "Hybrid GraphRAG"], horizontal=True)
    question = st.text_area("Research question", placeholder="Compare the approaches used for privacy-preserving federated learning.")
    top_k = st.slider("Evidence passages", 3, 12, 6)
    if st.button("Retrieve evidence and answer", type="primary", disabled=not question.strip()):
        with st.spinner("Retrieving evidence and generating an answer..."):
            evidence = retriever.retrieve(question, top_k=top_k, mode="hybrid" if mode == "Hybrid GraphRAG" else "vector")
            answer = agents.answer_question(question, evidence, api_key, model)
        st.session_state["last_qa"] = {"question": question, "answer": answer, "evidence": evidence, "created_at": datetime.now().isoformat(timespec="seconds")}
        st.markdown(answer)
        # Persist the interaction so Evaluation can reuse it without retyping questions or uploading PDFs.
        qa = st.session_state["last_qa"]
        qid = "QA-" + str(uuid.uuid4())[:8]
        evidence_sources = sorted({e.get("filename", "") for e in evidence if e.get("filename")})
        save_qa_record({
            "qid": qid, "created_at": qa["created_at"], "question": question, "answer": answer,
            "evidence_json": json.dumps(evidence, ensure_ascii=False),
            "relevant_filenames": ";".join(evidence_sources),
            # Prior model answer is a draft reference only; user must validate it before benchmarking.
            "reference_answer": answer, "approved": 0
        })
        st.success(f"Question saved to Evaluation history as {qid}. You can validate it in the Evaluation tab.")
        with st.expander("Retrieved evidence and citations"):
            for item in evidence:
                st.markdown(f"**[{item['citation']}] {item['paper']} — {item['filename']} — page {item['page']}**")
                st.write(item["text"])

with tab_agents:
    st.subheader("Run the multi-agent research workflow")
    research_topic = st.text_area("Research topic", placeholder="Graph neural networks for drug discovery")
    if st.button("Run all research agents", type="primary", disabled=not research_topic.strip()):
        with st.spinner("Research agents are analyzing the indexed literature..."):
            result = agents.run_research_workflow(research_topic, api_key, model)
        st.session_state["last_report"] = result
        st.success("Research workflow completed.")
    if "last_report" in st.session_state:
        report = st.session_state["last_report"]
        for section, content in report.items():
            with st.expander(section, expanded=True):
                st.markdown(content)
        md = build_markdown_report(research_topic, report)
        st.download_button("Download research report (.md)", md, file_name="research_report.md", mime="text/markdown")

with tab_graph:
    st.subheader("Research knowledge graph")
    graph_data = graph.to_table()
    if graph_data:
        st.dataframe(graph_data, width=True, hide_index=True)
        st.caption("Nodes and relationships are extracted from paper metadata and LLM-assisted paper analysis.")
    else:
        st.info("The graph is empty. Process research papers first.")

with tab_eval:
    st.subheader("Research experiment runner")
    st.write("Reuse questions already asked in Research Q&A. Your indexed PDFs and saved question history are shared automatically.")

    qa_history = load_qa_history()
    if not qa_history:
        st.info("No saved questions yet. Go to Research Q&A, ask questions, and they will appear here automatically.")
    else:
        st.markdown("### Questions from Research Q&A")
        st.caption("The previous AI answer is prefilled as a draft reference answer, NOT ground truth. Review and correct the reference answer and relevant filenames, then approve each question before running the benchmark.")
        qa_df = pd.DataFrame(qa_history)
        editable = qa_df[["qid", "question", "relevant_filenames", "reference_answer", "approved"]].copy()
        edited = st.data_editor(
            editable, width=True, hide_index=True,
            column_config={
                "qid": st.column_config.TextColumn("Question ID", disabled=True),
                "question": st.column_config.TextColumn("Previously asked question", disabled=True, width="large"),
                "relevant_filenames": st.column_config.TextColumn("Relevant PDF filenames (semicolon-separated)", width="large"),
                "reference_answer": st.column_config.TextColumn("Validated reference answer (edit the draft)", width="large"),
                "approved": st.column_config.CheckboxColumn("Approved for benchmark"),
            },
            disabled=["qid", "question"]
        )
        if st.button("Save benchmark validation", type="secondary"):
            updates = []
            for _, row in edited.iterrows():
                updates.append({
                    "qid": row["qid"],
                    "relevant_filenames": str(row["relevant_filenames"] or ""),
                    "reference_answer": str(row["reference_answer"] or ""),
                    "approved": int(bool(row["approved"]))
                })
            update_qa_benchmark(updates)
            st.success("Your benchmark corrections and approvals have been saved.")
            st.rerun()

        approved_ids = edited.loc[edited["approved"].fillna(False), "qid"].astype(str).tolist()
        selected_ids = st.multiselect("Choose saved questions to include", options=approved_ids, default=approved_ids,
                                      help="Only approved questions are available. No CSV upload is required.")
        top_k_eval = st.slider("K for retrieval metrics", min_value=3, max_value=15, value=5, key="eval_k")
        run_eval = st.button("Run benchmark on selected saved questions", type="primary",
                             disabled=(not selected_ids or store.count() == 0))

        if run_eval:
            if not api_key:
                st.error("Enter your Gemini API key in the sidebar before running answer-quality experiments.")
            else:
                selected_rows = edited[edited["qid"].astype(str).isin(selected_ids)]
                run_id = str(uuid.uuid4())[:8]
                created_at = datetime.now().isoformat(timespec="seconds")
                result_rows = []
                progress = st.progress(0.0, text="Starting experiments...")
                total = max(len(selected_rows) * 3, 1)
                done = 0
                history_by_id = {x["qid"]: x for x in qa_history}
                systems = [("Vector RAG", "vector", False), ("Hybrid GraphRAG", "hybrid", False),
                           ("Multi-Agent GraphRAG", "hybrid", True)]
                for _, question_row in selected_rows.iterrows():
                    qid = str(question_row["qid"])
                    question = str(question_row["question"])
                    relevant = [x.strip() for x in str(question_row["relevant_filenames"]).split(";") if x.strip()]
                    reference = str(question_row["reference_answer"] or "")
                    for system_name, mode, multi_agent in systems:
                        started = time.perf_counter()
                        evidence = retriever.retrieve(question, top_k=top_k_eval, mode=mode)
                        answer = (agents.multi_agent_answer(question, evidence, api_key, model) if multi_agent
                                  else agents.answer_question(question, evidence, api_key, model))
                        latency = time.perf_counter() - started
                        metrics = retrieval_metrics(evidence, relevant, k=top_k_eval)
                        result_rows.append({
                            "run_id": run_id, "created_at": created_at, "qid": qid, "question": question,
                            "system": system_name, "recall_k": metrics[f"recall@{top_k_eval}"],
                            "precision_k": metrics[f"precision@{top_k_eval}"], "mrr": metrics["mrr"],
                            "ndcg_k": metrics[f"ndcg@{top_k_eval}"],
                            "answer_f1": token_f1(answer, reference) if reference else None,
                            "answer_exact_match": exact_match(answer, reference) if reference else None,
                            "citation_validity": citation_validity(answer, evidence),
                            "latency_seconds": round(latency, 3),
                            "estimated_tokens": int((len(question) + len(answer) + sum(len(e["text"]) for e in evidence)) / 4),
                            "answer": answer,
                            "evidence_json": json.dumps([{"citation": e.get("citation"), "paper": e.get("paper"),
                                "filename": e.get("filename"), "page": e.get("page")} for e in evidence]),
                            "faithfulness_rating": None, "correctness_rating": None, "citation_support_rating": None,
                        })
                        done += 1
                        progress.progress(done / total, text=f"Completed {done}/{total} system runs")
                save_results(result_rows)
                st.session_state["latest_eval_rows"] = result_rows
                st.success(f"Benchmark completed. Run ID: {run_id}")

    latest = st.session_state.get("latest_eval_rows", [])
    if latest:
        st.markdown("### Latest experiment results")
        result_df = pd.DataFrame(latest)
        summary = result_df.groupby("system").agg(
            **{"Recall@K": ("recall_k", "mean"), "Precision@K": ("precision_k", "mean"),
               "MRR": ("mrr", "mean"), "nDCG@K": ("ndcg_k", "mean"),
               "Answer token-F1": ("answer_f1", "mean"), "Exact match": ("answer_exact_match", "mean"), "Citation validity": ("citation_validity", "mean"),
               "Latency (sec)": ("latency_seconds", "mean"), "Estimated tokens": ("estimated_tokens", "mean")}
        ).reset_index()
        st.dataframe(summary, width=True, hide_index=True)
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.markdown("**Retrieval comparison**")
            st.bar_chart(summary.set_index("system")[["Recall@K", "MRR", "nDCG@K"]])
        with chart_right:
            st.markdown("**Efficiency comparison**")
            st.bar_chart(summary.set_index("system")[["Latency (sec)", "Estimated tokens"]])
        st.caption("Answer token-F1 measures lexical overlap with your reference answer; exact match is deliberately strict. Neither is a complete semantic correctness measure. Citation validity checks whether citation IDs exist; it does not prove that the cited passage entails the claim. Estimated tokens are rough character-based estimates, not provider-reported token counts.")
        st.markdown("### Human assessment")
        st.write("Review each generated answer and enter ratings from 1 (poor) to 5 (excellent). These ratings provide the defensible human assessment of correctness, faithfulness and citation support.")
        review_df = result_df[["qid", "system", "question", "correctness_rating", "faithfulness_rating", "citation_support_rating"]].copy()
        edited = st.data_editor(review_df, width=True, hide_index=True,
            column_config={
                "correctness_rating": st.column_config.NumberColumn("Correctness (1–5)", min_value=1, max_value=5, step=1),
                "faithfulness_rating": st.column_config.NumberColumn("Faithfulness (1–5)", min_value=1, max_value=5, step=1),
                "citation_support_rating": st.column_config.NumberColumn("Citation support (1–5)", min_value=1, max_value=5, step=1),
            }, disabled=["qid", "system", "question"])
        if st.button("Save human assessment", key="save_human_eval"):
            # Keep the saved benchmark rows in sync with the editable review table.
            for i, row in edited.iterrows():
                latest[i]["correctness_rating"] = None if pd.isna(row["correctness_rating"]) else float(row["correctness_rating"])
                latest[i]["faithfulness_rating"] = None if pd.isna(row["faithfulness_rating"]) else float(row["faithfulness_rating"])
                latest[i]["citation_support_rating"] = None if pd.isna(row["citation_support_rating"]) else float(row["citation_support_rating"])
            # Save a new assessment snapshot; previous raw run remains auditable.
            save_results(latest)
            st.success("Human assessment saved to the local experiment database.")
        st.markdown("### Generated answers and evidence trace")
        for row in latest:
            with st.expander(f"{row['qid']} · {row['system']}"):
                st.markdown(row["answer"])
                st.caption(f"Latency: {row['latency_seconds']} sec · Estimated tokens: {row['estimated_tokens']}")
                st.json(json.loads(row["evidence_json"]))
        st.download_button("Download latest experiment results (CSV)", result_df.to_csv(index=False).encode("utf-8"),
                           file_name="research_experiment_results.csv", mime="text/csv")

    st.markdown("### Previous experiment history")
    history = load_results()
    if history:
        history_df = pd.DataFrame(history)
        st.dataframe(history_df.drop(columns=["answer", "evidence_json"], errors="ignore"), width=True, hide_index=True)
        st.download_button("Download all saved experiment records", history_df.to_csv(index=False).encode("utf-8"),
                           file_name="all_research_experiments.csv", mime="text/csv")
    else:
        st.info("No experiment runs saved yet. Ask questions in Research Q&A, validate them above, and run your first benchmark.")
