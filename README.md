# 🔬 ResearchLens AI
## Hybrid GraphRAG and Multi-Agent AI for Scientific Research and Research Gap Discovery

ResearchLens AI is an AI-powered scientific research assistant designed to help researchers analyze academic papers, retrieve relevant information, understand relationships between research concepts, and identify potential research gaps.

The application combines **Vector Retrieval-Augmented Generation (Vector RAG), Knowledge Graphs, Hybrid GraphRAG, and Multi-Agent AI** to provide a structured approach to scientific literature analysis and research discovery.

---

## 📌 Project Overview

Scientific research often involves reviewing large volumes of academic literature, identifying relationships between research contributions, understanding existing limitations, and discovering unexplored research opportunities.

Traditional literature review methods can be time-consuming and may not effectively capture relationships between concepts across multiple research papers.

ResearchLens AI addresses these challenges through a hybrid retrieval and multi-agent architecture that integrates semantic search, graph-based knowledge representation, and collaborative AI analysis.

### Key Objectives

- Automate the ingestion and processing of scientific research papers.
- Extract meaningful information from academic PDF documents.
- Enable semantic retrieval using vector embeddings.
- Represent relationships between research concepts through knowledge graphs.
- Combine vector-based and graph-based retrieval for contextual research analysis.
- Use multiple AI agents to investigate research topics from different perspectives.
- Identify potential research gaps and suggest future research directions.
- Generate structured research reports with supporting references.

---

## ✨ Key Features

### 1. Research Paper Upload and Processing
- Upload academic research papers in PDF format.
- Extract document text using PyMuPDF.
- Split documents into manageable text chunks.
- Generate semantic embeddings for document retrieval.
- Store processed document information for subsequent analysis.

### 2. Vector RAG
- Semantic search across uploaded research papers.
- Retrieve relevant document chunks based on user queries.
- Generate context-aware responses using retrieved information.

### 3. Knowledge Graph Construction
- Identify important research concepts and entities.
- Represent relationships between concepts.
- Build a graph-based representation of scientific literature.
- Explore connections between research topics and findings.

### 4. Hybrid GraphRAG
- Combine vector-based semantic retrieval with knowledge graph retrieval.
- Retrieve relevant information from both document chunks and graph relationships.
- Improve contextual understanding of research questions.
- Support relationship-aware scientific literature analysis.

### 5. Multi-Agent AI Research Workflow
The multi-agent research workflow coordinates AI-driven analysis to investigate research topics.

The workflow is designed to support tasks such as:

- Research topic analysis.
- Scientific literature investigation.
- Methodology comparison.
- Identification of limitations.
- Research gap discovery.
- Future research direction generation.
- Structured research report preparation.

### 6. Research Question Answering
- Ask questions about uploaded scientific papers.
- Retrieve relevant evidence from the indexed literature.
- Generate contextual answers.
- Support literature-based research investigation.

### 7. Research Evaluation
The application includes an evaluation module for assessing retrieval and answer-generation performance.

Evaluation metrics include:

| Category | Evaluation Metrics |
|---|---|
| Retrieval Performance | Precision@K, Recall@K, MRR, nDCG@K |
| Answer Quality | Answer Correctness, Completeness, Faithfulness |
| Evidence Quality | Citation Validity, Evidence Support |
| Text Similarity | Exact Match, Token F1 |
| Performance | Response Latency, Token Consumption |

---

## 🏗️ System Architecture

The ResearchLens AI architecture consists of the following major components:

1. **User Interface:** Streamlit-based interface for document uploads, research queries, graph exploration, and evaluation.
2. **Document Ingestion:** Extracts and processes academic research papers.
3. **Embedding and Vector Storage:** Generates semantic embeddings and stores document chunks using ChromaDB.
4. **Knowledge Graph:** Represents entities and relationships extracted from scientific literature.
5. **Hybrid Retrieval:** Combines semantic vector retrieval and graph-based retrieval.
6. **Multi-Agent Research Team:** Coordinates AI agents to perform research analysis and research gap discovery.
7. **Report Generation:** Produces structured research findings and reports.
8. **Evaluation Module:** Measures retrieval quality, answer quality, evidence support, and response performance.

### Research Workflow

```text
             Research Paper Upload
                      |
                      v
             PDF Text Extraction
                      |
                      v
              Text Chunking
                      |
                      v
             Embedding Generation
                      |
             +--------+--------+
             |                 |
             v                 v
       Vector Database    Knowledge Graph
          (ChromaDB)       (Relationships)
             |                 |
             +--------+--------+
                      |
                      v
               Hybrid Retrieval
                      |
                      v
                Research Query
                      |
                      v
              Multi-Agent AI
                 Workflow
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
       Research    Gap Analysis  Future
       Analysis                  Directions
          |           |           |
          +-----------+-----------+
                      |
                      v
             Research Report
                      |
                      v
                Evaluation
```

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend / User Interface | Streamlit |
| Large Language Model | Google Gemini |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| PDF Processing | PyMuPDF |
| Retrieval Architecture | Vector RAG and Hybrid GraphRAG |
| AI Architecture | Multi-Agent AI |
| Evaluation | Custom retrieval and answer evaluation metrics |

---

## 📂 Project Structure

```text
Hybrid_GraphRAG_Research/
│
├── app.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion.py
│   ├── store.py
│   ├── graph.py
│   ├── retrieval.py
│   ├── agents.py
│   ├── reporting.py
│   └── evaluation.py
│
├── data/
│   └── Uploaded research papers
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

*Note: The directory structure above represents the main application modules. Adjust it to match the final structure of your GitHub repository.*

---

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or later (use the version supported by your dependencies).
- pip package manager.
- Google Gemini API key.
- Git.

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate to the project directory:

```bash
cd Hybrid_GraphRAG_Research
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Gemini API Key

Obtain a Google Gemini API key and configure it through the API key input available in the ResearchLens AI application.

Keep API keys private. Do not commit API keys or other credentials to GitHub.

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will start in your default browser.

---

## 🚀 How to Use ResearchLens AI

1. Launch the Streamlit application.
2. Configure your Gemini API key.
3. Navigate to the **Upload Papers** section.
4. Upload one or more scientific research papers in PDF format.
5. Allow the application to process and index the documents.
6. Explore the indexed papers in the Paper Library.
7. Use Research Q&A to ask questions about the uploaded literature.
8. Explore relationships between research concepts through the Knowledge Graph.
9. Navigate to Research Agents and enter a research topic.
10. Run the multi-agent research workflow to investigate the topic and identify potential research gaps.
11. Review the generated research findings and reports.
12. Use the Evaluation section to assess the system's performance.

---

## 📊 Research Methodology and Comparative Analysis

The project explores three retrieval and reasoning approaches:

| Approach | Description |
|---|---|
| Vector RAG | Retrieves relevant research information using semantic similarity. |
| Hybrid GraphRAG | Combines semantic retrieval with graph-based relationships. |
| Multi-Agent GraphRAG | Extends hybrid retrieval through coordinated AI-agent research and analysis. |

The comparative evaluation focuses on retrieval performance, answer quality, evidence support, research analysis, and system efficiency.

---

## 🎯 Expected Outcomes

- Efficient processing and analysis of scientific research papers.
- Improved contextual retrieval through combined vector and graph-based approaches.
- Better understanding of relationships between scientific concepts.
- Structured multi-agent analysis of research topics.
- Identification of potential research gaps based on available literature.
- Generation of evidence-supported research reports.
- Quantitative comparison of Vector RAG, Hybrid GraphRAG, and Multi-Agent GraphRAG.

---

## 🔬 Research Scope

The project focuses on applying Hybrid GraphRAG and Multi-Agent AI to scientific literature analysis and research gap discovery.

The system is intended to assist researchers and postgraduate students in literature exploration, methodology comparison, research problem identification, and future research planning.

Research gap suggestions should be validated against a sufficiently broad collection of relevant academic publications before being treated as novel research contributions.

---

## 👨‍💻 Author

**Balaji Kudumu**

M.Tech – Artificial Intelligence and Data Science  
Indian Institute of Technology (IIT) Patna

**Project:** Hybrid GraphRAG and Multi-Agent AI for Scientific Research and Research Gap Discovery

---

## 📜 License

This project is developed for academic and research purposes. Add an appropriate open-source license if you intend to distribute the source code publicly.
