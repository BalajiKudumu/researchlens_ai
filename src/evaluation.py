import re
import math
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path('data/evaluation_results.sqlite3')
QA_DB_PATH = Path('data/research_qa_history.sqlite3')


def normalize(text):
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def exact_match(prediction, reference):
    return float(normalize(prediction) == normalize(reference)) if normalize(reference) else None

def token_f1(prediction, reference):
    pred, ref = normalize(prediction), normalize(reference)
    if not pred or not ref:
        return 0.0
    from collections import Counter
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def retrieval_metrics(retrieved, relevant_filenames, k=5):
    relevant = {x.strip().lower() for x in relevant_filenames if x.strip()}
    ranked = [str(x.get('filename') or '').strip().lower() for x in retrieved[:k]]
    if not relevant:
        return {f'recall@{k}': None, f'precision@{k}': None, 'mrr': None, f'ndcg@{k}': None}
    hits = [1 if name in relevant else 0 for name in ranked]
    recall = sum(hits) / len(relevant)
    precision = sum(hits) / k
    reciprocal_rank = next((1 / i for i, hit in enumerate(hits, 1) if hit), 0.0)
    dcg = sum(hit / math.log2(i + 1) for i, hit in enumerate(hits, 1))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return {f'recall@{k}': recall, f'precision@{k}': precision,
            'mrr': reciprocal_rank, f'ndcg@{k}': dcg / idcg if idcg else 0.0}


def citation_validity(answer, evidence):
    cited = re.findall(r"\[(E\d+)\]", answer or "")
    valid_ids = {x.get('citation') for x in evidence}
    if not cited:
        return 0.0
    return sum(1 for citation in cited if citation in valid_ids) / len(cited)


def save_results(rows):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS experiment_results (
            run_id TEXT, created_at TEXT, qid TEXT, question TEXT, system TEXT,
            recall_k REAL, precision_k REAL, mrr REAL, ndcg_k REAL,
            answer_f1 REAL, answer_exact_match REAL, citation_validity REAL, latency_seconds REAL,
            estimated_tokens INTEGER, answer TEXT, evidence_json TEXT,
            faithfulness_rating REAL, correctness_rating REAL, citation_support_rating REAL
        )''')
        conn.executemany('''INSERT INTO experiment_results VALUES (
            :run_id,:created_at,:qid,:question,:system,:recall_k,:precision_k,:mrr,:ndcg_k,
            :answer_f1,:answer_exact_match,:citation_validity,:latency_seconds,:estimated_tokens,:answer,:evidence_json,
            :faithfulness_rating,:correctness_rating,:citation_support_rating)''', rows)
        conn.commit()


def load_results():
    if not DB_PATH.exists():
        return []
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute('SELECT * FROM experiment_results ORDER BY created_at DESC')]


def save_qa_record(record):
    """Persist a Research Q&A interaction for later evaluation reuse."""
    QA_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(QA_DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS qa_history (
            qid TEXT PRIMARY KEY, created_at TEXT, question TEXT, answer TEXT,
            evidence_json TEXT, relevant_filenames TEXT, reference_answer TEXT, approved INTEGER DEFAULT 0
        )""")
        conn.execute("""INSERT OR REPLACE INTO qa_history
            (qid, created_at, question, answer, evidence_json, relevant_filenames, reference_answer, approved)
            VALUES (:qid,:created_at,:question,:answer,:evidence_json,:relevant_filenames,:reference_answer,:approved)""", record)
        conn.commit()

def load_qa_history():
    if not QA_DB_PATH.exists():
        return []
    with sqlite3.connect(QA_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute('SELECT * FROM qa_history ORDER BY created_at DESC')]

def update_qa_benchmark(rows):
    QA_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(QA_DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS qa_history (
            qid TEXT PRIMARY KEY, created_at TEXT, question TEXT, answer TEXT,
            evidence_json TEXT, relevant_filenames TEXT, reference_answer TEXT, approved INTEGER DEFAULT 0
        )""")
        conn.executemany("""UPDATE qa_history SET relevant_filenames=:relevant_filenames,
            reference_answer=:reference_answer, approved=:approved WHERE qid=:qid""", rows)
        conn.commit()
