import chromadb
from sentence_transformers import SentenceTransformer
from src.config import settings

class ResearchStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        self.collection = self.client.get_or_create_collection(settings.collection_name)
        self.embedder = SentenceTransformer(settings.embedding_model)
        self.paper_cache = {}

    def count(self):
        return self.collection.count()

    def add_paper(self, paper, chunks):
        self.paper_cache[paper["paper_id"]] = paper
        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            ids.append(f"{paper['paper_id']}_{i}")
            docs.append(chunk["text"])
            metas.append({
                "paper_id": paper["paper_id"], "paper": paper["title"],
                "filename": paper["filename"], "page": int(chunk["page"]),
                "chunk_index": i
            })
        embeddings = self.embedder.encode(docs, normalize_embeddings=True).tolist()
        # Upsert makes re-processing the same paper idempotent.
        self.collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

    def vector_search(self, query, top_k=6):
        if self.count() == 0:
            return []
        emb = self.embedder.encode([query], normalize_embeddings=True).tolist()
        result = self.collection.query(query_embeddings=emb, n_results=min(top_k, self.count()))
        rows = []
        for i, doc in enumerate(result["documents"][0]):
            meta = result["metadatas"][0][i]
            rows.append({"text": doc, "paper": meta["paper"], "paper_id": meta["paper_id"],
                         "page": meta["page"], "filename": meta.get("filename", ""), "score": 1 - result["distances"][0][i]})
        return rows

    def all_chunks(self, limit=5000):
        if self.count() == 0:
            return []
        result = self.collection.get(limit=limit, include=["documents", "metadatas"])
        return [{"text": d, **m} for d, m in zip(result["documents"], result["metadatas"])]

    def list_papers(self):
        rows = {}
        for item in self.all_chunks():
            rows[item["paper_id"]] = {"Paper": item["paper"], "Filename": item["filename"]}
        return [{"Paper": v["Paper"], "Filename": v["Filename"], "Paper ID": k} for k, v in rows.items()]
