import re
from collections import Counter
from src.graph import ResearchGraph

def terms(text):
    return set(re.findall(r"[a-zA-Z0-9_-]{2,}", text.lower()))

class HybridRetriever:
    def __init__(self, store, graph):
        self.store, self.graph = store, graph

    def retrieve(self, query, top_k=6, mode="hybrid"):
        vector = self.store.vector_search(query, top_k=max(top_k * 3, 12))
        if mode == "vector":
            selected = vector[:top_k]
        else:
            all_chunks = self.store.all_chunks()
            qterms = terms(query)
            keyword_scores = []
            for row in all_chunks:
                overlap = len(qterms & terms(row["text"]))
                if overlap:
                    keyword_scores.append((overlap / max(len(qterms), 1), row))
            keyword_scores.sort(key=lambda x: x[0], reverse=True)
            candidates = {}
            # Reciprocal-rank fusion combines semantic and lexical ranking.
            for rank, row in enumerate(vector, start=1):
                key = (row["paper_id"], row["page"], row["text"][:80])
                candidates[key] = {"row": row, "score": candidates.get(key, {}).get("score", 0) + 1/(60+rank)}
            for rank, (_, row) in enumerate(keyword_scores[:top_k*4], start=1):
                key = (row["paper_id"], row["page"], row["text"][:80])
                candidates.setdefault(key, {"row": row, "score": 0})
                candidates[key]["score"] += 1/(60+rank)
            ranked = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)
            seed_ids = [x["row"]["paper_id"] for x in ranked[:top_k]]
            related = self.graph.related_paper_ids(seed_ids)
            # Add graph-neighbor chunks with a modest score boost.
            for row in all_chunks:
                if row["paper_id"] in related and row["paper_id"] not in seed_ids:
                    ranked.append({"row": row, "score": 0.001})
            selected = [x["row"] for x in ranked[:top_k]]
        output = []
        for i, row in enumerate(selected, 1):
            output.append({**row, "citation": f"E{i}"})
        return output
