import json
from pathlib import Path
import networkx as nx
from src.config import settings

class ResearchGraph:
    def __init__(self):
        self.path = Path(settings.graph_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.graph = nx.MultiDiGraph()
        if self.path.exists():
            try:
                self.graph = nx.node_link_graph(json.loads(self.path.read_text(encoding="utf-8")))
            except Exception:
                self.graph = nx.MultiDiGraph()

    def _save(self):
        self.path.write_text(json.dumps(nx.node_link_data(self.graph), ensure_ascii=False, indent=2), encoding="utf-8")

    def add_paper(self, paper):
        pid = "paper:" + paper["paper_id"]
        self.graph.add_node(pid, kind="paper", label=paper["title"])
        count = 1
        fields = [("authors", "author", "AUTHORED_BY"), ("methods", "method", "USES_METHOD"),
                  ("datasets", "dataset", "USES_DATASET"), ("keywords", "keyword", "TAGGED_WITH")]
        for field, kind, relation in fields:
            for value in paper.get(field, []) or []:
                value = str(value).strip()
                if not value:
                    continue
                node = f"{kind}:{value.lower()}"
                self.graph.add_node(node, kind=kind, label=value)
                self.graph.add_edge(pid, node, relation=relation)
                count += 1
        for finding in paper.get("findings", []) or []:
            value = str(finding).strip()
            if value:
                node = f"finding:{paper['paper_id']}:{len(self.graph)}"
                self.graph.add_node(node, kind="finding", label=value)
                self.graph.add_edge(pid, node, relation="REPORTS_FINDING")
                count += 1
        self._save()
        return count

    def related_paper_ids(self, paper_ids):
        found = set(paper_ids)
        for pid in list(paper_ids):
            node = "paper:" + pid
            if node not in self.graph:
                continue
            for neighbor in self.graph.neighbors(node):
                for other in self.graph.predecessors(neighbor):
                    if self.graph.nodes[other].get("kind") == "paper":
                        found.add(other.split(":", 1)[1])
        return found

    def node_count(self):
        return self.graph.number_of_nodes()

    def edge_count(self):
        return self.graph.number_of_edges()

    def to_table(self):
        rows = []
        for u, v, data in self.graph.edges(data=True):
            rows.append({"Source": self.graph.nodes[u].get("label", u),
                         "Relationship": data.get("relation", ""),
                         "Target": self.graph.nodes[v].get("label", v)})
        return rows[:1000]
