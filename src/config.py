import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chroma_path: str = os.getenv("CHROMA_PATH", "./data/chroma")
    graph_path: str = os.getenv("GRAPH_PATH", "./data/research_graph.json")
    collection_name: str = os.getenv("COLLECTION_NAME", "research_papers")

settings = Settings()
