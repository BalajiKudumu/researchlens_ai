import io, re, hashlib
from pathlib import Path
import fitz
import pdfplumber
from sentence_transformers import SentenceTransformer
from src.config import settings

def extract_pdf(uploaded_file):
    raw = uploaded_file.getvalue()
    doc = fitz.open(stream=raw, filetype="pdf")
    pages = []
    for index, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page": index + 1, "text": text})
    metadata = dict(doc.metadata or {})
    doc.close()

    # pdfplumber is used as a complementary table extractor.
    tables = []
    try:
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for pno, page in enumerate(pdf.pages, start=1):
                for table in (page.extract_tables() or []):
                    cleaned = [[cell or "" for cell in row] for row in table]
                    if cleaned:
                        tables.append({"page": pno, "rows": cleaned})
    except Exception:
        pass

    return raw, pages, metadata, tables

def chunk_pages(pages, chunk_size=1100, overlap=180):
    chunks = []
    for page in pages:
        text = re.sub(r"\s+", " ", page["text"]).strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            piece = text[start:end].strip()
            if piece:
                chunks.append({"text": piece, "page": page["page"]})
            if end == len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks

def analyze_paper_with_gemini(paper_text, api_key, model):
    if not api_key:
        return {}
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        prompt = f"""
Extract research metadata from this scientific paper. Return ONLY valid JSON with keys:
title, authors (array), year, research_problem, methods (array), datasets (array),
key_findings (array), limitations (array), keywords (array), references (array of strings).
Do not invent missing values. Use empty arrays or empty strings when unknown.
PAPER TEXT:
{paper_text[:18000]}
"""
        response = client.models.generate_content(model=model, contents=prompt)
        import json
        text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(text)
    except Exception:
        return {}

def ingest_pdfs(uploads, store, graph, api_key="", model="gemini-2.5-flash"):
    total_chunks, entities, warnings = 0, 0, []
    for upload in uploads:
        try:
            raw, pages, metadata, tables = extract_pdf(upload)
            chunks = chunk_pages(pages)
            if not chunks:
                warnings.append(f"{upload.name}: no selectable text found; OCR may be needed for scanned PDFs.")
                continue
            paper_id = hashlib.sha1(raw).hexdigest()[:12]
            full_text = "\n".join(p["text"] for p in pages)
            analysis = analyze_paper_with_gemini(full_text, api_key, model)
            title = analysis.get("title") or metadata.get("title") or upload.name
            paper = {
                "paper_id": paper_id, "filename": upload.name, "title": title,
                "authors": analysis.get("authors", []), "year": str(analysis.get("year") or ""),
                "methods": analysis.get("methods", []), "datasets": analysis.get("datasets", []),
                "findings": analysis.get("key_findings", []), "limitations": analysis.get("limitations", []),
                "problem": analysis.get("research_problem", ""), "keywords": analysis.get("keywords", []),
                "references": analysis.get("references", []), "tables": tables
            }
            store.add_paper(paper, chunks)
            entities += graph.add_paper(paper)
            total_chunks += len(chunks)
        except Exception as exc:
            warnings.append(f"{upload.name}: {exc}")
    return {"papers": len(uploads), "chunks": total_chunks, "entities": entities, "warnings": warnings}
