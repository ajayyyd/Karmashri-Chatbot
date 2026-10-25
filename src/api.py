"""
FastAPI Statewide Service Endpoint for Karmashri Legal AI.
"""

import sys
from pathlib import Path
from typing import Optional, List

# Ensure project root is available on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.responses import FileResponse

from src.generation import KarmashriLegalAdvisor

# Ensure static asset folder exists
STATIC_DIR = PROJECT_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR = PROJECT_ROOT / "data" / "pdf"

app = FastAPI(
    title="Karmashri Legal AI Engine",
    description="Statewide intelligence platform for Kerala Government Orders",
    version="1.0.0"
)

# Instantiate advisor once on worker start
advisor = KarmashriLegalAdvisor()


class CitationModel(BaseModel):
    source_index: int
    go_number: Optional[str]
    date: Optional[str]
    department: Optional[str]
    parent_path: Optional[str]
    source_file: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationModel]


# Mount static assets directory
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def get_ui():
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return {
        "platform": "Karmashri Legal AI Engine",
        "docs": "/docs",
        "message": "UI asset 'static/index.html' not found. Visit /docs for OpenAPI documentation."
    }

@app.get("/api/v1/pdf/{filename}", include_in_schema=False)
def serve_pdf(filename: str):
    file_path = PDF_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Source PDF not found")
    return FileResponse(file_path, media_type="application/pdf")


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "karmashri-engine"}


# Synchronous 'def' enables FastAPI to offload CPU-heavy Ollama calls to worker threads
@app.get("/api/v1/query", response_model=QueryResponse)
def execute_query(
    q: str = Query(..., description="Administrative question or legal query"),
    department: Optional[str] = Query(None, description="Department Code filter (e.g. GAD, FIN)"),
    top_k: int = Query(4, ge=1, le=10, description="Number of source clauses to synthesize")
):
    try:
        result = advisor.answer_query(q, dept_code=department, top_k=top_k)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))