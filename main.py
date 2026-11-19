from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, func
from sqlalchemy.orm import Session

from database import AuditLog, DocumentChunk, get_db, init_db
from services.ingestion import chunk_text, generate_embeddings
from services.retrieval import hybrid_search_rbac
from services.validator import generate_guarded_answer

BASE_DIR = Path(__file__).resolve().parent
POLICY_SOURCE_DIR = BASE_DIR / "data" / "policies"
POLICY_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
UI_FILE = BASE_DIR / "ui.html"

app = FastAPI(title="DocuFlow AI", version="1.0.0")


@app.on_event("startup")
def on_startup() -> None:
    init_db()


class IngestRequest(BaseModel):
    document_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    department_access: List[str] = Field(default_factory=lambda: ["general"])
    sensitivity_level: str = Field(default="internal")


class QueryRequest(BaseModel):
    query_text: str = Field(..., min_length=1)
    user_roles: List[str] = Field(default_factory=list)


def _safe_source_path(filename: str) -> Path:
    clean_name = Path(filename).name
    if not clean_name or clean_name in {".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file name.",
        )
    target = (POLICY_SOURCE_DIR / clean_name).resolve()
    if POLICY_SOURCE_DIR.resolve() not in target.parents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path.",
        )
    return target


def _delete_document_chunks(db: Session, document_id: str) -> int:
    stmt = delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
    result = db.execute(stmt)
    db.commit()
    return int(result.rowcount or 0)


def _store_document_chunks(
    db: Session,
    document_id: str,
    title: str,
    content: str,
    department_access: List[str],
    sensitivity_level: str = "internal",
) -> int:
    _delete_document_chunks(db, document_id)
    chunks = chunk_text(content)
    if not chunks:
        return 0

    embeddings = generate_embeddings(chunks)
    for idx, (chunk_str, emb) in enumerate(zip(chunks, embeddings)):
        row = DocumentChunk(
            document_id=document_id,
            title=title,
            chunk_index=idx,
            content=chunk_str,
            embedding=emb,
            department_access=department_access,
            sensitivity_level=sensitivity_level,
        )
        db.add(row)

    db.commit()
    return len(chunks)


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "service": "docuflow-ai-backend"}


@app.post("/api/v1/ingest")
def api_ingest(payload: IngestRequest, db: Session = Depends(get_db)) -> dict:
    source_file = POLICY_SOURCE_DIR / f"{payload.document_id}.txt"
    source_file.write_text(payload.content, encoding="utf-8")

    chunks_created = _store_document_chunks(
        db=db,
        document_id=payload.document_id,
        title=payload.title,
        content=payload.content,
        department_access=payload.department_access,
        sensitivity_level=payload.sensitivity_level,
    )
    return {
        "status": "ingested",
        "document_id": payload.document_id,
        "source_file": str(source_file.relative_to(BASE_DIR)),
        "chunks_created": chunks_created,
    }


@app.post("/api/v1/ingest/file")
async def api_ingest_file(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(default=None),
    title: Optional[str] = Form(default=None),
    department_access: str = Form(default="general"),
    sensitivity_level: str = Form(default="internal"),
    db: Session = Depends(get_db),
) -> dict:
    target_path = _safe_source_path(file.filename or "uploaded_policy.txt")
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    target_path.write_bytes(raw_bytes)
    try:
        text_content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please upload a UTF-8 encoded .txt or .md policy file.",
        )

    resolved_doc_id = (document_id or "").strip() or target_path.stem
    resolved_title = (title or "").strip() or target_path.stem.replace("_", " ").title()
    roles = [r.strip() for r in department_access.split(",") if r.strip()] or ["general"]

    chunks_created = _store_document_chunks(
        db=db,
        document_id=resolved_doc_id,
        title=resolved_title,
        content=text_content,
        department_access=roles,
        sensitivity_level=sensitivity_level,
    )
    return {
        "status": "ingested",
        "document_id": resolved_doc_id,
        "title": resolved_title,
        "source_file": str(target_path.relative_to(BASE_DIR)),
        "chunks_created": chunks_created,
    }


@app.get("/api/v1/documents")
def list_documents(db: Session = Depends(get_db)) -> dict:
    rows = (
        db.query(
            DocumentChunk.document_id,
            func.min(DocumentChunk.title).label("title"),
            func.count(DocumentChunk.id).label("chunk_count"),
            func.max(DocumentChunk.sensitivity_level).label("sensitivity_level"),
            func.max(DocumentChunk.department_access).label("department_access"),
        )
        .group_by(DocumentChunk.document_id)
        .order_by(DocumentChunk.document_id)
        .all()
    )

    ingested = [
        {
            "document_id": r.document_id,
            "title": r.title,
            "chunk_count": int(r.chunk_count),
            "sensitivity_level": r.sensitivity_level,
            "department_access": r.department_access or [],
        }
        for r in rows
    ]

    source_files = sorted(
        str(p.relative_to(BASE_DIR))
        for p in POLICY_SOURCE_DIR.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )

    return {
        "source_directory": str(POLICY_SOURCE_DIR.relative_to(BASE_DIR)),
        "source_files": source_files,
        "ingested_documents": ingested,
    }


@app.delete("/api/v1/documents/{document_id}")
def delete_document(
    document_id: str,
    remove_file: bool = Query(default=False),
    db: Session = Depends(get_db),
) -> dict:
    deleted_chunks = _delete_document_chunks(db, document_id)
    removed_files: List[str] = []

    if remove_file:
        for candidate in POLICY_SOURCE_DIR.glob(f"{document_id}.*"):
            if candidate.is_file():
                candidate.unlink()
                removed_files.append(str(candidate.relative_to(BASE_DIR)))

    if deleted_chunks == 0 and not removed_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    return {
        "status": "deleted",
        "document_id": document_id,
        "chunks_deleted": deleted_chunks,
        "removed_files": removed_files,
    }


@app.post("/api/v1/query")
def api_query(payload: QueryRequest, db: Session = Depends(get_db)) -> dict:
    retrieved = hybrid_search_rbac(
        db=db,
        query_text=payload.query_text,
        user_roles=payload.user_roles,
    )
    parsed, citations = generate_guarded_answer(
        query_text=payload.query_text,
        retrieved=retrieved,
    )

    try:
        audit_entry = AuditLog(
            query_text=payload.query_text,
            user_roles=payload.user_roles,
            requires_human_approval=bool(parsed.requires_human_approval),
            created_at=datetime.now(timezone.utc),
        )
        db.add(audit_entry)
        db.commit()
    except Exception:
        db.rollback()

    return {
        "answer": parsed.answer,
        "confidence": parsed.confidence,
        "detected_amount": parsed.detected_amount,
        "requires_human_approval": parsed.requires_human_approval,
        "approval_reason": parsed.approval_reason,
        "citations": [c.model_dump() if hasattr(c, "model_dump") else c for c in citations],
    }


@app.get("/ui", response_class=HTMLResponse)
def docuflow_ui() -> str:
    if not UI_FILE.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ui.html not found.",
        )
    return UI_FILE.read_text(encoding="utf-8")