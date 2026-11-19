# DocuFlow AI

DocuFlow AI is a FastAPI-based policy knowledge and compliance query service built on PostgreSQL (vector + full-text hybrid search via RRF), role-based access control (RBAC), human-approval guardrails, and a browser console (`/ui`) for managing policy documents and running queries.

---

## Architecture Overview

* **API & Web Console (`main.py` + `ui.html`):** Exposes `/health`, `/api/v1/ingest`, `/api/v1/ingest/file`, `/api/v1/documents`, `/api/v1/documents/{document_id}`, `/api/v1/query`, and `/ui`.
* **Server Policy Source Store (`data/policies/`):** Saves raw policy files on disk so every ingested policy has a traceable origin on the server.
* **Ingestion & Embeddings (`services/ingestion.py`):** Splits policy text into chunks (`chunk_text`) and generates vector embeddings (`generate_embeddings`).
* **Hybrid Retrieval & RBAC (`services/retrieval.py`):** Runs `hybrid_search_rbac` scoped by `user_roles` so callers only retrieve policies permitted for their role(s).
* **Guarded Answer & Citations (`services/validator.py`):** Runs `generate_guarded_answer` to return structured answers, confidence, detected monetary amounts, `requires_human_approval`, approval reasons, and source citations.
* **Database & Audit Trail (`database.py`):** Manages `DocumentChunk` records, `AuditLog` entries, and database initialization (`init_db`).

---

## End-to-End Workflow

1. **Add & Ingest Policies:**
   * Submit raw text via `POST /api/v1/ingest` (or the text form in `/ui`) or upload a `.txt` / `.md` file via `POST /api/v1/ingest/file` (or the file upload form in `/ui`).
   * The raw file is saved under `data/policies/` on the server, split into chunks, embedded, and stored in PostgreSQL under `document_id`. Re-ingesting the same `document_id` replaces prior chunks for that document.
2. **Inspect & Delete Policies:**
   * Call `GET /api/v1/documents` (or click **Refresh List** in `/ui`) to view files in `data/policies/` alongside ingested `document_id` groups and chunk counts.
   * Call `DELETE /api/v1/documents/{document_id}?remove_file=true|false` (or use Section 3 in `/ui`) to delete chunks from PostgreSQL and optionally remove the source file from `data/policies/`.
3. **Query Policies (Human Users or External Systems):**
   * Send `query_text` and `user_roles` to `POST /api/v1/query` (or Section 2 in `/ui`).
   * The service filters chunks by `user_roles`, performs hybrid retrieval, evaluates guardrails (such as high-value / approval thresholds), logs an audit record when possible, and returns a structured JSON response.

---

## Expected Inputs and Outputs

### 1. Ingest Text (`POST /api/v1/ingest`)

**Input (`application/json`):**
```json
{
  "document_id": "POL-FIN-021",
  "title": "Vendor Payments & Software Spend Policy",
  "content": "Vendor invoices above INR 50,000 require Finance Director approval.",
  "department_access": ["finance"],
  "sensitivity_level": "confidential"
}
```

**Output (`application/json`):**
```json
{
  "status": "ingested",
  "document_id": "POL-FIN-021",
  "source_file": "data/policies/POL-FIN-021.txt",
  "chunks_created": 1
}
```

### 2. Ingest File (`POST /api/v1/ingest/file`)

**Input (`multipart/form-data`):**
* `file` *(required)*: UTF-8 `.txt` or `.md` policy file.
* `document_id` *(optional)*: Defaults to the uploaded filename stem.
* `title` *(optional)*: Defaults to a title-cased version of the filename stem.
* `department_access` *(optional, default `"general"`)*: Comma-separated role names (e.g., `"finance,ops"`).
* `sensitivity_level` *(optional, default `"internal"`)*: E.g., `"internal"` or `"confidential"`.

**Output (`application/json`):**
```json
{
  "status": "ingested",
  "document_id": "vendor_policy",
  "title": "Vendor Policy",
  "source_file": "data/policies/vendor_policy.md",
  "chunks_created": 2
}
```

### 3. List Documents (`GET /api/v1/documents`)

**Output (`application/json`):**
```json
{
  "source_directory": "data/policies",
  "source_files": ["data/policies/POL-FIN-021.txt"],
  "ingested_documents": [
    {
      "document_id": "POL-FIN-021",
      "title": "Vendor Payments & Software Spend Policy",
      "chunk_count": 1,
      "sensitivity_level": "confidential",
      "department_access": ["finance"]
    }
  ]
}
```

### 4. Delete Document (`DELETE /api/v1/documents/{document_id}?remove_file=false`)

**Output (`application/json`):**
```json
{
  "status": "deleted",
  "document_id": "POL-FIN-021",
  "chunks_deleted": 1,
  "removed_files": []
}
```

### 5. Query Policies (`POST /api/v1/query`)

**Input (`application/json`):**
```json
{
  "query_text": "Can we release an urgent vendor payment of INR 75,000?",
  "user_roles": ["finance"]
}
```

**Output (`application/json`):**
```json
{
  "answer": "Payments above INR 50,000 require Finance Director or CFO approval even if urgent.",
  "confidence": "high",
  "detected_amount": 75000.0,
  "requires_human_approval": true,
  "approval_reason": "Amount exceeds standard approval threshold.",
  "citations": [
    {
      "document_id": "POL-FIN-021",
      "title": "Vendor Payments & Software Spend Policy",
      "chunk_index": 0
    }
  ]
}
```

---

## Things to Consider

* **Secrets & `.gitignore`:** Never commit `.env` or API keys to GitHub. Exclude `.env` and `.venv/` via `.gitignore` and configure secrets via environment variables in your hosting environment.
* **RBAC Enforcement:** Keep role names consistent between `department_access` during ingestion and `user_roles` during queries, and follow least-privilege assignment.
* **Guardrail Enforcement:** Treat `requires_human_approval: true` as a mandatory workflow gate in calling applications or agents so high-value or sensitive actions pause for human sign-off.
* **Chunking & Audit Trail:** Structure policy documents with clear section headings for accurate chunking, and retain `AuditLog` entries for compliance history even after active policy chunks are deleted.
* **Supported File Formats:** File ingestion currently expects UTF-8 `.txt` or `.md` files; convert binary formats (such as `.pdf` or `.docx`) to plain text prior to upload.
* **Render Filesystem & Persistence:** On Render, local disk writes (such as `data/policies/`) can reset across container restarts or redeploys unless backed by persistent storage or re-ingested. Treat PostgreSQL as the primary persisted store and use `uvicorn main:app --host 0.0.0.0 --port $PORT` as your start command on Render.
* **API Security:** Add authentication/API keys and restrict CORS before exposing `/api/v1/*` publicly.

---

## Running Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install python-multipart
uvicorn main:app --reload --port 8000
```

Open **`http://127.0.0.1:8000/ui`** for the browser console or **`http://127.0.0.1:8000/docs`** for interactive OpenAPI docs.