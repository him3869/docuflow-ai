import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from database import DocumentChunk, get_db
from main import app
from services.validator import APPROVAL_THRESHOLD, generate_guarded_answer


def _fake_chunk(
    chunk_id: int,
    doc_id: str,
    title: str,
    content: str,
    roles: list[str],
    sensitivity: str = "internal",
) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id=doc_id,
        title=title,
        chunk_index=0,
        content=content,
        department_access=roles,
        sensitivity_level=sensitivity,
        embedding=[0.01] * 768,
    )


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_empty_retrieval_returns_safe_fallback():
    parsed, citations = generate_guarded_answer(
        query_text="Show confidential executive compensation",
        retrieved=[],
    )
    assert citations == []
    assert parsed.requires_human_approval is False
    assert "No authorized documents" in parsed.answer


@patch("services.validator.get_genai_client")
@patch("services.validator.get_genai_config")
def test_high_amount_triggers_human_approval_guardrail(mock_cfg, mock_client):
    mock_cfg.return_value = SimpleNamespace(
        api_key="test-key",
        model_name="test-gen-model",
        embedding_model_name="test-emb-model",
    )
    fake_llm_json = (
        '{"answer": "Invoices of INR 85,000 require director sign-off.", '
        '"confidence": 0.92, "detected_amount": 85000.0, '
        '"requires_human_approval": false, "approval_reason": null}'
    )
    mock_client.return_value.models.generate_content.return_value = SimpleNamespace(
        text=fake_llm_json
    )

    chunk = _fake_chunk(
        chunk_id=1,
        doc_id="POL-FIN-014",
        title="Vendor Invoice Approval Policy",
        content="Invoices at or above INR 50,000 require finance director sign-off.",
        roles=["finance"],
        sensitivity="confidential",
    )

    parsed, citations = generate_guarded_answer(
        query_text="Approval for INR 85,000 invoice?",
        retrieved=[(chunk, 0.032)],
    )

    assert len(citations) == 1
    assert citations[0].document_id == "POL-FIN-014"
    assert parsed.requires_human_approval is True
    assert str(int(APPROVAL_THRESHOLD)) in (parsed.approval_reason or "")


def test_rbac_role_isolation_via_api():
    """Verify unauthorized roles do not receive finance-restricted chunks."""
    finance_chunk = _fake_chunk(
        chunk_id=10,
        doc_id="POL-FIN-014",
        title="Vendor Invoice Approval Policy",
        content="Finance policy secret content.",
        roles=["finance"],
    )

    mock_db = MagicMock()

    def override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db
    try:
        with patch("main.hybrid_search_rbac", return_value=[]) as mock_search:
            client = TestClient(app)
            resp = client.post(
                "/api/v1/query",
                json={
                    "query_text": "What is the finance policy?",
                    "user_roles": ["marketing"],
                },
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["citations"] == []
            assert "No authorized documents" in body["answer"]
            mock_search.assert_called_once()
            assert finance_chunk.content not in body["answer"]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_RAG_TESTS") != "1",
    reason="Set RUN_LIVE_RAG_TESTS=1 with running Postgres + GEMINI_API_KEY to run live smoke test.",
)
def test_live_ingest_and_query_smoke():
    client = TestClient(app)
    ingest_resp = client.post(
        "/api/v1/ingest",
        json={
            "document_id": "SMOKE-DOC-001",
            "title": "Travel Reimbursement Policy",
            "content": "Domestic flight bookings up to INR 12,000 are auto-approved.",
            "department_access": ["ops"],
            "sensitivity_level": "internal",
        },
    )
    assert ingest_resp.status_code == 200

    query_resp = client.post(
        "/api/v1/query",
        json={
            "query_text": "What is the auto-approval limit for domestic flights?",
            "user_roles": ["ops"],
        },
    )
    assert query_resp.status_code == 200
    assert query_resp.json()["requires_human_approval"] is False
