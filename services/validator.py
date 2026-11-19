import re
from typing import List, Optional, Tuple

from google.genai import types
from pydantic import BaseModel, Field

from config import get_genai_config
from database import DocumentChunk
from services.ingestion import get_genai_client

APPROVAL_THRESHOLD = 50000.0


class CitationOut(BaseModel):
    document_id: str
    title: str
    chunk_index: int
    rrf_score: float


class GuardrailedRAGResponse(BaseModel):
    answer: str = Field(
        description="Concise answer grounded strictly in the provided context, or a refusal if missing."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence from 0.0 to 1.0 based only on retrieved context.",
    )
    detected_amount: Optional[float] = Field(
        default=None,
        description="Numeric amount if an invoice, payment, or limit is referenced.",
    )
    requires_human_approval: bool = Field(
        default=False,
        description="True if risky, ambiguous, or above approval policy threshold.",
    )
    approval_reason: Optional[str] = Field(
        default=None,
        description="Reason human review is triggered, if applicable.",
    )


def _extract_max_amount(text: str) -> Optional[float]:
    matches = re.findall(
        r"(?:INR|Rs\.?|₹|\$)?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)",
        text,
        flags=re.IGNORECASE,
    )
    values: List[float] = []
    for raw in matches:
        try:
            val = float(raw.replace(",", ""))
            if val >= 100:
                values.append(val)
        except ValueError:
            continue
    return max(values) if values else None


def generate_guarded_answer(
    query_text: str,
    retrieved: List[Tuple[DocumentChunk, float]],
) -> Tuple[GuardrailedRAGResponse, List[CitationOut]]:
    citations = [
        CitationOut(
            document_id=chunk.document_id,
            title=chunk.title,
            chunk_index=chunk.chunk_index,
            rrf_score=score,
        )
        for chunk, score in retrieved
    ]

    if not retrieved:
        return (
            GuardrailedRAGResponse(
                answer="No authorized documents matched your query and role permissions.",
                confidence=1.0,
                requires_human_approval=False,
                approval_reason=None,
            ),
            [],
        )

    context_block = "\n\n".join(
        f"[Source {i + 1} | doc_id={c.document_id} | title={c.title} | sensitivity={c.sensitivity_level}]\n{c.content}"
        for i, (c, _) in enumerate(retrieved)
    )

    system_instruction = (
        "You are an enterprise compliance and document assistant. "
        "Answer ONLY using the provided context excerpts. "
        "If the context does not contain enough information, say so clearly and set confidence below 0.5."
    )

    prompt = f"Context:\n{context_block}\n\nQuestion:\n{query_text}"

    cfg = get_genai_config()
    client = get_genai_client()
    result = client.models.generate_content(
        model=cfg.model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=GuardrailedRAGResponse,
        ),
    )

    parsed = GuardrailedRAGResponse.model_validate_json(result.text)

    reasons: List[str] = []
    inferred_amount = (
        parsed.detected_amount
        if parsed.detected_amount is not None
        else _extract_max_amount(parsed.answer)
    )
    if inferred_amount is not None and inferred_amount >= APPROVAL_THRESHOLD:
        reasons.append(
            f"Amount {inferred_amount} meets/exceeds threshold {APPROVAL_THRESHOLD}"
        )
    if parsed.confidence < 0.65:
        reasons.append(f"Low grounding confidence ({parsed.confidence:.2f})")

    if reasons:
        parsed.requires_human_approval = True
        parsed.approval_reason = "; ".join(reasons)
    elif not parsed.requires_human_approval:
        parsed.approval_reason = None

    return parsed, citations