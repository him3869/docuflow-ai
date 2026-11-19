from typing import Dict, List, Tuple
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database import DocumentChunk
from services.ingestion import generate_embeddings

RRF_K = 60


def hybrid_search_rbac(
    db: Session,
    query_text: str,
    user_roles: List[str],
    top_k: int = 5,
    candidate_limit: int = 20,
) -> List[Tuple[DocumentChunk, float]]:
    """
    Retrieve top-k chunks matching user_roles using Postgres full-text search
    and pgvector cosine distance merged via Reciprocal Rank Fusion (RRF).
    """
    if not user_roles:
        return []

    rbac_filter = DocumentChunk.department_access.op("&&")(user_roles)

    # 1. Keyword / full-text ranking within authorized chunks
    ts_vector = func.to_tsvector("english", DocumentChunk.content)
    ts_query = func.plainto_tsquery("english", query_text)
    ts_rank_score = func.ts_rank_cd(ts_vector, ts_query)

    keyword_stmt = (
        select(DocumentChunk.id)
        .where(rbac_filter, ts_vector.op("@@")(ts_query))
        .order_by(ts_rank_score.desc())
        .limit(candidate_limit)
    )
    keyword_ids = [row[0] for row in db.execute(keyword_stmt).all()]

    # 2. Semantic vector ranking within authorized chunks
    query_embedding = generate_embeddings([query_text], output_dimensionality=768)[0]
    cosine_dist = DocumentChunk.embedding.cosine_distance(query_embedding)

    vector_stmt = (
        select(DocumentChunk.id)
        .where(rbac_filter, DocumentChunk.embedding.is_not(None))
        .order_by(cosine_dist.asc())
        .limit(candidate_limit)
    )
    vector_ids = [row[0] for row in db.execute(vector_stmt).all()]

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores: Dict[int, float] = {}
    for rank, chunk_id in enumerate(keyword_ids, start=1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

    for rank, chunk_id in enumerate(vector_ids, start=1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

    if not rrf_scores:
        return []

    top_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
    chunks = db.execute(select(DocumentChunk).where(DocumentChunk.id.in_(top_ids))).scalars().all()
    chunk_map = {c.id: c for c in chunks}

    return [(chunk_map[cid], round(rrf_scores[cid], 6)) for cid in top_ids if cid in chunk_map]