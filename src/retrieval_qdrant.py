"""
Statewide Hybrid Retrieval Engine for Karmashri.
Combines Qdrant (Dense Vector Search) + PostgreSQL (Sparse Full-Text Search)
via Reciprocal Rank Fusion (RRF).
Adds Native Cross-Encoder Reranking for entity precision, followed by Temporal Recency Precedence.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

import psycopg
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("retrieval")

# ============================================================
# CONFIGURATION & SETTINGS
# ============================================================

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = "kerala_go_nodes"

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
PG_DB = os.getenv("PG_DB", "karmashri")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")


class KarmashriHybridRetriever:
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        log.info("Loading Embeddings and Cross-Encoder into VRAM...")
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    def _get_pg_conn(self):
        return psycopg.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            dbname=PG_DB,
            row_factory=dict_row
        )

    def search(
        self,
        query: str,
        dept_code: Optional[str] = None,
        date_from: Optional[str] = None,
        top_k: int = 4,
        fetch_parent_context: bool = True
    ) -> List[Dict[str, Any]]:
        
        # ----------------------------------------------------
        # 1. Dense Semantic Search via Qdrant
        # ----------------------------------------------------
        query_vector = self.embed_model.encode(query, normalize_embeddings=True).tolist()

        must_conditions = []
        must_not_conditions = [
            FieldCondition(key="parent_path", match=MatchValue(value="Endorsement"))
        ]

        if dept_code:
            must_conditions.append(
                FieldCondition(key="dept_code", match=MatchValue(value=dept_code))
            )
        if date_from:
            must_conditions.append(
                FieldCondition(key="order_date", range=Range(gte=date_from))
            )

        qdrant_filter = Filter(must=must_conditions, must_not=must_not_conditions)

        search_limit = max(15, top_k * 3)
        
        dense_hits = self.qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=search_limit
        ).points

        dense_ranks: Dict[str, int] = {}
        dense_payloads: Dict[str, Dict[str, Any]] = {}
        for rank, hit in enumerate(dense_hits, start=1):
            orig_node_id = hit.payload.get("node_id") or str(hit.id)
            dense_ranks[orig_node_id] = rank
            dense_payloads[orig_node_id] = hit.payload

        # ----------------------------------------------------
        # 2. Sparse Lexical Full-Text Search via PostgreSQL
        # ----------------------------------------------------
        sparse_ranks: Dict[str, int] = {}
        sparse_rows: Dict[str, Dict[str, Any]] = {}

        clean_query = re.sub(r'[^\w\s\.-]', ' ', query).strip()

        pg_filter_clauses = [
            "to_tsvector('english', n.text) @@ plainto_tsquery('english', %(query)s)",
            "n.parent_path != 'Endorsement'"
        ]
        pg_params: Dict[str, Any] = {"query": clean_query, "limit": search_limit}

        if dept_code:
            pg_filter_clauses.append("d.dept_code = %(dept)s")
            pg_params["dept"] = dept_code

        if date_from:
            pg_filter_clauses.append("d.order_date >= %(date_from)s")
            pg_params["date_from"] = date_from

        where_stmt = "WHERE " + " AND ".join(pg_filter_clauses)

        sparse_sql = f"""
        SELECT 
            n.id,
            n.go_number,
            d.department,
            d.order_date,
            d.source_file,
            n.section,
            n.clause_marker,
            n.parent_path,
            n.text,
            ts_rank_cd(to_tsvector('english', n.text), plainto_tsquery('english', %(query)s)) AS rank_score
        FROM kerala_go_nodes n
        JOIN kerala_go_documents d ON n.go_number = d.go_number
        {where_stmt}
        ORDER BY rank_score DESC
        LIMIT %(limit)s;
        """

        try:
            with self._get_pg_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sparse_sql, pg_params)
                    rows = cur.fetchall()
                    for rank, row in enumerate(rows, start=1):
                        sparse_ranks[row["id"]] = rank
                        sparse_rows[row["id"]] = row
        except Exception as exc:
            log.warning(f"PostgreSQL sparse FTS execution skipped: {exc}")

        # ----------------------------------------------------
        # 3. Reciprocal Rank Fusion (RRF) Broad Candidate Pool
        # ----------------------------------------------------
        all_ids = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))
        fused_scores: List[Tuple[str, float]] = []

        for node_id in all_ids:
            score = 0.0
            if node_id in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[node_id])
            if node_id in sparse_ranks:
                score += 1.0 / (self.rrf_k + sparse_ranks[node_id])
            fused_scores.append((node_id, score))

        fused_scores.sort(key=lambda x: x[1], reverse=True)
        candidates = fused_scores[:search_limit]

        # ----------------------------------------------------
        # 4. Native Cross-Encoder Reranking & Temporal Boost
        # ----------------------------------------------------
        candidate_texts = []
        for node_id, _ in candidates:
            if node_id in sparse_rows:
                candidate_texts.append(sparse_rows[node_id]["text"])
            else:
                candidate_texts.append(dense_payloads[node_id].get("text", ""))

        final_ranked_ids: List[Tuple[str, float]] = []

        if candidate_texts:
            sentence_pairs = [[query, text] for text in candidate_texts]
            cross_scores = self.reranker.predict(sentence_pairs)

            for idx, base_score in enumerate(cross_scores):
                node_id = candidates[idx][0]
                final_ranked_ids.append((node_id, float(base_score)))
            
            final_ranked_ids.sort(key=lambda x: x[1], reverse=True)
        else:
            final_ranked_ids = self._apply_temporal_boost(candidates, sparse_rows, dense_payloads)

        top_final = final_ranked_ids[:top_k]

        # ----------------------------------------------------
        # 5. Construct Final Response Payload
        # ----------------------------------------------------
        results: List[Dict[str, Any]] = []
        for node_id, final_score in top_final:
            record: Dict[str, Any] = {}

            if node_id in sparse_rows:
                row = sparse_rows[node_id]
                record = {
                    "node_id": node_id,
                    "rrf_score": final_score,
                    "go_number": row["go_number"],
                    "department": row["department"],
                    "date": str(row["order_date"]),
                    "source_file": row.get("source_file"),
                    "parent_path": row["parent_path"],
                    "section": row["section"],
                    "clause_marker": row["clause_marker"],
                    "text": row["text"]
                }
            elif node_id in dense_payloads:
                pl = dense_payloads[node_id]
                record = {
                    "node_id": node_id,
                    "rrf_score": final_score,
                    "go_number": pl.get("go_number"),
                    "department": pl.get("department"),
                    "date": str(pl.get("order_date")),
                    "source_file": pl.get("source_file"),
                    "parent_path": pl.get("parent_path"),
                    "section": pl.get("section"),
                    "clause_marker": pl.get("clause_marker"),
                    "text": pl.get("text")
                }

            if record and not record.get("source_file"):
                record["source_file"] = self._lookup_source_file(record["go_number"])

            if record and fetch_parent_context:
                record["full_parent_context"] = self._get_parent_context(
                    go_number=record["go_number"],
                    parent_path=record["parent_path"]
                )

            if record:
                results.append(record)

        return results

    def _apply_temporal_boost(self, candidates, sparse_rows, dense_payloads) -> List[Tuple[str, float]]:
        boosted = []
        for node_id, score in candidates:
            date_str = None
            if node_id in sparse_rows:
                date_str = str(sparse_rows[node_id].get("order_date", ""))
            elif node_id in dense_payloads:
                date_str = str(dense_payloads[node_id].get("order_date", ""))

            if date_str and len(date_str) >= 4:
                try:
                    year = int(date_str[:4])
                    if year >= 2025:
                        score *= 1.30
                    elif year >= 2023:
                        score *= 1.15
                except ValueError:
                    pass
            boosted.append((node_id, score))
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    def _lookup_source_file(self, go_number: str) -> Optional[str]:
        sql = "SELECT source_file FROM kerala_go_documents WHERE go_number = %(go_number)s LIMIT 1;"
        try:
            with self._get_pg_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, {"go_number": go_number})
                    row = cur.fetchone()
                    if row:
                        return row.get("source_file")
        except Exception:
            pass
        return None

    def _get_parent_context(self, go_number: str, parent_path: str) -> Optional[str]:
        if not parent_path or parent_path in {"Preamble", "Endorsement", "Intro"}:
            return None

        root_clause = parent_path.split(">")[0].strip()
        sql = """
        SELECT text FROM kerala_go_nodes
        WHERE go_number = %(go_number)s
          AND (parent_path = %(root)s OR parent_path LIKE %(root_prefix)s)
        ORDER BY id ASC;
        """
        try:
            with self._get_pg_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, {
                        "go_number": go_number,
                        "root": root_clause,
                        "root_prefix": f"{root_clause} >%"
                    })
                    rows = cur.fetchall()
                    if rows and len(rows) > 1:
                        return "\n\n".join(r["text"].strip() for r in rows)
        except Exception:
            pass
        return None