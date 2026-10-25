"""
Statewide Hybrid Retrieval Engine for Karmashri.
Combines Qdrant (Dense Vector Search) + PostgreSQL (Sparse Full-Text Search)
via Reciprocal Rank Fusion (RRF) with Temporal Recency Precedence,
Source File Metadata tracking, and Parent Context Assembly.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

import psycopg
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer
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
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)

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
        """
        Executes hybrid dense + sparse retrieval with Reciprocal Rank Fusion
        and temporal amendment re-weighting.
        """
        # ----------------------------------------------------
        # 1. Dense Semantic Search via Qdrant
        # ----------------------------------------------------
        query_vector = self.embed_model.encode(query, normalize_embeddings=True).tolist()

        must_conditions = []
        if dept_code:
            must_conditions.append(
                FieldCondition(key="dept_code", match=MatchValue(value=dept_code))
            )
        if date_from:
            must_conditions.append(
                FieldCondition(key="order_date", range=Range(gte=date_from))
            )

        qdrant_filter = Filter(must=must_conditions) if must_conditions else None

        dense_hits = self.qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=top_k * 3
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

        # Sanitize query string for plainto_tsquery
        clean_query = re.sub(r'[^\w\s\.-]', ' ', query).strip()

        pg_filter_clauses = ["to_tsvector('english', n.text) @@ plainto_tsquery('english', %(query)s)"]
        pg_params: Dict[str, Any] = {"query": clean_query, "limit": top_k * 3}

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
        # 3. Reciprocal Rank Fusion (RRF) with Temporal Recency Boost
        # ----------------------------------------------------
        all_ids = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))
        fused_scores: List[Tuple[str, float]] = []

        for node_id in all_ids:
            score = 0.0
            if node_id in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[node_id])
            if node_id in sparse_ranks:
                score += 1.0 / (self.rrf_k + sparse_ranks[node_id])

            # Extract order date for temporal precedence
            date_str = None
            if node_id in sparse_rows:
                date_str = str(sparse_rows[node_id].get("order_date", ""))
            elif node_id in dense_payloads:
                date_str = str(dense_payloads[node_id].get("order_date", ""))

            # Apply temporal multipliers so recent amendments (2025/2026) outrank older base rules
            if date_str and len(date_str) >= 4:
                try:
                    year = int(date_str[:4])
                    if year >= 2025:
                        score *= 1.30  # Active recent amendments prioritized
                    elif year >= 2023:
                        score *= 1.15
                except ValueError:
                    pass

            fused_scores.append((node_id, score))

        # Sort highest fused score first
        fused_scores.sort(key=lambda x: x[1], reverse=True)
        top_fused = fused_scores[:top_k]

        # ----------------------------------------------------
        # 4. Construct Final Response Payload
        # ----------------------------------------------------
        results: List[Dict[str, Any]] = []
        for node_id, rrf_score in top_fused:
            record: Dict[str, Any] = {}

            if node_id in sparse_rows:
                row = sparse_rows[node_id]
                record = {
                    "node_id": node_id,
                    "rrf_score": rrf_score,
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
                    "rrf_score": rrf_score,
                    "go_number": pl.get("go_number"),
                    "department": pl.get("department"),
                    "date": str(pl.get("order_date")),
                    "source_file": pl.get("source_file"),
                    "parent_path": pl.get("parent_path"),
                    "section": pl.get("section"),
                    "clause_marker": pl.get("clause_marker"),
                    "text": pl.get("text")
                }

            # If source_file was missing from payload, backfill from Postgres
            if record and not record.get("source_file"):
                record["source_file"] = self._lookup_source_file(record["go_number"])

            # 5. Fetch full parent context if requested
            if record and fetch_parent_context:
                record["full_parent_context"] = self._get_parent_context(
                    go_number=record["go_number"],
                    parent_path=record["parent_path"]
                )

            if record:
                results.append(record)

        return results

    def _lookup_source_file(self, go_number: str) -> Optional[str]:
        """Backfills source_file directly from kerala_go_documents table."""
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
        """
        Retrieves sibling sub-clauses when an atomic leaf node (e.g. 3 > (b)) matches,
        ensuring operative rules have their entire section context intact.
        """
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


if __name__ == "__main__":
    retriever = KarmashriHybridRetriever()

    test_queries = [
        "What is the reimbursement ceiling and distance condition for Hostel Subsidy?",
        "rates of children education allowance for differently abled children",
        "private hospital definition under AIS medical facilities scheme"
    ]

    for q in test_queries:
        print("\n" + "=" * 90)
        print(f"QUERY: {q}")
        print("=" * 90)
        hits = retriever.search(q, top_k=2)
        for i, hit in enumerate(hits, start=1):
            print(f"\n[Hit {i} | RRF Score: {hit['rrf_score']:.5f}]")
            print(f"Order: {hit['go_number']} | Date: {hit['date']} | File: {hit.get('source_file')}")
            print(f"Path: {hit['parent_path']}")
            print(f"Snippet: {hit['text'][:220]}...")