"""
Statewide Concurrent Storage Engine:
- PostgreSQL: Authoritative relational metadata, raw text, and audit trails.
- Qdrant: High-throughput vector search with in-graph payload filtering and scalar quantization.
"""

import os
import json
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any

import psycopg
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    PayloadSchemaType,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("statewide_storage")

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
PROCESSED_DIR = Path("data/processed")


def init_qdrant(client: QdrantClient):
    """Initializes Qdrant collection with 384-d Cosine vectors and Scalar Quantization."""
    collections = [c.name for c in client.get_collections().collections]

    if QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True
                )
            ),
        )
        # Fast payload filtering indexes
        client.create_payload_index(QDRANT_COLLECTION, "dept_code", PayloadSchemaType.KEYWORD)
        client.create_payload_index(QDRANT_COLLECTION, "go_number", PayloadSchemaType.KEYWORD)
        client.create_payload_index(QDRANT_COLLECTION, "order_date", PayloadSchemaType.KEYWORD)
        client.create_payload_index(QDRANT_COLLECTION, "section", PayloadSchemaType.KEYWORD)
        log.info(f"Qdrant collection '{QDRANT_COLLECTION}' created with quantized payload indexes.")


def to_valid_qdrant_uuid(sha256_str: str) -> str:
    """Converts a 64-char SHA256 string into an RFC-4122 compliant UUID string."""
    return str(uuid.UUID(hex=sha256_str[:32]))


def load_corpus_statewide():
    # 1. Connect to both engines
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    init_qdrant(qdrant)

    pg_conn = psycopg.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=PG_DB,
        row_factory=dict_row
    )

    node_files = sorted(PROCESSED_DIR.glob("*_nodes.json"))
    if not node_files:
        log.warning(f"No processed files found in {PROCESSED_DIR}")
        return

    log.info(f"Loading embedding model: {EMBED_MODEL_NAME}...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    total_synced_nodes = 0

    with pg_conn.cursor() as cur:
        for file_path in node_files:
            with open(file_path, "r", encoding="utf-8") as fp:
                nodes: List[Dict[str, Any]] = json.load(fp)

            if not nodes:
                continue

            first_meta = nodes[0]["metadata"]
            go_number = first_meta.get("go_number")
            source_file = first_meta.get("source_file")

            if not go_number or go_number == "UNKNOWN":
                log.warning(f"Skipping {file_path.name}: missing valid GO number")
                continue

            # 2. Persist relational document record in PostgreSQL
            cur.execute("""
            INSERT INTO kerala_go_documents (
                go_number, go_type, order_date, department, dept_code,
                source_file, total_pages, extraction_mode, cited_orders
            ) VALUES (
                %(go_number)s, %(go_type)s, %(order_date)s, %(department)s, %(dept_code)s,
                %(source_file)s, %(total_pages)s, %(extraction_mode)s, %(cited_orders)s
            )
            ON CONFLICT (go_number) DO UPDATE SET
                order_date = EXCLUDED.order_date,
                department = EXCLUDED.department,
                dept_code = EXCLUDED.dept_code,
                source_file = EXCLUDED.source_file,
                cited_orders = EXCLUDED.cited_orders;
            """, {
                "go_number": go_number,
                "go_type": first_meta.get("go_type"),
                "order_date": first_meta.get("order_date_iso"),
                "department": first_meta.get("department"),
                "dept_code": first_meta.get("dept_code"),
                "source_file": source_file,
                "total_pages": first_meta.get("total_pages"),
                "extraction_mode": first_meta.get("extraction_mode"),
                "cited_orders": json.dumps(first_meta.get("cited_orders", []))
            })

            # 3. Vector calculation with contextual prefix
            prepared_texts = [
                f"{go_number} | {n['metadata'].get('department', '')} | {n['metadata'].get('parent_path', '')}: {n['text']}"
                for n in nodes
            ]
            vectors = embed_model.encode(prepared_texts, batch_size=32, normalize_embeddings=True)

            # 4. Stream points into Qdrant & store relational nodes in PostgreSQL
            points = []
            for node, vec in zip(nodes, vectors):
                meta = node["metadata"]
                point_uuid = to_valid_qdrant_uuid(node["id"])

                # Store text chunk in PostgreSQL
                cur.execute("""
                INSERT INTO kerala_go_nodes (
                    id, go_number, section, clause_marker, parent_path,
                    is_clause, text, metadata
                ) VALUES (
                    %(id)s, %(go_number)s, %(section)s, %(clause_marker)s, %(parent_path)s,
                    %(is_clause)s, %(text)s, %(metadata)s
                )
                ON CONFLICT (id) DO UPDATE SET
                    text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata;
                """, {
                    "id": node["id"],
                    "go_number": go_number,
                    "section": meta.get("section"),
                    "clause_marker": meta.get("clause_marker"),
                    "parent_path": meta.get("parent_path"),
                    "is_clause": meta.get("is_clause", False),
                    "text": node["text"],
                    "metadata": json.dumps(meta)
                })

                # Prepare Qdrant Point with valid RFC-4122 UUID and source_file payload
                points.append(
                    PointStruct(
                        id=point_uuid,
                        vector=vec.tolist(),
                        payload={
                            "node_id": node["id"],
                            "go_number": go_number,
                            "dept_code": meta.get("dept_code"),
                            "department": meta.get("department"),
                            "order_date": meta.get("order_date_iso"),
                            "source_file": meta.get("source_file") or source_file,
                            "section": meta.get("section"),
                            "clause_marker": meta.get("clause_marker"),
                            "parent_path": meta.get("parent_path"),
                            "is_clause": meta.get("is_clause", False),
                            "text": node["text"]
                        }
                    )
                )

            qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
            pg_conn.commit()
            total_synced_nodes += len(points)
            log.info(f"[Synced] {go_number} -> {len(points)} nodes stored in Postgres & Qdrant.")

    pg_conn.close()
    log.info(f"Corpus synchronization complete. Total points stored: {total_synced_nodes}")


if __name__ == "__main__":
    load_corpus_statewide()