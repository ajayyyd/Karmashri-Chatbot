"""
Fault-Tolerant Multiprocessing Batch Runner for Kerala Government Orders.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Lock Tesseract to single thread per process to prevent thread oversubscription
os.environ["OMP_THREAD_LIMIT"] = "1"

from src.hybrid_ingestion import ingest_kerala_go_pdf, SUPPORTED_IMAGE_EXTENSIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("batch_processor")

INPUT_DIR = PROJECT_ROOT / "data" / "pdf"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEAD_LETTER_DIR = PROJECT_ROOT / "data" / "dead_letter"


def process_single_file(file_path: Path) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "file": file_path.name,
        "status": "FAILED",
        "nodes": 0,
        "go_number": None,
        "date": None,
        "error": None
    }
    out_file = OUTPUT_DIR / f"{file_path.stem}_nodes.json"

    # Idempotent skip: bypass already parsed files
    if out_file.exists():
        record["status"] = "SKIPPED_EXISTING"
        return record

    try:
        nodes = ingest_kerala_go_pdf(file_path)
        payload = [
            {
                "id": n.node_id,
                "text": n.text,
                "metadata": n.metadata
            }
            for n in nodes
        ]

        with open(out_file, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)

        first_meta = nodes[0].metadata if nodes else {}
        record["status"] = "SUCCESS"
        record["nodes"] = len(nodes)
        record["go_number"] = first_meta.get("go_number")
        record["date"] = first_meta.get("order_date_iso")

    except Exception as exc:
        record["error"] = str(exc)
        dead_record = DEAD_LETTER_DIR / f"{file_path.stem}.err.log"
        with open(dead_record, "w", encoding="utf-8") as ef:
            ef.write(f"File: {file_path.name}\nException:\n{exc}\n")

    return record


def run_batch():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEAD_LETTER_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() == ".pdf"])
    if not files:
        log.warning(f"No PDF orders found in {INPUT_DIR}")
        return

    log.info(f"Queuing {len(files)} documents across CPU workers...")

    results: List[Dict[str, Any]] = []
    max_workers = max(1, os.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, f): f for f in files}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            log.info(f"[{res['status']}] {res['file']} -> Nodes: {res['nodes']} | GO: {res['go_number']}")

    # Formatted Diagnostic Summary Table
    print("\n" + "=" * 115)
    print(f"{'File':<40} | {'Status':<12} | {'Nodes':<6} | {'Order Date':<12} | {'GO Number':<30}")
    print("-" * 115)
    for r in results:
        print(f"{r['file'][:38]:<40} | {r['status']:<12} | {r['nodes']:<6} | {str(r['date']):<12} | {str(r['go_number'])[:28]:<30}")
    print("=" * 115 + "\n")


if __name__ == "__main__":
    run_batch()