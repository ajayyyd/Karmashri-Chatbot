"""
Statewide Grounded Generation Engine for Karmashri.
Synthesizes retrieved Kerala Government Orders with strict source attribution,
supersession/amendment lineage tracking, and grounded PDF source mapping.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests

# Ensure project root is available on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe dual import supporting either filename
try:
    from src.retrieval_qdrant import KarmashriHybridRetriever
except ImportError:
    from src.retrieval_qdrant import KarmashriHybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("generation")

OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


class KarmashriLegalAdvisor:
    def __init__(self):
        self.retriever = KarmashriHybridRetriever()

    def answer_query(
        self,
        query: str,
        dept_code: Optional[str] = None,
        date_from: Optional[str] = None,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """
        Retrieves relevant legal nodes, organizes context by temporal precedence,
        and generates an authoritative, grounded response.
        """
        # 1. Retrieve Candidate Clauses
        hits = self.retriever.search(
            query=query,
            dept_code=dept_code,
            date_from=date_from,
            top_k=top_k,
            fetch_parent_context=True
        )

        if not hits:
            return {
                "query": query,
                "answer": "No relevant Kerala Government Orders found for the specified query.",
                "citations": []
            }

        # 2. Sort by Order Date (Newest to Oldest) for Temporal Precedence
        sorted_hits = sorted(hits, key=lambda x: str(x.get("date", "")), reverse=True)

        # 3. Build Grounded Context Block and Citation Manifest
        context_blocks = []
        citations_summary = []

        for idx, h in enumerate(sorted_hits, start=1):
            go_num = h.get("go_number") or "Unknown GO"
            date = h.get("date") or "Unknown Date"
            dept = h.get("department") or "Government of Kerala"
            path = h.get("parent_path") or "Operative Order"
            source_file = h.get("source_file") or ""
            
            # Prefer full parent context if available, otherwise use atomic text
            clause_body = (h.get("full_parent_context") or h.get("text", "")).strip()

            context_blocks.append(
                f"[Source Document {idx}]\n"
                f"GO Number: {go_num}\n"
                f"Date: {date}\n"
                f"Department: {dept}\n"
                f"Clause/Path: {path}\n"
                f"Source File: {source_file}\n"
                f"Text:\n{clause_body}\n"
            )

            citations_summary.append({
                "source_index": idx,
                "go_number": go_num,
                "date": date,
                "department": dept,
                "parent_path": path,
                "source_file": source_file,
            })

        grounded_context = "\n--------------------\n".join(context_blocks)

        # 4. Strict Grounding System Prompt
        system_prompt = f"""You are the official Legal and Administrative AI Advisor for the Government of Kerala.
Answer the official's question strictly and exclusively using the provided Government Order sources.

Mandatory Rules:
1. Base every assertion strictly on the text provided below. Never extrapolate, hallucinate, or cite rules not in the text.
2. Note temporal precedence: If a newer order (e.g., 2025 or 2026) modifies rates, terms, or definitions from an older order (e.g., 2020 or 2021), clearly state the current modified rule and explicitly mention what it amends or replaces.
3. State exact monetary ceilings, allowances, distance requirements, and eligibility conditions explicitly.
4. If no monetary ceiling or specific condition is mentioned for a clause, explicitly state that none is provided in the source orders.
5. Attribute every single rule directly to its GO Number, Issue Date, and Clause Path.

Provided Sources:
\"\"\"
{grounded_context}
\"\"\"

Question: {query}

Provide a direct, authoritative executive response formatted strictly under these headings:
- **Direct Ruling & Applicable Conditions**
- **Monetary Ceilings / Numerical Terms (if applicable)**
- **Statutory Authority & Precedence**"""

        # 5. Invoke Local Grounded Model
        try:
            res = requests.post(
                OLLAMA_ENDPOINT,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": system_prompt,
                    "stream": False,
                    "keep_alive": "30m",  # Keeps model warm in RAM across calls
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 700,
                        "num_ctx": 4096       # Ample context for multiple parent clauses
                    }
                },
                timeout=90  # Safe timeout for CPU inference
            )

            if res.status_code == 200:
                answer_text = res.json().get("response", "").strip()
                return {
                    "query": query,
                    "answer": answer_text,
                    "citations": citations_summary
                }
            else:
                log.error(f"Ollama error status: {res.status_code}")
                return {
                    "query": query,
                    "answer": f"Inference service returned status code {res.status_code}.",
                    "citations": citations_summary
                }

        except Exception as exc:
            log.error(f"Generation failure: {exc}")
            return {
                "query": query,
                "answer": f"Generation service unavailable ({exc}).",
                "citations": citations_summary
            }


if __name__ == "__main__":
    advisor = KarmashriLegalAdvisor()

    test_queries = [
        "What are the rates of Children Education Allowance and condition for claiming Hostel Subsidy?",
        "What is the definition of a Private Hospital under the All India Services Medical Facilities Scheme?"
    ]

    for q in test_queries:
        print("\n" + "=" * 90)
        print(f"OFFICIAL QUERY: {q}")
        print("=" * 90)
        res = advisor.answer_query(q)
        print("\n" + res["answer"])
        print("\n--- Verified Statutory Sources ---")
        for c in res["citations"]:
            print(f"[{c['source_index']}] {c['go_number']} (dated {c['date']}) | Path: {c['parent_path']} | File: {c.get('source_file')}")