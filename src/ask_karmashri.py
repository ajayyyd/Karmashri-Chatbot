#!/usr/bin/env python3
"""
Interactive CLI for Karmashri Legal AI.
Queries the statewide hybrid retrieval and grounded generation engine.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation import KarmashriLegalAdvisor


def main():
    print("=" * 80)
    print("  KARMASHRI: Kerala Government Orders Intelligence Platform")
    print("  Type your administrative query below (or 'exit' to quit).")
    print("=" * 80 + "\n")

    advisor = KarmashriLegalAdvisor()

    while True:
        try:
            query = input("\n[Kerala Admin Query] > ").strip()
            if not query:
                continue
            if query.lower() in {"exit", "quit", "q"}:
                print("Exiting Karmashri. Goodbye.")
                break

            dept_filter = input("Department Code filter (e.g. GAD, FIN or press Enter to skip): ").strip().upper() or None

            print("\nAnalyzing corpus and synthesizing grounded ruling...")
            res = advisor.answer_query(query, dept_code=dept_filter, top_k=4)

            print("\n" + res["answer"])
            print("\n" + "-" * 40 + " Verified Statutory Sources " + "-" * 40)
            for c in res["citations"]:
                print(f"[{c['source_index']}] {c['go_number']} (dated {c['date']}) | Path: {c['parent_path']}")
            print("-" * 108)

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break
        except Exception as err:
            print(f"\n[Error] {err}")


if __name__ == "__main__":
    main()