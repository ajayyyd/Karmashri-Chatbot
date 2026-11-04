"""
Production Hybrid Ingestion Engine for Kerala Government Orders (GOs).

Architectural Principles:
1. Zero Data Loss Invariant: Input characters equal emitted node characters.
2. Hierarchical Lineage: Structural clauses capture their ancestry (e.g., 2 > (a) > (ii)).
3. Anti-Hallucination Guard: Generative metadata must pass grounding assertions against source text.
4. Idempotency: Deterministic SHA-256 node keys guarantee reproducible runs.
"""

import os
import re
import json
import hashlib
import unicodedata
import difflib
import logging
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Dict, Any, List, Optional, Tuple, Sequence
import concurrent.futures
import pandas as pd
from img2table.document import Image as Img2TableImage
from img2table.ocr import TesseractOCR

import requests
from PIL import Image
import pymupdf
import pytesseract
from llama_index.core.schema import Document, TextNode, BaseNode
from llama_index.core.node_parser.interface import NodeParser

log = logging.getLogger("hybrid_ingest")

# ============================================================
# CONFIGURATION & SETTINGS
# ============================================================

DPI = int(os.getenv("INGEST_DPI", "300"))
ZOOM = DPI / 72.0
MIN_TEXT_QUALITY_SCORE = float(os.getenv("MIN_TEXT_QUALITY_SCORE", "0.65"))
TESSERACT_TIMEOUT = int(os.getenv("TESSERACT_TIMEOUT", "30"))

OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp", ".bmp"}

KERALA_DEPARTMENTS: Dict[str, str] = {
    "GAD": "General Administration Department",
    "FIN": "Finance Department",
    "HEDN": "Higher Education Department",
    "GEDN": "General Education Department",
    "H&FWD": "Health & Family Welfare Department",
    "DMD": "Disaster Management Department",
    "REV": "Revenue Department",
    "P&ARD": "Personnel & Administrative Reforms Department",
    "LSGD": "Local Self Government Department",
    "PWD": "Public Works Department",
    "WRD": "Water Resources Department",
    "ITD": "Electronics & Information Technology Department",
    "IND": "Industries & Commerce Department",
    "AGRI": "Agriculture Development & Farmers Welfare Department",
    "HOME": "Home Department",
    "LAW": "Law Department",
    "TAX": "Taxes Department",
    "TRANS": "Transport Department",
    "F&CSD": "Food, Civil Supplies & Consumer Affairs Department",
    "SCSTDD": "Scheduled Castes & Scheduled Tribes Development Department",
    "SJD": "Social Justice Department",
    "WCD": "Women & Child Development Department",
    "LBRD": "Labour & Skills Department",
    "CO-OP": "Co-operation Department",
    "FSHD": "Fisheries & Ports Department",
    "TOUR": "Tourism Department",
    "ENVT": "Environment Department",
    "FOR": "Forest & Wildlife Department",
    "POWER": "Power Department",
    "CAD": "Cultural Affairs Department",
    "AYUSH": "Ayush Department",
}

OLLAMA_STRICT_SCHEMA = {
    "type": "object",
    "properties": {
        "go_type": {"type": "string", "enum": ["Ms", "Rt", "P", "Ap", "Ord"]},
        "go_number": {"type": "string"},
        "order_date_iso": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
        "dept_code": {"type": "string"}
    },
    "required": ["go_type", "go_number", "order_date_iso", "dept_code"]
}


# ============================================================
# 1. NON-DESTRUCTIVE NORMALIZER & QUALITY ASSURANCE
# ============================================================

class SafeTextNormalizer:
    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""
        # Standardize Unicode (NFC) while preserving substantive characters
        text = unicodedata.normalize('NFC', text)
        # Normalize carriage returns and form feeds
        text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\f', '\n')
        # Standardize non-breaking space without mutating standard character set
        text = text.replace('\u00a0', ' ')
        return text

    @staticmethod
    def evaluate_quality(raw_text: str) -> float:
        """Computes readability density. Returns 0.0 for binary artifacts or garbage text."""
        stripped = raw_text.strip()
        if len(stripped) < 40:
            return 0.0

        replacement_count = stripped.count('\ufffd')
        if replacement_count / len(stripped) > 0.03:
            return 0.0

        words = re.findall(r'\b[A-Za-z0-9]{2,}\b', stripped)
        tokens = stripped.split()
        if not tokens:
            return 0.0

        return len(words) / len(tokens)


# ============================================================
# 2. ROBUST METADATA EXTRACTOR WITH GROUNDING GUARANTEES
# ============================================================

class HybridKeralaGOMetadataExtractor:
    # Captures GO numbers: eg, G.O.(Ms) No. 68/2026/GAD, G.O.(Rt) No. 12/09/FIN, G.O.(P) No.45/2024/SCSTDD
    GO_NUMBER_RE = re.compile(
        r'G[\.0O\s]*\(?\s*([A-Za-z]+)\s*\)?[\.0O\s]*N[o00]{1,2}[\.:\s]*'
        r'(\d+)[ \t\.]*[\/\-\\|lI1][ \t\.]*(\d{2,4})[ \t\.]*[\/\-\\|lI1][ \t\.]*'
        r'([A-Z][A-Z0-9 \t&\-]{1,15}(?:\([A-Za-z0-9\-]+\))?)',
        re.IGNORECASE
    )

    DATE_RE = re.compile(
    r'(?:Dated|Date)\s*[:,]?\s*'
    r'(?:[A-Za-z ]+,\s*)?'
    r'(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}'
    r'|\d{1,2}(?:st|nd|rd|th)?\s+'
    r'[A-Za-z]+\s*,?\s+\d{4})',
    re.IGNORECASE
    )

    CITED_ORDERS_RE = re.compile(
        r'(?:G\.O\.?\s*\([A-Za-z]+\)\s*No\.?\s*[\d\w\/\-\.\s]+?dt[\.:\s]*\d{1,2}[/\.\-]\d{1,2}[/\.\-]\d{2,4})',
        re.IGNORECASE
    )

    @classmethod
    def extract(cls, full_text: str, filename_hint: str) -> Dict[str, Any]:
        read_split = re.split(r'\n\s*Read\s*:', full_text[:5000], maxsplit=1, flags=re.IGNORECASE)
        header_window = read_split[0] if len(read_split) > 1 else full_text[:2500]
        read_window = read_split[1][:3000] if len(read_split) > 1 else ""

        # Deterministic Pass
        meta = cls._extract_regex(header_window, filename_hint)
        meta["cited_orders"] = cls._extract_citations(read_window)

        # Trigger LLM if core fields are absent
        if not meta["go_number"] or not meta["order_date_iso"]:
            log.info("Deterministic extractor produced incomplete metadata. Invoking local LLM...")
            llm_meta = cls._extract_via_local_llm(header_window)
            if cls._verify_llm_grounding(llm_meta, header_window, filename_hint):
                meta = cls._merge_metadata(meta, llm_meta)
            else:
                log.warning("LLM output failed grounding check against raw header. Rejected.")

        return cls._canonicalize_metadata(meta, filename_hint)

    @classmethod
    def _derive_department_from_go(cls, go_number: str) -> Tuple[Optional[str], Optional[str]]:
        """Derives dept_code and department cleanly from an already parsed go_number."""
        if not go_number:
            return None, None

        parts = [p.strip() for p in go_number.split('/')]
        dept_code = None
        if len(parts) >= 3:
            # Clean trailing punctuation or whitespace from the last token
            dept_code = re.sub(r'[^A-Z&]', '', parts[-1].upper())

        department_name = KERALA_DEPARTMENTS.get(dept_code, dept_code if dept_code else None)
        return dept_code, department_name

    @classmethod
    def _extract_regex(cls, header_text: str, filename_hint: str) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "go_type": None,
            "go_number": None,
            "order_date_iso": None,
            "department": None,
            "dept_code": None,
        }

        go_m = cls.GO_NUMBER_RE.search(header_text)
        if go_m:
            raw_type = cls._normalize_go_type(go_m.group(1))
            order_seq = go_m.group(2).strip()
            order_year = cls._expand_year(go_m.group(3).strip())
            raw_dept = "".join(go_m.group(4).split())
            clean_dept = re.sub(r'(?:GOVERNMENT|DATED|DATE|ABSTRACT|\bDT\b).*$', '', raw_dept, flags=re.IGNORECASE).rstrip('/-_ ')
            
            # Form initial GO number to leverage clean derivation
            temp_go_number = f"G.O.({raw_type}) No. {order_seq}/{order_year}/{clean_dept}"
            dept_code, department = cls._derive_department_from_go(temp_go_number)

            meta["go_type"] = raw_type
            meta["dept_code"] = dept_code
            meta["go_number"] = temp_go_number
            meta["department"] = department

        d_m = cls.DATE_RE.search(header_text)
        if d_m:
            meta["order_date_iso"] = cls._parse_iso_date(d_m.group(1).strip())

        return meta

    @classmethod
    def _extract_citations(cls, read_text: str) -> List[str]:
        if not read_text:
            return []
        matches = cls.CITED_ORDERS_RE.findall(read_text)
        cleaned = [" ".join(m.split()) for m in matches]
        return list(dict.fromkeys(cleaned))


    @classmethod
    def _extract_via_local_llm(cls, header_window: str) -> Dict[str, Any]:
        prompt = (
            "Extract metadata from this Kerala Government Order header into the strict JSON schema provided.\n"
            f"Header:\n\"\"\"\n{header_window[:1500]}\n\"\"\""
        )
        try:
            res = requests.post(
                OLLAMA_ENDPOINT,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "format": OLLAMA_STRICT_SCHEMA,
                    "stream": False,
                    "options": {"temperature": 0.0}
                },
                timeout=25
            )
            if res.status_code == 200:
                payload = res.json().get("response", "{}")
                data = json.loads(payload)
                if isinstance(data, dict):
                    # Also normalize fields derived from LLM response go_number if present
                    if data.get("go_number"):
                        d_code, d_name = cls._derive_department_from_go(data["go_number"])
                        if d_code:
                            data["dept_code"] = d_code
                    return data
        except Exception as exc:
            log.warning(f"Ollama local inference failed: {exc}")
        return {}

    @classmethod
    def _verify_llm_grounding(cls, llm_meta: Dict[str, Any], header_text: str, filename_hint: str) -> bool:
        if not llm_meta or not isinstance(llm_meta, dict):
            return False

        go_num = str(llm_meta.get("go_number", "")).strip()
        if not go_num or go_num.lower() == "null":
            return False

        target_tokens = "".join(c for c in go_num if c.isalnum()).upper()
        header_tokens = "".join(c for c in header_text if c.isalnum()).upper()
        file_tokens = "".join(c for c in filename_hint if c.isalnum()).upper()

        if target_tokens in header_tokens:
            return True

        ratio = difflib.SequenceMatcher(None, target_tokens, file_tokens).ratio()
        return ratio >= 0.75

    @classmethod
    def _canonicalize_metadata(cls, meta: Dict[str, Any], filename_hint: str) -> Dict[str, Any]:
        if not meta.get("go_number") and filename_hint:
            fn_m = cls.GO_NUMBER_RE.search(filename_hint)
            if fn_m:
                raw_type = cls._normalize_go_type(fn_m.group(1))
                seq = fn_m.group(2)
                yr = cls._expand_year(fn_m.group(3))
                code = "".join(fn_m.group(4).split())
                meta["go_number"] = f"G.O.({raw_type}) No. {seq}/{yr}/{code}"
                meta["go_type"] = raw_type

        if not meta.get("order_date_iso") and filename_hint:
            d_m = cls.DATE_RE.search(filename_hint)
            if d_m:
                meta["order_date_iso"] = cls._parse_iso_date(d_m.group(1).strip())

        # Clean trailing noise on go_number
        if meta.get("go_number"):
            meta["go_number"] = re.sub(r'(?:dt|dated|\bdt\b).*$', '', meta["go_number"], flags=re.IGNORECASE).rstrip('/-_ ')
            
            # Ensure dept_code and department are always structurally derived from go_number
            dept_code, department = cls._derive_department_from_go(meta["go_number"])
            if dept_code:
                meta["dept_code"] = dept_code
            if department:
                meta["department"] = department

        return meta

    @staticmethod
    def _normalize_go_type(raw_type: str) -> str:
        clean = re.sub(r'[^A-Za-z]', '', raw_type).capitalize()
        return "Ms" if clean in {"Ms", "M"} else ("Rt" if clean == "Rt" else clean)

    @staticmethod
    def _expand_year(yr_str: str) -> str:
        if len(yr_str) == 2:
            val = int(yr_str)
            return f"19{yr_str}" if val > 50 else f"20{yr_str}"
        return yr_str

    @staticmethod
    def _parse_iso_date(raw_date: str) -> Optional[str]:
        clean = re.sub(r'(st|nd|rd|th)', '', raw_date).strip()
        formats = [
            "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
            "%d/%m/%y", "%d-%m-%y", "%d.%m.%y",
            "%d %B %Y", "%d %b %Y", "%d %B, %Y", "%d %b, %Y"
        ]
        for fmt in formats:
            try:
                return datetime.strptime(clean, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None

    @staticmethod
    def _merge_metadata(primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
        for k in ["go_type", "go_number", "order_date_iso", "dept_code"]:
            if not primary.get(k) and secondary.get(k):
                primary[k] = secondary[k]
        
        if primary.get("go_number"):
            d_code, d_name = HybridKeralaGOMetadataExtractor._derive_department_from_go(primary["go_number"])
            if d_code:
                primary["dept_code"] = d_code
            if d_name:
                primary["department"] = d_name
        return primary


# ============================================================
# 3. NON-DESTRUCTIVE HIERARCHICAL PARSER
# ============================================================
_ROMAN_NUMERAL = r'(?:X{0,3}(?:IX|IV|V?I{0,3}))'

class KeralaLegalClauseParser(NodeParser):
    ORDER_HEADING: ClassVar[re.Pattern] = re.compile(
        r'(?:\n|\A)[ \t]*[\"\'\*\~\-]*[ \t]*(?:ORDER|O\s*R\s*D\s*E\s*R)[ \t]*[\"\'\*\~\-:]*[ \t]*(?:\n|\Z)',
        re.IGNORECASE
    )

    ANNEXURE_HEADING: ClassVar[re.Pattern] = re.compile(
        r'(?:\n|\A)[ \t]*ANNEXURE\b[ \t\-–—]*(?:[I|V|X\d]+)?',
        re.IGNORECASE
    )

    SIGNATORY_DELIMITER: ClassVar[re.Pattern] = re.compile(
        r'(?=(?:\n|\A)[ \t]*\(?By\s+order\s+of\s+the\s+Governor\)?)',
        re.IGNORECASE
    )

    FORM_HEADING: ClassVar[re.Pattern] = re.compile(
        r'(?:\n|\A)[ \t]*(?:Form|FORM|Iornr|f\s*orm)[ \t\.\-–—]*[a-zA-Z0-9\|! ]{1,10}(?=\s*\n)',
        re.IGNORECASE
    )

    CLAUSE_PATTERN: ClassVar[re.Pattern] = re.compile(
        rf'^(?:'
        rf'(?P<num>\d{{1,3}})[\.\)]|'
        rf'\((?P<paren_num>\d{{1,3}})\)|'
        rf'\((?P<alpha>[a-z]{{1,2}})\)|'
        rf'\((?P<roman>{_ROMAN_NUMERAL})\)|'
        rf'(?P<standalone_roman>{_ROMAN_NUMERAL})[\.\)]'
        rf')[ \t]+',
        re.IGNORECASE | re.MULTILINE
    )
    
    @staticmethod
    def detect_statutory_form(text: str, current_section: str) -> str:
        text_lower = text.lower()
        
        # 1. Fill-in-the-blanks (Strictly dots and underscores, NOT dashes)
        blank_lines_count = len(re.findall(r"\.{4,}|_{4,}", text))
        
        # 2. Markdown Table Detection
        has_markdown_table = "|---" in text or "| ---" in text
        
        # 3. Form Vocabulary Point System
        form_score = 0
        if "signature" in text_lower or "sign & seal" in text_lower: form_score += 2
        if "designation" in text_lower: form_score += 1
        if "office seal" in text_lower or "(seal)" in text_lower: form_score += 1
        if "name of" in text_lower: form_score += 1
        if "date:" in text_lower or "date :" in text_lower: form_score += 1
        if "applicant" in text_lower: form_score += 1
        if "bill no" in text_lower or "rupees" in text_lower: form_score += 1
        
        has_title = False
        if "essentiality certificate" in text_lower or "application for" in text_lower:
            has_title = True
        if re.search(r"\bform[- \w]{1,5}[ivx\d]+", text_lower):
            has_title = True

        # DECISION TREE
        if blank_lines_count >= 3:
            return "statutory_form"
            
        # If it has a table AND form keywords (e.g. signature), it's a form. 
        # If it's just a table with 0 form keywords, it stays an annexure_rule.
        if (has_markdown_table or blank_lines_count >= 1) and form_score >= 1:
            return "statutory_form"
            
        if has_title and (has_markdown_table or blank_lines_count >= 1 or form_score >= 1):
            return "statutory_form"
            
        return current_section

    def _parse_nodes(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any
    ) -> List[BaseNode]:
        nodes: List[BaseNode] = []

        for doc in documents:
            text = doc.text
            meta = doc.metadata
            go_id = meta.get("go_number") or meta.get("source_file", "KERALA_GO")

            # 1. Non-destructive boundary identification
            order_match = self.ORDER_HEADING.search(text)
            if order_match:
                preamble_slice = text[:order_match.start()]
                post_preamble_slice = text[order_match.start():]
            else:
                preamble_slice = ""
                post_preamble_slice = text

            if preamble_slice:
                # Apply form detection to preamble
                final_section = self.detect_statutory_form(preamble_slice, "preamble")
                nodes.append(self._build_node(
                    text=preamble_slice,
                    meta=meta,
                    section=final_section,
                    marker="Preamble",
                    path="Preamble",
                    go_id=go_id
                ))

            # 2. Separate Annexures
            annex_match = self.ANNEXURE_HEADING.search(post_preamble_slice)
            if annex_match:
                order_body = post_preamble_slice[:annex_match.start()]
                annexure_body = post_preamble_slice[annex_match.start():]
            else:
                order_body = post_preamble_slice
                annexure_body = ""

            # 3. Separate Operative from Signatory block
            sig_match = self.SIGNATORY_DELIMITER.search(order_body)
            if sig_match:
                operative_text = order_body[:sig_match.start()]
                endorsement_text = order_body[sig_match.start():]
            else:
                operative_text = order_body
                endorsement_text = ""

            # Segment operative clauses hierarchically
            nodes.extend(self._segment_hierarchy(operative_text, "operative_clause", meta, go_id))

            if endorsement_text:
                nodes.append(self._build_node(
                    text=endorsement_text,
                    meta=meta,
                    section="endorsement",
                    marker="Endorsement",
                    path="Endorsement",
                    go_id=go_id
                ))

            # 4. Partition Annexure and Statutory Forms
            if annexure_body:
                form_splits = [m.start() for m in self.FORM_HEADING.finditer(annexure_body)]
                if form_splits:
                    rules_text = annexure_body[:form_splits[0]]
                    nodes.extend(self._segment_hierarchy(rules_text, "annexure_rule", meta, go_id))

                    for idx, start_idx in enumerate(form_splits):
                        end_idx = form_splits[idx + 1] if idx + 1 < len(form_splits) else len(annexure_body)
                        form_slice = annexure_body[start_idx:end_idx]
                        
                        # Explicitly label this block as a statutory_form without further chunking
                        final_section = self.detect_statutory_form(form_slice, "statutory_form")
                        
                        tag = self.FORM_HEADING.search(form_slice)
                        marker_val = tag.group(0).strip() if tag else f"Form-{idx+1}"

                        nodes.append(self._build_node(
                            text=form_slice,
                            meta=meta,
                            section=final_section,
                            marker=marker_val,
                            path=f"Annexure > {marker_val}",
                            go_id=go_id
                        ))
                else:
                    nodes.extend(self._segment_hierarchy(annexure_body, "annexure_rule", meta, go_id))

        return nodes

    def _segment_hierarchy(
        self,
        text: str,
        section: str,
        meta: Dict[str, Any],
        go_id: str
    ) -> List[TextNode]:
        """State machine that tracks parent clause ancestry and injects context."""
        if not text:
            return []

        nodes: List[TextNode] = []
        
        matches = list(self.CLAUSE_PATTERN.finditer(text))
        if not matches:
            # Safely evaluate if this un-chunked block is a form
            final_section = self.detect_statutory_form(text, section)
            return [self._build_node(text, meta, final_section, "Intro", "Intro", go_id)]

        slices: List[Tuple[int, int, Any]] = []
        if matches[0].start() > 0:
            slices.append((0, matches[0].start(), None))

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            slices.append((start, end, m))

        # The State Machine
        stack = {1: None, 2: None, 3: None, 4: None}
        top_level_context = ""

        for start, end, match_obj in slices:
            chunk = text[start:end].strip()
            if not chunk:
                continue
                
            # Intercept inner chunk: Evaluate ONLY this specific sub-clause
            chunk_section = self.detect_statutory_form(chunk, section)

            if match_obj is None:
                marker = "Intro"
                path_str = "Intro"
                final_chunk_text = chunk
            else:
                if match_obj.group('num'):  # Depth 1: "1."
                    val = match_obj.group('num')
                    marker = f"{val}."
                    
                    # Reset deeper states
                    stack[1] = marker
                    stack[2], stack[3], stack[4] = None, None, None
                    
                    # Capture the title line for context injection
                    top_level_context = chunk.split('\n')[0].strip()
                    final_chunk_text = chunk

                elif match_obj.group('paren_num'):  # Depth 2: "(1)"
                    val = match_obj.group('paren_num')
                    marker = f"({val})"
                    
                    stack[2] = marker
                    stack[3], stack[4] = None, None
                    
                    # Inject Context
                    if top_level_context:
                        final_chunk_text = f"[Context: This is a sub-clause of {top_level_context}]\n\n{chunk}"
                    else:
                        final_chunk_text = chunk

                elif match_obj.group('alpha'):  # Depth 3: "(a)"
                    val = match_obj.group('alpha')
                    marker = f"({val})"
                    
                    stack[3] = marker
                    stack[4] = None
                    
                    # Inject Context
                    if top_level_context:
                        final_chunk_text = f"[Context: This is a sub-clause of {top_level_context}]\n\n{chunk}"
                    else:
                        final_chunk_text = chunk

                elif match_obj.group('roman') or match_obj.group('standalone_roman'):  # Depth 4: "(i)"
                    val = match_obj.group('roman') or match_obj.group('standalone_roman')
                    marker = f"({val})"
                    
                    stack[4] = marker
                    
                    # Inject Context
                    if top_level_context:
                        final_chunk_text = f"[Context: This is a sub-clause of {top_level_context}]\n\n{chunk}"
                    else:
                        final_chunk_text = chunk
                else:
                    marker = "Clause"
                    final_chunk_text = chunk

                # Rebuild current path dynamically based on active stack levels
                active_levels = [stack[lvl] for lvl in range(1, 5) if stack[lvl]]
                path_str = " > ".join(active_levels) if active_levels else marker

            nodes.append(self._build_node(final_chunk_text, meta, chunk_section, marker, path_str, go_id))

        return nodes

    def _build_node(
        self,
        text: str,
        meta: Dict[str, Any],
        section: str,
        marker: str,
        path: str,
        go_id: str
    ) -> TextNode:
        seed = f"{go_id}::{section}::{path}::{text.strip()}".encode('utf-8')
        node_id = hashlib.sha256(seed).hexdigest()

        node_metadata = {
            **meta,
            "section": section,
            "clause_marker": marker,
            "parent_path": path,
            "is_clause": bool(marker not in {"Preamble", "Endorsement", "Intro"})
        }

        return TextNode(
            id_=node_id,
            text=text,
            metadata=node_metadata,
            excluded_embed_metadata_keys=[
                "source_file", "total_pages", "extraction_mode",
                "cited_orders", "parent_path", "is_clause"
            ],
            excluded_llm_metadata_keys=[
                "source_file", "total_pages", "extraction_mode"
            ]
        )


# ============================================================
# 4. ENGINE PIPELINE (EXTRACTION, AUDIT & BATCH API)
# ============================================================

def assert_complete_text_coverage(source_text: str, nodes: List[BaseNode]) -> None:
    def strip_ws(t: str) -> str:
        # Ignore our dynamically injected context when validating the text invariant
        clean_t = re.sub(r'\[Context: This is a sub-clause of .*?\]', '', t)
        return "".join(c for c in unicodedata.normalize('NFC', clean_t) if not c.isspace())

    src_stripped = strip_ws(source_text)
    node_stripped = "".join(strip_ws(n.text) for n in nodes)

    if src_stripped != node_stripped:
        diff = len(src_stripped) - len(node_stripped)
        raise AssertionError(
            f"Zero-Loss Invariant Violation: Input ({len(src_stripped)} chars) "
            f"!= Output ({len(node_stripped)} chars). Net variance: {diff} characters."
        )


def extract_page_data(page: pymupdf.Page) -> Tuple[str, str]:
    # 1. Try Digital Extraction First
    tab_finder = page.find_tables()
    table_md_chunks = []
    if tab_finder.tables:
        for tab in tab_finder.tables:
            table_md_chunks.append(tab.to_markdown())

    # Extract every individual word with its exact X/Y visual coordinates
    words = page.get_text("words")
    lines = {}
    
    # Extract every individual word
    words = page.get_text("words")
    
    # Sort primarily by top Y-coordinate, then left X-coordinate
    words.sort(key=lambda w: (w[1], w[0]))

    lines = []
    current_line = []
    current_top = -1
    current_bottom = -1

    for w in words:
        w_x0, w_top, w_x1, w_bottom, text = w[0], w[1], w[2], w[3], w[4]
        w_center = (w_top + w_bottom) / 2

        if not current_line:
            current_line.append((w_x0, text))
            current_top, current_bottom = w_top, w_bottom
        else:
            # If the word's vertical center is within the current line's bounding box, it belongs here
            if current_top <= w_center <= current_bottom:
                current_line.append((w_x0, text))
                # Expand the bounding box slightly if this word is taller
                current_top = min(current_top, w_top)
                current_bottom = max(current_bottom, w_bottom)
            else:
                # Word is outside the box. Save current line and start a new one.
                lines.append(current_line)
                current_line = [(w_x0, text)]
                current_top, current_bottom = w_top, w_bottom

    if current_line:
        lines.append(current_line)

    sorted_lines = []
    for line in lines:
        # Sort all words on this specific line horizontally (left-to-right)
        line.sort(key=lambda x: x[0])
        # Extract just the text and join with spaces
        sorted_lines.append(" ".join([word_data[1] for word_data in line]))
        
    raw_digital = "\n".join(sorted_lines).strip()
    
    score = SafeTextNormalizer.evaluate_quality(raw_digital)

    if score >= MIN_TEXT_QUALITY_SCORE:
        full_text = raw_digital + ("\n\n" + "\n\n".join(table_md_chunks) if table_md_chunks else "")
        return full_text, "digital"

    # 2. Scanned Image Pipeline
    pix = page.get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM), colorspace=pymupdf.csGRAY)
    img_bytes = pix.tobytes("png")

    # STRICT DETECTOR: Prevents hallucinating tables from regular text like distribution lists
    img_doc = Img2TableImage(img_bytes)
    detected_tables = img_doc.extract_tables(ocr=None, implicit_rows=False, borderless_tables=False)

    table_markdowns = []
    if detected_tables:
        log.info("Strict table grids detected. Extracting...")
        ocr = TesseractOCR(n_threads=1, lang="eng")
        extracted_tables = img_doc.extract_tables(ocr=ocr, implicit_rows=False, borderless_tables=False)
        
        for ext_table in extracted_tables:
            if ext_table.df is not None and not ext_table.df.empty:
                df = ext_table.df.fillna("")
                table_markdowns.append(df.to_markdown(index=False))

    # 3. Standard text extraction for the rest of the page
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    raw_ocr = pytesseract.image_to_string(
        img,
        config="--oem 1 --psm 3 -c tessedit_create_txt=1",
        timeout=TESSERACT_TIMEOUT
    )

    # 4. Stitch text and tables together
    full_text = raw_ocr
    if table_markdowns:
        full_text += "\n\n### Extracted Tables:\n\n" + "\n\n".join(table_markdowns)

    return full_text, "tesseract_hybrid"


def _process_single_page(page_num: int, pdf_path: Path) -> Tuple[int, str, str]:
    """Isolated worker function for parallel execution."""
    with pymupdf.open(pdf_path) as local_doc:
        page = local_doc[page_num]
        t, m = extract_page_data(page)
        return page_num, t, m

def ingest_kerala_go_pdf(pdf_path: Path) -> List[BaseNode]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    log.info(f"Opening Document: {pdf_path.name}")
    with pymupdf.open(pdf_path) as doc:
        if doc.needs_pass:
            raise PermissionError(f"PDF is encrypted: {pdf_path.name}")
        total_pages = len(doc)

    page_texts: List[str] = [""] * total_pages
    modes: List[str] = [""] * total_pages

    # Execute OCR and Table extraction concurrently across CPU cores
    log.info(f"Processing {total_pages} pages in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(_process_single_page, i, pdf_path) for i in range(total_pages)]
        
        for future in concurrent.futures.as_completed(futures):
            page_num, t, m = future.result()
            page_texts[page_num] = t
            modes[page_num] = m

    full_text = SafeTextNormalizer.clean("\n\n".join(page_texts))

    metadata = HybridKeralaGOMetadataExtractor.extract(full_text, filename_hint=pdf_path.name)
    metadata["source_file"] = pdf_path.name
    metadata["total_pages"] = total_pages
    metadata["extraction_mode"] = "digital" if all(m == "digital" for m in modes) else "ocr_hybrid"

    parent_doc = Document(text=full_text, metadata=metadata)
    parser = KeralaLegalClauseParser()
    nodes = parser.get_nodes_from_documents([parent_doc])

    assert_complete_text_coverage(full_text, nodes)
    log.info(f"Zero text-loss verified. Emitted {len(nodes)} hierarchical nodes.")
    return nodes