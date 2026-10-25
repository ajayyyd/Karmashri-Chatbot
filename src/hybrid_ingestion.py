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
    # Captures: G.O.(Ms) No. 68/2026/GAD, G.O.(Rt) No. 12/09/FIN, G.O.(P) No.45/2024/SCSTDD
    GO_NUMBER_RE = re.compile(
        r'G[\.0O\s]*\(?\s*([A-Za-z]+)\s*\)?[\.0O\s]*N[o00]{1,2}[\.:\s]*'
        r'(\d+)[ \t\.]*[\/\-\\|lI1][ \t\.]*(\d{2,4})[ \t\.]*[\/\-\\|lI1][ \t\.]*'
        r'([A-Z][A-Z0-9 \t&\-]{1,15}(?:\([A-Za-z0-9\-]+\))?)',
        re.IGNORECASE
    )

    DATE_RE = re.compile(
        r'(?:Dated|Date)[\s,:]*(?:Thiruvananthapuram[,\s]*)?'
        r'(\d{1,2}[/\.\-]\d{1,2}[/\.\-]\d{2,4}|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+[,\s]+\d{4})',
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
            dept_code = re.sub(r'(?:GOVERNMENT|DATED|DATE|ABSTRACT|\bDT\b).*$', '', raw_dept, flags=re.IGNORECASE).rstrip('/-_ ')

            meta["go_type"] = raw_type
            meta["dept_code"] = dept_code
            meta["go_number"] = f"G.O.({raw_type}) No. {order_seq}/{order_year}/{dept_code}"
            meta["department"] = KERALA_DEPARTMENTS.get(dept_code)

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

        # Extract alphanumeric sequence
        target_tokens = "".join(c for c in go_num if c.isalnum()).upper()
        header_tokens = "".join(c for c in header_text if c.isalnum()).upper()
        file_tokens = "".join(c for c in filename_hint if c.isalnum()).upper()

        if target_tokens in header_tokens:
            return True

        # Fuzzy match against filename hint
        ratio = difflib.SequenceMatcher(None, target_tokens, file_tokens).ratio()
        return ratio >= 0.75

    @classmethod
    def _canonicalize_metadata(cls, meta: Dict[str, Any], filename_hint: str) -> Dict[str, Any]:
        # Cross-validate against filename if still unresolved
        if not meta.get("go_number") and filename_hint:
            fn_m = cls.GO_NUMBER_RE.search(filename_hint)
            if fn_m:
                raw_type = cls._normalize_go_type(fn_m.group(1))
                seq = fn_m.group(2)
                yr = cls._expand_year(fn_m.group(3))
                code = "".join(fn_m.group(4).split())
                meta["go_type"] = raw_type
                meta["dept_code"] = code
                meta["go_number"] = f"G.O.({raw_type}) No. {seq}/{yr}/{code}"
                meta["department"] = KERALA_DEPARTMENTS.get(code)

        if not meta.get("order_date_iso") and filename_hint:
            d_m = cls.DATE_RE.search(filename_hint)
            if d_m:
                meta["order_date_iso"] = cls._parse_iso_date(d_m.group(1).strip())

        # Ensure uniform capitalization
        if meta.get("go_number"):
            meta["go_number"] = re.sub(r'(?:dt|dated|\bdt\b).*$', '', meta["go_number"], flags=re.IGNORECASE).rstrip('/-_ ')

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
        if not primary.get("department") and primary.get("dept_code"):
            primary["department"] = KERALA_DEPARTMENTS.get(primary["dept_code"])
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
        r'(?=(?:\n|\A)[ \t]*(?:Form|FORM)[ \t\-–—]*(?:I|II|III|IV|\d+)\b)',
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
                nodes.append(self._build_node(
                    text=preamble_slice,
                    meta=meta,
                    section="preamble",
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
                        tag = self.FORM_HEADING.search(form_slice)
                        marker_val = tag.group(0).strip() if tag else f"Form-{idx+1}"

                        nodes.append(self._build_node(
                            text=form_slice,
                            meta=meta,
                            section="statutory_form",
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
        """State machine that tracks parent clause ancestry."""
        if not text:
            return []

        nodes: List[TextNode] = []
        matches = list(self.CLAUSE_PATTERN.finditer(text))
        if not matches:
            return [self._build_node(text, meta, section, "Intro", "Intro", go_id)]

        slices: List[Tuple[int, int, Any]] = []
        # Leading chunk before the first clause marker
        if matches[0].start() > 0:
            slices.append((0, matches[0].start(), None))

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            slices.append((start, end, m))

        path_stack: List[str] = []

        for start, end, match_obj in slices:
            chunk = text[start:end]
            if not chunk:
                continue

            if match_obj is None:
                marker = "Intro"
                path_str = "Intro"
            else:
                if match_obj.group('num') or match_obj.group('paren_num'):
                    val = match_obj.group('num') or match_obj.group('paren_num')
                    path_stack = [val]
                    marker = f"{val}."
                elif match_obj.group('alpha'):
                    val = f"({match_obj.group('alpha')})"
                    path_stack = path_stack[:1] + [val]
                    marker = val
                elif match_obj.group('roman') or match_obj.group('standalone_roman'):
                    val = f"({match_obj.group('roman') or match_obj.group('standalone_roman')})"
                    path_stack = path_stack[:2] + [val]
                    marker = val
                else:
                    marker = "Clause"

                path_str = " > ".join(path_stack) if path_stack else marker

            nodes.append(self._build_node(chunk, meta, section, marker, path_str, go_id))

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
        # Generate idempotent SHA-256 identifier
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
    """Verifies that 100% of the substantive text from the PDF was partitioned into nodes."""
    def strip_ws(t: str) -> str:
        return "".join(c for c in unicodedata.normalize('NFC', t) if not c.isspace())

    src_stripped = strip_ws(source_text)
    node_stripped = "".join(strip_ws(n.text) for n in nodes)

    if src_stripped != node_stripped:
        diff = len(src_stripped) - len(node_stripped)
        raise AssertionError(
            f"Zero-Loss Invariant Violation: Input ({len(src_stripped)} chars) "
            f"!= Output ({len(node_stripped)} chars). Net variance: {diff} characters."
        )


def extract_page_data(page: pymupdf.Page) -> Tuple[str, str]:
    # 1. Structure preservation: Extract tables as Markdown first
    tab_finder = page.find_tables()
    table_bboxes = []
    table_md_chunks = []

    if tab_finder.tables:
        for tab in tab_finder.tables:
            table_bboxes.append(tab.bbox)
            table_md_chunks.append(tab.to_markdown())

    # 2. Extract plain text layer
    raw_digital = page.get_text("text").strip()
    score = SafeTextNormalizer.evaluate_quality(raw_digital)

    if score >= MIN_TEXT_QUALITY_SCORE:
        # If valid tables exist, append them cleanly to preserve structured data
        full_text = raw_digital + ("\n\n" + "\n\n".join(table_md_chunks) if table_md_chunks else "")
        return full_text, "digital"

    # 3. Direct memory OCR fallback using grayscale conversion (no disk/PNG overhead)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM), colorspace=pymupdf.csGRAY)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)

    raw_ocr = pytesseract.image_to_string(
        img,
        config=f"--oem 1 --psm 3 -c tessedit_create_txt=1",
        timeout=TESSERACT_TIMEOUT
    )
    return raw_ocr, "tesseract_ocr"


def ingest_kerala_go_pdf(pdf_path: Path) -> List[BaseNode]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    log.info(f"Opening Document: {pdf_path.name}")
    with pymupdf.open(pdf_path) as doc:
        if doc.needs_pass:
            raise PermissionError(f"PDF is encrypted: {pdf_path.name}")

        page_texts: List[str] = []
        modes: List[str] = []

        for p in doc:
            t, m = extract_page_data(p)
            page_texts.append(t)
            modes.append(m)

        full_text = SafeTextNormalizer.clean("\n\n".join(page_texts))

        metadata = HybridKeralaGOMetadataExtractor.extract(full_text, filename_hint=pdf_path.name)
        metadata["source_file"] = pdf_path.name
        metadata["total_pages"] = len(doc)
        metadata["extraction_mode"] = "digital" if all(m == "digital" for m in modes) else "ocr_hybrid"

    parent_doc = Document(text=full_text, metadata=metadata)
    parser = KeralaLegalClauseParser()
    nodes = parser.get_nodes_from_documents([parent_doc])

    # Enforce non-destructive invariant
    assert_complete_text_coverage(full_text, nodes)
    log.info(f"Zero text-loss verified. Emitted {len(nodes)} hierarchical nodes.")
    return nodes