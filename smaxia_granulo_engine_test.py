# smaxia_granulo_engine_test.py
# ENGINE SAFE PATCH — ne doit jamais casser l'UI à l'import

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import hashlib

import requests

BASE_URLS = ["https://www.apmep.fr/-Terminale-"]

WORKDIR = Path("data_granulo")
PDF_DIR = WORKDIR / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

PDF_HREF_RE = re.compile(r'href=["\']([^"\']+\.pdf)["\']', re.IGNORECASE)

CHAPTER_KEYWORDS = ["suite", "suites", "récurrence", "arithmétique", "géométrique", "raison", "terme général", "u_n", "v_n"]
TRIGGERS = ["calculer", "déterminer", "montrer que", "démontrer", "étudier", "exprimer", "en déduire", "justifier"]


def _absolute_apmep_url(href: str) -> str:
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return "https://www.apmep.fr" + href
    return "https://www.apmep.fr/" + href


def scrape_pdf_urls() -> List[str]:
    pdf_urls = set()
    for url in BASE_URLS:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        for m in PDF_HREF_RE.finditer(r.text):
            pdf_urls.add(_absolute_apmep_url(m.group(1).strip()))
    return sorted(pdf_urls)


def download_pdf(url: str) -> Optional[Path]:
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    pdf_path = PDF_DIR / f"{h}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 1024:
        return pdf_path

    r = requests.get(url, timeout=30)
    if r.status_code != 200 or not r.content:
        return None

    pdf_path.write_bytes(r.content)
    if pdf_path.stat().st_size < 1024:
        return None
    return pdf_path


def extract_text_from_pdf(pdf_path: Path) -> str:
    # Imports retardés pour éviter ImportError au chargement
    try:
        import pdfplumber  # type: ignore
        texts = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t.strip():
                    texts.append(t)
        return "\n".join(texts).strip()
    except Exception:
        # Fallback PyPDF2 si dispo
        try:
            from PyPDF2 import PdfReader  # type: ignore
            reader = PdfReader(str(pdf_path))
            texts = []
            for page in reader.pages:
                t = page.extract_text() or ""
                if t.strip():
                    texts.append(t)
            return "\n".join(texts).strip()
        except Exception:
            return ""


def is_suites_numeriques(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in CHAPTER_KEYWORDS)


def extract_qi(text: str) -> List[str]:
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 18]
    qi = []
    for l in lines:
        if re.search(r"(suite|u_n|v_n|récurrence|raison|arithmétique|géométrique)", l, re.IGNORECASE):
            qi.append(l)
    return qi


def compute_frt(qc: List[str]) -> Dict[str, Any]:
    joined = " ".join(qc).lower()
    triggers = [t for t in TRIGGERS if t in joined]
    return {
        "declencheurs": triggers,
        "ari": list(range(1, len(qc) + 1)),
        "n_q": len(qc),
        "score": round(len(triggers) / max(len(qc), 1), 2),
    }


def run_granulo_test() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    urls = scrape_pdf_urls()

    for url in urls:
        pdf_path = download_pdf(url)
        if not pdf_path:
            continue

        text = extract_text_from_pdf(pdf_path)
        if not text or not is_suites_numeriques(text):
            continue

        qi = extract_qi(text)
        if not qi:
            continue

        # Regroupement simple : 1 QC = lot de Qi (mode minimal stable)
        # (On pourra remettre le clustering ensuite, mais d'abord ZÉRO crash)
        qc = qi[:10]
        results.append({"qc": qc, "frt": compute_frt(qc)})

    return results
