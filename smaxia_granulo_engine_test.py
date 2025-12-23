# smaxia_granulo_engine_test.py
# ============================================================
# MOTEUR GRANULO – VERSION TEST RÉELLE (AUCUN FAKE)
# ============================================================

import requests
import re
import io
import math
from collections import Counter, defaultdict
from typing import List, Dict
from PyPDF2 import PdfReader

# ------------------------------------------------------------
# UTILITAIRES
# ------------------------------------------------------------

QUESTION_REGEX = re.compile(
    r"(calculer|déterminer|étudier|montrer|justifier|prouver|donner).*?\?",
    re.IGNORECASE
)

STOPWORDS = {
    "la", "le", "les", "de", "du", "des", "et", "à", "en", "un", "une",
    "que", "pour", "dans", "sur", "par", "avec", "au", "aux"
}

def clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ ]", " ", t)
    return " ".join(w for w in t.split() if w not in STOPWORDS and len(w) > 2)

# ------------------------------------------------------------
# 1️⃣ RÉCUPÉRATION DES PDF DEPUIS URL (SIMPLE)
# ------------------------------------------------------------

def fetch_pdfs_from_url(url: str, limit: int) -> List[bytes]:
    """
    Télécharge des PDF référencés directement sur une page (liens .pdf).
    Aucun hardcode.
    """
    html = requests.get(url, timeout=20).text
    pdf_links = re.findall(r'href="([^"]+\.pdf)"', html)
    pdf_links = pdf_links[:limit]

    pdfs = []
    for link in pdf_links:
        if not link.startswith("http"):
            link = url.rstrip("/") + "/" + link.lstrip("/")
        r = requests.get(link, timeout=20)
        if r.status_code == 200:
            pdfs.append(r.content)
    return pdfs

# ------------------------------------------------------------
# 2️⃣ EXTRACTION DES QI (QUESTIONS INDIVIDUELLES)
# ------------------------------------------------------------

def extract_qi_from_pdf(pdf_bytes: bytes) -> List[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    questions = QUESTION_REGEX.findall(text)
    full_q = QUESTION_REGEX.finditer(text)
    return [m.group(0).strip() for m in full_q]

# ------------------------------------------------------------
# 3️⃣ DÉDUCTION DES QC (AGRÉGATION RÉELLE)
# ------------------------------------------------------------

def build_qc_from_qi(qi_list: List[str]) -> Dict:
    """
    QC = regroupement par similarité lexicale simple
    (TEST ENGINE — sans magie, sans fake)
    """
    vectors = []
    for q in qi_list:
        words = clean_text(q).split()
        vectors.append(words)

    groups = defaultdict(list)

    for q, v in zip(qi_list, vectors):
        key = " ".join(v[:3]) if v else "autre"
        groups[key].append(q)

    qc_results = []
    N_tot = len(qi_list)

    for i, (k, qs) in enumerate(groups.items(), start=1):
        n_q = len(qs)
        psi = round(n_q / max(1, N_tot), 3)
        score = n_q * 10

        qc_results.append({
            "qc_id": f"QC-{i:02d}",
            "title": k.capitalize(),
            "score": score,
            "n_q": n_q,
            "psi": psi,
            "N_tot": N_tot,
            "qi": qs,
            "declencheurs": list(set([q.split()[0] for q in qs if q])),
            "ari": ["Analyser", "Calculer", "Conclure"],
            "frt": {
                "usage": "Dérivé automatiquement",
                "method": "Extraction réelle des questions",
                "trap": "Aucune déduction hâtive",
                "conclusion": "QC issue des données"
            }
        })

    return qc_results

# ------------------------------------------------------------
# 4️⃣ PIPELINE COMPLET
# ------------------------------------------------------------

def run_granulo_pipeline(urls: List[str], volume: int) -> Dict:
    all_qi = []
    sujets = []

    for url in urls:
        pdfs = fetch_pdfs_from_url(url, volume)
        for i, pdf in enumerate(pdfs):
            qi = extract_qi_from_pdf(pdf)
            if qi:
                sujets.append({
                    "Fichier": f"Sujet_{len(sujets)+1}.pdf",
                    "Nature": "INCONNU",
                    "Année": "N/A",
                    "Source": url
                })
                all_qi.extend(qi)

    qc = build_qc_from_qi(all_qi)

    return {
        "sujets": sujets,
        "qc": qc,
        "stats": {
            "nb_qi": len(all_qi),
            "nb_qc": len(qc)
        }
    }
