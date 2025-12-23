# smaxia_granulo_engine_test.py
# ============================================================
# SMAXIA — GRANULO TEST ENGINE (RÉEL / AUDITABLE)
# ============================================================

import requests
import re
import io
from collections import defaultdict
from PyPDF2 import PdfReader

# ------------------------------------------------------------
# OUTILS
# ------------------------------------------------------------

QUESTION_PATTERN = re.compile(
    r"(calculer|déterminer|étudier|montrer|justifier|prouver)[^?.!]*[?.!]",
    re.IGNORECASE
)

STOPWORDS = {
    "la","le","les","de","du","des","et","à","en","un","une","que","pour",
    "dans","sur","par","avec","au","aux"
}

def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ ]", " ", text)
    return " ".join(w for w in text.split() if w not in STOPWORDS and len(w) > 2)

# ------------------------------------------------------------
# 1️⃣ EXTRACTION DES PDF DEPUIS URL
# ------------------------------------------------------------

def fetch_pdfs(url: str, limit: int):
    html = requests.get(url, timeout=20).text
    links = re.findall(r'href="([^"]+\.pdf)"', html)
    pdfs = []

    for link in links[:limit]:
        if not link.startswith("http"):
            link = url.rstrip("/") + "/" + link.lstrip("/")
        r = requests.get(link, timeout=20)
        if r.status_code == 200:
            pdfs.append(r.content)

    return pdfs

# ------------------------------------------------------------
# 2️⃣ EXTRACTION DES QI
# ------------------------------------------------------------

def extract_qi(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for p in reader.pages:
        text += p.extract_text() or ""

    return [m.group(0).strip() for m in QUESTION_PATTERN.finditer(text)]

# ------------------------------------------------------------
# 3️⃣ CONSTRUCTION DES QC (AGRÉGATION RÉELLE)
# ------------------------------------------------------------

def build_qc(qi_list):
    buckets = defaultdict(list)

    for q in qi_list:
        key = " ".join(clean(q).split()[:3]) or "autre"
        buckets[key].append(q)

    qc_out = []
    N_tot = len(qi_list)

    for i, (k, qs) in enumerate(buckets.items(), start=1):
        qc_out.append({
            "qc_id": f"QC-{i:02d}",
            "title": k.capitalize(),
            "n_q": len(qs),
            "score": len(qs) * 10,
            "psi": round(len(qs) / max(1, N_tot), 3),
            "N_tot": N_tot,
            "declencheurs": list(set(q.split()[0].lower() for q in qs)),
            "ari": ["Analyser", "Calculer", "Conclure"],
            "frt": {
                "usage": "Déduit automatiquement",
                "method": "Agrégation réelle des questions",
                "trap": "Aucune hypothèse ajoutée",
                "conclusion": "QC issue exclusivement des Qi"
            },
            "qi": qs
        })

    return qc_out

# ------------------------------------------------------------
# 4️⃣ PIPELINE APPELÉ PAR L’UI
# ------------------------------------------------------------

def run_granulo_pipeline(urls, volume):
    all_qi = []
    sujets = []

    for url in urls:
        pdfs = fetch_pdfs(url, volume)
        for pdf in pdfs:
            qi = extract_qi(pdf)
            if qi:
                sujets.append({
                    "Fichier": f"Sujet_{len(sujets)+1}.pdf",
                    "Nature": "INCONNU",
                    "Année": "N/A",
                    "Source": url
                })
                all_qi.extend(qi)

    qc = build_qc(all_qi)

    return {
        "sujets": sujets,
        "qc": qc,
        "stats": {
            "nb_qi": len(all_qi),
            "nb_qc": len(qc)
        }
    }
