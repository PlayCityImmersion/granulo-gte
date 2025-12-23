# smaxia_granulo_engine_test.py
# GRANULO TEST ENGINE — SMAXIA
# Objectif : extraction réelle → Qi → QC → FRT

import os
import re
import requests
import hashlib
from bs4 import BeautifulSoup
from pathlib import Path
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIGURATION
# =========================
BASE_URLS = [
    "https://www.apmep.fr/-Terminale-"
]

WORKDIR = Path("data_granulo")
PDF_DIR = WORKDIR / "pdfs"
TEXT_DIR = WORKDIR / "texts"

PDF_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)

CHAPTER_KEYWORDS = [
    "suite",
    "suites numériques",
    "raison",
    "terme général",
    "récurrence",
    "arithmétique",
    "géométrique"
]

TRIGGERS = [
    "calculer",
    "déterminer",
    "montrer que",
    "démontrer",
    "étudier",
    "exprimer",
    "en déduire"
]

# =========================
# SCRAPING PDF URLS
# =========================
def scrape_pdf_urls():
    pdf_urls = set()

    for url in BASE_URLS:
        r = requests.get(url, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf"):
                if href.startswith("http"):
                    pdf_urls.add(href)
                else:
                    pdf_urls.add("https://www.apmep.fr" + href)

    return sorted(pdf_urls)

# =========================
# DOWNLOAD PDF
# =========================
def download_pdf(url):
    h = hashlib.md5(url.encode()).hexdigest()
    pdf_path = PDF_DIR / f"{h}.pdf"

    if pdf_path.exists():
        return pdf_path

    r = requests.get(url, timeout=30)
    if r.status_code == 200:
        with open(pdf_path, "wb") as f:
            f.write(r.content)
        return pdf_path

    return None

# =========================
# PDF TEXT EXTRACTION
# =========================
def extract_text_from_pdf(pdf_path):
    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                full_text.append(t)

    return "\n".join(full_text)

# =========================
# QI EXTRACTION
# =========================
def extract_qi(text):
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 20]

    qi = []
    for l in lines:
        if re.search(r"(suite|u_n|v_n|raison|récurrence)", l.lower()):
            qi.append(l)

    return qi

# =========================
# FILTER CHAPTER
# =========================
def is_suites_numeriques(text):
    t = text.lower()
    return any(k in t for k in CHAPTER_KEYWORDS)

# =========================
# QC GROUPING
# =========================
def group_qi_to_qc(qi_list, threshold=0.45):
    if len(qi_list) < 2:
        return [[q] for q in qi_list]

    vect = TfidfVectorizer(stop_words="french")
    X = vect.fit_transform(qi_list)
    sim = cosine_similarity(X)

    clusters = []
    used = set()

    for i in range(len(qi_list)):
        if i in used:
            continue

        cluster = [qi_list[i]]
        used.add(i)

        for j in range(i + 1, len(qi_list)):
            if sim[i, j] >= threshold:
                cluster.append(qi_list[j])
                used.add(j)

        clusters.append(cluster)

    return clusters

# =========================
# FRT COMPUTATION
# =========================
def compute_frt(qc):
    text = " ".join(qc).lower()

    triggers = [t for t in TRIGGERS if t in text]

    return {
        "declencheurs": triggers,
        "ari": list(range(1, len(qc) + 1)),
        "n_q": len(qc),
        "score": round(len(triggers) / max(len(qc), 1), 2)
    }

# =========================
# MAIN ENGINE
# =========================
def run_engine():
    results = []

    pdf_urls = scrape_pdf_urls()
    print(f"[INFO] PDFs trouvés : {len(pdf_urls)}")

    for url in pdf_urls:
        pdf_path = download_pdf(url)
        if not pdf_path:
            continue

        text = extract_text_from_pdf(pdf_path)
        if not is_suites_numeriques(text):
            continue

        qi = extract_qi(text)
        if not qi:
            continue

        qcs = group_qi_to_qc(qi)

        for qc in qcs:
            frt = compute_frt(qc)
            results.append({
                "qc": qc,
                "frt": frt
            })

    return results

# =========================
# EXEC
# =========================
if __name__ == "__main__":
    data = run_engine()
    print(f"[OK] QC générées : {len(data)}")

    for i, d in enumerate(data[:5]):
        print(f"\nQC {i+1}")
        print(d["qc"][:2])
        print(d["frt"])
