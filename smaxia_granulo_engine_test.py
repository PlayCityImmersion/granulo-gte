# smaxia_granulo_engine_test.py
# GRANULO TEST ENGINE — SMAXIA (VERSION STABLE)

import os
import re
import requests
import hashlib
from bs4 import BeautifulSoup
from pathlib import Path
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def scrape_pdf_urls():
    pdf_urls = set()
    for url in BASE_URLS:
        r = requests.get(url, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf"):
                pdf_urls.add(
                    href if href.startswith("http")
                    else "https://www.apmep.fr" + href
                )
    return sorted(pdf_urls)

def download_pdf(url):
    h = hashlib.md5(url.encode()).hexdigest()
    pdf_path = PDF_DIR / f"{h}.pdf"
    if pdf_path.exists():
        return pdf_path
    r = requests.get(url, timeout=30)
    if r.status_code == 200:
        pdf_path.write_bytes(r.content)
        return pdf_path
    return None

def extract_text_from_pdf(pdf_path):
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text.append(t)
    return "\n".join(text)

def extract_qi(text):
    qi = []
    for line in text.split("\n"):
        l = line.strip()
        if len(l) > 20 and re.search(r"(suite|u_n|v_n|récurrence)", l.lower()):
            qi.append(l)
    return qi

def is_suites_numeriques(text):
    t = text.lower()
    return any(k in t for k in CHAPTER_KEYWORDS)

def group_qi_to_qc(qi_list, threshold=0.45):
    if len(qi_list) < 2:
        return [[q] for q in qi_list]

    vect = TfidfVectorizer(stop_words="french")
    X = vect.fit_transform(qi_list)
    sim = cosine_similarity(X)

    clusters, used = [], set()
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

def compute_frt(qc):
    text = " ".join(qc).lower()
    triggers = [t for t in TRIGGERS if t in text]
    return {
        "declencheurs": triggers,
        "ari": list(range(1, len(qc) + 1)),
        "n_q": len(qc),
        "score": round(len(triggers) / max(len(qc), 1), 2)
    }

# ✅ FONCTION OFFICIELLE EXPORTÉE
def run_granulo_test():
    results = []
    pdf_urls = scrape_pdf_urls()

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
            results.append({
                "qc": qc,
                "frt": compute_frt(qc)
            })

    return results
