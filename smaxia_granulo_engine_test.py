# smaxia_granulo_engine_test.py
import requests
from bs4 import BeautifulSoup
import pdfplumber
import re
import io
from collections import defaultdict
from difflib import SequenceMatcher
import time

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
HEADERS = {"User-Agent": "SMAXIA-GRANULO/1.0"}
MATH_KEYWORDS = [
    "suite", "limite", "converge", "diverge", "u_n", "n tend vers",
    "croissante", "décroissante", "récurrence"
]

# --------------------------------------------------
# UTILS
# --------------------------------------------------
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def extract_pdf_links(url):
    html = requests.get(url, headers=HEADERS, timeout=10).text
    soup = BeautifulSoup(html, "html.parser")
    return list(set(a["href"] for a in soup.find_all("a", href=True) if a["href"].lower().endswith(".pdf")))

def download_pdf(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    return io.BytesIO(r.content)

def extract_text_from_pdf(pdf_bytes):
    text = ""
    with pdfplumber.open(pdf_bytes) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += "\n" + t
    return text.lower()

def extract_qi(text):
    sentences = re.split(r"[?.!]\s+", text)
    qi = []
    for s in sentences:
        if any(k in s for k in MATH_KEYWORDS) and len(s) > 40:
            qi.append(s.strip())
    return qi

# --------------------------------------------------
# GRANULO CORE
# --------------------------------------------------
def granulo_run(urls, max_subjects=20):
    start = time.time()
    subjects = []
    all_qi = []

    for url in urls:
        try:
            pdf_links = extract_pdf_links(url)
        except:
            continue

        for link in pdf_links[:max_subjects]:
            try:
                pdf_bytes = download_pdf(link)
                text = extract_text_from_pdf(pdf_bytes)
                qi = extract_qi(text)
                if qi:
                    subjects.append({
                        "file": link.split("/")[-1],
                        "source": url,
                        "qi": qi
                    })
                    all_qi.extend(qi)
            except:
                continue

    # QC clustering (simple but réel)
    qc_map = defaultdict(list)
    for qi in all_qi:
        assigned = False
        for qc in qc_map:
            if similar(qc, qi) > 0.75:
                qc_map[qc].append(qi)
                assigned = True
                break
        if not assigned:
            qc_map[qi].append(qi)

    qc_list = []
    for i, (qc_label, qis) in enumerate(qc_map.items(), 1):
        qc_list.append({
            "qc_id": f"QC-{i:03}",
            "title": qc_label[:120],
            "n_q": len(qis),
            "declencheurs": list(set(re.findall(r"\b\w{6,}\b", qc_label)))[:6],
            "ari": [
                "Identifier la suite",
                "Étudier le comportement",
                "Utiliser les théorèmes",
                "Conclure"
            ],
            "frt": {
                "usage": "Étude de suite numérique",
                "method": "Analyse – théorème – conclusion",
                "trap": "Confusion convergence/divergence",
                "conclusion": "Conclusion rigoureuse"
            },
            "qi": qis
        })

    return {
        "subjects": subjects,
        "qc": qc_list,
        "audit": {
            "n_urls": len(urls),
            "n_subjects": len(subjects),
            "n_qi": len(all_qi),
            "n_qc": len(qc_list),
            "elapsed_s": round(time.time() - start, 2)
        }
    }
