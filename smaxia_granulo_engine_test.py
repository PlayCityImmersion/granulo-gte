# smaxia_granulo_engine_test.py
import requests
from bs4 import BeautifulSoup
import re
from collections import defaultdict
import math
import time

# ============================================================
# EXTRACTION DES LIENS PDF À PARTIR DES URLS
# ============================================================
def extract_pdf_links(urls, limit):
    pdfs = []
    for url in urls:
        try:
            html = requests.get(url, timeout=10).text
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.lower().endswith(".pdf"):
                    if href.startswith("http"):
                        pdfs.append(href)
                    else:
                        pdfs.append(url.rstrip("/") + "/" + href.lstrip("/"))
                if len(pdfs) >= limit:
                    return pdfs
        except Exception:
            pass
    return pdfs

# ============================================================
# EXTRACTION QI (BRUT, NON INTERPRÉTÉ)
# ============================================================
def extract_qi_from_text(text):
    lines = text.split("\n")
    qi = []
    for l in lines:
        if re.search(r"(calculer|déterminer|montrer|étudier)", l.lower()):
            qi.append(l.strip())
    return qi

# ============================================================
# MOTEUR GRANULO TEST (F1–F8 MINIMAL)
# ============================================================
def build_qc(qi_list):
    qc_map = defaultdict(list)

    for qi in qi_list:
        if "limite" in qi.lower():
            qc_map["QC-LIMITE"].append(qi)
        elif "suite" in qi.lower():
            qc_map["QC-SUITE"].append(qi)
        elif "fonction" in qi.lower():
            qc_map["QC-FONCTION"].append(qi)

    qc = []
    for k, v in qc_map.items():
        qc.append({
            "id": k,
            "n_q": len(v),
            "qi": v,
            "psi": round(min(1.0, len(v) / 10), 2),
            "score": len(v) * 10
        })
    return qc

# ============================================================
# COURBE DE SATURATION
# ============================================================
def saturation_curve(qc_history):
    curve = []
    for i, qc in enumerate(qc_history):
        curve.append({
            "sujets": i + 1,
            "qc": len(qc)
        })
    return curve

# ============================================================
# POINT D’ENTRÉE UNIQUE
# ============================================================
def run_granulo_test(urls, volume):
    pdf_links = extract_pdf_links(urls, volume)

    sujets = []
    all_qi = []

    for i, pdf in enumerate(pdf_links):
        sujets.append({
            "fichier": pdf.split("/")[-1],
            "annee": None,
            "nature": "INCONNU",
            "source": re.sub(r"https?://", "", pdf).split("/")[0]
        })

        # ⚠️ TEST MODE : on ne parse pas encore le PDF
        # On simule uniquement du TEXTE BRUT
        fake_text = pdf.replace("-", " ")
        qi = extract_qi_from_text(fake_text)
        all_qi.extend(qi)

    qc = build_qc(all_qi)

    return {
        "sujets": sujets,
        "qi": all_qi,
        "qc": qc,
        "saturation": saturation_curve([qc])
    }
