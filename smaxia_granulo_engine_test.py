# smaxia_granulo_engine_test.py
from __future__ import annotations

import io
import math
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import requests
import pdfplumber
from bs4 import BeautifulSoup


# =========================
# CONFIG TEST (TRANSPARENT)
# =========================
UA = "SMAXIA-GTE/1.0 (+test-engine)"
REQ_TIMEOUT = 20
MAX_PDF_MB = 25  # sécurité
MIN_QI_CHARS = 18

# Filtrage "Suites numériques" (phase 1 France Terminale spé)
SUITES_KEYWORDS = {
    "suite", "suites", "arithmétique", "géométrique", "raison", "u_n", "un",
    "récurrence", "recurrence", "limite", "convergence", "monotone", "bornée",
    "borne", "majorée", "minorée", "sommes", "somme", "terme général", "terme general"
}


# =========================
# OUTILS TEXTE
# =========================
def _normalize(text: str) -> str:
    t = text.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _tokenize(text: str) -> List[str]:
    t = _normalize(text)
    # tokens simples, alphanum
    toks = re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ0-9]+", t)
    return toks


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _contains_suites_signal(text: str) -> bool:
    toks = set(_tokenize(text))
    return len(toks & SUITES_KEYWORDS) > 0


# =========================
# SCRAPING PDF LINKS
# =========================
def extract_pdf_links_from_url(url: str) -> List[str]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT)
        r.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue
        if ".pdf" not in href.lower():
            continue

        # Absolutisation
        if href.startswith("http://") or href.startswith("https://"):
            links.append(href)
        else:
            base = url.rstrip("/")
            if href.startswith("/"):
                # récupère domaine
                m = re.match(r"^(https?://[^/]+)", base)
                if m:
                    links.append(m.group(1) + href)
            else:
                links.append(base + "/" + href)

    # dédoublonnage en conservant ordre
    seen = set()
    out = []
    for x in links:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_pdf_links(urls: List[str], limit: int) -> List[str]:
    all_links: List[str] = []
    for u in urls:
        all_links.extend(extract_pdf_links_from_url(u))
        if len(all_links) >= limit:
            break
    # dédoublonnage global
    seen = set()
    uniq = []
    for x in all_links:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
        if len(uniq) >= limit:
            break
    return uniq


# =========================
# TELECHARGEMENT PDF
# =========================
def download_pdf(url: str) -> Optional[bytes]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT, stream=True)
        r.raise_for_status()

        # check taille si dispo
        cl = r.headers.get("Content-Length")
        if cl:
            mb = int(cl) / (1024 * 1024)
            if mb > MAX_PDF_MB:
                return None

        data = r.content
        if len(data) > MAX_PDF_MB * 1024 * 1024:
            return None
        return data
    except Exception:
        return None


# =========================
# EXTRACTION TEXTE PDF
# =========================
def extract_text_from_pdf_bytes(pdf_bytes: bytes, max_pages: int = 25) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        n = min(len(pdf.pages), max_pages)
        for i in range(n):
            page = pdf.pages[i]
            t = page.extract_text() or ""
            if t.strip():
                text_parts.append(t)
    return "\n".join(text_parts)


# =========================
# EXTRACTION QI (HEURISTIQUE TRANSPARENTE)
# =========================
QI_SPLIT_PATTERNS = [
    r"\bexercice\s*\d+\b",
    r"\bquestion\s*\d+\b",
    r"\bpartie\s*[a-z0-9]+\b",
    r"^\s*\d+\s*[\).\:-]\s+",
]


def extract_qi_from_text(text: str) -> List[str]:
    raw = text.replace("\r", "\n")
    raw = re.sub(r"\n{2,}", "\n\n", raw)

    # segmentation grossière par blocs
    blocks = re.split(r"\n\s*\n", raw)
    candidates = []

    for b in blocks:
        b2 = b.strip()
        if len(b2) < MIN_QI_CHARS:
            continue

        # signal "énoncé" : présence de verbes fréquents
        if re.search(r"\b(calculer|déterminer|montrer|justifier|étudier|prouver)\b", b2, re.IGNORECASE):
            candidates.append(b2)
            continue

        # ou présence de mot-clé suites (phase 1)
        if _contains_suites_signal(b2):
            candidates.append(b2)

    # nettoyage léger : ne garder qu'une phrase / ligne représentative si bloc énorme
    qi = []
    for c in candidates:
        c = re.sub(r"\s+", " ", c).strip()
        if len(c) > 350:
            # garder le début informatif
            c = c[:350].rsplit(" ", 1)[0] + "…"
        if len(c) >= MIN_QI_CHARS:
            qi.append(c)

    # dédoublonnage
    seen = set()
    out = []
    for x in qi:
        k = _normalize(x)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


# =========================
# CLUSTERING -> QC (GREEDY JACCARD)
# =========================
@dataclass
class QiItem:
    subject_id: str
    subject_file: str
    text: str


def cluster_qi_to_qc(qis: List[QiItem], sim_threshold: float = 0.28) -> List[Dict]:
    clusters: List[Dict] = []  # each: {id, rep_tokens, qis: [QiItem]}
    qc_idx = 1

    for qi in qis:
        toks = _tokenize(qi.text)
        if not toks:
            continue

        best_i = None
        best_sim = 0.0

        for i, c in enumerate(clusters):
            sim = _jaccard(toks, c["rep_tokens"])
            if sim > best_sim:
                best_sim = sim
                best_i = i

        if best_i is not None and best_sim >= sim_threshold:
            clusters[best_i]["qis"].append(qi)
            # rep_tokens = union léger (stabilise)
            clusters[best_i]["rep_tokens"] = list(set(clusters[best_i]["rep_tokens"]) | set(toks))
        else:
            clusters.append({
                "id": f"QC-{qc_idx:03d}",
                "rep_tokens": toks,
                "qis": [qi],
            })
            qc_idx += 1

    # build QC objects
    qc_out = []
    for c in clusters:
        qi_texts = [q.text for q in c["qis"]]

        # titre QC = phrase représentative (premier Qi)
        title = qi_texts[0]
        title_short = title
        if len(title_short) > 90:
            title_short = title_short[:90].rsplit(" ", 1)[0] + "…"

        # triggers = top keywords (simple)
        tokens_all = []
        for t in qi_texts:
            tokens_all.extend(_tokenize(t))
        freq = {}
        for tok in tokens_all:
            if tok in {"le", "la", "les", "de", "des", "du", "un", "une", "et", "à", "a", "en", "pour"}:
                continue
            if len(tok) < 4:
                continue
            freq[tok] = freq.get(tok, 0) + 1
        triggers = [k for k, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:6]]

        # métriques transparentes
        n_q = len(qi_texts)
        psi = round(min(1.0, n_q / 25.0), 2)  # simple, assumé
        score = int(round(50 + 10 * math.log(1 + n_q), 0))

        # ARI/FRT : pas de fake -> placeholders explicites (phase test)
        ari = ["(TEST) ARI non généré automatiquement à ce stade."]
        frt = {
            "usage": "(TEST) FRT non générée automatiquement à ce stade.",
            "method": "(TEST) FRT non générée automatiquement à ce stade.",
            "trap": "(TEST) FRT non générée automatiquement à ce stade.",
            "conc": "(TEST) FRT non générée automatiquement à ce stade.",
        }

        qc_out.append({
            "chapter": "SUITES NUMÉRIQUES",  # phase 1
            "qc_id": c["id"],
            "qc_title": title_short,
            "score": score,
            "n_q": n_q,
            "psi": psi,
            "n_tot": len(qis),
            "t_rec": 0.0,
            "triggers": triggers,
            "ari": ari,
            "frt": frt,
            # mapping Qi par fichier
            "qi_by_file": _group_qi_by_file(c["qis"]),
        })

    return qc_out


def _group_qi_by_file(qis: List[QiItem]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for q in qis:
        out.setdefault(q.subject_file, []).append(q.text)
    return out


# =========================
# SATURATION
# =========================
def compute_saturation(history_counts: List[int]) -> List[Dict]:
    # history_counts[i] = nb QC après i+1 sujets
    sat = []
    for i, v in enumerate(history_counts):
        sat.append({"Nombre de sujets injectés": i + 1, "Nombre de QC découvertes": v})
    return sat


# =========================
# API PRINCIPALE POUR UI
# =========================
def run_granulo_test(urls: List[str], volume: int) -> Dict:
    """
    Retourne un dict :
    - sujets: rows pour dataframe
    - qc: liste QC structurées
    - saturation: points courbe
    - audit: métriques
    """
    start = time.time()

    pdf_links = extract_pdf_links(urls, limit=volume)

    sujets_rows = []
    all_qis: List[QiItem] = []
    qc_history = []

    for idx, pdf_url in enumerate(pdf_links, start=1):
        pdf_bytes = download_pdf(pdf_url)
        if not pdf_bytes:
            continue

        text = extract_text_from_pdf_bytes(pdf_bytes)
        if not text.strip():
            continue

        # extraction Qi
        qi_texts = extract_qi_from_text(text)

        # Filtrage phase 1 suites (aucun fake : on jette si pas signal)
        qi_texts = [q for q in qi_texts if _contains_suites_signal(q)]

        subject_file = pdf_url.split("/")[-1].split("?")[0]
        source_host = re.sub(r"^https?://", "", pdf_url).split("/")[0]

        sujets_rows.append({
            "Fichier": subject_file,
            "Nature": "INCONNU",
            "Année": None,
            "Source": source_host,
        })

        subject_id = f"S{idx:04d}"
        for q in qi_texts:
            all_qis.append(QiItem(subject_id=subject_id, subject_file=subject_file, text=q))

        # QC cumulées (saturation)
        qc_current = cluster_qi_to_qc(all_qis)
        qc_history.append(len(qc_current))

    qc_list = cluster_qi_to_qc(all_qis)
    sat_points = compute_saturation(qc_history)

    elapsed = round(time.time() - start, 2)

    return {
        "sujets": sujets_rows,
        "qc": qc_list,
        "saturation": sat_points,
        "audit": {
            "n_urls": len(urls),
            "n_pdf_links": len(pdf_links),
            "n_subjects_ok": len(sujets_rows),
            "n_qi": len(all_qis),
            "n_qc": len(qc_list),
            "elapsed_s": elapsed,
        }
    }
