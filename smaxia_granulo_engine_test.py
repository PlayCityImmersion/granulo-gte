from __future__ import annotations

import io
import math
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests

# =========================
# CONFIG TEST (TRANSPARENT)
# =========================
UA = "SMAXIA-GTE/1.1 (+saturation-proof)"
REQ_TIMEOUT = 20
MAX_PDF_MB = 25  # sécurité
MIN_QI_CHARS = 18

# Crawl léger (1 niveau) pour trouver plus de PDFs depuis une page "index"
MAX_CRAWL_PAGES_PER_SOURCE = 25
MAX_HTML_BYTES = 2_000_000

# Filtrage "Suites numériques" (phase 1 France Terminale spé)
SUITES_KEYWORDS: Set[str] = {
    "suite", "suites", "arithmétique", "arithmetique", "géométrique", "geometrique",
    "raison", "u_n", "un", "v_n", "récurrence", "recurrence", "limite", "convergence",
    "monotone", "bornée", "bornee", "borne", "majorée", "majoree", "minorée", "minoree",
    "sommes", "somme", "terme", "général", "general"
}

# déclencheurs “réels” à détecter dans le texte
TRIGGER_PHRASES = [
    "montrer que",
    "démontrer que",
    "demontrer que",
    "en déduire",
    "en deduire",
    "justifier",
    "calculer",
    "déterminer",
    "determiner",
    "étudier",
    "etudier",
    "prouver",
]

STOP_TOKENS = {
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "à", "a", "en", "pour",
    "dans", "sur", "avec", "par", "au", "aux", "d", "l", "si", "que", "qui", "on",
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
    toks = re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ0-9_]+", t)
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


def _extract_trigger_phrases(text: str) -> List[str]:
    t = _normalize(text)
    found = []
    for p in TRIGGER_PHRASES:
        if p in t:
            found.append(p)
    # dédoublonnage ordre
    out, seen = [], set()
    for x in found:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _top_keywords(texts: List[str], k: int = 6) -> List[str]:
    freq: Dict[str, int] = {}
    for s in texts:
        for tok in _tokenize(s):
            if tok in STOP_TOKENS:
                continue
            if len(tok) < 4:
                continue
            freq[tok] = freq.get(tok, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:k]]


# =========================
# HTML FETCH + LINK EXTRACTION
# =========================
def _safe_get_html(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        # limiter taille
        txt = r.text
        if len(txt.encode("utf-8", errors="ignore")) > MAX_HTML_BYTES:
            return txt[:MAX_HTML_BYTES]
        return txt
    except Exception:
        return None


PDF_HREF_RE = re.compile(r'href=["\']([^"\']+?\.pdf(?:\?[^"\']*)?)["\']', re.IGNORECASE)
A_HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)

def _extract_pdf_links_from_html(base_url: str, html: str) -> List[str]:
    links = []
    for m in PDF_HREF_RE.finditer(html):
        href = m.group(1).strip()
        if not href:
            continue
        links.append(urljoin(base_url, href))
    # dédoublonnage ordre
    out, seen = [], set()
    for x in links:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _extract_internal_pages(base_url: str, html: str, same_host: str) -> List[str]:
    pages = []
    for m in A_HREF_RE.finditer(html):
        href = m.group(1).strip()
        if not href:
            continue
        if href.lower().endswith(".pdf"):
            continue
        full = urljoin(base_url, href)
        try:
            host = urlparse(full).netloc
        except Exception:
            continue
        if host != same_host:
            continue
        pages.append(full)

    out, seen = [], set()
    for x in pages:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_pdf_links_from_url(url: str) -> List[str]:
    """
    Extraction PDF depuis :
    - la page source
    - un crawl léger (1 niveau, borné) des pages internes
    """
    html = _safe_get_html(url)
    if not html:
        return []

    same_host = urlparse(url).netloc

    pdfs = _extract_pdf_links_from_html(url, html)

    # crawl 1 niveau (borné) si la page est un index
    internal_pages = _extract_internal_pages(url, html, same_host)[:MAX_CRAWL_PAGES_PER_SOURCE]
    for p in internal_pages:
        html2 = _safe_get_html(p)
        if not html2:
            continue
        pdfs.extend(_extract_pdf_links_from_html(p, html2))

    # dédoublonnage global ordre
    out, seen = [], set()
    for x in pdfs:
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
# TELECHARGEMENT PDF + VALIDATION
# =========================
def _looks_like_pdf(data: bytes) -> bool:
    return data[:4] == b"%PDF"

def download_pdf(url: str) -> Optional[bytes]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT, stream=True)
        r.raise_for_status()

        ct = (r.headers.get("Content-Type") or "").lower()
        # On accepte si content-type PDF ou si ça ressemble à un PDF
        data = r.content
        if not data:
            return None

        cl = r.headers.get("Content-Length")
        if cl:
            mb = int(cl) / (1024 * 1024)
            if mb > MAX_PDF_MB:
                return None

        if len(data) > MAX_PDF_MB * 1024 * 1024:
            return None

        if ("pdf" not in ct) and (not _looks_like_pdf(data)):
            return None

        if not _looks_like_pdf(data):
            # certains serveurs ajoutent un header, on cherche %PDF
            pos = data.find(b"%PDF")
            if pos == -1:
                return None
            data = data[pos:]

        return data
    except Exception:
        return None


# =========================
# EXTRACTION TEXTE PDF (IMPORTS RETARDÉS)
# =========================
def extract_text_from_pdf_bytes(pdf_bytes: bytes, max_pages: int = 25) -> str:
    # 1) pdfplumber si dispo
    try:
        import pdfplumber  # type: ignore
        text_parts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            n = min(len(pdf.pages), max_pages)
            for i in range(n):
                t = pdf.pages[i].extract_text() or ""
                if t.strip():
                    text_parts.append(t)
        return "\n".join(text_parts)
    except Exception:
        pass

    # 2) fallback PyPDF2 si dispo
    try:
        from PyPDF2 import PdfReader  # type: ignore
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text_parts = []
        n = min(len(reader.pages), max_pages)
        for i in range(n):
            t = reader.pages[i].extract_text() or ""
            if t.strip():
                text_parts.append(t)
        return "\n".join(text_parts)
    except Exception:
        return ""


# =========================
# EXTRACTION QI (HEURISTIQUE TRANSPARENTE)
# =========================
def extract_qi_from_text(text: str) -> List[str]:
    raw = text.replace("\r", "\n")
    raw = re.sub(r"\n{2,}", "\n\n", raw)

    blocks = re.split(r"\n\s*\n", raw)
    candidates = []

    for b in blocks:
        b2 = b.strip()
        if len(b2) < MIN_QI_CHARS:
            continue

        # signal “question” : verbes déclencheurs
        if re.search(r"\b(calculer|déterminer|determiner|montrer|démontrer|demontrer|justifier|étudier|etudier|prouver)\b", b2, re.IGNORECASE):
            candidates.append(b2)
            continue

        # ou signal “suites”
        if _contains_suites_signal(b2):
            candidates.append(b2)

    qi = []
    for c in candidates:
        c = re.sub(r"\s+", " ", c).strip()
        if len(c) > 420:
            c = c[:420].rsplit(" ", 1)[0] + "…"
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


def _group_qi_by_file(qis: List[QiItem]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for q in qis:
        out.setdefault(q.subject_file, []).append(q.text)
    return out


def _build_frt(qi_texts: List[str], triggers: List[str]) -> Dict[str, str]:
    """
    FRT standard (déterministe) basée sur signaux suites :
    - usage: quand appliquer
    - method: étapes
    - trap: erreurs classiques
    - conc: format de réponse
    """
    txt = " ".join(qi_texts).lower()
    is_arith = ("arithm" in txt) or ("raison" in txt and "arith" in txt)
    is_geo = ("géométr" in txt) or ("geomet" in txt)
    is_recur = ("récurr" in txt) or ("recurr" in txt) or ("u_{n+1}" in txt) or ("un+1" in txt)
    wants_limit = ("limite" in txt) or ("converg" in txt)

    chap = "Suites numériques"
    if is_arith:
        subtype = "suite arithmétique"
    elif is_geo:
        subtype = "suite géométrique"
    elif is_recur:
        subtype = "suite définie par récurrence"
    else:
        subtype = "suite (cadre général)"

    trig_txt = ", ".join(triggers) if triggers else "analyser / déterminer"

    usage = (
        f"À utiliser lorsque l’énoncé demande : {trig_txt} "
        f"et porte sur {chap} ({subtype})."
    )

    steps = []
    steps.append("Identifier la définition de la suite (explicite / récurrence) et les données initiales.")
    if is_arith:
        steps.append("Reconnaître la forme u(n+1)=u(n)+r et déterminer la raison r puis u(n)=u(0)+nr.")
    if is_geo:
        steps.append("Reconnaître la forme u(n+1)=q·u(n) et déterminer la raison q puis u(n)=u(0)·q^n.")
    if is_recur and (not is_arith) and (not is_geo):
        steps.append("Écrire clairement l’hypothèse de récurrence et dérouler 2–3 termes pour valider le schéma.")
    if wants_limit:
        steps.append("Étudier monotonie + bornes, conclure convergence, puis calculer la limite (équation de point fixe si récurrence).")
    steps.append("Rédiger la conclusion avec la formule demandée et/ou la valeur numérique.")

    method = "<br>".join([f"{i+1}) {s}" for i, s in enumerate(steps)])

    traps = []
    traps.append("Oublier la condition initiale (u0/u1) ou confondre l’indice n.")
    if is_arith:
        traps.append("Confondre la raison r avec un terme de la suite.")
    if is_geo:
        traps.append("Confondre q et u0 ; oublier q^n (et les cas q=0, q=1, q<0 si pertinent).")
    if is_recur:
        traps.append("Manipuler la récurrence sans vérifier le domaine / la stabilité (bornes).")
    if wants_limit:
        traps.append("Conclure la limite sans justifier monotonie + bornes (ou sans théorème adapté).")

    trap = "<br>".join([f"• {t}" for t in traps])

    conc = (
        "Conclusion attendue : donner l’expression de u(n) / la raison / la limite, "
        "avec justification minimale (monotonie + bornes si convergence)."
    )

    return {"usage": usage, "method": method, "trap": trap, "conc": conc}


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
            clusters[best_i]["rep_tokens"] = list(set(clusters[best_i]["rep_tokens"]) | set(toks))
        else:
            clusters.append({
                "id": f"QC-{qc_idx:03d}",
                "rep_tokens": toks,
                "qis": [qi],
            })
            qc_idx += 1

    qc_out = []
    for c in clusters:
        qi_texts = [q.text for q in c["qis"]]

        # titre QC = premier Qi (représentatif)
        title = qi_texts[0]
        title_short = title if len(title) <= 90 else title[:90].rsplit(" ", 1)[0] + "…"

        # déclencheurs réels (présents)
        trig = []
        for t in qi_texts:
            trig.extend(_extract_trigger_phrases(t))
        # dédoublonnage
        trig2, seen = [], set()
        for x in trig:
            if x not in seen:
                seen.add(x)
                trig2.append(x)

        # si aucun déclencheur détecté, fallback top keywords
        if not trig2:
            trig2 = _top_keywords(qi_texts, k=6)

        n_q = len(qi_texts)
        psi = round(min(1.0, n_q / 25.0), 2)
        score = int(round(50 + 10 * math.log(1 + n_q), 0))

        # ARI réel : séquence de Qi (tronquée, mais réelle)
        ari = []
        for i, qtxt in enumerate(qi_texts[:8], 1):
            short = qtxt if len(qtxt) <= 120 else qtxt[:120].rsplit(" ", 1)[0] + "…"
            ari.append(f"{i}. {short}")
        if len(qi_texts) > 8:
            ari.append(f"… +{len(qi_texts)-8} Qi suivantes")

        # FRT standard (règles déterministes)
        frt = _build_frt(qi_texts, trig2[:6])

        qc_out.append({
            "chapter": "SUITES NUMÉRIQUES",  # phase 1
            "qc_id": c["id"],
            "qc_title": title_short,
            "score": score,
            "n_q": n_q,
            "psi": psi,
            "n_tot": len(qis),
            "t_rec": 0.0,
            "triggers": trig2[:6],
            "ari": ari,
            "frt": frt,
            "qi_by_file": _group_qi_by_file(c["qis"]),
        })

    return qc_out


# =========================
# SATURATION
# =========================
def compute_saturation(history_counts: List[int]) -> List[Dict]:
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

    # nettoyage urls
    urls = [u.strip() for u in (urls or []) if u and u.strip()]
    if not urls:
        urls = ["https://www.apmep.fr"]

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

        qi_texts = extract_qi_from_text(text)

        # Filtrage phase 1 suites : on garde seulement les Qi avec signal suites
        qi_texts = [q for q in qi_texts if _contains_suites_signal(q)]
        if not qi_texts:
            continue

        subject_file = pdf_url.split("/")[-1].split("?")[0]
        source_host = urlparse(pdf_url).netloc or re.sub(r"^https?://", "", pdf_url).split("/")[0]

        sujets_rows.append({
            "Fichier": subject_file,
            "Nature": "INCONNU",
            "Année": None,
            "Source": source_host,
        })

        subject_id = f"S{idx:04d}"
        for q in qi_texts:
            all_qis.append(QiItem(subject_id=subject_id, subject_file=subject_file, text=q))

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
