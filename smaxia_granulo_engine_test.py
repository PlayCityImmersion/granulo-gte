# smaxia_granulo_engine_test.py
from __future__ import annotations

import io
import math
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
import pdfplumber
from bs4 import BeautifulSoup
from collections import Counter


# =========================
# CONFIG TEST (TRANSPARENT)
# =========================
UA = "SMAXIA-GTE/1.0 (+granulo-test-engine)"
REQ_TIMEOUT = 25
MAX_PDF_MB = 25          # sécurité
MAX_PDF_PAGES = 35       # extraction texte plafonnée
MIN_QI_CHARS = 18

# Crawl (HTML)
MAX_HTML_PAGES = 60      # garde-fou Streamlit Cloud
MAX_CRAWL_DEPTH = 2      # 0 seed, 1 liens, 2 liens de liens

# Filtrage "Suites numériques" (phase 1 France Terminale spé)
SUITES_KEYWORDS = {
    "suite", "suites", "arithmétique", "arithmetique", "géométrique", "geometrique",
    "raison", "u_n", "u(n)", "un", "u0", "u1", "u2", "u_{n}", "u_{n+1}",
    "récurrence", "recurrence", "limite", "convergence", "divergence",
    "monotone", "croissante", "décroissante", "decroissante", "bornée", "bornee",
    "majorée", "majoree", "minorée", "minoree",
    "somme", "sommes", "terme", "terme général", "terme general"
}

# Liens HTML à privilégier (APMEP Annales)
FOLLOW_HINTS = (
    "annales", "terminale", "generale", "générale", "specialite", "spécialité",
    "enseignement", "math", "bac", "sujets", "corrig", "ts", "tle", "examen"
)

# PDFs à rejeter (parasites)
PDF_URL_BLACKLIST = (
    "pv", "bulletin", "sommaire", "édito", "edito", "regionale", "régionale",
    "journal", "vie-de", "vie de", "litteramath", "littéramath", "revue",
    "compte-rendu", "compte rendu", "editorial", "éditorial", "agenda"
)

# Contenu (texte) à rejeter (parasites) : détecté sur le texte extrait
TEXT_BLACKLIST = (
    "sommaire", "éditorial", "editorial", "procès-verbal", "proces-verbal",
    "compte rendu", "compte-rendu", "assemblée générale", "bureau", "adhérents",
    "vie de la régionale", "agenda", "courrier des lecteurs"
)

# Stopwords FR minimaux (déclencheurs)
STOPWORDS_FR = {
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "ou", "à", "a", "au",
    "aux", "en", "pour", "par", "sur", "dans", "avec", "sans", "que", "qui", "quoi",
    "dont", "où", "est", "sont", "être", "etre", "soit", "on", "il", "elle", "ils",
    "elles", "ce", "cet", "cette", "ces", "se", "sa", "son", "ses", "leur", "leurs",
    "plus", "moins", "très", "tres", "afin", "ainsi", "alors", "donc"
}


# =========================
# OUTILS TEXTE
# =========================
def _normalize(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _tokenize(text: str) -> List[str]:
    """
    Tokenisation simple + tolérance accents.
    On garde aussi certains tokens utiles (un, u_n) via normalisation.
    """
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


def _looks_like_subject_text(text: str) -> bool:
    """
    Rejette PV/sommaires via blacklist.
    """
    t = _normalize(text)
    if not t:
        return False
    bad_hits = sum(1 for b in TEXT_BLACKLIST if b in t)
    # si trop de signaux "magazine", on rejette
    if bad_hits >= 2:
        return False
    return True


def _is_blacklisted_pdf_url(url: str) -> bool:
    u = (url or "").lower()
    return any(b in u for b in PDF_URL_BLACKLIST)


# =========================
# SCRAPING PDF LINKS (CRAWL SAFE)
# =========================
def _is_http_url(u: str) -> bool:
    try:
        p = urlparse(u)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


def _same_domain(seed_netloc: str, u: str) -> bool:
    try:
        netloc = urlparse(u).netloc.lower()
        seed_netloc = seed_netloc.lower()
        return netloc == seed_netloc or netloc.endswith("." + seed_netloc)
    except Exception:
        return False


def _looks_like_pdf(u: str) -> bool:
    # couvre .pdf?download=1 etc.
    return ".pdf" in (u or "").lower()


def _should_follow_link(href_abs: str, anchor_text: str, seed_netloc: str) -> bool:
    if not _is_http_url(href_abs):
        return False
    if not _same_domain(seed_netloc, href_abs):
        return False
    h = href_abs.lower()
    if any(h.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".zip", ".rar", ".7z")):
        return False
    if "mailto:" in h or "javascript:" in h:
        return False

    t = (anchor_text or "").strip().lower()
    return any(k in h for k in FOLLOW_HINTS) or any(k in t for k in FOLLOW_HINTS)


def extract_pdf_links_from_url(url: str) -> List[str]:
    """
    Crawl contrôlé : extrait PDFs sur la seed + pages Annales à 1–2 niveaux.
    """
    seed_netloc = urlparse(url).netloc or ""
    if not seed_netloc:
        return []

    visited_pages = set()
    seen_pdf = set()
    pdf_links: List[str] = []

    queue: List[Tuple[str, int]] = [(url, 0)]

    while queue and len(visited_pages) < MAX_HTML_PAGES and len(pdf_links) < 2000:
        page_url, depth = queue.pop(0)
        if page_url in visited_pages:
            continue
        visited_pages.add(page_url)

        try:
            r = requests.get(page_url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT)
            r.raise_for_status()
        except Exception:
            continue

        ctype = (r.headers.get("Content-Type") or "").lower()
        # si c'est un PDF direct
        if "application/pdf" in ctype or page_url.lower().endswith(".pdf"):
            if _looks_like_pdf(page_url) and (page_url not in seen_pdf) and not _is_blacklisted_pdf_url(page_url):
                seen_pdf.add(page_url)
                pdf_links.append(page_url)
            continue

        soup = BeautifulSoup(r.text, "html.parser")

        for a in soup.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            href_abs = urljoin(page_url, href)

            # PDF
            if _looks_like_pdf(href_abs):
                if _is_blacklisted_pdf_url(href_abs):
                    continue
                if href_abs not in seen_pdf:
                    seen_pdf.add(href_abs)
                    pdf_links.append(href_abs)
                continue

            # Follow HTML
            if depth < MAX_CRAWL_DEPTH:
                text = a.get_text(" ", strip=True) if a else ""
                if _should_follow_link(href_abs, text, seed_netloc):
                    if href_abs not in visited_pages:
                        queue.append((href_abs, depth + 1))

    # dédoublonnage (ordre)
    out = []
    seen = set()
    for x in pdf_links:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_pdf_links(urls: List[str], limit: int) -> List[str]:
    """
    Récolte plus large que 'limit' pour compenser les rejets PDF (scan, parasite, etc.).
    """
    target = max(limit * 10, limit)  # buffer
    all_links: List[str] = []
    seen = set()

    for u in urls:
        links = extract_pdf_links_from_url(u)
        for x in links:
            if x not in seen:
                seen.add(x)
                all_links.append(x)
            if len(all_links) >= target:
                break
        if len(all_links) >= target:
            break

    return all_links[:target]


# =========================
# TELECHARGEMENT PDF
# =========================
def download_pdf(url: str) -> Optional[bytes]:
    """
    Télécharge un PDF (stream) et applique un plafond taille.
    """
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT, stream=True)
        r.raise_for_status()

        ctype = (r.headers.get("Content-Type") or "").lower()
        if "pdf" not in ctype and ".pdf" not in url.lower():
            return None

        cl = r.headers.get("Content-Length")
        if cl:
            mb = int(cl) / (1024 * 1024)
            if mb > MAX_PDF_MB:
                return None

        buf = io.BytesIO()
        max_bytes = MAX_PDF_MB * 1024 * 1024
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if not chunk:
                continue
            buf.write(chunk)
            if buf.tell() > max_bytes:
                return None

        return buf.getvalue()
    except Exception:
        return None


# =========================
# EXTRACTION TEXTE PDF
# =========================
def extract_text_from_pdf_bytes(pdf_bytes: bytes, max_pages: int = MAX_PDF_PAGES) -> str:
    """
    Extraction texte via pdfplumber (si PDF scanné -> texte vide).
    """
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        n = min(len(pdf.pages), max_pages)
        for i in range(n):
            page = pdf.pages[i]
            t = page.extract_text() or ""
            # normalisation légère
            t = t.replace("\u00ad", "")  # soft hyphen
            if t.strip():
                text_parts.append(t)
    return "\n".join(text_parts)


# =========================
# EXTRACTION QI (HEURISTIQUE ROBUSTE)
# =========================
VERBS = (
    "calculer", "déterminer", "determiner", "montrer", "démontrer", "demontrer",
    "justifier", "étudier", "etudier", "résoudre", "resoudre", "vérifier", "verifier",
    "prouver", "en déduire", "en deduire", "exprimer"
)

# signaux "math" simples
MATH_SIGNALS_RE = re.compile(r"(\bu[_\s\{\(]*n\b|\bu[_\s\{\(]*n\+1\b|\blim\b|\b\d+\b|[=<>±+\-*/^]|sqrt|ln|log|exp)", re.IGNORECASE)

# repère début de question
QI_START_RE = re.compile(
    r"^\s*(?:exercice\s*\d+|question\s*\d+|partie\s*[a-z0-9]+|\d+\s*[\).\:-]|[a-z]\)\s*)\s*(.*)$",
    re.IGNORECASE
)

# phrase “impérative” de maths
QI_VERB_RE = re.compile(r"\b(" + "|".join([re.escape(v) for v in VERBS]) + r")\b", re.IGNORECASE)


def extract_qi_from_text(text: str) -> List[str]:
    """
    Stratégie :
    - découpe par lignes
    - détecte lignes question (verbe + signal math) et agrège sur 1–3 lignes si besoin
    - filtre longueur
    - dédoublonne
    """
    raw = (text or "").replace("\r", "\n")
    raw = re.sub(r"\n{3,}", "\n\n", raw)

    lines = [ln.strip() for ln in raw.split("\n")]
    lines = [ln for ln in lines if ln]

    qi_out: List[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]

        # normaliser un peu
        ln_clean = re.sub(r"\s+", " ", ln).strip()

        # capturer contenu après "1." / "a)" / "Exercice"
        m = QI_START_RE.match(ln_clean)
        candidate = m.group(1).strip() if m else ln_clean

        # test Qi : verbe + signal math OU suites signal
        verb_ok = bool(QI_VERB_RE.search(candidate))
        math_ok = bool(MATH_SIGNALS_RE.search(candidate))
        suites_ok = _contains_suites_signal(candidate)

        if (verb_ok and (math_ok or suites_ok)) or (suites_ok and len(candidate) >= MIN_QI_CHARS):
            # tenter d’agréger la suite si c’est une phrase coupée
            agg = [candidate]
            # ajoute 1–2 lignes suivantes si elles semblent continuation (pas un nouveau numéro)
            for j in range(1, 3):
                if i + j >= len(lines):
                    break
                nxt = lines[i + j]
                if QI_START_RE.match(nxt):
                    break
                # continuation si contient math ou suites
                if MATH_SIGNALS_RE.search(nxt) or _contains_suites_signal(nxt):
                    agg.append(nxt)
            qi = " ".join(agg)
            qi = re.sub(r"\s+", " ", qi).strip()

            if len(qi) > 420:
                qi = qi[:420].rsplit(" ", 1)[0] + "…"

            if len(qi) >= MIN_QI_CHARS:
                qi_out.append(qi)

        i += 1

    # dédoublonnage
    seen = set()
    final = []
    for q in qi_out:
        k = _normalize(q)
        if k not in seen:
            seen.add(k)
            final.append(q)
    return final


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


def _extract_triggers(qi_texts: List[str], k: int = 7) -> List[str]:
    tokens_all: List[str] = []
    for t in qi_texts:
        tokens_all.extend(_tokenize(t))

    # filtrage stopwords + bruit
    filtered = []
    for tok in tokens_all:
        if tok in STOPWORDS_FR:
            continue
        if tok.isdigit():
            continue
        if len(tok) < 4:
            continue
        filtered.append(tok)

    freq = Counter(filtered)
    return [w for w, _ in freq.most_common(k)]


def _infer_type(blob: str) -> str:
    b = _normalize(blob)
    if "récurrence" in b or "recurrence" in b:
        return "RECURRENCE"
    if "géométrique" in b or "geometrique" in b:
        return "GEOMETRIQUE"
    if "arithmétique" in b or "arithmetique" in b:
        return "ARITHMETIQUE"
    if "limite" in b and ("indétermination" in b or "indetermination" in b):
        return "LIM_INDET"
    if "limite" in b:
        return "LIMITE"
    if "monotone" in b or "croissante" in b or "décroissante" in b or "decroissante" in b:
        return "MONOTONIE"
    if "somme" in b or "sommes" in b:
        return "SOMME"
    return "GENERIC"


def _build_ari_and_frt(qi_texts: List[str], triggers: List[str]) -> Tuple[List[str], Dict[str, str], str]:
    blob = " ".join(qi_texts)
    t = _infer_type(blob)

    # Titre QC : première Qi (raccourcie)
    title = qi_texts[0].strip() if qi_texts else "Question"
    if len(title) > 90:
        title = title[:90].rsplit(" ", 1)[0] + "…"

    if t == "RECURRENCE":
        ari = [
            "Identifier la propriété P(n) à prouver.",
            "Initialisation : vérifier P(0) ou P(1) selon l’énoncé.",
            "Hérédité : supposer P(n) vraie, démontrer P(n+1).",
            "Conclusion : P(n) vraie pour tout n (principe de récurrence)."
        ]
        frt = {
            "usage": "Utiliser si l’énoncé demande « montrer pour tout n » avec une relation liant n et n+1.",
            "method": "Écrire P(n). Faire initialisation, puis hérédité (P(n) ⇒ P(n+1)), conclure.",
            "trap": "Oublier l’initialisation, ou écrire une hérédité qui n’utilise pas vraiment P(n).",
            "conc": "Conclure explicitement « donc pour tout n … »."
        }

    elif t == "GEOMETRIQUE":
        ari = [
            "Exprimer u(n+1) en fonction de u(n) (ou de n).",
            "Calculer le quotient u(n+1)/u(n).",
            "Vérifier que ce quotient est constant : raison q.",
            "Conclure : suite géométrique, puis écrire u(n)=u(0)·q^n."
        ]
        frt = {
            "usage": "Utiliser si l’énoncé parle de « suite géométrique », « raison », ou si u(n+1)=q·u(n).",
            "method": "Calculer u(n+1)/u(n), montrer que c’est une constante q, puis donner la forme explicite.",
            "trap": "Confondre u(n+1)-u(n) (arithmétique) et u(n+1)/u(n) (géométrique).",
            "conc": "Donner q et l’expression explicite u(n)=u(0)·q^n."
        }

    elif t == "ARITHMETIQUE":
        ari = [
            "Exprimer u(n+1) et u(n).",
            "Calculer la différence u(n+1) - u(n).",
            "Vérifier que la différence est constante : raison r.",
            "Conclure : u(n)=u(0)+n·r."
        ]
        frt = {
            "usage": "Utiliser si l’énoncé parle de « suite arithmétique » ou si u(n+1)=u(n)+r.",
            "method": "Calculer u(n+1)-u(n), montrer que c’est constant r, puis écrire u(n)=u(0)+n·r.",
            "trap": "Calculer un quotient au lieu d’une différence.",
            "conc": "Donner r et l’expression explicite."
        }

    elif t == "MONOTONIE":
        ari = [
            "Choisir une méthode : u(n+1)-u(n) ou u(n+1)/u(n) selon la forme.",
            "Étudier le signe de u(n+1)-u(n) (ou comparer u(n+1) à u(n)).",
            "Conclure : croissante/décroissante, puis exploiter bornes si nécessaire.",
        ]
        frt = {
            "usage": "Utiliser si l’énoncé demande « variations », « monotone », « sens de variation ».",
            "method": "Étudier u(n+1)-u(n) : si ≥0 alors croissante, si ≤0 alors décroissante.",
            "trap": "Oublier la justification du signe, ou conclure sans préciser « pour tout n ».",
            "conc": "Conclure clairement le sens de variation."
        }

    elif t == "LIM_INDET":
        ari = [
            "Identifier la forme indéterminée (0/0, ∞/∞, etc.).",
            "Transformer : factoriser, rationaliser, ou utiliser un équivalent.",
            "Simplifier, puis calculer la limite.",
            "Conclure (valeur finie / ±∞ / inexistence)."
        ]
        frt = {
            "usage": "Utiliser si la limite donne une indétermination.",
            "method": "Factoriser/équivalents, simplifier, recalculer la limite sur l’expression simplifiée.",
            "trap": "Simplifier illégalement (division par 0) ou oublier le domaine.",
            "conc": "Donner la limite et justifier la transformation utilisée."
        }

    elif t == "LIMITE":
        ari = [
            "Identifier le comportement des termes (borné, dominant, etc.).",
            "Appliquer théorèmes usuels (somme/produit/quotient) ou encadrement.",
            "Conclure la limite."
        ]
        frt = {
            "usage": "Utiliser si l’énoncé demande « calculer la limite de u(n) ».",
            "method": "Analyser les termes, appliquer les règles sur les limites ou un encadrement.",
            "trap": "Appliquer une règle sans vérifier les hypothèses (division, signe, etc.).",
            "conc": "Conclure explicitement « lim u(n) = … »."
        }

    elif t == "SOMME":
        ari = [
            "Identifier la forme (somme de termes, télescopage, géométrique…).",
            "Écrire la somme partielle S(n) et simplifier.",
            "Conclure (expression de S(n) ou limite si demandée)."
        ]
        frt = {
            "usage": "Utiliser si l’énoncé demande une somme de termes de suite.",
            "method": "Écrire S(n), utiliser une forme connue (géométrique / télescopique), simplifier.",
            "trap": "Se tromper d’indices (0..n, 1..n) ou oublier un terme.",
            "conc": "Donner l’expression finale de la somme."
        }

    else:
        ari = [
            "Identifier les données de l’énoncé (définition de la suite).",
            "Choisir l’outil : récurrence / comparaison / calcul explicite.",
            "Dérouler les étapes de calcul.",
            "Conclure."
        ]
        frt = {
            "usage": "Utiliser pour une question standard sur suite si le type exact n’est pas détecté.",
            "method": "Reformuler, choisir l’outil, calculer proprement, conclure.",
            "trap": "Sauter une justification ou mélanger plusieurs méthodes incohérentes.",
            "conc": "Conclusion explicite + résultat final."
        }

    return ari, frt, title


def cluster_qi_to_qc(qis: List[QiItem], sim_threshold: float = 0.30) -> List[Dict]:
    """
    Clustering greedy basé sur Jaccard tokens.
    """
    clusters: List[Dict] = []  # {id, rep_tokens, qis:[QiItem]}
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

        if best_i is not None a_
