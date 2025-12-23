# smaxia_granulo_engine_test.py
from __future__ import annotations

import io
import math
import re
import time
import urllib.parse
from collections import deque, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import requests
import pdfplumber
from bs4 import BeautifulSoup


# =========================
# CONFIG (safe for Streamlit Cloud)
# =========================
UA = "SMAXIA-GTE/1.1 (+granulo-test-engine)"
REQ_TIMEOUT = 25
MAX_PDF_MB = 30
MAX_HTML_PAGES = 120               # limite de crawl
MAX_PDFS_COLLECT_MULT = 12         # on collecte plus que "volume" puis on filtre
MAX_PDF_PAGES_TEXT = 30            # extraction texte (pages max)
MIN_TEXT_CHARS_PDF = 400           # sinon c'est souvent un scan (ou vide)
MIN_QI_CHARS = 25

# Crawl: tokens utiles
FOLLOW_TOKENS = {
    "annales", "terminale", "bac", "sujet", "sujets", "corrig", "corrigé", "corriges",
    "enseignement", "specialite", "spécialité", "math", "maths", "examen",
    "ts", "tale", "générale", "generale", "e3c", "epreuve", "épreuve"
}
AVOID_TOKENS = {
    "bulletin", "édito", "edito", "sommaire", "pv", "regionale", "régionale", "compte-rendu",
    "cr", "association", "statuts", "adhesion", "adhésion", "newsletter", "actualite", "actualité",
    "agenda", "forum", "contact", "mentions-legales", "privacy", "cookie", "don", "boutique"
}

# Filtrage "Suites numériques" (Terminale - Maths)
SUITES_KEYWORDS = {
    "suite", "suites", "arithmétique", "arithmetique", "géométrique", "geometrique", "raison",
    "u_n", "un", "u(n)", "u(n+1)", "u_{n}", "u_{n+1}", "récurrence", "recurrence",
    "limite", "convergence", "monotone", "bornée", "bornee", "borne", "majorée", "majoree",
    "minorée", "minoree", "croissante", "décroissante", "decroissante", "somme", "sommes",
    "terme général", "terme general", "rang", "n∈", "n in", "∀n", "pour tout n"
}

# Stopwords minimalistes
STOP = {
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "à", "a", "en", "pour", "dans",
    "sur", "au", "aux", "par", "avec", "que", "qui", "quoi", "dont", "où", "ou", "ce", "cet",
    "cette", "ces", "il", "elle", "on", "se", "sa", "son", "ses", "leur", "leurs", "plus",
    "moins", "très", "tres"
}

# Indices "math"
MATH_SYMBOL_RE = re.compile(r"([=<>≤≥±∑∫√π]|\\frac|\\sqrt|\bu_n\b|\bu_{n}\b|\bu_{n\+1}\b|\bn\in\b|\b∀\b|\blim\b|\bf\(x\)|\bx\^)")
MATH_VERBS_RE = re.compile(r"\b(calculer|déterminer|determiner|montrer|justifier|étudier|etudier|prouver|démontrer|demontrer|conjecturer|vérifier|verifier)\b", re.IGNORECASE)


# =========================
# UTILS
# =========================
def _norm(text: str) -> str:
    t = text.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _tokenize(text: str) -> List[str]:
    t = _norm(text)
    return re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ0-9]+", t)


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    u = len(sa | sb)
    return (len(sa & sb) / u) if u else 0.0


def _is_pdf_bytes(b: bytes) -> bool:
    return bool(b) and b[:5] == b"%PDF-"


def _safe_urljoin(base: str, href: str) -> str:
    return urllib.parse.urljoin(base, href)


def _same_site_seed(seed_host: str, url: str) -> bool:
    h = urllib.parse.urlparse(url).netloc.lower()
    # autorise domaine principal + sous-domaines
    return h == seed_host or h.endswith("." + seed_host)


def _link_score(href: str, text: str) -> int:
    blob = _norm((href or "") + " " + (text or ""))
    score = 0
    for tok in FOLLOW_TOKENS:
        if tok in blob:
            score += 2
    for tok in AVOID_TOKENS:
        if tok in blob:
            score -= 3
    # bonus si ressemble à page annales Terminale
    if "annales-terminale" in blob or "terminale" in blob and "annales" in blob:
        score += 6
    return score


def _looks_like_non_subject_by_name(filename: str) -> bool:
    f = filename.lower()
    return (
        f.startswith("pv") or "bulletin" in f or "sommaire" in f or "edito" in f or "édito" in f
        or "regionale" in f or "régionale" in f
    )


def _looks_like_non_subject_by_text(text: str) -> bool:
    t = _norm(text)
    bad = ["sommaire", "édito", "edito", "vie de la régionale", "vie de la regionale", "bulletin", "procès-verbal", "proces-verbal"]
    return any(x in t for x in bad)


def _suites_score(text: str) -> int:
    t = _norm(text)
    score = 0
    for k in SUITES_KEYWORDS:
        if k in t:
            score += 2
    # bonus u_n / recurrence
    if "u_n" in t or "u_{n" in t or "u(n+1)" in t:
        score += 6
    if "récurrence" in t or "recurrence" in t:
        score += 4
    if "limite" in t or "convergence" in t:
        score += 3
    return score


def _math_score(text: str) -> int:
    s = 0
    if MATH_VERBS_RE.search(text):
        s += 4
    if MATH_SYMBOL_RE.search(text):
        s += 6
    # token density: chiffres + lettres
    toks = _tokenize(text)
    if toks:
        digits = sum(1 for x in toks if x.isdigit())
        if digits >= 2:
            s += 2
    return s


def _is_terminal_maths_context(text: str) -> bool:
    t = _norm(text)
    # très permissif (APMEP n'affiche pas toujours "Terminale" dans le PDF)
    return ("math" in t or "maths" in t or "spécialité" in t or "specialite" in t or "bac" in t or "terminale" in t)


# =========================
# HTTP
# =========================
def _http_get(url: str) -> Optional[requests.Response]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT)
        if r.status_code >= 400:
            return None
        return r
    except Exception:
        return None


def _http_get_stream(url: str) -> Optional[requests.Response]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT, stream=True)
        if r.status_code >= 400:
            return None
        return r
    except Exception:
        return None


# =========================
# CRAWLER (HTML -> PDF links)
# =========================
def crawl_pdf_links(seeds: List[str], target_min: int) -> List[str]:
    """
    Crawl BFS depuis seeds, collecte des liens PDF + suit des pages internes "pertinentes".
    """
    pdfs: List[str] = []
    seen_pdfs: Set[str] = set()

    # seeds hosts (on autorise apmep.fr + sous-domaines si seed est apmep.fr)
    seed_hosts = set()
    for s in seeds:
        try:
            seed_hosts.add(urllib.parse.urlparse(s).netloc.lower() or "apmep.fr")
        except Exception:
            seed_hosts.add("apmep.fr")
    # si apmep.fr présent, autoriser tous sous-domaines apmep.fr
    allow_root = "apmep.fr" in seed_hosts

    q = deque()
    seen_pages: Set[str] = set()

    for s in seeds:
        if not s.startswith("http"):
            s = "https://" + s
        q.append(s)

    pages_visited = 0

    while q and pages_visited < MAX_HTML_PAGES and len(pdfs) < target_min:
        url = q.popleft()
        u0 = url.split("#")[0]
        if u0 in seen_pages:
            continue
        seen_pages.add(u0)

        r = _http_get(u0)
        if not r:
            continue

        ctype = (r.headers.get("Content-Type") or "").lower()
        if "text/html" not in ctype and "<html" not in (r.text[:500].lower() if r.text else ""):
            continue

        pages_visited += 1

        soup = BeautifulSoup(r.text, "html.parser")
        anchors = soup.find_all("a", href=True)

        for a in anchors:
            href = (a.get("href") or "").strip()
            txt = (a.get_text(" ", strip=True) or "").strip()
            if not href:
                continue

            full = _safe_urljoin(u0, href)
            full = full.split("#")[0]

            # PDF direct
            if ".pdf" in full.lower():
                if full not in seen_pdfs:
                    seen_pdfs.add(full)
                    pdfs.append(full)
                continue

            # Pages à suivre (même site + score)
            parsed = urllib.parse.urlparse(full)
            if not parsed.scheme.startswith("http"):
                continue

            host = parsed.netloc.lower()
            if allow_root:
                if not (host == "apmep.fr" or host.endswith(".apmep.fr")):
                    continue
            else:
                # autorise uniquement mêmes hosts que seeds
                if host not in seed_hosts and not any(host.endswith("." + h) for h in seed_hosts):
                    continue

            # filtrage sur score
            score = _link_score(full, txt)
            if score >= 3:
                if full not in seen_pages:
                    q.append(full)

    # dédoublonnage ordre conservé
    out = []
    seen = set()
    for p in pdfs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# =========================
# PDF DOWNLOAD + TEXT
# =========================
def download_pdf(url: str) -> Optional[bytes]:
    r = _http_get_stream(url)
    if not r:
        return None

    cl = r.headers.get("Content-Length")
    if cl:
        try:
            mb = int(cl) / (1024 * 1024)
            if mb > MAX_PDF_MB:
                return None
        except Exception:
            pass

    # read with cap
    try:
        data = r.content
    except Exception:
        return None

    if not data or len(data) > MAX_PDF_MB * 1024 * 1024:
        return None
    if not _is_pdf_bytes(data):
        return None
    return data


def extract_text_from_pdf_bytes(pdf_bytes: bytes, max_pages: int = MAX_PDF_PAGES_TEXT) -> str:
    parts: List[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        n = min(len(pdf.pages), max_pages)
        for i in range(n):
            page = pdf.pages[i]
            t = page.extract_text() or ""
            t = t.strip()
            if t:
                parts.append(t)
    return "\n".join(parts)


# =========================
# QI EXTRACTION (stronger heuristics)
# =========================
def extract_qi_from_text(text: str) -> List[str]:
    raw = text.replace("\r", "\n")
    raw = re.sub(r"\n{3,}", "\n\n", raw)

    # supprimer entêtes/pieds répétés grossiers
    raw = re.sub(r"Page\s+\d+\s*/\s*\d+", " ", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s+", " ", raw)

    # segmentation par marqueurs fréquents
    # On split sur "Exercice", "Question", numérotation, etc.
    split_re = re.compile(r"(?:\bExercice\s*\d+\b|\bQuestion\s*\d+\b|\bPartie\s*[A-Za-z0-9]+\b|(?:(?:^|\s)\d+\s*[\).\:-]\s+))", re.IGNORECASE)
    chunks = split_re.split(raw)

    candidates: List[str] = []
    for c in chunks:
        c = c.strip()
        if len(c) < MIN_QI_CHARS:
            continue

        # garder uniquement ce qui ressemble à une question/consigne math
        ms = _math_score(c)
        if ms < 4:
            continue

        if not MATH_VERBS_RE.search(c) and "?" not in c:
            # sans verbe, on est souvent sur du texte narratif
            continue

        # tronquer si trop long
        if len(c) > 520:
            c = c[:520].rsplit(" ", 1)[0] + "…"

        candidates.append(c)

    # fallback si split vide: blocs par ponctuation
    if not candidates:
        blocks = re.split(r"[.;]\s+", raw)
        for b in blocks:
            b = b.strip()
            if len(b) < MIN_QI_CHARS:
                continue
            if _math_score(b) >= 6 and (MATH_VERBS_RE.search(b) or "?" in b):
                if len(b) > 520:
                    b = b[:520].rsplit(" ", 1)[0] + "…"
                candidates.append(b)

    # dédoublonnage
    out: List[str] = []
    seen = set()
    for x in candidates:
        k = _norm(x)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


# =========================
# ARI + FRT (simple deterministic rules)
# =========================
def build_ari(qi_texts: List[str]) -> List[str]:
    blob = _norm(" ".join(qi_texts[:6]))
    steps: List[str] = []

    if "récurrence" in blob or "recurrence" in blob:
        steps.extend([
            "Identifier la propriété P(n) à démontrer.",
            "Initialisation : vérifier P(0) ou P(1) selon l’énoncé.",
            "Hérédité : supposer P(n) vraie et démontrer P(n+1).",
            "Conclure : P(n) vraie pour tout n (récurrence).",
        ])
    elif "monotone" in blob or "croissante" in blob or "décroissante" in blob or "decroissante" in blob or "bornée" in blob or "bornee" in blob:
        steps.extend([
            "Exprimer u_{n+1} − u_n ou u_{n+1}/u_n selon la forme.",
            "Établir le sens de variation (monotonie).",
            "Montrer que la suite est bornée.",
            "Conclure : la suite converge, puis identifier la limite (si demandé).",
        ])
    elif "arithm" in blob:
        steps.extend([
            "Identifier que la suite est arithmétique (u_{n+1} = u_n + r).",
            "Déterminer la raison r et le premier terme.",
            "Écrire le terme général u_n.",
            "Répondre (calcul de terme, somme, etc.).",
        ])
    elif "géométr" in blob or "geomet" in blob:
        steps.extend([
            "Identifier que la suite est géométrique (u_{n+1} = q·u_n).",
            "Déterminer la raison q et le premier terme.",
            "Écrire le terme général u_n.",
            "Répondre (calcul de terme, somme, etc.).",
        ])
    else:
        steps.extend([
            "Identifier la définition de la suite (explicite ou par récurrence).",
            "Mettre en place l’outil adapté (variation, encadrement, récurrence, limite).",
            "Effectuer les calculs nécessaires et justifier.",
            "Conclure clairement avec la réponse demandée.",
        ])

    # format UI monospace-friendly
    return [f"{i+1}. {s}" for i, s in enumerate(steps)]


def build_frt(qi_texts: List[str]) -> Dict[str, str]:
    blob = _norm(" ".join(qi_texts[:6]))

    usage = "Exercices sur suites (définition, terme général, variation, convergence)."
    method = "1) Identifier la nature (explicite/récurrence)  2) Choisir l’outil (variation/borne/récurrence)  3) Justifier chaque étape  4) Conclure."
    trap = "Confondre u_{n+1}−u_n et u_{n+1}/u_n, oublier l’indice de départ, conclure une limite sans bornitude/monotonie, erreurs de calcul de raison."
    conc = "Écrire la conclusion sous forme de phrase mathématique (ex : « la suite (u_n) est croissante et bornée, donc convergente, et sa limite vaut L » si établi)."

    if "récurrence" in blob or "recurrence" in blob:
        usage = "Questions de démonstration par récurrence sur suites."
        method = "Définir P(n) → Initialisation → Hérédité (P(n) ⇒ P(n+1)) → Conclusion (∀n)."
        trap = "Oublier l’initialisation, mal formuler P(n), faire l’hérédité sans hypothèse de récurrence."
    if "limite" in blob or "convergence" in blob:
        usage = "Étude de convergence (monotonie + bornes ou point fixe)."
        method = "Montrer monotone + bornée ⇒ convergence, puis identifier la limite (équation de point fixe ou encadrement)."
        trap = "Chercher la limite sans avoir prouvé la convergence, résoudre une équation de limite non justifiée."

    return {"usage": usage, "method": method, "trap": trap, "conc": conc}


# =========================
# CLUSTERING -> QC (simple semantic)
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


def cluster_qi_to_qc(qis: List[QiItem], sim_threshold: float = 0.30) -> List[Dict]:
    clusters: List[Dict] = []  # each: {id, rep_tokens, qis}
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
            # union léger
            clusters[best_i]["rep_tokens"] = list(set(clusters[best_i]["rep_tokens"]) | set(toks))
        else:
            clusters.append({"id": f"QC-{qc_idx:03d}", "rep_tokens": toks, "qis": [qi]})
            qc_idx += 1

    qc_out: List[Dict] = []
    n_tot = len(qis)

    for c in clusters:
        qi_texts = [q.text for q in c["qis"]]
        title = qi_texts[0]
        qc_title = title if len(title) <= 92 else (title[:92].rsplit(" ", 1)[0] + "…")

        # triggers: tokens fréquents + suites keywords présents
        all_toks: List[str] = []
        for t in qi_texts:
            all_toks.extend(_tokenize(t))

        freq = Counter(tok for tok in all_toks if tok not in STOP and len(tok) >= 4)
        top = [k for k, _ in freq.most_common(10)]

        # garder des déclencheurs "suites" si présents
        blob = _norm(" ".join(qi_texts))
        suites_hits = []
        for k in ["suite", "récurrence", "recurrence", "limite", "convergence", "monotone", "arithmétique", "géométrique", "raison"]:
            if k in blob:
                suites_hits.append(k)
        triggers = list(dict.fromkeys(suites_hits + top))[:8]

        n_q = len(qi_texts)
        psi = round(min(1.0, n_q / 30.0), 2)
        score = int(round(50 + 12 * math.log(1 + n_q), 0))

        ari = build_ari(qi_texts)
        frt = build_frt(qi_texts)

        qc_out.append({
            "chapter": "SUITES NUMÉRIQUES",
            "qc_id": c["id"],
            "qc_title": qc_title,
            "score": score,
            "n_q": n_q,
            "psi": psi,
            "n_tot": n_tot,
            "t_rec": 0.0,  # rempli au niveau run
            "triggers": triggers,
            "ari": ari,
            "frt": frt,
            "qi_by_file": _group_qi_by_file(c["qis"]),
        })

    # Tri: QC les plus denses d’abord
    qc_out.sort(key=lambda x: (x["n_q"], x["score"]), reverse=True)
    return qc_out


# =========================
# SATURATION
# =========================
def compute_saturation(history_counts: List[int]) -> List[Dict]:
    return [{"Nombre de sujets injectés": i + 1, "Nombre de QC découvertes": v} for i, v in enumerate(history_counts)]


# =========================
# SUBJECT FILTER (reject PV/bulletin etc.)
# =========================
def is_probable_subject_pdf(pdf_url: str, filename: str, text: str) -> bool:
    # nom de fichier
    if _looks_like_non_subject_by_name(filename):
        return False
    # contenu texte
    if _looks_like_non_subject_by_text(text):
        return False
    # contexte math/terminale (très permissif)
    if not _is_terminal_maths_context(text):
        return False
    return True


# =========================
# MAIN API FOR UI (unchanged contract)
# =========================
def run_granulo_test(urls: List[str], volume: int) -> Dict:
    """
    Contract UI (NE PAS CASSER):
    return {
      "sujets": [ {Fichier,Nature,Année,Source}, ... ],
      "qc": [ {chapter,qc_id,qc_title,score,n_q,psi,n_tot,t_rec,triggers,ari,frt,qi_by_file}, ... ],
      "saturation": [ {Nombre de sujets injectés, Nombre de QC découvertes}, ... ],
      "audit": {...}
    }
    """
    start = time.time()

    # 1) Crawl + collecte PDF links (beaucoup plus que volume)
    target_min = max(volume * MAX_PDFS_COLLECT_MULT, volume)
    pdf_links_all = crawl_pdf_links(urls, target_min=target_min)

    # 2) Limiter pour traitement (évite explosion temps)
    #    On essaye jusqu’à obtenir "volume" sujets valides (pas juste "volume" liens).
    max_to_try = max(volume * 6, volume)
    pdf_links = pdf_links_all[:max_to_try]

    sujets_rows: List[Dict] = []
    all_qis: List[QiItem] = []
    qc_history: List[int] = []

    rejected_pdf_non_subject = 0
    rejected_pdf_no_text = 0
    rejected_pdf_no_qi = 0
    downloaded_ok = 0

    for idx, pdf_url in enumerate(pdf_links, start=1):
        if len(sujets_rows) >= volume:
            break

        pdf_bytes = download_pdf(pdf_url)
        if not pdf_bytes:
            continue

        filename = pdf_url.split("/")[-1].split("?")[0]
        host = re.sub(r"^https?://", "", pdf_url).split("/")[0]

        text = extract_text_from_pdf_bytes(pdf_bytes)
        if not text or len(text) < MIN_TEXT_CHARS_PDF:
            rejected_pdf_no_text += 1
            continue

        # filtre "sujet" vs PV/bulletin
        if not is_probable_subject_pdf(pdf_url, filename, text):
            rejected_pdf_non_subject += 1
            continue

        # filtre contexte Terminale/Maths (soft)
        if not _is_terminal_maths_context(text):
            rejected_pdf_non_subject += 1
            continue

        # 3) Qi extraction
        qi_candidates = extract_qi_from_text(text)

        # 4) Filtrer suites numériques (score suites + score math)
        qi_filtered: List[str] = []
        for q in qi_candidates:
            ss = _suites_score(q)
            ms = _math_score(q)
            if ss >= 4 and ms >= 4:
                qi_filtered.append(q)

        if not qi_filtered:
            rejected_pdf_no_qi += 1
            continue

        downloaded_ok += 1

        sujets_rows.append({
            "Fichier": filename,
            "Nature": "SUJET",
            "Année": _extract_year_from_text_or_name(text, filename),
            "Source": host,
        })

        subject_id = f"S{len(sujets_rows):04d}"
        for q in qi_filtered:
            all_qis.append(QiItem(subject_id=subject_id, subject_file=filename, text=q))

        # saturation (QC cumulées)
        qc_current = cluster_qi_to_qc(all_qis)
        qc_history.append(len(qc_current))

    # QC final
    qc_list = cluster_qi_to_qc(all_qis)
    sat_points = compute_saturation(qc_history)

    elapsed = round(time.time() - start, 2)

    # t_rec: temps total / nb sujets
    t_rec = 0.0
    if sujets_rows:
        t_rec = round(elapsed / max(1, len(sujets_rows)), 2)
    for qc in qc_list:
        qc["t_rec"] = t_rec
        qc["n_tot"] = len(all_qis)

    # booléen central (simplifié): y a-t-il une Qi non mappée vers une QC ?
    # Ici, clustering met toujours toute Qi dans un cluster -> False
    binary_unmapped_exists = False

    return {
        "sujets": sujets_rows,
        "qc": qc_list,
        "saturation": sat_points,
        "audit": {
            "n_urls": len(urls),
            "n_pdf_links_found": len(pdf_links_all),
            "n_pdf_links_tested": len(pdf_links),
            "n_subjects_ok": len(sujets_rows),
            "n_qi": len(all_qis),
            "n_qc": len(qc_list),
            "downloaded_ok": downloaded_ok,
            "rejected_pdf_non_subject": rejected_pdf_non_subject,
            "rejected_pdf_no_text": rejected_pdf_no_text,
            "rejected_pdf_no_qi": rejected_pdf_no_qi,
            "elapsed_s": elapsed,
            "binary_qi_to_qc_unmapped_exists": binary_unmapped_exists,
        }
    }


def _extract_year_from_text_or_name(text: str, filename: str) -> Optional[int]:
    # essaye d’extraire année (1990-2035)
    m = re.search(r"\b(19\d{2}|20\d{2}|203[0-5])\b", filename)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    m = re.search(r"\b(19\d{2}|20\d{2}|203[0-5])\b", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return None
