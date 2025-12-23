from __future__ import annotations

import io
import math
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urldefrag
from html.parser import HTMLParser

import requests

# =========================
# CONFIG
# =========================
UA = "SMAXIA-GTE/1.3 (pdf-crawler+math-filter)"
REQ_TIMEOUT = 25
MAX_PDF_MB = 25
MIN_QI_CHARS = 18

# Crawl (borné) : suffisamment profond pour atteindre "Année 2025/2024/..."
CRAWL_MAX_PAGES = 80
CRAWL_MAX_DEPTH = 3
MAX_PDF_LINKS_GLOBAL = 3000
MAX_HTML_BYTES = 2_000_000

# Filtrage Suites (Terminale Maths)
SUITES_KEYWORDS: Set[str] = {
    "suite", "suites", "arithmétique", "arithmetique", "géométrique", "geometrique",
    "raison", "u_n", "v_n", "récurrence", "recurrence", "terme", "général", "general",
    "monotone", "bornée", "bornee", "majorée", "majoree", "minorée", "minoree",
    "limite", "convergence"
}

TRIGGER_PHRASES = [
    "montrer que", "démontrer", "demontrer", "prouver",
    "calculer", "déterminer", "determiner", "étudier", "etudier",
    "justifier", "en déduire", "en deduire",
]

STOP_TOKENS = {
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "à", "a", "en", "pour",
    "dans", "sur", "avec", "par", "au", "aux", "d", "l", "si", "que", "qui", "on",
}

# Exclusions par nom (fortement non sujet)
UNWANTED_PDF_NAME_RE = re.compile(
    r"(?:^|/)(pv\d+|compte[- ]rendu|convocation|bulletin|newsletter|sommaire|édito|edito|lettre|adh[eé]sion)",
    re.IGNORECASE
)

# Indices texte "non-sujet" (associatif/administratif)
NON_SUBJECT_TEXT_RE = re.compile(
    r"(proc[eè]s[- ]verbal|compte[- ]rendu|convocation|adh[eé]sion|association|r[eé]gionale|bureau|assembl[eé]e|tr[eé]sorier)",
    re.IGNORECASE
)

# Indices texte "math sujet"
MATH_SIGNAL_RE = re.compile(
    r"(exercice|question|sujet|corrig[eé]|sp[eé]cialit[eé]|math|u_n|v_n|r[eé]currence|suite|limite|convergence|monotone|major[eé]e|minor[eé]e|raison|terme g[eé]n[eé]ral)",
    re.IGNORECASE
)

# Détection de liens "PDF potentiels" même sans .pdf (SPIP etc.)
PDF_CANDIDATE_RE = re.compile(
    r"(\.pdf\b|telecharger|t[eé]l[eé]charger|download|file=|fichier=|doc=|document=|pdf=)",
    re.IGNORECASE
)

DEFAULT_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.7",
    "Connection": "keep-alive",
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
    return re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ0-9_]+", t)

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
# HTML PARSER
# =========================
class _HrefCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        for k, v in attrs:
            if k.lower() == "href" and v:
                self.hrefs.append(v.strip())

def _safe_get_html(url: str) -> Optional[Tuple[str, str]]:
    """
    Retourne (final_url, html) ou None
    """
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=REQ_TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        final_url = r.url
        txt = r.text
        b = txt.encode("utf-8", errors="ignore")
        if len(b) > MAX_HTML_BYTES:
            txt = b[:MAX_HTML_BYTES].decode("utf-8", errors="ignore")
        return final_url, txt
    except Exception:
        return None

def _same_host(a: str, b: str) -> bool:
    try:
        return urlparse(a).netloc == urlparse(b).netloc
    except Exception:
        return False

def _url_priority(u: str) -> int:
    """
    Priorise les pages pertinentes : Annales/Terminale/Bac/Sujets/Spécialité/Maths
    """
    uu = u.lower()
    score = 0
    for kw in ["annale", "annales", "terminale", "baccalaureat", "bac", "sujet", "sujets", "corrige", "corrigé", "specialite", "spécialité", "math"]:
        if kw in uu:
            score += 3
    # pages d'année
    if re.search(r"\b(19\d{2}|20\d{2})\b", uu):
        score += 2
    return -score  # plus petit = plus prioritaire (tri croissant)

def _extract_links(base_url: str, html: str) -> Tuple[List[str], List[str]]:
    """
    Retourne (pages_internes, pdf_candidates)
    """
    parser = _HrefCollector()
    parser.feed(html)
    hrefs = parser.hrefs

    pages: List[str] = []
    pdfs: List[str] = []

    for href in hrefs:
        if not href:
            continue
        # remove fragment (#...)
        href, _frag = urldefrag(href)
        if not href:
            continue
        full = urljoin(base_url, href)

        # candidat pdf ?
        if PDF_CANDIDATE_RE.search(full):
            pdfs.append(full)
            continue

        # page interne
        pages.append(full)

    # filtrer pages internes (même host)
    pages2 = [p for p in pages if _same_host(base_url, p)]
    # dédoublonnage ordre
    def _dedup(xs: List[str]) -> List[str]:
        out, seen = [], set()
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return _dedup(pages2), _dedup(pdfs)

# =========================
# PDF VALIDATION + DOWNLOAD
# =========================
def _looks_like_pdf(data: bytes) -> bool:
    return data[:4] == b"%PDF" or b"%PDF" in data[:1024]

def download_pdf(url: str) -> Optional[bytes]:
    # Exclure par nom
    if UNWANTED_PDF_NAME_RE.search(url):
        return None

    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=REQ_TIMEOUT, stream=True, allow_redirects=True)
        r.raise_for_status()

        cl = r.headers.get("Content-Length")
        if cl:
            mb = int(cl) / (1024 * 1024)
            if mb > MAX_PDF_MB:
                return None

        data = r.content
        if not data:
            return None
        if len(data) > MAX_PDF_MB * 1024 * 1024:
            return None

        # Certains liens "telecharger" renvoient du HTML (erreur ou page)
        if not _looks_like_pdf(data):
            return None

        # réalignement sur %PDF si nécessaire
        if data[:4] != b"%PDF":
            pos = data.find(b"%PDF")
            if pos >= 0:
                data = data[pos:]
            else:
                return None

        return data
    except Exception:
        return None

# =========================
# TEXT EXTRACTION (PDF réel)
# =========================
def extract_text_from_pdf_bytes(pdf_bytes: bytes, max_pages: int = 25) -> str:
    # pdfplumber
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

    # PyPDF2 fallback
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

def _is_probably_math_subject(text: str) -> bool:
    """
    Nouveau filtre : score, pas rejet binaire.
    - Un vrai sujet peut contenir des mentions APMEP/régionale dans l’entête : on ne rejette plus pour ça.
    """
    if not text or len(text.strip()) < 200:
        return False

    math_hits = len(MATH_SIGNAL_RE.findall(text))
    non_hits = len(NON_SUBJECT_TEXT_RE.findall(text))

    # règle simple : assez de signaux maths
    if math_hits >= 10:
        return True

    # règle équilibrée : maths dominent
    if math_hits >= 6 and (math_hits - non_hits) >= 4:
        return True

    return False

# =========================
# Qi extraction
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

        if re.search(r"\b(exercice|question)\b", b2, re.IGNORECASE):
            candidates.append(b2)
            continue

        if re.search(r"\b(calculer|déterminer|determiner|montrer|démontrer|demontrer|justifier|étudier|etudier|prouver)\b", b2, re.IGNORECASE):
            candidates.append(b2)
            continue

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
# QC clustering + FRT/ARI
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
    txt = " ".join(qi_texts).lower()

    is_arith = "arithm" in txt
    is_geo = ("géométr" in txt) or ("geomet" in txt)
    is_recur = ("récurr" in txt) or ("recurr" in txt) or ("u_{n+1}" in txt) or ("un+1" in txt)
    wants_limit = ("limite" in txt) or ("converg" in txt)

    if is_arith:
        subtype = "suite arithmétique"
    elif is_geo:
        subtype = "suite géométrique"
    elif is_recur:
        subtype = "suite définie par récurrence"
    else:
        subtype = "suite (cadre général)"

    trig_txt = ", ".join(triggers) if triggers else "analyser / déterminer"
    usage = f"À utiliser lorsque l’énoncé demande : {trig_txt}, sur le chapitre Suites numériques ({subtype})."

    steps = ["Identifier la définition de la suite (explicite / récurrence) et les données initiales."]
    if is_arith:
        steps.append("Si u(n+1)=u(n)+r : déterminer r puis utiliser u(n)=u(0)+nr.")
    if is_geo:
        steps.append("Si u(n+1)=q·u(n) : déterminer q puis utiliser u(n)=u(0)·q^n.")
    if is_recur and (not is_arith) and (not is_geo):
        steps.append("Dérouler 2–3 termes, puis structurer la preuve (récurrence / invariants / bornes).")
    if wants_limit:
        steps.append("Étudier monotonie + bornes, conclure convergence, puis calculer la limite (point fixe si récurrence).")
    steps.append("Conclure avec la réponse demandée (expression / valeur / justification).")

    method = "<br>".join([f"{i+1}) {s}" for i, s in enumerate(steps)])

    traps = [
        "Oublier la condition initiale (u0/u1) ou confondre les indices.",
        "Conclure une limite sans justification (monotonie + bornes / théorème adapté).",
    ]
    if is_geo:
        traps.append("Confondre q et u0 ; oublier q^n ; ne pas traiter les cas particuliers (q=1, q=0, q<0).")
    if is_arith:
        traps.append("Confondre la raison r avec un terme de la suite.")
    if is_recur:
        traps.append("Manipuler la récurrence sans vérifier domaine/bornes (stabilité).")

    trap = "<br>".join([f"• {t}" for t in traps])
    conc = "Conclusion : donner u(n) / raison / limite, avec justification conforme au niveau Terminale."

    return {"usage": usage, "method": method, "trap": trap, "conc": conc}

def cluster_qi_to_qc(qis: List[QiItem], sim_threshold: float = 0.28) -> List[Dict]:
    clusters: List[Dict] = []
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
            clusters.append({"id": f"QC-{qc_idx:03d}", "rep_tokens": toks, "qis": [qi]})
            qc_idx += 1

    qc_out = []
    for c in clusters:
        qi_texts = [q.text for q in c["qis"]]
        title = qi_texts[0]
        qc_title = title if len(title) <= 90 else title[:90].rsplit(" ", 1)[0] + "…"

        triggers = []
        for t in qi_texts:
            triggers.extend(_extract_trigger_phrases(t))
        if not triggers:
            triggers = _top_keywords(qi_texts, k=6)

        trig2, seen = [], set()
        for x in triggers:
            if x not in seen:
                seen.add(x)
                trig2.append(x)

        n_q = len(qi_texts)
        psi = round(min(1.0, n_q / 25.0), 2)
        score = int(round(50 + 10 * math.log(1 + n_q), 0))

        ari = []
        for i, qtxt in enumerate(qi_texts[:10], 1):
            short = qtxt if len(qtxt) <= 120 else qtxt[:120].rsplit(" ", 1)[0] + "…"
            ari.append(f"{i}. {short}")
        if len(qi_texts) > 10:
            ari.append(f"… +{len(qi_texts)-10} Qi suivantes")

        frt = _build_frt(qi_texts, trig2[:6])

        qc_out.append({
            "chapter": "SUITES NUMÉRIQUES",
            "qc_id": c["id"],
            "qc_title": qc_title,
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
# PDF DISCOVERY (BFS)
# =========================
def discover_pdf_links_bfs(start_urls: List[str], soft_limit: int) -> List[str]:
    """
    Crawl BFS borné :
    - collecte PDF candidates sur toutes pages internes
    - profondeur jusqu’à CRAWL_MAX_DEPTH
    - puis dédoublonnage + filtre par nom
    """
    start_urls = [u.strip() for u in start_urls if u and u.strip()]
    if not start_urls:
        start_urls = ["https://apmep.fr"]

    # si racine APMEP : on ajoute la page Annales Terminale (vue sur votre capture)
    seeds = []
    for u in start_urls:
        seeds.append(u)
        if "apmep.fr" in urlparse(u).netloc and urlparse(u).path.strip("/") == "":
            seeds.append("https://apmep.fr/Annales-Terminale-Generale")
    # dédoublonnage
    q = []
    seen_seed = set()
    for s in seeds:
        if s not in seen_seed:
            seen_seed.add(s)
            q.append((s, 0))

    visited_pages = set()
    pdf_links: List[str] = []

    while q and len(visited_pages) < CRAWL_MAX_PAGES and len(pdf_links) < MAX_PDF_LINKS_GLOBAL:
        q.sort(key=lambda x: _url_priority(x[0]))
        url, depth = q.pop(0)

        if url in visited_pages:
            continue
        visited_pages.add(url)

        got = _safe_get_html(url)
        if not got:
            continue
        final_url, html = got

        pages, pdfs = _extract_links(final_url, html)
        pdf_links.extend(pdfs)

        if depth < CRAWL_MAX_DEPTH:
            for p in pages:
                if p not in visited_pages:
                    q.append((p, depth + 1))

        # si on a déjà beaucoup de PDFs, on peut s’arrêter tôt
        if len(pdf_links) >= max(soft_limit * 15, 200):
            break

    # nettoyage + filtre
    cleaned = []
    seen = set()
    for link in pdf_links:
        link, _ = urldefrag(link)
        if not link:
            continue
        if UNWANTED_PDF_NAME_RE.search(link):
            continue
        if link not in seen:
            seen.add(link)
            cleaned.append(link)

    return cleaned[:MAX_PDF_LINKS_GLOBAL]

# =========================
# SATURATION
# =========================
def compute_saturation(history_counts: List[int]) -> List[Dict]:
    return [{"Nombre de sujets injectés": i + 1, "Nombre de QC découvertes": v} for i, v in enumerate(history_counts)]

# =========================
# API PRINCIPALE POUR UI
# =========================
def run_granulo_test(urls: List[str], volume: int) -> Dict:
    """
    Retour dict UI:
    - sujets: rows
    - qc: QC list
    - saturation: points
    - audit: metrics
    """
    start = time.time()
    volume = int(volume) if volume else 15
    urls = [u.strip() for u in (urls or []) if u and u.strip()]

    # 1) discovery PDF links
    pdf_candidates = discover_pdf_links_bfs(urls, soft_limit=volume)

    sujets_rows = []
    all_qis: List[QiItem] = []
    qc_history = []

    rejected_pdf_download = 0
    rejected_pdf_no_text = 0
    rejected_pdf_non_subject = 0
    rejected_pdf_no_qi = 0

    # 2) iterate PDFs until "volume sujets OK"
    for candidate in pdf_candidates:
        pdf_bytes = download_pdf(candidate)
        if not pdf_bytes:
            rejected_pdf_download += 1
            continue

        text = extract_text_from_pdf_bytes(pdf_bytes)
        if not text.strip():
            rejected_pdf_no_text += 1
            continue

        if not _is_probably_math_subject(text):
            rejected_pdf_non_subject += 1
            continue

        qi_texts = extract_qi_from_text(text)
        qi_texts = [q for q in qi_texts if _contains_suites_signal(q)]
        if not qi_texts:
            rejected_pdf_no_qi += 1
            continue

        subject_file = candidate.split("/")[-1].split("?")[0]
        source_host = urlparse(candidate).netloc

        m_year = re.search(r"(19\d{2}|20\d{2})", subject_file)
        year = int(m_year.group(1)) if m_year else None

        sujets_rows.append({
            "Fichier": subject_file,
            "Nature": "SUJET/CORRIGÉ (probable)",
            "Année": year,
            "Source": source_host,
        })

        subject_id = f"S{len(sujets_rows):04d}"
        for q in qi_texts:
            all_qis.append(QiItem(subject_id=subject_id, subject_file=subject_file, text=q))

        qc_current = cluster_qi_to_qc(all_qis)
        qc_history.append(len(qc_current))

        if len(sujets_rows) >= volume:
            break

    qc_list = cluster_qi_to_qc(all_qis)
    sat_points = compute_saturation(qc_history)
    elapsed = round(time.time() - start, 2)

    return {
        "sujets": sujets_rows,
        "qc": qc_list,
        "saturation": sat_points,
        "audit": {
            "n_urls": len(urls),
            "n_pdf_links": len(pdf_candidates),
            "n_subjects_ok": len(sujets_rows),
            "n_qi": len(all_qis),
            "n_qc": len(qc_list),
            "rejected_pdf_download": rejected_pdf_download,
            "rejected_pdf_no_text": rejected_pdf_no_text,
            "rejected_pdf_non_subject": rejected_pdf_non_subject,
            "rejected_pdf_no_qi": rejected_pdf_no_qi,
            "elapsed_s": elapsed,
            "binary_qi_to_qc_unmapped_exists": False,  # par construction du clustering
        }
    }
