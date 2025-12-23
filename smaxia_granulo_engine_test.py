from __future__ import annotations

import io
import math
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from html.parser import HTMLParser


# =========================
# CONFIG
# =========================
UA = "SMAXIA-GTE/1.4 (apmep-bfs+pdf+scoring)"
REQ_TIMEOUT = 25
MAX_PDF_MB = 25
MIN_QI_CHARS = 18

CRAWL_MAX_PAGES = 120
CRAWL_MAX_DEPTH = 3
MAX_PDF_LINKS_GLOBAL = 4000
MAX_HTML_BYTES = 2_000_000

# Terminale Maths – Suites numériques
SUITES_KEYWORDS: Set[str] = {
    "suite", "suites", "arithmétique", "arithmetique", "géométrique", "geometrique",
    "raison", "u_n", "v_n", "u(n)", "v(n)", "u0", "u1",
    "récurrence", "recurrence", "terme", "général", "general",
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

# Exclusions fortes par nom
UNWANTED_PDF_NAME_RE = re.compile(
    r"(?:^|/)(pv\d+|compte[- ]rendu|convocation|adh[eé]sion|newsletter|sommaire|édito|edito|lettre)",
    re.IGNORECASE
)

# Indices "administratif / associatif"
NON_SUBJECT_TEXT_RE = re.compile(
    r"(proc[eè]s[- ]verbal|compte[- ]rendu|convocation|adh[eé]sion|association|assembl[eé]e|bureau|tr[eé]sorier|secr[eé]taire)",
    re.IGNORECASE
)

# Indices "math sujet/corrigé"
MATH_SIGNAL_RE = re.compile(
    r"(exercice|question|sujet|corrig[eé]|sp[eé]cialit[eé]|math|"
    r"fonction|d[eé]riv[eé]e|primitive|int[eé]grale|"
    r"probabilit|loi|variance|esp[eé]rance|"
    r"g[eé]om[eé]tr|vecteur|rep[eè]re|"
    r"u_n|v_n|r[eé]currence|suite|limite|convergence|monotone|raison|terme g[eé]n[eé]ral)",
    re.IGNORECASE
)

# Détection “PDF candidate” même sans .pdf
PDF_CANDIDATE_RE = re.compile(
    r"(\.pdf\b|/IMG/pdf/|telecharger|t[eé]l[eé]charger|download|file=|fichier=|doc=|document=|pdf=)",
    re.IGNORECASE
)

# Extraction brute de liens PDF présents dans le HTML (pas seulement href)
PDF_URL_IN_HTML_RE = re.compile(
    r"(https?://[^\s\"\'<>]+?\.pdf(?:\?[^\s\"\'<>]+)?)",
    re.IGNORECASE
)

DEFAULT_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.7",
    "Connection": "keep-alive",
}


# =========================
# SESSION HTTP ROBUSTE
# =========================
def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


# =========================
# OUTILS TEXTE
# =========================
def _normalize(text: str) -> str:
    t = text.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _tokenize(text: str) -> List[str]:
    t = _normalize(text)
    return re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ0-9_()]+", t)

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0

def _contains_suites_signal(text: str) -> bool:
    toks = set(_tokenize(text))
    # signal “suite” fort
    if len(toks & SUITES_KEYWORDS) > 0:
        return True
    # signal “u_n / v_n” en texte OCR
    if re.search(r"\bu\s*[_]?\s*n\b|\bv\s*[_]?\s*n\b", text, re.IGNORECASE):
        return True
    return False

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
# HTML PARSER href
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


def _same_site(base: str, other: str) -> bool:
    """
    Autorise les sous-domaines APMEP (apmep.fr, www.apmep.fr, partage.apmep...).
    """
    try:
        b = urlparse(base).netloc.lower()
        o = urlparse(other).netloc.lower()
        if not b or not o:
            return False
        if b == o:
            return True
        if b.endswith("apmep.fr") and o.endswith("apmep.fr"):
            return True
        return False
    except Exception:
        return False


def _safe_get_html(session: requests.Session, url: str) -> Optional[Tuple[str, str]]:
    try:
        r = session.get(url, headers=DEFAULT_HEADERS, timeout=REQ_TIMEOUT, allow_redirects=True)
        if r.status_code >= 400:
            return None
        final_url = r.url
        txt = r.text or ""
        b = txt.encode("utf-8", errors="ignore")
        if len(b) > MAX_HTML_BYTES:
            txt = b[:MAX_HTML_BYTES].decode("utf-8", errors="ignore")
        return final_url, txt
    except Exception:
        return None


def _url_priority(u: str) -> int:
    uu = u.lower()
    score = 0
    for kw in [
        "annale", "annales", "terminale", "baccalaureat", "bac",
        "sujet", "sujets", "corrige", "corrigé",
        "specialite", "spécialité", "math"
    ]:
        if kw in uu:
            score += 4
    if re.search(r"\b(19\d{2}|20\d{2})\b", uu):
        score += 2
    return -score


def _extract_links(base_url: str, html: str) -> Tuple[List[str], List[str]]:
    """
    Retourne (pages, pdf_candidates)
    - pages : liens internes (même site)
    - pdf_candidates : liens candidats PDF (href + URLs PDF brutes dans le HTML)
    """
    parser = _HrefCollector()
    parser.feed(html)
    hrefs = parser.hrefs

    pages: List[str] = []
    pdfs: List[str] = []

    # 1) href
    for href in hrefs:
        href, _ = urldefrag(href)
        if not href:
            continue
        full = urljoin(base_url, href)

        if PDF_CANDIDATE_RE.search(full):
            pdfs.append(full)
        else:
            pages.append(full)

    # 2) PDF URLs brutes (souvent présentes dans des scripts / JSON)
    for m in PDF_URL_IN_HTML_RE.finditer(html):
        pdfs.append(m.group(1))

    # nettoyage
    def _dedup(xs: List[str]) -> List[str]:
        out, seen = [], set()
        for x in xs:
            x, _ = urldefrag(x)
            if not x:
                continue
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    pages = [p for p in pages if _same_site(base_url, p)]
    pages = _dedup(pages)
    pdfs = _dedup(pdfs)

    return pages, pdfs


# =========================
# PDF DOWNLOAD + VALIDATION
# =========================
def _looks_like_pdf(data: bytes) -> bool:
    return data[:4] == b"%PDF" or b"%PDF" in data[:1024]

def download_pdf(session: requests.Session, url: str) -> Optional[bytes]:
    if UNWANTED_PDF_NAME_RE.search(url):
        return None
    try:
        r = session.get(
            url,
            headers={**DEFAULT_HEADERS, "Accept": "application/pdf,*/*"},
            timeout=REQ_TIMEOUT,
            stream=True,
            allow_redirects=True
        )
        if r.status_code >= 400:
            return None

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

        if not _looks_like_pdf(data):
            return None

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
# TEXT EXTRACTION (PDF RÉEL)
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


def _is_probably_math_subject(text: str, filename: str = "") -> Tuple[bool, Dict]:
    """
    Nouveau : scoring permissif.
    Retour (ok, debug_metrics).
    """
    t = text or ""
    fn = (filename or "").lower()

    math_hits = len(MATH_SIGNAL_RE.findall(t))
    non_hits = len(NON_SUBJECT_TEXT_RE.findall(t))
    ex_hits = len(re.findall(r"\bexercice\b", t, re.IGNORECASE))
    q_hits = len(re.findall(r"\bquestion\b", t, re.IGNORECASE))

    # score
    score = 0
    score += min(20, math_hits) * 2
    score += min(10, ex_hits) * 4
    score += min(10, q_hits) * 3
    score -= min(20, non_hits) * 3

    # bonus filename
    if any(k in fn for k in ["terminale", "ts", "specialite", "spécialité", "bac", "sujet", "corrige", "corrig"]):
        score += 6
    if re.search(r"(19\d{2}|20\d{2})", fn):
        score += 2

    # Règle d’acceptation : on accepte dès qu’il y a des indices de sujet
    ok = False
    if ex_hits >= 1 and math_hits >= 4:
        ok = True
    elif q_hits >= 2 and math_hits >= 4:
        ok = True
    elif score >= 18:
        ok = True

    debug = {
        "math_hits": math_hits,
        "non_hits": non_hits,
        "ex_hits": ex_hits,
        "q_hits": q_hits,
        "score": score
    }
    return ok, debug


# =========================
# EXTRACTION Qi (marqueurs)
# =========================
_MARKER_RE = re.compile(
    r"^\s*(exercice\s*\d+|question\s*\d+|\d+\s*[\).\:-])\s*",
    re.IGNORECASE
)

_VERB_RE = re.compile(
    r"\b(calculer|déterminer|determiner|montrer|démontrer|demontrer|justifier|étudier|etudier|prouver)\b",
    re.IGNORECASE
)

def extract_qi_from_text(text: str) -> List[str]:
    raw = (text or "").replace("\r", "\n")
    raw = re.sub(r"\n{3,}", "\n\n", raw)

    lines = [ln.strip() for ln in raw.split("\n")]

    # segmentation par marqueurs
    segments: List[str] = []
    cur: List[str] = []
    saw_marker = False

    for ln in lines:
        if not ln:
            continue
        if _MARKER_RE.match(ln):
            saw_marker = True
            if cur:
                segments.append(" ".join(cur).strip())
                cur = []
            cur.append(ln)
        else:
            cur.append(ln)

    if cur:
        segments.append(" ".join(cur).strip())

    # fallback si aucun marqueur
    if not saw_marker:
        blocks = re.split(r"\n\s*\n", raw)
        segments = [re.sub(r"\s+", " ", b).strip() for b in blocks if len(b.strip()) >= MIN_QI_CHARS]

    # filtrage Qi candidates
    qi: List[str] = []
    for seg in segments:
        s = re.sub(r"\s+", " ", seg).strip()
        if len(s) < MIN_QI_CHARS:
            continue

        # critères : verbes ou suites
        if _VERB_RE.search(s) or _contains_suites_signal(s):
            if len(s) > 520:
                s = s[:520].rsplit(" ", 1)[0] + "…"
            qi.append(s)

    # dédoublonnage
    out, seen = [], set()
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
    is_recur = ("récurr" in txt) or ("recurr" in txt) or ("u(n+1)" in txt) or ("u_{n+1}" in txt)
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
def discover_pdf_links_bfs(session: requests.Session, start_urls: List[str], soft_limit: int) -> Tuple[List[str], Dict]:
    start_urls = [u.strip() for u in start_urls if u and u.strip()]
    if not start_urls:
        start_urls = ["https://apmep.fr"]

    # seed APMEP "Annales Terminale"
    seeds = []
    for u in start_urls:
        seeds.append(u)
        if "apmep.fr" in urlparse(u).netloc.lower() and urlparse(u).path.strip("/") == "":
            seeds.append("https://apmep.fr/Annales-Terminale-Generale")

    # queue
    q: List[Tuple[str, int]] = []
    seen_seed = set()
    for s in seeds:
        s, _ = urldefrag(s)
        if s and s not in seen_seed:
            seen_seed.add(s)
            q.append((s, 0))

    visited = set()
    pdfs: List[str] = []
    pages_crawled = 0

    while q and pages_crawled < CRAWL_MAX_PAGES and len(pdfs) < MAX_PDF_LINKS_GLOBAL:
        q.sort(key=lambda x: _url_priority(x[0]))
        url, depth = q.pop(0)

        if url in visited:
            continue
