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
# CONFIG
# =========================
UA = "SMAXIA-GTE/1.2 (terminales-maths-suites)"
REQ_TIMEOUT = 20
MAX_PDF_MB = 25
MIN_QI_CHARS = 18

# Crawl borné : on évite de parcourir tout le web
MAX_CRAWL_PAGES_PER_SOURCE = 40
MAX_PDF_LINKS_GLOBAL = 2000  # plafond sécurité
MAX_HTML_BYTES = 2_000_000

# Signaux forts "suites numériques"
SUITES_KEYWORDS: Set[str] = {
    "suite", "suites", "arithmétique", "arithmetique", "géométrique", "geometrique",
    "raison", "u_n", "v_n", "récurrence", "recurrence", "terme", "général", "general",
    "monotone", "bornée", "bornee", "majorée", "majoree", "minorée", "minoree",
    "limite", "convergence"
}

# Déclencheurs (réels)
TRIGGER_PHRASES = [
    "montrer que", "démontrer", "demontrer", "prouver",
    "calculer", "déterminer", "determiner", "étudier", "etudier",
    "justifier", "en déduire", "en deduire",
]

STOP_TOKENS = {
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "à", "a", "en", "pour",
    "dans", "sur", "avec", "par", "au", "aux", "d", "l", "si", "que", "qui", "on",
}

# PDFs à exclure par nom (PV, bulletins, sommaires…)
UNWANTED_PDF_NAME_RE = re.compile(
    r"(?:^|/)(pv\d+|bulletin|lettre|news|newsletter|edito|édito|sommaire|compte[- ]rendu|cr_)",
    re.IGNORECASE
)

# Signaux texte “non sujet”
NON_SUBJECT_TEXT_RE = re.compile(
    r"(édito|edito|sommaire|proc[eè]s[- ]verbal|compte[- ]rendu|adh[eé]sion|association|r[eé]gionale|bureau|convocation)",
    re.IGNORECASE
)

# Signaux texte “sujet maths”
MATH_SIGNAL_RE = re.compile(
    r"(exercice|question|\bu_n\b|\bv_n\b|r[eé]currence|suite|limite|convergence|monotone|major[eé]e|minor[eé]e|raison|terme g[eé]n[eé]ral)",
    re.IGNORECASE
)

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

def _safe_get_html(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        txt = r.text
        b = txt.encode("utf-8", errors="ignore")
        if len(b) > MAX_HTML_BYTES:
            return b[:MAX_HTML_BYTES].decode("utf-8", errors="ignore")
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
        full = urljoin(base_url, href)
        links.append(full)
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
        host = urlparse(full).netloc
        if host != same_host:
            continue
        pages.append(full)

    # Filtre pages "pertinentes" (Terminale/Bac/Sujets/Maths)
    def _relevant(u: str) -> bool:
        uu = u.lower()
        return any(k in uu for k in ["terminale", "bac", "sujet", "sujets", "annale", "math", "specialite", "spécialité", "lycee", "lycée"])

    pages = [p for p in pages if _relevant(p)]

    out, seen = [], set()
    for x in pages:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _seed_urls(urls: List[str]) -> List[str]:
    """
    Si l’utilisateur met juste https://apmep.fr, on ajoute automatiquement
    des entrées qui mènent à des sujets Terminale.
    """
    out = []
    for u in urls:
        u = u.strip()
        if not u:
            continue
        out.append(u)

        try:
            p = urlparse(u)
            host = p.netloc.lower()
            path = (p.path or "").strip("/")
            if "apmep.fr" in host and path == "":
                # seeds contractuels : Terminale + Bac + Sujets (sans casser si la page n’existe pas)
                out.append("https://www.apmep.fr/-Terminale-")
                out.append("https://www.apmep.fr/-Baccalaureat-")
                out.append("https://www.apmep.fr/-Sujets-")
        except Exception:
            pass

    # dédoublonnage ordre
    uniq, seen = [], set()
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq

def extract_pdf_links_from_url(url: str) -> List[str]:
    html = _safe_get_html(url)
    if not html:
        return []

    same_host = urlparse(url).netloc
    pdfs = _extract_pdf_links_from_html(url, html)

    # crawl 1 niveau borné
    internal_pages = _extract_internal_pages(url, html, same_host)[:MAX_CRAWL_PAGES_PER_SOURCE]
    for p in internal_pages:
        html2 = _safe_get_html(p)
        if not html2:
            continue
        pdfs.extend(_extract_pdf_links_from_html(p, html2))
        if len(pdfs) >= MAX_PDF_LINKS_GLOBAL:
            break

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
        if len(all_links) >= MAX_PDF_LINKS_GLOBAL:
            break

    # Filtre par nom (PV/bulletin/etc.)
    filtered = []
    for link in all_links:
        if UNWANTED_PDF_NAME_RE.search(link):
            continue
        filtered.append(link)

    # dédoublonnage global
    seen = set()
    uniq = []
    for x in filtered:
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
    return data[:4] == b"%PDF" or b"%PDF" in data[:1024]

def download_pdf(url: str) -> Optional[bytes]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT, stream=True)
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
# EXTRACTION TEXTE PDF (PDF RÉEL)
# =========================
def extract_text_from_pdf_bytes(pdf_bytes: bytes, max_pages: int = 25) -> str:
    # pdfplumber si dispo
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

    # fallback PyPDF2
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
    Garde-fou anti PV/bulletin :
    - rejette si trop de signaux "non sujet"
    - exige des signaux "exercice/question/suite/récurrence..."
    """
    if not text or len(text.strip()) < 200:
        return False

    if NON_SUBJECT_TEXT_RE.search(text):
        # pas une preuve absolue, mais très discriminant
        return False

    hits = len(MATH_SIGNAL_RE.findall(text))
    # seuil simple mais efficace
    return hits >= 6

# =========================
# EXTRACTION QI RÉELLES
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

        # indices "question/exercice"
        if re.search(r"\b(exercice|question)\b", b2, re.IGNORECASE):
            candidates.append(b2)
            continue

        # verbes déclencheurs + signal suites
        if re.search(r"\b(calculer|déterminer|determiner|montrer|démontrer|demontrer|justifier|étudier|etudier|prouver)\b", b2, re.IGNORECASE):
            candidates.append(b2)
            continue

        # signal suites (fort)
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
# CLUSTERING -> QC
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

    steps = [
        "Identifier la définition de la suite (explicite / récurrence) et les données initiales.",
    ]
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

    conc = "Conclusion attendue : expression de u(n) / raison / limite, avec justification minimale conforme au niveau Terminale."

    return {"usage": usage, "method": method, "trap": trap, "conc": conc}

def cluster_qi_to_qc(qis: List[QiItem], sim_threshold: float = 0.28) -> List[Dict]:
    clusters: List[Dict] = []  # {id, rep_tokens, qis}
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
        # si aucun déclencheur, fallback mots clés
        if not triggers:
            triggers = _top_keywords(qi_texts, k=6)
        # dédoublonnage
        trig2, seen = [], set()
        for x in triggers:
            if x not in seen:
                seen.add(x)
                trig2.append(x)

        n_q = len(qi_texts)
        psi = round(min(1.0, n_q / 25.0), 2)
        score = int(round(50 + 10 * math.log(1 + n_q), 0))

        # ARI = séquence logique (ordre des Qi dans la QC)
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
# SATURATION
# =========================
def compute_saturation(history_counts: List[int]) -> List[Dict]:
    return [{"Nombre de sujets injectés": i + 1, "Nombre de QC découvertes": v} for i, v in enumerate(history_counts)]

# =========================
# API PRINCIPALE POUR UI (SIGNATURE INCHANGÉE)
# =========================
def run_granulo_test(urls: List[str], volume: int) -> Dict:
    start = time.time()

    urls = [u.strip() for u in (urls or []) if u and u.strip()]
    urls = _seed_urls(urls)

    pdf_links = extract_pdf_links(urls, limit=max(5, int(volume)))

    sujets_rows = []
    all_qis: List[QiItem] = []
    qc_history = []

    rejected_pdf_non_subject = 0
    rejected_pdf_no_text = 0
    rejected_pdf_no_qi = 0

    for idx, pdf_url in enumerate(pdf_links, start=1):
        pdf_bytes = download_pdf(pdf_url)
        if not pdf_bytes:
            continue

        text = extract_text_from_pdf_bytes(pdf_bytes)
        if not text.strip():
            rejected_pdf_no_text += 1
            continue

        # ✅ garde-fou : on rejette les PV/sommaires/etc.
        if not _is_probably_math_subject(text):
            rejected_pdf_non_subject += 1
            continue

        qi_texts = extract_qi_from_text(text)

        # Filtre Suites numériques (fort)
        qi_texts = [q for q in qi_texts if _contains_suites_signal(q)]
        if not qi_texts:
            rejected_pdf_no_qi += 1
            continue

        subject_file = pdf_url.split("/")[-1].split("?")[0]
        source_host = urlparse(pdf_url).netloc

        # année éventuelle
        m_year = re.search(r"(19\d{2}|20\d{2})", subject_file)
        year = int(m_year.group(1)) if m_year else None

        sujets_rows.append({
            "Fichier": subject_file,
            "Nature": "SUJET (probable)",
            "Année": year,
            "Source": source_host,
        })

        subject_id = f"S{len(sujets_rows):04d}"
        for q in qi_texts:
            all_qis.append(QiItem(subject_id=subject_id, subject_file=subject_file, text=q))

        qc_current = cluster_qi_to_qc(all_qis)
        qc_history.append(len(qc_current))

        # stop si on a réellement traité "volume" sujets OK
        if len(sujets_rows) >= int(volume):
            break

    qc_list = cluster_qi_to_qc(all_qis)
    sat_points = compute_saturation(qc_history)

    elapsed = round(time.time() - start, 2)

    # Audit binaire “Qi -> QC” (ici, par construction : 1 Qi -> 1 cluster)
    n_qi = len(all_qis)
    n_qc = len(qc_list)
    n_subjects_ok = len(sujets_rows)

    return {
        "sujets": sujets_rows,
        "qc": qc_list,
        "saturation": sat_points,
        "audit": {
            "n_urls": len(urls),
            "n_pdf_links": len(pdf_links),
            "n_subjects_ok": n_subjects_ok,
            "n_qi": n_qi,
            "n_qc": n_qc,
            "rejected_pdf_non_subject": rejected_pdf_non_subject,
            "rejected_pdf_no_text": rejected_pdf_no_text,
            "rejected_pdf_no_qi": rejected_pdf_no_qi,
            "elapsed_s": elapsed,
            "binary_qi_to_qc_unmapped_exists": False,  # par construction
        }
    }
