# -*- coding: utf-8 -*-
# SMAXIA Granulo Test Engine - V31 SAFE (NO UI CHANGE, NO HARDCODE DATA)
# Provides: run_granulo_test(urls, volume) -> {sujets,qc,saturation,audit}

import io
import re
import time
import math
from collections import Counter
from urllib.parse import urljoin, urlparse

import requests
import pdfplumber
from bs4 import BeautifulSoup


UA = "SMAXIA-GTE-V31SAFE/1.0"
TIMEOUT = 25

MAX_HTML_PAGES = 60
MAX_DEPTH = 2
MAX_PDF_MB = 25
MAX_PAGES_TEXT = 35

# crawl hints for APMEP / annales
FOLLOW_HINTS = (
    "annales", "terminale", "generale", "générale", "specialite", "spécialité",
    "enseignement", "math", "bac", "sujets", "corrig", "ts", "tle", "examen"
)

# reject obvious non-subject PDFs
PDF_URL_BLACKLIST = (
    "pv", "bulletin", "sommaire", "edito", "édito", "regionale", "régionale",
    "journal", "vie-de", "litteramath", "revue", "compte-rendu", "editorial"
)

TEXT_BLACKLIST = (
    "sommaire", "éditorial", "editorial", "procès-verbal", "proces-verbal",
    "compte rendu", "assemblée générale", "bureau", "adhérents",
    "vie de la régionale", "agenda"
)

# Terminale Maths - Suites (simple keyword filter)
SUITES_KEYWORDS = set([
    "suite", "suites", "arithmetique", "arithmétique", "geometrique", "géométrique",
    "raison", "u_n", "u(n)", "u0", "u1", "u2", "u_{n}", "u_{n+1}",
    "recurrence", "récurrence", "limite", "convergence", "divergence",
    "monotone", "croissante", "decroissante", "décroissante",
    "bornee", "bornée", "majoree", "majorée", "minoree", "minorée",
    "terme", "somme"
])

STOPWORDS_FR = set([
    "le","la","les","de","des","du","un","une","et","ou","a","à","au","aux","en","pour","par",
    "sur","dans","avec","sans","que","qui","quoi","dont","ou","où","est","sont","etre","être",
    "soit","on","il","elle","ils","elles","ce","cet","cette","ces","se","sa","son","ses",
    "leur","leurs","plus","moins","tres","très","afin","ainsi","alors","donc"
])


def _norm(s):
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize(s):
    s = _norm(s)
    return re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ0-9_]+", s)


def _jaccard(a, b):
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    uni = len(sa | sb)
    return float(inter) / float(uni) if uni else 0.0


def _is_http(u):
    try:
        p = urlparse(u)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


def _same_domain(seed_netloc, u):
    try:
        n = urlparse(u).netloc.lower()
        seed = (seed_netloc or "").lower()
        return (n == seed) or n.endswith("." + seed)
    except Exception:
        return False


def _looks_like_pdf(u):
    return ".pdf" in (u or "").lower()


def _blacklisted_pdf_url(u):
    u = (u or "").lower()
    for b in PDF_URL_BLACKLIST:
        if b in u:
            return True
    return False


def _looks_like_subject_text(text):
    t = _norm(text)
    if not t:
        return False
    hits = 0
    for b in TEXT_BLACKLIST:
        if b in t:
            hits += 1
    if hits >= 2:
        return False
    return True


def _contains_suites_signal(text):
    toks = set(_tokenize(text))
    return len(toks & SUITES_KEYWORDS) > 0


def _should_follow(href_abs, anchor_text, seed_netloc):
    if not _is_http(href_abs):
        return False
    if not _same_domain(seed_netloc, href_abs):
        return False

    h = href_abs.lower()
    if any(h.endswith(ext) for ext in (".jpg",".jpeg",".png",".gif",".zip",".rar",".7z")):
        return False
    if "mailto:" in h or "javascript:" in h:
        return False

    t = (anchor_text or "").strip().lower()
    for k in FOLLOW_HINTS:
        if (k in h) or (k in t):
            return True
    return False


def extract_pdf_links_from_url(seed_url):
    seed_netloc = urlparse(seed_url).netloc or ""
    if not seed_netloc:
        return []

    visited = set()
    seen_pdf = set()
    pdfs = []

    queue = [(seed_url, 0)]

    while queue and len(visited) < MAX_HTML_PAGES and len(pdfs) < 2000:
        page_url, depth = queue.pop(0)
        if page_url in visited:
            continue
        visited.add(page_url)

        try:
            r = requests.get(page_url, headers={"User-Agent": UA}, timeout=TIMEOUT)
            r.raise_for_status()
        except Exception:
            continue

        ctype = (r.headers.get("Content-Type") or "").lower()
        if ("application/pdf" in ctype) or page_url.lower().endswith(".pdf"):
            if _looks_like_pdf(page_url) and (page_url not in seen_pdf) and (not _blacklisted_pdf_url(page_url)):
                seen_pdf.add(page_url)
                pdfs.append(page_url)
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            href_abs = urljoin(page_url, href)

            if _looks_like_pdf(href_abs):
                if _blacklisted_pdf_url(href_abs):
                    continue
                if href_abs not in seen_pdf:
                    seen_pdf.add(href_abs)
                    pdfs.append(href_abs)
                continue

            if depth < MAX_DEPTH:
                txt = a.get_text(" ", strip=True)
                if _should_follow(href_abs, txt, seed_netloc):
                    if href_abs not in visited:
                        queue.append((href_abs, depth + 1))

    # stable dedupe
    out = []
    s = set()
    for x in pdfs:
        if x not in s:
            s.add(x)
            out.append(x)
    return out


def extract_pdf_links(urls, volume):
    # buffer to compensate rejections
    target = max(int(volume) * 10, int(volume))
    all_links = []
    seen = set()
    for u in (urls or []):
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


def download_pdf(url):
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT, stream=True)
        r.raise_for_status()

        ctype = (r.headers.get("Content-Type") or "").lower()
        if ("pdf" not in ctype) and (".pdf" not in url.lower()):
            return None

        cl = r.headers.get("Content-Length")
        if cl:
            try:
                mb = float(int(cl)) / (1024.0 * 1024.0)
                if mb > MAX_PDF_MB:
                    return None
            except Exception:
                pass

        buf = io.BytesIO()
        max_bytes = int(MAX_PDF_MB) * 1024 * 1024
        for chunk in r.iter_content(chunk_size=262144):
            if not chunk:
                continue
            buf.write(chunk)
            if buf.tell() > max_bytes:
                return None
        return buf.getvalue()
    except Exception:
        return None


def extract_text_from_pdf_bytes(pdf_bytes):
    parts = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            n = min(len(pdf.pages), int(MAX_PAGES_TEXT))
            for i in range(n):
                t = pdf.pages[i].extract_text() or ""
                t = t.replace("\u00ad", "")
                if t.strip():
                    parts.append(t)
    except Exception:
        return ""
    return "\n".join(parts)


# --- Qi extraction (heuristic, no OCR) ---
VERBS = ("calculer","determiner","déterminer","montrer","demontrer","démontrer","justifier",
         "etudier","étudier","resoudre","résoudre","verifier","vérifier","prouver","en deduire","en déduire","exprimer")

QI_START_RE = re.compile(r"^\s*(?:exercice\s*\d+|question\s*\d+|partie\s*[a-z0-9]+|\d+\s*[\).\:-]|[a-z]\)\s*)\s*(.*)$", re.IGNORECASE)
QI_VERB_RE = re.compile(r"\b(" + "|".join([re.escape(v) for v in VERBS]) + r")\b", re.IGNORECASE)
MATH_SIG_RE = re.compile(r"(\bu[_\s\{\(]*n\b|\bu[_\s\{\(]*n\+1\b|\blim\b|\b\d+\b|[=<>+\-*/^]|sqrt|ln|log|exp)", re.IGNORECASE)


def extract_qi_from_text(text):
    raw = (text or "").replace("\r", "\n")
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    out = []
    i = 0
    while i < len(lines):
        ln = re.sub(r"\s+", " ", lines[i]).strip()
        m = QI_START_RE.match(ln)
        cand = m.group(1).strip() if m else ln

        verb_ok = bool(QI_VERB_RE.search(cand))
        math_ok = bool(MATH_SIG_RE.search(cand))
        suites_ok = _contains_suites_signal(cand)

        if (verb_ok and (math_ok or suites_ok)) or (suites_ok and len(cand) >= 18):
            agg = [cand]
            for j in (1, 2):
                if i + j >= len(lines):
                    break
                nxt = lines[i + j].strip()
                if QI_START_RE.match(nxt):
                    break
                if MATH_SIG_RE.search(nxt) or _contains_suites_signal(nxt):
                    agg.append(nxt)
            qi = " ".join(agg)
            qi = re.sub(r"\s+", " ", qi).strip()
            if len(qi) > 420:
                qi = qi[:420].rsplit(" ", 1)[0] + "..."
            if len(qi) >= 18:
                out.append(qi)

        i += 1

    # dedupe
    seen = set()
    final = []
    for q in out:
        k = _norm(q)
        if k not in seen:
            seen.add(k)
            final.append(q)
    return final


def _extract_triggers(qi_texts, k):
    toks = []
    for t in qi_texts:
        toks.extend(_tokenize(t))
    filtered = []
    for tok in toks:
        if tok in STOPWORDS_FR:
            continue
        if tok.isdigit():
            continue
        if len(tok) < 4:
            continue
        filtered.append(tok)
    freq = Counter(filtered)
    return [w for (w, _) in freq.most_common(int(k))]


def _infer_type(blob):
    b = _norm(blob)
    if ("recurrence" in b) or ("récurrence" in b):
        return "RECURRENCE"
    if ("geometrique" in b) or ("géométrique" in b):
        return "GEOMETRIQUE"
    if ("arithmetique" in b) or ("arithmétique" in b):
        return "ARITHMETIQUE"
    if "limite" in b and ("indetermination" in b or "indétermination" in b):
        return "LIM_INDET"
    if "limite" in b:
        return "LIMITE"
    if ("monotone" in b) or ("croissante" in b) or ("decroissante" in b) or ("décroissante" in b):
        return "MONOTONIE"
    if ("somme" in b) or ("sommes" in b):
        return "SOMME"
    return "GENERIC"


def _build_ari_frt_title(qi_texts):
    blob = " ".join(qi_texts)
    t = _infer_type(blob)

    title = qi_texts[0].strip() if qi_texts else "Question"
    if len(title) > 90:
        title = title[:90].rsplit(" ", 1)[0] + "..."

    if t == "RECURRENCE":
        ari = "1) Poser P(n)\n2) Initialisation\n3) Hérédité (P(n)=>P(n+1))\n4) Conclusion"
        frt = {
            "quand_utiliser": "Quand l'enonce demande 'pour tout n' avec une relation n -> n+1.",
            "methode_redigee": "Ecrire P(n). Faire initialisation, puis heredite, puis conclure.",
            "pieges": "Oublier l'initialisation ou ne pas utiliser P(n) dans l'heredite.",
            "conclusion": "Conclure explicitement : 'Donc, pour tout n, ...'."
        }
    elif t == "GEOMETRIQUE":
        ari = "1) Calculer u(n+1)/u(n)\n2) Montrer que c'est une constante q\n3) Conclure geometrie\n4) Donner u(n)=u(0)*q^n"
        frt = {
            "quand_utiliser": "Quand l'enonce parle de suite geometrique ou raison q.",
            "methode_redigee": "Calculer le quotient et montrer qu'il est constant.",
            "pieges": "Confondre difference et quotient.",
            "conclusion": "Donner q puis l'expression explicite de u(n)."
        }
    elif t == "ARITHMETIQUE":
        ari = "1) Calculer u(n+1)-u(n)\n2) Montrer que c'est une constante r\n3) Conclure arithmetique\n4) Donner u(n)=u(0)+n*r"
        frt = {
            "quand_utiliser": "Quand l'enonce parle de suite arithmetique ou pas constant.",
            "methode_redigee": "Calculer la difference et montrer qu'elle est constante.",
            "pieges": "Calculer un quotient au lieu d'une difference.",
            "conclusion": "Donner r puis l'expression explicite."
        }
    elif t == "MONOTONIE":
        ari = "1) Etudier u(n+1)-u(n)\n2) Determiner le signe\n3) Conclure croissance/decroissance"
        frt = {
            "quand_utiliser": "Quand l'enonce demande les variations ou monotonicite.",
            "methode_redigee": "Etudier le signe de u(n+1)-u(n) pour tout n.",
            "pieges": "Conclure sans justifier le signe pour tout n.",
            "conclusion": "Conclure clairement le sens de variation."
        }
    elif t == "LIM_INDET":
        ari = "1) Identifier l'indetermination\n2) Transformer (factoriser/rationaliser/equivalent)\n3) Simplifier\n4) Calculer la limite"
        frt = {
            "quand_utiliser": "Quand une limite donne 0/0 ou inf/inf.",
            "methode_redigee": "Transformer pour lever l'indetermination puis recalculer la limite.",
            "pieges": "Simplifier illegalement (division par 0) ou oublier le domaine.",
            "conclusion": "Donner la limite et la transformation utilisee."
        }
    elif t == "LIMITE":
        ari = "1) Analyser les termes dominants\n2) Appliquer les regles sur les limites/encadrement\n3) Conclure"
        frt = {
            "quand_utiliser": "Quand l'enonce demande la limite de u(n).",
            "methode_redigee": "Utiliser regles usuelles ou encadrement selon la forme.",
            "pieges": "Appliquer une regle sans verifier les hypotheses.",
            "conclusion": "Conclure explicitement : lim u(n)=..."
        }
    elif t == "SOMME":
        ari = "1) Ecrire la somme partielle S(n)\n2) Utiliser forme connue (geo/telescopique)\n3) Simplifier\n4) Conclure"
        frt = {
            "quand_utiliser": "Quand l'enonce demande une somme de termes de suite.",
            "methode_redigee": "Ecrire S(n), utiliser une formule, puis simplifier.",
            "pieges": "Erreur d'indices (0..n vs 1..n).",
            "conclusion": "Donner l'expression finale de la somme."
        }
    else:
        ari = "1) Identifier les donnees\n2) Choisir l'outil (calcul/recurrence/limite)\n3) Executer proprement\n4) Conclure"
        frt = {
            "quand_utiliser": "Question standard sur suite (type non detecte).",
            "methode_redigee": "Reformuler, choisir un outil coherent, calculer, conclure.",
            "pieges": "Sauter une justification ou melanger des methodes.",
            "conclusion": "Conclusion explicite + resultat final."
        }

    return ari, frt, title


def _group_qi_by_file(qi_items):
    out = {}
    for it in qi_items:
        out.setdefault(it["subject_file"], []).append(it["text"])
    return out


def cluster_qi_to_qc(qi_items, sim_threshold):
    clusters = []  # each: {id, rep_tokens, items}
    qc_idx = 1

    for it in qi_items:
        toks = _tokenize(it["text"])
        if not toks:
            continue

        best_i = None
        best_sim = 0.0
        for i, c in enumerate(clusters):
            sim = _jaccard(toks, c["rep_tokens"])
            if sim > best_sim:
                best_sim = sim
                best_i = i

        if best_i is not None and best_sim >= float(sim_threshold):
            clusters[best_i]["items"].append(it)
            clusters[best_i]["rep_tokens"] = list(set(clusters[best_i]["rep_tokens"]) | set(toks))
        else:
            clusters.append({
                "id": "QC-{0:03d}".format(qc_idx),
                "rep_tokens": toks,
                "items": [it]
            })
            qc_idx += 1

    n_tot = len(qi_items)
    qc_out = []
    for c in clusters:
        qi_texts = [x["text"] for x in c["items"]]
        triggers = _extract_triggers(qi_texts, 7)
        ari, frt, title
