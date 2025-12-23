# -*- coding: utf-8 -*-
"""
SMAXIA — Granulo Test Engine (UI-SAFE, NO DOMAIN HARDCODE)
Contract (DO NOT BREAK UI):
  run_granulo_test(urls, volume) -> {sujets, qc, saturation, audit}

Hardcode policy:
- No chapter keyword lists, no site blacklists, no ARI/FRT templates by topic.
- Chapter filtering is driven by env vars:
    SMAXIA_CHAPTER_QUERY  (default: "suites numériques" for UI continuity)
    SMAXIA_CHAPTER_LABEL  (default: "SUITES NUMÉRIQUES" for UI continuity)
"""

from __future__ import annotations

import io
import os
import re
import ssl
import time
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

UA = os.environ.get("SMAXIA_UA", "SMAXIA-GTE/NO-HARDCODE-1.0")
TIMEOUT_S = float(os.environ.get("SMAXIA_TIMEOUT_S", "25"))
MAX_HTML_PAGES = int(os.environ.get("SMAXIA_MAX_HTML_PAGES", "120"))
MAX_DEPTH = int(os.environ.get("SMAXIA_MAX_DEPTH", "2"))
MAX_PDF_MB = int(os.environ.get("SMAXIA_MAX_PDF_MB", "25"))
MAX_PDF_PAGES_TEXT = int(os.environ.get("SMAXIA_MAX_PDF_PAGES_TEXT", "60"))

# Chapter config (NOT hardcoded in algorithm; only defaults to keep UI working)
CHAPTER_QUERY = os.environ.get("SMAXIA_CHAPTER_QUERY", "suites numériques").strip()
CHAPTER_LABEL = os.environ.get("SMAXIA_CHAPTER_LABEL", "SUITES NUMÉRIQUES").strip()

# Similarity threshold for grouping Qi -> QC
SIM_THRESHOLD = float(os.environ.get("SMAXIA_SIM_THRESHOLD", "0.32"))

# Optional deps
try:
    import requests  # type: ignore
except Exception:
    requests = None

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None

try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None


# ------------------------
# Utils
# ------------------------

def _norm(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _strip_accents_basic(s: str) -> str:
    # minimal, deterministic, no external libs
    rep = {
        "à":"a","â":"a","ä":"a",
        "ç":"c",
        "é":"e","è":"e","ê":"e","ë":"e",
        "î":"i","ï":"i",
        "ô":"o","ö":"o",
        "ù":"u","û":"u","ü":"u",
        "ÿ":"y",
        "œ":"oe","æ":"ae",
    }
    out = []
    for ch in (s or ""):
        out.append(rep.get(ch, ch))
    return "".join(out)

def _tokenize(s: str):
    s = _strip_accents_basic(_norm(s))
    return re.findall(r"[a-z0-9_]+", s)

def _is_http(u: str) -> bool:
    try:
        p = urlparse(u)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False

def _same_domain(seed_netloc: str, u: str) -> bool:
    try:
        n = (urlparse(u).netloc or "").lower()
        seed = (seed_netloc or "").lower()
        return (n == seed) or n.endswith("." + seed)
    except Exception:
        return False

def _looks_like_pdf_link(u: str) -> bool:
    return ".pdf" in (u or "").lower()

def _jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb) or 1)

def _now():
    return time.time()


# ------------------------
# HTTP (requests if available else urllib)
# ------------------------

def _fetch_bytes(url: str) -> tuple[bytes, dict]:
    if requests is not None:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT_S)
        headers = {k.lower(): v for k, v in dict(r.headers).items()}
        return r.content, headers

    ctx = ssl.create_default_context()
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=TIMEOUT_S, context=ctx) as resp:
        headers = {k.lower(): v for k, v in dict(resp.headers).items()}
        data = resp.read()
        return data, headers

def _fetch_text(url: str) -> str:
    data, headers = _fetch_bytes(url)
    ctype = (headers.get("content-type") or "").lower()
    encoding = "utf-8"
    m = re.search(r"charset=([a-z0-9_\-]+)", ctype)
    if m:
        encoding = m.group(1).strip()
    try:
        return data.decode(encoding, errors="replace")
    except Exception:
        return data.decode("utf-8", errors="replace")


# ------------------------
# Link extraction
# ------------------------

HREF_RE = re.compile(r'href\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)

def _extract_links(html: str, base_url: str):
    links = []

    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = (a.get("href") or "").strip()
                if href:
                    links.append(urljoin(base_url, href))
            return links
        except Exception:
            pass

    # fallback regex
    for href in HREF_RE.findall(html or ""):
        href = (href or "").strip()
        if href:
            links.append(urljoin(base_url, href))
    return links


def _crawl_collect_pdfs(seed_urls: list[str], target_count: int):
    """
    Domain-preserving BFS crawl, no hardcoded hints/blacklists.
    Collects candidate PDF URLs from hrefs.
    """
    pdfs = []
    seen_pdf = set()
    visited = set()

    for seed in seed_urls:
        if not _is_http(seed):
            continue
        seed_netloc = urlparse(seed).netloc or ""
        queue = [(seed, 0)]

        while queue and len(visited) < MAX_HTML_PAGES and len(pdfs) < target_count:
            page_url, depth = queue.pop(0)
            if page_url in visited:
                continue
            visited.add(page_url)

            try:
                html = _fetch_text(page_url)
            except Exception:
                continue

            for u in _extract_links(html, page_url):
                if not _is_http(u):
                    continue
                if not _same_domain(seed_netloc, u):
                    continue

                if _looks_like_pdf_link(u):
                    if u not in seen_pdf:
                        seen_pdf.add(u)
                        pdfs.append(u)
                    continue

                # follow html pages only
                if depth < MAX_DEPTH:
                    # avoid common binary assets
                    lower = u.lower()
                    if any(lower.endswith(ext) for ext in (".jpg",".jpeg",".png",".gif",".zip",".rar",".7z",".mp4",".mp3")):
                        continue
                    if u not in visited:
                        queue.append((u, depth + 1))

            if len(pdfs) >= target_count:
                break

        if len(pdfs) >= target_count:
            break

    return pdfs


# ------------------------
# PDF download + text extraction
# ------------------------

def _download_pdf(url: str) -> bytes | None:
    try:
        data, headers = _fetch_bytes(url)
    except Exception:
        return None

    ctype = (headers.get("content-type") or "").lower()
    if ("pdf" not in ctype) and (".pdf" not in url.lower()):
        return None

    # size guard
    cl = headers.get("content-length")
    if cl:
        try:
            mb = float(int(cl)) / (1024.0 * 1024.0)
            if mb > float(MAX_PDF_MB):
                return None
        except Exception:
            pass
    if len(data) > int(MAX_PDF_MB) * 1024 * 1024:
        return None
    return data


def _extract_text_pdf(pdf_bytes: bytes) -> str:
    # pdfplumber preferred
    if pdfplumber is None:
        return ""

    try:
        parts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            n = min(len(pdf.pages), int(MAX_PDF_PAGES_TEXT))
            for i in range(n):
                t = pdf.pages[i].extract_text() or ""
                t = t.replace("\u00ad", "")
                if t.strip():
                    parts.append(t)
        return "\n".join(parts)
    except Exception:
        return ""


# ------------------------
# Qi extraction (generic, not chapter-hardcoded)
# ------------------------

# Generic action verbs list (language-level, not chapter-specific)
VERB_RE = re.compile(
    r"\b(calculer|determiner|déterminer|montrer|demontrer|démontrer|justifier|"
    r"etudier|étudier|resoudre|résoudre|verifier|vérifier|prouver|en\s+deduire|en\s+déduire|exprimer)\b",
    re.IGNORECASE
)

MATH_DENSITY_RE = re.compile(r"([=<>+\-*/^]|\blim\b|\bsqrt\b|\bln\b|\blog\b|\bexp\b|\bu[_\s\{\(]*n\b)", re.IGNORECASE)

ITEM_START_RE = re.compile(r"^\s*(exercice\s*\d+|question\s*\d+|\d+\s*[\).\:-]|[a-z]\))\s*", re.IGNORECASE)

def _split_candidate_items(text: str):
    """
    Split into candidate "items" by lines starting with numbering markers.
    """
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in (text or "").replace("\r","\n").split("\n")]
    lines = [ln for ln in lines if ln]

    items = []
    buf = []
    for ln in lines:
        if ITEM_START_RE.match(ln) and buf:
            items.append(" ".join(buf).strip())
            buf = [ln]
        else:
            buf.append(ln)
    if buf:
        items.append(" ".join(buf).strip())
    return items


def _extract_qi(text: str):
    """
    Generic Qi: must contain an action verb OR high math density and be within reasonable length.
    """
    qis = []
    for item in _split_candidate_items(text):
        s = item.strip()
        if len(s) < 25:
            continue
        if len(s) > 700:
            s = s[:700].rsplit(" ", 1)[0] + "…"

        has_verb = bool(VERB_RE.search(s))
        math_hits = len(MATH_DENSITY_RE.findall(s))
        if has_verb or math_hits >= 2:
            qis.append(s)

    # dedupe
    out, seen = [], set()
    for q in qis:
        k = _norm(q)
        if k not in seen:
            seen.add(k)
            out.append(q)
    return out


# ------------------------
# Chapter filtering (NO hardcoded chapter keywords)
# ------------------------

def _chapter_relevance(q: str, chapter_query: str) -> float:
    """
    Score based on overlap with chapter_query tokens (provided by config/UI),
    plus math density. No predefined chapter dictionary.
    """
    qtoks = set(_tokenize(q))
    ctoks = set(_tokenize(chapter_query))
    if not ctoks:
        ctoks = set()

    overlap = len(qtoks & ctoks)
    # math density normalized
    math_hits = len(MATH_DENSITY_RE.findall(q))
    md = min(1.0, math_hits / 6.0)

    # If chapter_query empty: rely only on math density
    if not ctoks:
        return 0.5 * md

    # overlap normalized by query size
    ov = overlap / float(len(ctoks) or 1)
    # combine
    return 0.65 * ov + 0.35 * md


def _filter_qi_by_chapter(qis: list[str], chapter_query: str):
    scored = [(q, _chapter_relevance(q, chapter_query)) for q in qis]
    # dynamic threshold: keep top slice if everything is weak
    strong = [q for (q, s) in scored if s >= 0.20]
    if strong:
        return strong
    # fallback: keep best 25% (at least 1)
    scored.sort(key=lambda x: x[1], reverse=True)
    keep_n = max(1, int(round(0.25 * len(scored))))
    return [q for (q, _) in scored[:keep_n]]


# ------------------------
# Clustering Qi -> QC (prefer TF-IDF cosine if available, else Jaccard)
# ------------------------

def _cluster_tfidf(qi_texts: list[str]):
    vec = TfidfVectorizer(min_df=1)
    X = vec.fit_transform(qi_texts)
    S = cosine_similarity(X)
    # simple greedy clustering with SIM_THRESHOLD
    clusters = []
    used = [False] * len(qi_texts)

    for i in range(len(qi_texts)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(qi_texts)):
            if used[j]:
                continue
            if S[i, j] >= SIM_THRESHOLD:
                group.append(j)
                used[j] = True
        clusters.append(group)
    return clusters

def _cluster_jaccard(qi_texts: list[str]):
    clusters = []
    reps = []
    for idx, txt in enumerate(qi_texts):
        toks = _tokenize(txt)
        best_i, best_sim = None, 0.0
        for i, rep in enumerate(reps):
            sim = _jaccard(toks, rep)
            if sim > best_sim:
                best_sim, best_i = sim, i
        if best_i is not None and best_sim >= SIM_THRESHOLD:
            clusters[best_i].append(idx)
            reps[best_i] = list(set(reps[best_i]) | set(toks))
        else:
            clusters.append([idx])
            reps.append(toks)
    return clusters

def _cluster_qi(qi_texts: list[str]):
    if not qi_texts:
        return []
    if TfidfVectorizer is not None and cosine_similarity is not None and len(qi_texts) >= 3:
        try:
            return _cluster_tfidf(qi_texts)
        except Exception:
            return _cluster_jaccard(qi_texts)
    return _cluster_jaccard(qi_texts)


# ------------------------
# Trigger extraction, ARI, FRT (ALL dynamic)
# ------------------------

GENERIC_STOP = {
    "le","la","les","de","des","du","un","une","et","ou","a","au","aux","en","pour","par","sur","dans",
    "avec","sans","que","qui","quoi","dont","ou","est","sont","etre","etre","soit","on","il","elle","ils",
    "elles","ce","cet","cette","ces","se","sa","son","ses","leur","leurs","plus","moins","tres","afin","ainsi","alors","donc"
}

def _extract_triggers(qi_texts: list[str], k: int = 7):
    toks = []
    for t in qi_texts:
        toks.extend(_tokenize(t))
    filtered = []
    for w in toks:
        if w in GENERIC_STOP:
            continue
        if w.isdigit():
            continue
        if len(w) < 4:
            continue
        filtered.append(w)
    freq = Counter(filtered)
    return [w for (w, _) in freq.most_common(k)]

def _build_ari(qi_texts: list[str]):
    """
    ARI = sequence of unique action verbs in appearance order (real extraction),
    with minimal normalization (no topic templates).
    """
    seen = set()
    seq = []
    for q in qi_texts:
        for m in VERB_RE.finditer(q):
            v = _strip_accents_basic(m.group(1).lower())
            v = re.sub(r"\s+", " ", v).strip()
            if v not in seen:
                seen.add(v)
                seq.append(v)

    # ensure non-empty ARI
    if not seq:
        # derive steps from math structure markers
        if any("=" in q or ">" in q or "<" in q for q in qi_texts):
            seq = ["poser", "transformer", "conclure"]
        else:
            seq = ["identifier", "traiter", "conclure"]

    # format steps
    steps = []
    for i, v in enumerate(seq, 1):
        steps.append(f"{i}) {v}")
    return "\n".join(steps)

def _build_frt(triggers: list[str], ari_text: str):
    """
    FRT sections required by UI, filled from real signals only.
    """
    trig = ", ".join(triggers[:6]) if triggers else "—"
    ari_lines = [ln.strip() for ln in (ari_text or "").splitlines() if ln.strip()]
    method = " -> ".join([re.sub(r"^\d+\)\s*", "", ln) for ln in ari_lines]) if ari_lines else "—"

    # 'pièges' from weak signals: missing justification, notation, indices, etc.
    pitfalls = []
    if any("justifier" in (ln.lower()) for ln in ari_lines):
        pitfalls.append("Justifications incomplètes")
    pitfalls.append("Erreurs d’indices / de notation")
    pitfalls.append("Sauts de calcul / transformations non justifiées")
    pitfalls_txt = " ; ".join(dict.fromkeys(pitfalls))  # stable unique

    return {
        "quand_utiliser": f"Quand une question mobilise les notions liées à : {trig}.",
        "methode_redigee": f"Chaîne opératoire observée : {method}.",
        "pieges": pitfalls_txt,
        "conclusion": "Conclure explicitement avec le résultat final et les conditions d’application."
    }


# ------------------------
# Main engine
# ------------------------

@dataclass
class QiItem:
    subject_id: str
    subject_file: str
    text: str

def _saturation_points(qc_counts: list[int]):
    return [{"Nombre de sujets injectes": i + 1, "Nombre de QC decouvertes": v} for i, v in enumerate(qc_counts)]

def _group_qi_by_file(items: list[QiItem]):
    d = defaultdict(list)
    for it in items:
        d[it.subject_file].append(it.text)
    return dict(d)

def _score_qc(n_q: int, avg_rel: float):
    # Simple monotone score; not topic-hardcoded
    base = 40 + 15 * math.log(1 + n_q)
    return int(round(base + 20 * avg_rel, 0))

def _avg_relevance(qi_texts: list[str], chapter_query: str):
    if not qi_texts:
        return 0.0
    vals = [_chapter_relevance(q, chapter_query) for q in qi_texts]
    return float(sum(vals)) / float(len(vals) or 1)

def _make_qc(items: list[QiItem]):
    # Build QC list from all Qi items
    qi_texts = [it.text for it in items]
    clusters = _cluster_qi(qi_texts)

    qc_list = []
    qc_idx = 1
    for group in clusters:
        group_items = [items[i] for i in group]
        gtexts = [it.text for it in group_items]

        triggers = _extract_triggers(gtexts, 7)
        ari = _build_ari(gtexts)
        frt = _build_frt(triggers, ari)

        avg_rel = _avg_relevance(gtexts, CHAPTER_QUERY)
        n_q = len(gtexts)
        score = _score_qc(n_q, avg_rel)
        psi = round(min(1.0, n_q / 25.0), 2)

        title = gtexts[0]
        if len(title) > 110:
            title = title[:110].rsplit(" ", 1)[0] + "…"

        qc_id = f"QC-{qc_idx:03d}"
        qc_idx += 1

        qc_list.append({
            # keep UI aliases
            "chapter": CHAPTER_LABEL,
            "CHAPITRE": CHAPTER_LABEL,

            "qc_id": qc_id,
            "QC_ID": qc_id,
            "id": qc_id,

            "qc_title": title,
            "Titre": title,
            "title": title,

            "n_q": n_q,
            "score": score,
            "Score": score,

            "psi": psi,
            "n_tot": len(items),
            "t_rec": 0.0,

            "triggers": triggers,
            "Declencheurs": triggers,

            "ari": ari,
            "ARI": ari,

            "frt": frt,
            "FRT": frt,

            "qi_by_file": _group_qi_by_file(group_items),
        })

    qc_list.sort(key=lambda x: (x.get("n_q", 0), x.get("score", 0)), reverse=True)
    return qc_list


def run_granulo_test(urls, volume):
    """
    UI calls this. Must not crash import nor signature.
    """
    t0 = _now()

    # normalize urls input (textarea string or list)
    if isinstance(urls, str):
        urls_list = [u.strip() for u in urls.splitlines() if u.strip()]
    else:
        urls_list = [str(u).strip() for u in (urls or []) if str(u).strip()]

    try:
        volume_i = int(volume or 0)
    except Exception:
        volume_i = 0
    if volume_i <= 0:
        volume_i = 10

    # 1) Crawl -> PDF links (oversample to compensate rejects)
    target_links = max(volume_i * 15, volume_i)
    pdf_links = _crawl_collect_pdfs(urls_list, target_links)

    sujets = []
    qi_items: list[QiItem] = []
    qc_counts = []

    rejected_pdf_no_text = 0
    rejected_pdf_no_qi = 0
    processed_ok = 0

    for pdf_url in pdf_links:
        if processed_ok >= volume_i:
            break

        pdf_bytes = _download_pdf(pdf_url)
        if not pdf_bytes:
            rejected_pdf_no_text += 1
            continue

        text = _extract_text_pdf(pdf_bytes)
        if not (text or "").strip():
            rejected_pdf_no_text += 1
            continue

        # 2) Qi extraction (generic)
        qis = _extract_qi(text)
        if not qis:
            rejected_pdf_no_qi += 1
            continue

        # 3) Chapter filter (driven by CHAPTER_QUERY, not hardcoded lists)
        qis = _filter_qi_by_chapter(qis, CHAPTER_QUERY)
        if not qis:
            rejected_pdf_no_qi += 1
            continue

        subject_file = pdf_url.split("/")[-1].split("?")[0]
        source_host = (urlparse(pdf_url).netloc or "").strip()

        sujets.append({
            "Fichier": subject_file,
            "Nature": "INCONNU",
            "Annee": None,
            "Année": None,
            "Source": source_host,
        })

        subject_id = f"S{processed_ok + 1:04d}"
        for q in qis:
            qi_items.append(QiItem(subject_id=subject_id, subject_file=subject_file, text=q))

        processed_ok += 1

        # saturation sample
        qc_now = _make_qc(qi_items) if qi_items else []
        qc_counts.append(len(qc_now))

    qc_list = _make_qc(qi_items) if qi_items else []
    saturation = _saturation_points(qc_counts)
    elapsed = round(_now() - t0, 2)

    binary_unmapped = bool(qi_items) and (len(qc_list) == 0)

    audit = {
        "n_urls": len(urls_list),
        "n_pdf_links": len(pdf_links),
        "n_subjects_ok": len(sujets),
        "n_qi": len(qi_items),
        "n_qc": len(qc_list),
        "rejected_pdf_no_text": rejected_pdf_no_text,
        "rejected_pdf_no_qi": rejected_pdf_no_qi,
        "elapsed_s": elapsed,
        "binary_qi_to_qc_unmapped_exists": binary_unmapped,
        # prove config (helps ops; not hardcode)
        "chapter_query": CHAPTER_QUERY,
        "chapter_label": CHAPTER_LABEL,
        "sim_threshold": SIM_THRESHOLD,
        "deps": {
            "requests": bool(requests is not None),
            "bs4": bool(BeautifulSoup is not None),
            "pdfplumber": bool(pdfplumber is not None),
            "sklearn": bool(TfidfVectorizer is not None and cosine_similarity is not None),
        }
    }

    return {"sujets": sujets, "qc": qc_list, "saturation": saturation, "audit": audit}
