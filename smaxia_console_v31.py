# smaxia_granulo_engine_real.py
# =============================================================================
# SMAXIA - MOTEUR GRANULO V3 (POST-AUDIT GPT)
# =============================================================================
# Correctifs appliqués:
# 1. Suppression "un", "vn", "wn" ambigus → patterns explicites u_n, u(n)
# 2. is_math_content() strict: question + math obligatoire
# 3. Split non-capturant dans extract_qi_from_text()
# 4. detect_year() sans invention (None si inconnu)
# 5. BFS récursif réel pour scraping
# 6. Audit log des rejets
# =============================================================================
# NOTE: Paramètres hardcodés marqués [P3-CONFIG] pour migration vers Academic Pack
# =============================================================================

from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
from datetime import datetime
from urllib.parse import urljoin

import requests
import pdfplumber
from bs4 import BeautifulSoup


# =============================================================================
# CONFIGURATION [P3-CONFIG: À charger depuis Academic Pack]
# =============================================================================
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
REQ_TIMEOUT = 25
MAX_PDF_MB = 30
MIN_QI_CHARS = 25

# [P3-CONFIG] Sources par pays - France pour test
SEED_URLS_FRANCE = [
    "https://www.apmep.fr/Annee-2025",
    "https://www.apmep.fr/Annee-2024",
    "https://www.apmep.fr/Annee-2023",
    "https://www.apmep.fr/Annales-Terminale-Generale",
]

# =============================================================================
# TAXONOMIES [P3-CONFIG: À charger depuis Academic Pack par pays]
# =============================================================================
# CORRECTIF 1: Suppression des tokens ambigus (un, vn, wn)
# Remplacés par patterns explicites
CHAPTER_KEYWORDS = {
    "SUITES NUMÉRIQUES": {
        "suite", "suites", "arithmétique", "géométrique", "raison", "récurrence",
        "limite", "convergence", "monotone", "bornée", "terme général", "somme",
        "croissante", "décroissante", "adjacentes"
        # SUPPRIMÉ: "un", "vn", "wn" - trop ambigus
    },
    "FONCTIONS": {
        "fonction", "dérivée", "dérivation", "primitive", "intégrale", "limite",
        "continuité", "asymptote", "tangente", "extremum", "maximum", "minimum",
        "convexe", "concave", "logarithme", "exponentielle"
    },
    "PROBABILITÉS": {
        "probabilité", "aléatoire", "événement", "indépendance", "conditionnelle",
        "binomiale", "espérance", "variance", "écart-type", "loi normale", "arbre"
    },
    "GÉOMÉTRIE": {
        "vecteur", "droite", "plan", "espace", "repère", "coordonnées",
        "orthogonal", "colinéaire", "produit scalaire", "équation cartésienne"
    },
}

QUESTION_VERBS = {
    "calculer", "déterminer", "montrer", "démontrer", "justifier", "prouver",
    "étudier", "vérifier", "exprimer", "établir", "résoudre", "tracer",
    "conjecturer", "interpréter", "expliciter", "préciser", "donner", "déduire"
}

EXCLUDE_WORDS = {
    "sommaire", "édito", "éditorial", "rédaction", "abonnement", "adhésion",
    "bulletin", "revue", "publication", "copyright", "tous droits", "flux rss",
    "table des matières", "index", "préface", "avant-propos"
}

# [P3-CONFIG] Niveaux et coefficients δ
DELTA_NIVEAU = {"Terminale": 1.0, "Première": 0.8, "Seconde": 0.6}

# Transformations cognitives pour F1
COGNITIVE_TRANSFORMS = {
    "calculer": 0.3, "simplifier": 0.25, "factoriser": 0.35,
    "développer": 0.3, "substituer": 0.25,
    "dériver": 0.4, "intégrer": 0.45, "résoudre": 0.4,
    "démontrer": 0.5, "raisonner": 0.45,
    "récurrence": 0.6, "limite": 0.5, "convergence": 0.55,
    "théorème": 0.5, "optimisation": 0.7, "modélisation": 0.65
}

EPSILON_PSI = 0.1


# =============================================================================
# PATTERNS MATHÉMATIQUES STRICTS (CORRECTIF 2)
# =============================================================================
# Pattern pour u_n, u(n), v_n, etc. - plus strict que le mot "un"
SUITE_PATTERN_RE = re.compile(r'\b[uvw]\s*[_\(]\s*n\s*[\)\}]?|\b[uvw]\s*[_\(]\s*n\s*[+\-]\s*\d', re.IGNORECASE)

# Symboles mathématiques
MATH_SYMBOL_RE = re.compile(r'[=≤≥≠∞∑∫√→×÷±]|\\frac|\\sum|\\int|\d+[,\.]\d+|[a-z]\s*\([a-z]\)')

# Pattern exercice/question
EXERCISE_RE = re.compile(r'\b(?:exercice|question|partie|problème)\s*\d*\b', re.IGNORECASE)


# =============================================================================
# OUTILS TEXTE
# =============================================================================
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ]{3,}", normalize_text(text))


def jaccard_similarity(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# =============================================================================
# CORRECTIF 2: is_math_content() STRICT
# =============================================================================
def is_math_content(text: str) -> bool:
    """
    Vérifie si le texte est du contenu mathématique RÉEL.
    Condition stricte: (verbe + indice_math) OU (>=2 keywords + indice_math)
    """
    text_lower = text.lower()
    
    # Exclusion dure
    if any(excl in text_lower for excl in EXCLUDE_WORDS):
        return False
    
    # Indices forts
    has_question_verb = any(re.search(rf"\b{v}\b", text_lower) for v in QUESTION_VERBS)
    has_math_symbol = bool(MATH_SYMBOL_RE.search(text))
    has_suite_pattern = bool(SUITE_PATTERN_RE.search(text))
    has_exercise = bool(EXERCISE_RE.search(text))
    
    # Indices math (symbole OU pattern suite OU exercice)
    has_math_indicator = has_math_symbol or has_suite_pattern or has_exercise
    
    # Keywords: exiger >=2 mots-clés stricts
    all_math_keywords = set()
    for keywords in CHAPTER_KEYWORDS.values():
        all_math_keywords.update(keywords)
    
    toks = set(tokenize(text))
    kw_hits = len(toks & all_math_keywords)
    
    # Condition stricte
    return (has_question_verb and has_math_indicator) or (kw_hits >= 2 and has_math_indicator)


# =============================================================================
# CORRECTIF 5: BFS RÉCURSIF RÉEL
# =============================================================================
def scrape_pdf_links_bfs(seed_urls: List[str], limit: int, max_pages: int = 100) -> Tuple[List[str], List[dict]]:
    """
    BFS récursif réel pour collecter les PDFs de sujets.
    Retourne (pdfs, audit_log) pour traçabilité.
    """
    base = "https://www.apmep.fr"
    queue = list(dict.fromkeys(seed_urls))
    visited = set()
    pdfs = []
    audit_log = []
    
    def normalize_link(href: str) -> str:
        if href.startswith("http"):
            return href
        return urljoin(base + "/", href.lstrip("/"))
    
    while queue and len(visited) < max_pages and len(pdfs) < limit * 3:
        url = queue.pop(0).split("#")[0]
        if url in visited:
            continue
        visited.add(url)
        
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            audit_log.append({"url": url, "status": "error", "reason": str(e)})
            continue
        
        # 1) Collecter les PDFs directs
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if ".pdf" not in href.lower():
                continue
                
            pdf_url = normalize_link(href)
            fn_lower = pdf_url.lower().split("/")[-1]
            
            # Filtrer les non-sujets
            if any(x in fn_lower for x in ["bulletin", "lettre", "actualite", "pv1", "pv2"]):
                audit_log.append({"url": pdf_url, "status": "rejected", "reason": "non-sujet (bulletin/lettre)"})
                continue
            
            if pdf_url not in pdfs:
                # Priorité aux sujets (non-corrigés en premier)
                if "corrig" not in fn_lower:
                    pdfs.insert(0, pdf_url)
                else:
                    pdfs.append(pdf_url)
                audit_log.append({"url": pdf_url, "status": "accepted", "reason": "PDF sujet candidat"})
        
        # 2) Explorer les sous-pages pertinentes (BFS)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            nxt = normalize_link(href)
            nxt_lower = nxt.lower()
            
            if "apmep.fr" not in nxt_lower:
                continue
            
            # Pages pertinentes pour les sujets
            if any(k in nxt_lower for k in ["annee-", "bac-", "annales", "terminale", "sujets"]):
                nxt_clean = nxt.split("#")[0]
                if nxt_clean not in visited and nxt_clean not in queue:
                    queue.append(nxt_clean)
        
        time.sleep(0.15)  # Politesse anti-ban
    
    # Dédoublonner et limiter
    out, seen = [], set()
    for p in pdfs:
        if p not in seen:
            seen.add(p)
            out.append(p)
        if len(out) >= limit:
            break
    
    return out, audit_log


# =============================================================================
# TÉLÉCHARGEMENT PDF
# =============================================================================
def download_pdf(url: str) -> Optional[bytes]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT, stream=True)
        r.raise_for_status()
        
        cl = r.headers.get("Content-Length")
        if cl and int(cl) / (1024 * 1024) > MAX_PDF_MB:
            return None
        
        data = r.content
        return data if len(data) <= MAX_PDF_MB * 1024 * 1024 else None
    except Exception:
        return None


# =============================================================================
# EXTRACTION TEXTE PDF
# =============================================================================
def extract_pdf_text(pdf_bytes: bytes, max_pages: int = 30) -> str:
    text_parts = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i in range(min(len(pdf.pages), max_pages)):
                t = pdf.pages[i].extract_text() or ""
                if t.strip():
                    text_parts.append(t)
    except Exception:
        pass
    return "\n".join(text_parts)


# =============================================================================
# DÉTECTION CHAPITRE / NATURE / ANNÉE
# =============================================================================
def detect_chapter(text: str, matiere: str = "MATHS") -> str:
    toks = set(tokenize(text))
    
    # Vérifier aussi les patterns de suite
    has_suite_pattern = bool(SUITE_PATTERN_RE.search(text))
    
    chapters = list(CHAPTER_KEYWORDS.keys())
    best_chapter = chapters[0]
    best_score = 0
    
    for chapter in chapters:
        keywords = CHAPTER_KEYWORDS.get(chapter, set())
        score = len(toks & keywords)
        
        # Bonus pour pattern suite
        if chapter == "SUITES NUMÉRIQUES" and has_suite_pattern:
            score += 3
        
        if score > best_score:
            best_score = score
            best_chapter = chapter
    
    return best_chapter


def detect_nature(filename: str, text: str) -> str:
    combined = (filename + " " + text[:2000]).lower()
    
    if any(k in combined for k in ["bac", "baccalauréat", "métropole", "polynesie", "antilles", "asie", "amerique"]):
        return "BAC"
    if any(k in combined for k in ["concours"]):
        return "CONCOURS"
    if any(k in combined for k in ["dst", "devoir"]):
        return "DST"
    if any(k in combined for k in ["interro"]):
        return "INTERRO"
    
    return "EXAMEN"  # Neutre par défaut


# CORRECTIF 4: Ne pas inventer l'année
def detect_year(filename: str, text: str) -> Optional[int]:
    """Retourne None si année non trouvée (pas d'invention)."""
    match = re.search(r"20[12]\d", filename)
    if match:
        return int(match.group())
    
    match = re.search(r"20[12]\d", text[:1500])
    if match:
        return int(match.group())
    
    return None  # CORRECTIF: pas d'invention


# =============================================================================
# CORRECTIF 3: EXTRACTION Qi AVEC SPLIT NON-CAPTURANT
# =============================================================================
def extract_qi_from_text(text: str, chapter_filter: str = None) -> Tuple[List[str], List[dict]]:
    """
    Extrait les Qi avec audit log.
    Retourne (qi_list, audit_log).
    """
    audit_log = []
    
    # Nettoyer
    raw = text.replace("\r", "\n")
    raw = re.sub(r'A\.?P\.?M\.?E\.?P\.?', '', raw)
    raw = re.sub(r'Baccalauréat.*?sujet\s*\d*', '', raw, flags=re.IGNORECASE)
    
    # CORRECTIF 3: Patterns NON-CAPTURANTS
    question_patterns = [
        r'\n\s*(?:\d+)\.\s+',           # "1. "
        r'\n\s*(?:\d+)\)\s+',           # "1) "
        r'\n\s*(?:[a-z])\.\s+',         # "a. "
        r'\n\s*(?:[a-z])\)\s+',         # "a) "
        r'\n\s*Affirmation\s*\d+\s*:',  # "Affirmation 1:"
        r'\n\s*EXERCICE\s+\d+',         # "EXERCICE 1"
    ]
    combined_pattern = '|'.join(question_patterns)
    segments = re.split(combined_pattern, raw)
    
    candidates = []
    
    for segment in segments:
        if not segment or not segment.strip():
            continue
        
        segment = segment.strip()
        
        # Filtres de base
        if len(segment) < MIN_QI_CHARS:
            audit_log.append({"text": segment[:50], "status": "rejected", "reason": "trop court"})
            continue
        
        if len(segment) > 500:
            segment = segment[:500]
        
        # CORRECTIF 2: Validation stricte
        if not is_math_content(segment):
            audit_log.append({"text": segment[:50], "status": "rejected", "reason": "pas de contenu math"})
            continue
        
        # Filtre chapitre si demandé
        if chapter_filter:
            keywords = CHAPTER_KEYWORDS.get(chapter_filter, set())
            toks = set(tokenize(segment))
            has_keyword = len(toks & keywords) >= 1 or (chapter_filter == "SUITES NUMÉRIQUES" and SUITE_PATTERN_RE.search(segment))
            if not has_keyword:
                audit_log.append({"text": segment[:50], "status": "rejected", "reason": f"hors chapitre {chapter_filter}"})
                continue
        
        # Nettoyage final
        segment = re.sub(r'\s+', ' ', segment).strip()
        if len(segment) > 350:
            segment = segment[:350].rsplit(' ', 1)[0] + "..."
        
        candidates.append(segment)
        audit_log.append({"text": segment[:50], "status": "accepted", "reason": "Qi valide"})
    
    # Fallback par blocs si peu de résultats
    if len(candidates) < 3:
        blocks = re.split(r'\n\s*\n', raw)
        for b in blocks:
            b = b.strip()
            if len(b) < MIN_QI_CHARS or len(b) > 500:
                continue
            if is_math_content(b) and b not in candidates:
                b = re.sub(r'\s+', ' ', b).strip()
                if len(b) > 350:
                    b = b[:350].rsplit(' ', 1)[0] + "..."
                candidates.append(b)
    
    # Dédoublonnage
    seen = set()
    out = []
    for x in candidates:
        k = normalize_text(x)
        if k not in seen and len(k) > 20:
            seen.add(k)
            out.append(x)
    
    return out[:50], audit_log


# =============================================================================
# F1: Ψ_q (Poids Prédictif Purifié)
# =============================================================================
def compute_psi_q(qi_texts: List[str], niveau: str = "Terminale") -> float:
    if not qi_texts:
        return EPSILON_PSI
    
    combined = " ".join(qi_texts).lower()
    
    sum_tj = sum(w for t, w in COGNITIVE_TRANSFORMS.items() if t in combined)
    psi_brut = sum_tj + EPSILON_PSI
    delta_c = DELTA_NIVEAU.get(niveau, 1.0)
    
    return round(min(1.0, psi_brut * delta_c / 3.0), 2)


# =============================================================================
# F2: Score(q) (Sélection Granulo) - CORRECTIF 4: t_rec prudent
# =============================================================================
def compute_score_f2(n_q: int, n_total: int, t_rec: Optional[float], psi_q: float, alpha: float = 5.0) -> float:
    if n_total == 0:
        return 0.0
    
    freq_ratio = n_q / n_total
    
    # CORRECTIF 4: Si année inconnue, pénaliser (t_rec=5 ans)
    t_rec_safe = max(0.5, t_rec) if t_rec is not None else 5.0
    recency_factor = 1 + (alpha / t_rec_safe)
    
    return round(freq_ratio * recency_factor * psi_q * 100, 1)


# =============================================================================
# GÉNÉRATION ARI
# =============================================================================
def generate_ari(qi_texts: List[str], chapter: str) -> List[str]:
    combined = " ".join(qi_texts).lower()
    
    if chapter == "SUITES NUMÉRIQUES":
        if any(k in combined for k in ["géométrique", "quotient"]):
            return ["1. Exprimer u(n+1)", "2. Quotient u(n+1)/u(n)", "3. Simplifier", "4. Constante q"]
        if any(k in combined for k in ["arithmétique", "différence"]):
            return ["1. Exprimer u(n+1)", "2. Différence u(n+1)-u(n)", "3. Simplifier", "4. Constante r"]
        if any(k in combined for k in ["limite", "convergence"]):
            return ["1. Terme dominant", "2. Factorisation", "3. Limites usuelles", "4. Conclure"]
        if any(k in combined for k in ["récurrence"]):
            return ["1. Initialisation", "2. Hérédité", "3. Démontrer P(n+1)", "4. Conclure"]
    
    elif chapter == "FONCTIONS":
        if any(k in combined for k in ["dérivée"]):
            return ["1. Identifier f", "2. Dériver", "3. Simplifier f'", "4. Signe"]
    
    return ["1. Analyser", "2. Méthode", "3. Calculer", "4. Conclure"]


# =============================================================================
# GÉNÉRATION FRT
# =============================================================================
def generate_frt(qi_texts: List[str], chapter: str, triggers: List[str]) -> List[Dict]:
    combined = " ".join(qi_texts).lower()
    
    if chapter == "SUITES NUMÉRIQUES" and any(k in combined for k in ["géométrique"]):
        return [
            {"type": "usage", "title": "🔔 1. QUAND UTILISER", "text": "Prouver qu'une suite est géométrique."},
            {"type": "method", "title": "✅ 2. MÉTHODE", "text": "1. Exprimer u(n+1).\n2. Calculer u(n+1)/u(n).\n3. Simplifier.\n4. Constante q."},
            {"type": "trap", "title": "⚠️ 3. PIÈGES", "text": "Vérifier u(n) ≠ 0."},
            {"type": "conc", "title": "✍️ 4. CONCLUSION", "text": "Suite géométrique de raison q."}
        ]
    
    return [
        {"type": "usage", "title": "🔔 1. QUAND UTILISER", "text": f"Questions: {', '.join(triggers[:3]) if triggers else 'voir déclencheurs'}"},
        {"type": "method", "title": "✅ 2. MÉTHODE", "text": "1. Identifier.\n2. Appliquer.\n3. Calculer.\n4. Conclure."},
        {"type": "trap", "title": "⚠️ 3. PIÈGES", "text": "Vérifier les conditions."},
        {"type": "conc", "title": "✍️ 4. CONCLUSION", "text": "Répondre à la question."}
    ]


# =============================================================================
# EXTRACTION TRIGGERS
# =============================================================================
def extract_triggers(qi_texts: List[str]) -> List[str]:
    stopwords = {"les", "des", "une", "pour", "que", "qui", "est", "sont", "dans", "par", "sur", "avec"}
    
    bigrams = Counter()
    for qi in qi_texts:
        toks = [t for t in tokenize(qi) if t not in stopwords and len(t) >= 3]
        for i in range(len(toks) - 1):
            bigrams[f"{toks[i]} {toks[i+1]}"] += 1
    
    return [phrase for phrase, _ in bigrams.most_common(4)]


# =============================================================================
# DATACLASS
# =============================================================================
@dataclass
class QiItem:
    subject_id: str
    subject_file: str
    text: str
    chapter: str = ""
    year: Optional[int] = None


# =============================================================================
# CLUSTERING Qi → QC
# =============================================================================
def cluster_qi_to_qc(qis: List[QiItem], sim_threshold: float = 0.25) -> List[Dict]:
    if not qis:
        return []
    
    clusters = []
    
    for qi in qis:
        toks = tokenize(qi.text)
        if not toks:
            continue
        
        best_i, best_sim = None, 0.0
        for i, c in enumerate(clusters):
            sim = jaccard_similarity(toks, c["rep_tokens"])
            if sim > best_sim:
                best_sim, best_i = sim, i
        
        if best_i is not None and best_sim >= sim_threshold:
            clusters[best_i]["qis"].append(qi)
            clusters[best_i]["rep_tokens"] = list(set(clusters[best_i]["rep_tokens"]) | set(toks))
        else:
            clusters.append({"id": f"QC-{len(clusters)+1:02d}", "rep_tokens": toks, "qis": [qi]})
    
    qc_out = []
    total_qi = len(qis)
    
    for c in clusters:
        qi_texts = [q.text for q in c["qis"]]
        chapter = c["qis"][0].chapter if c["qis"] else "SUITES NUMÉRIQUES"
        
        title = min(qi_texts, key=lambda x: len(x) if len(x) > 30 else 1000)
        if len(title) > 80:
            title = title[:80].rsplit(" ", 1)[0] + "..."
        
        triggers = extract_triggers(qi_texts)
        ari = generate_ari(qi_texts, chapter)
        frt_data = generate_frt(qi_texts, chapter, triggers)
        
        n_q = len(qi_texts)
        psi_q = compute_psi_q(qi_texts, "Terminale")
        
        # CORRECTIF 4: Gestion année None
        years = [q.year for q in c["qis"] if q.year is not None]
        if years:
            max_year = max(years)
            t_rec = max(0.5, datetime.now().year - max_year)
        else:
            t_rec = None  # Année inconnue
        
        score = compute_score_f2(n_q, total_qi, t_rec, psi_q)
        
        qi_by_file = defaultdict(list)
        for q in c["qis"]:
            qi_by_file[q.subject_file].append(q.text)
        
        evidence = [{"Fichier": f, "Qi": qi_txt} for f, qlist in qi_by_file.items() for qi_txt in qlist]
        
        qc_out.append({
            "Chapitre": chapter, "QC_ID": c["id"], "FRT_ID": c["id"],
            "Titre": title, "Score": score, "n_q": n_q, "Psi": psi_q,
            "N_tot": total_qi, "t_rec": round(t_rec, 1) if t_rec else "N/A",
            "Triggers": triggers, "ARI": ari, "FRT_DATA": frt_data, "Evidence": evidence
        })
    
    qc_out.sort(key=lambda x: x["Score"], reverse=True)
    return qc_out


# =============================================================================
# FONCTION PRINCIPALE D'INGESTION
# =============================================================================
def ingest_real(urls: List[str], volume: int, matiere: str, chapter_filter: str = None, progress_callback=None):
    """
    Ingestion RÉELLE avec BFS récursif et audit complet.
    """
    import pandas as pd
    
    cols_src = ["Fichier", "Nature", "Annee", "Telechargement", "Qi_Data"]
    cols_atm = ["FRT_ID", "Qi", "File", "Year", "Chapitre"]
    
    # Déterminer les seeds (ignorer page d'accueil APMEP)
    seeds = []
    for url in urls:
        url_lower = url.lower().strip().rstrip("/")
        if url_lower in ["https://apmep.fr", "https://www.apmep.fr", "http://apmep.fr"]:
            seeds.extend(SEED_URLS_FRANCE)  # [P3-CONFIG]
        else:
            seeds.append(url)
    
    if not seeds:
        seeds = SEED_URLS_FRANCE
    
    # CORRECTIF 5: BFS récursif réel
    pdf_links, scrape_audit = scrape_pdf_links_bfs(seeds, limit=volume * 2)
    
    if not pdf_links:
        return pd.DataFrame(columns=cols_src), pd.DataFrame(columns=cols_atm)
    
    subjects = []
    atoms = []
    processed = 0
    
    for idx, pdf_url in enumerate(pdf_links):
        if processed >= volume:
            break
        
        if progress_callback:
            progress_callback((idx + 1) / len(pdf_links))
        
        pdf_bytes = download_pdf(pdf_url)
        if not pdf_bytes:
            continue
        
        text = extract_pdf_text(pdf_bytes)
        if not text.strip() or len(text) < 200:
            continue
        
        # Validation globale du PDF
        if not is_math_content(text[:3000]):
            continue
        
        filename = pdf_url.split("/")[-1].split("?")[0]
        nature = detect_nature(filename, text)
        year = detect_year(filename, text)
        
        qi_texts, qi_audit = extract_qi_from_text(text, chapter_filter)
        
        if not qi_texts:
            continue
        
        qi_data = []
        for qi_txt in qi_texts:
            chapter = detect_chapter(qi_txt, matiere) if not chapter_filter else chapter_filter
            atoms.append({"FRT_ID": None, "Qi": qi_txt, "File": filename, "Year": year, "Chapitre": chapter})
            qi_data.append({"Qi": qi_txt, "FRT_ID": None})
        
        subjects.append({
            "Fichier": filename, "Nature": nature, "Annee": year if year else "N/A",
            "Telechargement": pdf_url, "Qi_Data": qi_data
        })
        
        processed += 1
    
    return (
        pd.DataFrame(subjects) if subjects else pd.DataFrame(columns=cols_src),
        pd.DataFrame(atoms) if atoms else pd.DataFrame(columns=cols_atm)
    )


# =============================================================================
# CALCUL QC
# =============================================================================
def compute_qc_real(df_atoms) -> 'pd.DataFrame':
    import pandas as pd
    
    if df_atoms.empty:
        return pd.DataFrame()
    
    all_qis = [
        QiItem(f"S{idx:04d}", row.get("File", ""), row.get("Qi", ""), row.get("Chapitre", ""), row.get("Year"))
        for idx, row in df_atoms.iterrows()
    ]
    
    qc_list = cluster_qi_to_qc(all_qis)
    return pd.DataFrame(qc_list) if qc_list else pd.DataFrame()


# =============================================================================
# SATURATION
# =============================================================================
def compute_saturation_real(df_atoms) -> 'pd.DataFrame':
    import pandas as pd
    
    if df_atoms.empty:
        return pd.DataFrame(columns=["Sujets (N)", "QC Découvertes", "Saturation (%)"])
    
    files = df_atoms["File"].unique().tolist()
    data_points = []
    cumulative = []
    
    for i, f in enumerate(files):
        cumulative.extend(df_atoms[df_atoms["File"] == f].to_dict('records'))
        qis = [QiItem(f"S{j}", r.get("File", ""), r.get("Qi", ""), r.get("Chapitre", ""), r.get("Year")) for j, r in enumerate(cumulative)]
        n_qc = len(cluster_qi_to_qc(qis))
        data_points.append({"Sujets (N)": i + 1, "QC Découvertes": n_qc, "Saturation (%)": 0})
    
    if data_points:
        max_qc = max(d["QC Découvertes"] for d in data_points)
        for d in data_points:
            d["Saturation (%)"] = round((d["QC Découvertes"] / max(max_qc, 1)) * 100, 1)
    
    return pd.DataFrame(data_points)


# =============================================================================
# AUDIT
# =============================================================================
def audit_internal_real(subject_qis: List[Dict], qc_df) -> List[Dict]:
    if qc_df.empty or not subject_qis:
        return []
    
    results = []
    qc_list = qc_df.to_dict('records')
    
    for qi_item in subject_qis:
        qi_toks = tokenize(qi_item.get("Qi", ""))
        best_qc, best_sim = None, 0.0
        
        for qc in qc_list:
            for ev in qc.get("Evidence", []):
                sim = jaccard_similarity(qi_toks, tokenize(ev.get("Qi", "")))
                if sim > best_sim:
                    best_sim, best_qc = sim, qc
        
        qi_short = qi_item.get("Qi", "")[:80] + "..." if len(qi_item.get("Qi", "")) > 80 else qi_item.get("Qi", "")
        results.append({
            "Qi": qi_short,
            "Statut": "✅ MATCH" if best_sim >= 0.25 else "❌ GAP",
            "QC": best_qc["QC_ID"] if best_qc and best_sim >= 0.25 else None
        })
    
    return results


def audit_external_real(pdf_bytes: bytes, qc_df, chapter_filter: str = None) -> Tuple[float, List[Dict]]:
    text = extract_pdf_text(pdf_bytes)
    qi_texts, _ = extract_qi_from_text(text, chapter_filter)
    
    if not qi_texts or qc_df.empty:
        return 0.0, []
    
    qc_list = qc_df.to_dict('records')
    results, matched = [], 0
    
    for qi_text in qi_texts:
        qi_toks = tokenize(qi_text)
        best_qc, best_sim = None, 0.0
        
        for qc in qc_list:
            for ev in qc.get("Evidence", []):
                sim = jaccard_similarity(qi_toks, tokenize(ev.get("Qi", "")))
                if sim > best_sim:
                    best_sim, best_qc = sim, qc
        
        if best_sim >= 0.20:
            matched += 1
        
        results.append({
            "Qi": qi_text[:80] + "..." if len(qi_text) > 80 else qi_text,
            "Statut": "✅ MATCH" if best_sim >= 0.20 else "❌ GAP",
            "QC": best_qc["QC_ID"] if best_qc and best_sim >= 0.20 else None
        })
    
    return round((matched / len(qi_texts)) * 100 if qi_texts else 0, 1), results

# =============================================================================
# VERSION MARKER - V3.1 POST-AUDIT GPT - 2024-12-24
# Si vous voyez PV164.pdf, ce fichier N'EST PAS déployé correctement!
# =============================================================================
VERSION = "V3.1-AUDIT-GPT-20241224"
