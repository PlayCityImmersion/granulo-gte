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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
REQ_TIMEOUT = 20
MAX_PDF_MB = 30
MIN_QI_CHARS = 25

# Parallélisation
MAX_WORKERS = 10  # Threads simultanés pour téléchargement

# [P3-CONFIG] Sources par pays - France pour test
SEED_URLS_FRANCE = [
    "https://www.apmep.fr/Annee-2025",
    "https://www.apmep.fr/Annee-2024",
    "https://www.apmep.fr/Annee-2023",
    "https://www.apmep.fr/Annales-Terminale-Generale",
]

# Session HTTP réutilisable (keep-alive)
_session = None

def get_session():
    """Session HTTP avec keep-alive pour performance."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": UA})
    return _session

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
def scrape_pdf_links_bfs(seed_urls: List[str], limit: int, max_pages: int = 100) -> Tuple[List[Dict], List[dict]]:
    """
    BFS récursif réel pour collecter les PDFs de sujets ET leurs corrigés.
    Retourne (sujets_avec_corriges, audit_log).
    
    Chaque élément de sujets_avec_corriges est un dict:
    {"sujet_url": "...", "corrige_url": "..." ou None}
    """
    base = "https://www.apmep.fr"
    queue = list(dict.fromkeys(seed_urls))
    visited = set()
    
    # Collecter séparément sujets et corrigés
    sujets = []  # URLs des sujets
    corriges = []  # URLs des corrigés
    audit_log = []
    
    def normalize_link(href: str) -> str:
        if href.startswith("http"):
            return href
        return urljoin(base + "/", href.lstrip("/"))
    
    def get_base_name(url: str) -> str:
        """Extrait le nom de base pour matcher sujet/corrigé."""
        fn = url.split("/")[-1].lower()
        # Supprimer les variantes de "corrigé"
        fn = re.sub(r'corr?ig[eé]?_?', '', fn)
        fn = re.sub(r'_corr?_?', '', fn)
        # Supprimer extensions et numéros de version
        fn = re.sub(r'_?\d*\.pdf$', '', fn)
        fn = re.sub(r'_[a-z]{2,3}$', '', fn)  # _DV, _FK, etc.
        return fn
    
    while queue and len(visited) < max_pages and (len(sujets) + len(corriges)) < limit * 4:
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
        
        # Collecter les PDFs
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if ".pdf" not in href.lower():
                continue
                
            pdf_url = normalize_link(href)
            fn_lower = pdf_url.lower().split("/")[-1]
            
            # Filtrer les non-sujets (bulletins, lettres)
            if any(x in fn_lower for x in ["bulletin", "lettre", "actualite", "pv1", "pv2"]):
                continue
            
            # Classer en sujet ou corrigé
            is_corrige = any(x in fn_lower for x in ["corrig", "corr_", "_corr"])
            
            if is_corrige:
                if pdf_url not in corriges:
                    corriges.append(pdf_url)
                    audit_log.append({"url": pdf_url, "status": "corrige", "reason": "Corrigé détecté"})
            else:
                if pdf_url not in sujets:
                    sujets.append(pdf_url)
                    audit_log.append({"url": pdf_url, "status": "sujet", "reason": "Sujet détecté"})
        
        # Explorer les sous-pages (BFS)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            nxt = normalize_link(href)
            nxt_lower = nxt.lower()
            
            if "apmep.fr" not in nxt_lower:
                continue
            
            if any(k in nxt_lower for k in ["annee-", "bac-", "annales", "terminale", "sujets"]):
                nxt_clean = nxt.split("#")[0]
                if nxt_clean not in visited and nxt_clean not in queue:
                    queue.append(nxt_clean)
        
        time.sleep(0.15)
    
    # Matcher sujets avec leurs corrigés
    result = []
    for sujet_url in sujets:
        sujet_base = get_base_name(sujet_url)
        
        # Chercher le corrigé correspondant
        corrige_match = None
        best_score = 0
        
        for corrige_url in corriges:
            corrige_base = get_base_name(corrige_url)
            
            # Score de similarité simple
            if sujet_base in corrige_base or corrige_base in sujet_base:
                score = len(set(sujet_base) & set(corrige_base))
                if score > best_score:
                    best_score = score
                    corrige_match = corrige_url
        
        result.append({
            "sujet_url": sujet_url,
            "corrige_url": corrige_match
        })
        
        if len(result) >= limit:
            break
    
    return result, audit_log


# =============================================================================
# TÉLÉCHARGEMENT PDF (AVEC SESSION KEEP-ALIVE)
# =============================================================================
def download_pdf(url: str) -> Optional[bytes]:
    """Télécharge un PDF avec session keep-alive."""
    try:
        session = get_session()
        r = session.get(url, timeout=REQ_TIMEOUT, stream=True)
        r.raise_for_status()
        
        cl = r.headers.get("Content-Length")
        if cl and int(cl) / (1024 * 1024) > MAX_PDF_MB:
            return None
        
        data = r.content
        return data if len(data) <= MAX_PDF_MB * 1024 * 1024 else None
    except Exception:
        return None


def download_and_process_subject(item: Dict, chapter_filter: str, matiere: str) -> Optional[Dict]:
    """
    Télécharge et traite UN sujet (pour parallélisation).
    Retourne un dict avec les données ou None si échec.
    """
    sujet_url = item["sujet_url"]
    corrige_url = item["corrige_url"]
    
    # Télécharger le sujet
    pdf_bytes = download_pdf(sujet_url)
    if not pdf_bytes:
        return None
    
    # Extraction texte (optimisée : 8 pages max d'abord)
    text = extract_pdf_text(pdf_bytes, max_pages=8)
    if not text.strip() or len(text) < 150:
        return None
    
    filename = sujet_url.split("/")[-1].split("?")[0]
    
    # Pour les PDFs BAC identifiés par nom, être plus permissif
    is_bac_by_name = any(k in filename.lower() for k in [
        "bac", "metropole", "polynesie", "asie", "amerique", "spe_", "terminale", "etranger"
    ])
    
    # Validation du contenu
    if not is_bac_by_name and not is_math_content(text[:2000]):
        return None
    
    nature = detect_nature(filename, text)
    year = detect_year(filename, text)
    
    # Extraction Qi
    qi_texts, _ = extract_qi_from_text(text, chapter_filter)
    
    # Si pas assez de Qi, parser plus de pages
    if len(qi_texts) < 3 and is_bac_by_name:
        text_full = extract_pdf_text(pdf_bytes, max_pages=20)
        qi_texts, _ = extract_qi_from_text(text_full, chapter_filter)
    
    # Si toujours pas de Qi avec filtre, essayer sans
    if not qi_texts and is_bac_by_name and chapter_filter:
        qi_texts, _ = extract_qi_from_text(text, None)
    
    if not qi_texts:
        return None
    
    # Construire les atomes
    atoms = []
    qi_data = []
    for qi_txt in qi_texts:
        chapter = detect_chapter(qi_txt, matiere) if not chapter_filter else chapter_filter
        atoms.append({
            "FRT_ID": None, 
            "Qi": qi_txt, 
            "File": filename, 
            "Year": year, 
            "Chapitre": chapter
        })
        qi_data.append({"Qi": qi_txt, "FRT_ID": None})
    
    return {
        "subject": {
            "Fichier": filename,
            "Nature": nature,
            "Annee": year if year else "N/A",
            "Telechargement": sujet_url,
            "Corrige": corrige_url if corrige_url else "Non trouvé",
            "Qi_Data": qi_data
        },
        "atoms": atoms
    }


# =============================================================================
# EXTRACTION TEXTE PDF (AMÉLIORÉE - GESTION MOTS COLLÉS)
# =============================================================================
def extract_pdf_text(pdf_bytes: bytes, max_pages: int = 30) -> str:
    text_parts = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i in range(min(len(pdf.pages), max_pages)):
                page = pdf.pages[i]
                
                # Méthode 1: extraction standard
                t = page.extract_text() or ""
                
                # Méthode 2: si texte collé, essayer avec layout
                if t and len(t) > 100:
                    # Détecter mots collés (mots très longs sans espaces)
                    words = t.split()
                    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
                    
                    if avg_word_len > 15:  # Mots anormalement longs = collés
                        # Essayer extraction avec paramètres différents
                        t2 = page.extract_text(x_tolerance=3, y_tolerance=3) or t
                        if t2:
                            t = t2
                        
                        # Ajouter espaces avant majuscules (heuristique)
                        t = re.sub(r'([a-zéèêëàâùûîïôç])([A-ZÉÈÊËÀÂÙÛÎÏÔÇ])', r'\1 \2', t)
                        # Ajouter espaces autour des chiffres isolés
                        t = re.sub(r'(\d)([A-Za-zéèêëàâùûîïôç])', r'\1 \2', t)
                        t = re.sub(r'([A-Za-zéèêëàâùûîïôç])(\d)', r'\1 \2', t)
                
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
# F1: Ψ_q (Poids Prédictif Purifié) - AVEC DÉTAIL DES COMPOSANTES
# =============================================================================
def compute_psi_q_detailed(qi_texts: List[str], niveau: str = "Terminale") -> Dict:
    """
    Calcule Ψ_q avec toutes les composantes pour affichage.
    Retourne: {psi, sum_tj, delta_c, transforms_found}
    """
    if not qi_texts:
        return {"psi": EPSILON_PSI, "sum_tj": 0, "delta_c": 1.0, "transforms_found": []}
    
    combined = " ".join(qi_texts).lower()
    
    # Calculer Σ T_j (somme des transformations cognitives détectées)
    transforms_found = []
    sum_tj = 0.0
    for transform, weight in COGNITIVE_TRANSFORMS.items():
        if transform in combined:
            sum_tj += weight
            transforms_found.append(f"{transform}({weight})")
    
    # Ψ_brut = Σ T_j + ε
    psi_brut = sum_tj + EPSILON_PSI
    
    # δ_c = coefficient de niveau
    delta_c = DELTA_NIVEAU.get(niveau, 1.0)
    
    # Ψ ajusté et normalisé
    psi_ajuste = psi_brut * delta_c
    psi_normalise = min(1.0, psi_ajuste / 3.0)
    
    return {
        "psi": round(psi_normalise, 2),
        "sum_tj": round(sum_tj, 2),
        "delta_c": delta_c,
        "transforms_found": transforms_found
    }


def compute_psi_q(qi_texts: List[str], niveau: str = "Terminale") -> float:
    """Version simple pour compatibilité."""
    return compute_psi_q_detailed(qi_texts, niveau)["psi"]


# =============================================================================
# F2: Score(q) (Sélection Granulo) - AVEC DÉTAIL DES COMPOSANTES
# =============================================================================
def compute_score_f2_detailed(n_q: int, n_total: int, t_rec: Optional[float], psi_q: float, 
                               alpha: float = 5.0, redundancy_penalty: float = 1.0) -> Dict:
    """
    Calcule Score(q) avec toutes les composantes pour affichage.
    
    Formule A2: Score(q) = (n_q / N_tot) × (1 + α/t_réc) × Ψ_q × R_penalty × 100
    
    Retourne: {score, freq_ratio, recency_factor, alpha, t_rec, redundancy}
    """
    if n_total == 0:
        return {"score": 0, "freq_ratio": 0, "recency_factor": 0, "alpha": alpha, 
                "t_rec": t_rec, "redundancy": redundancy_penalty}
    
    # Fréquence relative
    freq_ratio = n_q / n_total
    
    # Facteur de récence: (1 + α/t_réc)
    t_rec_safe = max(0.5, t_rec) if t_rec is not None else 5.0
    recency_factor = 1 + (alpha / t_rec_safe)
    
    # Score final
    score = freq_ratio * recency_factor * psi_q * redundancy_penalty * 100
    
    return {
        "score": round(score, 1),
        "freq_ratio": round(freq_ratio, 4),
        "recency_factor": round(recency_factor, 2),
        "alpha": alpha,
        "t_rec": t_rec_safe if t_rec is not None else None,
        "redundancy": redundancy_penalty
    }


def compute_score_f2(n_q: int, n_total: int, t_rec: Optional[float], psi_q: float, alpha: float = 5.0) -> float:
    """Version simple pour compatibilité."""
    return compute_score_f2_detailed(n_q, n_total, t_rec, psi_q, alpha)["score"]


# =============================================================================
# EXTRACTION TRIGGERS (ABSTRAITS)
# =============================================================================
def extract_triggers(qi_texts: List[str], qc_type: Optional[str] = None) -> List[str]:
    """
    Extrait les déclencheurs abstraits.
    Si un template existe, utilise ses triggers prédéfinis.
    Sinon, extrait des n-grams abstraits.
    """
    # Si template reconnu, utiliser ses triggers
    if qc_type and qc_type in QC_TEMPLATES:
        return QC_TEMPLATES[qc_type]["triggers"][:5]
    
    # Sinon, extraire des bigrams abstraits
    stopwords = {"les", "des", "une", "pour", "que", "qui", "est", "sont", "dans", "par", "sur", "avec", 
                 "sur", "que", "cet", "cette", "son", "ses", "leur"}
    
    bigrams = Counter()
    for qi in qi_texts:
        # Abstraire d'abord
        abstract = abstract_qi_text(qi)
        toks = [t for t in tokenize(abstract) if t not in stopwords and len(t) >= 3]
        for i in range(len(toks) - 1):
            bigram = f"{toks[i]} {toks[i+1]}"
            # Éviter les bigrams avec des nombres/variables spécifiques
            if not re.search(r'\b[nkab]\b', bigram):
                bigrams[bigram] += 1
    
    return [phrase for phrase, _ in bigrams.most_common(5)]


# =============================================================================
# GÉNÉRATION ARI (TEMPLATE-BASED)
# =============================================================================
def generate_ari(qi_texts: List[str], chapter: str, qc_type: Optional[str] = None) -> List[str]:
    """
    Génère un ARI basé sur le template ou la détection de mots-clés.
    """
    # Si template reconnu, utiliser son ARI
    if qc_type and qc_type in QC_TEMPLATES:
        return QC_TEMPLATES[qc_type]["ari"]
    
    # Sinon, fallback sur détection par mots-clés
    combined = " ".join(qi_texts).lower()
    
    if chapter == "SUITES NUMÉRIQUES":
        if any(k in combined for k in ["géométrique", "quotient"]):
            return QC_TEMPLATES["SUITE_GEOMETRIQUE"]["ari"]
        if any(k in combined for k in ["arithmétique", "différence"]):
            return QC_TEMPLATES["SUITE_ARITHMETIQUE"]["ari"]
        if any(k in combined for k in ["limite", "convergence", "tend vers"]):
            return QC_TEMPLATES["LIMITE_SUITE"]["ari"]
        if any(k in combined for k in ["récurrence"]):
            return QC_TEMPLATES["RECURRENCE"]["ari"]
    
    if chapter == "FONCTIONS":
        if any(k in combined for k in ["unique solution", "admet une", "équation"]):
            return QC_TEMPLATES["TVI_UNIQUE"]["ari"]
        if any(k in combined for k in ["dérivée", "variations", "signe"]):
            return QC_TEMPLATES["DERIVEE_SIGNE"]["ari"]
    
    # ARI générique
    return [
        "1. Identifier le type de problème",
        "2. Rappeler les outils/théorèmes nécessaires",
        "3. Appliquer la méthode appropriée",
        "4. Conclure en répondant à la question"
    ]


# =============================================================================
# GÉNÉRATION FRT (TEMPLATE-BASED)
# =============================================================================
def generate_frt(qi_texts: List[str], chapter: str, triggers: List[str], qc_type: Optional[str] = None) -> List[Dict]:
    """
    Génère une FRT basée sur le template ou la détection de mots-clés.
    """
    # Si template reconnu, utiliser sa FRT
    if qc_type and qc_type in QC_TEMPLATES:
        return QC_TEMPLATES[qc_type]["frt"]
    
    # Sinon, fallback sur détection
    combined = " ".join(qi_texts).lower()
    
    if any(k in combined for k in ["unique solution", "admet une"]):
        return QC_TEMPLATES["TVI_UNIQUE"]["frt"]
    
    if chapter == "SUITES NUMÉRIQUES":
        if any(k in combined for k in ["géométrique"]):
            return QC_TEMPLATES["SUITE_GEOMETRIQUE"]["frt"]
        if any(k in combined for k in ["arithmétique"]):
            return QC_TEMPLATES["SUITE_ARITHMETIQUE"]["frt"]
        if any(k in combined for k in ["récurrence"]):
            return QC_TEMPLATES["RECURRENCE"]["frt"]
        if any(k in combined for k in ["limite"]):
            return QC_TEMPLATES["LIMITE_SUITE"]["frt"]
    
    if chapter == "FONCTIONS":
        if any(k in combined for k in ["dérivée", "variations"]):
            return QC_TEMPLATES["DERIVEE_SIGNE"]["frt"]
    
    # FRT générique
    return [
        {"type": "usage", "title": "🔔 1. QUAND UTILISER", "text": f"Questions contenant: {', '.join(triggers[:3]) if triggers else 'voir déclencheurs'}"},
        {"type": "method", "title": "✅ 2. MÉTHODE", "text": "1. Identifier le problème\n2. Appliquer les outils\n3. Calculer\n4. Conclure"},
        {"type": "trap", "title": "⚠️ 3. PIÈGES", "text": "Vérifier les hypothèses et conditions d'application"},
        {"type": "conc", "title": "✍️ 4. CONCLUSION", "text": "Répondre précisément à la question posée."}
    ]


# =============================================================================
# ABSTRACTION QC (AXIOME SMAXIA: Instance → Classe)
# =============================================================================
# Une QC est une CLASSE abstraite, pas une instance.
# "Démontrer que f(t)=0,7 sur [0;6]" → "Démontrer l'existence et l'unicité d'une solution f(x)=k sur un intervalle"

# Patterns d'abstraction: valeurs concrètes → variables génériques
ABSTRACTION_PATTERNS = [
    # Intervalles AVANT les nombres isolés
    (r'\[\s*-?\d+[,\.]?\d*\s*[;,]\s*-?\d+[,\.]?\d*\s*\]', '[a;b]'),
    (r'\[\s*-?\d+[,\.]?\d*\s*[;,]\s*\+?∞\s*\[', '[a;+∞['),
    (r'\]\s*-∞\s*[;,]\s*-?\d+[,\.]?\d*\s*\]', ']-∞;b]'),
    
    # Nombres décimaux et fractions
    (r'\b\d+[,\.]\d+\b', 'k'),           # 0,7 → k
    
    # Fonctions spécifiques → génériques (avant les nombres)
    (r'\bf\s*\(\s*[txnab]\s*\)', 'f(x)'),
    (r'\bg\s*\(\s*[txnab]\s*\)', 'f(x)'),
    (r'\bh\s*\(\s*[txnab]\s*\)', 'f(x)'),
    
    # Suites spécifiques → génériques
    (r'\bu\s*[_\(]\s*n\s*[\)\}]?', 'u_n'),
    (r'\bv\s*[_\(]\s*n\s*[\)\}]?', 'u_n'),
    (r'\bw\s*[_\(]\s*n\s*[\)\}]?', 'u_n'),
    
    # Années, sessions → supprimées
    (r'\b20\d{2}\b', ''),
]

# Templates de QC abstraites par type de problème
QC_TEMPLATES = {
    "TVI_UNIQUE": {
        "pattern": r"(montrer|démontrer|prouver).*équation.*admet.*unique.*solution",
        "title": "Démontrer l'existence et l'unicité d'une solution f(x)=k sur un intervalle (TVI)",
        "triggers": ["admet une unique solution", "montrer que l'équation", "unique réel", "solution unique"],
        "ari": [
            "1. Vérifier la continuité de f sur [a;b]",
            "2. Étudier la monotonie stricte (via f')",
            "3. Calculer f(a) et f(b) (images des bornes)",
            "4. Vérifier que k ∈ [f(a);f(b)]",
            "5. Appliquer le corollaire du TVI",
            "6. Conclure sur l'unicité de α"
        ],
        "frt": [
            {"type": "usage", "title": "🔔 1. QUAND UTILISER", "text": "Dès que l'énoncé contient: 'Montrer que... admet une unique solution'"},
            {"type": "method", "title": "✅ 2. MÉTHODE", "text": "• f est continue sur [a;b]\n• f est strictement monotone (tableau de variations)\n• Calcul: f(a)=... et f(b)=...\n• Or k ∈ [f(a);f(b)]\n• D'après le corollaire du TVI, l'équation f(x)=k admet une unique solution"},
            {"type": "trap", "title": "⚠️ 3. PIÈGES", "text": "• Oublier 'continue'\n• Oublier 'strictement' monotone\n• Confondre f(x)=k et f'(x)"},
            {"type": "conc", "title": "✍️ 4. CONCLUSION", "text": "L'équation admet une unique solution α sur [a;b]."}
        ]
    },
    "SUITE_GEOMETRIQUE": {
        "pattern": r"(montrer|démontrer|prouver).*suite.*géométrique",
        "title": "Démontrer qu'une suite est géométrique de raison q",
        "triggers": ["suite géométrique", "raison q", "u(n+1)/u(n)", "quotient constant"],
        "ari": [
            "1. Exprimer u(n+1) en fonction de n",
            "2. Calculer le quotient u(n+1)/u(n)",
            "3. Simplifier l'expression",
            "4. Montrer que le quotient est constant = q"
        ],
        "frt": [
            {"type": "usage", "title": "🔔 1. QUAND UTILISER", "text": "Prouver qu'une suite est géométrique"},
            {"type": "method", "title": "✅ 2. MÉTHODE", "text": "• Exprimer u(n+1)\n• Calculer u(n+1)/u(n)\n• Simplifier\n• Montrer = constante q"},
            {"type": "trap", "title": "⚠️ 3. PIÈGES", "text": "• Vérifier u(n) ≠ 0 pour tout n\n• Ne pas confondre raison et premier terme"},
            {"type": "conc", "title": "✍️ 4. CONCLUSION", "text": "(u_n) est géométrique de raison q et de premier terme u_0."}
        ]
    },
    "SUITE_ARITHMETIQUE": {
        "pattern": r"(montrer|démontrer|prouver).*suite.*arithmétique",
        "title": "Démontrer qu'une suite est arithmétique de raison r",
        "triggers": ["suite arithmétique", "raison r", "u(n+1)-u(n)", "différence constante"],
        "ari": [
            "1. Exprimer u(n+1) en fonction de n",
            "2. Calculer la différence u(n+1)-u(n)",
            "3. Simplifier l'expression",
            "4. Montrer que la différence est constante = r"
        ],
        "frt": [
            {"type": "usage", "title": "🔔 1. QUAND UTILISER", "text": "Prouver qu'une suite est arithmétique"},
            {"type": "method", "title": "✅ 2. MÉTHODE", "text": "• Exprimer u(n+1)\n• Calculer u(n+1)-u(n)\n• Simplifier\n• Montrer = constante r"},
            {"type": "trap", "title": "⚠️ 3. PIÈGES", "text": "• Ne pas confondre avec géométrique\n• Bien identifier le premier terme"},
            {"type": "conc", "title": "✍️ 4. CONCLUSION", "text": "(u_n) est arithmétique de raison r et de premier terme u_0."}
        ]
    },
    "RECURRENCE": {
        "pattern": r"(montrer|démontrer|prouver).*récurrence|par récurrence",
        "title": "Démontrer une propriété par récurrence",
        "triggers": ["par récurrence", "pour tout n", "démontrer que pour tout", "P(n)"],
        "ari": [
            "1. INITIALISATION: Vérifier P(n_0)",
            "2. HÉRÉDITÉ: Supposer P(n) vraie",
            "3. Démontrer P(n+1) à partir de P(n)",
            "4. CONCLURE par le principe de récurrence"
        ],
        "frt": [
            {"type": "usage", "title": "🔔 1. QUAND UTILISER", "text": "Prouver une propriété 'pour tout entier n ≥ n_0'"},
            {"type": "method", "title": "✅ 2. MÉTHODE", "text": "• Initialisation: P(n_0) vraie? Vérification\n• Hérédité: Soit n≥n_0, supposons P(n) vraie\n• Montrons P(n+1): [développement]\n• Donc P(n+1) vraie"},
            {"type": "trap", "title": "⚠️ 3. PIÈGES", "text": "• Oublier l'initialisation\n• Ne pas écrire 'supposons P(n) vraie'\n• Oublier la conclusion"},
            {"type": "conc", "title": "✍️ 4. CONCLUSION", "text": "D'après le principe de récurrence, P(n) est vraie pour tout n ≥ n_0."}
        ]
    },
    "LIMITE_SUITE": {
        "pattern": r"(calculer|déterminer).*limite.*suite|limite.*tend.*infini",
        "title": "Calculer la limite d'une suite",
        "triggers": ["limite de la suite", "n tend vers +∞", "convergence", "lim u_n"],
        "ari": [
            "1. Identifier la forme de u_n",
            "2. Factoriser par le terme dominant",
            "3. Appliquer les limites usuelles",
            "4. Conclure"
        ],
        "frt": [
            {"type": "usage", "title": "🔔 1. QUAND UTILISER", "text": "Calculer une limite quand n → +∞"},
            {"type": "method", "title": "✅ 2. MÉTHODE", "text": "• Identifier forme (quotient, exponentielle...)\n• Factoriser par terme dominant\n• Appliquer théorèmes (croissances comparées...)\n• Conclure"},
            {"type": "trap", "title": "⚠️ 3. PIÈGES", "text": "• Formes indéterminées: ∞-∞, 0×∞, ∞/∞\n• Ne pas oublier les croissances comparées"},
            {"type": "conc", "title": "✍️ 4. CONCLUSION", "text": "lim(n→+∞) u_n = L (ou +∞ ou -∞)."}
        ]
    },
    "DERIVEE_SIGNE": {
        "pattern": r"(étudier|déterminer).*signe.*dérivée|variations.*fonction",
        "title": "Étudier le signe de la dérivée et les variations",
        "triggers": ["signe de f'", "tableau de variations", "croissante décroissante", "extremum"],
        "ari": [
            "1. Calculer f'(x)",
            "2. Résoudre f'(x) = 0",
            "3. Étudier le signe de f'(x)",
            "4. Dresser le tableau de variations"
        ],
        "frt": [
            {"type": "usage", "title": "🔔 1. QUAND UTILISER", "text": "Étudier les variations d'une fonction"},
            {"type": "method", "title": "✅ 2. MÉTHODE", "text": "• Calculer f'(x)\n• Résoudre f'(x)=0\n• Tableau de signes de f'\n• En déduire les variations de f"},
            {"type": "trap", "title": "⚠️ 3. PIÈGES", "text": "• Erreurs de calcul de dérivée\n• Oublier le domaine de définition\n• Confondre f et f'"},
            {"type": "conc", "title": "✍️ 4. CONCLUSION", "text": "f est croissante sur... et décroissante sur..."}
        ]
    }
}


def abstract_qi_text(text: str) -> str:
    """
    Abstrait un texte Qi en remplaçant les valeurs concrètes par des variables génériques.
    "f(t)=0,7 sur [0;6]" → "f(x)=k sur [a;b]"
    """
    result = text
    for pattern_tuple in ABSTRACTION_PATTERNS:
        if len(pattern_tuple) == 2:
            pattern, replacement = pattern_tuple
            flags = 0
        else:
            pattern, replacement, flags = pattern_tuple
        result = re.sub(pattern, replacement, result, flags=flags)
    
    # Nettoyer les espaces multiples
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def detect_qc_type(qi_texts: List[str]) -> Optional[str]:
    """
    Détecte le type de QC à partir des Qi pour appliquer le bon template.
    Retourne la clé du template ou None.
    """
    combined = " ".join(qi_texts).lower()
    
    for qc_type, template in QC_TEMPLATES.items():
        if re.search(template["pattern"], combined, re.IGNORECASE):
            return qc_type
    
    return None


def generate_abstract_title(qi_texts: List[str], qc_type: Optional[str]) -> str:
    """
    Génère un titre de QC abstrait (classe, pas instance).
    """
    # Si on a un template reconnu, utiliser son titre
    if qc_type and qc_type in QC_TEMPLATES:
        return QC_TEMPLATES[qc_type]["title"]
    
    # Sinon, abstraire la Qi la plus courte
    if not qi_texts:
        return "Question type non identifiée"
    
    # Prendre la Qi la plus représentative (ni trop courte ni trop longue)
    candidates = [q for q in qi_texts if 40 < len(q) < 200]
    if not candidates:
        candidates = qi_texts
    
    best = min(candidates, key=len)
    
    # Abstraire
    abstract = abstract_qi_text(best)
    
    # Tronquer si trop long
    if len(abstract) > 100:
        abstract = abstract[:100].rsplit(' ', 1)[0] + "..."
    
    return abstract


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
# CLUSTERING Qi → QC (AVEC ABSTRACTION SMAXIA)
# =============================================================================
def cluster_qi_to_qc(qis: List[QiItem], sim_threshold: float = 0.25) -> List[Dict]:
    """
    Clustering des Qi en QC avec ABSTRACTION SMAXIA.
    
    AXIOME: Une QC est une CLASSE, pas une instance.
    - Le titre ne contient jamais de valeurs spécifiques
    - Les Qi (instances) sont conservées comme preuves
    """
    if not qis:
        return []
    
    clusters = []
    ALPHA = 5.0
    
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
        
        # ÉTAPE CLÉ: Détecter le type de QC pour appliquer le bon template
        qc_type = detect_qc_type(qi_texts)
        
        # TITRE ABSTRAIT (jamais de valeurs concrètes)
        title = generate_abstract_title(qi_texts, qc_type)
        
        # Déclencheurs, ARI, FRT basés sur le template
        triggers = extract_triggers(qi_texts, qc_type)
        ari = generate_ari(qi_texts, chapter, qc_type)
        frt_data = generate_frt(qi_texts, chapter, triggers, qc_type)
        
        n_q = len(qi_texts)
        
        # F1: Calcul détaillé de Ψ_q
        psi_details = compute_psi_q_detailed(qi_texts, "Terminale")
        psi_q = psi_details["psi"]
        sum_tj = psi_details["sum_tj"]
        
        # Calcul de t_réc
        years = [q.year for q in c["qis"] if q.year is not None]
        if years:
            max_year = max(years)
            t_rec = max(0.5, datetime.now().year - max_year)
        else:
            t_rec = None
        
        # F2: Score
        score_details = compute_score_f2_detailed(n_q, total_qi, t_rec, psi_q, ALPHA)
        score = score_details["score"]
        
        # Organisation des Qi par fichier source (PREUVES)
        qi_by_file = defaultdict(list)
        for q in c["qis"]:
            qi_by_file[q.subject_file].append({
                "text": q.text,
                "year": q.year
            })
        
        # Evidence structurée par sujet
        evidence_by_subject = []
        for f, qi_list in qi_by_file.items():
            evidence_by_subject.append({
                "Fichier": f,
                "Qis": [q["text"] for q in qi_list],
                "Count": len(qi_list)
            })
        
        # Evidence plate (compatibilité)
        evidence = [{"Fichier": f, "Qi": q["text"]} for f, qi_list in qi_by_file.items() for q in qi_list]
        
        qc_out.append({
            # Identifiants
            "Chapitre": chapter,
            "QC_ID": c["id"],
            "FRT_ID": c["id"],
            "QC_Type": qc_type if qc_type else "GENERIC",
            "Titre": title,
            
            # Variables F2
            "Score": score,
            "n_q": n_q,
            "Psi": psi_q,
            "N_tot": total_qi,
            "t_rec": round(t_rec, 1) if t_rec else "N/A",
            "Alpha": ALPHA,
            "Sum_Tj": sum_tj,
            
            # Détails F1/F2
            "F1_details": psi_details,
            "F2_details": score_details,
            
            # Déclencheurs, ARI, FRT
            "Triggers": triggers,
            "ARI": ari,
            "FRT_DATA": frt_data,
            
            # Preuves (Qi - instances concrètes)
            "Evidence": evidence,
            "EvidenceBySubject": evidence_by_subject
        })
    
    qc_out.sort(key=lambda x: x["Score"], reverse=True)
    return qc_out


# =============================================================================
# FONCTION PRINCIPALE D'INGESTION (PARALLÉLISÉE)
# =============================================================================
def ingest_real(urls: List[str], volume: int, matiere: str, chapter_filter: str = None, progress_callback=None):
    """
    Ingestion RÉELLE avec BFS + téléchargement PARALLÈLE.
    Objectif: < 30 secondes pour 20 sujets.
    """
    import pandas as pd
    
    cols_src = ["Fichier", "Nature", "Annee", "Telechargement", "Corrige", "Qi_Data"]
    cols_atm = ["FRT_ID", "Qi", "File", "Year", "Chapitre"]
    
    # Déterminer les seeds
    seeds = []
    for url in urls:
        url_lower = url.lower().strip().rstrip("/")
        if url_lower in ["https://apmep.fr", "https://www.apmep.fr", "http://apmep.fr"]:
            seeds.extend(SEED_URLS_FRANCE)
        else:
            seeds.append(url)
    
    if not seeds:
        seeds = SEED_URLS_FRANCE
    
    # Phase 1: BFS pour collecter les URLs (rapide)
    if progress_callback:
        progress_callback(0.1)
    
    sujets_corriges, _ = scrape_pdf_links_bfs(seeds, limit=volume * 2)
    
    if not sujets_corriges:
        return pd.DataFrame(columns=cols_src), pd.DataFrame(columns=cols_atm)
    
    # Limiter au volume demandé + marge
    candidates = sujets_corriges[:volume + 10]
    
    if progress_callback:
        progress_callback(0.2)
    
    # Phase 2: Téléchargement et traitement PARALLÈLE
    subjects = []
    all_atoms = []
    processed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Soumettre tous les téléchargements en parallèle
        future_to_item = {
            executor.submit(download_and_process_subject, item, chapter_filter, matiere): item 
            for item in candidates
        }
        
        # Collecter les résultats au fur et à mesure
        completed = 0
        for future in as_completed(future_to_item):
            completed += 1
            
            if progress_callback:
                progress_callback(0.2 + 0.7 * (completed / len(candidates)))
            
            if processed >= volume:
                continue  # On continue pour finir les threads mais on n'ajoute plus
            
            try:
                result = future.result()
                if result:
                    subjects.append(result["subject"])
                    all_atoms.extend(result["atoms"])
                    processed += 1
            except Exception:
                pass
    
    if progress_callback:
        progress_callback(1.0)
    
    return (
        pd.DataFrame(subjects) if subjects else pd.DataFrame(columns=cols_src),
        pd.DataFrame(all_atoms) if all_atoms else pd.DataFrame(columns=cols_atm)
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
# SATURATION AVEC TRACKING NOUVELLES QC
# =============================================================================
def compute_saturation_real(df_atoms) -> 'pd.DataFrame':
    """
    Calcule la courbe de saturation avec:
    - Total QC cumulées
    - Nouvelles QC à chaque injection
    - Détection du point de saturation
    """
    import pandas as pd
    
    if df_atoms.empty:
        return pd.DataFrame(columns=["Sujets (N)", "QC Total", "Nouvelles QC", "Saturation (%)"])
    
    files = df_atoms["File"].unique().tolist()
    data_points = []
    cumulative_atoms = []
    seen_qc_signatures = set()
    
    for i, f in enumerate(files):
        # Ajouter les atomes du nouveau sujet
        file_atoms = df_atoms[df_atoms["File"] == f].to_dict('records')
        cumulative_atoms.extend(file_atoms)
        
        # Calculer les QC avec tous les atomes jusqu'ici
        qis = [
            QiItem(f"S{j}", r.get("File", ""), r.get("Qi", ""), r.get("Chapitre", ""), r.get("Year")) 
            for j, r in enumerate(cumulative_atoms)
        ]
        
        qc_list = cluster_qi_to_qc(qis)
        
        # Identifier les nouvelles QC (par signature/titre)
        current_signatures = set()
        for qc in qc_list:
            # Signature = premiers 50 chars du titre normalisé
            sig = normalize_text(qc.get("Titre", ""))[:50]
            current_signatures.add(sig)
        
        new_qc_count = len(current_signatures - seen_qc_signatures)
        seen_qc_signatures.update(current_signatures)
        
        total_qc = len(qc_list)
        
        data_points.append({
            "Sujets (N)": i + 1,
            "QC Total": total_qc,
            "Nouvelles QC": new_qc_count,
            "Saturation (%)": 0
        })
    
    # Calculer le % de saturation
    if data_points:
        max_qc = max(d["QC Total"] for d in data_points)
        for d in data_points:
            d["Saturation (%)"] = round((d["QC Total"] / max(max_qc, 1)) * 100, 1)
    
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
# Si vous voyez PV164.pdf, ce fichier N'EST PAS déployé correctement!
# =============================================================================


VERSION = "V3.5-ABSTRACTION-20241224"
