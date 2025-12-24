# smaxia_granulo_engine_real.py
# =============================================================================
# SMAXIA - MOTEUR GRANULO RÉEL V2 (SCRAPING AMÉLIORÉ)
# =============================================================================
# - Navigation récursive pour trouver les vrais PDFs
# - Filtrage strict du contenu mathématique
# - Support multi-sources (APMEP, etc.)
# =============================================================================

from __future__ import annotations

import io
import math
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict
from datetime import datetime
from urllib.parse import urljoin, urlparse

import requests
import pdfplumber
from bs4 import BeautifulSoup


# =============================================================================
# CONFIGURATION
# =============================================================================
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
REQ_TIMEOUT = 25
MAX_PDF_MB = 30
MIN_QI_CHARS = 25

# URLs de pages contenant des sujets (à explorer récursivement)
APMEP_SUBJECT_PAGES = [
    "https://www.apmep.fr/Annee-2025",
    "https://www.apmep.fr/Annee-2024",
    "https://www.apmep.fr/Annee-2023",
    "https://www.apmep.fr/Annee-2022",
    "https://www.apmep.fr/Annee-2021",
    "https://www.apmep.fr/Annales-Terminale-Generale",
]


# =============================================================================
# MOTS-CLÉS PAR CHAPITRE
# =============================================================================
CHAPTER_KEYWORDS = {
    "SUITES NUMÉRIQUES": {
        "suite", "suites", "arithmétique", "géométrique", "raison", "récurrence",
        "limite", "convergence", "monotone", "bornée", "terme général", "somme",
        "croissante", "décroissante", "adjacentes", "u_n", "un", "vn", "wn"
    },
    "FONCTIONS": {
        "fonction", "dérivée", "dérivation", "primitive", "intégrale", "limite",
        "continuité", "asymptote", "tangente", "extremum", "maximum", "minimum",
        "convexe", "concave", "tvi", "logarithme", "exponentielle", "ln", "exp"
    },
    "PROBABILITÉS": {
        "probabilité", "aléatoire", "événement", "indépendance", "conditionnelle",
        "binomiale", "espérance", "variance", "écart-type", "loi normale", "arbre"
    },
    "GÉOMÉTRIE": {
        "vecteur", "droite", "plan", "espace", "repère", "coordonnées",
        "orthogonal", "colinéaire", "produit scalaire", "équation"
    },
    "MÉCANIQUE": {
        "force", "mouvement", "vitesse", "accélération", "énergie", "travail",
        "puissance", "newton", "cinétique", "potentielle"
    },
    "ONDES": {
        "onde", "fréquence", "période", "longueur", "amplitude", "propagation"
    }
}

# Verbes indicateurs de questions mathématiques
QUESTION_VERBS = {
    "calculer", "déterminer", "montrer", "démontrer", "justifier", "prouver",
    "étudier", "vérifier", "exprimer", "établir", "résoudre", "tracer",
    "conjecturer", "interpréter", "expliciter", "préciser", "donner",
    "en déduire", "déterminer", "retrouver"
}

# Mots à EXCLURE (éditoriaux, sommaires, etc.)
EXCLUDE_WORDS = {
    "sommaire", "édito", "éditorial", "rédaction", "abonnement", "adhésion",
    "bulletin", "revue", "publication", "copyright", "tous droits", "flux rss"
}


# =============================================================================
# TRANSFORMATIONS COGNITIVES (F1)
# =============================================================================
COGNITIVE_TRANSFORMS = {
    "identifier": 0.1, "lire": 0.1, "recopier": 0.05,
    "calculer": 0.3, "simplifier": 0.25, "factoriser": 0.35,
    "développer": 0.3, "substituer": 0.25,
    "dériver": 0.4, "intégrer": 0.45, "résoudre": 0.4,
    "démontrer": 0.5, "raisonner": 0.45,
    "récurrence": 0.6, "limite": 0.5, "convergence": 0.55,
    "théorème": 0.5, "changement_variable": 0.55,
    "optimisation": 0.7, "modélisation": 0.65
}

EPSILON_PSI = 0.1
DELTA_NIVEAU = {"Terminale": 1.0, "Première": 0.8, "Seconde": 0.6}


# =============================================================================
# OUTILS TEXTE
# =============================================================================
def normalize_text(text: str) -> str:
    t = text.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize(text: str) -> List[str]:
    t = normalize_text(text)
    return re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ0-9]+", t)


def jaccard_similarity(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def is_math_content(text: str) -> bool:
    """Vérifie si le texte contient du contenu mathématique réel."""
    text_lower = text.lower()
    
    # Exclure les contenus éditoriaux
    if any(excl in text_lower for excl in EXCLUDE_WORDS):
        return False
    
    # Vérifier la présence de verbes de question
    has_verb = any(verb in text_lower for verb in QUESTION_VERBS)
    
    # Vérifier la présence de termes mathématiques
    all_math_keywords = set()
    for keywords in CHAPTER_KEYWORDS.values():
        all_math_keywords.update(keywords)
    
    toks = set(tokenize(text))
    has_math = len(toks & all_math_keywords) >= 1
    
    # Vérifier la présence de formules (symboles math)
    has_formula = bool(re.search(r'[=+\-×÷∑∫√∞αβγδ]|u[_\(]|f\(|ln\(|exp\(|\d+[,\.]\d+', text))
    
    return has_verb or (has_math and has_formula) or (has_math and len(text) > 50)


# =============================================================================
# SCRAPING AMÉLIORÉ
# =============================================================================
def get_base_url(url: str) -> str:
    """Extrait l'URL de base."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def scrape_pdf_links_deep(start_urls: List[str], limit: int) -> List[str]:
    """
    Scraping en profondeur pour trouver les vrais PDFs de sujets.
    """
    all_pdfs = []
    visited = set()
    base_url = "https://www.apmep.fr"
    
    # D'abord, explorer les pages d'années spécifiques
    pages_to_explore = list(APMEP_SUBJECT_PAGES)
    
    # Ajouter les URLs fournies par l'utilisateur
    for url in start_urls:
        if url not in pages_to_explore:
            pages_to_explore.append(url)
    
    for page_url in pages_to_explore:
        if len(all_pdfs) >= limit:
            break
            
        if page_url in visited:
            continue
        visited.add(page_url)
        
        try:
            r = requests.get(page_url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            
            # Trouver tous les liens PDF
            for a in soup.find_all("a", href=True):
                href = a["href"]
                
                if ".pdf" in href.lower():
                    # Construire l'URL complète
                    if href.startswith("http"):
                        pdf_url = href
                    elif href.startswith("/"):
                        pdf_url = base_url + href
                    else:
                        pdf_url = base_url + "/" + href
                    
                    # Filtrer : garder seulement les sujets (pas les corrigés pour l'instant, ou les deux)
                    filename_lower = pdf_url.lower()
                    
                    # Éviter les doublons
                    if pdf_url not in all_pdfs:
                        # Priorité aux sujets non corrigés
                        if "corrig" not in filename_lower:
                            all_pdfs.insert(0, pdf_url)  # Priorité
                        else:
                            all_pdfs.append(pdf_url)
                    
                    if len(all_pdfs) >= limit * 2:  # Marge
                        break
                        
        except Exception as e:
            print(f"Erreur scraping {page_url}: {e}")
            continue
    
    # Dédoublonner et limiter
    seen = set()
    unique_pdfs = []
    for pdf in all_pdfs:
        if pdf not in seen:
            seen.add(pdf)
            unique_pdfs.append(pdf)
        if len(unique_pdfs) >= limit:
            break
    
    return unique_pdfs


def scrape_single_page(url: str) -> List[str]:
    """Scrape une seule page pour les PDFs."""
    pdfs = []
    base_url = get_base_url(url)
    
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if ".pdf" in href.lower():
                if href.startswith("http"):
                    pdfs.append(href)
                elif href.startswith("/"):
                    pdfs.append(base_url + href)
                else:
                    pdfs.append(url.rsplit("/", 1)[0] + "/" + href)
                    
    except Exception as e:
        print(f"Erreur: {e}")
    
    return pdfs


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
        if len(data) > MAX_PDF_MB * 1024 * 1024:
            return None
        return data
    except Exception as e:
        print(f"Erreur download {url}: {e}")
        return None


# =============================================================================
# EXTRACTION TEXTE PDF
# =============================================================================
def extract_pdf_text(pdf_bytes: bytes, max_pages: int = 30) -> str:
    text_parts = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            n = min(len(pdf.pages), max_pages)
            for i in range(n):
                page = pdf.pages[i]
                t = page.extract_text() or ""
                if t.strip():
                    text_parts.append(t)
    except Exception as e:
        print(f"Erreur extraction PDF: {e}")
    return "\n".join(text_parts)


# =============================================================================
# DÉTECTION CHAPITRE / NATURE / ANNÉE
# =============================================================================
def detect_chapter(text: str, matiere: str = "MATHS") -> str:
    toks = set(tokenize(text))
    
    if matiere == "MATHS":
        chapters = ["SUITES NUMÉRIQUES", "FONCTIONS", "PROBABILITÉS", "GÉOMÉTRIE"]
    else:
        chapters = ["MÉCANIQUE", "ONDES"]
    
    best_chapter = chapters[0]
    best_score = 0
    
    for chapter in chapters:
        keywords = CHAPTER_KEYWORDS.get(chapter, set())
        score = len(toks & keywords)
        if score > best_score:
            best_score = score
            best_chapter = chapter
    
    return best_chapter


def detect_nature(filename: str, text: str) -> str:
    combined = (filename + " " + text[:2000]).lower()
    
    if any(k in combined for k in ["bac", "baccalauréat", "métropole", "metropole", "polynesie", "antilles", "asie"]):
        return "BAC"
    if any(k in combined for k in ["concours"]):
        return "CONCOURS"
    if any(k in combined for k in ["dst", "devoir"]):
        return "DST"
    if any(k in combined for k in ["interro"]):
        return "INTERRO"
    
    return "BAC"  # Par défaut pour APMEP


def detect_year(filename: str, text: str) -> Optional[int]:
    # Chercher dans le nom de fichier
    match = re.search(r"20[12]\d", filename)
    if match:
        return int(match.group())
    
    # Chercher dans le texte
    match = re.search(r"20[12]\d", text[:1500])
    if match:
        return int(match.group())
    
    return datetime.now().year


# =============================================================================
# EXTRACTION Qi (AMÉLIORÉE POUR FORMAT APMEP/BAC)
# =============================================================================
def extract_qi_from_text(text: str, chapter_filter: str = None) -> List[str]:
    """Extrait les questions individuelles avec parsing adapté au format BAC."""
    
    # Nettoyer le texte
    raw = text.replace("\r", "\n")
    
    # Supprimer les headers APMEP répétitifs
    raw = re.sub(r'\.?P\.?E\.?M\.?P\.?A\.?\s*\d*', '', raw)
    raw = re.sub(r'Baccalauréat.*?sujet\s*\d*', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'A\.P\.M\.E\.P\.', '', raw)
    
    # Découper par questions numérotées (1., 2., a., b., etc.)
    # Pattern pour détecter les débuts de questions
    question_patterns = [
        r'\n\s*(\d+)\.\s+',           # "1. ", "2. "
        r'\n\s*(\d+)\)\s+',           # "1) ", "2) "
        r'\n\s*([a-z])\.\s+',         # "a. ", "b. "
        r'\n\s*([a-z])\)\s+',         # "a) ", "b) "
        r'\n\s*Affirmation\s*\d+\s*:', # "Affirmation 1:"
        r'\n\s*EXERCICE\s+\d+',       # "EXERCICE 1"
    ]
    
    # Combiner les patterns
    combined_pattern = '|'.join(question_patterns)
    
    # Découper le texte en segments
    segments = re.split(combined_pattern, raw)
    
    candidates = []
    
    for segment in segments:
        if not segment:
            continue
            
        segment = segment.strip()
        
        # Ignorer les segments trop courts
        if len(segment) < MIN_QI_CHARS:
            continue
        
        # Ignorer les contenus éditoriaux
        segment_lower = segment.lower()
        if any(excl in segment_lower for excl in EXCLUDE_WORDS):
            continue
        
        # Vérifier si c'est une vraie question mathématique
        has_verb = any(re.search(rf"\b{v}\b", segment, re.IGNORECASE) for v in QUESTION_VERBS)
        
        # Vérifier la présence de termes mathématiques
        all_math_keywords = set()
        for keywords in CHAPTER_KEYWORDS.values():
            all_math_keywords.update(keywords)
        
        toks = set(tokenize(segment))
        has_math = len(toks & all_math_keywords) >= 1
        
        # Vérifier les mots-clés du chapitre si filtre actif
        if chapter_filter:
            keywords = CHAPTER_KEYWORDS.get(chapter_filter, set())
            has_chapter_keyword = len(toks & keywords) >= 1
        else:
            has_chapter_keyword = True
        
        # Accepter si c'est une question avec du contenu math
        if (has_verb or has_math) and has_chapter_keyword:
            # Nettoyer le segment
            segment = re.sub(r'\s+', ' ', segment).strip()
            
            # Tronquer si trop long
            if len(segment) > 350:
                segment = segment[:350].rsplit(' ', 1)[0] + "..."
            
            if len(segment) >= MIN_QI_CHARS:
                candidates.append(segment)
    
    # Si pas assez de résultats, essayer une approche par blocs
    if len(candidates) < 3:
        blocks = re.split(r'\n\s*\n', raw)
        for b in blocks:
            b = b.strip()
            if len(b) < MIN_QI_CHARS or len(b) > 500:
                continue
            
            b_lower = b.lower()
            if any(excl in b_lower for excl in EXCLUDE_WORDS):
                continue
            
            has_verb = any(v in b_lower for v in QUESTION_VERBS)
            if has_verb:
                b = re.sub(r'\s+', ' ', b).strip()
                if b not in candidates:
                    candidates.append(b)
    
    # Dédoublonnage
    seen = set()
    out = []
    for x in candidates:
        k = normalize_text(x)
        if k not in seen and len(k) > 20:
            seen.add(k)
            out.append(x)
    
    return out[:50]  # Limiter à 50 Qi max par PDF


# =============================================================================
# F1 : Ψ_q (Poids Prédictif Purifié)
# =============================================================================
def compute_psi_q(qi_texts: List[str], niveau: str = "Terminale") -> float:
    if not qi_texts:
        return EPSILON_PSI
    
    combined_text = " ".join(qi_texts).lower()
    
    sum_tj = 0.0
    for transform, weight in COGNITIVE_TRANSFORMS.items():
        if transform in combined_text:
            sum_tj += weight
    
    psi_brut = sum_tj + EPSILON_PSI
    delta_c = DELTA_NIVEAU.get(niveau, 1.0)
    psi_ajuste = psi_brut * delta_c
    
    max_psi_local = 3.0
    psi_normalise = min(1.0, psi_ajuste / max_psi_local)
    
    return round(psi_normalise, 2)


# =============================================================================
# F2 : Score(q) (Sélection Granulo)
# =============================================================================
def compute_score_f2(n_q: int, n_total: int, t_rec: float, psi_q: float, 
                     redundancy_penalty: float = 1.0, alpha: float = 5.0) -> float:
    if n_total == 0:
        return 0.0
    
    freq_ratio = n_q / n_total
    t_rec_safe = max(0.5, t_rec)
    recency_factor = 1 + (alpha / t_rec_safe)
    
    score = freq_ratio * recency_factor * psi_q * redundancy_penalty * 100
    
    return round(score, 1)


# =============================================================================
# GÉNÉRATION ARI
# =============================================================================
def generate_ari(qi_texts: List[str], chapter: str) -> List[str]:
    combined = " ".join(qi_texts).lower()
    
    if chapter == "SUITES NUMÉRIQUES":
        if any(k in combined for k in ["géométrique", "geometrique", "quotient"]):
            return ["1. Exprimer u(n+1)", "2. Quotient u(n+1)/u(n)", "3. Simplifier", "4. Constante"]
        if any(k in combined for k in ["arithmétique", "arithmetique", "différence"]):
            return ["1. Exprimer u(n+1)", "2. Différence u(n+1)-u(n)", "3. Simplifier", "4. Constante r"]
        if any(k in combined for k in ["limite", "convergence", "tend vers"]):
            return ["1. Terme dominant", "2. Factorisation", "3. Limites usuelles", "4. Conclure"]
        if any(k in combined for k in ["récurrence", "recurrence"]):
            return ["1. Initialisation P(n0)", "2. Hérédité: supposer P(n)", "3. Démontrer P(n+1)", "4. Conclure"]
    
    elif chapter == "FONCTIONS":
        if any(k in combined for k in ["tvi", "valeurs intermédiaires", "unique solution"]):
            return ["1. Continuité", "2. Monotonie", "3. Bornes", "4. TVI"]
        if any(k in combined for k in ["dérivée", "derivee"]):
            return ["1. Identifier f", "2. Dériver", "3. Simplifier f'", "4. Signe de f'"]
    
    return ["1. Analyser", "2. Méthode", "3. Calculer", "4. Conclure"]


# =============================================================================
# GÉNÉRATION FRT
# =============================================================================
def generate_frt(qi_texts: List[str], chapter: str, triggers: List[str]) -> List[Dict]:
    combined = " ".join(qi_texts).lower()
    
    if chapter == "SUITES NUMÉRIQUES":
        if any(k in combined for k in ["géométrique", "geometrique"]):
            return [
                {"type": "usage", "title": "🔔 1. QUAND UTILISER", "text": "L'énoncé demande de prouver qu'une suite est géométrique."},
                {"type": "method", "title": "✅ 2. MÉTHODE RÉDIGÉE", "text": "1. Exprimer u(n+1).\n2. Calculer u(n+1)/u(n).\n3. Simplifier.\n4. Trouver q constant."},
                {"type": "trap", "title": "⚠️ 3. PIÈGES", "text": "Oublier de vérifier u(n) ≠ 0."},
                {"type": "conc", "title": "✍️ 4. CONCLUSION", "text": "La suite est géométrique de raison q."}
            ]
        if any(k in combined for k in ["limite", "convergence"]):
            return [
                {"type": "usage", "title": "🔔 1. QUAND UTILISER", "text": "Forme indéterminée ∞/∞."},
                {"type": "method", "title": "✅ 2. MÉTHODE RÉDIGÉE", "text": "1. Terme dominant.\n2. Factoriser.\n3. Limites usuelles."},
                {"type": "trap", "title": "⚠️ 3. PIÈGES", "text": "Erreur de signe."},
                {"type": "conc", "title": "✍️ 4. CONCLUSION", "text": "La suite converge vers L."}
            ]
        if any(k in combined for k in ["récurrence", "recurrence"]):
            return [
                {"type": "usage", "title": "🔔 1. QUAND UTILISER", "text": "Démontrer une propriété pour tout n."},
                {"type": "method", "title": "✅ 2. MÉTHODE RÉDIGÉE", "text": "1. Initialisation.\n2. Hérédité.\n3. Conclure."},
                {"type": "trap", "title": "⚠️ 3. PIÈGES", "text": "Oublier l'initialisation."},
                {"type": "conc", "title": "✍️ 4. CONCLUSION", "text": "Par récurrence, P(n) vraie pour tout n."}
            ]
    
    return [
        {"type": "usage", "title": "🔔 1. QUAND UTILISER", "text": f"Questions avec: {', '.join(triggers[:3]) if triggers else 'termes du chapitre'}"},
        {"type": "method", "title": "✅ 2. MÉTHODE RÉDIGÉE", "text": "1. Identifier.\n2. Appliquer.\n3. Calculer.\n4. Conclure."},
        {"type": "trap", "title": "⚠️ 3. PIÈGES", "text": "Vérifier les conditions."},
        {"type": "conc", "title": "✍️ 4. CONCLUSION", "text": "Répondre à la question."}
    ]


# =============================================================================
# EXTRACTION TRIGGERS
# =============================================================================
def extract_triggers(qi_texts: List[str]) -> List[str]:
    stopwords = {"le", "la", "les", "de", "des", "du", "un", "une", "et", "à", "a", "en", "pour", "que", "qui", "est", "sont", "on", "dans"}
    
    bigrams = Counter()
    for qi in qi_texts:
        toks = tokenize(qi)
        toks_clean = [t for t in toks if t not in stopwords and len(t) >= 3]
        for i in range(len(toks_clean) - 1):
            bigrams[f"{toks_clean[i]} {toks_clean[i+1]}"] += 1
    
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
    
    clusters: List[Dict] = []
    qc_idx = 1

    for qi in qis:
        toks = tokenize(qi.text)
        if not toks:
            continue

        best_i = None
        best_sim = 0.0

        for i, c in enumerate(clusters):
            sim = jaccard_similarity(toks, c["rep_tokens"])
            if sim > best_sim:
                best_sim = sim
                best_i = i

        if best_i is not None and best_sim >= sim_threshold:
            clusters[best_i]["qis"].append(qi)
            clusters[best_i]["rep_tokens"] = list(set(clusters[best_i]["rep_tokens"]) | set(toks))
        else:
            clusters.append({"id": f"QC-{qc_idx:02d}", "rep_tokens": toks, "qis": [qi]})
            qc_idx += 1

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
        
        years = [q.year for q in c["qis"] if q.year]
        max_year = max(years) if years else datetime.now().year
        t_rec = max(0.5, datetime.now().year - max_year)
        
        score = compute_score_f2(n_q, total_qi, t_rec, psi_q)
        
        qi_by_file = defaultdict(list)
        for q in c["qis"]:
            qi_by_file[q.subject_file].append(q.text)
        
        evidence = [{"Fichier": f, "Qi": qi_txt} for f, qlist in qi_by_file.items() for qi_txt in qlist]
        
        qc_out.append({
            "Chapitre": chapter, "QC_ID": c["id"], "FRT_ID": c["id"],
            "Titre": title, "Score": score, "n_q": n_q, "Psi": psi_q,
            "N_tot": total_qi, "t_rec": round(t_rec, 1),
            "Triggers": triggers, "ARI": ari, "FRT_DATA": frt_data, "Evidence": evidence
        })

    qc_out.sort(key=lambda x: x["Score"], reverse=True)
    return qc_out


# =============================================================================
# FONCTION PRINCIPALE D'INGESTION
# =============================================================================
def ingest_real(urls: List[str], volume: int, matiere: str, chapter_filter: str = None, progress_callback=None):
    """Ingestion RÉELLE avec scraping en profondeur."""
    import pandas as pd
    
    cols_src = ["Fichier", "Nature", "Annee", "Telechargement", "Qi_Data"]
    cols_atm = ["FRT_ID", "Qi", "File", "Year", "Chapitre"]
    
    # Scraping en profondeur
    pdf_links = scrape_pdf_links_deep(urls, limit=volume)
    
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
        if not text.strip() or len(text) < 100:
            continue
        
        # Vérifier que c'est du contenu mathématique
        if not is_math_content(text):
            continue
        
        filename = pdf_url.split("/")[-1].split("?")[0]
        if not filename.endswith(".pdf"):
            filename = f"sujet_{idx+1}.pdf"
        
        nature = detect_nature(filename, text)
        year = detect_year(filename, text)
        
        qi_texts = extract_qi_from_text(text, chapter_filter)
        
        if not qi_texts:
            continue
        
        qi_data = []
        subject_id = f"S{processed+1:04d}"
        
        for qi_txt in qi_texts:
            chapter = detect_chapter(qi_txt, matiere) if not chapter_filter else chapter_filter
            
            atoms.append({
                "FRT_ID": None, "Qi": qi_txt, "File": filename,
                "Year": year, "Chapitre": chapter
            })
            qi_data.append({"Qi": qi_txt, "FRT_ID": None})
        
        subjects.append({
            "Fichier": filename, "Nature": nature, "Annee": year,
            "Telechargement": pdf_url, "Qi_Data": qi_data
        })
        
        processed += 1
    
    df_sources = pd.DataFrame(subjects) if subjects else pd.DataFrame(columns=cols_src)
    df_atoms = pd.DataFrame(atoms) if atoms else pd.DataFrame(columns=cols_atm)
    
    return df_sources, df_atoms


# =============================================================================
# CALCUL QC
# =============================================================================
def compute_qc_real(df_atoms) -> 'pd.DataFrame':
    import pandas as pd
    
    if df_atoms.empty:
        return pd.DataFrame()
    
    all_qis = []
    for idx, row in df_atoms.iterrows():
        all_qis.append(QiItem(
            subject_id=f"S{idx:04d}",
            subject_file=row.get("File", "unknown.pdf"),
            text=row.get("Qi", ""),
            chapter=row.get("Chapitre", "SUITES NUMÉRIQUES"),
            year=row.get("Year")
        ))
    
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
    cumulative_atoms = []
    
    for i, f in enumerate(files):
        file_atoms = df_atoms[df_atoms["File"] == f].to_dict('records')
        cumulative_atoms.extend(file_atoms)
        
        qis = [QiItem(f"S{j:04d}", r.get("File", ""), r.get("Qi", ""), r.get("Chapitre", ""), r.get("Year"))
               for j, r in enumerate(cumulative_atoms)]
        
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
        qi_text = qi_item.get("Qi", "")
        qi_toks = tokenize(qi_text)
        
        best_qc, best_sim = None, 0.0
        for qc in qc_list:
            for ev in qc.get("Evidence", []):
                sim = jaccard_similarity(qi_toks, tokenize(ev.get("Qi", "")))
                if sim > best_sim:
                    best_sim, best_qc = sim, qc
        
        results.append({
            "Qi": qi_text[:80] + "..." if len(qi_text) > 80 else qi_text,
            "Statut": "✅ MATCH" if best_sim >= 0.25 else "❌ GAP",
            "QC": best_qc["QC_ID"] if best_qc and best_sim >= 0.25 else None
        })
    
    return results


def audit_external_real(pdf_bytes: bytes, qc_df, chapter_filter: str = None) -> Tuple[float, List[Dict]]:
    text = extract_pdf_text(pdf_bytes)
    qi_texts = extract_qi_from_text(text, chapter_filter)
    
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
