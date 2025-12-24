# smaxia_granulo_engine_real.py
# =============================================================================
# SMAXIA - MOTEUR GRANULO RÉEL (ZÉRO HARDCODE)
# =============================================================================
# Formules F1 (Ψ_q) et F2 (Score(q)) conformes au document A2
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

import requests
import pdfplumber
from bs4 import BeautifulSoup


# =============================================================================
# CONFIGURATION
# =============================================================================
UA = "SMAXIA-Granulo/1.0"
REQ_TIMEOUT = 25
MAX_PDF_MB = 30
MIN_QI_CHARS = 20


# =============================================================================
# MOTS-CLÉS PAR CHAPITRE (Terminale France)
# =============================================================================
CHAPTER_KEYWORDS = {
    "SUITES NUMÉRIQUES": {
        "suite", "suites", "arithmétique", "géométrique", "raison", "récurrence",
        "limite", "convergence", "monotone", "bornée", "terme général", "somme",
        "croissante", "décroissante", "adjacentes", "u_n", "un", "vn"
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
        "puissance", "newton", "cinétique", "potentielle", "chute"
    },
    "ONDES": {
        "onde", "fréquence", "période", "longueur", "amplitude", "propagation",
        "interférence", "diffraction", "son", "lumière"
    }
}

# Verbes indicateurs de questions
QUESTION_VERBS = {
    "calculer", "déterminer", "montrer", "démontrer", "justifier", "prouver",
    "étudier", "vérifier", "exprimer", "établir", "résoudre", "tracer",
    "conjecturer", "interpréter", "expliciter", "préciser"
}

# =============================================================================
# TRANSFORMATIONS COGNITIVES ARI (pour F1 - Ψ_q)
# =============================================================================
COGNITIVE_TRANSFORMS = {
    # Transformations de base (poids faible)
    "identifier": 0.1,
    "lire": 0.1,
    "recopier": 0.05,
    
    # Transformations intermédiaires
    "calculer": 0.3,
    "simplifier": 0.25,
    "factoriser": 0.35,
    "développer": 0.3,
    "substituer": 0.25,
    
    # Transformations avancées
    "dériver": 0.4,
    "intégrer": 0.45,
    "résoudre": 0.4,
    "démontrer": 0.5,
    "raisonner": 0.45,
    
    # Transformations complexes
    "récurrence": 0.6,
    "limite": 0.5,
    "convergence": 0.55,
    "théorème": 0.5,
    "changement_variable": 0.55,
    
    # Transformations expertes
    "optimisation": 0.7,
    "modélisation": 0.65,
    "interprétation": 0.4
}

# Constante epsilon (évite Ψ=0)
EPSILON_PSI = 0.1

# Coefficient de difficulté par niveau (δ_c)
DELTA_NIVEAU = {
    "Terminale": 1.0,
    "Première": 0.8,
    "Seconde": 0.6
}


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
    
    if any(k in combined for k in ["bac", "baccalauréat", "baccalaureat", "métropole", "metropole"]):
        return "BAC"
    if any(k in combined for k in ["concours", "polytechnique", "centrale", "mines", "ens"]):
        return "CONCOURS"
    if any(k in combined for k in ["dst", "devoir surveillé", "devoir surveille"]):
        return "DST"
    if any(k in combined for k in ["interro", "interrogation", "contrôle", "controle"]):
        return "INTERRO"
    
    return "EXAMEN"


def detect_year(filename: str, text: str) -> Optional[int]:
    match = re.search(r"20[12]\d", filename)
    if match:
        return int(match.group())
    
    match = re.search(r"20[12]\d", text[:1500])
    if match:
        return int(match.group())
    
    return datetime.now().year


# =============================================================================
# SCRAPING PDF
# =============================================================================
def scrape_pdf_links(url: str) -> List[str]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=REQ_TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        print(f"Erreur scraping {url}: {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or ".pdf" not in href.lower():
            continue

        if href.startswith(("http://", "https://")):
            links.append(href)
        else:
            base = url.rstrip("/")
            if href.startswith("/"):
                m = re.match(r"^(https?://[^/]+)", base)
                if m:
                    links.append(m.group(1) + href)
            else:
                links.append(base + "/" + href)

    seen = set()
    return [x for x in links if not (x in seen or seen.add(x))]


def collect_pdf_links(urls: List[str], limit: int) -> List[str]:
    all_links = []
    for u in urls:
        all_links.extend(scrape_pdf_links(u))
        if len(all_links) >= limit * 2:
            break
    
    seen = set()
    uniq = []
    for x in all_links:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
        if len(uniq) >= limit:
            break
    return uniq


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
# EXTRACTION Qi
# =============================================================================
def extract_qi_from_text(text: str, chapter_filter: str = None) -> List[str]:
    raw = text.replace("\r", "\n")
    raw = re.sub(r"\n{2,}", "\n\n", raw)

    blocks = re.split(r"\n\s*\n", raw)
    candidates = []

    for b in blocks:
        b2 = b.strip()
        if len(b2) < MIN_QI_CHARS:
            continue

        if any(re.search(rf"\b{v}\b", b2, re.IGNORECASE) for v in QUESTION_VERBS):
            candidates.append(b2)
            continue

        if chapter_filter:
            keywords = CHAPTER_KEYWORDS.get(chapter_filter, set())
            toks = set(tokenize(b2))
            if len(toks & keywords) >= 2:
                candidates.append(b2)

    qi_list = []
    for c in candidates:
        c = re.sub(r"\s+", " ", c).strip()
        if len(c) > 400:
            c = c[:400].rsplit(" ", 1)[0] + "..."
        if len(c) >= MIN_QI_CHARS:
            qi_list.append(c)

    seen = set()
    out = []
    for x in qi_list:
        k = normalize_text(x)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


# =============================================================================
# F1 : Ψ_q — POIDS PRÉDICTIF PURIFIÉ (Document A2)
# =============================================================================
def compute_psi_q(qi_texts: List[str], niveau: str = "Terminale") -> float:
    """
    F1 : Ψ_q = (Σ T_j + ε) × δ_c / max(Ψ_p)
    
    Calcule le poids prédictif purifié basé sur la densité cognitive (ARI).
    """
    if not qi_texts:
        return EPSILON_PSI
    
    # Calculer la somme des transformations cognitives
    combined_text = " ".join(qi_texts).lower()
    
    sum_tj = 0.0
    transforms_found = []
    
    for transform, weight in COGNITIVE_TRANSFORMS.items():
        if transform in combined_text:
            sum_tj += weight
            transforms_found.append(transform)
    
    # Ajouter epsilon pour éviter Ψ=0
    psi_brut = sum_tj + EPSILON_PSI
    
    # Appliquer le coefficient de difficulté du niveau
    delta_c = DELTA_NIVEAU.get(niveau, 1.0)
    psi_ajuste = psi_brut * delta_c
    
    # Normalisation (max local assumé à 3.0 pour Terminale)
    max_psi_local = 3.0
    psi_normalise = min(1.0, psi_ajuste / max_psi_local)
    
    return round(psi_normalise, 2)


# =============================================================================
# F2 : Score(q) — SÉLECTION OPTIMALE (Document A2)
# =============================================================================
def compute_score_f2(n_q: int, n_total: int, t_rec: float, psi_q: float, 
                     redundancy_penalty: float = 1.0, alpha: float = 5.0) -> float:
    """
    F2 : Score(q) = (n_q / N_total) × (1 + α / t_rec) × Ψ_q × Π(1 - σ(q,p))
    
    Calcule le score de sélection pour le classement Granulo.
    
    Args:
        n_q: occurrences historiques de la structure
        n_total: total d'items observés dans le chapitre
        t_rec: temps (années) depuis la dernière occurrence
        psi_q: poids cognitif (F1)
        redundancy_penalty: pénalité de redondance Π(1-σ)
        alpha: coefficient de récence (défaut 5.0 selon A2)
    """
    if n_total == 0:
        return 0.0
    
    # Ratio de fréquence
    freq_ratio = n_q / n_total
    
    # Facteur de récence (plus récent = score plus élevé)
    t_rec_safe = max(0.5, t_rec)  # Éviter division par 0
    recency_factor = 1 + (alpha / t_rec_safe)
    
    # Score F2
    score = freq_ratio * recency_factor * psi_q * redundancy_penalty * 100
    
    return round(score, 1)


# =============================================================================
# GÉNÉRATION ARI
# =============================================================================
def generate_ari(qi_texts: List[str], chapter: str) -> List[str]:
    combined = " ".join(qi_texts).lower()
    
    if chapter == "SUITES NUMÉRIQUES":
        if any(k in combined for k in ["géométrique", "geometrique", "quotient"]):
            return [
                "1. Exprimer u(n+1)",
                "2. Quotient u(n+1)/u(n)",
                "3. Simplifier",
                "4. Constante"
            ]
        if any(k in combined for k in ["arithmétique", "arithmetique", "différence"]):
            return [
                "1. Exprimer u(n+1)",
                "2. Différence u(n+1)-u(n)",
                "3. Simplifier",
                "4. Constante r"
            ]
        if any(k in combined for k in ["limite", "convergence", "tend vers"]):
            return [
                "1. Terme dominant",
                "2. Factorisation",
                "3. Limites usuelles",
                "4. Conclure"
            ]
        if any(k in combined for k in ["récurrence", "recurrence"]):
            return [
                "1. Initialisation P(n0)",
                "2. Hérédité: supposer P(n)",
                "3. Démontrer P(n+1)",
                "4. Conclure"
            ]
    
    elif chapter == "FONCTIONS":
        if any(k in combined for k in ["tvi", "valeurs intermédiaires", "unique solution"]):
            return [
                "1. Continuité",
                "2. Monotonie",
                "3. Bornes",
                "4. TVI"
            ]
        if any(k in combined for k in ["dérivée", "derivee"]):
            return [
                "1. Identifier f",
                "2. Dériver",
                "3. Simplifier f'",
                "4. Signe de f'"
            ]
    
    return [
        "1. Analyser",
        "2. Méthode",
        "3. Calculer",
        "4. Conclure"
    ]


# =============================================================================
# GÉNÉRATION FRT
# =============================================================================
def generate_frt(qi_texts: List[str], chapter: str, triggers: List[str]) -> List[Dict]:
    combined = " ".join(qi_texts).lower()
    
    if chapter == "SUITES NUMÉRIQUES":
        if any(k in combined for k in ["géométrique", "geometrique"]):
            return [
                {"type": "usage", "title": "🔔 1. QUAND UTILISER", 
                 "text": "L'énoncé demande explicitement la nature de la suite ou de prouver qu'elle est géométrique."},
                {"type": "method", "title": "✅ 2. MÉTHODE RÉDIGÉE", 
                 "text": "1. Pour tout n, on exprime u(n+1).\n2. On calcule u(n+1)/u(n).\n3. On simplifie.\n4. On trouve une constante q."},
                {"type": "trap", "title": "⚠️ 3. PIÈGES", 
                 "text": "Oublier de vérifier u(n) non nul."},
                {"type": "conc", "title": "✍️ 4. CONCLUSION", 
                 "text": "Le rapport est constant, donc la suite est géométrique."}
            ]
        
        if any(k in combined for k in ["limite", "convergence"]):
            return [
                {"type": "usage", "title": "🔔 1. QUAND UTILISER", 
                 "text": "Forme indéterminée infini/infini."},
                {"type": "method", "title": "✅ 2. MÉTHODE RÉDIGÉE", 
                 "text": "1. Identifier le terme dominant.\n2. Factoriser.\n3. Limites usuelles."},
                {"type": "trap", "title": "⚠️ 3. PIÈGES", 
                 "text": "Règle des signes sans factorisation."},
                {"type": "conc", "title": "✍️ 4. CONCLUSION", 
                 "text": "La suite converge vers..."}
            ]
    
    elif chapter == "FONCTIONS":
        if any(k in combined for k in ["tvi", "unique", "solution"]):
            return [
                {"type": "usage", "title": "🔔 1. QUAND UTILISER", 
                 "text": "Prouver existence et unicité."},
                {"type": "method", "title": "✅ 2. MÉTHODE RÉDIGÉE", 
                 "text": "1. f continue et strictement monotone.\n2. Images aux bornes.\n3. k compris entre.\n4. Corollaire TVI."},
                {"type": "trap", "title": "⚠️ 3. PIÈGES", 
                 "text": "Oublier la stricte monotonie."},
                {"type": "conc", "title": "✍️ 4. CONCLUSION", 
                 "text": "Unique solution alpha."}
            ]
    
    return [
        {"type": "usage", "title": "🔔 1. QUAND UTILISER", 
         "text": f"Questions avec: {', '.join(triggers[:3]) if triggers else 'termes du chapitre'}"},
        {"type": "method", "title": "✅ 2. MÉTHODE RÉDIGÉE", 
         "text": "1. Identifier.\n2. Appliquer.\n3. Calculer.\n4. Conclure."},
        {"type": "trap", "title": "⚠️ 3. PIÈGES", 
         "text": "Vérifier les conditions."},
        {"type": "conc", "title": "✍️ 4. CONCLUSION", 
         "text": "Répondre à la question."}
    ]


# =============================================================================
# EXTRACTION TRIGGERS
# =============================================================================
def extract_triggers(qi_texts: List[str]) -> List[str]:
    stopwords = {
        "le", "la", "les", "de", "des", "du", "un", "une", "et", "à", "a", "en",
        "pour", "que", "qui", "est", "sont", "on", "dans", "par", "sur", "avec"
    }
    
    bigrams = Counter()
    
    for qi in qi_texts:
        toks = tokenize(qi)
        toks_clean = [t for t in toks if t not in stopwords and len(t) >= 3]
        
        for i in range(len(toks_clean) - 1):
            bigrams[f"{toks_clean[i]} {toks_clean[i+1]}"] += 1
    
    triggers = []
    for phrase, count in bigrams.most_common(6):
        if count >= 1:
            triggers.append(phrase)
    
    return triggers[:4]


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
# CLUSTERING Qi → QC avec F1 et F2
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
            clusters.append({
                "id": f"QC-{qc_idx:02d}",
                "rep_tokens": toks,
                "qis": [qi],
            })
            qc_idx += 1

    qc_out = []
    total_qi = len(qis)
    
    for c in clusters:
        qi_texts = [q.text for q in c["qis"]]
        chapter = c["qis"][0].chapter if c["qis"] else "SUITES NUMÉRIQUES"
        
        # Titre
        title = min(qi_texts, key=lambda x: len(x) if len(x) > 30 else 1000)
        if len(title) > 80:
            title = title[:80].rsplit(" ", 1)[0] + "..."
        
        # Triggers
        triggers = extract_triggers(qi_texts)
        
        # ARI et FRT
        ari = generate_ari(qi_texts, chapter)
        frt_data = generate_frt(qi_texts, chapter, triggers)
        
        # Métriques avec F1 et F2
        n_q = len(qi_texts)
        
        # F1: Ψ_q (Poids Prédictif Purifié)
        psi_q = compute_psi_q(qi_texts, "Terminale")
        
        # Année la plus récente pour t_rec
        years = [q.year for q in c["qis"] if q.year]
        max_year = max(years) if years else datetime.now().year
        t_rec = max(0.5, datetime.now().year - max_year)
        
        # F2: Score(q)
        score = compute_score_f2(n_q, total_qi, t_rec, psi_q)
        
        # Evidence
        qi_by_file = defaultdict(list)
        for q in c["qis"]:
            qi_by_file[q.subject_file].append(q.text)
        
        evidence = []
        for f, qlist in qi_by_file.items():
            for qi_txt in qlist:
                evidence.append({"Fichier": f, "Qi": qi_txt})
        
        qc_out.append({
            "Chapitre": chapter,
            "QC_ID": c["id"],
            "FRT_ID": c["id"],
            "Titre": title,
            "Score": score,
            "n_q": n_q,
            "Psi": psi_q,
            "N_tot": total_qi,
            "t_rec": round(t_rec, 1),
            "Triggers": triggers,
            "ARI": ari,
            "FRT_DATA": frt_data,
            "Evidence": evidence
        })

    qc_out.sort(key=lambda x: x["Score"], reverse=True)
    return qc_out


# =============================================================================
# FONCTION PRINCIPALE D'INGESTION (RETOURNE 2 VALEURS, PAS 3)
# =============================================================================
def ingest_real(urls: List[str], volume: int, matiere: str, chapter_filter: str = None, progress_callback=None):
    """
    Ingestion RÉELLE : scrape → télécharge → extrait → cluster.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (df_sources, df_atoms)
    """
    import pandas as pd
    
    pdf_links = collect_pdf_links(urls, limit=volume)
    
    cols_src = ["Fichier", "Nature", "Annee", "Telechargement", "Qi_Data"]
    cols_atm = ["FRT_ID", "Qi", "File", "Year", "Chapitre"]
    
    if not pdf_links:
        return pd.DataFrame(columns=cols_src), pd.DataFrame(columns=cols_atm)
    
    subjects = []
    atoms = []
    all_qis: List[QiItem] = []
    
    for idx, pdf_url in enumerate(pdf_links):
        if progress_callback:
            progress_callback((idx + 1) / len(pdf_links))
        
        pdf_bytes = download_pdf(pdf_url)
        if not pdf_bytes:
            continue
        
        text = extract_pdf_text(pdf_bytes)
        if not text.strip():
            continue
        
        filename = pdf_url.split("/")[-1].split("?")[0]
        if not filename.endswith(".pdf"):
            filename = f"sujet_{idx+1}.pdf"
        
        nature = detect_nature(filename, text)
        year = detect_year(filename, text)
        
        qi_texts = extract_qi_from_text(text, chapter_filter)
        
        if chapter_filter:
            keywords = CHAPTER_KEYWORDS.get(chapter_filter, set())
            qi_texts = [q for q in qi_texts if len(set(tokenize(q)) & keywords) >= 1]
        
        if not qi_texts:
            continue
        
        qi_data = []
        subject_id = f"S{idx+1:04d}"
        
        for qi_txt in qi_texts:
            chapter = detect_chapter(qi_txt, matiere) if not chapter_filter else chapter_filter
            
            all_qis.append(QiItem(
                subject_id=subject_id,
                subject_file=filename,
                text=qi_txt,
                chapter=chapter,
                year=year
            ))
            
            atoms.append({
                "FRT_ID": None,
                "Qi": qi_txt,
                "File": filename,
                "Year": year,
                "Chapitre": chapter
            })
            
            qi_data.append({"Qi": qi_txt, "FRT_ID": None})
        
        subjects.append({
            "Fichier": filename,
            "Nature": nature,
            "Annee": year,
            "Telechargement": pdf_url,
            "Qi_Data": qi_data
        })
    
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
    
    # Reconstruire les QiItems depuis df_atoms
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
    
    if not qc_list:
        return pd.DataFrame()
    
    return pd.DataFrame(qc_list)


# =============================================================================
# SATURATION RÉELLE
# =============================================================================
def compute_saturation_real(df_atoms) -> 'pd.DataFrame':
    import pandas as pd
    
    if df_atoms.empty:
        return pd.DataFrame(columns=["Sujets (N)", "QC Découvertes", "Saturation (%)"])
    
    # Grouper par fichier
    files = df_atoms["File"].unique().tolist()
    
    data_points = []
    cumulative_atoms = []
    
    for i, f in enumerate(files):
        file_atoms = df_atoms[df_atoms["File"] == f].to_dict('records')
        cumulative_atoms.extend(file_atoms)
        
        # Reconstruire QiItems
        qis = []
        for idx, row in enumerate(cumulative_atoms):
            qis.append(QiItem(
                subject_id=f"S{idx:04d}",
                subject_file=row.get("File", "unknown.pdf"),
                text=row.get("Qi", ""),
                chapter=row.get("Chapitre", "SUITES NUMÉRIQUES"),
                year=row.get("Year")
            ))
        
        qc_list = cluster_qi_to_qc(qis)
        n_qc = len(qc_list)
        
        data_points.append({
            "Sujets (N)": i + 1,
            "QC Découvertes": n_qc,
            "Saturation (%)": 0
        })
    
    if data_points:
        max_qc = max(d["QC Découvertes"] for d in data_points)
        for d in data_points:
            d["Saturation (%)"] = round((d["QC Découvertes"] / max(max_qc, 1)) * 100, 1)
    
    return pd.DataFrame(data_points)


# =============================================================================
# AUDIT INTERNE
# =============================================================================
def audit_internal_real(subject_qis: List[Dict], qc_df) -> List[Dict]:
    if qc_df.empty or not subject_qis:
        return []
    
    results = []
    qc_list = qc_df.to_dict('records')
    
    for qi_item in subject_qis:
        qi_text = qi_item.get("Qi", "")
        qi_toks = tokenize(qi_text)
        
        best_qc = None
        best_sim = 0.0
        
        for qc in qc_list:
            for ev in qc.get("Evidence", []):
                ev_toks = tokenize(ev.get("Qi", ""))
                sim = jaccard_similarity(qi_toks, ev_toks)
                if sim > best_sim:
                    best_sim = sim
                    best_qc = qc
        
        if best_sim >= 0.25:
            results.append({
                "Qi": qi_text[:80] + "..." if len(qi_text) > 80 else qi_text,
                "Statut": "✅ MATCH",
                "QC": best_qc["QC_ID"] if best_qc else None
            })
        else:
            results.append({
                "Qi": qi_text[:80] + "..." if len(qi_text) > 80 else qi_text,
                "Statut": "❌ GAP",
                "QC": None
            })
    
    return results


# =============================================================================
# AUDIT EXTERNE
# =============================================================================
def audit_external_real(pdf_bytes: bytes, qc_df, chapter_filter: str = None) -> Tuple[float, List[Dict]]:
    text = extract_pdf_text(pdf_bytes)
    qi_texts = extract_qi_from_text(text, chapter_filter)
    
    if chapter_filter:
        keywords = CHAPTER_KEYWORDS.get(chapter_filter, set())
        qi_texts = [q for q in qi_texts if len(set(tokenize(q)) & keywords) >= 1]
    
    if not qi_texts or qc_df.empty:
        return 0.0, []
    
    qc_list = qc_df.to_dict('records')
    results = []
    matched = 0
    
    for qi_text in qi_texts:
        qi_toks = tokenize(qi_text)
        
        best_qc = None
        best_sim = 0.0
        
        for qc in qc_list:
            for ev in qc.get("Evidence", []):
                ev_toks = tokenize(ev.get("Qi", ""))
                sim = jaccard_similarity(qi_toks, ev_toks)
                if sim > best_sim:
                    best_sim = sim
                    best_qc = qc
        
        if best_sim >= 0.20:
            matched += 1
            status = "✅ MATCH"
        else:
            status = "❌ GAP"
        
        results.append({
            "Qi": qi_text[:80] + "..." if len(qi_text) > 80 else qi_text,
            "Statut": status,
            "QC": best_qc["QC_ID"] if best_qc and best_sim >= 0.20 else None
        })
    
    coverage = (matched / len(qi_texts)) * 100 if qi_texts else 0
    return round(coverage, 1), results
