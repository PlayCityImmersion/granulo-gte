# smaxia_granulo_engine_real.py
# =============================================================================
# SMAXIA - MOTEUR GRANULO V6 (ARCHITECTURE INVARIANTE)
# =============================================================================
# INVARIANTS SMAXIA :
#  - Toute QC commence par "Comment" et finit par "?"
#  - Chapitre sélectionné = FILTRE BLOQUANT (aucun hors-chapitre)
#  - Clean First, Compute Later (Sanitizer à l'ingestion)
#  - QC ≈ 15 (±15) par chapitre = SATURATION
#  - Zéro "déclarations" non prouvées
# =============================================================================

from __future__ import annotations

import io
import re
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
from datetime import datetime
from urllib.parse import urljoin

import requests
import pdfplumber
from bs4 import BeautifulSoup

VERSION = "V6.1-SMAXIA-GATE-SANITIZER-20251225"

# =============================================================================
# CONFIGURATION
# =============================================================================
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
REQ_TIMEOUT = 25
MAX_PDF_MB = 30
MIN_QI_CHARS = 25
MAX_QI_CHARS = 500
MAX_QI_RETURN = 80
MAX_WORKERS = 10
MAX_PAGES = 18
EPSILON_PSI = 0.1

# Seuil de clustering SMAXIA : augmenté pour réduire le nombre de QC
SIM_THRESHOLD_DEFAULT = 0.30

SEED_URLS_FRANCE = [
    "https://www.apmep.fr/Annee-2025",
    "https://www.apmep.fr/Annee-2024",
    "https://www.apmep.fr/Annee-2023",
]

_session = None

def get_session():
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": UA})
    return _session


# =============================================================================
# TAXONOMIES SMAXIA : GATE CHAPITRE (FILTRAGE STRICT)
# =============================================================================
CHAPTERS = ["SUITES NUMÉRIQUES", "FONCTIONS", "PROBABILITÉS", "GÉOMÉTRIE", "INCONNU"]

# GATE = Filtrage strict avec must_any (au moins un) et must_not_any (aucun)
CHAPTER_GATES = {
    "SUITES NUMÉRIQUES": {
        "must_any": [
            r"\bsuite(s)?\b",
            r"\bu\s*[_\(]?\s*n\b",
            r"\bu\s*\(\s*n\s*\)",
            r"\bu\s*\(\s*n\s*\+\s*1\s*\)",
            r"\b[vw]\s*[_\(]?\s*n\b",
            r"\brécurrence\b",
            r"\bgéométrique\b.*\braison\b",
            r"\braison\b.*\bgéométrique\b",
            r"\barithmétique\b.*\braison\b",
        ],
        "must_not_any": [
            r"\bprobabilit[ée]\b",
            r"\besp[ée]rance\b",
            r"\bvariance\b",
            r"\bloi\s+(binomiale|normale|uniforme)\b",
            r"\bvariable\s+al[ée]atoire\b",
            r"\b[ée]v[ée]nement\b",
            r"\bconnexion\b",
            r"\bserveur\b",
        ],
    },
    "FONCTIONS": {
        "must_any": [
            r"\bfonction\b",
            r"\bf\s*\(\s*x\s*\)",
            r"\bf'\s*\(",
            r"\bd[ée]riv[ée]e\b",
            r"\bprimitive\b",
            r"\bint[ée]grale\b",
            r"\basymptote\b",
            r"\bcontinuit[ée]\b",
        ],
        "must_not_any": [
            r"\bprobabilit[ée]\b",
            r"\bvariable\s+al[ée]atoire\b",
            r"\bsuite\s*\(",
            r"\bu\s*_\s*n\b",
        ],
    },
    "PROBABILITÉS": {
        "must_any": [
            r"\bprobabilit[ée]\b",
            r"\bvariable\s+al[ée]atoire\b",
            r"\bloi\s+(binomiale|normale|uniforme)\b",
            r"\besp[ée]rance\b",
            r"\bvariance\b",
            r"\b[ée]v[ée]nement\b",
            r"\bindépendance\b",
            r"\btirage\b",
            r"\burne\b",
        ],
        "must_not_any": [],
    },
    "GÉOMÉTRIE": {
        "must_any": [
            r"\bvecteur(s)?\b",
            r"\bdroite(s)?\b",
            r"\bplan\b",
            r"\bespace\b",
            r"\bproduit\s+scalaire\b",
            r"\bcoordonn[ée]es?\b",
            r"\borthogonal",
            r"\bcolinéaire\b",
        ],
        "must_not_any": [],
    },
}

# Scoring soft (fallback si gate ne matche pas)
CHAPTER_KEYWORDS = {
    "SUITES NUMÉRIQUES": {"suite", "suites", "arithmétique", "géométrique", "récurrence", "convergence", "terme"},
    "FONCTIONS": {"fonction", "dérivée", "primitive", "intégrale", "continuité", "asymptote", "limite"},
    "PROBABILITÉS": {"probabilité", "aléatoire", "binomiale", "espérance", "variance", "loi", "événement"},
    "GÉOMÉTRIE": {"vecteur", "droite", "plan", "espace", "coordonnées", "scalaire", "orthogonal"},
}

QUESTION_VERBS = {
    "calculer", "déterminer", "montrer", "démontrer", "justifier", "prouver", "étudier",
    "vérifier", "résoudre", "exprimer", "encadrer", "conclure", "en déduire", "déduire"
}

EXCLUDE_WORDS = {"sommaire", "édito", "éditorial", "bulletin", "revue", "publication", "copyright", "apmep"}

DELTA_NIVEAU = {"Terminale": 1.0, "Première": 0.8, "Seconde": 0.6}

COGNITIVE_TRANSFORMS = {
    "calculer": 0.3, "simplifier": 0.25, "factoriser": 0.35,
    "dériver": 0.4, "intégrer": 0.45, "résoudre": 0.4,
    "démontrer": 0.5, "montrer": 0.5, "justifier": 0.45,
    "récurrence": 0.6, "limite": 0.5, "convergence": 0.5,
    "étudier": 0.4, "encadrer": 0.35, "géométrique": 0.4,
    "arithmétique": 0.4, "probabilité": 0.4
}

MATH_SYMBOL_RE = re.compile(r'[=≤≥≠∞∑∫√→×÷±]|\\frac|\\sum|\d+[,\.]\d+')


# =============================================================================
# SANITIZER SMAXIA V6 : CLEAN FIRST (Appliqué à l'ingestion)
# =============================================================================
class SmaxiaSanitizer:
    """
    NETTOYEUR V6 : Dictionnaire précis + Règles physiques universelles.
    PAS de glue breaker agressif qui casse les mots.
    """
    
    # Dictionnaire de séparations PRÉCISES (testées)
    CORE_SPLITS = [
        # Verbes + que/articles
        ('Démontrerque', 'Démontrer que '), ('démontrerque', 'démontrer que '),
        ('Montrerque', 'Montrer que '), ('montrerque', 'montrer que '),
        ('Prouverque', 'Prouver que '), ('prouverque', 'prouver que '),
        ('Calculerla', 'Calculer la '), ('calculerla', 'calculer la '),
        ('Calculerle', 'Calculer le '), ('calculerle', 'calculer le '),
        ('Déterminerla', 'Déterminer la '), ('déterminerla', 'déterminer la '),
        ('Résoudreune', 'Résoudre une '), ('résoudreune', 'résoudre une '),
        ('Étudierle', 'Étudier le '), ('étudierle', 'étudier le '),
        ('Vérifierque', 'Vérifier que '), ('vérifierque', 'vérifier que '),
        ('Endéduire', 'En déduire '), ('endéduire', 'en déduire '),
        
        # Expressions courantes
        ('Onadmet', 'On admet '), ('onadmet', 'on admet '),
        ('Onconsidère', 'On considère '), ('onconsidère', 'on considère '),
        ('Onsuppose', 'On suppose '), ('onsuppose', 'on suppose '),
        ('parailleurs', 'par ailleurs '), ('Parailleurs', 'Par ailleurs '),
        
        # Articles + noms mathématiques
        ('lalimite', 'la limite '), ('lasuite', 'la suite '), ('lafonction', 'la fonction '),
        ('laprobabilité', 'la probabilité '), ('laconnexion', 'la connexion '),
        ('unesuite', 'une suite '), ('unefonction', 'une fonction '),
        ('uniquesolution', 'unique solution '),
        
        # Verbes d'état
        ('estégaleà', 'est égale à '), ('estégal', 'est égal '),
        ('égaleà', 'égale à '),
        ('estarithmétique', 'est arithmétique '), ('estgéométrique', 'est géométrique '),
        ('estcroissante', 'est croissante '), ('estdécroissante', 'est décroissante '),
        ('estconvergente', 'est convergente '),
        ('suiteest', 'suite est '),
        
        # Prépositions
        ('pourtout', 'pour tout '), ('pourtoutentier', 'pour tout entier '),
        ('entiernaturel', 'entier naturel '),
        ('tendvers', 'tend vers '), ('ntend', 'n tend '),
        ('quandn', 'quand n '),
        ('deraison', 'de raison '), ('raisonq', 'raison q'),
        ('géométriquede', 'géométrique de '),
        ('solutionsur', 'solution sur '),
        ('admetune', 'admet une '),
        
        # Cas spécifiques probas
        ('soitstable', 'soit stable '),
        ('stableet', 'stable et '),
        ('passeparle', 'passe par le '),
        ('etpasse', 'et passe '),
        ('serveurB', 'serveur B'),
        
        # Extensions V6.1
        ('Justifierque', 'Justifier que '), ('justifierque', 'justifier que '),
        ('Exprimerla', 'Exprimer la '), ('Exprimerle', 'Exprimer le '),
        ('ExprimerSn', 'Exprimer Sn '),
        ('Onpose', 'On pose '), ('onpose', 'on pose '),
        ('Sachantque', 'Sachant que '), ('sachantque', 'sachant que '),
        ('Soitla', 'Soit la '), ('soitla', 'soit la '),
        ('Soitf', 'Soit f '), ('soitf', 'soit f '),
        ('Pourtout', 'Pour tout '), ('Pourtoutentier', 'Pour tout entier '),
        ('toutentier', 'tout entier '),
        ('parrécurrence', 'par récurrence '),
        ('parrécurrenceque', 'par récurrence que '),
        ('définiesur', 'définie sur '), ('définiepar', 'définie par '),
        ('enfonctionde', 'en fonction de '), ('enfonctionden', 'en fonction de n'),
        ('probabilitéque', 'probabilité que '),
        ('quela', 'que la '), ('quele', 'que le '),
        ('etdéterminer', 'et déterminer '), ('etcalculer', 'et calculer '),
        ('Calculerf', 'Calculer f'), ("f'(x)", "f'(x) "),
        ('sontorthogonales', 'sont orthogonales'),
        ('lesdroites', 'les droites '),
        ('estconvergent', 'est convergent '),
    ]

    def clean_garbage_chars(self, text: str) -> str:
        """Nettoie les artefacts d'encodage PDF."""
        t = (text or "")
        t = t.replace("\u00ad", "")  # soft hyphen
        t = t.replace("â€™", "'").replace("'", "'").replace("'", "'")
        t = t.replace("Â", "")
        t = t.replace("\r", " ").replace("\n", " ")
        return re.sub(r"\s+", " ", t).strip()

    def apply_physical_invariants(self, text: str) -> str:
        """
        Règles physiques universelles (INVARIANT) :
        - CamelCase : minuscule + Majuscule → espace
        - Chiffre/Lettre transitions
        """
        t = text
        # CamelCase : serveurBest → serveur Best
        t = re.sub(r'([a-zà-ÿ])([A-ZÉÈÊËÀÂÙÛÎÏÔÇ])', r'\1 \2', t)
        # Lettre puis chiffre : solution1 → solution 1
        t = re.sub(r'([a-zà-ÿ])(\d)', r'\1 \2', t)
        # Chiffre puis lettre : 10mai → 10 mai
        t = re.sub(r'(\d)([a-zA-Zà-ÿ])', r'\1 \2', t)
        # Ponctuation collée
        t = re.sub(r'([,;:])([a-zA-Zà-ÿ])', r'\1 \2', t)
        return t

    def apply_dictionary_splits(self, text: str) -> str:
        """Applique le dictionnaire de séparations précises."""
        res = text
        for glued, separated in self.CORE_SPLITS:
            res = res.replace(glued, separated)
        return res

    def sanitize(self, raw_text: str) -> str:
        """PIPELINE COMPLET : Nettoyage → Physique → Dictionnaire → Final"""
        if not raw_text:
            return ""
        t1 = self.clean_garbage_chars(raw_text)
        t2 = self.apply_physical_invariants(t1)
        t3 = self.apply_dictionary_splits(t2)
        return re.sub(r"\s+", " ", t3).strip()


# Instance globale
_sanitizer = SmaxiaSanitizer()


# =============================================================================
# DÉTECTION CHAPITRE : GATE STRICT
# =============================================================================
def _gate_match(text: str, chapter: str) -> bool:
    """Vérifie si le texte passe le GATE du chapitre."""
    t = (text or "").lower()
    gate = CHAPTER_GATES.get(chapter)
    if not gate:
        return False
    
    # Doit avoir AU MOINS UN mot-clé positif
    if gate["must_any"]:
        if not any(re.search(p, t, flags=re.IGNORECASE) for p in gate["must_any"]):
            return False
    
    # Ne doit avoir AUCUN mot-clé négatif
    if gate["must_not_any"]:
        if any(re.search(p, t, flags=re.IGNORECASE) for p in gate["must_not_any"]):
            return False
    
    return True


def detect_chapter_strict(text: str) -> str:
    """
    Détection de chapitre avec GATE prioritaire.
    Si aucun gate ne matche → scoring soft → INCONNU si score < 2.
    """
    # 1. Vérifier les GATES (prioritaire)
    for ch in ["SUITES NUMÉRIQUES", "PROBABILITÉS", "FONCTIONS", "GÉOMÉTRIE"]:
        if _gate_match(text, ch):
            return ch
    
    # 2. Fallback : scoring soft
    toks = set(tokenize(text))
    best_ch, best_score = "INCONNU", 0
    for chapter, keywords in CHAPTER_KEYWORDS.items():
        score = len(toks & keywords)
        if score > best_score:
            best_score = score
            best_ch = chapter
    
    return best_ch if best_score >= 2 else "INCONNU"


def chapter_gate_filter(qi_texts: List[str], chapter_filter: Optional[str]) -> List[str]:
    """
    FILTRE BLOQUANT : Si chapitre demandé → rejeter tout ce qui ne passe pas le gate.
    C'est LE FIX PRINCIPAL pour éviter les probas dans Suites.
    """
    if not chapter_filter or chapter_filter.strip().upper() == "INCONNU":
        return qi_texts
    
    ch = chapter_filter.strip().upper()
    if ch not in CHAPTERS:
        return qi_texts
    
    out = []
    for qi in qi_texts:
        if _gate_match(qi, ch):
            out.append(qi)
    return out


# =============================================================================
# OUTILS TEXTE
# =============================================================================
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ]{3,}", normalize_text(text))


def jaccard_similarity(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_math_content(text: str) -> bool:
    """Vérifie qu'un texte est une question mathématique valide."""
    text_lower = (text or "").lower()
    if any(excl in text_lower for excl in EXCLUDE_WORDS):
        return False
    
    has_verb = any(v in text_lower for v in QUESTION_VERBS)
    has_math = bool(MATH_SYMBOL_RE.search(text or "")) or \
               "suite" in text_lower or \
               "fonction" in text_lower or \
               "probabilité" in text_lower or \
               re.search(r'\bu\s*[_\(]?\s*n', text_lower)
    
    return has_verb and has_math


# =============================================================================
# F1 / F2 (Sur texte PROPRE)
# =============================================================================
def compute_psi_q(qi_texts: List[str], niveau: str = "Terminale") -> Tuple[float, float]:
    """Calcule F1 (Densité Cognitive) sur texte nettoyé."""
    if not qi_texts:
        return EPSILON_PSI, 0.0
    
    combined = " ".join(qi_texts).lower()
    sum_tj = sum(w for t, w in COGNITIVE_TRANSFORMS.items() if t in combined)
    delta_c = DELTA_NIVEAU.get(niveau, 1.0)
    psi = min(1.0, (sum_tj + EPSILON_PSI) * delta_c / 3.0)
    return round(psi, 2), round(sum_tj, 2)


def compute_score_f2(n_q: int, n_total: int, t_rec: Optional[float], psi_q: float, alpha: float = 5.0) -> float:
    """Calcule F2 (Score de Sélection Granulo)."""
    if n_total <= 0:
        return 0.0
    freq_ratio = n_q / n_total
    t_rec_safe = max(0.5, t_rec) if t_rec is not None else 5.0
    recency_factor = 1 + (alpha / t_rec_safe)
    return round(freq_ratio * recency_factor * psi_q * 100, 1)


# =============================================================================
# DATACLASS
# =============================================================================
@dataclass
class QiItem:
    subject_id: str
    subject_file: str
    text: str  # TOUJOURS PROPRE (sanitizé)
    chapter: str = ""
    year: Optional[int] = None


# =============================================================================
# EXTRACTION PDF → LIGNES → Qi (CLEAN FIRST)
# =============================================================================
START_Q_RE = re.compile(
    r"""(?ix)^\s*(
        (?:\d+\s*[\)\.]\s+)|
        (?:[a-z]\s*[\)\.]\s+)|
        (?:question\s*\d*\s*:?\s+)|
        (?:partie\s*[A-D]\s*:?\s+)|
        (?:calculer|déterminer|determiner|montrer|démontrer|demonstrer|justifier|prouver|
            resoudre|résoudre|exprimer|étudier|etudier|vérifier|verifier|encadrer|conclure|
            en\s+déduire|deduire)\b
    )"""
)

HEADER_NOISE_RE = re.compile(
    r"(?i)\b(apmep|baccalaur|corrig|session|sp[ée]cialit[ée]|m[ée]tropole|polyn[ée]sie|asie|am[ée]rique|page\s+\d+)\b"
)


def extract_pdf_lines(pdf_bytes: bytes, max_pages: int = MAX_PAGES) -> List[str]:
    """Extrait et nettoie les lignes d'un PDF."""
    lines: List[str] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            n = min(len(pdf.pages), max_pages)
            for i in range(n):
                txt = pdf.pages[i].extract_text() or ""
                if not txt.strip():
                    continue
                for raw_ln in txt.split("\n"):
                    # SANITIZER APPLIQUÉ ICI (Clean First)
                    ln = _sanitizer.sanitize(raw_ln.strip())
                    if not ln or len(ln) < 3:
                        continue
                    # Filtrer bruit header
                    if HEADER_NOISE_RE.search(ln) and len(ln) < 40:
                        continue
                    lines.append(ln)
    except Exception:
        return []
    return lines


def extract_qi_from_lines(lines: List[str], chapter_filter: Optional[str] = None) -> List[str]:
    """
    Extraction Qi ligne par ligne avec détection début de question.
    GATE CHAPITRE appliqué ici.
    """
    qis: List[str] = []
    buf: List[str] = []
    in_q = False

    def flush():
        nonlocal buf, in_q
        if not buf:
            return
        q = _sanitizer.sanitize(" ".join(buf))
        q = re.sub(r"\s+", " ", q).strip()
        
        # Conditions de validité
        if MIN_QI_CHARS <= len(q) <= MAX_QI_CHARS and is_math_content(q):
            qis.append(q)
        buf = []
        in_q = False

    for ln in lines:
        # Nouveau départ de question
        if START_Q_RE.match(ln):
            flush()
            buf = [ln]
            in_q = True
            continue

        if in_q:
            # Arrêt si nouveau titre/section
            if re.match(r"(?i)^\s*(exercice|annexe|corrig[ée])\b", ln):
                flush()
                continue
            buf.append(ln)

    flush()

    # GATE CHAPITRE BLOQUANT
    if chapter_filter:
        qis = chapter_gate_filter(qis, chapter_filter)

    # Dédoublonnage
    seen = set()
    out = []
    for q in qis:
        k = normalize_text(q)
        if k not in seen:
            seen.add(k)
            out.append(q)

    return out[:MAX_QI_RETURN]


# =============================================================================
# OPÉRATIONS DE MUE (Nettoyage → Traduction → Standardisation)
# =============================================================================
VERBES_CANON = {
    "montrer": "démontrer", "démontrer": "démontrer", "prouver": "démontrer", "justifier": "démontrer",
    "calculer": "calculer",
    "déterminer": "déterminer", "trouver": "déterminer",
    "étudier": "étudier",
    "résoudre": "résoudre",
    "vérifier": "vérifier",
    "exprimer": "exprimer",
    "encadrer": "encadrer",
    "conclure": "conclure",
    "en déduire": "en déduire", "déduire": "en déduire"
}


def operation_nettoyage(texte: str) -> str:
    """OPÉRATION 1 : Remplacer constantes par invariants."""
    result = texte

    # Canon suites
    result = re.sub(r"\b[uvw]\s*_\s*n\b", "u_n", result, flags=re.IGNORECASE)
    result = re.sub(r"\b[uvw]\s*\(\s*n\s*\)", "u_n", result, flags=re.IGNORECASE)
    result = re.sub(r"la\s+suite\s*\(u_n\)", "une suite", result, flags=re.IGNORECASE)
    result = re.sub(r"la\s+suite\s+u_n", "une suite", result, flags=re.IGNORECASE)

    # Intervalles → invariants
    result = re.sub(r"\[\s*-?\d+[,\.]?\d*\s*[;,]\s*-?\d+[,\.]?\d*\s*\]", "[a;b]", result)
    result = re.sub(r"\[\s*-?\d+[,\.]?\d*\s*[;,]\s*\+?∞\s*\[", "[a;+∞[", result)

    # Valeurs → k
    result = re.sub(r"\best\s+égale?\s+à\s+-?\d+[,\.]?\d*", "est égale à k", result, flags=re.IGNORECASE)
    result = re.sub(r"=\s*-?\d+[,\.]?\d*", "= k", result)

    # Raison chiffrée → raison q
    result = re.sub(r"raison\s+[qr]\s*=\s*-?\d+[,\.]?\d*", "raison q", result, flags=re.IGNORECASE)

    # Années
    result = re.sub(r"\b20\d{2}\b", "", result)

    return re.sub(r"\s+", " ", result).strip()


def operation_traduction(texte: str) -> str:
    """OPÉRATION 2 : Verbe → "Comment + verbe canonique"."""
    result = texte.strip()
    low = result.lower()

    mapping = {
        "montrer": "démontrer", "prouver": "démontrer", "établir": "démontrer", "justifier": "démontrer",
        "démontrer": "démontrer",
        "en déduire": "en déduire", "déduire": "en déduire",
        "calculer": "calculer",
        "déterminer": "déterminer", "trouver": "déterminer",
        "résoudre": "résoudre",
        "étudier": "étudier",
        "vérifier": "vérifier",
        "exprimer": "exprimer",
        "encadrer": "encadrer",
        "conclure": "conclure",
    }

    verbe_trouve = None
    reste = result

    # Multi-mots d'abord
    for v_source in ["en déduire"]:
        if low.startswith(v_source):
            verbe_trouve = mapping[v_source]
            reste = result[len(v_source):].strip()
            break

    if verbe_trouve is None:
        for v_source, v_target in mapping.items():
            if low.startswith(v_source):
                verbe_trouve = v_target
                reste = result[len(v_source):].strip()
                break

    if verbe_trouve:
        if reste.lower().startswith("que ") or reste.lower().startswith("qu'"):
            return f"Comment {verbe_trouve} {reste}"
        if verbe_trouve in ["démontrer", "vérifier"]:
            return f"Comment {verbe_trouve} que {reste}"
        return f"Comment {verbe_trouve} {reste}"

    if not low.startswith("comment"):
        return f"Comment {result}"
    return result


def operation_standardisation(texte: str) -> str:
    """OPÉRATION 3 : Élisions + ponctuation finale."""
    result = texte

    # Élisions françaises
    result = re.sub(r"\bde une\b", "d'une", result, flags=re.IGNORECASE)
    result = re.sub(r"\bque une\b", "qu'une", result, flags=re.IGNORECASE)
    result = re.sub(r"\bde un\b", "d'un", result, flags=re.IGNORECASE)
    result = re.sub(r"\bque un\b", "qu'un", result, flags=re.IGNORECASE)

    # Nettoyer fin
    result = re.sub(r"\s*[\.,:;]+\s*$", "", result).strip()

    # Point d'interrogation
    if not result.endswith("?"):
        result += " ?"
    result = result.replace(" ?", "?")

    return re.sub(r"\s+", " ", result).strip()


def mue_qi_vers_qc(qi_championne: str) -> str:
    """MUE complète : Qi Championne → Titre QC."""
    qi_clean = (qi_championne or "").strip()

    # Garder uniquement la première instruction
    first_end = len(qi_clean)
    for m in re.finditer(r"\.\s+[A-ZÉÈÊËÀÂÙÛÎÏÔÇ+−]", qi_clean):
        first_end = m.start() + 1
        break
    qi_first = qi_clean[:first_end].strip().rstrip(".")

    e0 = _sanitizer.sanitize(qi_first)
    e1 = operation_nettoyage(e0)
    e2 = operation_traduction(e1)
    e3 = operation_standardisation(e2)

    res = e3.strip()
    if res:
        res = res[0].upper() + res[1:]

    # Limite longueur
    if len(res) > 100:
        res = res[:100].rsplit(" ", 1)[0] + "...?"

    return res


# =============================================================================
# VALIDATION QC
# =============================================================================
BANNED_QC_TOKENS = {
    "ce type", "ce genre", "ce problème", "cette question", "traiter", "comment faire",
    "ci-dessus", "ci-dessous", "cet objet mathématique", "l'objet principal",
    "pour tout entier", "pour tout n", "on pose", "on considère", "on admet",
    "soit la suite", "soit f", "soit la fonction"
}


def qc_title_is_valid(title: str) -> bool:
    if not title.startswith("Comment"):
        return False
    if not title.endswith("?"):
        return False
    low = title.lower()
    if any(tok in low for tok in BANNED_QC_TOKENS):
        return False
    if len(title) < 20:
        return False
    # Vérifier que le mot après "Comment" est un verbe, pas "Pour" ou "Soit"
    m = re.match(r"Comment\s+([A-Za-zàâçéèêëîïôûùüÿñæœ]+)", title)
    if m:
        word = m.group(1).lower()
        # Liste des verbes valides
        valid_verbs = {"démontrer", "montrer", "prouver", "calculer", "déterminer", 
                       "justifier", "vérifier", "étudier", "résoudre", "exprimer",
                       "encadrer", "conclure", "en"}  # "en" pour "en déduire"
        if word not in valid_verbs:
            return False
    return True


# =============================================================================
# GÉNÉRATION QC : CHAMPION + MAJORITÉ
# =============================================================================
CHAPTER_DEMANDES = {
    "SUITES NUMÉRIQUES": [
        (r"géométrique", "le caractère géométrique"),
        (r"arithmétique", "le caractère arithmétique"),
        (r"récurrence", "une propriété par récurrence"),
        (r"limite|convergen|tend vers", "la limite"),
        (r"croissant|décroissant|monoton", "la monotonie"),
        (r"born|encadr|major|minor", "un encadrement"),
        (r"raison\b", "la raison"),
        (r"terme général|forme explicite", "le terme général"),
    ],
    "FONCTIONS": [
        (r"unique solution|solution unique", "l'unicité d'une solution"),
        (r"variation|croissant|décroissant", "les variations"),
        (r"limite\b", "une limite"),
        (r"asymptot", "une asymptote"),
        (r"dérivée|f'", "la dérivée"),
    ],
    "PROBABILITÉS": [
        (r"probabilit", "la probabilité"),
        (r"espérance\b", "l'espérance"),
        (r"variance\b", "la variance"),
        (r"indépendant", "l'indépendance"),
    ],
    "GÉOMÉTRIE": [
        (r"distance\b", "une distance"),
        (r"angle\b", "un angle"),
        (r"parall", "le parallélisme"),
        (r"orthogon", "l'orthogonalité"),
    ],
    "INCONNU": [],
}

CHAPTER_OBJETS = {
    "SUITES NUMÉRIQUES": [(r"\b(suite|suites)\b", "une suite"), (r"\bu_n\b", "une suite")],
    "FONCTIONS": [(r"\bfonction\b", "une fonction"), (r"\bf\s*\(\s*x\s*\)", "une fonction")],
    "PROBABILITÉS": [(r"\bprobabilit", "une probabilité"), (r"\bévénement\b", "un événement")],
    "GÉOMÉTRIE": [(r"\bvecteur\b", "un vecteur"), (r"\bdroite\b", "une droite")],
    "INCONNU": [],
}

VERBES_CANON_STRICT = {
    "démontrer": "justifier", "montrer": "justifier", "prouver": "justifier", "justifier": "justifier",
    "calculer": "calculer", "déterminer": "déterminer", "trouver": "déterminer",
    "résoudre": "résoudre", "exprimer": "exprimer", "encadrer": "encadrer",
    "vérifier": "vérifier", "étudier": "étudier", "établir": "établir",
}


def _pick_first_match(text: str, patterns: List[tuple], default: str) -> str:
    t = (text or "").lower()
    for pat, label in patterns:
        if re.search(pat, t, flags=re.IGNORECASE):
            return label
    return default


def extract_action_strict(text: str) -> str:
    t = (text or "").lower()
    m = re.match(r"^\s*([a-zàâçéèêëîïôûùüÿñæœ]+)", t)
    if m:
        w = m.group(1)
        if w in VERBES_CANON_STRICT:
            return VERBES_CANON_STRICT[w]
    for w, canon in VERBES_CANON_STRICT.items():
        if re.search(rf"\b{re.escape(w)}\b", t):
            return canon
    return "déterminer"


def build_qc_title_champion_majorite(cluster_qis: List['QiItem'], qi_champion: 'QiItem') -> str:
    """Vote majoritaire sur Action + Demande + Objet."""
    texts = [q.text for q in cluster_qis if q.text]
    chapter = qi_champion.chapter or "INCONNU"

    actions = Counter(extract_action_strict(t) for t in texts if t)
    demandes = Counter(_pick_first_match(t, CHAPTER_DEMANDES.get(chapter, []), "") for t in texts if t)
    objets = Counter(_pick_first_match(t, CHAPTER_OBJETS.get(chapter, []), "") for t in texts if t)

    # Filtrer vides
    actions = Counter({k: v for k, v in actions.items() if k})
    demandes = Counter({k: v for k, v in demandes.items() if k})
    objets = Counter({k: v for k, v in objets.items() if k})

    action = actions.most_common(1)[0][0] if actions else extract_action_strict(qi_champion.text)
    demande = demandes.most_common(1)[0][0] if demandes else _pick_first_match(qi_champion.text, CHAPTER_DEMANDES.get(chapter, []), "")
    objet = objets.most_common(1)[0][0] if objets else _pick_first_match(qi_champion.text, CHAPTER_OBJETS.get(chapter, []), "")

    if not action:
        action = "déterminer"

    # Éviter redondance
    if objet and demande:
        oc = re.sub(r"^(une|un|la|le)\s+", "", objet.lower()).strip()
        dc = re.sub(r"^(une|un|la|le)\s+", "", demande.lower()).strip()
        if oc and dc and (oc in dc or dc in oc):
            objet = ""

    # Construction
    if objet and demande:
        title = f"Comment {action} {demande} pour {objet}?"
    elif demande:
        title = f"Comment {action} {demande}?"
    elif objet:
        title = f"Comment {action} {objet}?"
    else:
        # Fallback basé sur le chapitre
        if chapter == "SUITES NUMÉRIQUES":
            title = f"Comment {action} une propriété d'une suite?"
        elif chapter == "FONCTIONS":
            title = f"Comment {action} une propriété d'une fonction?"
        elif chapter == "PROBABILITÉS":
            title = f"Comment {action} une probabilité?"
        else:
            title = f"Comment {action} le résultat demandé?"

    title = re.sub(r"\s+", " ", title).strip().replace(" ?", "?")

    if not qc_title_is_valid(title):
        # Essayer MUE
        fb = mue_qi_vers_qc(qi_champion.text)
        if qc_title_is_valid(fb):
            return fb
        # Dernier fallback basé sur chapitre
        if chapter == "SUITES NUMÉRIQUES":
            return f"Comment {action} une propriété d'une suite?"
        return f"Comment {action} le résultat?"

    return title


# =============================================================================
# REPRÉSENTATIVITÉ Qi
# =============================================================================
def compute_qi_representativite(qi_text: str, all_qi_texts: List[str]) -> float:
    qi_tokens = set(tokenize(qi_text))
    if not qi_tokens:
        return 0.0
    all_tokens = Counter()
    for txt in all_qi_texts:
        all_tokens.update(tokenize(txt))
    return float(sum(all_tokens.get(t, 0) for t in qi_tokens))


# =============================================================================
# ARI / FRT / DÉCLENCHEURS
# =============================================================================
CONCEPTS_PATTERNS = [
    (r"unique.*solution|solution.*unique|admet.*une.*seule", "l'unicité d'une solution"),
    (r"suite.*géométrique|géométrique.*raison", "qu'une suite est géométrique"),
    (r"suite.*arithmétique|arithmétique.*raison", "qu'une suite est arithmétique"),
    (r"récurrence|par récurrence", "une propriété par récurrence"),
    (r"limite.*suite|convergence|tend vers", "la limite d'une suite"),
    (r"born|encadr|major|minor", "un encadrement d'une suite"),
    (r"croissante|décroissante|monoton", "la monotonie d'une suite"),
    (r"dérivée|f'|variations", "les variations d'une fonction"),
    (r"probabilité|événement", "une probabilité"),
]

ARI_PAR_CONCEPT = {
    "l'unicité d'une solution": ["1. Vérifier continuité sur [a;b]", "2. Étudier monotonie stricte", "3. Calculer f(a) et f(b)", "4. Appliquer TVI", "5. Conclure unicité"],
    "qu'une suite est géométrique": ["1. Exprimer u(n+1)", "2. Calculer u(n+1)/u(n)", "3. Simplifier", "4. Montrer quotient = constante q", "5. Conclure"],
    "qu'une suite est arithmétique": ["1. Exprimer u(n+1)", "2. Calculer u(n+1) - u(n)", "3. Simplifier", "4. Montrer différence = constante r", "5. Conclure"],
    "une propriété par récurrence": ["1. INITIALISATION : Vérifier P(n₀)", "2. HÉRÉDITÉ : Supposer P(n) vraie", "3. Démontrer P(n+1)", "4. CONCLUSION"],
    "la limite d'une suite": ["1. Identifier la forme de u_n", "2. Factoriser terme dominant", "3. Appliquer limites usuelles", "4. Conclure"],
    "un encadrement d'une suite": ["1. Identifier bornes candidates", "2. Prouver inégalité", "3. Exploiter encadrement", "4. Conclure"],
    "la monotonie d'une suite": ["1. Étudier signe de u(n+1)-u(n)", "2. Déduire croissante/décroissante", "3. Exploiter", "4. Conclure"],
    "les variations d'une fonction": ["1. Calculer f'(x)", "2. Résoudre f'(x) = 0", "3. Tableau signes f'", "4. Tableau variations", "5. Extremums"],
    "une probabilité": ["1. Identifier l'univers", "2. Identifier l'événement", "3. Calculer probabilités", "4. Conclure"],
}

FRT_PAR_CONCEPT = {
    "l'unicité d'une solution": [
        {"type": "usage", "title": "🔔 1. QUAND UTILISER", "text": "Quand l'énoncé demande de montrer qu'une équation admet UNE SEULE solution."},
        {"type": "method", "title": "✅ 2. MÉTHODE", "text": "• f continue sur [a;b]\n• f strictement monotone\n• Calculer f(a) et f(b)\n• Appliquer TVI"},
        {"type": "trap", "title": "⚠️ 3. PIÈGES", "text": "• Oublier 'continue'\n• Oublier 'strictement' monotone"},
        {"type": "conc", "title": "✍️ 4. CONCLUSION", "text": "L'équation admet une unique solution α sur [a;b]."}
    ],
    "qu'une suite est géométrique": [
        {"type": "usage", "title": "🔔 1. QUAND UTILISER", "text": "Quand l'énoncé demande de prouver qu'une suite est géométrique."},
        {"type": "method", "title": "✅ 2. MÉTHODE", "text": "• Calculer u(n+1)/u(n)\n• Simplifier\n• Montrer = constante q"},
        {"type": "trap", "title": "⚠️ 3. PIÈGES", "text": "• Vérifier u(n) ≠ 0\n• Ne pas confondre raison et premier terme"},
        {"type": "conc", "title": "✍️ 4. CONCLUSION", "text": "(u_n) est géométrique de raison q."}
    ],
    "qu'une suite est arithmétique": [
        {"type": "usage", "title": "🔔 1. QUAND UTILISER", "text": "Quand l'énoncé demande de prouver qu'une suite est arithmétique."},
        {"type": "method", "title": "✅ 2. MÉTHODE", "text": "• Calculer u(n+1) - u(n)\n• Simplifier\n• Montrer = constante r"},
        {"type": "trap", "title": "⚠️ 3. PIÈGES", "text": "• Ne pas confondre raison et premier terme"},
        {"type": "conc", "title": "✍️ 4. CONCLUSION", "text": "(u_n) est arithmétique de raison r."}
    ],
    "une propriété par récurrence": [
        {"type": "usage", "title": "🔔 1. QUAND UTILISER", "text": "Quand l'énoncé demande de démontrer 'pour tout n ≥ n₀'."},
        {"type": "method", "title": "✅ 2. MÉTHODE", "text": "• INIT: Vérifier P(n₀)\n• HÉRÉDITÉ: Supposer P(n), montrer P(n+1)\n• CONCLUSION"},
        {"type": "trap", "title": "⚠️ 3. PIÈGES", "text": "• Oublier l'initialisation\n• Oublier 'supposons P(n) vraie'"},
        {"type": "conc", "title": "✍️ 4. CONCLUSION", "text": "Par récurrence, P(n) vraie pour tout n ≥ n₀."}
    ],
    "la limite d'une suite": [
        {"type": "usage", "title": "🔔 1. QUAND UTILISER", "text": "Quand l'énoncé demande de calculer une limite."},
        {"type": "method", "title": "✅ 2. MÉTHODE", "text": "• Identifier la forme\n• Factoriser\n• Appliquer théorèmes"},
        {"type": "trap", "title": "⚠️ 3. PIÈGES", "text": "• Formes indéterminées"},
        {"type": "conc", "title": "✍️ 4. CONCLUSION", "text": "lim u_n = L."}
    ],
}

ARI_GENERIQUE = ["1. Identifier le problème", "2. Appliquer la méthode", "3. Calculer", "4. Conclure"]
FRT_GENERIQUE = [
    {"type": "usage", "title": "🔔 1. QUAND UTILISER", "text": "Identifier les mots-clés de l'énoncé."},
    {"type": "method", "title": "✅ 2. MÉTHODE", "text": "Appliquer la méthode appropriée."},
    {"type": "trap", "title": "⚠️ 3. PIÈGES", "text": "Vérifier les conditions."},
    {"type": "conc", "title": "✍️ 4. CONCLUSION", "text": "Répondre à la question."}
]

DECLENCHEURS_PAR_CONCEPT = {
    "l'unicité d'une solution": ["admet une unique solution", "une seule solution", "solution unique"],
    "qu'une suite est géométrique": ["suite géométrique", "raison q", "quotient constant"],
    "qu'une suite est arithmétique": ["suite arithmétique", "raison r", "différence constante"],
    "une propriété par récurrence": ["par récurrence", "pour tout n", "pour tout entier"],
    "la limite d'une suite": ["limite de la suite", "quand n tend vers", "convergence"],
    "un encadrement d'une suite": ["encadrer", "majorer", "minorer"],
    "la monotonie d'une suite": ["croissante", "décroissante", "monotone"],
    "les variations d'une fonction": ["tableau de variations", "signe de f'"],
    "une probabilité": ["probabilité", "événement", "tirage"],
}


def extraire_concept_cle(qi_texts: List[str]) -> str:
    combined = " ".join(qi_texts).lower()
    for pattern, concept in CONCEPTS_PATTERNS:
        if re.search(pattern, combined):
            return concept
    return "cet objet mathématique"


def extraire_concept_depuis_titre(titre_qc: str) -> str:
    match = re.search(r"Comment (?:justifier|calculer|étudier|déterminer|résoudre|vérifier)\s+(.+)\?$", titre_qc.strip())
    if match:
        return match.group(1).strip()
    return "cet objet mathématique"


def generer_ari(concept: str) -> List[str]:
    return ARI_PAR_CONCEPT.get(concept, ARI_GENERIQUE)


def generer_frt(concept: str) -> List[Dict]:
    return FRT_PAR_CONCEPT.get(concept, FRT_GENERIQUE)


def generer_declencheurs(concept: str, qi_texts: List[str]) -> List[str]:
    declencheurs = DECLENCHEURS_PAR_CONCEPT.get(concept, [])[:3]
    if len(declencheurs) < 4 and qi_texts:
        stopwords = {"les", "des", "une", "pour", "que", "qui", "est", "dans", "par", "sur", "avec"}
        bigrams = Counter()
        for qi in qi_texts:
            toks = re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ]{3,}", qi.lower())
            toks = [t for t in toks if t not in stopwords]
            for i in range(len(toks) - 1):
                bigrams[f"{toks[i]} {toks[i+1]}"] += 1
        for phrase, _ in bigrams.most_common(4):
            if phrase not in declencheurs:
                declencheurs.append(phrase)
    return declencheurs[:5]


# =============================================================================
# CLUSTERING Qi → QC
# =============================================================================
def cluster_qi_to_qc(qis: List[QiItem], sim_threshold: float = SIM_THRESHOLD_DEFAULT) -> List[Dict]:
    """Clustering SMAXIA V6 avec seuil augmenté."""
    if not qis:
        return []

    ALPHA = 5.0
    total_qi = len(qis)

    # 1) Clustering lexical
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
            clusters.append({"rep_tokens": toks, "qis": [qi]})

    qc_out = []
    for c in clusters:
        cluster_qis = c["qis"]
        qi_texts = [q.text for q in cluster_qis]
        n_q = len(cluster_qis)

        # 2) Championne
        best_qi = None
        best_rep = -1.0
        for qi in cluster_qis:
            rep = compute_qi_representativite(qi.text, qi_texts)
            if rep > best_rep:
                best_rep = rep
                best_qi = qi
        if not best_qi:
            best_qi = cluster_qis[0]

        # 3) Titre QC
        titre = build_qc_title_champion_majorite(cluster_qis, best_qi)
        if not qc_title_is_valid(titre):
            titre = mue_qi_vers_qc(best_qi.text)

        # 4) Concept
        concept = extraire_concept_depuis_titre(titre)
        if concept == "cet objet mathématique":
            concept = extraire_concept_cle(qi_texts)

        # 5) ARI/FRT/Triggers
        ari = generer_ari(concept)
        frt_data = generer_frt(concept)
        triggers = generer_declencheurs(concept, qi_texts)

        # 6) F1/F2
        psi_q, sum_tj = compute_psi_q(qi_texts, "Terminale")
        years = [q.year for q in cluster_qis if q.year]
        t_rec = max(0.5, datetime.now().year - max(years)) if years else None
        score = compute_score_f2(n_q, total_qi, t_rec, psi_q, alpha=ALPHA)

        chapter = best_qi.chapter or ""

        qi_by_file = defaultdict(list)
        for q in cluster_qis:
            qi_by_file[q.subject_file].append(q.text)

        evidence = [{"Fichier": f, "Qi": qi} for f, qlist in qi_by_file.items() for qi in qlist]
        evidence_by_subject = [{"Fichier": f, "Qis": qlist, "Count": len(qlist)} for f, qlist in qi_by_file.items()]

        qc_out.append({
            "Chapitre": chapter,
            "QC_ID": "",
            "FRT_ID": "",
            "Titre": titre,
            "Concept": concept,
            "Qi_Championne": best_qi.text[:150] + "..." if len(best_qi.text) > 150 else best_qi.text,
            "Score": score,
            "n_q": n_q,
            "Psi": round(psi_q, 2),
            "N_tot": total_qi,
            "t_rec": round(t_rec, 1) if t_rec else "N/A",
            "Alpha": ALPHA,
            "Sum_Tj": round(sum_tj, 2),
            "Triggers": triggers,
            "ARI": ari,
            "FRT_DATA": frt_data,
            "Evidence": evidence,
            "EvidenceBySubject": evidence_by_subject
        })

    qc_out.sort(key=lambda x: x["Score"], reverse=True)
    for i, qc in enumerate(qc_out):
        qc["QC_ID"] = f"QC-{i+1:02d}"
        qc["FRT_ID"] = f"QC-{i+1:02d}"
    return qc_out


# =============================================================================
# SCRAPING / INGESTION
# =============================================================================
def download_pdf(url: str) -> Optional[bytes]:
    try:
        r = get_session().get(url, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        if len(r.content) > MAX_PDF_MB * 1024 * 1024:
            return None
        return r.content
    except Exception:
        return None


def detect_year(filename: str, text_hint: str) -> Optional[int]:
    m = re.search(r"20[12]\d", filename) or re.search(r"20[12]\d", (text_hint or "")[:1500])
    return int(m.group()) if m else None


def detect_nature(filename: str, text_hint: str) -> str:
    combined = (filename + " " + (text_hint or "")[:1200]).lower()
    if any(k in combined for k in ["bac", "baccalauréat", "métropole", "polynésie", "spécialité"]):
        return "BAC"
    if any(k in combined for k in ["concours", "centrale", "mines", "polytechnique"]):
        return "CONCOURS"
    if any(k in combined for k in ["dst", "devoir surveillé"]):
        return "DST"
    if any(k in combined for k in ["interro", "interrogation"]):
        return "INTERRO"
    return "EXAMEN"


def scrape_pdf_links_bfs(seed_urls: List[str], limit: int) -> List[Dict]:
    """BFS pour trouver des paires Sujet + Corrigé."""
    base = "https://www.apmep.fr"
    queue = list(seed_urls)
    visited = set()

    sujets = []
    corriges = []

    def norm_key(u: str) -> str:
        fn = u.split("/")[-1].lower()
        fn = re.sub(r"(corrig[eé]|correction|corrige)", "", fn)
        fn = re.sub(r"[^a-z0-9]+", "_", fn)
        return fn.strip("_")[:80]

    while queue and len(visited) < 100 and len(sujets) < limit * 3:
        url = queue.pop(0).split("#")[0]
        if url in visited:
            continue
        visited.add(url)

        try:
            r = get_session().get(url, timeout=REQ_TIMEOUT)
            soup = BeautifulSoup(r.text, "html.parser")
        except Exception:
            continue

        for a in soup.find_all("a", href=True):
            href = a["href"]
            hlow = href.lower()
            if ".pdf" in hlow:
                pdf_url = href if href.startswith("http") else urljoin(base + "/", href.lstrip("/"))
                fn = pdf_url.lower().split("/")[-1]
                if any(x in fn for x in ["bulletin", "lettre", "pv1", "doc_"]):
                    continue
                if any(x in fn for x in ["corrig", "correction"]):
                    corriges.append(pdf_url)
                else:
                    sujets.append(pdf_url)
            elif "apmep.fr" in hlow and any(k in hlow for k in ["annee-", "bac-", "terminal", "sujet"]):
                nxt = href if href.startswith("http") else urljoin(base + "/", href.lstrip("/"))
                nxt = nxt.split("#")[0]
                if nxt not in visited:
                    queue.append(nxt)

        time.sleep(0.05)

    # Matching corrigé
    corr_map = {norm_key(c): c for c in corriges}

    result = []
    for s in sujets[:limit]:
        k = norm_key(s)
        corr = corr_map.get(k)
        result.append({"sujet_url": s, "corrige_url": corr})
    return result


def ingest_real(urls: List[str], volume: int, matiere: str, chapter_filter: str = None, progress_callback=None):
    """
    Ingestion V6 : scrape → download → extract lines → extract Qi → GATE chapitre.
    """
    import pandas as pd

    cols_src = ["Fichier", "Nature", "Annee", "Telechargement", "Corrige", "Qi_Data"]
    cols_atm = ["FRT_ID", "Qi", "File", "Year", "Chapitre"]

    seeds = []
    for url in (urls or []):
        if "apmep" in url.lower():
            seeds.extend(SEED_URLS_FRANCE)
        else:
            seeds.append(url)
    if not seeds:
        seeds = SEED_URLS_FRANCE

    sujets_corriges = scrape_pdf_links_bfs(seeds, limit=max(volume * 2, volume + 10))
    if not sujets_corriges:
        return pd.DataFrame(columns=cols_src), pd.DataFrame(columns=cols_atm)

    subjects = []
    all_atoms = []

    def process_one(item: Dict) -> Optional[Dict]:
        pdf_bytes = download_pdf(item["sujet_url"])
        if not pdf_bytes:
            return None

        filename = item["sujet_url"].split("/")[-1]
        lines = extract_pdf_lines(pdf_bytes, max_pages=MAX_PAGES)
        if not lines or len(lines) < 15:
            return None

        # Extraction Qi avec GATE CHAPITRE
        qi_texts = extract_qi_from_lines(lines, chapter_filter=chapter_filter)
        if not qi_texts:
            return None

        hint = " ".join(lines[:60])
        year = detect_year(filename, hint)
        nature = detect_nature(filename, hint)

        atoms = []
        qi_data = []
        for qi in qi_texts:
            chap = chapter_filter if chapter_filter else detect_chapter_strict(qi)
            atoms.append({"FRT_ID": None, "Qi": qi, "File": filename, "Year": year, "Chapitre": chap})
            qi_data.append({"Qi": qi})

        return {
            "Fichier": filename,
            "Nature": nature,
            "Annee": year or "N/A",
            "Telechargement": item["sujet_url"],
            "Corrige": item.get("corrige_url") or "Non trouvé",
            "Qi_Data": qi_data,
            "_atoms": atoms
        }

    wanted = volume
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(process_one, item) for item in sujets_corriges]

        for idx, fut in enumerate(as_completed(futs), 1):
            if progress_callback:
                progress_callback(min(1.0, idx / max(1, len(futs))))
            res = fut.result()
            if not res:
                continue

            subjects.append({k: v for k, v in res.items() if k != "_atoms"})
            all_atoms.extend(res["_atoms"])
            done += 1

            if done >= wanted:
                break

    df_subjects = pd.DataFrame(subjects) if subjects else pd.DataFrame(columns=cols_src)
    df_atoms = pd.DataFrame(all_atoms) if all_atoms else pd.DataFrame(columns=cols_atm)
    return df_subjects, df_atoms


def compute_qc_real(df_atoms):
    import pandas as pd
    if df_atoms is None or df_atoms.empty:
        return pd.DataFrame()

    qis = [
        QiItem(f"S{idx}", row.get("File", ""), row.get("Qi", ""), row.get("Chapitre", ""), row.get("Year"))
        for idx, row in df_atoms.iterrows()
    ]

    qc_list = cluster_qi_to_qc(qis)
    return pd.DataFrame(qc_list) if qc_list else pd.DataFrame()


def compute_saturation_real(df_atoms):
    import pandas as pd
    if df_atoms is None or df_atoms.empty:
        return pd.DataFrame(columns=["Sujets (N)", "QC Total", "Nouvelles QC"])

    files = df_atoms["File"].unique().tolist()
    data = []
    cumul = []
    seen_sigs = set()

    for i, f in enumerate(files):
        cumul.extend(df_atoms[df_atoms["File"] == f].to_dict("records"))
        qis = [
            QiItem(f"S{j}", r.get("File", ""), r.get("Qi", ""), r.get("Chapitre", ""), r.get("Year"))
            for j, r in enumerate(cumul)
        ]
        qc_list = cluster_qi_to_qc(qis)
        sigs = {normalize_text(qc.get("Titre", ""))[:70] for qc in qc_list}
        new_count = len(sigs - seen_sigs)
        seen_sigs.update(sigs)
        data.append({"Sujets (N)": i + 1, "QC Total": len(qc_list), "Nouvelles QC": new_count})

    return pd.DataFrame(data)


# =============================================================================
# AUDITS
# =============================================================================
def audit_internal_real(df_atoms, df_qc) -> Dict:
    if df_atoms is None or df_atoms.empty or df_qc is None or df_qc.empty:
        return {"status": "EMPTY", "coverage": 0.0, "orphans": 0, "total_qi": 0, "covered_qi": 0}

    total_qi = len(df_atoms)
    covered_qi = 0

    if "Evidence" in df_qc.columns:
        for _, row in df_qc.iterrows():
            evidence = row.get("Evidence", [])
            if isinstance(evidence, list):
                covered_qi += len(evidence)

    orphans = max(0, total_qi - covered_qi)
    coverage = (covered_qi / total_qi * 100) if total_qi > 0 else 0.0
    return {
        "status": "PASS" if orphans == 0 else "FAIL",
        "coverage": round(coverage, 1),
        "orphans": orphans,
        "total_qi": total_qi,
        "covered_qi": covered_qi
    }


def audit_external_real(df_atoms_test, df_qc_train, sim_threshold: float = 0.22) -> Dict:
    if df_atoms_test is None or df_atoms_test.empty or df_qc_train is None or df_qc_train.empty:
        return {"status": "EMPTY", "coverage": 0.0, "gaps": 0, "total_test": 0}

    qc_tokens = []
    for _, row in df_qc_train.iterrows():
        parts = [str(row.get("Titre", ""))]
        tr = row.get("Triggers", [])
        if isinstance(tr, list):
            parts.append(" ".join(tr))
        parts.append(str(row.get("Qi_Championne", "")))
        qc_tokens.append(tokenize(" ".join(parts)))

    total = len(df_atoms_test)
    covered = 0

    for _, row in df_atoms_test.iterrows():
        qi = str(row.get("Qi", ""))
        qi_toks = tokenize(qi)
        if not qi_toks:
            continue

        best = max((jaccard_similarity(qi_toks, qct) for qct in qc_tokens), default=0.0)
        if best >= sim_threshold:
            covered += 1

    coverage = (covered / total * 100) if total > 0 else 0.0
    return {
        "status": "PASS" if covered == total else "FAIL",
        "coverage": round(coverage, 1),
        "gaps": total - covered,
        "total_test": total,
        "covered": covered,
        "sim_threshold": sim_threshold
    }


# =============================================================================
# WRAPPER
# =============================================================================
def run_granulo_test(urls: List[str], volume: int, matiere: str = "MATHS", chapter_filter: str = None, progress_callback=None):
    df_sujets, df_atoms = ingest_real(urls, volume, matiere, chapter_filter=chapter_filter, progress_callback=progress_callback)
    df_qc = compute_qc_real(df_atoms)
    df_sat = compute_saturation_real(df_atoms)
    audit = audit_internal_real(df_atoms, df_qc)
    return {
        "sujets": df_sujets,
        "qc": df_qc,
        "saturation": df_sat,
        "audit": audit
    }
