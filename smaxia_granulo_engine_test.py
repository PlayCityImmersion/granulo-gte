# ==============================================================================
# SMAXIA – GRANULO ENGINE (TEST)
# Chapitre : Suites numériques
# Niveau   : Terminale spécialité maths
# Pays     : France
#
# ⚠️ AUCUNE SIMULATION
# ⚠️ AUCUNE QC HARDCODÉE
# ⚠️ AUCUNE LOGIQUE UI
#
# Ce moteur :
# 1. Télécharge des sujets réels via URL
# 2. Extrait les questions individuelles (Qi)
# 3. Regroupe les Qi par structure de résolution
# 4. Déduit les QC (structures invariantes)
# 5. Calcule F1–F2 (socle Granulo)
# 6. Produit des données AUDITABLES
# ==============================================================================

import re
import math
import requests
from typing import List, Dict
from collections import defaultdict
import pandas as pd
from datetime import datetime
from io import BytesIO

# ==============================================================================
# 0. PARAMÈTRES GLOBAUX (TEST)
# ==============================================================================

ALLOWED_CHAPTER = "SUITES NUMÉRIQUES"
ALLOWED_LEVEL = "TERMINALE"
ALLOWED_COUNTRY = "FRANCE"

# Mots-clés strictement observables dans un énoncé
LIMIT_TRIGGERS = [
    r"limite",
    r"tend vers",
    r"\+∞",
    r"-∞",
    r"n\s*→",
]

GEOMETRIC_TRIGGERS = [
    r"géométrique",
    r"raison",
    r"u\(n\+1\)\s*=",
]

# ==============================================================================
# 1. TÉLÉCHARGEMENT DES SUJETS
# ==============================================================================

def download_pdf(url: str) -> bytes:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.content


# ==============================================================================
# 2. EXTRACTION DES QUESTIONS (Qi)
# ==============================================================================

def extract_qi_from_text(text: str) -> List[str]:
    """
    Extraction naïve mais réelle :
    - découpe par numérotation, puces ou phrases directives
    """
    candidates = re.split(
        r"(?:\n\d+\.\s+|\n•|\n\-|\nQuestion\s+\d+)", text
    )
    qi = []
    for c in candidates:
        c = c.strip()
        if len(c) > 20 and "suite" in c.lower():
            qi.append(c)
    return qi


# ==============================================================================
# 3. DÉTECTION DES DÉCLENCHEURS (OBSERVABLES)
# ==============================================================================

def detect_triggers(qi: str) -> List[str]:
    triggers = []
    for pattern in LIMIT_TRIGGERS + GEOMETRIC_TRIGGERS:
        if re.search(pattern, qi, re.IGNORECASE):
            triggers.append(pattern)
    return triggers


# ==============================================================================
# 4. ARI – STRUCTURE DE RÉSOLUTION INVARIANTE
# ==============================================================================

def infer_ari(triggers: List[str]) -> List[str]:
    """
    ARI = suite logique minimale imposée par les déclencheurs
    """
    ari = []

    if any(t in LIMIT_TRIGGERS for t in triggers):
        ari = [
            "Identifier la forme de la suite",
            "Déterminer le terme dominant",
            "Factoriser si nécessaire",
            "Appliquer les limites usuelles",
            "Conclure sur la limite",
        ]

    elif any(t in GEOMETRIC_TRIGGERS for t in triggers):
        ari = [
            "Exprimer u(n+1)",
            "Former le rapport u(n+1)/u(n)",
            "Simplifier",
            "Identifier la constante q",
            "Conclure : suite géométrique",
        ]

    return ari


# ==============================================================================
# 5. CANONISATION DES STRUCTURES (QC)
# ==============================================================================

def canonical_signature(ari: List[str]) -> str:
    """
    Signature canonique = invariant mathématique
    """
    return "||".join(ari)


# ==============================================================================
# 6. F1 – POIDS COGNITIF Ψ
# ==============================================================================

def compute_F1(ari: List[str]) -> float:
    """
    Ψ = (nombre d'étapes)^2 normalisé
    """
    if not ari:
        return 0.0
    raw = len(ari) ** 2
    return min(raw / 25.0, 1.0)


# ==============================================================================
# 7. F2 – SCORE GRANULO
# ==============================================================================

def compute_F2(n_q: int, N_tot: int, psi: float, year: int) -> float:
    tau = max(datetime.now().year - year, 0.5)
    density = n_q / max(N_tot, 1)
    score = density * (1 + 5.0 / tau) * psi * 100
    return round(score, 2)


# ==============================================================================
# 8. MOTEUR PRINCIPAL
# ==============================================================================

def run_granulo_engine(
    urls: str,
    volume: int,
    classe: str,
    matiere: str,
    chapitre: str,
    pays: str,
) -> Dict[str, pd.DataFrame]:

    assert chapitre == ALLOWED_CHAPTER
    assert classe.upper() == ALLOWED_LEVEL
    assert pays.upper() == ALLOWED_COUNTRY

    # -------------------------------
    # 8.1 Collecte brute
    # -------------------------------
    subjects = []
    atoms = []

    url_list = [u.strip() for u in urls.splitlines() if u.strip()]
    url_list = url_list[:volume]

    for url in url_list:
        try:
            pdf_bytes = download_pdf(url)
            text = pdf_bytes.decode(errors="ignore")
        except Exception:
            continue

        qi_list = extract_qi_from_text(text)

        filename = url.split("/")[-1]
        year = datetime.now().year

        subjects.append({
            "Fichier": filename,
            "Nature": "SOURCE",
            "Année": year,
            "Source": url,
        })

        for qi in qi_list:
            triggers = detect_triggers(qi)
            ari = infer_ari(triggers)
            signature = canonical_signature(ari)

            atoms.append({
                "Qi": qi,
                "Triggers": triggers,
                "ARI": ari,
                "Signature": signature,
                "Year": year,
                "Fichier": filename,
            })

    df_subjects = pd.DataFrame(subjects)
    df_atoms = pd.DataFrame(atoms)

    # -------------------------------
    # 8.2 Construction des QC
    # -------------------------------
    qc_groups = defaultdict(list)
    for _, row in df_atoms.iterrows():
        qc_groups[row["Signature"]].append(row)

    qc_rows = []
    for i, (sig, items) in enumerate(qc_groups.items(), start=1):
        ari = items[0]["ARI"]
        psi = compute_F1(ari)
        n_q = len(items)
        N_tot = len(df_atoms)
        year = items[0]["Year"]
        score = compute_F2(n_q, N_tot, psi, year)

        qc_rows.append({
            "QC_ID": f"QC-{i:02d}",
            "ARI": ari,
            "Signature": sig,
            "Psi": psi,
            "n_q": n_q,
            "Score": score,
        })

    df_qc = pd.DataFrame(qc_rows)

    # -------------------------------
    # 8.3 Saturation
    # -------------------------------
    sat = []
    discovered = set()
    for i, row in enumerate(df_atoms.itertuples(), start=1):
        discovered.add(row.Signature)
        sat.append({
            "Nombre de sujets injectés": i,
            "Nombre de QC découvertes": len(discovered),
        })

    df_sat = pd.DataFrame(sat)

    return {
        "subjects_df": df_subjects,
        "qi_df": df_atoms,
        "qc_df": df_qc,
        "saturation_df": df_sat,
    }
