# =============================================================================
# SMAXIA — Granulo Engine (TEST)
# Fichier : smaxia_granulo_engine_test.py
# Périmètre : France | Terminale | Spécialité Maths | Suites Numériques
# MODE : TEST SCIENTIFIQUE — AUCUNE SIMULATION — AUCUN HARDCODE QC
# =============================================================================

import requests
import re
import math
import uuid
from typing import List, Dict, Tuple
from collections import defaultdict
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# PARAMÈTRES GÉNÉRAUX (TEST)
# =============================================================================

HEADERS = {
    "User-Agent": "SMAXIA-Granulo-Test/1.0"
}

CHAPITRE_CIBLE = "SUITES NUMÉRIQUES"
NIVEAU = "TERMINALE"
PAYS = "FRANCE"

EPSILON = 0.1

# =============================================================================
# OUTILS BAS NIVEAU
# =============================================================================

def download_html(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.text


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def split_into_questions(text: str) -> List[str]:
    """
    Extraction STRICTE de Qi :
    - phrases interrogatives
    - consignes évaluatives classiques
    """
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 20]

    qi = []
    for l in lines:
        if (
            "limite" in l.lower()
            or "suite" in l.lower()
            or "u_n" in l.lower()
            or "u(n)" in l.lower()
            or "déterminer" in l.lower()
            or "montrer que" in l.lower()
            or "calculer" in l.lower()
            or "étudier" in l.lower()
        ):
            qi.append(l)

    return list(dict.fromkeys(qi))  # déduplication stricte


# =============================================================================
# ARI — ALGORITHME DE RÉSOLUTION INVARIANT
# =============================================================================

def build_ari_vector(qi_text: str) -> np.ndarray:
    """
    Construction d’un vecteur ARI RÉEL.
    Chaque composante = présence d’une transformation cognitive.
    """

    transformations = {
        "factorisation": ["factoriser", "terme dominant"],
        "limite": ["limite", "tend vers", "+∞", "-∞"],
        "quotient": ["/"],
        "difference": ["-"],
        "comparaison": ["major", "minor", "born"],
        "recurrence": ["récurrence"],
    }

    vector = []
    text = qi_text.lower()

    for _, keywords in transformations.items():
        vector.append(1.0 if any(k in text for k in keywords) else 0.0)

    return np.array(vector, dtype=float)


# =============================================================================
# F1 — POIDS PRÉDICTIF PURIFIÉ
# =============================================================================

def compute_psi(ari_vector: np.ndarray, delta_c: float = 1.0) -> float:
    tj_sum = np.sum(ari_vector)
    psi_raw = delta_c * (EPSILON + tj_sum) ** 2
    return psi_raw


# =============================================================================
# CLUSTERING QC (INVARIANTS)
# =============================================================================

def cluster_qi_into_qc(qi_df: pd.DataFrame, similarity_threshold=0.95):
    """
    QC = cluster de Qi ayant des ARI quasi identiques
    """
    vectors = np.stack(qi_df["ari_vector"].values)
    sim = cosine_similarity(vectors)

    visited = set()
    clusters = []

    for i in range(len(qi_df)):
        if i in visited:
            continue

        cluster_idx = {i}
        for j in range(len(qi_df)):
            if sim[i, j] >= similarity_threshold:
                cluster_idx.add(j)

        visited |= cluster_idx
        clusters.append(cluster_idx)

    return clusters


# =============================================================================
# F2 — SCORE GRANULO
# =============================================================================

def compute_score_f2(n_q, N_tot, psi, t_rec):
    density = n_q / max(N_tot, 1)
    recency = 1 + 5.0 / max(t_rec, 0.5)
    return density * recency * psi * 100


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_granulo_pipeline(urls: List[str], volume: int):
    """
    === FONCTION UNIQUE À APPELER DEPUIS L’UI ===
    """

    # -------------------------------------------------------------------------
    # 1. RÉCUPÉRATION DES SUJETS
    # -------------------------------------------------------------------------
    subjects = []
    qi_rows = []

    for url in urls[:volume]:
        try:
            html = download_html(url)
            text = extract_text_from_html(html)
            qi_list = split_into_questions(text)

            doc_id = str(uuid.uuid4())[:8]

            for qi in qi_list:
                ari = build_ari_vector(qi)

                qi_rows.append({
                    "qi_id": str(uuid.uuid4()),
                    "texte": qi,
                    "chapitre": CHAPITRE_CIBLE,
                    "ari_vector": ari,
                    "document_id": doc_id,
                    "source": url
                })

            subjects.append({
                "document_id": doc_id,
                "source": url,
                "nb_qi": len(qi_list)
            })

        except Exception as e:
            print(f"[ERREUR] {url} → {e}")

    qi_df = pd.DataFrame(qi_rows)
    subjects_df = pd.DataFrame(subjects)

    if qi_df.empty:
        return subjects_df, qi_df, pd.DataFrame(), pd.DataFrame()

    # -------------------------------------------------------------------------
    # 2. CLUSTERING → QC
    # -------------------------------------------------------------------------
    clusters = cluster_qi_into_qc(qi_df)

    qc_rows = []
    N_tot = len(qi_df)

    for idx, cluster in enumerate(clusters, start=1):
        cluster_qi = qi_df.iloc[list(cluster)]
        psi_vals = cluster_qi["ari_vector"].apply(compute_psi)

        psi_max = psi_vals.max()
        t_rec = 1.0  # test (pas encore temporel)

        score = compute_score_f2(
            n_q=len(cluster_qi),
            N_tot=N_tot,
            psi=psi_max,
            t_rec=t_rec
        )

        qc_rows.append({
            "QC_ID": f"QC-{idx:02d}",
            "chapitre": CHAPITRE_CIBLE,
            "n_q": len(cluster_qi),
            "Psi": round(psi_max, 3),
            "Score": round(score, 2),
            "qi_ids": cluster_qi["qi_id"].tolist(),
        })

    qc_df = pd.DataFrame(qc_rows)

    # -------------------------------------------------------------------------
    # 3. SATURATION
    # -------------------------------------------------------------------------
    sat_points = []
    seen_qc = set()

    for i in range(1, len(qi_df) + 1):
        partial = qi_df.iloc[:i]
        clusters_i = cluster_qi_into_qc(partial)
        sat_points.append({
            "Nombre de Qi": i,
            "Nombre de QC": len(clusters_i)
        })

    saturation_df = pd.DataFrame(sat_points)

    # -------------------------------------------------------------------------
    # SORTIES OFFICIELLES
    # -------------------------------------------------------------------------
    return subjects_df, qi_df, qc_df, saturation_df


# =============================================================================
# FIN DU FICHIER
# =============================================================================
