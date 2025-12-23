# ==============================================================================
# SMAXIA – GRANULO ENGINE (TEST)
# Version : TEST F1–F8
# Fichier : smaxia_granulo_engine_test.py
#
# ⚠️ AUCUNE LOGIQUE UI
# ⚠️ AUCUN HARDCODING DE QC
# ⚠️ TOUT EST DÉRIVÉ DES Qi
# ==============================================================================

from dataclasses import dataclass
from typing import List, Dict
import math
import hashlib
from collections import defaultdict

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class Qi:
    """
    Question Individuelle (Qi)
    """
    texte: str
    chapitre: str
    source: str
    annee: int


@dataclass
class ARIStep:
    """
    Étape ARI (Algorithme de Résolution Invariant)
    """
    label: str
    poids: float


@dataclass
class QCResult:
    """
    QC générée par le moteur
    """
    qc_id: str
    chapitre: str
    ari_signature: str
    qi_list: List[Qi]
    score_q: float
    psi: float
    n_q: int


# ==============================================================================
# ARI – RÉFÉRENTIEL INVARIANT (TEST)
# (sera enrichi plus tard – ici strictement TEST)
# ==============================================================================

ARI_LIBRARY = {
    "SUITES_NUM_LIMITES": [
        ARIStep("identifier_terme_dominant", 1.0),
        ARIStep("factoriser", 1.2),
        ARIStep("appliquer_limites_usuelles", 1.5),
        ARIStep("conclure", 0.8),
    ],
    "SUITES_GEOMETRIQUES": [
        ARIStep("exprimer_u_n_plus_1", 1.0),
        ARIStep("calculer_rapport", 1.3),
        ARIStep("simplifier", 1.0),
        ARIStep("identifier_constante", 1.4),
    ],
}


# ==============================================================================
# F1 – SIGNATURE ARI (INVARIANT STRUCTUREL)
# ==============================================================================

def compute_ari_signature(ari_steps: List[ARIStep]) -> str:
    """
    Signature structurelle d'une QC (hash invariant)
    """
    concat = "|".join(step.label for step in ari_steps)
    return hashlib.sha256(concat.encode()).hexdigest()


# ==============================================================================
# F2 – POIDS PRÉDICTIF Ψ(q)
# ==============================================================================
def compute_psi(ari_steps: List[ARIStep]) -> float:
    """
    Ψ(q) = (Σ poids ARI)^2 normalisé
    """
    raw = sum(step.poids for step in ari_steps)
    return round((raw ** 2) / 10.0, 3)


# ==============================================================================
# F3 – SIMILARITÉ σ(qi, qc)
# (TEST : matching lexical simple, remplaçable plus tard)
# ==============================================================================

def similarity_sigma(qi: Qi, ari_key: str) -> float:
    txt = qi.texte.lower()
    if "limite" in txt and "limite" in ari_key.lower():
        return 1.0
    if "géométrique" in txt and "geo" in ari_key.lower():
        return 1.0
    return 0.0


# ==============================================================================
# F4 – GROUPEMENT DES Qi PAR STRUCTURE ARI
# ==============================================================================

def cluster_qi_by_ari(qi_list: List[Qi]) -> Dict[str, List[Qi]]:
    clusters = defaultdict(list)

    for qi in qi_list:
        for ari_key, ari_steps in ARI_LIBRARY.items():
            if similarity_sigma(qi, ari_key) > 0.9:
                clusters[ari_key].append(qi)
                break

    return clusters


# ==============================================================================
# F5 – SCORE GRANULO
# ==============================================================================
def compute_score(n_q: int, psi: float, n_tot: int) -> float:
    """
    Score(q) = (n_q / N_tot) × Ψ × 100
    """
    if n_tot == 0:
        return 0.0
    return round((n_q / n_tot) * psi * 100, 2)


# ==============================================================================
# F6–F8 – MOTEUR GLOBAL
# ==============================================================================

def run_granulo_engine(qi_list: List[Qi]) -> Dict:
    """
    Moteur Granulo TEST
    Entrée : liste de Qi
    Sortie : QC générées + métriques
    """

    clusters = cluster_qi_by_ari(qi_list)
    qc_results: List[QCResult] = []

    n_tot = len(qi_list)
    qc_index = 1

    for ari_key, qis in clusters.items():
        ari_steps = ARI_LIBRARY[ari_key]
        signature = compute_ari_signature(ari_steps)
        psi = compute_psi(ari_steps)
        score = compute_score(len(qis), psi, n_tot)

        qc_results.append(
            QCResult(
                qc_id=f"QC-{qc_index:02d}",
                chapitre=qis[0].chapitre if qis else "UNKNOWN",
                ari_signature=signature,
                qi_list=qis,
                score_q=score,
                psi=psi,
                n_q=len(qis)
            )
        )
        qc_index += 1

    # ============================
    # SORTIE NORMALISÉE
    # ============================
    return {
        "N_Qi": n_tot,
        "N_QC": len(qc_results),
        "QC": [
            {
                "QC_ID": qc.qc_id,
                "Chapitre": qc.chapitre,
                "Score(q)": qc.score_q,
                "Ψ": qc.psi,
                "n_q": qc.n_q,
                "ARI_signature": qc.ari_signature,
                "Qi": [qi.texte for qi in qc.qi_list],
            }
            for qc in qc_results
        ]
    }


# ==============================================================================
# FIN DU FICHIER
# ==============================================================================
