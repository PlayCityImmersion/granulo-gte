# ==============================================================================
# SMAXIA – GRANULO ENGINE (TEST)
# F1 → F8 – ENVIRONNEMENT DE VALIDATION SCIENTIFIQUE
# Aucun affichage UI – Aucun hardcoding QC
# ==============================================================================

import math
import hashlib
from collections import defaultdict
from datetime import datetime

# ==============================================================================
# STRUCTURES DE BASE
# ==============================================================================

class Qi:
    """
    Question individuelle extraite d’un sujet
    """
    def __init__(self, texte, chapitre, source, annee):
        self.texte = texte
        self.chapitre = chapitre
        self.source = source
        self.annee = annee


class QC:
    """
    Question Clé construite par agrégation de Qi
    """
    def __init__(self, signature):
        self.signature = signature          # invariant ARI
        self.qis = []                        # Qi associées
        self.chapitre = None
        self.creation_year = datetime.now().year

    def add_qi(self, qi: Qi):
        self.qis.append(qi)
        self.chapitre = qi.chapitre


# ==============================================================================
# F1 – EXTRACTION DES INVARIANTS ARI
# ==============================================================================

def F1_extract_ari(qi: Qi):
    """
    Transforme une Qi en signature ARI minimale.
    Ici : version test → normalisation lexicale + patterns cognitifs.
    """
    txt = qi.texte.lower()

    ari = []
    if "limite" in txt:
        ari.append("LIMIT")
    if "dériver" in txt or "variation" in txt:
        ari.append("VARIATION")
    if "suite" in txt:
        ari.append("SUITE")
    if "convergence" in txt:
        ari.append("CONVERGENCE")
    if "démontrer" in txt or "montrer" in txt:
        ari.append("PROUVER")

    if not ari:
        ari.append("AUTRE")

    return tuple(sorted(set(ari)))


# ==============================================================================
# F2 – SIGNATURE CANONIQUE (HASH)
# ==============================================================================

def F2_signature(ari_tuple):
    """
    Génère une signature unique et stable à partir de l’ARI
    """
    raw = "|".join(ari_tuple)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ==============================================================================
# F3 – AGRÉGATION Qi → QC
# ==============================================================================

def F3_cluster_qi(qi_list):
    """
    Regroupe les Qi par signature ARI
    """
    qc_map = {}

    for qi in qi_list:
        ari = F1_extract_ari(qi)
        sig = F2_signature(ari)

        if sig not in qc_map:
            qc_map[sig] = QC(signature=sig)

        qc_map[sig].add_qi(qi)

    return list(qc_map.values())


# ==============================================================================
# F4 – SCORE PREDICTIF Ψ(q)
# ==============================================================================

def F4_score_psi(qc: QC):
    """
    Ψ(q) – poids prédictif de la QC
    Version test : croissance logarithmique contrôlée
    """
    n_q = len(qc.qis)
    return round(min(1.0, math.log(1 + n_q) / 3), 2)


# ==============================================================================
# F5 – DENSITÉ RELATIVE
# ==============================================================================

def F5_density(qc: QC, total_qi):
    return len(qc.qis) / max(1, total_qi)


# ==============================================================================
# F6 – RÉCENCE
# ==============================================================================

def F6_recency(qc: QC):
    age = datetime.now().year - qc.creation_year
    return max(1.0, 1 + age)


# ==============================================================================
# F7 – SCORE GRANULO GLOBAL
# ==============================================================================

def F7_score_granulo(qc: QC, total_qi):
    psi = F4_score_psi(qc)
    density = F5_density(qc, total_qi)
    recency = F6_recency(qc)

    score = 100 * psi * density * (1 / recency)
    return round(score, 1)


# ==============================================================================
# F8 – SATURATION (CONVERGENCE QC)
# ==============================================================================

def F8_saturation_curve(qc_counts):
    """
    qc_counts : liste cumulative [QC après N sujets]
    """
    saturation = []
    prev = 0

    for n, qc in enumerate(qc_counts, start=1):
        delta = qc - prev
        saturation.append({
            "N_sujets": n,
            "QC_total": qc,
            "Δ_QC": delta
        })
        prev = qc

    return saturation


# ==============================================================================
# PIPELINE GLOBAL – APPEL UNIQUE
# ==============================================================================

def run_granulo_engine(qi_list):
    """
    Point d’entrée unique du moteur Granulo (TEST)
    """
    total_qi = len(qi_list)

    # F3
    qcs = F3_cluster_qi(qi_list)

    # Calcul scores
    results = []
    for qc in qcs:
        results.append({
            "QC_signature": qc.signature,
            "Chapitre": qc.chapitre,
            "n_q": len(qc.qis),
            "Ψ": F4_score_psi(qc),
            "Score(q)": F7_score_granulo(qc, total_qi),
            "Qi": qc.qis
        })

    return {
        "QC": results,
        "QC_count": len(qcs)
    }
