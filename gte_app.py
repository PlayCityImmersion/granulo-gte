import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(layout="wide", page_title="SMAXIA - Moteur F2 (Correction)")

st.title("🛡️ SMAXIA - Moteur de Sélection F2 (Correctif)")
st.markdown("### Algorithme de sélection des 15 meilleures QC (Anti-Redondance)")

# --- 1. CONFIGURATION DES PARAMÈTRES (A2) ---
ALPHA = 365.0       # Coefficient de récence (1 an pèse lourd)
SEUIL_SIMILARITE = 0.1 # Sigma : Si similarité > 0.1, on pénalise
NB_TARGET = 15      # Objectif : 15 QC optimales

# --- 2. JEU DE DONNÉES SIMULÉ (CORPUS HISTORIQUE P3) ---
CANDIDATE_POOL = [
    {"id": "ANA_LIM_INF", "txt": "Calculer la limite en +infini", "years": [2015, 2018, 2021, 2023, 2024], "trigs": {"calculer", "limite", "infini"}},
    {"id": "ANA_LIM_POINT", "txt": "Calculer la limite en un point", "years": [2016, 2019], "trigs": {"calculer", "limite", "point"}},
    {"id": "ANA_DERIV_VAR", "txt": "Étudier les variations de la fonction", "years": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024], "trigs": {"variations", "dérivée"}},
    {"id": "ANA_PRIM_UNIQUE", "txt": "Déterminer la primitive F qui s'annule en 0", "years": [2018, 2022, 2024], "trigs": {"primitive", "unique", "condition"}},
    {"id": "ANA_PRIM_GEN", "txt": "Déterminer une primitive quelconque", "years": [2017, 2021], "trigs": {"primitive", "fonction"}}, 
    {"id": "GEO_ORTHO", "txt": "Démontrer que la droite est orthogonale au plan", "years": [2019, 2023, 2024], "trigs": {"orthogonal", "plan", "droite"}},
    {"id": "GEO_COPLAN", "txt": "Justifier que les points sont coplanaires", "years": [2020, 2022], "trigs": {"coplanaires", "points"}},
    {"id": "PROBA_LOI_NORM", "txt": "Calculer une probabilité loi normale", "years": [2021, 2022, 2023, 2024], "trigs": {"loi", "normale", "probabilité"}},
    {"id": "PROBA_BINOM", "txt": "Justifier le schéma de Bernoulli", "years": [2015, 2016], "trigs": {"bernoulli", "binomiale"}},
    {"id": "SUITE_REC", "txt": "Démontrer par récurrence que Un > 0", "years": [2015, 2017, 2019, 2021, 2023], "trigs": {"récurrence", "initialisation"}},
    {"id": "SUITE_GEO", "txt": "Montrer que la suite est géométrique", "years": [2016, 2018, 2020, 2022, 2024], "trigs": {"géométrique", "raison"}},
    {"id": "COMPLEXE_ALG", "txt": "Déterminer la forme algébrique", "years": [2015, 2018], "trigs": {"algébrique", "complexe"}},
    {"id": "COMPLEXE_GEO", "txt": "Déterminer l'ensemble des points M", "years": [2017, 2019, 2023], "trigs": {"ensemble", "points", "affixe"}},
    {"id": "EQUA_DIFF", "txt": "Résoudre l'équation différentielle (E)", "years": [2015, 2016, 2020], "trigs": {"équation", "différentielle"}},
    {"id": "INT_CALCUL", "txt": "Calculer l'intégrale I", "years": [2019, 2021, 2023], "trigs": {"intégrale", "calculer"}},
    {"id": "INT_AIRE", "txt": "Interpréter géométriquement l'intégrale (Aire)", "years": [2018, 2022], "trigs": {"aire", "intégrale", "unités"}} 
]

# --- 3. FONCTIONS MATHÉMATIQUES (DEFINITIONS STRICTES) ---

def calc_psi(text):
    """Ψ_q : Potentiel d'Impact Cognitif (Densité sémantique)"""
    words = re.findall(r'\w+', text.lower())
    stopwords = ["le", "la", "de", "une", "que", "est", "les", "en"]
    meaningful = [w for w in words if w not in stopwords and len(w) > 2]
    return round(len(set(meaningful)) / len(words), 3) if words else 0

def calc_sigma(trigs_q, trigs_p):
    """σ(q,p) : Similarité vectorielle (Jaccard sur les triggers)"""
    # Mesure le recouvrement entre deux QC. Si elles partagent trop de triggers, Sigma augmente.
    if not isinstance(trigs_q, set): trigs_q = set(trigs_q)
    if not isinstance(trigs_p, set): trigs_p = set(trigs_p)
    
    intersection = len(trigs_q.intersection(trigs_p))
    union = len(trigs_q.union(trigs_p))
    return intersection / union if union > 0 else 0

def calc_time_rec(years):
    """t_rec : Temps écoulé (en jours approx) depuis la dernière occurrence"""
    current_year = datetime.now().year
    last_year = max(years)
    delta_years = current_year - last_year
    # On convertit en jours pour la formule, minimum 1 jour pour éviter div/0
    t_rec_days = max(delta_years * 365, 1) 
    return t_rec_days

# --- 4. MOTEUR DE SÉLECTION (ARGMAX LOOP) ---

def run_smaxia_selection(candidates):
    logs = [] # Pour stocker l'historique des pénalités
    
    # 1. Calcul de N_total (Volume historique total des occurrences)
    N_total_occurrences = sum(len(c["years"]) for c in candidates)
    
    # 2. Pré-calcul des scores "Base" (Intrinsèques)
    pool = []
    for c in candidates:
        n_q = len(c["years"])
        t_rec = calc_time_rec(c["years"])
        psi = calc_psi(c["txt"])
        
        # Bloc Fréquence
        freq_term = n_q / N_total_occurrences
        
        # Bloc Récence
        recency_term = 1 + (ALPHA / t_rec)
        
        # Bloc Valeur
        base_score = freq_term * recency_term * psi * 100 # x100 pour lisibilité
        
        pool.append({
            "id": c["id"],
            "obj": c,
            "base_score": base_score,
            "current_score": base_score,
            "n_q": n_q,
            "t_rec": t_rec,
            "psi": psi,
            "selected": False
        })

    # 3. Boucle de Sélection (Argmax itératif)
    selected_qcs = []
    
    while len(selected_qcs) < NB_TARGET and len(pool) > len(selected_qcs):
        # A. Trouver le MAX parmi les non-sélectionnés
        candidates_left = [p for p in pool if not p["selected"]]
        if not candidates_left: break
        
        # Argmax
        best_candidate = max(candidates_left, key=lambda x: x["current_score"])
        
        # B. Sélectionner
        best_candidate["selected"] = True
        rank = len(selected_qcs) + 1
        selected_qcs.append(best_candidate)
        
        # C. Mettre à jour les scores des RESTANTS (Anti-Redondance)
        for item in pool:
            if not item["selected"]:
                # Calcul de Sigma entre l'item et celui qu'on vient de choisir
                sigma = calc_sigma(item["obj"]["trigs"], best_candidate["obj"]["trigs"])
                
                # Application de la pénalité si Sigma significatif
                penalty_factor = (1 - sigma)
                
                # Mise à jour du score courant
                old_score = item["current_score"]
                item["current_score"] *= penalty_factor
                
                if sigma > SEUIL_SIMILARITE:
                    logs.append(f"⚠️ Pénalité Redondance sur **{item['id']}** (Sim avec {best_candidate['id']} = {sigma:.2f}) : {old_score:.4f} -> {item['current_score']:.4f}")

    return pd.DataFrame(selected_qcs), logs

# --- 5. INTERFACE UTILISATEUR (STREAMLIT) ---

if st.button("LANCER LE CALCULATEUR F2 🚀"):
    df_result, logs = run_smaxia_selection(CANDIDATE_POOL)
    
    st.success(f"Calcul terminé. {len(df_result)} QC sélectionnées.")
    
    # Affichage du Tableau Principal
