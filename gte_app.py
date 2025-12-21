import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Moteur F2 (Safe Mode)")
st.title("🛡️ SMAXIA - Moteur F2 (Mode Robuste)")

# --- PARAMÈTRES ---
ALPHA = 365.0
SEUIL_SIMILARITE = 0.1
NB_TARGET = 15

# --- DONNÉES SIMULÉES ---
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

# --- FONCTIONS ---
def calc_psi(text):
    words = re.findall(r'\w+', text.lower())
    stopwords = ["le", "la", "de", "une", "que", "est", "les", "en"]
    meaningful = [w for w in words if w not in stopwords and len(w) > 2]
    return round(len(set(meaningful)) / len(words), 3) if words else 0

def calc_sigma(trigs_q, trigs_p):
    if not isinstance(trigs_q, set): trigs_q = set(trigs_q)
    if not isinstance(trigs_p, set): trigs_p = set(trigs_p)
    intersection = len(trigs_q.intersection(trigs_p))
    union = len(trigs_q.union(trigs_p))
    return intersection / union if union > 0 else 0

def calc_time_rec(years):
    current_year = datetime.now().year
    last_year = max(years)
    delta_years = current_year - last_year
    t_rec_days = max(delta_years * 365, 1) 
    return t_rec_days

def run_smaxia_selection(candidates):
    logs = []
    N_total_occurrences = sum(len(c["years"]) for c in candidates)
    
    pool = []
    for c in candidates:
        n_q = len(c["years"])
        t_rec = calc_time_rec(c["years"])
        psi = calc_psi(c["txt"])
        freq_term = n_q / N_total_occurrences
        recency_term = 1 + (ALPHA / t_rec)
        base_score = freq_term * recency_term * psi * 100 
        
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

    selected_qcs = []
    
    while len(selected_qcs) < NB_TARGET and len(pool) > len(selected_qcs):
        candidates_left = [p for p in pool if not p["selected"]]
        if not candidates_left: break
        
        best_candidate = max(candidates_left, key=lambda x: x["current_score"])
        
        best_candidate["selected"] = True
        selected_qcs.append(best_candidate)
        
        for item in pool:
            if not item["selected"]:
                sigma = calc_sigma(item["obj"]["trigs"], best_candidate["obj"]["trigs"])
                penalty_factor = (1 - sigma)
                old_score = item["current_score"]
                item["current_score"] *= penalty_factor
                
                if sigma > SEUIL_SIMILARITE:
                    logs.append(f"Pénalité sur {item['id']} (Sim avec {best_candidate['id']} = {sigma:.2f})")

    return pd.DataFrame(selected_qcs), logs

# --- INTERFACE ---
if st.button("LANCER LE CALCULATEUR"):
    df_result, logs = run_smaxia_selection(CANDIDATE_POOL)
    
    st.success(f"Calcul terminé. {len(df_result)} QC sélectionnées.")
    
    st.subheader("1. Tableau des Résultats")
    # Affichage SIMPLE sans configuration avancée
    st.dataframe(df_result[["id", "current_score", "base_score", "n_q", "t_rec", "psi"]])
    
    st.subheader("2. Logs Anti-Redondance")
    if logs:
        st.write(logs)
    else:
        st.write("Aucune pénalité majeure.")
