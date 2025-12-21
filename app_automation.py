import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console CEO V2")

# --- 1. BIBLIOTHÈQUE DE CONTENU (GÉNÉRATEUR DE VARIANTES) ---
# On simule ici que pour une même compétence, les phrases changent légèrement d'une année à l'autre.
CONTENT_GENERATOR = {
    "DERIVATION": [
        "Étudier les variations de la fonction f sur l'intervalle I",
        "Dresser le tableau de variations complet de g",
        "Déterminer les variations de la fonction h",
        "Justifier le sens de variation de la suite (Un)"
    ],
    "LIMITES": [
        "Calculer la limite de f en +infini",
        "Déterminer la limite de la suite (Un) quand n tend vers l'infini",
        "Quelle est la limite de f en 0 ?",
        "En déduire l'existence d'une asymptote"
    ],
    "GEOMETRIE": [
        "Démontrer que la droite (D) est orthogonale au plan (P)",
        "Prouver que les vecteurs u et v sont orthogonaux",
        "Déterminer une représentation paramétrique de la droite",
        "Vérifier que le point A appartient au plan (P)"
    ],
    "INTEGRATION": [
        "Déterminer une primitive F de f sur R",
        "Calculer l'intégrale I entre a et b",
        "Montrer que F est une primitive de f",
        "En déduire l'aire sous la courbe"
    ]
}

# --- 2. FONCTIONS MOTEUR ---

def simulate_smart_crawl(url, n_sujets):
    """
    Simule la récupération de N sujets.
    Chaque sujet contient entre 3 et 5 Qi (Exercices).
    Cela rend N_total réaliste (N_sujets * 4 approx).
    """
    crawled_atoms = []
    progress = st.progress(0)
    status = st.empty()
    
    total_steps = n_sujets
    
    for i in range(n_sujets):
        # Simulation visuelle
        if i % 5 == 0: # On met à jour tous les 5 pour aller vite
            progress.progress((i+1)/total_steps)
            status.text(f"Scraping : {url}/sujet_bac_{2024-i%8}_{i}.pdf ... Extraction Qi...")
        
        # GÉNÉRATION DU CONTENU DU SUJET (3 à 5 exercices par sujet)
        nb_exos = random.randint(3, 5)
        year = random.choice(range(2016, 2025))
        
        for _ in range(nb_exos):
            # Choix d'un thème au hasard
            theme = random.choice(list(CONTENT_GENERATOR.keys()))
            # Choix d'une formulation (Variante)
            raw_text = random.choice(CONTENT_GENERATOR[theme])
            
            # Mapping simulé
            chapitre_map = {
                "DERIVATION": "Dérivation", "LIMITES": "Limites", 
                "GEOMETRIE": "Espace", "INTEGRATION": "Intégration"
            }
            
            crawled_atoms.append({
                "ID_Sujet": f"SUJET_{i:03d}",
                "Année": year,
                "Niveau": "TERMINALE", # Fixé pour l'exemple
                "Matière": "MATHS",
                "Chapitre": chapitre_map[theme],
                "Qi_Brut": raw_text,
                "Déclencheur": raw_text.split()[0].upper() # Ex: CALCULER, ETUDIER
            })
            
    progress.empty()
    status.empty()
    return pd.DataFrame(crawled_atoms)

def process_engine_logic(df_atoms):
    """
    Calcule les invariants QC et les scores.
    """
    N_total_global = len(df_atoms) # C'est le VRAI volume d'atomes (ex: 85)
    current_year = datetime.now().year
    
    # 1. Normalisation QC (On regroupe les variantes sous une même bannière "COMMENT")
    # Dans la vraie vie, c'est le moteur sémantique. Ici on simule par string matching simple.
    df_atoms["QC_Tag"] = df_atoms["Qi_Brut"].apply(lambda x: "COMMENT " + " ".join(x.split()[:4]) + "...")
    
    # 2. Agrégation
    grouped = df_atoms.groupby(["Niveau", "Chapitre", "QC_Tag"]).agg({
        "Qi_Brut": list,           # On garde la LISTE des sources (Preuve)
        "ID_Sujet": "count",       # n_q (Fréquence)
        "Année": "max"             # Récence max
    }).reset_index()
    
    results = []
    
    for idx, row in grouped.iterrows():
        n_q = row["ID_Sujet"]
        qi_list = row["Qi_Brut"]
        
        # Variables SMAXIA
        delta_t = (current_year - row["Année"])
        tau = delta_t if delta_t > 0 else 0.5
        alpha = 5.0
        
        # Psi moyen des phrases sources
        psi_avg = np.mean([len(set(s.split()))/len(s.split()) for s in qi_list])
        
        # Calcul Score
        freq = n_q / N_total_global
        recency = 1 + (alpha / max(tau, 1))
        
        score = freq * recency * psi_avg * 100
        
        results.append({
            "NIVEAU": row["Niveau"],
            "CHAPITRE": row["Chapitre"],
            "QC_CIBLE": row["QC_Tag"],
            "SCORE_F2": score,
            "n_q": n_q,
            "N_tot": N_total_global,
            "Tau": tau,
            "Psi": round(psi_avg, 2),
            "SOURCES_QI": qi_list # La liste pour l'audit
        })
        
    return pd.DataFrame(results).sort_values(by="SCORE_F2", ascending=False)

# --- INTERFACE ---
st.title("🚀 SMAXIA - Console CEO V2 (Audit Ready)")

# 1. INPUTS
with st.container():
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        url = st.text_input("URL Cible", value="https://www.apmep.fr/")
    with c2:
        n_sujets = st.number_input("Nombre de Sujets PDF", value=20, step=5)
    with c3:
        st.write("")
        run = st.button("LANCER L'ANALYSE", type="primary")

if run:
    st.divider()
    
    # 2. PROCESS
    with st.spinner("Simulation Extraction & Atomisation..."):
        df_raw = simulate_smart_crawl(url, n_sujets)
        df_final = process_engine_logic(df_raw)
    
    # 3. GLOBAL STATS (KPIs)
    st.subheader("📊 Métriques Globales")
    k1, k2, k3 = st.columns(3)
    k1.metric("Sujets Traités", n_sujets)
    k2.metric("N_total (Atomes Qi)", len(df_raw), delta="Volume de calcul")
    k3.metric("QC Identifiées", len(df_final))
    
    st.divider()
    
    # 4. LIVRABLE DÉTAILLÉ
    st.header("📦 Livrable SMAXIA (Classé par Pertinence)")
    
    # Onglets par Chapitre (plus propre que tout d'un coup)
    chapitres = df_final["CHAPITRE"].unique()
    tabs = st.tabs(list(chapitres))
    
    for i, chap in enumerate(chapitres):
        with tabs[i]:
            df_chap = df_final[df_final["CHAPITRE"] == chap]
            
            for index, row in df_chap.iterrows():
                # AFFICHAGE CARTE QC
                with st.container():
                    # En-tête de la QC
                    col_score, col_qc = st.columns([1, 4])
                    
                    with col_score:
                        st.metric("Score F2", f"{row['SCORE_F2']:.1f}")
                    
                    with col_qc:
                        st.subheader(f"🗝️ {row['QC_CIBLE']}")
                        st.caption(f"Fréquence: **{row['n_q']}** / {row['N_tot']} | Récence: **{row['Tau']} ans** | Densité Ψ: **{row['Psi']}**")
                    
                    # PREUVE (EXPANDER) - C'est ici qu'on voit les Qi
                    with st.expander(f"🔎 VOIR LES {len(row['SOURCES_QI'])} PHRASES SOURCES (Preuve de Mapping)"):
                        for source in row['SOURCES_QI']:
                            st.markdown(f"- 📄 *{source}*")
                            
                    st.markdown("---")
