import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime
import re

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - CEO Crawler & Engine")

# --- 1. BIBLIOTHÈQUE DE CONTENU (SIMULATION DU WEB) ---
# Ce sont les types d'exercices que le robot "trouverait" sur un site comme APMEP
WEB_CONTENT_DB = [
    # TERMINALE - ANALYSE
    {"txt": "Calculer la limite de la fonction f en +infini", "chap": "Limites", "niv": "TERMINALE", "mat": "MATHS", "trigs": ["calculer", "limite"], "frt": "Valeur Numérique / Exacte"},
    {"txt": "Démontrer par récurrence que la suite est bornée", "chap": "Suites", "niv": "TERMINALE", "mat": "MATHS", "trigs": ["démontrer", "récurrence"], "frt": "Preuve Logique (Bloc Rédactionnel)"},
    {"txt": "Étudier les variations de la fonction f sur I", "chap": "Dérivation", "niv": "TERMINALE", "mat": "MATHS", "trigs": ["étudier", "variations"], "frt": "Tableau de Signes + Variations"},
    {"txt": "Déterminer une primitive F de la fonction f", "chap": "Intégration", "niv": "TERMINALE", "mat": "MATHS", "trigs": ["déterminer", "primitive"], "frt": "Expression Algébrique F(x)"},
    # TERMINALE - GÉOMÉTRIE
    {"txt": "Démontrer que la droite (D) est orthogonale au plan (P)", "chap": "Espace", "niv": "TERMINALE", "mat": "MATHS", "trigs": ["démontrer", "orthogonal"], "frt": "Preuve Produit Scalaire"},
    {"txt": "Déterminer une représentation paramétrique de la droite", "chap": "Espace", "niv": "TERMINALE", "mat": "MATHS", "trigs": ["déterminer", "paramétrique"], "frt": "Système d'équations {x,y,z}"},
    # PREMIÈRE
    {"txt": "Calculer les racines du trinôme du second degré", "chap": "Second Degré", "niv": "PREMIERE", "mat": "MATHS", "trigs": ["calculer", "racines"], "frt": "Ensemble S = {x1, x2}"},
    {"txt": "Déterminer le produit scalaire des vecteurs u et v", "chap": "Produit Scalaire", "niv": "PREMIERE", "mat": "MATHS", "trigs": ["déterminer", "produit scalaire"], "frt": "Valeur Réelle"},
]

# --- 2. FONCTIONS MOTEUR (NOYAU SMAXIA) ---

def simulate_crawl(url, n_target):
    """Simule la récupération de N sujets depuis une URL"""
    crawled = []
    progress = st.progress(0)
    status = st.empty()
    
    for i in range(n_target):
        # Simulation d'attente réseau
        time.sleep(0.02) 
        progress.progress((i+1)/n_target)
        status.text(f"Scraping : {url}/sujet_bac_{2024-i%10}_{i}.pdf ... OK")
        
        # On pioche un contenu aléatoire pour simuler la lecture du PDF
        content = random.choice(WEB_CONTENT_DB)
        
        # On génère des métadonnées réalistes
        year = random.choice(range(2015, 2025))
        
        crawled.append({
            "ID_Source": f"DOC_{i:04d}",
            "Source_Url": f"{url}/doc_{i}.pdf",
            "Année": year,
            "Niveau": content["niv"],
            "Matière": content["mat"],
            "Chapitre": content["chap"],
            "Qi_Brut": content["txt"],
            "Triggers": content["trigs"],
            "FRT_Type": content["frt"]
        })
        
    progress.empty()
    status.empty()
    return pd.DataFrame(crawled)

def calculate_score_f2(df_data):
    """Calcule le Score F2 et toutes les variables pour chaque QC identifiée"""
    
    # 1. Regroupement par QC (Invariant)
    grouped = df_data.groupby(["Niveau", "Matière", "Chapitre", "Qi_Brut"]).agg({
        "ID_Source": "count",      # n_q (Fréquence)
        "Année": "max",            # Pour Récence
        "Triggers": "first",
        "FRT_Type": "first"
    }).reset_index()
    
    results = []
    current_year = datetime.now().year
    N_total_global = len(df_data) # Volume total injecté
    
    for idx, row in grouped.iterrows():
        # --- CALCUL VARIABLES ---
        n_q = row["ID_Source"]
        
        # Tau (Récence)
        delta_t = (current_year - row["Année"])
        tau_rec = delta_t if delta_t > 0 else 0.5 # Année courante
        
        # Alpha (Coeff Récence fixe pour test)
        alpha = 5.0 
        
        # Psi (Densité)
        words = row["Qi_Brut"].split()
        psi = len(set(words)) / len(words)
        
        # Sigma (Similarité/Bruit - simplifié ici à 0 pour l'exemple)
        sigma = 0.0
        
        # --- ÉQUATION ---
        # Score = (n_q / N_tot) * [1 + alpha/tau] * psi * (1-sigma)
        freq = n_q / N_total_global
        recency = 1 + (alpha / max(tau_rec, 1)) # Sécurité div/0
        
        score_f2 = freq * recency * psi * 100
        
        # Création QC Format SMAXIA
        qc_name = f"COMMENT {row['Qi_Brut']}..."
        
        results.append({
            "NIVEAU": row["Niveau"],
            "MATIERE": row["Matière"],
            "CHAPITRE": row["Chapitre"],
            "QC_INVARIANTE": qc_name,
            "FRT": row["FRT_Type"],
            # VARIABLES DE PREUVE
            "Score_F2": score_f2,
            "n_q": n_q,
            "N_tot": N_total_global,
            "Tau_rec": tau_rec,
            "Psi": round(psi, 2),
            "Sigma": sigma,
            "Nb_Qi_Source": n_q # Nombre de Qi qui pointent vers cette QC
        })
        
    return pd.DataFrame(results).sort_values(by=["NIVEAU", "CHAPITRE", "Score_F2"], ascending=[True, True, False])

# --- INTERFACE CEO ---
st.title("🚀 SMAXIA - Automation Console (Source -> Engine -> Livrable)")

with st.container():
    st.markdown("### 1. Paramètres d'Injection")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        url = st.text_input("URL Source", value="https://www.apmep.fr/", placeholder="ex: apmep.fr")
    with col2:
        n_sujets = st.number_input("Nombre de Sujets (N)", min_value=10, max_value=1000, value=50, step=10)
    with col3:
        st.write("")
        start = st.button("LANCER L'EXTRACTION", type="primary")

if start:
    st.divider()
    
    # 1. CRAWLING
    st.subheader("2. Journal d'Exécution")
    with st.spinner(f"Récupération de {n_sujets} sujets sur {url}..."):
        df_raw = simulate_crawl(url, n_sujets)
    st.success(f"✅ Extraction terminée : {len(df_raw)} Qi brutes extraites.")
    
    # 2. CALCUL MOTEUR
    df_livrable = calculate_score_f2(df_raw)
    
    # 3. AFFICHAGE LIVRABLE
    st.divider()
    st.header("📦 3. LIVRABLE FINAL (Classé par Hiérarchie)")
    
    # Navigation par Niveau
    niveaux = df_livrable["NIVEAU"].unique()
    tabs = st.tabs(list(niveaux))
    
    for i, niv in enumerate(niveaux):
        with tabs[i]:
            df_niv = df_livrable[df_livrable["NIVEAU"] == niv]
            
            # Boucle par Chapitre
            chapitres = df_niv["CHAPITRE"].unique()
            for chap in chapitres:
                st.markdown(f"#### 📘 {chap}")
                
                df_chap = df_niv[df_niv["CHAPITRE"] == chap]
                
                # TABLEAU COMPLET AVEC VARIABLES
                st.dataframe(
                    df_chap[[
                        "QC_INVARIANTE", 
                        "Score_F2", 
                        "n_q", "N_tot", "Tau_rec", "Psi", "Sigma", 
                        "FRT"
                    ]],
                    column_config={
                        "QC_INVARIANTE": st.column_config.TextColumn("Question Clé (Invariant)", width="large"),
                        "Score_F2": st.column_config.ProgressColumn("Pertinence F2", format="%.2f", min_value=0, max_value=max(df_livrable["Score_F2"])),
                        "n_q": st.column_config.NumberColumn("n_q", format="%d"),
                        "Tau_rec": st.column_config.NumberColumn("τ (Ans)", format="%.1f"),
                        "FRT": st.column_config.TextColumn("FRT (Type Réponse)", width="medium"),
                    },
                    use_container_width=True,
                    hide_index=True
                )
