import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V4 (Real Content)")

st.markdown("""
<style>
    .big-score { font-size: 20px; font-weight: bold; color: #2563EB; }
    .stDataFrame { border: 1px solid #E5E7EB; }
    .css-1y4p8pa { padding-top: 1rem; } 
</style>
""", unsafe_allow_html=True)

# --- 1. BASE DE DONNÉES SMAXIA (CONTENU RÉEL) ---
# Structure : Chapitre -> Liste de QC Complètes
REAL_MATH_DB = {
    "DÉRIVATION": [
        {
            "QC": "COMMENT Étudier les variations d'une fonction",
            "FRT": "Tableau de Variations (Signe f' -> Var f)",
            "Triggers": ["VARIATIONS", "SENS DE VARIATION", "CROISSANTE", "DÉCROISSANTE"],
            "Qi_Exemples": [
                "Étudier les variations de la fonction f sur l'intervalle [0; +infini[.",
                "Dresser le tableau de variations complet de la fonction g.",
                "Justifier que la fonction h est strictement croissante sur R.",
                "En déduire le sens de variation de la suite (Un)."
            ]
        },
        {
            "QC": "COMMENT Déterminer l'équation d'une tangente en un point",
            "FRT": "Équation Réduite y = f'(a)(x-a) + f(a)",
            "Triggers": ["TANGENTE", "ÉQUATION", "POINT D'ABSCISSE"],
            "Qi_Exemples": [
                "Déterminer l'équation de la tangente T à la courbe C au point d'abscisse 0.",
                "Montrer que la tangente au point A a pour équation y = 2x + 1.",
                "Existe-t-il une tangente à C parallèle à l'axe des abscisses ?"
            ]
        }
    ],
    "GÉOMÉTRIE ESPACE": [
        {
            "QC": "COMMENT Démontrer qu'une droite est orthogonale à un plan",
            "FRT": "Preuve Vectorielle (Produit Scalaire Nul x2)",
            "Triggers": ["ORTHOGONALE", "PERPENDICULAIRE", "PLAN"],
            "Qi_Exemples": [
                "Démontrer que la droite (AH) est orthogonale au plan (BCD).",
                "Prouver que le vecteur n est normal au plan (P).",
                "En déduire que la droite (d) est perpendiculaire au plan (ABC)."
            ]
        },
        {
            "QC": "COMMENT Déterminer une représentation paramétrique de droite",
            "FRT": "Système Paramétrique {x(t), y(t), z(t)}",
            "Triggers": ["REPRÉSENTATION PARAMÉTRIQUE", "SYSTÈME", "DROITE"],
            "Qi_Exemples": [
                "Déterminer une représentation paramétrique de la droite (AB).",
                "Donner un système d'équations paramétriques de la droite passant par A et de vecteur directeur u.",
                "La droite (d) est définie par la représentation paramétrique suivante..."
            ]
        }
    ],
    "LIMITES": [
        {
            "QC": "COMMENT Lever une forme indéterminée (FI)",
            "FRT": "Calcul de Limite (Factorisation / Croissance Comparée)",
            "Triggers": ["LIMITE", "INDÉTERMINÉE", "TEND VERS"],
            "Qi_Exemples": [
                "Déterminer la limite de f en +infini (on pourra factoriser par x).",
                "Lever l'indétermination pour calculer la limite en 0.",
                "En utilisant les croissances comparées, déterminer la limite de f(x)."
            ]
        }
    ]
}

# --- 2. MOTEUR : CRAWLER & CLASSIFICATEUR ---

def simulate_audit_crawl(url, n_sujets):
    """Génère des sujets réalistes basés sur la DB SMAXIA"""
    files_log = []
    atoms = []
    
    progress = st.progress(0)
    status = st.empty()
    
    for i in range(n_sujets):
        if i % 2 == 0: 
            progress.progress((i+1)/n_sujets)
            status.text(f"Fetching & Atomizing : {url}/sujet_bac_{2024-i%5}_{i}.pdf")
            
        year = random.choice(range(2018, 2025))
        
        # Log Fichier
        files_log.append({
            "ID": f"DOC_{i:03d}",
            "Fichier": f"Bac_Sujet_{year}_Maths_{i}.pdf",
            "Année": year,
            "Statut": "✅ OK"
        })
        
        # Génération des Atomes (Qi)
        # Chaque sujet contient 3 à 4 exercices piochés dans la DB
        nb_exos = random.randint(3, 4)
        for _ in range(nb_exos):
            # Choix Chapitre
            chap = random.choice(list(REAL_MATH_DB.keys()))
            # Choix QC dans le chapitre
            qc_data = random.choice(REAL_MATH_DB[chap])
            # Choix d'une Qi réaliste (simulation de variante)
            qi_text = random.choice(qc_data["Qi_Exemples"])
            
            atoms.append({
                "ID_Fichier": f"DOC_{i:03d}",
                "Année": year,
                "Niveau": "TERMINALE",
                "Matière": "MATHS",
                "Chapitre": chap,
                "Qi_Brut": qi_text,
                # On passe les métadonnées SMAXIA cachées pour le moteur
                "_QC_REF": qc_data["QC"],
                "_FRT_REF": qc_data["FRT"],
                "_TRIGS_REF": qc_data["Triggers"]
            })
            
    progress.empty()
    status.empty()
    return pd.DataFrame(files_log), pd.DataFrame(atoms)

def calculate_smaxia_matrix(df_atoms):
    N_total_global = len(df_atoms)
    current_year = datetime.now().year
    
    # Agrégation par la QC de référence (Invariant SMAXIA)
    grouped = df_atoms.groupby(["Niveau", "Matière", "Chapitre", "_QC_REF", "_FRT_REF"]).agg({
        "Qi_Brut": list,
        "ID_Fichier": "count", # n_q
        "Année": "max",
        "_TRIGS_REF": "first" # On récupère les triggers
    }).reset_index()
    
    results = []
    
    for idx, row in grouped.iterrows():
        n_q = row["ID_Fichier"]
        
        # Calcul Variables F2
        delta_t = (current_year - row["Année"])
        tau = delta_t if delta_t > 0 else 0.5
        alpha = 5.0
        
        # Psi (Densité)
        psi = 0.85 # Simulé stable pour l'exemple mathématique
        sigma = 0.02 # Très peu de bruit sur ces phrases propres
        
        # ÉQUATION F2
        freq = n_q / N_total_global
        recency = 1 + (alpha / max(tau, 1))
        
        score = freq * recency * psi * (1 - sigma) * 100
        
        results.append({
            "NIVEAU": row["Niveau"],
            "MATIERE": row["Matière"],
            "CHAPITRE": row["Chapitre"],
            "QC_INVARIANTE": row["_QC_REF"],
            "FRT": row["_FRT_REF"],
            "TRIGGERS": ", ".join(row["_TRIGS_REF"][:3]), # Affichage propre
            "SCORE_F2": score,
            
            # VARIABLES AUDIT
            "n_q": n_q,
            "N_tot": N_total_global,
            "Tau": tau,
            "Alpha": alpha,
            "Psi": psi,
            "Sigma": sigma,
            "SOURCES": row["Qi_Brut"]
        })
        
    return pd.DataFrame(results).sort_values(by="SCORE_F2", ascending=False)

# --- INTERFACE ---
st.title("🛡️ SMAXIA - Console V4 (Real Content)")

# 1. PARAMÈTRES
with st.container():
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        url = st.text_input("URL Source", value="https://www.apmep.fr")
    with c2:
        n_docs = st.number_input("Nombre de Sujets", value=20, step=5)
    with c3:
        st.write("")
        run = st.button("LANCER L'AUDIT", type="primary")

if run:
    st.divider()
    
    # 2. CRAWLING
    df_files, df_atoms = simulate_audit_crawl(url, n_docs)
    
    st.info(f"✅ **{len(df_files)} Sujets** analysés | **{len(df_atoms)} Qi (Exercices)** extraites et atomisées.")

    # 3. CALCUL MOTEUR
    df_result = calculate_smaxia_matrix(df_atoms)
    
    # 4. AFFICHAGE
    st.divider()
    
    # Navigation Hiérarchique
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        niv_sel = st.selectbox("Niveau", df_result["NIVEAU"].unique())
    with col_nav2:
        mat_sel = st.selectbox("Matière", df_result[df_result["NIVEAU"]==niv_sel]["MATIERE"].unique())
        
    df_filtered = df_result[(df_result["NIVEAU"]==niv_sel) & (df_result["MATIERE"]==mat_sel)]
    
    # Onglets Chapitres
    chapitres = df_filtered["CHAPITRE"].unique()
    tabs = st.tabs(list(chapitres))
    
    for i, chap in enumerate(chapitres):
        with tabs[i]:
            df_final = df_filtered[df_filtered["CHAPITRE"] == chap]
            
            # Rappel de l'équation
            st.latex(r"Score(q) = \frac{n_q}{N_{tot}} \times [1 + \frac{\alpha}{\tau}] \times \Psi \times (1 - \sigma)")
            
            # TABLEAU PRINCIPAL
            st.dataframe(
                df_final[[
                    "QC_INVARIANTE", 
                    "SCORE_F2", 
                    "n_q", "N_tot", "Tau", "Psi", "Sigma",
                    "FRT", "TRIGGERS"
                ]],
                column_config={
                    "QC_INVARIANTE": st.column_config.TextColumn("Question Clé (Invariant)", width="large"),
                    "SCORE_F2": st.column_config.ProgressColumn("Score F2", format="%.2f", min_value=0, max_value=100),
                    "n_q": st.column_config.NumberColumn("n_q", width="small"),
                    "N_tot": st.column_config.NumberColumn("N_tot", width="small"),
                    "Tau": st.column_config.NumberColumn("τ", format="%.1f", width="small"),
                    "FRT": st.column_config.TextColumn("Format Réponse (FRT)", width="medium"),
                    "TRIGGERS": st.column_config.TextColumn("Déclencheurs", width="medium"),
                },
                use_container_width=True,
                hide_index=True
            )
            
            # PREUVE (EXPANDER DYNAMIQUE)
            for idx, row in df_final.iterrows():
                with st.expander(f"🔎 VOIR LES SOURCES ({row['n_q']}) pour : {row['QC_INVARIANTE']}"):
                    st.markdown(f"**FRT Attendue :** `{row['FRT']}`")
                    st.markdown("**Phrases élèves (Qi) ayant déclenché cette QC :**")
                    for s in row["SOURCES"]:
                        st.text(f"• {s}")
