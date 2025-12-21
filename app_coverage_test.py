import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Factory V4")
st.title("🏭 SMAXIA - Console Factory (Sourcing & Calcul)")

# --- 0. SIMULATEUR DE DONNÉES MATHÉMATIQUES ---
DB_MATHS = {
    "SUITES NUMÉRIQUES": [
        "Démontrer par récurrence que la suite est majorée",
        "Étudier le sens de variation de la suite (Un)",
        "Déterminer la limite de la suite par comparaison",
        "Montrer que la suite est géométrique de raison q",
        "Exprimer Un en fonction de n",
        "Calculer la somme des termes consécutifs"
    ],
    "NOMBRES COMPLEXES": [
        "Déterminer la forme algébrique de z",
        "Calculer le module et l'argument",
        "Résoudre l'équation z² + az + b = 0",
        "Placer les points images dans le plan complexe",
        "Montrer que le triangle ABC est équilatéral"
    ],
    "GÉOMÉTRIE ESPACE": [
        "Démontrer que la droite est orthogonale au plan",
        "Déterminer une représentation paramétrique de droite",
        "Calculer le produit scalaire u.v",
        "Vérifier que le point M appartient au plan (P)",
        "Déterminer une équation cartésienne du plan"
    ]
}

# --- 1. FONCTIONS MOTEUR ---

def ingest_and_calculate(urls, n_per_url, chapitres_cibles):
    """
    Simule la chaîne complète : Sourcing -> Granulation -> Calcul QC -> Score F2
    """
    sources_log = []
    all_qi = []
    
    natures = ["BAC", "DST", "INTERRO", "CONCOURS"]
    
    # 1. SOURCING & GRANULATION (F1)
    progress = st.progress(0)
    total_ops = len(urls) * n_per_url
    counter = 0
    
    for i, url in enumerate(urls):
        if not url.strip(): continue
        
        for j in range(n_per_url):
            counter += 1
            progress.progress(min(counter/total_ops, 1.0))
            time.sleep(0.005) # Micro-latence
            
            # Création Fichier Virtuel
            nature = random.choice(natures)
            year = random.choice(range(2019, 2025))
            file_id = f"DOC_{i}_{j}"
            filename = f"Sujet_{nature}_{year}_{j}.pdf"
            
            sources_log.append({
                "ID": file_id,
                "Fichier": filename,
                "Source (URL)": url, # Colonne demandée
                "Nature": nature, 
                "Année": year,
                "Statut": "📥 OK"
            })
            
            # Extraction Qi
            chaps_sujet = random.sample(chapitres_cibles, k=min(len(chapitres_cibles), 2))
            for chap in chaps_sujet:
                nb_exos = random.randint(2, 4)
                for _ in range(nb_exos):
                    qi_txt = random.choice(DB_MATHS[chap])
                    all_qi.append({
                        "ID_Source": file_id,
                        "Nature_Source": nature,
                        "Année": year,
                        "Chapitre": chap,
                        "Qi_Brut": qi_txt,
                        "Fichier_Origine": filename
                    })
    
    progress.empty()
    df_sources = pd.DataFrame(sources_log)
    df_qi = pd.DataFrame(all_qi)
    
    # 2. CALCUL MOTEUR QC (F2)
    # On regroupe immédiatement pour l'affichage Factory
    if df_qi.empty:
        return df_sources, df_qi, pd.DataFrame()

    grouped = df_qi.groupby(["Chapitre", "Qi_Brut"]).agg({
        "ID_Source": "count",      # n_q
        "Année": "max",            # Récence
        "Fichier_Origine": list    # Liste des fichiers sources (Preuve)
    }).reset_index()
    
    qcs = []
    N_total = len(df_qi)
    current_year = datetime.now().year
    
    for idx, row in grouped.iterrows():
        n_q = row["ID_Source"]
        tau = max((current_year - row["Année"]), 0.5)
        alpha = 5.0
        psi = 1.0 
        sigma = 0.00
        
        # EQUATION F2
        score = (n_q / N_total) * (1 + alpha/tau) * psi * (1-sigma) * 100
        
        qc_name = f"COMMENT {row['Qi_Brut']}..."
        
        qcs.append({
            "CHAPITRE": row["Chapitre"],
            "QC_INVARIANTE": qc_name,
            "SCORE_F2": score,
            
            # VARIABLES VISIBLES (DEMANDE UTILISATEUR)
            "n_q": n_q,
            "N_tot": N_total,
            "Tau": tau,
            "Alpha": alpha,
            "Psi": psi,
            "Sigma": sigma,
            
            "QI_ASSOCIES": row["Fichier_Origine"] # Liste des fichiers preuves
        })
        
    df_qc = pd.DataFrame(qcs).sort_values(by=["CHAPITRE", "SCORE_F2"], ascending=[True, False])
    
    return df_sources, df_qi, df_qc

# --- INTERFACE ---

# SIDEBAR
with st.sidebar:
    st.header("1. Périmètre")
    chapitres_actifs = st.multiselect(
        "Chapitres Cibles", 
        list(DB_MATHS.keys()), 
        default=["SUITES NUMÉRIQUES"]
    )

# LAYOUT PRINCIPAL
st.subheader("A. Usine de Sourcing & Génération QC")

col_input, col_act = st.columns([3, 1])
with col_input:
    urls_input = st.text_area("Sources (URLs)", "https://apmep.fr/terminale\nhttps://sujetdebac.fr", height=70)
with col_act:
    n_sujets = st.number_input("Vol. par URL", 5, 100, 10)
    btn_run = st.button("LANCER L'USINE 🚀", type="primary")

if btn_run:
    url_list = urls_input.split('\n')
    with st.spinner("Collecte + Granulation + Calcul F2..."):
        df_src, df_qi, df_qc = ingest_and_calculate(url_list, n_sujets, chapitres_actifs)
        
        # Sauvegarde Session
        st.session_state['df_src'] = df_src
        st.session_state['df_qi'] = df_qi
        st.session_state['df_qc'] = df_qc
        st.success("Traitement terminé.")

st.divider()

# VUE SPLIT (GAUCHE / DROITE)
if 'df_qc' in st.session_state:
    
    col_left, col_right = st.columns([1, 1.2]) # Droite un peu plus large pour les QC
    
    # --- COLONNE GAUCHE : SOURCES ---
    with col_left:
        st.markdown("### 📥 1. Liste des Sujets Sourcés")
        st.caption("Fichiers PDF collectés et indexés par Nature.")
        
        df_display_src = st.session_state['df_src'][["Fichier", "Source (URL)", "Nature", "Année"]]
        st.dataframe(
            df_display_src,
            column_config={
                "Source (URL)": st.column_config.LinkColumn("Lien Source"),
            },
            use_container_width=True,
            height=600
        )

    # --- COLONNE DROITE : QC + QI + PARAMÈTRES ---
    with col_right:
        st.markdown("### 🧠 2. QC Générées (Avec Qi & Variables)")
        st.caption("Regroupement sémantique et Calcul du Score F2.")
        
        # Filtre local pour l'affichage
        if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
            
            # Gestion du bug KeyError : On vérifie les chapitres disponibles
            available_chaps = st.session_state['df_qc']["CHAPITRE"].unique()
            chap_filter = st.selectbox("Filtrer par Chapitre", available_chaps)
            
            df_view_qc = st.session_state['df_qc'][st.session_state['df_qc']["CHAPITRE"] == chap_filter]
            
            if not df_view_qc.empty:
                # Boucle d'affichage "CARTE QC"
                for idx, row in df_view_qc.iterrows():
                    with st.container():
                        # EN-TÊTE QC
                        st.info(f"🗝️ **{row['QC_INVARIANTE']}**")
                        
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.metric("Score F2", f"{row['SCORE_F2']:.2f}")
                        
                        with c2:
                            # TABLEAU DES VARIABLES (L'équation visible)
                            var_data = pd.DataFrame([{
                                "n_q": row['n_q'],
                                "N_tot": row['N_tot'],
                                "Tau": row['Tau'],
                                "Alpha": row['Alpha'],
                                "Psi": row['Psi'],
                                "Sigma": row['Sigma']
                            }])
                            st.dataframe(var_data, hide_index=True, use_container_width=True)
                        
                        # LISTE DES Qi ASSOCIÉS (Preuve)
                        with st.expander(f"Voir les {row['n_q']} Qi associés (Preuve Source)"):
                            # On reconstruit la phrase source (ici simulé par le nom du fichier)
                            qi_data = pd.DataFrame({
                                "Source (Fichier)": row['QI_ASSOCIES'],
                                "Qi (Question Élève)": [row['QC_INVARIANTE'].replace("COMMENT ", "").replace("...", "")] * row['n_q']
                            })
                            st.dataframe(qi_data, hide_index=True, use_container_width=True)
                        
                        st.write("---")
            else:
                st.warning(f"Aucune QC trouvée pour le chapitre {chap_filter}.")
        else:
            st.warning("Aucune donnée. Lancez l'usine.")

else:
    st.info("👈 Configurez et lancez l'usine depuis le menu de gauche.")
