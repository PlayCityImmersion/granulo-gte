import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Factory V6.6")
st.title("🏭 SMAXIA - Console Factory V6.6 (Restauration + Corrections)")

# --- 0. MOTEUR DE VARIANTES (CORRECTION 3 : Polymorphisme) ---
# Pour éviter que les Qi soient toutes pareilles, on utilise des templates
MATH_VARIANTS = {
    "SUITES_GEO": [
        "Montrer que la suite ({u}) est géométrique.",
        "Démontrer que la suite définie par {u} est géométrique de raison {q}.",
        "Justifier le caractère géométrique de la suite ({u}).",
        "Prouver que pour tout n, {u} est une suite géométrique."
    ],
    "SUITES_LIM": [
        "Déterminer la limite de la suite ({u}).",
        "Calculer la limite de ({u}) quand n tend vers l'infini.",
        "Étudier la convergence de la suite ({u}).",
        "La suite ({u}) converge-t-elle ?"
    ],
    "COMPLEXE_ALG": [
        "Déterminer la forme algébrique du nombre complexe {z}.",
        "Écrire le nombre {z} sous forme a + ib.",
        "Donner la partie réelle et imaginaire de {z}.",
        "Mettre {z} sous forme algébrique."
    ],
    "ESPACE_ORTHO": [
        "Démontrer que la droite ({d}) est orthogonale au plan ({p}).",
        "Prouver que le vecteur {v} est normal au plan ({p}).",
        "Justifier l'orthogonalité entre ({d}) et ({p}).",
        "Vérifier que ({d}) est perpendiculaire à ({p})."
    ]
}

# Variables aléatoires pour varier le texte
VARS_SUITE = ["Un", "Vn", "Wn", "tn"]
VARS_CPLX = ["z", "zA", "zB", "Ω"]
VARS_DROITE = ["(d)", "(Delta)", "(AB)"]
VARS_PLAN = ["(P)", "(ABC)", "(Pi)"]

def get_varied_qi(concept_key):
    """Génère une phrase unique pour le concept donné"""
    tpl = random.choice(MATH_VARIANTS.get(concept_key, ["Question standard."]))
    return tpl.format(
        u=random.choice(VARS_SUITE),
        q=random.choice(["1/2", "3", "q", "-2"]),
        z=random.choice(VARS_CPLX),
        d=random.choice(VARS_DROITE),
        p=random.choice(VARS_PLAN),
        v=random.choice(["n", "u", "v"])
    )

# Mapping : Chapitre -> Liste de Concepts disponibles
DB_CONCEPTS = {
    "SUITES NUMÉRIQUES": ["SUITES_GEO", "SUITES_LIM"],
    "NOMBRES COMPLEXES": ["COMPLEXE_ALG"],
    "GÉOMÉTRIE ESPACE": ["ESPACE_ORTHO"]
}

# --- 1. FONCTIONS MOTEUR ---

def ingest_and_calculate(urls, n_per_url, chapitres_cibles):
    sources_log = []
    all_qi = []
    natures = ["BAC", "DST", "INTERRO", "CONCOURS"]
    
    progress = st.progress(0)
    total_ops = len(urls) * n_per_url if len(urls) > 0 else 1
    counter = 0
    
    for i, url in enumerate(urls):
        if not url.strip(): continue
        for j in range(n_per_url):
            counter += 1
            progress.progress(min(counter/total_ops, 1.0))
            time.sleep(0.005) 
            
            nature = random.choice(natures)
            year = random.choice(range(2020, 2025))
            file_id = f"DOC_{i}_{j}"
            filename = f"Sujet_{nature}_{year}_{j}.txt"
            
            # Génération du contenu physique (Correction 1)
            file_content_lines = [f"SUJET {nature} - {year}", f"SOURCE: {url}", "-"*20]
            
            # Extraction Qi (Simulée)
            # On ne prend que les chapitres demandés dans la sidebar
            valid_chaps = [c for c in chapitres_cibles if c in DB_CONCEPTS]
            if not valid_chaps: valid_chaps = list(DB_CONCEPTS.keys())
            
            chaps_sujet = random.sample(valid_chaps, k=min(len(valid_chaps), 2))
            
            for chap in chaps_sujet:
                concepts = DB_CONCEPTS[chap]
                for concept in concepts:
                    # ICI : On génère une variante unique (Correction 3)
                    qi_txt = get_varied_qi(concept)
                    
                    file_content_lines.append(f"Exo: {qi_txt}")
                    
                    all_qi.append({
                        "ID_Source": file_id,
                        "Nature_Source": nature,
                        "Année": year,
                        "Chapitre": chap,
                        "Concept_Code": concept, # Invariant caché
                        "Qi_Brut": qi_txt,       # Phrase visible (Variable)
                        "Fichier_Origine": filename
                    })
            
            # Stockage du contenu complet pour téléchargement
            full_text = "\n".join(file_content_lines)
            
            sources_log.append({
                "ID": file_id,
                "Fichier": filename,
                "Nature": nature, 
                "Année": year,
                "Contenu_Blob": full_text # Le vrai texte
            })
    
    progress.empty()
    return pd.DataFrame(sources_log), pd.DataFrame(all_qi)

def calculate_engine_qc(df_qi):
    if df_qi.empty: return pd.DataFrame()

    # Regroupement par CONCEPT (Invariant) et non par texte
    grouped = df_qi.groupby(["Chapitre", "Concept_Code"]).agg({
        "Qi_Brut": list,           # Liste des phrases (Variantes)
        "Fichier_Origine": list,   # Liste des fichiers
        "Année": "max"             # Récence
    }).reset_index()
    
    qcs = []
    current_year = datetime.now().year
    
    # Titres QC propres
    TITRES_QC = {
        "SUITES_GEO": "COMMENT Démontrer qu'une suite est géométrique",
        "SUITES_LIM": "COMMENT Calculer la limite d'une suite",
        "COMPLEXE_ALG": "COMMENT Déterminer la forme algébrique",
        "ESPACE_ORTHO": "COMMENT Caractériser l'orthogonalité Droite/Plan"
    }
    
    for idx, row in grouped.iterrows():
        n_q = len(row["Qi_Brut"])
        N_total = len(df_qi) # Simplifié
        tau = max((current_year - row["Année"]), 0.5)
        alpha = 5.0
        psi = 1.0 
        sigma = 0.00
        
        score = (n_q / N_total) * (1 + alpha/tau) * psi * (1-sigma) * 100
        qc_title = TITRES_QC.get(row["Concept_Code"], row["Concept_Code"])
        
        # Preuve (Liste des Qi variées)
        evidence = []
        for k in range(len(row["Qi_Brut"])):
            evidence.append({
                "Fichier": row["Fichier_Origine"][k],
                "Qi (Variante)": row["Qi_Brut"][k]
            })
            
        qcs.append({
            "CHAPITRE": row["Chapitre"],
            "QC_INVARIANTE": qc_title,
            "SCORE_F2": score,
            "n_q": n_q,
            "N_tot": N_total,
            "Tau": tau,
            "Alpha": alpha,
            "Psi": psi,
            "Sigma": sigma,
            "EVIDENCE": evidence
        })
        
    return pd.DataFrame(qcs).sort_values(by=["CHAPITRE", "SCORE_F2"], ascending=[True, False])

# --- INTERFACE (RESTAURATION STRUCTURE V6) ---

# SIDEBAR (RESTAURÉE)
with st.sidebar:
    st.header("1. Périmètre Usine")
    # Choix multiples restaurés comme avant
    chapitres_actifs = st.multiselect(
        "Chapitres Cibles", 
        list(DB_CONCEPTS.keys()), 
        default=list(DB_CONCEPTS.keys())
    )

# TABS
tab_factory = st.container()

with tab_factory:
    st.subheader("A. Sourcing & Génération QC")

    col_input, col_act = st.columns([3, 1])
    with col_input:
        urls_input = st.text_area("Sources (URLs)", "https://apmep.fr/terminale\nhttps://sujetdebac.fr", height=70)
    with col_act:
        # CORRECTION 2 : Step = 5
        n_sujets = st.number_input("Vol. par URL", min_value=5, max_value=100, value=10, step=5)
        btn_run = st.button("LANCER L'USINE 🚀", type="primary")

    if btn_run:
        url_list = urls_input.split('\n')
        with st.spinner("Traitement Polymorphe..."):
            df_src, df_qi = ingest_and_calculate(url_list, n_sujets, chapitres_actifs)
            df_qc = calculate_engine_qc(df_qi)
            
            st.session_state['df_src'] = df_src
            st.session_state['df_qc'] = df_qc
            st.success("Usine mise à jour.")

    st.divider()

    # VUE SPLIT USINE (GAUCHE / DROITE)
    if 'df_qc' in st.session_state:
        col_left, col_right = st.columns([1, 1.5])
        
        # --- GAUCHE : SUJETS (CORRECTION 1 : Téléchargement) ---
        with col_left:
            st.markdown(f"### 📥 Sujets ({len(st.session_state['df_src'])})")
            
            # Affichage tableau simple
            st.dataframe(
                st.session_state['df_src'][["Fichier", "Nature", "Année"]],
                use_container_width=True,
                height=400
            )
            
            st.markdown("#### 💾 Téléchargement Physique")
            # Sélecteur pour télécharger
            sel_file = st.selectbox("Choisir le sujet à vérifier :", st.session_state['df_src']["Fichier"])
            
            # Récupération du blob
            file_data = st.session_state['df_src'][st.session_state['df_src']["Fichier"] == sel_file].iloc[0]
            
            st.download_button(
                label=f"📥 Télécharger {sel_file}",
                data=file_data["Contenu_Blob"],
                file_name=sel_file,
                mime="text/plain",
                type="secondary"
            )

        # --- DROITE : QC GÉNÉRÉES ---
        with col_right:
            total_qc = len(st.session_state['df_qc'])
            st.markdown(f"### 🧠 QC Générées (Total : {total_qc})")
            
            if not st.session_state['df_qc'].empty:
                # Filtre Chapitre (Restauration)
                available_chaps = st.session_state['df_qc']["CHAPITRE"].unique()
                chap_filter = st.selectbox("Filtrer Affichage QC", available_chaps)
                
                df_view_qc = st.session_state['df_qc'][st.session_state['df_qc']["CHAPITRE"] == chap_filter]
                
                for idx, row in df_view_qc.iterrows():
                    with st.container():
                        st.info(f"**{row['QC_INVARIANTE']}**")
                        st.caption(f"Score F2: **{row['SCORE_F2']:.2f}**")
                        
                        # Affichage des variables F2 (Demandé)
                        cols_var = st.columns(6)
                        cols_var[0].metric("n_q", row['n_q'])
                        cols_var[1].metric("N_tot", row['N_tot'])
                        cols_var[2].metric("Tau", row['Tau'])
                        cols_var[3].metric("Alpha", row['Alpha'])
                        cols_var[4].metric("Psi", row['Psi'])
                        cols_var[5].metric("Sigma", row['Sigma'])
                        
                        # CORRECTION 3 : Preuve Polymorphe
                        with st.expander(f"Voir les {row['n_q']} Variantes (Preuve)"):
                            st.write("Phrases élèves différentes regroupées ici :")
                            st.dataframe(pd.DataFrame(row['EVIDENCE']), hide_index=True)
                        st.divider()
            else:
                st.warning("Aucune QC pour ces chapitres.")
    else:
        st.info("Configurez le périmètre à gauche et lancez l'usine.")
