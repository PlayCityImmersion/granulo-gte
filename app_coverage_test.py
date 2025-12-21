import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Factory V3")
st.title("🏭 SMAXIA - Console Factory & Validation (End-to-End)")

# --- 0. SIMULATEUR DE DONNÉES MATHÉMATIQUES ---
# Base de connaissance simulée pour générer du contenu réaliste
DB_MATHS = {
    "SUITES NUMÉRIQUES": [
        "Démontrer par récurrence que la suite est majorée",
        "Étudier le sens de variation de la suite (Un)",
        "Déterminer la limite de la suite par comparaison",
        "Montrer que la suite est géométrique de raison q",
        "Exprimer Un en fonction de n"
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

# --- 1. FONCTIONS DU MOTEUR ---

def ingest_sources(urls, n_per_url, chapitres_cibles):
    """
    Simule la collecte et l'étiquetage par NATURE (DST, BAC...)
    """
    sources_log = []
    all_qi = []
    
    natures = ["BAC", "DST", "INTERRO", "CONCOURS"]
    
    progress = st.progress(0)
    for i, url in enumerate(urls):
        if not url.strip(): continue
        
        for j in range(n_per_url):
            # Simulation
            time.sleep(0.01)
            progress.progress((j+1)/n_per_url)
            
            # 1. Création du Fichier Source
            nature = random.choice(natures)
            year = random.choice(range(2018, 2025))
            file_id = f"DOC_{i}_{j}"
            filename = f"Sujet_{nature}_{year}_{j}.pdf"
            
            sources_log.append({
                "ID": file_id,
                "Fichier": filename,
                "URL_Source": url,
                "Nature": nature, # DST, BAC...
                "Année": year,
                "Statut": "📥 COLLECTÉ"
            })
            
            # 2. Extraction des Qi (Granulation)
            # Un sujet couvre souvent 1 ou 2 chapitres
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
                        "Trigger": qi_txt.split()[0].upper()
                    })
                    
    progress.empty()
    return pd.DataFrame(sources_log), pd.DataFrame(all_qi)

def calculate_engine_qc(df_qi):
    """
    Regroupe les Qi par Chapitre -> Génère les QC -> Calcule F2
    """
    if df_qi.empty: return pd.DataFrame()
    
    # Agrégation par (Chapitre + Concept) pour former une QC
    # On simule le regroupement sémantique par le texte exact ici
    grouped = df_qi.groupby(["Chapitre", "Qi_Brut"]).agg({
        "ID_Source": "count",      # n_q (Fréquence)
        "Année": "max",            # Récence
        "Nature_Source": list      # Pour voir si ça tombe en BAC ou DST
    }).reset_index()
    
    qcs = []
    N_total = len(df_qi)
    current_year = datetime.now().year
    
    for idx, row in grouped.iterrows():
        n_q = row["ID_Source"]
        tau = max((current_year - row["Année"]), 0.5)
        alpha = 5.0
        psi = 1.0 # Simplifié
        sigma = 0.05
        
        # SCORE F2
        score = (n_q / N_total) * (1 + alpha/tau) * psi * (1-sigma) * 100
        
        qc_name = f"COMMENT {row['Qi_Brut']}..."
        
        qcs.append({
            "CHAPITRE": row["Chapitre"],
            "QC_INVARIANTE": qc_name,
            "SCORE_F2": score,
            "n_q": n_q,
            "N_tot": N_total,
            "Tau": tau,
            "SOURCES_TYPES": list(set(row["Nature_Source"])) # Ex: [BAC, DST]
        })
        
    return pd.DataFrame(qcs).sort_values(by=["CHAPITRE", "SCORE_F2"], ascending=[True, False])

def test_mapping(qi_text, df_qc_engine):
    """
    Vérifie si une Qi injectée individuellement trouve sa QC
    """
    for idx, row in df_qc_engine.iterrows():
        # Matching sémantique simulé (inclusion)
        core_qc = row["QC_INVARIANTE"].replace("COMMENT ", "").replace("...", "")
        if core_qc in qi_text:
            return True, row["QC_INVARIANTE"], row["SCORE_F2"]
    return False, None, 0

# --- INTERFACE ---

# BARRE LATÉRALE (CONFIG)
with st.sidebar:
    st.header("1. Configuration Usine")
    chapitres_actifs = st.multiselect(
        "Chapitres à Traiter", 
        list(DB_MATHS.keys()), 
        default=list(DB_MATHS.keys())
    )

# ONGLETS PRINCIPAUX
tab_source, tab_engine, tab_test = st.tabs([
    "📥 ZONE 1 & 2 : Sourcing & Granulation", 
    "⚙️ ZONE 3 : Moteur & Calcul F2", 
    "🎯 ZONE 4 : Crash Test (Validation)"
])

# --- ZONE 1 & 2 : COLLECTE ET CLASSEMENT Qi ---
with tab_source:
    st.subheader("A. Zone de Positionnement (URLs)")
    urls_input = st.text_area("URLs cibles", "https://apmep.fr/terminale\nhttps://sujetdebac.fr", height=70)
    n_sujets = st.number_input("Volume par URL", 5, 50, 10)
    
    if st.button("LANCER LA COLLECTE & GRANULATION"):
        with st.spinner("Collecte des sujets et extraction des Qi..."):
            url_list = urls_input.split('\n')
            df_sources, df_qi = ingest_sources(url_list, n_sujets, chapitres_actifs)
            
            # Sauvegarde Session
            st.session_state['df_sources'] = df_sources
            st.session_state['df_qi'] = df_qi
            st.session_state['data_fresh'] = True # Signal pour recalculer le moteur
            
            st.success(f"Collecte terminée : {len(df_sources)} Sujets | {len(df_qi)} Atomes (Qi) extraits.")

    if 'df_sources' in st.session_state:
        st.divider()
        col_src, col_qi = st.columns(2)
        
        with col_src:
            st.markdown("### 📋 Liste des Sujets Sourcés")
            st.dataframe(
                st.session_state['df_sources'][["Fichier", "Nature", "Année"]],
                use_container_width=True
            )
            
        with col_qi:
            st.markdown("### 🧬 Qi Extraites (Par Chapitre)")
            # Filtre dynamique
            chap_filter = st.selectbox("Filtrer Qi par Chapitre", chapitres_actifs)
            df_show = st.session_state['df_qi'][st.session_state['df_qi']["Chapitre"] == chap_filter]
            st.dataframe(df_show[["Qi_Brut", "Nature_Source"]], use_container_width=True)

# --- ZONE 3 : TRAITEMENT ET GÉNÉRATION QC ---
with tab_engine:
    st.subheader("B. Moteur de Génération QC (Mise à jour dynamique)")
    
    if st.button("LANCER / METTRE À JOUR LE MOTEUR"):
        if 'df_qi' in st.session_state:
            df_qc = calculate_engine_qc(st.session_state['df_qi'])
            st.session_state['df_qc'] = df_qc
            st.success("✅ Moteur mis à jour avec les dernières données.")
        else:
            st.error("Aucune donnée Qi. Veuillez lancer la collecte en Zone 1.")
            
    if 'df_qc' in st.session_state:
        st.divider()
        
        # Affichage rangé par Chapitre
        for chap in chapitres_actifs:
            st.markdown(f"#### 📘 {chap}")
            df_view = st.session_state['df_qc'][st.session_state['df_qc']["CHAPITRE"] == chap]
            
            if not df_view.empty:
                # Tableau Moteur
                st.dataframe(
                    df_view[[
                        "QC_INVARIANTE", "SCORE_F2", "n_q", "N_tot", "Tau", "SOURCES_TYPES"
                    ]],
                    column_config={
                        "SCORE_F2": st.column_config.ProgressColumn("Score F2", format="%.2f"),
                        "SOURCES_TYPES": st.column_config.ListColumn("Vu dans (Nature)")
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                # Preuve Qi Associés (Expander)
                with st.expander(f"Voir les Qi associés pour {chap}"):
                    # On retrouve les Qi brutes qui matchent ce chapitre
                    raw_qi = st.session_state['df_qi'][st.session_state['df_qi']["Chapitre"] == chap]
                    st.dataframe(raw_qi)
            else:
                st.caption("Pas de QC détectée pour ce chapitre.")

# --- ZONE 4 : CRASH TEST (VALIDATION INDIVIDUELLE) ---
with tab_test:
    st.subheader("C. Test de Couverture Individuel")
    st.markdown("Téléversez (simulé) un exercice ou un sujet pour voir si le moteur le couvre.")
    
    col_upload, col_res = st.columns([1, 2])
    
    with col_upload:
        # Simulation Upload
        upload_type = st.selectbox("Nature du document", ["DST", "INTERRO", "EXO_LIVRE"])
        target_chap = st.selectbox("Chapitre du document", chapitres_actifs)
        
        if st.button("ANALISER CE DOCUMENT"):
            if 'df_qc' not in st.session_state:
                st.error("Moteur vide.")
            else:
                # Simulation extraction Qi du document uploadé
                # On prend 3 phrases au hasard de la DB pour ce chapitre
                extracted_qi = random.sample(DB_MATHS[target_chap], 3)
                
                st.session_state['test_results'] = []
                for qi in extracted_qi:
                    found, qc_ref, score = test_mapping(qi, st.session_state['df_qc'])
                    st.session_state['test_results'].append({
                        "Qi_Extrait": qi,
                        "Statut": "✅ COUVERT" if found else "❌ ANGLE MORT",
                        "QC_Moteur": qc_ref if found else "---",
                        "Pertinence_QC": score
                    })

    with col_res:
        if 'test_results' in st.session_state:
            st.markdown("### Résultat du Mapping")
            df_res = pd.DataFrame(st.session_state['test_results'])
            
            # Coloration
            def color_row(row):
                bg = '#d1fae5' if row['Statut'] == "✅ COUVERT" else '#fee2e2'
                return [f'background-color: {bg}; color: black'] * len(row)

            st.dataframe(df_res.style.apply(color_row, axis=1), use_container_width=True)
            
            # KPI
            couverts = len(df_res[df_res["Statut"]=="✅ COUVERT"])
            total = len(df_res)
            st.metric("Taux de Couverture Document", f"{(couverts/total)*100:.0f}%")
