import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Factory V5.1")
st.title("🏭 SMAXIA - Console Factory & Crash Test (V5.1)")

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
    
    # 1. SOURCING
    progress = st.progress(0)
    total_ops = len(urls) * n_per_url
    counter = 0
    
    for i, url in enumerate(urls):
        if not url.strip(): continue
        for j in range(n_per_url):
            counter += 1
            progress.progress(min(counter/total_ops, 1.0))
            time.sleep(0.002) 
            
            nature = random.choice(natures)
            year = random.choice(range(2019, 2025))
            file_id = f"DOC_{i}_{j}"
            filename = f"Sujet_{nature}_{year}_{j}.pdf"
            
            # On simule un lien de téléchargement
            download_link = f"https://fake-smaxia-cloud.com/dl/{filename}"
            
            sources_log.append({
                "ID": file_id,
                "Fichier": filename,
                "Télécharger": download_link, # Lien simulé
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
    if df_qi.empty:
        return df_sources, df_qi, pd.DataFrame()

    grouped = df_qi.groupby(["Chapitre", "Qi_Brut"]).agg({
        "ID_Source": "count",      # n_q
        "Année": "max",            # Récence
        "Fichier_Origine": list
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
        
        score = (n_q / N_total) * (1 + alpha/tau) * psi * (1-sigma) * 100
        qc_name = f"COMMENT {row['Qi_Brut']}..."
        
        qcs.append({
            "CHAPITRE": row["Chapitre"],
            "QC_INVARIANTE": qc_name,
            "SCORE_F2": score,
            "n_q": n_q,
            "N_tot": N_total,
            "Tau": tau,
            "QI_ASSOCIES": row["Fichier_Origine"]
        })
        
    df_qc = pd.DataFrame(qcs).sort_values(by=["CHAPITRE", "SCORE_F2"], ascending=[True, False])
    
    # AJOUT DES IDs UNIQUES (QC_1, QC_2...)
    df_qc = df_qc.reset_index(drop=True)
    df_qc["QC_ID"] = df_qc.index + 1
    df_qc["QC_ID"] = df_qc["QC_ID"].apply(lambda x: f"QC_{x:03d}")
    
    # Réorganiser les colonnes
    cols = ["QC_ID"] + [c for c in df_qc.columns if c != "QC_ID"]
    df_qc = df_qc[cols]
    
    return df_sources, df_qi, df_qc

def analyze_external_subject(target_chapitre, doc_type, df_qc_engine):
    """
    Simule l'analyse d'un sujet externe injecté pour le test
    """
    extracted_qi = []
    if target_chapitre in DB_MATHS:
        existing_qi = random.sample(DB_MATHS[target_chapitre], k=min(3, len(DB_MATHS[target_chapitre])))
        extracted_qi.extend(existing_qi)
    extracted_qi.append("Démontrer la conjecture de Riemann (Question hors programme)")
    
    results = []
    
    for qi in extracted_qi:
        match_found = False
        match_id = "---"
        match_text = "---"
        match_score = 0
        
        for idx, row in df_qc_engine.iterrows():
            core_qc = row["QC_INVARIANTE"].replace("COMMENT ", "").replace("...", "")
            if core_qc in qi:
                match_found = True
                match_id = row["QC_ID"]
                match_text = row["QC_INVARIANTE"]
                match_score = row["SCORE_F2"]
                break
        
        results.append({
            "Qi_Enonce": qi,
            "Statut": "✅ MATCH" if match_found else "❌ GAP",
            "QC_ID": match_id,
            "QC_Moteur": match_text,
            "Score_F2": match_score
        })
        
    return pd.DataFrame(results)

# --- INTERFACE ---

# SIDEBAR
with st.sidebar:
    st.header("1. Périmètre Usine")
    chapitres_actifs = st.multiselect(
        "Chapitres Cibles", 
        list(DB_MATHS.keys()), 
        default=["SUITES NUMÉRIQUES"]
    )

# TABS
tab_factory, tab_test = st.tabs(["🏭 USINE (Production)", "🧪 CRASH TEST (Validation)"])

# --- TAB 1 : USINE ---
with tab_factory:
    st.subheader("A. Sourcing & Génération QC")

    col_input, col_act = st.columns([3, 1])
    with col_input:
        urls_input = st.text_area("Sources (URLs)", "https://apmep.fr/terminale\nhttps://sujetdebac.fr", height=70)
    with col_act:
        n_sujets = st.number_input("Vol. par URL", 5, 100, 10)
        btn_run = st.button("LANCER L'USINE 🚀", type="primary")

    if btn_run:
        url_list = urls_input.split('\n')
        with st.spinner("Traitement en cours..."):
            df_src, df_qi, df_qc = ingest_and_calculate(url_list, n_sujets, chapitres_actifs)
            st.session_state['df_src'] = df_src
            st.session_state['df_qi'] = df_qi
            st.session_state['df_qc'] = df_qc
            st.success("Usine mise à jour.")
            # On force le rechargement pour afficher les nouvelles colonnes proprement
            st.rerun()

    st.divider()

    # VUE SPLIT USINE
    if 'df_qc' in st.session_state:
        col_left, col_right = st.columns([1, 1.5])
        
        # --- GAUCHE : SUJETS SOURCÉS ---
        with col_left:
            st.markdown(f"### 📥 Sujets ({len(st.session_state['df_src'])})")
            
            # SÉCURITÉ ANTI-CRASH V5
            # Si l'utilisateur a de vieilles données en cache sans la colonne 'Télécharger', on gère l'erreur
            if "Télécharger" not in st.session_state['df_src'].columns:
                st.warning("⚠️ Données obsolètes détectées. Veuillez relancer l'usine (bouton rouge).")
                df_display_src = st.session_state['df_src'][["Fichier", "Nature", "Année"]]
            else:
                df_display_src = st.session_state['df_src'][["Fichier", "Nature", "Année", "Télécharger"]]

            st.dataframe(
                df_display_src,
                column_config={
                    "Télécharger": st.column_config.LinkColumn("Action", display_text="📥 Télécharger"),
                },
                use_container_width=True,
                height=600
            )

        # --- DROITE : QC GÉNÉRÉES ---
        with col_right:
            total_qc = len(st.session_state['df_qc'])
            st.markdown(f"### 🧠 QC Générées (Total : {total_qc})")
            
            if not st.session_state['df_qc'].empty:
                # SÉCURITÉ : Vérification que QC_ID existe
                if "QC_ID" not in st.session_state['df_qc'].columns:
                     st.warning("⚠️ Ancienne structure QC détectée. Relancez l'usine.")
                else:
                    # Filtre Chapitre
                    available_chaps = st.session_state['df_qc']["CHAPITRE"].unique()
                    chap_filter = st.selectbox("Filtrer par Chapitre", available_chaps)
                    
                    df_view_qc = st.session_state['df_qc'][st.session_state['df_qc']["CHAPITRE"] == chap_filter]
                    
                    if not df_view_qc.empty:
                        for idx, row in df_view_qc.iterrows():
                            with st.container():
                                # En-tête avec QC_ID
                                c1, c2 = st.columns([0.5, 3])
                                with c1:
                                    st.markdown(f"**`{row['QC_ID']}`**")
                                with c2:
                                    st.info(f"**{row['QC_INVARIANTE']}**")
                                
                                # Détails Score
                                k1, k2, k3, k4 = st.columns(4)
                                k1.caption(f"Score F2: **{row['SCORE_F2']:.1f}**")
                                k2.caption(f"Freq (n_q): {row['n_q']}")
                                k3.caption(f"Récence (τ): {row['Tau']}")
                                k4.caption(f"Densité (Ψ): 1.0")
                                
                                # Preuve
                                with st.expander("Voir les Qi sources"):
                                    st.dataframe(pd.DataFrame(row['QI_ASSOCIES'], columns=["Fichiers Sources"]), hide_index=True)
                                st.divider()
                    else:
                        st.info("Aucune QC pour ce chapitre.")

# --- TAB 2 : CRASH TEST ---
with tab_test:
    st.subheader("B. Zone de Test (Mapping Enoncé -> QC)")
    
    if 'df_qc' in st.session_state and "QC_ID" in st.session_state['df_qc'].columns:
        
        # 1. SIMULATION UPLOAD
        col_up, col_param = st.columns([2, 1])
        with col_up:
            st.file_uploader("Télécharger un sujet (PDF/Image)", type=["pdf", "png", "jpg"])
            st.caption("*(Simulation : le système va extraire le texte automatiquement)*")
        with col_param:
            doc_type = st.selectbox("Type Document", ["DST", "BAC", "EXO"])
            target_chap = st.selectbox("Chapitre Supposé", chapitres_actifs)
            btn_test = st.button("ANALYSER L'ÉNONCÉ")
        
        # 2. RÉSULTAT ANALYSE
        if btn_test:
            st.divider()
            st.markdown("#### Résultats de l'Atomisation & Mapping")
            
            # Lancer l'analyse simulée
            df_res_test = analyze_external_subject(target_chap, doc_type, st.session_state['df_qc'])
            
            # KPI
            nb_qi = len(df_res_test)
            nb_match = len(df_res_test[df_res_test["Statut"] == "✅ MATCH"])
            taux = (nb_match / nb_qi) * 100
            
            k1, k2 = st.columns(2)
            k1.metric("Qi extraites de l'énoncé", nb_qi)
            k2.metric("Taux de Couverture", f"{taux:.0f}%")
            
            # TABLEAU DE MAPPING (La demande clé)
            st.markdown("##### Tableau de Correspondance (Qi vs QC)")
            
            def highlight_status(val):
                color = '#dcfce7' if val == '✅ MATCH' else '#fee2e2'
                return f'background-color: {color}; color: black'

            st.dataframe(
                df_res_test[["Qi_Enonce", "Statut", "QC_ID", "QC_Moteur"]].style.map(highlight_status, subset=['Statut']),
                column_config={
                    "Qi_Enonce": st.column_config.TextColumn("1. Qi (Enoncé Élève)", width="large"),
                    "Statut": st.column_config.TextColumn("2. Verdict", width="small"),
                    "QC_ID": st.column_config.TextColumn("3. Ref ID", width="small"),
                    "QC_Moteur": st.column_config.TextColumn("4. QC SMAXIA (Réponse)", width="large")
                },
                use_container_width=True
            )
            
            if taux < 100:
                st.error("⚠️ Attention : Certaines questions de ce sujet ne trouvent pas de réponse dans le moteur actuel.")
            else:
                st.success("✅ Succès : Le moteur couvre intégralement ce sujet.")
                
    else:
        st.warning("⚠️ Le moteur est vide ou obsolète. Veuillez lancer l'usine dans l'onglet 1.")
