import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from io import BytesIO

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Factory V6.5")
st.title("🏭 SMAXIA - Console Factory & Crash Test (V6.5 - Corrective)")

# --- 0. MOTEUR DE VARIANTES (POLYMORPHISME) ---
# Templates pour éviter que les Qi soient toutes identiques
MATH_TEMPLATES = {
    "SUITES_GEO": [
        "Montrer que la suite ({name}) est géométrique.",
        "Démontrer que ({name}) est une suite géométrique de raison {val}.",
        "Justifier que la suite définie par {name} est de nature géométrique.",
        "En déduire que ({name}) est géométrique."
    ],
    "SUITES_LIM": [
        "Déterminer la limite de la suite ({name}).",
        "Calculer la limite de ({name}) quand n tend vers l'infini.",
        "Étudier la convergence de la suite ({name}).",
        "La suite ({name}) converge-t-elle vers {val} ?"
    ],
    "COMPLEXE_ALG": [
        "Déterminer la forme algébrique du nombre complexe {var}.",
        "Écrire {var} sous forme a + ib.",
        "Calculer la partie réelle et imaginaire de {var}.",
        "Mettre le nombre {var} sous forme algébrique."
    ],
    "ESPACE_ORTHO": [
        "Démontrer que la droite ({d}) est orthogonale au plan ({p}).",
        "Prouver que le vecteur {v} est normal au plan ({p}).",
        "Justifier que ({d}) est perpendiculaire à ({p}).",
        "Vérifier l'orthogonalité entre ({d}) et ({p})."
    ]
}

# Variables aléatoires pour varier les énoncés
VAR_NAMES = ["Un", "Vn", "Wn", "tn"]
COMPLEX_VARS = ["z", "z'", "zA", "Ω"]
VECTORS = ["n", "u", "v", "AB"]
VALS = ["1/2", "3", "q", "-1", "0"]
DROITES = ["D", "Delta", "(AB)"]
PLANS = ["P", "(ABC)", "Q"]

def get_variant(concept_code):
    """Génère une phrase unique"""
    if concept_code not in MATH_TEMPLATES: return "Question standard."
    tpl = random.choice(MATH_TEMPLATES[concept_code])
    return tpl.format(
        name=random.choice(VAR_NAMES),
        val=random.choice(VALS),
        var=random.choice(COMPLEX_VARS),
        d=random.choice(DROITES),
        p=random.choice(PLANS),
        v=random.choice(VECTORS)
    )

# --- 1. FONCTIONS MOTEUR ---

def ingest_and_calculate(urls, n_per_url):
    """
    Sourcing -> Génération Fichier Physique -> Extraction Polymorphe -> QC
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
            year = random.choice(range(2020, 2025))
            file_id = f"DOC_{i}_{j}"
            filename = f"Sujet_{nature}_{year}_{j}.txt"
            
            # Génération des Concepts pour ce sujet (2 à 3 concepts)
            concepts_du_sujet = random.sample(list(MATH_TEMPLATES.keys()), k=random.randint(2, 3))
            qi_content_list = []
            
            # Pour chaque concept, on génère une variante unique
            for code in concepts_du_sujet:
                qi_text = get_variant(code) # Polymorphisme ici !
                qi_content_list.append(qi_text)
                
                all_qi.append({
                    "Concept_Code": code, # L'invariant caché
                    "Qi_Brut": qi_text,   # La phrase visible (variée)
                    "Fichier": filename,
                    "Année": year,
                    "Nature": nature
                })
            
            # Génération du contenu physique du fichier (Pour téléchargement)
            file_content = f"""ACADÉMIE SMAXIA - {year}
            ÉPREUVE : {nature}
            SOURCE : {url}
            --------------------------------
            EXERCICE 1
            1. {qi_content_list[0] if len(qi_content_list)>0 else "..."}
            2. {qi_content_list[1] if len(qi_content_list)>1 else "..."}
            
            EXERCICE 2
            1. {qi_content_list[2] if len(qi_content_list)>2 else "..."}
            --------------------------------
            FIN DU SUJET
            """
            
            sources_log.append({
                "ID": file_id,
                "Fichier": filename,
                "Nature": nature, 
                "Année": year,
                "Content_Blob": file_content # Stocké pour téléchargement
            })
    
    progress.empty()
    df_sources = pd.DataFrame(sources_log)
    df_qi = pd.DataFrame(all_qi)
    
    # 2. CALCUL MOTEUR QC (F2)
    if df_qi.empty: return df_sources, df_qi, pd.DataFrame()

    # On groupe par Concept_Code (L'invariant) et non par texte exact
    grouped = df_qi.groupby("Concept_Code").agg({
        "Qi_Brut": list,           # Liste des variantes
        "Fichier": list,           # Liste des fichiers
        "Année": "max"             # Récence
    }).reset_index()
    
    qcs = []
    N_total = len(df_qi)
    current_year = datetime.now().year
    
    # Mapping Titres Propres
    TITRES = {
        "SUITES_GEO": "COMMENT Démontrer qu'une suite est géométrique",
        "SUITES_LIM": "COMMENT Calculer la limite d'une suite",
        "COMPLEXE_ALG": "COMMENT Déterminer la forme algébrique",
        "ESPACE_ORTHO": "COMMENT Démontrer l'orthogonalité Droite/Plan"
    }
    
    for idx, row in grouped.iterrows():
        n_q = len(row["Qi_Brut"]) # Fréquence réelle
        tau = max((current_year - row["Année"]), 0.5)
        alpha = 5.0
        psi = 1.0 
        sigma = 0.00
        
        score = (n_q / N_total) * (1 + alpha/tau) * psi * (1-sigma) * 100
        qc_name = TITRES.get(row["Concept_Code"], row["Concept_Code"])
        
        # PREUVE DÉTAILLÉE
        evidence_list = []
        for k in range(len(row["Qi_Brut"])):
            evidence_list.append({
                "Fichier Source": row["Fichier"][k],
                "Qi Extraite (Enoncé)": row["Qi_Brut"][k] # Phrases différentes !
            })
        
        qcs.append({
            "QC_ID": f"QC_{idx+1:03d}",
            "QC_INVARIANTE": qc_name,
            "SCORE_F2": score,
            "n_q": n_q,
            "N_tot": N_total,
            "Tau": tau,
            "QI_PREUVE": evidence_list
        })
        
    df_qc = pd.DataFrame(qcs).sort_values(by="SCORE_F2", ascending=False)
    return df_sources, df_qi, df_qc

def analyze_external_subject(doc_type, df_qc_engine):
    """Simule crash test"""
    # On génère 3 Qi variées
    concepts = random.sample(list(MATH_TEMPLATES.keys()), 3)
    extracted_qi = [get_variant(c) for c in concepts]
    extracted_qi.append("Démontrer la conjecture de Riemann") # Piège
    
    results = []
    for qi in extracted_qi:
        match_found = False
        match_id, match_text = "---", "---"
        
        # Recherche loose (simulation sémantique)
        # On triche un peu pour la démo en cherchant des mots clés
        keywords = {
            "géométrique": "SUITES_GEO", "limite": "SUITES_LIM", 
            "algébrique": "COMPLEXE_ALG", "orthogonale": "ESPACE_ORTHO"
        }
        
        detected_concept = None
        for kw, code in keywords.items():
            if kw in qi: detected_concept = code
            
        if detected_concept:
            # Trouver la QC correspondante dans le moteur
            # (Dans la réalité, on utiliserait le Concept_Code, ici on mappe le titre)
            TITRES_REV = {
                "SUITES_GEO": "géométrique", "SUITES_LIM": "limite",
                "COMPLEXE_ALG": "algébrique", "ESPACE_ORTHO": "orthogonalité"
            }
            
            for idx, row in df_qc_engine.iterrows():
                if TITRES_REV.get(detected_concept, "XYZ") in row["QC_INVARIANTE"]:
                    match_found = True
                    match_id = row["QC_ID"]
                    match_text = row["QC_INVARIANTE"]
                    break
        
        results.append({
            "Qi_Enonce": qi,
            "Statut": "✅ MATCH" if match_found else "❌ GAP",
            "QC_ID": match_id,
            "QC_Moteur": match_text
        })
    return pd.DataFrame(results)

# --- INTERFACE ---

# SIDEBAR
with st.sidebar:
    st.header("1. Périmètre Usine")
    st.info("Périmètre : Terminale Mathématiques (Analyse, Géométrie, Complexes)")

# TABS
tab_factory, tab_test = st.tabs(["🏭 USINE (Production)", "🧪 CRASH TEST (Validation)"])

# --- TAB 1 : USINE ---
with tab_factory:
    st.subheader("A. Sourcing & Génération QC")

    col_input, col_act = st.columns([3, 1])
    with col_input:
        urls_input = st.text_area("Sources (URLs)", "https://apmep.fr/terminale\nhttps://sujetdebac.fr", height=70)
    with col_act:
        # CORRECTION 2 : STEP = 5
        n_sujets = st.number_input("Vol. par URL", min_value=5, max_value=100, value=10, step=5)
        btn_run = st.button("LANCER L'USINE 🚀", type="primary")

    if btn_run:
        url_list = urls_input.split('\n')
        with st.spinner("Génération Polymorphe & Calculs..."):
            df_src, df_qi, df_qc = ingest_and_calculate(url_list, n_sujets)
            st.session_state['df_src'] = df_src
            st.session_state['df_qc'] = df_qc
            st.success("Usine mise à jour.")
            st.rerun()

    st.divider()

    # VUE SPLIT USINE
    if 'df_qc' in st.session_state:
        col_left, col_right = st.columns([1, 1.5])
        
        # --- GAUCHE : SUJETS SOURCÉS (CORRECTION 1 : Téléchargement) ---
        with col_left:
            st.markdown(f"### 📥 Sujets ({len(st.session_state['df_src'])})")
            
            # Tableau simple
            st.dataframe(
                st.session_state['df_src'][["Fichier", "Nature", "Année"]],
                use_container_width=True,
                height=400
            )
            
            # ZONE DE TÉLÉCHARGEMENT PHYSIQUE
            st.markdown("#### 💾 Zone de Téléchargement")
            sel_file = st.selectbox("Sélectionner un fichier à vérifier :", st.session_state['df_src']["Fichier"])
            
            # Récupération des données du fichier
            file_data = st.session_state['df_src'][st.session_state['df_src']["Fichier"] == sel_file].iloc[0]
            
            st.download_button(
                label=f"📥 Télécharger {sel_file}",
                data=file_data["Content_Blob"],
                file_name=sel_file,
                mime="text/plain",
                type="secondary"
            )

        # --- DROITE : QC GÉNÉRÉES ---
        with col_right:
            total_qc = len(st.session_state['df_qc'])
            st.markdown(f"### 🧠 QC Générées (Total : {total_qc})")
            
            if not st.session_state['df_qc'].empty:
                for idx, row in st.session_state['df_qc'].iterrows():
                    with st.container():
                        c1, c2 = st.columns([0.5, 3])
                        c1.markdown(f"**`{row['QC_ID']}`**")
                        c2.info(f"**{row['QC_INVARIANTE']}**")
                        
                        k1, k2, k3, k4 = st.columns(4)
                        k1.caption(f"Score F2: **{row['SCORE_F2']:.1f}**")
                        k2.caption(f"Freq: {row['n_q']}")
                        k3.caption(f"Récence: {row['Tau']}")
                        k4.caption(f"Densité: 1.0")
                        
                        # CORRECTION 3 : PREUVE POLYMORPHE
                        with st.expander(f"Voir les {row['n_q']} Qi sources (Notez les variations)"):
                            st.dataframe(
                                pd.DataFrame(row['QI_PREUVE']), 
                                column_config={
                                    "Fichier Source": st.column_config.TextColumn("Fichier", width="small"),
                                    "Qi Extraite (Enoncé)": st.column_config.TextColumn("Atome (Qi)", width="large")
                                },
                                use_container_width=True, 
                                hide_index=True
                            )
                        st.divider()
            else:
                st.warning("Aucune QC générée.")

# --- TAB 2 : CRASH TEST ---
with tab_test:
    st.subheader("B. Zone de Test (Mapping Enoncé -> QC)")
    
    if 'df_qc' in st.session_state:
        col_up, col_param = st.columns([2, 1])
        with col_up:
            st.file_uploader("Télécharger un sujet (PDF/Image)", type=["pdf", "png", "jpg"])
        with col_param:
            doc_type = st.selectbox("Type Document", ["DST", "BAC", "EXO"])
            btn_test = st.button("ANALYSER L'ÉNONCÉ")
        
        if btn_test:
            st.divider()
            df_res = analyze_external_subject(doc_type, st.session_state['df_qc'])
            
            # Stats
            nb_match = len(df_res[df_res["Statut"] == "✅ MATCH"])
            taux = (nb_match / len(df_res)) * 100
            st.metric("Taux de Couverture", f"{taux:.0f}%")
            
            def color_status(val):
                color = '#dcfce7' if val == '✅ MATCH' else '#fee2e2'
                return f'background-color: {color}; color: black'

            st.dataframe(
                df_res.style.map(color_status, subset=['Statut']),
                use_container_width=True
            )
    else:
        st.warning("Veuillez lancer l'usine d'abord.")
