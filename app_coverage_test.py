import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Factory V9.2")
st.title("🏭 SMAXIA - Console Factory V9.2 (Audit Complet)")

# ==============================================================================
# 🧱 ETAPE 0 : KERNEL SMAXIA (Espace Canonique E_c)
# ==============================================================================

KERNEL_MAPPING = {
    "FRT_SUITE_01": "SUITES NUMÉRIQUES", "FRT_SUITE_02": "SUITES NUMÉRIQUES", "FRT_SUITE_03": "SUITES NUMÉRIQUES",
    "FRT_FCT_01": "FONCTIONS", "FRT_FCT_02": "FONCTIONS", "FRT_FCT_03": "FONCTIONS",
    "FRT_GEO_01": "GÉOMÉTRIE", "FRT_GEO_02": "GÉOMÉTRIE",
    "FRT_PROBA_01": "PROBABILITÉS", "FRT_PROBA_02": "PROBABILITÉS"
}

SMAXIA_KERNEL = {
    # SUITES
    "FRT_SUITE_01": {"QC": "COMMENT Démontrer qu'une suite est géométrique", "Weights": [0.2, 0.3, 0.2, 0.1], "Delta": 1.2},
    "FRT_SUITE_02": {"QC": "COMMENT Calculer la limite d'une suite (Théorèmes)", "Weights": [0.2, 0.3, 0.3, 0.2], "Delta": 1.1},
    "FRT_SUITE_03": {"QC": "COMMENT Démontrer par récurrence", "Weights": [0.1, 0.2, 0.6, 0.1], "Delta": 1.5},
    # FONCTIONS
    "FRT_FCT_01": {"QC": "COMMENT Étudier les variations d'une fonction", "Weights": [0.3, 0.3, 0.2, 0.1], "Delta": 1.3},
    "FRT_FCT_02": {"QC": "COMMENT Appliquer le TVI (Solution unique)", "Weights": [0.1, 0.2, 0.2, 0.4], "Delta": 1.4},
    "FRT_FCT_03": {"QC": "COMMENT Déterminer l'équation de la tangente", "Weights": [0.1, 0.2, 0.2, 0.1], "Delta": 0.9},
    # GEO
    "FRT_GEO_01": {"QC": "COMMENT Démontrer Droite Orthogonale Plan", "Weights": [0.1, 0.1, 0.4, 0.2], "Delta": 1.3},
    "FRT_GEO_02": {"QC": "COMMENT Déterminer une représentation paramétrique", "Weights": [0.2, 0.2, 0.4], "Delta": 1.0},
    # PROBA
    "FRT_PROBA_01": {"QC": "COMMENT Calculer une probabilité totale (Arbre)", "Weights": [0.1, 0.3, 0.2, 0.2], "Delta": 1.1},
    "FRT_PROBA_02": {"QC": "COMMENT Utiliser la Loi Binomiale", "Weights": [0.3, 0.1, 0.3, 0.1], "Delta": 1.2}
}

# --- GÉNÉRATEUR DE QI (Polymorphisme) ---
QI_TEMPLATES = {
    "FRT_SUITE_01": ["Montrer que (Un) est géométrique.", "Prouver que Vn est une suite géométrique.", "Justifier le caractère géométrique."],
    "FRT_SUITE_02": ["Déterminer la limite de Un.", "Calculer la limite quand n tend vers l'infini.", "Étudier la convergence."],
    "FRT_SUITE_03": ["Démontrer par récurrence que Un > 0.", "Montrer par récurrence que Un < 5.", "Prouver la propriété P(n)."],
    "FRT_FCT_01": ["Étudier les variations de f.", "Dresser le tableau de variations.", "Donner le sens de variation."],
    "FRT_FCT_02": ["Montrer que f(x)=0 a une solution unique.", "Démontrer l'existence d'une solution alpha.", "Résoudre f(x)=k."],
    "FRT_FCT_03": ["Donner l'équation de la tangente T.", "Déterminer la tangente en a.", "Quelle est l'équation réduite ?"],
    "FRT_GEO_01": ["Montrer que (d) est orthogonale à (P).", "Prouver que la droite est perpendiculaire.", "Vérifier l'orthogonalité."],
    "FRT_GEO_02": ["Donner une représentation paramétrique.", "Déterminer le système paramétrique."],
    "FRT_PROBA_01": ["Calculer P(B).", "Quelle est la probabilité totale ?", "Utiliser l'arbre pour calculer p."],
    "FRT_PROBA_02": ["Calculer P(X=k).", "Quelle est la probabilité de k succès ?", "Appliquer la loi binomiale."]
}

def generate_smart_qi(frt_id):
    if frt_id not in QI_TEMPLATES: return "Question Standard"
    text = random.choice(QI_TEMPLATES[frt_id])
    context = random.choice(["", " sur I", " dans le repère", " pour tout n"])
    return text + context

# ==============================================================================
# ⚙️ MOTEUR KERNEL
# ==============================================================================

def calculate_psi_real(frt_id):
    """Calcul F1 Réel : Psi dépend de la complexité ARI"""
    data = SMAXIA_KERNEL[frt_id]
    psi_raw = data["Delta"] * (0.1 + sum(data["Weights"]))**2
    return round(min(psi_raw / 4.0, 1.0), 4)

def ingest_and_process(urls, n_per_url, selected_chapters):
    """Usine de génération"""
    sources_log = []
    atoms_db = []
    progress = st.progress(0)
    
    # Filtre des FRT actives
    active_frts = [k for k, v in KERNEL_MAPPING.items() if v in selected_chapters]
    if not active_frts: return pd.DataFrame(), pd.DataFrame()

    total_ops = len(urls) * n_per_url if len(urls) > 0 else 1
    counter = 0
    natures = ["BAC", "DST", "CONCOURS"]
    
    for i, url in enumerate(urls):
        if not url.strip(): continue
        for j in range(n_per_url):
            counter += 1
            progress.progress(min(counter/total_ops, 1.0))
            time.sleep(0.002)
            
            nature = random.choice(natures)
            year = random.choice(range(2020, 2025))
            filename = f"Sujet_{nature}_{year}_{j}.txt"
            file_id = f"DOC_{i}_{j}"
            
            # Pioche 2-3 exos
            nb_exos = random.randint(2, 3)
            frts_in_doc = random.sample(active_frts, k=min(nb_exos, len(active_frts)))
            
            qi_list = []
            for frt_id in frts_in_doc:
                qi_txt = generate_smart_qi(frt_id)
                atoms_db.append({
                    "ID_Source": file_id, "Année": year, "Qi_Brut": qi_txt,
                    "FRT_ID": frt_id, "Fichier": filename, 
                    "Chapitre": KERNEL_MAPPING[frt_id]
                })
                qi_list.append(qi_txt)
                
            content = f"SUJET {filename}\nTYPE: {nature}\n" + "\n".join([f"- {q}" for q in qi_list])
            sources_log.append({"Fichier": filename, "Nature": nature, "Année": year, "Contenu": content})
            
    progress.empty()
    return pd.DataFrame(sources_log), pd.DataFrame(atoms_db)

def compute_engine_metrics(df_atoms):
    if df_atoms.empty: return pd.DataFrame()
    
    # Regroupement par FRT (Structure)
    grouped = df_atoms.groupby("FRT_ID").agg({
        "ID_Source": "count", "Année": "max", 
        "Qi_Brut": list, "Fichier": list, "Chapitre": "first"
    }).reset_index()
    
    qcs = []
    N_total = len(df_atoms)
    current_year = datetime.now().year
    
    for idx, row in grouped.iterrows():
        frt_id = row["FRT_ID"]
        kernel = SMAXIA_KERNEL[frt_id]
        
        # Variables
        n_q = row["ID_Source"]
        tau = max((current_year - row["Année"]), 0.5)
        alpha = 5.0
        psi = calculate_psi_real(frt_id)
        sigma = 0.05
        
        # Équation F2
        score = (n_q / N_total) * (1 + alpha/tau) * psi * (1-sigma) * 100
        
        # Preuve (Liste Qi)
        evidence = [{"Fichier": f, "Qi": q} for f, q in zip(row["Fichier"], row["Qi_Brut"])]
        
        qcs.append({
            "Chapitre": row["Chapitre"],
            "QC_ID": f"QC_{idx+1:02d}", # ID Simple (QC_01)
            "FRT_ID": frt_id,           # ID Technique
            "QC_Titre": kernel["QC"],   # Titre FRT
            "Score_F2": score,
            # Variables pour Audit
            "n_q": n_q, "N_tot": N_total, "Tau": tau, 
            "Alpha": alpha, "Psi": psi, "Sigma": sigma,
            "Evidence": evidence
        })
        
    return pd.DataFrame(qcs).sort_values(by=["Chapitre", "Score_F2"], ascending=[True, False])

# ==============================================================================
# 🖥️ INTERFACE
# ==============================================================================

with st.sidebar:
    st.header("1. Paramètres")
    matiere = st.selectbox("Matière", ["MATHS"])
    all_chaps = list(set(KERNEL_MAPPING.values()))
    selected_chaps = st.multiselect("Chapitres", all_chaps, default=all_chaps)

# USINE
st.subheader("A. Sourcing & Génération QC")
c1, c2 = st.columns([3, 1])
with c1: urls_input = st.text_area("Sources", "https://apmep.fr", height=70)
with c2: 
    n_sujets = st.number_input("Vol. par URL", 5, 100, 10, step=5)
    btn_run = st.button("LANCER USINE 🚀", type="primary")

if btn_run:
    url_list = urls_input.split('\n')
    with st.spinner("Traitement..."):
        df_src, df_atoms = ingest_and_process(url_list, n_sujets, selected_chaps)
        df_qc = compute_engine_metrics(df_atoms)
        st.session_state['df_src'] = df_src
        st.session_state['df_qc'] = df_qc
        st.success("Terminé.")

st.divider()

if 'df_qc' in st.session_state:
    col_left, col_right = st.columns([1, 1.5])
    
    # GAUCHE : SUJETS
    with col_left:
        st.markdown(f"### 📥 Sujets ({len(st.session_state['df_src'])})")
        st.dataframe(st.session_state['df_src'][["Fichier", "Nature", "Année"]], use_container_width=True, height=400)
        
        sel = st.selectbox("Sélectionner pour télécharger", st.session_state['df_src']["Fichier"])
        if not st.session_state['df_src'].empty:
            txt = st.session_state['df_src'][st.session_state['df_src']["Fichier"]==sel].iloc[0]["Contenu"]
            st.download_button("📥 Télécharger (.txt)", txt, file_name=sel)

    # DROITE : QC (AUDIT)
    with col_right:
        st.markdown("### 🧠 QC Générées (Détail F2 & Preuve)")
        
        if not st.session_state['df_qc'].empty:
            chapters = st.session_state['df_qc']["Chapitre"].unique()
            
            for chap in chapters:
                st.markdown(f"#### 📘 {chap}")
                df_view = st.session_state['df_qc'][st.session_state['df_qc']["Chapitre"] == chap]
                
                for idx, row in df_view.iterrows():
                    with st.container():
                        # EN-TÊTE : ID + FRT
                        c1, c2 = st.columns([0.5, 3])
                        c1.markdown(f"**`{row['QC_ID']}`**") # QC_01
                        c2.info(f"**{row['QC_Titre']}**")    # FRT
                        
                        st.caption(f"Score F2 : **{row['Score_F2']:.2f}**")
                        
                        # TABLEAU VARIABLES (Demandé)
                        vars_df = pd.DataFrame({
                            "Variable": ["n_q", "N_tot", "Tau (τ)", "Alpha (α)", "Psi (Ψ)", "Sigma (σ)"],
                            "Valeur": [row['n_q'], row['N_tot'], row['Tau'], row['Alpha'], row['Psi'], row['Sigma']]
                        }).T
                        st.table(vars_df) # Table statique pour lisibilité
                        
                        # LISTE DES Qi (Preuve)
                        with st.expander(f"Voir les {row['n_q']} Qi sources"):
                            st.dataframe(pd.DataFrame(row['Evidence']), hide_index=True, use_container_width=True)
                        st.divider()
        else:
            st.warning("Rien à afficher.")
