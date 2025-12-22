import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V10.1")
st.title("🛡️ SMAXIA - Console V10.1 (UI Finalisée)")

# Styles CSS
st.markdown("""
<style>
    .qc-id { font-weight: bold; color: #d97706; font-size: 1.1em; }
    .qc-text { font-size: 1.1em; font-family: sans-serif; font-weight: 500; }
    .qc-score { color: #dc2626; font-weight: bold; }
    .stExpander { border: 1px solid #e5e7eb; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

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
    "FRT_SUITE_01": {"QC": "COMMENT Démontrer qu'une suite est géométrique", "ARI": ["Calculer u(n+1)", "Factoriser par u(n)", "Identifier la raison q", "Conclure"], "Weights": [0.2, 0.3, 0.2, 0.1], "Delta": 1.2},
    "FRT_SUITE_02": {"QC": "COMMENT Calculer la limite d'une suite", "ARI": ["Identifier termes dominants", "Factoriser", "Limites usuelles", "Conclure"], "Weights": [0.2, 0.3, 0.3, 0.2], "Delta": 1.1},
    "FRT_SUITE_03": {"QC": "COMMENT Démontrer par récurrence", "ARI": ["Initialisation", "Hérédité", "Conclusion"], "Weights": [0.1, 0.2, 0.6, 0.1], "Delta": 1.5},
    # FONCTIONS
    "FRT_FCT_01": {"QC": "COMMENT Étudier les variations d'une fonction", "ARI": ["Dérivée f'", "Signe f'", "Tableau var", "Images aux bornes"], "Weights": [0.3, 0.3, 0.2, 0.1], "Delta": 1.3},
    "FRT_FCT_02": {"QC": "COMMENT Appliquer le TVI (Solution unique)", "ARI": ["Continuité", "Monotonie stricte", "Images bornes", "Corollaire TVI"], "Weights": [0.1, 0.2, 0.2, 0.4], "Delta": 1.4},
    "FRT_FCT_03": {"QC": "COMMENT Déterminer l'équation de la tangente", "ARI": ["Calculer f(a)", "Calculer f'(a)", "Formule y=f'(a)(x-a)+f(a)"], "Weights": [0.1, 0.2, 0.2, 0.1], "Delta": 0.9},
    # GEO
    "FRT_GEO_01": {"QC": "COMMENT Démontrer Droite Orthogonale Plan", "ARI": ["Vecteur directeur u", "Vecteurs plan v1, v2", "Produits scalaires nuls"], "Weights": [0.1, 0.1, 0.4, 0.2], "Delta": 1.3},
    "FRT_GEO_02": {"QC": "COMMENT Déterminer une représentation paramétrique", "ARI": ["Point A", "Vecteur u", "Système {x,y,z}"], "Weights": [0.2, 0.2, 0.4], "Delta": 1.0},
    # PROBA
    "FRT_PROBA_01": {"QC": "COMMENT Calculer une probabilité totale", "ARI": ["Arbre pondéré", "Identifier chemins", "Somme des probas"], "Weights": [0.1, 0.3, 0.2, 0.2], "Delta": 1.1},
    "FRT_PROBA_02": {"QC": "COMMENT Utiliser la Loi Binomiale", "ARI": ["Justifier Bernoulli (n,p)", "Formule P(X=k)", "Calcul"], "Weights": [0.3, 0.1, 0.3, 0.1], "Delta": 1.2}
}

# --- GÉNÉRATEUR ---
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
# ⚙️ MOTEUR
# ==============================================================================

def calculate_psi_real(frt_id):
    data = SMAXIA_KERNEL[frt_id]
    psi_raw = data["Delta"] * (0.1 + sum(data["Weights"]))**2
    return round(min(psi_raw / 4.0, 1.0), 4)

def ingest_and_process(urls, n_per_url, selected_chapters):
    sources_log = []
    atoms_db = []
    progress = st.progress(0)
    
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
            filename = f"Sujet_{nature}_{year}_{j}.pdf"
            file_id = f"DOC_{i}_{j}"
            
            nb_exos = random.randint(2, 4)
            frts_in_doc = random.sample(active_frts, k=min(nb_exos, len(active_frts)))
            
            qi_list_in_file = []
            
            for frt_id in frts_in_doc:
                qi_txt = generate_smart_qi(frt_id)
                atoms_db.append({
                    "ID_Source": file_id, "Année": year, "Qi_Brut": qi_txt,
                    "FRT_ID": frt_id, "Fichier": filename, 
                    "Chapitre": KERNEL_MAPPING[frt_id]
                })
                qi_list_in_file.append({"Qi": qi_txt, "FRT_ID": frt_id})
                
            content = f"CONTENU SIMULÉ PDF\nFICHIER: {filename}\n" + "\n".join([f"- {q['Qi']}" for q in qi_list_in_file])
            
            sources_log.append({
                "Fichier": filename, "Nature": nature, "Année": year, 
                "Contenu": content,
                "Qi_Data": qi_list_in_file 
            })
            
    progress.empty()
    return pd.DataFrame(sources_log), pd.DataFrame(atoms_db)

def compute_engine_metrics(df_atoms):
    if df_atoms.empty: return pd.DataFrame()
    
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
        
        n_q = row["ID_Source"]
        tau = max((current_year - row["Année"]), 0.5)
        alpha = 5.0
        psi = calculate_psi_real(frt_id)
        sigma = 0.05
        
        score = (n_q / N_total) * (1 + alpha/tau) * psi * (1-sigma) * 100
        evidence = [{"Fichier": f, "Qi": q} for f, q in zip(row["Fichier"], row["Qi_Brut"])]
        qc_clean = kernel["QC"].replace("COMMENT ", "comment ")
        
        qcs.append({
            "Chapitre": row["Chapitre"],
            "QC_ID_Simple": f"QC_{idx+1:02d}", 
            "FRT_ID": frt_id,
            "QC_Texte": qc_clean,
            "Score_F2": score,
            "n_q": n_q, "N_tot": N_total, "Tau": tau, "Alpha": alpha, "Psi": psi, "Sigma": sigma,
            "Evidence": evidence,
            "ARI": kernel["ARI"]
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

# TABS
tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine (Prod)", "✅ Onglet 2 : Audit (Validation Booléenne)"])

# --- TAB 1 : USINE ---
with tab_usine:
    # 1. INPUT
    c1, c2 = st.columns([3, 1])
    with c1: urls_input = st.text_area("Sources", "https://apmep.fr", height=70)
    with c2: 
        n_sujets = st.number_input("Vol. par URL", 5, 100, 10, step=5)
        btn_run = st.button("LANCER USINE 🚀", type="primary")

    if btn_run:
        with st.spinner("Traitement..."):
            df_src, df_atoms = ingest_and_process(urls_input.split('\n'), n_sujets, selected_chaps)
            df_qc = compute_engine_metrics(df_atoms)
            st.session_state['df_src'] = df_src
            st.session_state['df_qc'] = df_qc
            st.session_state['sel_chaps'] = selected_chaps
            st.success("Traitement terminé.")

    st.divider()

    if 'df_qc' in st.session_state:
        col_left, col_right = st.columns([1, 1.8])
        
        # GAUCHE : SUJETS
        with col_left:
            st.markdown(f"### 📥 Sujets ({len(st.session_state['df_src'])})")
            
            df_display = st.session_state['df_src'][["Fichier", "Nature", "Année"]].copy()
            df_display["Action"] = "📥 PDF" 
            
            st.dataframe(df_display, use_container_width=True, height=500)
            
            st.caption("Télécharger un sujet :")
            sel = st.selectbox("Fichier cible", st.session_state['df_src']["Fichier"], label_visibility="collapsed")
            if not st.session_state['df_src'].empty:
                txt = st.session_state['df_src'][st.session_state['df_src']["Fichier"]==sel].iloc[0]["Contenu"]
                st.download_button(f"Télécharger {sel}", txt, file_name=sel)

        # DROITE : QC
        with col_right:
            if not st.session_state['df_qc'].empty:
                chapters = st.session_state['df_qc']["Chapitre"].unique()
                
                for chap in chapters:
                    df_view = st.session_state['df_qc'][st.session_state['df_qc']["Chapitre"] == chap]
                    nb_qc = len(df_view)
                    
                    st.markdown(f"#### 📘 Chapitre {chap} : {nb_qc} QC")
                    
                    for idx, row in df_view.iterrows():
                        with st.container():
                            # HEADER : ID | Texte | Variables
                            k1, k2, k3 = st.columns([0.8, 3.5, 1.5])
                            
                            with k1: st.markdown(f"<span class='qc-id'>{row['QC_ID_Simple']}</span>", unsafe_allow_html=True)
                            with k2: st.markdown(f"<span class='qc-text'>{row['QC_Texte']}</span>", unsafe_allow_html=True)
                            with k3:
                                st.caption(f"Score F2: **{row['Score_F2']:.0f}**")
                                st.caption(f"n_q: {row['n_q']} | Ψ: {row['Psi']}")

                            # BOUTON 1 : FRT (RESOLUTION)
                            frt_label = f"🧬 FRT_{row['QC_ID_Simple']} (Résolution Type ARI)"
                            with st.expander(frt_label):
                                st.info("Algorithme de Résolution Invariant :")
                                for i, step in enumerate(row['ARI']):
                                    st.write(f"{i+1}. {step}")
                                    
                            # BOUTON 2 : QI (SOURCES)
                            qi_label = f"📋 Qi Associées ({row['n_q']}) - Voir les sources"
                            with st.expander(qi_label):
                                st.dataframe(pd.DataFrame(row['Evidence']), hide_index=True, use_container_width=True)
                            
                            st.write("---")
            else:
                st.warning("Aucune QC.")

# --- TAB 2 : AUDIT BOOLEEN ---
with tab_audit:
    st.subheader("Validation Booléenne")
    
    if 'df_qc' in st.session_state and 'df_src' in st.session_state:
        
        st.markdown("#### 1. Test Interne (Sujet Traité)")
        test_file = st.selectbox("Choisir un sujet traité", st.session_state['df_src']["Fichier"])
        
        if st.button("Lancer Test Interne"):
            file_data = st.session_state['df_src'][st.session_state['df_src']["Fichier"]==test_file].iloc[0]
            qi_list = file_data["Qi_Data"]
            
            c_ok = 0
            res_log = []
            available_frt_ids = st.session_state['df_qc']["FRT_ID"].unique()
            
            for item in qi_list:
                is_covered = item["FRT_ID"] in available_frt_ids
                status = "✅ COUVERT" if is_covered else "❌ ERREUR"
                if is_covered: c_ok += 1
                res_log.append({"Qi": item["Qi"], "Statut": status})
                
            taux = (c_ok / len(qi_list)) * 100
            k1, k2 = st.columns(2)
            k1.metric("Questions", len(qi_list))
            k2.metric("Taux Couverture", f"{taux:.0f}%")
            st.dataframe(pd.DataFrame(res_log), use_container_width=True)

        st.divider()

        st.markdown("#### 2. Test Externe (Nouveau Sujet)")
        if st.button("Générer Sujet Crash Test"):
            all_kernel_frts = list(SMAXIA_KERNEL.keys())
            test_frts = random.sample(all_kernel_frts, 3)
            
            ext_log = []
            c_ok_ext = 0
            available_frt_ids = st.session_state['df_qc']["FRT_ID"].unique()
            
            for frt_id in test_frts:
                qi_txt = generate_smart_qi(frt_id)
                is_covered = frt_id in available_frt_ids
                status = "✅ MATCH" if is_covered else "❌ HORS PÉRIMÈTRE"
                if is_covered: c_ok_ext += 1
                ext_log.append({"Qi (Nouveau)": qi_txt, "Résultat": status})
                
            taux_ext = (c_ok_ext / 3) * 100
            st.metric("Taux Couverture Externe", f"{taux_ext:.0f}%")
            st.dataframe(pd.DataFrame(ext_log), use_container_width=True)

    else:
        st.info("Veuillez lancer l'usine dans l'onglet 1.")
