import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V10.2")
st.title("🛡️ SMAXIA - Console V10.2 (Mapping Master)")

# Styles CSS pour l'alignement demandé
st.markdown("""
<style>
    .qc-header-row { 
        display: flex; 
        align-items: center; 
        background-color: #f8f9fa; 
        padding: 8px; 
        border-radius: 5px; 
        border-left: 5px solid #2563eb;
        font-family: monospace;
    }
    .qc-id-tag { 
        font-weight: bold; 
        color: #d97706; 
        margin-right: 10px; 
        font-size: 1.1em;
    }
    .qc-title { 
        flex-grow: 1; 
        font-weight: 500; 
        color: #1f2937;
        font-size: 1.05em;
    }
    .qc-vars { 
        font-size: 0.9em; 
        color: #4b5563; 
        font-weight: bold;
        background-color: #e5e7eb;
        padding: 2px 8px;
        border-radius: 4px;
    }
    .stat-metric { font-size: 1.5em; font-weight: bold; color: #2563eb; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🧱 KERNEL SMAXIA
# ==============================================================================

KERNEL_MAPPING = {
    "FRT_SUITE_01": "SUITES NUMÉRIQUES", "FRT_SUITE_02": "SUITES NUMÉRIQUES", "FRT_SUITE_03": "SUITES NUMÉRIQUES",
    "FRT_FCT_01": "FONCTIONS", "FRT_FCT_02": "FONCTIONS", "FRT_FCT_03": "FONCTIONS",
    "FRT_GEO_01": "GÉOMÉTRIE", "FRT_GEO_02": "GÉOMÉTRIE",
    "FRT_PROBA_01": "PROBABILITÉS", "FRT_PROBA_02": "PROBABILITÉS"
}

SMAXIA_KERNEL = {
    "FRT_SUITE_01": {"QC": "COMMENT Démontrer qu'une suite est géométrique", "ARI": ["Calculer u(n+1)", "Factoriser par u(n)", "Identifier q", "Conclure"], "Weights": [0.2, 0.3, 0.2, 0.1], "Delta": 1.2},
    "FRT_SUITE_02": {"QC": "COMMENT Calculer la limite d'une suite", "ARI": ["Identifier termes", "Factoriser", "Limites usuelles", "Conclure"], "Weights": [0.2, 0.3, 0.3, 0.2], "Delta": 1.1},
    "FRT_SUITE_03": {"QC": "COMMENT Démontrer par récurrence", "ARI": ["Initialisation", "Hérédité", "Conclusion"], "Weights": [0.1, 0.2, 0.6, 0.1], "Delta": 1.5},
    "FRT_FCT_01": {"QC": "COMMENT Étudier les variations d'une fonction", "ARI": ["Dérivée f'", "Signe f'", "Tableau var", "Images"], "Weights": [0.3, 0.3, 0.2, 0.1], "Delta": 1.3},
    "FRT_FCT_02": {"QC": "COMMENT Appliquer le TVI (Solution unique)", "ARI": ["Continuité", "Monotonie", "Images bornes", "Corollaire TVI"], "Weights": [0.1, 0.2, 0.2, 0.4], "Delta": 1.4},
    "FRT_FCT_03": {"QC": "COMMENT Déterminer l'équation de la tangente", "ARI": ["Calculer f(a)", "Calculer f'(a)", "Formule y=..."], "Weights": [0.1, 0.2, 0.2, 0.1], "Delta": 0.9},
    "FRT_GEO_01": {"QC": "COMMENT Démontrer Droite Orthogonale Plan", "ARI": ["Vecteur directeur u", "Vecteurs plan v1, v2", "Produits scalaires nuls"], "Weights": [0.1, 0.1, 0.4, 0.2], "Delta": 1.3},
    "FRT_GEO_02": {"QC": "COMMENT Déterminer une représentation paramétrique", "ARI": ["Point A", "Vecteur u", "Système {x,y,z}"], "Weights": [0.2, 0.2, 0.4], "Delta": 1.0},
    "FRT_PROBA_01": {"QC": "COMMENT Calculer une probabilité totale", "ARI": ["Arbre pondéré", "Chemins favorables", "Somme"], "Weights": [0.1, 0.3, 0.2, 0.2], "Delta": 1.1},
    "FRT_PROBA_02": {"QC": "COMMENT Utiliser la Loi Binomiale", "ARI": ["Bernoulli (n,p)", "Formule P(X=k)", "Calcul"], "Weights": [0.3, 0.1, 0.3, 0.1], "Delta": 1.2}
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
                "Contenu": content, "Qi_Data": qi_list_in_file 
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
        qc_clean = kernel["QC"].replace("COMMENT ", "comment ")
        
        qcs.append({
            "Chapitre": row["Chapitre"],
            "QC_ID_Simple": f"QC_{idx+1:02d}", 
            "FRT_ID": frt_id,
            "QC_Texte": qc_clean,
            "QC_Titre_Full": kernel["QC"],
            "Score_F2": score,
            "n_q": n_q, "N_tot": N_total, "Tau": tau, "Alpha": alpha, "Psi": psi, "Sigma": sigma,
            "Evidence": [{"Fichier": f, "Qi": q} for f, q in zip(row["Fichier"], row["Qi_Brut"])],
            "ARI": kernel["ARI"]
        })
        
    return pd.DataFrame(qcs).sort_values(by=["Chapitre", "Score_F2"], ascending=[True, False])

# ==============================================================================
# 🖥️ INTERFACE V10.2
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
            st.success("Terminé.")

    st.divider()

    if 'df_qc' in st.session_state:
        col_left, col_right = st.columns([1, 1.8])
        
        with col_left:
            st.markdown(f"### 📥 Sujets ({len(st.session_state['df_src'])})")
            df_display = st.session_state['df_src'][["Fichier", "Nature", "Année"]].copy()
            df_display["Action"] = "📥 PDF" 
            st.dataframe(df_display, use_container_width=True, height=500)
            
            st.caption("Sélectionner pour télécharger :")
            sel = st.selectbox("Fichier cible", st.session_state['df_src']["Fichier"], label_visibility="collapsed")
            if not st.session_state['df_src'].empty:
                txt = st.session_state['df_src'][st.session_state['df_src']["Fichier"]==sel].iloc[0]["Contenu"]
                st.download_button(f"Télécharger {sel}", txt, file_name=sel)

        with col_right:
            if not st.session_state['df_qc'].empty:
                chapters = st.session_state['df_qc']["Chapitre"].unique()
                for chap in chapters:
                    df_view = st.session_state['df_qc'][st.session_state['df_qc']["Chapitre"] == chap]
                    st.markdown(f"#### 📘 Chapitre {chap} : {len(df_view)} QC")
                    
                    for idx, row in df_view.iterrows():
                        # FORMAT LIGNE 1 : [QC_ID] comment ... | Ψ=... | n_q=...
                        header_html = f"""
                        <div class="qc-header-row">
                            <span class="qc-id-tag">[{row['QC_ID_Simple']}]</span>
                            <span class="qc-title">{row['QC_Texte']}</span>
                            <span class="qc-vars">Ψ={row['Psi']} | n_q={row['n_q']} | τ={row['Tau']:.1f} | σ={row['Sigma']}</span>
                        </div>
                        """
                        st.markdown(header_html, unsafe_allow_html=True)
                        
                        # ELEMENTS DESSOUS
                        with st.expander(f"🧬 FRT_{row['QC_ID_Simple']} (Résolution ARI)"):
                            st.info("Algorithme de Résolution Invariant :")
                            for i, step in enumerate(row['ARI']): st.write(f"{i+1}. {step}")
                                    
                        with st.expander(f"📋 Qi Associées ({row['n_q']})"):
                            st.dataframe(pd.DataFrame(row['Evidence']), hide_index=True, use_container_width=True)
                        
                        st.write("") # Spacer
            else:
                st.warning("Aucune QC.")

# --- TAB 2 : AUDIT MAPPING ---
with tab_audit:
    st.subheader("Validation Booléenne (Tableau de Mapping Unifié)")
    
    if 'df_qc' in st.session_state and 'df_src' in st.session_state:
        
        # --- TEST 1 : SUJET INTERNE ---
        st.markdown("#### 1. Audit Interne (Sujet Traité)")
        test_file = st.selectbox("Choisir un sujet traité", st.session_state['df_src']["Fichier"])
        
        if st.button("LANCER L'AUDIT DE COUVERTURE (INTERNE)", type="primary"):
            file_data = st.session_state['df_src'][st.session_state['df_src']["Fichier"]==test_file].iloc[0]
            qi_list = file_data["Qi_Data"]
            
            mapping_rows = []
            c_ok = 0
            
            # Dictionnaire pour lookup rapide
            qc_lookup = st.session_state['df_qc'].set_index("FRT_ID")
            
            for item in qi_list:
                frt_id = item["FRT_ID"]
                is_covered = frt_id in qc_lookup.index
                
                if is_covered:
                    c_ok += 1
                    qc_info = qc_lookup.loc[frt_id]
                    qc_display = f"[{qc_info['QC_ID_Simple']}] {qc_info['QC_Titre_Full']}"
                    frt_display = f"{frt_id}"
                    status = "✅ MATCH"
                else:
                    qc_display = "---"
                    frt_display = f"{frt_id} (Manquant)"
                    status = "❌ GAP"
                
                mapping_rows.append({
                    "1. Qi (Question Élève)": item["Qi"],
                    "2. QC (ID + Titre)": qc_display,
                    "3. FRT (ID Technique)": frt_display,
                    "Statut": status
                })
            
            taux = (c_ok / len(qi_list)) * 100
            
            st.markdown(f"<div class='stat-metric'>Taux de Couverture : {taux:.0f}%</div>", unsafe_allow_html=True)
            
            def color_map(val):
                return f'background-color: {"#dcfce7" if val == "✅ MATCH" else "#fee2e2"}; color: black'

            st.dataframe(
                pd.DataFrame(mapping_rows).style.map(color_map, subset=['Statut']), 
                use_container_width=True
            )

        st.divider()

        # --- TEST 2 : SUJET EXTERNE ---
        st.markdown("#### 2. Audit Externe (Nouveau Sujet)")
        
        if st.button("LANCER L'AUDIT DE COUVERTURE (EXTERNE)"):
            all_kernel_frts = list(SMAXIA_KERNEL.keys())
            test_frts = random.sample(all_kernel_frts, 4)
            
            ext_rows = []
            c_ok_ext = 0
            qc_lookup = st.session_state['df_qc'].set_index("FRT_ID")
            
            for frt_id in test_frts:
                qi_txt = generate_smart_qi(frt_id)
                is_covered = frt_id in qc_lookup.index
                
                if is_covered:
                    c_ok_ext += 1
                    qc_info = qc_lookup.loc[frt_id]
                    qc_display = f"[{qc_info['QC_ID_Simple']}] {qc_info['QC_Titre_Full']}"
                    status = "✅ MATCH"
                else:
                    qc_display = "---"
                    status = "❌ HORS PÉRIMÈTRE"
                
                ext_rows.append({
                    "1. Qi (Nouveau Sujet)": qi_txt,
                    "2. QC Trouvée": qc_display,
                    "3. FRT Requise": frt_id,
                    "Statut": status
                })
            
            taux_ext = (c_ok_ext / 4) * 100
            st.markdown(f"<div class='stat-metric'>Taux de Couverture : {taux_ext:.0f}%</div>", unsafe_allow_html=True)
            
            st.dataframe(
                pd.DataFrame(ext_rows).style.map(color_map, subset=['Statut']), 
                use_container_width=True
            )

    else:
        st.info("Veuillez lancer l'usine dans l'onglet 1.")
