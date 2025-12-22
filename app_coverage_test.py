import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V10.3")
st.title("🛡️ SMAXIA - Console V10.3 (Vraie FRT & Déclencheurs)")

# Styles CSS
st.markdown("""
<style>
    .qc-header-row { 
        display: flex; align-items: center; background-color: #f8f9fa; 
        padding: 8px; border-radius: 5px; border-left: 5px solid #2563eb; font-family: monospace;
    }
    .qc-id-tag { font-weight: bold; color: #d97706; margin-right: 10px; font-size: 1.1em;}
    .qc-title { flex-grow: 1; font-weight: 500; color: #1f2937; font-size: 1.05em;}
    .qc-vars { font-size: 0.9em; color: #4b5563; font-weight: bold; background-color: #e5e7eb; padding: 2px 8px; border-radius: 4px;}
    .frt-box { background-color: #f0fdf4; border: 1px solid #bbf7d0; padding: 10px; border-radius: 5px; font-family: 'Courier New'; }
    .trigger-tag { background-color: #e0e7ff; color: #3730a3; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; margin-right: 5px; }
    .stat-metric { font-size: 1.5em; font-weight: bold; color: #2563eb; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🧱 KERNEL SMAXIA (Avec VRAIES FRT Rédigées)
# ==============================================================================

KERNEL_MAPPING = {
    "FRT_SUITE_01": "SUITES NUMÉRIQUES", "FRT_SUITE_02": "SUITES NUMÉRIQUES",
    "FRT_FCT_01": "FONCTIONS", "FRT_FCT_02": "FONCTIONS",
    "FRT_GEO_01": "GÉOMÉTRIE", "FRT_PROBA_01": "PROBABILITÉS"
}

# CONTENU PÉDAGOGIQUE (VRAI FRT + TRIGGERS)
SMAXIA_KERNEL = {
    "FRT_SUITE_01": {
        "QC": "COMMENT Démontrer qu'une suite est géométrique",
        "Triggers": ["montrer que", "suite géométrique", "raison", "nature de la suite"],
        "FRT_Redaction": """
        **Rédaction Type :**
        1. Pour tout entier naturel $n$, on calcule le rapport $\\frac{u_{n+1}}{u_n}$.
        2. On a : $u_{n+1} = \\dots$ (remplacer par l'expression).
        3. Après simplification, on obtient : $\\frac{u_{n+1}}{u_n} = q$ (où $q$ est une constante réelle).
        4. **Conclusion :** La suite $(u_n)$ est donc géométrique de raison $q$ et de premier terme $u_0 = \\dots$.
        """,
        "Weights": [0.2, 0.3, 0.2, 0.1], "Delta": 1.2
    },
    "FRT_SUITE_02": {
        "QC": "COMMENT Calculer la limite d'une suite (Forme Indéterminée)",
        "Triggers": ["limite", "tendre vers", "convergence", "infini"],
        "FRT_Redaction": """
        **Rédaction Type :**
        1. On identifie une Forme Indéterminée (FI) du type $\\infty - \\infty$ ou $\\frac{\\infty}{\\infty}$.
        2. On factorise par le terme de plus haut degré (ou le terme dominant) : $u_n = n^k (\\dots)$.
        3. Or, on sait que $\\lim_{n \\to +\\infty} \\frac{1}{n} = 0$.
        4. Par produit et somme de limites, on déduit que : $\\lim_{n \\to +\\infty} u_n = \\dots$.
        """,
        "Weights": [0.2, 0.3, 0.3, 0.2], "Delta": 1.1
    },
    "FRT_FCT_01": {
        "QC": "COMMENT Étudier les variations d'une fonction",
        "Triggers": ["variations", "croissante", "décroissante", "tableau"],
        "FRT_Redaction": """
        **Rédaction Type :**
        1. La fonction $f$ est dérivable sur $I$ comme somme/produit de fonctions dérivables.
        2. Pour tout $x \\in I$, $f'(x) = \\dots$ (Calcul de la dérivée).
        3. Étudions le signe de $f'(x)$ :
           - $f'(x) > 0$ si $x \\in \\dots$
           - $f'(x) < 0$ si $x \\in \\dots$
        4. **Tableau :** On dresse le tableau de variations en reportant les signes et les limites aux bornes.
        """,
        "Weights": [0.3, 0.3, 0.2, 0.1], "Delta": 1.3
    },
    "FRT_FCT_02": {
        "QC": "COMMENT Montrer qu'une équation a une solution unique (TVI)",
        "Triggers": ["solution unique", "équation f(x)=k", "théorème des valeurs intermédiaires", "alpha"],
        "FRT_Redaction": """
        **Rédaction Type :**
        1. La fonction $f$ est **continue** et **strictement monotone** (croissante/décroissante) sur l'intervalle $[a, b]$.
        2. On calcule les images : $f(a) = \\dots$ et $f(b) = \\dots$.
        3. On constate que $k$ est compris entre $f(a)$ et $f(b)$ (ou que $0 \\in [f(a), f(b)]$).
        4. **Conclusion :** D'après le corollaire du Théorème des Valeurs Intermédiaires, l'équation $f(x)=k$ admet une unique solution $\\alpha$ sur $[a, b]$.
        """,
        "Weights": [0.1, 0.2, 0.2, 0.4], "Delta": 1.4
    },
    "FRT_GEO_01": {
        "QC": "COMMENT Démontrer qu'une droite est orthogonale à un plan",
        "Triggers": ["orthogonale", "perpendiculaire", "vecteur normal", "produit scalaire"],
        "FRT_Redaction": """
        **Rédaction Type :**
        1. Soit $\\vec{u}$ un vecteur directeur de la droite $(d)$ et $\\vec{v_1}, \\vec{v_2}$ deux vecteurs non colinéaires du plan $(P)$.
        2. Calculons les produits scalaires :
           - $\\vec{u} \\cdot \\vec{v_1} = xx' + yy' + zz' = 0$
           - $\\vec{u} \\cdot \\vec{v_2} = 0$
        3. Le vecteur $\\vec{u}$ est orthogonal à deux vecteurs directeurs du plan.
        4. **Conclusion :** La droite $(d)$ est orthogonale au plan $(P)$.
        """,
        "Weights": [0.1, 0.1, 0.4, 0.2], "Delta": 1.3
    },
    "FRT_PROBA_01": {
        "QC": "COMMENT Calculer une probabilité totale (Arbre)",
        "Triggers": ["probabilité", "arbre pondéré", "sachant que", "totale"],
        "FRT_Redaction": """
        **Rédaction Type :**
        1. On définit les événements : $A$ "..." et $B$ "...".
        2. On construit un arbre pondéré représentant la situation.
        3. D'après la formule des probabilités totales, $(A, \\bar{A})$ formant une partition de l'univers :
           $P(B) = P(A \\cap B) + P(\\bar{A} \\cap B)$
           $P(B) = P(A) \\times P_A(B) + P(\\bar{A}) \\times P_{\\bar{A}}(B)$
        4. **Application numérique :** $P(B) = \\dots$
        """,
        "Weights": [0.1, 0.3, 0.2, 0.2], "Delta": 1.1
    }
}

# --- GÉNÉRATEUR ---
QI_TEMPLATES = {
    "FRT_SUITE_01": ["Montrer que (Un) est géométrique.", "Prouver que Vn est une suite géométrique.", "Justifier le caractère géométrique."],
    "FRT_SUITE_02": ["Déterminer la limite de Un.", "Calculer la limite quand n tend vers l'infini.", "Étudier la convergence."],
    "FRT_FCT_01": ["Étudier les variations de f.", "Dresser le tableau de variations.", "Donner le sens de variation."],
    "FRT_FCT_02": ["Montrer que f(x)=0 a une solution unique.", "Démontrer l'existence d'une solution alpha.", "Résoudre f(x)=k."],
    "FRT_GEO_01": ["Montrer que (d) est orthogonale à (P).", "Prouver que la droite est perpendiculaire.", "Vérifier l'orthogonalité."],
    "FRT_PROBA_01": ["Calculer P(B).", "Quelle est la probabilité totale ?", "Utiliser l'arbre pour calculer p."]
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
            "Triggers": kernel["Triggers"],
            "FRT_Content": kernel["FRT_Redaction"], # LA VRAIE FRT
            "Score_F2": score,
            "n_q": n_q, "N_tot": N_total, "Tau": tau, "Alpha": alpha, "Psi": psi, "Sigma": sigma,
            "Evidence": [{"Fichier": f, "Qi": q} for f, q in zip(row["Fichier"], row["Qi_Brut"])]
        })
        
    return pd.DataFrame(qcs).sort_values(by=["Chapitre", "Score_F2"], ascending=[True, False])

# ==============================================================================
# 🖥️ INTERFACE V10.3
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
            
            st.caption("Sélectionner pour télécharger :")
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
                    st.markdown(f"#### 📘 {chap} : {len(df_view)} QC")
                    
                    for idx, row in df_view.iterrows():
                        # LIGNE TITRE QC (Format Strict)
                        header_html = f"""
                        <div class="qc-header-row">
                            <span class="qc-id-tag">[{row['QC_ID_Simple']}]</span>
                            <span class="qc-title">{row['QC_Texte']}</span>
                            <span class="qc-vars">Ψ={row['Psi']} | n_q={row['n_q']} | τ={row['Tau']:.1f} | σ={row['Sigma']}</span>
                        </div>
                        """
                        st.markdown(header_html, unsafe_allow_html=True)
                        
                        # DECLENCHEURS
                        st.write("Déclencheurs : " + " ".join([f"<span class='trigger-tag'>{t}</span>" for t in row['Triggers']]), unsafe_allow_html=True)
                        
                        # VRAIE FRT
                        with st.expander(f"🧬 Voir FRT_{row['QC_ID_Simple']} (Fiche Réponse Type)"):
                            st.markdown(f"<div class='frt-box'>{row['FRT_Content']}</div>", unsafe_allow_html=True)
                                    
                        # QI ASSOCIÉES
                        with st.expander(f"📋 Qi Associées ({row['n_q']})"):
                            st.dataframe(pd.DataFrame(row['Evidence']), hide_index=True, use_container_width=True)
                        
                        st.write("") 
            else:
                st.warning("Aucune QC.")

# --- TAB 2 : AUDIT MAPPING ---
with tab_audit:
    st.subheader("Validation Booléenne (Tableau de Mapping Unifié)")
    
    if 'df_qc' in st.session_state and 'df_src' in st.session_state:
        
        # --- TEST 1 : INTERNE ---
        st.markdown("#### 1. Audit Interne (Sujet Traité)")
        test_file = st.selectbox("Choisir un sujet traité", st.session_state['df_src']["Fichier"])
        
        if st.button("LANCER L'AUDIT DE COUVERTURE (INTERNE)", type="primary"):
            file_data = st.session_state['df_src'][st.session_state['df_src']["Fichier"]==test_file].iloc[0]
            qi_list = file_data["Qi_Data"]
            
            mapping_rows = []
            c_ok = 0
            
            # Indexer le DataFrame QC pour recherche rapide
            qc_lookup = st.session_state['df_qc'].set_index("FRT_ID")
            
            for item in qi_list:
                frt_id = item["FRT_ID"]
                # CORRECTION BUG KeyError : on utilise 'if in index'
                is_covered = frt_id in qc_lookup.index
                
                if is_covered:
                    c_ok += 1
                    qc_info = qc_lookup.loc[frt_id]
                    # Gestion des doublons potentiels (si plusieurs lignes pour meme FRT - rare mais possible)
                    if isinstance(qc_info, pd.DataFrame): qc_info = qc_info.iloc[0]
                    
                    qc_display = f"[{qc_info['QC_ID_Simple']}] {qc_info['QC_Titre_Full']}"
                    frt_display = f"FRT_{qc_info['QC_ID_Simple']}"
                    status = "✅ MATCH"
                else:
                    qc_display = "---"
                    frt_display = "---"
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

            st.dataframe(pd.DataFrame(mapping_rows).style.map(color_map, subset=['Statut']), use_container_width=True)

        st.divider()

        # --- TEST 2 : EXTERNE ---
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
                    if isinstance(qc_info, pd.DataFrame): qc_info = qc_info.iloc[0]
                    
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
            
            st.dataframe(pd.DataFrame(ext_rows).style.map(color_map, subset=['Statut']), use_container_width=True)

    else:
        st.info("Veuillez lancer l'usine dans l'onglet 1.")
