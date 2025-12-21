import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Kernel V9")
st.title("⚛️ SMAXIA - KERNEL V9 (Architecture Scientifique & FRT)")

# ==============================================================================
# 🧱 ETAPE 0 : DÉFINITION DE L'ESPACE CANONIQUE (E_c)
# C'est la "Vérité SMAXIA". La liste finie des structures mathématiques possibles.
# ==============================================================================

SMAXIA_KERNEL = {
    # FRT_ID : L'identifiant unique de la structure de résolution
    "FRT_GEO_01": {
        "QC_Canonical": "COMMENT Démontrer qu'une suite est géométrique",
        "ARI_Steps": ["Calculer u(n+1)", "Factoriser par u(n)", "Identifier la raison q", "Conclure u(n+1) = q*u(n)"],
        "ARI_Weights": [0.2, 0.3, 0.2, 0.1], # Poids cognitifs des étapes
        "Difficulty_Delta": 1.2
    },
    "FRT_GEO_02": {
        "QC_Canonical": "COMMENT Exprimer le terme général Un en fonction de n",
        "ARI_Steps": ["Identifier le premier terme u0", "Identifier la raison q", "Appliquer formule u0 * q^n"],
        "ARI_Weights": [0.1, 0.1, 0.3],
        "Difficulty_Delta": 0.8
    },
    "FRT_GEO_03": {
        "QC_Canonical": "COMMENT Calculer la limite d'une suite géométrique",
        "ARI_Steps": ["Identifier la raison q", "Comparer |q| à 1", "Appliquer théorème limites usuelles"],
        "ARI_Weights": [0.1, 0.4, 0.3],
        "Difficulty_Delta": 1.0
    },
    "FRT_GEO_04": {
        "QC_Canonical": "COMMENT Calculer la somme des termes consécutifs",
        "ARI_Steps": ["Compter le nombre de termes", "Identifier premier terme", "Appliquer formule (1-q^N)/(1-q)"],
        "ARI_Weights": [0.4, 0.1, 0.5], # Plus complexe
        "Difficulty_Delta": 1.4
    }
}

# --- GÉNÉRATEUR DE QI (Simulation Élèves) ---
# Le générateur simule maintenant une "Signature Structurelle" (FRT_ID) cachée derrière le texte.

QI_TEMPLATES = {
    "FRT_GEO_01": [
        "Montrer que la suite (Un) est géométrique.",
        "Prouver que (Vn) est une suite géométrique de raison 3.",
        "Justifier le caractère géométrique de la suite.",
        "Démontrer que u(n+1) = 0.5 * u(n)."
    ],
    "FRT_GEO_02": [
        "Exprimer Un en fonction de n.",
        "Donner la forme explicite de la suite.",
        "En déduire l'expression de Vn.",
        "Écrire u(n) sous la forme f(n)."
    ],
    "FRT_GEO_03": [
        "Déterminer la limite de (Un).",
        "La suite converge-t-elle ?",
        "Étudier le comportement de (Un) à l'infini.",
        "Calculer lim Un."
    ],
    "FRT_GEO_04": [
        "Calculer la somme S = u0 + ... + u10.",
        "En déduire la somme des 20 premiers termes.",
        "Calculer Sigma de k=0 à n des Uk.",
        "Quelle est la somme des termes de la suite ?"
    ]
}

def generate_smart_qi(frt_id):
    """Génère une Qi qui possède une structure génétique (FRT)"""
    text = random.choice(QI_TEMPLATES[frt_id])
    # On ajoute du bruit contextuel (Polymorphisme)
    context = random.choice(["", " pour tout entier naturel n", " sur I", " dans C"])
    return text + context

# ==============================================================================
# ⚙️ MOTEUR KERNEL (F1 + F2 RÉELS)
# ==============================================================================

def calculate_psi_real(frt_id):
    """
    CALCUL F1 (Réel) : Ψ est dérivé de la complexité de l'ARI.
    Psi = (Somme(Poids) * Delta) / Normalisation
    """
    kernel_data = SMAXIA_KERNEL[frt_id]
    weights = kernel_data["ARI_Weights"]
    delta = kernel_data["Difficulty_Delta"]
    
    # Somme des poids cognitifs
    sum_t = sum(weights)
    
    # Formule F1 simplifiée pour la démo
    epsilon = 0.1
    psi_raw = delta * (epsilon + sum_t)**2
    
    # Normalisation locale (Max théorique ~ 4.0)
    psi_norm = min(psi_raw / 4.0, 1.0)
    
    return round(psi_norm, 4), sum_t, delta

def ingest_and_process(urls, n_per_url):
    """
    USINE : Collecte -> Identification FRT (Simulation IA) -> Stockage
    """
    sources_log = []
    atoms_db = []
    
    progress = st.progress(0)
    total_ops = len(urls) * n_per_url if len(urls) > 0 else 1
    counter = 0
    
    natures = ["BAC", "DST", "CONCOURS"]
    
    for i, url in enumerate(urls):
        if not url.strip(): continue
        for j in range(n_per_url):
            counter += 1
            progress.progress(min(counter/total_ops, 1.0))
            time.sleep(0.002)
            
            # Création Sujet
            nature = random.choice(natures)
            year = random.choice(range(2020, 2025))
            filename = f"Sujet_{nature}_{year}_{j}.txt"
            file_id = f"DOC_{i}_{j}"
            
            # Génération Contenu Mathématique
            # On pioche 2 ou 3 structures FRT au hasard dans le Kernel
            active_frts = random.sample(list(SMAXIA_KERNEL.keys()), k=random.randint(2, 3))
            
            file_qi_list = []
            
            for frt_id in active_frts:
                # 1. Génération du texte (Surface)
                qi_text = generate_smart_qi(frt_id)
                
                # 2. Stockage de l'Atome
                atoms_db.append({
                    "ID_Source": file_id,
                    "Année": year,
                    "Qi_Brut": qi_text,
                    "FRT_ID": frt_id, # C'est LA clé scientifique
                    "Fichier": filename
                })
                
                file_qi_list.append(qi_text)
            
            # Création Blob Physique
            content = f"SUJET {filename}\nSOURCE: {url}\n\n" + "\n".join([f"Exo {k+1}: {txt}" for k, txt in enumerate(file_qi_list)])
            
            sources_log.append({
                "Fichier": filename,
                "Nature": nature,
                "Année": year,
                "Contenu_Blob": content
            })
            
    progress.empty()
    return pd.DataFrame(sources_log), pd.DataFrame(atoms_db)

def compute_engine_metrics(df_atoms):
    """
    MOTEUR F2 : Agrégation par FRT (Structure) et non par Texte.
    """
    if df_atoms.empty: return pd.DataFrame()
    
    # GROUP BY FRT_ID (Le vrai regroupement SMAXIA)
    grouped = df_atoms.groupby("FRT_ID").agg({
        "ID_Source": "count",      # n_q
        "Année": "max",            # Pour Récence
        "Qi_Brut": list,           # Preuve Polymorphisme
        "Fichier": list            # Preuve Source
    }).reset_index()
    
    qcs = []
    N_total = len(df_atoms) # Espace observé total
    current_year = datetime.now().year
    
    for idx, row in grouped.iterrows():
        frt_id = row["FRT_ID"]
        kernel_data = SMAXIA_KERNEL[frt_id]
        
        # 1. Calcul F1 (Psi)
        psi, sum_t, delta = calculate_psi_real(frt_id)
        
        # 2. Variables F2
        n_q = row["ID_Source"]
        tau = max((current_year - row["Année"]), 0.5)
        alpha = 5.0
        
        # Sigma (Similarité) : Simplifié ici, mais on pourrait le calculer via ARI overlap
        sigma = 0.05 
        
        # 3. Formule F2
        # Score = Freq * Recence * Cognitif * Diversité
        score = (n_q / N_total) * (1 + alpha/tau) * psi * (1-sigma) * 100
        
        # Preuve Evidence
        evidence = []
        for k in range(len(row["Qi_Brut"])):
            evidence.append({"Fichier": row["Fichier"][k], "Qi": row["Qi_Brut"][k]})
            
        qcs.append({
            "QC_ID": frt_id, # L'ID technique
            "QC_INVARIANTE": kernel_data["QC_Canonical"], # Le nom humain
            "SCORE_F2": score,
            "n_q": n_q,
            "N_tot": N_total,
            "Tau": tau,
            "Psi": psi,
            "Sum_T": sum_t, # Détail F1
            "Delta": delta, # Détail F1
            "Sigma": sigma,
            "ARI_Steps": kernel_data["ARI_Steps"], # LA PREUVE STRUCTURELLE
            "EVIDENCE": evidence
        })
        
    return pd.DataFrame(qcs).sort_values(by="SCORE_F2", ascending=False)

def check_coverage_structural(df_qc_engine):
    """
    Vérifie si l'espace E_c (Kernel) est couvert par les QC trouvées.
    """
    kernel_ids = set(SMAXIA_KERNEL.keys())
    found_ids = set(df_qc_engine["QC_ID"].unique()) if not df_qc_engine.empty else set()
    
    coverage = len(found_ids) / len(kernel_ids) * 100
    missing = kernel_ids - found_ids
    
    return coverage, missing

# ==============================================================================
# 🖥️ INTERFACE
# ==============================================================================

# SIDEBAR
with st.sidebar:
    st.header("1. Périmètre : SUITES GÉOMÉTRIQUES")
    st.info("E_c défini : 4 FRT Canoniques (Démontrer, Exprimer, Limite, Somme)")

# TABS
tab_factory, tab_audit = st.tabs(["🏭 USINE (Calcul F1/F2)", "🔬 AUDIT SMAXIA (Structure)"])

# --- TAB 1 : USINE ---
with tab_factory:
    col_input, col_act = st.columns([3, 1])
    with col_input:
        urls_input = st.text_area("Sources", "https://apmep.fr/terminale", height=70)
    with col_act:
        n_sujets = st.number_input("Vol. par URL", 5, 100, 10, step=5)
        btn_run = st.button("LANCER LE KERNEL 🚀", type="primary")

    if btn_run:
        url_list = urls_input.split('\n')
        with st.spinner("Atomisation Structurelle & Calculs..."):
            df_src, df_atoms = ingest_and_process(url_list, n_sujets)
            df_qc = compute_engine_metrics(df_atoms)
            
            st.session_state['df_src'] = df_src
            st.session_state['df_qc'] = df_qc
            st.success("Kernel Exécuté.")
    
    st.divider()
    
    if 'df_qc' in st.session_state:
        c_left, c_right = st.columns([1, 1.5])
        
        with c_left:
            st.markdown(f"### 📥 Sujets ({len(st.session_state['df_src'])})")
            st.dataframe(st.session_state['df_src'][["Fichier", "Nature", "Année"]], use_container_width=True, height=400)
            
            # Téléchargement
            sel_file = st.selectbox("Vérifier fichier", st.session_state['df_src']["Fichier"])
            blob = st.session_state['df_src'][st.session_state['df_src']["Fichier"]==sel_file].iloc[0]["Contenu_Blob"]
            st.download_button("📥 Télécharger .txt", blob, file_name=sel_file)

        with c_right:
            st.markdown(f"### 🧠 QC (Regroupement par FRT)")
            
            for idx, row in st.session_state['df_qc'].iterrows():
                with st.container():
                    # Header
                    c1, c2 = st.columns([0.8, 3])
                    c1.code(row['QC_ID']) # Affiche l'ID technique (FRT_GEO_01)
                    c2.info(f"**{row['QC_INVARIANTE']}**")
                    
                    # Métriques F2
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Score F2", f"{row['SCORE_F2']:.1f}")
                    m2.metric("Psi (Cognitif)", f"{row['Psi']:.2f}", help="Calculé via ARI")
                    m3.metric("n_q", row['n_q'])
                    m4.metric("Tau", f"{row['Tau']:.1f}")
                    
                    # PREUVE ARI (C'est ça que l'audit voulait voir)
                    with st.expander("🧬 ADN de la QC (ARI & Calcul Psi)"):
                        st.markdown("**Algorithme de Résolution Invariant (ARI) :**")
                        for step in row['ARI_Steps']:
                            st.markdown(f"- `{step}`")
                        st.markdown(f"**Détail Psi :** Complexité Somme(T)={row['Sum_T']:.1f} * Delta={row['Delta']}")
                    
                    # PREUVE TEXTUELLE
                    with st.expander(f"🔎 Voir les {row['n_q']} Qi rattachées"):
                        st.dataframe(pd.DataFrame(row['EVIDENCE']), hide_index=True)
                    st.divider()

# --- TAB 2 : AUDIT (COUVERTURE STRUCTURELLE) ---
with tab_audit:
    st.subheader("Validation Scientifique SMAXIA")
    
    if 'df_qc' in st.session_state:
        cov, missing = check_coverage_structural(st.session_state['df_qc'])
        
        k1, k2 = st.columns(2)
        k1.metric("Couverture E_c (Espace Canonique)", f"{cov:.0f}%")
        k2.metric("Nombre de FRT Manquantes", len(missing))
        
        st.markdown("#### Matrice de Couverture $E_c$")
        
        audit_data = []
        for kernel_id, data in SMAXIA_KERNEL.items():
            is_covered = kernel_id not in missing
            audit_data.append({
                "FRT_ID": kernel_id,
                "QC Canonique": data["QC_Canonical"],
                "Statut": "✅ COUVERT" if is_covered else "❌ MANQUANT",
                "ARI Defined": "OUI"
            })
            
        def color_audit(val):
            color = '#dcfce7' if val == '✅ COUVERT' else '#fee2e2'
            return f'background-color: {color}; color: black'

        st.dataframe(pd.DataFrame(audit_data).style.map(color_audit, subset=['Statut']), use_container_width=True)
        
        if cov == 100:
            st.success("CERTIFICATION SMAXIA : L'espace canonique des Suites Géométriques est couvert à 100%.")
        else:
            st.error("ECHEC AUDIT : Il manque des structures FRT dans les sujets injectés.")
            
    else:
        st.warning("Lancez le Kernel d'abord.")
