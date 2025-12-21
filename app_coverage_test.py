import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Kernel V9.1")
st.title("⚛️ SMAXIA - KERNEL V9.1 (Multi-Chapitres Terminale)")

# ==============================================================================
# 🧱 ETAPE 0 : DÉFINITION DE L'ESPACE CANONIQUE (E_c)
# 4 Chapitres Clés de Terminale Spécialité
# ==============================================================================

# Mapping pour savoir quel FRT appartient à quel Chapitre
KERNEL_MAPPING = {
    "FRT_SUITE_01": "SUITES NUMÉRIQUES",
    "FRT_SUITE_02": "SUITES NUMÉRIQUES",
    "FRT_SUITE_03": "SUITES NUMÉRIQUES",
    
    "FRT_FCT_01": "FONCTIONS & DÉRIVATION",
    "FRT_FCT_02": "FONCTIONS & DÉRIVATION",
    "FRT_FCT_03": "FONCTIONS & DÉRIVATION",
    
    "FRT_GEO_01": "GÉOMÉTRIE ESPACE",
    "FRT_GEO_02": "GÉOMÉTRIE ESPACE",
    
    "FRT_PROBA_01": "PROBABILITÉS",
    "FRT_PROBA_02": "PROBABILITÉS"
}

SMAXIA_KERNEL = {
    # --- CHAPITRE : SUITES ---
    "FRT_SUITE_01": {
        "QC_Canonical": "COMMENT Démontrer qu'une suite est géométrique",
        "ARI_Steps": ["Calculer u(n+1)", "Factoriser par u(n)", "Identifier la raison q", "Conclure u(n+1) = q*u(n)"],
        "ARI_Weights": [0.2, 0.3, 0.2, 0.1],
        "Difficulty_Delta": 1.2
    },
    "FRT_SUITE_02": {
        "QC_Canonical": "COMMENT Calculer la limite d'une suite (Théorèmes)",
        "ARI_Steps": ["Identifier les termes dominants", "Factoriser le terme de plus haut degré", "Appliquer les limites usuelles", "Conclure sur la convergence"],
        "ARI_Weights": [0.2, 0.3, 0.3, 0.2],
        "Difficulty_Delta": 1.1
    },
    "FRT_SUITE_03": {
        "QC_Canonical": "COMMENT Démontrer par récurrence une propriété",
        "ARI_Steps": ["Initialisation (P0)", "Hérédité (Supposer Pk vraie)", "Démontrer P(k+1)", "Conclusion"],
        "ARI_Weights": [0.1, 0.2, 0.6, 0.1], # Hérédité lourde
        "Difficulty_Delta": 1.5
    },

    # --- CHAPITRE : FONCTIONS ---
    "FRT_FCT_01": {
        "QC_Canonical": "COMMENT Étudier les variations d'une fonction",
        "ARI_Steps": ["Calculer la dérivée f'(x)", "Étudier le signe de f'(x)", "Dresser le tableau de variations", "Calculer les images aux bornes"],
        "ARI_Weights": [0.3, 0.3, 0.2, 0.1],
        "Difficulty_Delta": 1.3
    },
    "FRT_FCT_02": {
        "QC_Canonical": "COMMENT Montrer qu'une équation admet une solution unique (TVI)",
        "ARI_Steps": ["Vérifier la continuité", "Vérifier la stricte monotonie", "Calculer les images de l'intervalle", "Invoquer le corollaire du TVI"],
        "ARI_Weights": [0.1, 0.2, 0.2, 0.4],
        "Difficulty_Delta": 1.4
    },
    "FRT_FCT_03": {
        "QC_Canonical": "COMMENT Déterminer l'équation de la tangente",
        "ARI_Steps": ["Calculer f(a)", "Calculer f'(a)", "Appliquer y = f'(a)(x-a) + f(a)", "Réduire l'expression"],
        "ARI_Weights": [0.1, 0.2, 0.2, 0.1],
        "Difficulty_Delta": 0.9
    },

    # --- CHAPITRE : GÉOMÉTRIE ---
    "FRT_GEO_01": {
        "QC_Canonical": "COMMENT Démontrer qu'une droite est orthogonale à un plan",
        "ARI_Steps": ["Identifier un vecteur directeur u de la droite", "Identifier deux vecteurs directeurs v1, v2 du plan", "Calculer u.v1 et u.v2", "Vérifier que les produits scalaires sont nuls"],
        "ARI_Weights": [0.1, 0.1, 0.4, 0.2],
        "Difficulty_Delta": 1.3
    },
    "FRT_GEO_02": {
        "QC_Canonical": "COMMENT Déterminer une représentation paramétrique de droite",
        "ARI_Steps": ["Identifier un point A(x,y,z)", "Identifier un vecteur directeur u(a,b,c)", "Écrire le système {x=x0+at...}"],
        "ARI_Weights": [0.2, 0.2, 0.4],
        "Difficulty_Delta": 1.0
    },

    # --- CHAPITRE : PROBABILITÉS ---
    "FRT_PROBA_01": {
        "QC_Canonical": "COMMENT Calculer une probabilité totale (Arbre)",
        "ARI_Steps": ["Définir les événements A et B", "Construire l'arbre pondéré", "Identifier les chemins favorables", "Sommer les probabilités des chemins"],
        "ARI_Weights": [0.1, 0.3, 0.2, 0.2],
        "Difficulty_Delta": 1.1
    },
    "FRT_PROBA_02": {
        "QC_Canonical": "COMMENT Calculer une probabilité avec la Loi Binomiale",
        "ARI_Steps": ["Justifier le schéma de Bernoulli (Succès/Echec)", "Identifier n et p", "Appliquer la formule P(X=k)", "Calculer la valeur numérique"],
        "ARI_Weights": [0.3, 0.1, 0.3, 0.1],
        "Difficulty_Delta": 1.2
    }
}

# --- GÉNÉRATEUR DE QI (Simulation Élèves) ---
QI_TEMPLATES = {
    # SUITES
    "FRT_SUITE_01": ["Montrer que (Un) est géométrique.", "Prouver que Vn est une suite géométrique.", "Justifier le caractère géométrique."],
    "FRT_SUITE_02": ["Déterminer la limite de Un.", "Calculer la limite quand n tend vers l'infini.", "Étudier la convergence."],
    "FRT_SUITE_03": ["Démontrer par récurrence que Un > 0.", "Montrer par récurrence que Un < 5.", "Prouver la propriété P(n) pour tout n."],
    # FONCTIONS
    "FRT_FCT_01": ["Étudier les variations de f.", "Dresser le tableau de variations complet.", "Donner le sens de variation de g."],
    "FRT_FCT_02": ["Montrer que f(x)=0 a une solution unique alpha.", "Démontrer l'existence d'une unique solution.", "Combien l'équation admet-elle de solutions ?"],
    "FRT_FCT_03": ["Donner l'équation de la tangente T.", "Déterminer la tangente au point d'abscisse 1.", "Quelle est l'équation réduite de la tangente ?"],
    # GEO
    "FRT_GEO_01": ["Montrer que (d) est orthogonale au plan (P).", "Prouver que la droite est perpendiculaire au plan.", "Vérifier l'orthogonalité."],
    "FRT_GEO_02": ["Donner une représentation paramétrique de (AB).", "Déterminer le système paramétrique de la droite D."],
    # PROBA
    "FRT_PROBA_01": ["Calculer P(B).", "En utilisant l'arbre, déterminer la probabilité de l'événement.", "Quelle est la probabilité totale ?"],
    "FRT_PROBA_02": ["Calculer P(X=3).", "Quelle est la probabilité d'obtenir exactement 2 succès ?", "Déterminer la probabilité que X soit égal à k."]
}

def generate_smart_qi(frt_id):
    """Génère une Qi polymorphe"""
    if frt_id not in QI_TEMPLATES: return "Question Standard"
    text = random.choice(QI_TEMPLATES[frt_id])
    context = random.choice(["", " sur I", " dans le repère", " pour tout n"])
    return text + context

# ==============================================================================
# ⚙️ MOTEUR KERNEL
# ==============================================================================

def calculate_psi_real(frt_id):
    """CALCUL F1 (Réel)"""
    kernel_data = SMAXIA_KERNEL[frt_id]
    weights = kernel_data["ARI_Weights"]
    delta = kernel_data["Difficulty_Delta"]
    sum_t = sum(weights)
    epsilon = 0.1
    psi_raw = delta * (epsilon + sum_t)**2
    psi_norm = min(psi_raw / 4.0, 1.0)
    return round(psi_norm, 4), sum_t, delta

def ingest_and_process(urls, n_per_url, selected_chapters):
    """USINE avec filtre de Chapitres"""
    sources_log = []
    atoms_db = []
    progress = st.progress(0)
    total_ops = len(urls) * n_per_url if len(urls) > 0 else 1
    counter = 0
    natures = ["BAC", "DST", "CONCOURS"]
    
    # Filtrer les FRT disponibles selon les chapitres sélectionnés
    available_frts = [k for k, v in KERNEL_MAPPING.items() if v in selected_chapters]
    
    if not available_frts:
        return pd.DataFrame(), pd.DataFrame() # Rien à générer
    
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
            
            # On pioche 2 ou 3 structures parmi celles autorisées par le filtre
            nb_exos = random.randint(2, 4)
            active_frts = random.sample(available_frts, k=min(nb_exos, len(available_frts)))
            
            file_qi_list = []
            for frt_id in active_frts:
                qi_text = generate_smart_qi(frt_id)
                atoms_db.append({
                    "ID_Source": file_id,
                    "Année": year,
                    "Qi_Brut": qi_text,
                    "FRT_ID": frt_id,
                    "Fichier": filename,
                    "Chapitre": KERNEL_MAPPING[frt_id] # Ajout du Chapitre pour l'affichage
                })
                file_qi_list.append(qi_text)
            
            content = f"SUJET {filename}\nSOURCE: {url}\n\n" + "\n".join([f"Exo {k+1}: {txt}" for k, txt in enumerate(file_qi_list)])
            sources_log.append({"Fichier": filename, "Nature": nature, "Année": year, "Contenu_Blob": content})
            
    progress.empty()
    return pd.DataFrame(sources_log), pd.DataFrame(atoms_db)

def compute_engine_metrics(df_atoms):
    if df_atoms.empty: return pd.DataFrame()
    
    grouped = df_atoms.groupby("FRT_ID").agg({
        "ID_Source": "count", "Année": "max", "Qi_Brut": list, "Fichier": list, "Chapitre": "first"
    }).reset_index()
    
    qcs = []
    N_total = len(df_atoms)
    current_year = datetime.now().year
    
    for idx, row in grouped.iterrows():
        frt_id = row["FRT_ID"]
        kernel_data = SMAXIA_KERNEL[frt_id]
        psi, sum_t, delta = calculate_psi_real(frt_id)
        
        n_q = row["ID_Source"]
        tau = max((current_year - row["Année"]), 0.5)
        alpha = 5.0
        sigma = 0.05 
        score = (n_q / N_total) * (1 + alpha/tau) * psi * (1-sigma) * 100
        
        evidence = [{"Fichier": row["Fichier"][k], "Qi": row["Qi_Brut"][k]} for k in range(len(row["Qi_Brut"]))]
        
        qcs.append({
            "CHAPITRE": row["Chapitre"],
            "QC_ID": frt_id,
            "QC_INVARIANTE": kernel_data["QC_Canonical"],
            "SCORE_F2": score,
            "n_q": n_q, "N_tot": N_total, "Tau": tau, "Psi": psi, "Sum_T": sum_t, "Delta": delta, "Sigma": sigma,
            "ARI_Steps": kernel_data["ARI_Steps"],
            "EVIDENCE": evidence
        })
        
    return pd.DataFrame(qcs).sort_values(by=["CHAPITRE", "SCORE_F2"], ascending=[True, False])

def check_coverage_structural(df_qc_engine, selected_chapters):
    # On ne vérifie que les chapitres demandés
    target_ids = {k for k, v in KERNEL_MAPPING.items() if v in selected_chapters}
    found_ids = set(df_qc_engine["QC_ID"].unique()) if not df_qc_engine.empty else set()
    
    # Intersection avec ce qu'on attendait
    expected_found = found_ids.intersection(target_ids)
    
    if len(target_ids) == 0: return 0, set()
    
    coverage = len(expected_found) / len(target_ids) * 100
    missing = target_ids - found_ids
    return coverage, missing

# ==============================================================================
# 🖥️ INTERFACE
# ==============================================================================

# SIDEBAR AVEC SÉLECTEURS
with st.sidebar:
    st.header("1. Paramètres Académiques")
    matiere = st.selectbox("Matière", ["MATHS"]) # Pour l'instant 1 seule, extensible
    
    # LISTE DES 4 CHAPITRES CLÉS
    all_chapters = list(set(KERNEL_MAPPING.values()))
    selected_chapters = st.multiselect(
        "Chapitres Cibles",
        options=all_chapters,
        default=all_chapters # Tout coché par défaut
    )
    
    st.info(f"Périmètre E_c : {len(selected_chapters)} Chapitres actifs.")

# TABS
tab_factory, tab_audit = st.tabs(["🏭 USINE (Calcul F1/F2)", "🔬 AUDIT SMAXIA (Structure)"])

# --- TAB 1 : USINE ---
with tab_factory:
    col_input, col_act = st.columns([3, 1])
    with col_input:
        urls_input = st.text_area("Sources", "https://apmep.fr/terminale\nhttps://sujetdebac.fr", height=70)
    with col_act:
        n_sujets = st.number_input("Vol. par URL", 5, 100, 10, step=5)
        btn_run = st.button("LANCER LE KERNEL 🚀", type="primary")

    if btn_run:
        url_list = urls_input.split('\n')
        with st.spinner("Atomisation Structurelle & Calculs..."):
            df_src, df_atoms = ingest_and_process(url_list, n_sujets, selected_chapters)
            df_qc = compute_engine_metrics(df_atoms)
            
            st.session_state['df_src'] = df_src
            st.session_state['df_qc'] = df_qc
            st.session_state['sel_chaps'] = selected_chapters
            st.success("Kernel Exécuté.")
    
    st.divider()
    
    if 'df_qc' in st.session_state:
        c_left, c_right = st.columns([1, 1.5])
        
        with c_left:
            st.markdown(f"### 📥 Sujets ({len(st.session_state['df_src'])})")
            st.dataframe(st.session_state['df_src'][["Fichier", "Nature", "Année"]], use_container_width=True, height=400)
            
            sel_file = st.selectbox("Vérifier fichier", st.session_state['df_src']["Fichier"])
            if not st.session_state['df_src'].empty:
                blob = st.session_state['df_src'][st.session_state['df_src']["Fichier"]==sel_file].iloc[0]["Contenu_Blob"]
                st.download_button("📥 Télécharger .txt", blob, file_name=sel_file)

        with c_right:
            st.markdown(f"### 🧠 QC par Chapitre (Regroupement FRT)")
            
            if not st.session_state['df_qc'].empty:
                # Affichage groupé par Chapitre
                active_chaps = st.session_state['df_qc']['CHAPITRE'].unique()
                
                for chap in active_chaps:
                    st.markdown(f"#### 📘 {chap}")
                    df_view = st.session_state['df_qc'][st.session_state['df_qc']['CHAPITRE'] == chap]
                    
                    for idx, row in df_view.iterrows():
                        with st.container():
                            c1, c2 = st.columns([0.8, 3])
                            c1.code(row['QC_ID'])
                            c2.info(f"**{row['QC_INVARIANTE']}**")
                            
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Score F2", f"{row['SCORE_F2']:.1f}")
                            m2.metric("Psi", f"{row['Psi']:.2f}")
                            m3.metric("n_q", row['n_q'])
                            m4.metric("Tau", f"{row['Tau']:.1f}")
                            
                            with st.expander("🧬 ADN (ARI) & Preuve"):
                                st.markdown("**ARI (Étapes de Résolution) :**")
                                for step in row['ARI_Steps']: st.markdown(f"- `{step}`")
                                st.markdown("---")
                                st.markdown("**Phrases élèves (Qi) :**")
                                st.dataframe(pd.DataFrame(row['EVIDENCE']), hide_index=True)
                            st.divider()
            else:
                st.warning("Aucune donnée générée pour ces chapitres.")

# --- TAB 2 : AUDIT ---
with tab_audit:
    st.subheader("Validation Scientifique Multi-Chapitres")
    
    if 'df_qc' in st.session_state:
        cov, missing = check_coverage_structural(st.session_state['df_qc'], st.session_state.get('sel_chaps', []))
        
        k1, k2 = st.columns(2)
        k1.metric("Couverture E_c (Périmètre Sélectionné)", f"{cov:.0f}%")
        k2.metric("FRT Manquantes", len(missing))
        
        st.markdown("#### Matrice de Couverture $E_c$")
        
        audit_data = []
        for kernel_id, data in SMAXIA_KERNEL.items():
            # On n'affiche que les chapitres demandés
            chap = KERNEL_MAPPING[kernel_id]
            if chap in st.session_state.get('sel_chaps', []):
                is_covered = kernel_id not in missing
                audit_data.append({
                    "Chapitre": chap,
                    "FRT_ID": kernel_id,
                    "QC Canonique": data["QC_Canonical"],
                    "Statut": "✅ COUVERT" if is_covered else "❌ MANQUANT"
                })
            
        def color_audit(val):
            return f'background-color: {"#dcfce7" if val == "✅ COUVERT" else "#fee2e2"}; color: black'

        st.dataframe(pd.DataFrame(audit_data).style.map(color_audit, subset=['Statut']), use_container_width=True)
        
    else:
        st.warning("Lancez le Kernel d'abord.")
