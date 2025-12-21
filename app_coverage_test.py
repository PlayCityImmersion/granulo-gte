import streamlit as st
import pandas as pd
import numpy as np
import random
import time

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Test Couverture Chapitre")
st.title("🛡️ SMAXIA - Protocole de Validation de Couverture (P6)")

# --- 0. SIMULATEUR DE DONNÉES (POUR LE TEST) ---
# Ce dictionnaire simule la richesse sémantique d'un chapitre (ex: Nombres Complexes)
# Il contient les "Atomes de Savoir" (Qi) possibles.
DATABASE_SIMULATION = {
    "COMPLEXES": {
        "ALG": ["Mettre sous forme algébrique", "Déterminer partie réelle et imaginaire", "Calculer z barre"],
        "GEO": ["Déterminer l'ensemble des points M", "Interpréter géométriquement le module", "Montrer que le triangle est rectangle"],
        "EQ": ["Résoudre l'équation dans C", "Déterminer les racines carrées", "Factoriser le polynôme"],
        "EXP": ["Passer sous forme exponentielle", "Utiliser la formule de Moivre", "Calculer la puissance n-ième"]
    }
}

# --- 1. MOTEUR F1 (ATOMISATION) & F2 (SCORING) ---
def atomize_and_score(sujets_type, n_sujets, chapitre):
    """
    PHASE 2 & 3 : Simule l'injection de sujets, l'extraction des Qi et le calcul F2.
    Retourne la liste des QC identifiées (Le 'Cerveau').
    """
    qc_detected = {} # Clé = QC_Name, Val = Score
    
    # Simulation de l'extraction F1
    for i in range(n_sujets):
        # On pioche des concepts aléatoires dans le chapitre pour simuler le contenu des sujets
        themes = list(DATABASE_SIMULATION[chapitre].keys())
        for t in themes:
            if random.random() > 0.3: # 70% de chance qu'un thème soit dans un sujet BAC
                concepts = DATABASE_SIMULATION[chapitre][t]
                qi_raw = random.choice(concepts)
                
                # Construction de la QC (Invariant)
                qc_name = f"COMMENT {qi_raw}..."
                
                # F2 : Scoring (Simulation de fréquence)
                if qc_name in qc_detected:
                    qc_detected[qc_name]["n_q"] += 1
                else:
                    qc_detected[qc_name] = {"n_q": 1, "Triggers": [qi_raw.split()[0].upper()]}

    # F3 : Sélection (On ne garde que ce qui est significatif)
    final_qcs = []
    for name, data in qc_detected.items():
        # Score F2 simplifié pour la simulation
        score = data["n_q"] * 10 # Plus c'est fréquent, plus le score est haut
        final_qcs.append({
            "QC_INVARIANTE": name,
            "SCORE_F2": score,
            "TRIGGERS": data["Triggers"]
        })
        
    return pd.DataFrame(final_qcs).sort_values(by="SCORE_F2", ascending=False)

# --- 2. MOTEUR DE VALIDATION (TEST SUR DST) ---
def run_coverage_test(df_qc_ref, n_dst, chapitre):
    """
    PHASE 4 : Prend des DST spécifiques et vérifie si leurs Qi matchent les QC.
    """
    report_log = []
    total_qi_dst = 0
    covered_qi_dst = 0
    
    # On simule N DST qui portent spécifiquement sur ce chapitre
    for i in range(n_dst):
        dst_name = f"DST_Chapitre_{i+1}"
        
        # Un DST contient 3 à 5 exercices (Qi)
        nb_exos = random.randint(3, 5)
        for _ in range(nb_exos):
            total_qi_dst += 1
            
            # Génération d'une Qi de DST (Peut être une variante rare)
            themes = list(DATABASE_SIMULATION[chapitre].keys())
            t = random.choice(themes)
            qi_dst_text = random.choice(DATABASE_SIMULATION[chapitre][t])
            
            # TEST DE COUVERTURE : Est-ce que cette Qi existe dans nos QC générées ?
            # (Matching sémantique simulé)
            match_found = False
            matching_qc = None
            
            for idx, row in df_qc_ref.iterrows():
                # On nettoie les strings pour comparer
                qc_core = row["QC_INVARIANTE"].replace("COMMENT ", "").replace("...", "")
                if qc_core in qi_dst_text:
                    match_found = True
                    matching_qc = row["QC_INVARIANTE"]
                    break
            
            if match_found:
                covered_qi_dst += 1
                status = "✅ COUVERT"
            else:
                status = "❌ ANGLE MORT (Non couvert)"
                
            report_log.append({
                "DST_Source": dst_name,
                "Qi_Élève (Question posée)": qi_dst_text,
                "Statut": status,
                "QC_Référente (Moteur)": matching_qc if match_found else "AUCUNE"
            })
            
    return pd.DataFrame(report_log), covered_qi_dst, total_qi_dst

# --- INTERFACE ---
# 1. PARAMÈTRES (PHASE 1)
with st.sidebar:
    st.header("1. Paramètres du Scope")
    niveau = st.selectbox("Niveau", ["TERMINALE", "PREMIERE"])
    matiere = st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    chapitre = st.selectbox("Chapitre Cible", ["COMPLEXES", "FONCTIONS", "PROBAS"]) # Pour l'exemple on utilise COMPLEXES

# TABS POUR SÉPARER L'APPRENTISSAGE DU TEST
tab1, tab2 = st.tabs(["🏗️ PHASE 2 & 3 : Injection & Atomisation", "🎯 PHASE 4 : Test Couverture (DST)"])

# --- PHASE 2 & 3 : CONSTRUCTION DU CERVEAU ---
with tab1:
    st.subheader("Injection Massive (BAC, Concours, Annales)")
    st.info("On injecte des sujets globaux pour faire émerger les Invariants (QC).")
    
    n_sujets_train = st.slider("Nombre de Sujets d'Entraînement", 10, 100, 30)
    
    if st.button("LANCER L'ATOMISATION (F1 + F2)", key="btn_train"):
        with st.spinner("Atomisation des sujets en cours..."):
            # Simulation (Si on avait le vrai moteur, on appellerait la fonction réelle ici)
            if chapitre == "COMPLEXES":
                df_qcs = atomize_and_score("BAC", n_sujets_train, "COMPLEXES")
                st.session_state['df_qcs'] = df_qcs # Sauvegarde en mémoire pour l'onglet 2
                
                st.success(f"✅ Moteur Construit ! {len(df_qcs)} QC Invariantes identifiées.")
                st.dataframe(df_qcs, use_container_width=True)
            else:
                st.warning("Pour cette démo, seul le chapitre 'Nombres Complexes' est simulé.")

# --- PHASE 4 : VALIDATION SUR DST ---
with tab2:
    st.subheader("Validation sur Évaluations Spécifiques (DST/Interro)")
    st.markdown("On prend des DST qui portent **uniquement** sur ce chapitre et on vérifie si le moteur a la réponse.")
    
    if 'df_qcs' in st.session_state:
        n_dst_test = st.number_input("Nombre de DST de contrôle", 1, 10, 3)
        
        if st.button("LANCER LE TEST DE COUVERTURE", type="primary"):
            df_log, covered, total = run_coverage_test(st.session_state['df_qcs'], n_dst_test, "COMPLEXES")
            
            # CALCUL DU TAUX
            taux = (covered / total) * 100
            
            # AFFICHAGE DES RÉSULTATS
            c1, c2 = st.columns([1, 3])
            with c1:
                st.metric("Taux de Couverture", f"{taux:.1f}%")
                if taux == 100:
                    st.success("PERFECT MATCH")
                elif taux > 80:
                    st.warning("QUELQUES ANGLES MORTS")
                else:
                    st.error("MODÈLE INSUFFISANT")
            
            with c2:
                st.progress(taux / 100)
            
            st.divider()
            st.markdown("### 🔍 Détail du Mapping (Preuve)")
            
            # Coloration du tableau pour voir les erreurs
            def color_status(val):
                color = '#d4edda' if val == '✅ COUVERT' else '#f8d7da'
                return f'background-color: {color}; color: black'

            st.dataframe(
                df_log.style.map(lambda x: color_status(x) if x in ['✅ COUVERT', '❌ ANGLE MORT (Non couvert)'] else '', subset=['Statut']),
                use_container_width=True
            )
            
            # IDENTIFICATION DES MANQUANTS
            missed = df_log[df_log["Statut"].str.contains("ANGLE MORT")]
            if not missed.empty:
                st.error("🚨 ATTENTION : Les concepts suivants sont dans les DST mais pas dans vos QC :")
                st.write(missed["Qi_Élève (Question posée)"].unique())
                st.caption("👉 Action P6 : Il faut injecter plus de sujets contenant ces notions ou affiner F3.")
            else:
                st.success("🎉 Le moteur couvre 100% des cas rencontrés dans ces DST.")
            
    else:
        st.warning("Veuillez d'abord générer les QC dans l'onglet 'Phase 2 & 3'.")
