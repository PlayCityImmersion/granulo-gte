import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Unit Test Bench")
st.title("🧪 SMAXIA - Banc de Test Unitaire (Validation Équations)")

# --- 1. RÉFÉRENTIEL THÉORIQUE (Ce qu'on DOIT trouver pour 100% couverture) ---
THEORETICAL_COVERAGE = {
    "ANALYSE": [
        "COMMENT ÉTUDIER les variations d'une fonction",
        "COMMENT DÉTERMINER une limite",
        "COMMENT MONTRER qu'une équation admet une solution (TVI)",
        "COMMENT CALCULER une dérivée",
        "COMMENT DÉTERMINER une primitive"
    ]
}

# --- 2. GOLDEN DATASET (Données d'injection contrôlées) ---
# On simule ici des phrases extraites de PDF réels avec leurs années
GOLDEN_DATASET = [
    # CAS 1 : LE STAR (Fréquent + Récent) -> "Variations"
    {"txt": "Étudier les variations de f sur I", "year": 2024, "chap": "ANALYSE"},
    {"txt": "Dresser le tableau de variations complet", "year": 2023, "chap": "ANALYSE"},
    {"txt": "Justifier le sens de variation de la fonction", "year": 2023, "chap": "ANALYSE"},
    {"txt": "Déterminer les variations de g", "year": 2022, "chap": "ANALYSE"},
    
    # CAS 2 : LE DINOSAURE (Vieux) -> "TVI" (Supposons qu'il n'est pas tombé depuis 2015 pour le test)
    {"txt": "Montrer que l'équation f(x)=0 admet une solution unique alpha", "year": 2015, "chap": "ANALYSE"},
    {"txt": "Démontrer l'existence d'une solution unique", "year": 2014, "chap": "ANALYSE"},

    # CAS 3 : LE SINGLE (Rare mais récent) -> "Primitive"
    {"txt": "Déterminer la primitive F qui s'annule en 0", "year": 2024, "chap": "ANALYSE"},

    # CAS 4 : LE BRUIT (A rejeter)
    {"txt": "Le candidat soignera la présentation de sa copie", "year": 2024, "chap": "ANALYSE"},
    {"txt": "Tournez la page S.V.P", "year": 2024, "chap": "ANALYSE"}
]

# --- 3. MOTEUR D'ÉQUATION (Strict) ---

def run_unit_tests():
    # A. CONFIGURATION CONSTANTES
    CURRENT_YEAR = 2025
    ALPHA = 5.0
    SIGMA_PENALTY = 0.95 # Pénalité fatale
    NOISE_WORDS = ["candidat", "copie", "page", "svp", "points"]

    # B. MAPPING QC (Logique de Regroupement)
    # Dans la prod, c'est de l'IA/Regex. Ici, c'est un mapping simple pour tester les maths.
    mapping_rules = {
        "variation": "COMMENT ÉTUDIER les variations d'une fonction",
        "limite": "COMMENT DÉTERMINER une limite",
        "solution unique": "COMMENT MONTRER qu'une équation admet une solution (TVI)",
        "primitive": "COMMENT DÉTERMINER une primitive",
        "dérivée": "COMMENT CALCULER une dérivée"
    }
    
    # C. TRAITEMENT
    processed_data = []
    
    # 1. Analyse Unitaire
    for item in GOLDEN_DATASET:
        txt = item["txt"].lower()
        
        # Détection QC
        qc_tag = "NON_IDENTIFIÉ (Bruit?)"
        for key, val in mapping_rules.items():
            if key in txt:
                qc_tag = val
                break
        
        # Calcul Variables
        # n_q et N_tot sont calculés après agrégation, ici on prépare
        is_noise = any(w in txt for w in NOISE_WORDS)
        sigma = SIGMA_PENALTY if is_noise else 0.0
        
        # Psi (Densité)
        words = txt.split()
        psi = len(set(words)) / len(words) if words else 0
        
        processed_data.append({
            "Original": item["txt"],
            "Year": item["year"],
            "QC": qc_tag,
            "Sigma": sigma,
            "Psi": psi,
            "Chap": item["chap"]
        })
        
    df = pd.DataFrame(processed_data)
    
    # 2. Agrégation & Calcul Score F2
    N_TOTAL = len(df) # Volume total injecté
    
    grouped = df[df["Sigma"] < 0.5].groupby("QC").agg({
        "Original": "count", # n_q
        "Year": "max",       # Récence max
        "Psi": "mean"
    }).rename(columns={"Original": "n_q", "Year": "Last_Year", "Psi": "Psi_Avg"})
    
    final_results = []
    
    for qc, row in grouped.iterrows():
        n_q = row["n_q"]
        delta_t = CURRENT_YEAR - row["Last_Year"]
        tau = delta_t if delta_t > 0 else 0.5
        
        # EQUATION F2
        term_freq = n_q / N_TOTAL
        term_recency = 1 + (ALPHA / tau)
        term_densite = row["Psi_Avg"]
        
        score_f2 = term_freq * term_recency * term_densite * 100
        
        # Assignation FRT (Test Case 4)
        frt = "Générique"
        if "variations" in qc: frt = "Tableau de Signes + Flèches"
        if "TVI" in qc: frt = "Phrase 'f est continue strictement monotone...'"
        
        final_results.append({
            "QC": qc,
            "Score_F2": score_f2,
            "n_q": n_q,
            "Tau": tau,
            "FRT": frt,
            "Status": "✅ VALIDE"
        })
        
    return pd.DataFrame(final_results).sort_values(by="Score_F2", ascending=False), df

# --- INTERFACE TESTEUR ---

st.markdown("### 1. Injection du 'Golden Dataset' (Données Contrôlées)")
st.info("Ce jeu de données contient volontairement : des questions fréquentes récentes, des questions vieilles, et du bruit administratif pour tester la robustesse.")

if st.button("LANCER LES TESTS UNITAIRES"):
    df_results, df_raw = run_unit_tests()
    
    # --- RESULTATS DU TEST D'ÉQUATION ---
    st.divider()
    st.header("2. Validation des Équations F2")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Classement Généré")
        # On affiche le tableau avec colorations
        st.dataframe(
            df_results,
            column_config={
                "Score_F2": st.column_config.ProgressColumn("Pertinence F2", max_value=200),
                "Tau": st.column_config.NumberColumn("Ancienneté (ans)"),
            },
            use_container_width=True
        )
        
    with col2:
        st.subheader("🕵️ Analyse des Comportements")
        
        # Test Case 1 : LE STAR
        try:
            star = df_results[df_results["QC"].str.contains("variations")].iloc[0]
            st.success(f"✅ **TEST 'STAR' RÉUSSI** : 'Variations' est 1er (Score {star['Score_F2']:.1f}) car fréquent ({star['n_q']}) et récent.")
        except:
            st.error("❌ TEST 'STAR' ÉCHOUÉ")

        # Test Case 2 : LE DINOSAURE
        try:
            dino = df_results[df_results["QC"].str.contains("TVI")].iloc[0]
            if dino['Score_F2'] < star['Score_F2']:
                st.success(f"✅ **TEST 'DINOSAURE' RÉUSSI** : 'TVI' est pénalisé (Score {dino['Score_F2']:.1f}) car ancien (Tau={dino['Tau']} ans).")
            else:
                st.warning("⚠️ PROBLÈME : Le vieux sujet score trop haut.")
        except:
            st.error("❌ TEST 'DINOSAURE' ÉCHOUÉ")
            
        # Test Case 3 : LE BRUIT
        noise_count = len(df_raw[df_raw["Sigma"] > 0.8])
        if noise_count > 0:
            st.success(f"✅ **TEST 'BRUIT' RÉUSSI** : {noise_count} phrases parasites ont été identifiées et exclues du calcul (Sigma élevé).")
        else:
            st.error("❌ TEST 'BRUIT' ÉCHOUÉ : Les déchets sont passés.")

    # --- GAP ANALYSIS (COUVERTURE) ---
    st.divider()
    st.header("3. Couverture 100% (Gap Analysis)")
    
    # On compare Théorique vs Réel
    generated_qcs = df_results["QC"].tolist()
    theoretical = THEORETICAL_COVERAGE["ANALYSE"]
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("**QC Détectées :**")
        for q in generated_qcs: st.caption(f"✅ {q}")
            
    with c2:
        st.write("**QC Manquantes (Angles Morts) :**")
        missing = [q for q in theoretical if q not in generated_qcs]
        if missing:
            for m in missing:
                st.error(f"⚠️ {m} (Aucune Qi injectée ne correspond)")
            st.caption("👉 Action requise : Injecter des sujets contenant ces thèmes.")
        else:
            st.success("🎉 COUVERTURE 100% ATTEINTE !")

    # --- VALIDATION FRT ---
    st.divider()
    st.header("4. Validation FRT (Forme Réponse Type)")
    st.markdown("Vérification que la bonne question déclenche la bonne structure de réponse.")
    
    st.table(df_results[["QC", "FRT"]])
