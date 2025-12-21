import streamlit as st
import pandas as pd
import pdfplumber
import re
import numpy as np

# --- CONFIGURATION SMAXIA ---
st.set_page_config(layout="wide", page_title="SMAXIA - Audit Console P6")

# --- 1. DÉFINITION DES TRIGGERS (LISTE FERMÉE SMAXIA) ---
# Seuls ces 5 déclencheurs existent. Tout le reste est un angle mort.
AUTHORIZED_TRIGGERS = {
    "CALCULER":   {"ID": "T1_RES", "Cat": "RÉSOLUTION", "Poids": 1.2},
    "DÉTERMINER": {"ID": "T1_RES", "Cat": "RÉSOLUTION", "Poids": 1.2},
    "DÉMONTRER":  {"ID": "T2_DEM", "Cat": "DÉMONSTRATION", "Poids": 1.5},
    "MONTRER":    {"ID": "T2_DEM", "Cat": "DÉMONSTRATION", "Poids": 1.5},
    "JUSTIFIER":  {"ID": "T3_ARG", "Cat": "ARGUMENTATION", "Poids": 1.1},
    "INTERPRÉTER":{"ID": "T4_INT", "Cat": "INTERPRÉTATION", "Poids": 1.3},
    "TRACER":     {"ID": "T5_GRA", "Cat": "GRAPHIQUE",      "Poids": 1.0}
}

# --- 2. EXTRACTION (MOTEUR GRANULO) ---
def extract_qi_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            extract = page.extract_text()
            if extract: text += extract + "\n"
    text = text.replace('\n', ' ')
    # Découpage par phrase pour atomisation
    raw_segments = re.split(r'[.;?!]', text)
    return [s.strip() for s in raw_segments if len(s) > 15]

# --- 3. CALCULATEUR DÉTERMINISTE (FORMULE SMAXIA) ---
def compute_smaxia_variables(segment, verb_found):
    # --- A. VARIABLES PRIMAIRES ---
    words = [w for w in re.findall(r'\w+', segment.lower()) if len(w) > 3]
    
    # n_q (Nombre de termes sémantiques dans le Qi)
    n_q = len(words)
    
    # Psi (Potentiel Sémantique - Densité)
    unique_words = set(words)
    Psi = len(unique_words) / n_q if n_q > 0 else 0
    
    # Alpha (Facteur de Contexte / Recouvrement)
    # Simulation: on regarde si des mots clés du chapitre sont présents
    keywords = ['fonction', 'intégrale', 'probabilité', 'suite', 'guerre', 'loi']
    matches = sum(1 for w in words if w in keywords)
    Alpha = matches * 0.5 
    
    # Tau_rec (Constante de récurrence - fixée pour le test)
    Tau_rec = 5.0 
    
    # Sigma (Facteur de Pénalité / Bruit)
    # On pénalise si le texte contient des "mots polluants" (ex: "candidat", "page", "points")
    noise_words = ['candidat', 'points', 'feuille', 'annexe', 'sujet']
    noise_count = sum(1 for w in words if w in noise_words)
    Sigma = noise_count * 0.1 # 10% de pénalité par mot de bruit
    if Sigma > 0.9: Sigma = 0.9 # Plafond

    # --- B. FORMULE FINALE (D'après votre image) ---
    # Score = (Base) * (1 + Alpha/Tau) * Psi * (1 - Sigma)
    # Note: N_total est normalisé à 1 ici pour l'échelle locale
    
    trigger_weight = AUTHORIZED_TRIGGERS[verb_found]["Poids"]
    
    Score_F2 = (n_q / 20) * (1 + (Alpha / Tau_rec)) * Psi * (1 - Sigma) * trigger_weight
    
    return {
        "n_q": n_q,
        "Psi": round(Psi, 3),
        "Alpha": Alpha,
        "Sigma": round(Sigma, 2),
        "Score_F2": round(Score_F2, 4)
    }

# --- 4. PROCESSEUR PRINCIPAL ---
def run_p6_audit(segments):
    audit_data = []
    
    for segment in segments:
        segment_upper = segment.upper()
        detected_trigger = None
        trigger_info = None
        
        # 1. IDENTIFICATION DU TRIGGER (STRICT)
        for verb, info in AUTHORIZED_TRIGGERS.items():
            if verb in segment_upper:
                detected_trigger = verb
                trigger_info = info
                break
        
        # 2. CALCUL SI TRIGGER VALIDE
        if detected_trigger:
            # Nettoyage pour le QC
            qc_text = f"COMMENT {segment.strip()}"
            
            # Appel des variables mathématiques
            vars = compute_smaxia_variables(segment, detected_trigger)
            
            status = "PASS" if vars["Score_F2"] > 0.4 else "FAIL_SCORE" # Seuil de qualité
            
            audit_data.append({
                "Statut": status,
                "ID_Trigger": trigger_info["ID"],
                "Déclencheur": detected_trigger,
                "QC_Générée (Cible)": qc_text,
                "Qi_Source (Mapping)": segment[:60] + "...",
                # --- VARIABLES VISIBLES POUR ANALYSE ---
                "n_q (Vol)": vars["n_q"],
                "Psi (Dens)": vars["Psi"],
                "Alpha (Ctx)": vars["Alpha"],
                "Sigma (Bruit)": vars["Sigma"],
                "SCORE F2": vars["Score_F2"]
            })
        else:
            # REJETÉ (Pas de trigger valide)
            pass 
            
    return pd.DataFrame(audit_data)

# --- INTERFACE ---
st.title("🛡️ SMAXIA PROD - Rapport de Validation P6")
st.markdown("### Contrôle des Variables Sémantiques & Booléennes")

uploaded_files = st.file_uploader("Injecter PDF Sujets", type=['pdf'], accept_multiple_files=True)

if uploaded_files:
    all_segments = []
    for f in uploaded_files:
        all_segments.extend(extract_qi_from_pdf(f))
        
    if all_segments:
        df = run_p6_audit(all_segments)
        
        if not df.empty:
            # SÉPARATION PASS / FAIL
            df_pass = df[df["Statut"] == "PASS"]
            df_fail = df[df["Statut"] == "FAIL_SCORE"]
            
            # --- VUE 1 : LE RAPPORT DE VALIDATION (LES PASS) ---
            st.success(f"✅ {len(df_pass)} QC Validées et Prêtes pour P6")
            
            st.markdown("#### Détail des Variables de Calcul (Preuve de Score)")
            
            # Configuration de l'affichage pour la lisibilité
            st.dataframe(
                df_pass,
                column_config={
                    "Statut": st.column_config.TextColumn("Verdict", width="small"),
                    "ID_Trigger": st.column_config.TextColumn("Ref Trig", width="small"),
                    "SCORE F2": st.column_config.ProgressColumn("Score F2", min_value=0, max_value=2, format="%.4f"),
                    "Sigma (Bruit)": st.column_config.NumberColumn("Sigma (Penalité)", format="%.2f"),
                },
                use_container_width=True,
                hide_index=True
            )
            
            # --- VUE 2 : ANALYSE DES REJETS (FAIL) ---
            if not df_fail.empty:
                st.markdown("---")
                st.error(f"❌ {len(df_fail)} QC Rejetées (Score Insuffisant - Voir Sigma/Psi)")
                with st.expander("Voir les éléments rejetés pour calibration"):
                    st.dataframe(df_fail, use_container_width=True)
            
        else:
            st.warning("Aucun Trigger SMAXIA (T1-T5) détecté dans ces documents.")
