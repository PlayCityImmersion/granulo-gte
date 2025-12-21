import streamlit as st
import pandas as pd
import pdfplumber
import re
import numpy as np

# --- CONFIGURATION SMAXIA ---
st.set_page_config(layout="wide", page_title="SMAXIA - Moteur Congruence V5")
st.markdown("""
<style>
    .stDataFrame { border: 1px solid #444; }
    .highlight { color: #1E3A8A; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 1. MATRICE DE DÉFINITION QC (CERVEAU SMAXIA) ---
# Structure : QC Invariante = Liste fermée de Déclencheurs + Liste de Mots-Clés
QC_MATRIX = {
    # --- ANALYSE ---
    "ANA_LIM_01": {
        "Chapitre": "ANALYSE - LIMITES",
        "QC_Invariant": "COMMENT Calculer une limite en l'infini",
        "Triggers": ["calculer", "déterminer", "étudier", "en déduire"], # Max 4-5
        "Keywords": ["limite", "tend vers", "infini", "+∞", "-∞", "asymptote"]
    },
    "ANA_PRIM_02": {
        "Chapitre": "ANALYSE - INTÉGRATION",
        "QC_Invariant": "COMMENT Déterminer une primitive d'une fonction",
        "Triggers": ["déterminer", "montrer", "vérifier", "justifier"],
        "Keywords": ["primitive", "intégrale", "f(x)"]
    },
    "ANA_VAR_03": {
        "Chapitre": "ANALYSE - DÉRIVATION",
        "QC_Invariant": "COMMENT Étudier les variations d'une fonction",
        "Triggers": ["étudier", "dresser", "démontrer", "justifier"],
        "Keywords": ["variations", "dérivée", "croissante", "décroissante", "tableau"]
    },
    "ANA_REC_04": {
        "Chapitre": "ANALYSE - SUITES",
        "QC_Invariant": "COMMENT Démontrer une propriété par récurrence",
        "Triggers": ["démontrer", "montrer", "prouver"],
        "Keywords": ["récurrence", "initialisation", "hérédité", "entier naturel"]
    },
    # --- GÉOMÉTRIE ---
    "GEO_POS_01": {
        "Chapitre": "GÉOMÉTRIE ESPACE",
        "QC_Invariant": "COMMENT Caractériser la position relative (Droites/Plans)",
        "Triggers": ["démontrer", "caractériser", "déterminer", "justifier"],
        "Keywords": ["orthogonal", "coplanaires", "sécants", "parallèles", "vecteur normal"]
    }
}

# --- 2. MOTEUR D'EXTRACTION (PDF) ---
def extract_qi_segments(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            extract = page.extract_text()
            if extract: text += extract + "\n"
    text = text.replace('\n', ' ')
    # Atomisation par phrase
    raw_segments = re.split(r'[.;?!]', text)
    return [s.strip() for s in raw_segments if len(s) > 20]

# --- 3. CALCULATEUR SCORE & PREUVE ---
def compute_congruity(qi_text, trigger_found, keyword_found):
    words = re.findall(r'\w+', qi_text.lower())
    clean_words = [w for w in words if len(w) > 2]
    
    # --- VARIABLES DE L'ÉQUATION SMAXIA ---
    
    # 1. n_q (Volume sémantique utile)
    n_q = len(clean_words)
    
    # 2. N_total (Constante de Normalisation Globale)
    # Fixée à 40 (taille moyenne idéale d'une phrase complexe de bac)
    N_total = 40.0 
    
    # 3. Tau_rec (Constante de Récurrence/Calibration)
    # Fixée à 5.0 pour le modèle actuel
    Tau_rec = 5.0
    
    # 4. Alpha (Pertinence Contextuelle)
    # Si le Trigger ET le Keyword sont proches dans la phrase, Alpha augmente
    # Pour simplifier ici : 1.0 si présence, 0.0 sinon
    Alpha = 1.0 
    
    # 5. Psi (Densité d'Information)
    unique_words = set(clean_words)
    Psi = len(unique_words) / n_q if n_q > 0 else 0
    
    # 6. Sigma (Pénalité de Bruit)
    noise_list = ['candidat', 'copie', 'points', 'annexe', 'sujet', 'calculatrice']
    noise_count = sum(1 for w in clean_words if w in noise_list)
    Sigma = noise_count * 0.15
    if Sigma > 0.9: Sigma = 0.9

    # --- ÉQUATION FINALE ---
    # Score = (n_q/N_tot) * [1 + Alpha/Tau] * Psi * (1-Sigma)
    
    term_vol = (n_q / N_total)
    term_ctx = (1 + (Alpha / Tau_rec))
    term_penal = (1 - Sigma)
    
    Score = term_vol * term_ctx * Psi * term_penal * 10 
    
    return {
        "n_q": n_q,
        "N_tot": N_total,
        "Alpha": Alpha,
        "Tau_rec": Tau_rec,
        "Psi": round(Psi, 3),
        "Sigma": round(Sigma, 2),
        "SCORE_FINAL": round(Score, 4)
    }

# --- 4. PIPELINE DE CONGRUENCE ---
def run_smaxia_engine(files):
    results = []
    all_qi = []
    for f in files: all_qi.extend(extract_qi_segments(f))
    
    for qi in all_qi:
        qi_lower = qi.lower()
        matched_qc = False
        
        # On scanne la Matrice SMAXIA
        for code, defs in QC_MATRIX.items():
            
            # A. RECHERCHE DU DÉCLENCHEUR (TRIGGER)
            found_trigger = None
            for trig in defs["Triggers"]:
                # On cherche le mot exact (boundary \b)
                if re.search(rf"\b{trig}\b", qi_lower):
                    found_trigger = trig
                    break
            
            # B. RECHERCHE DU MOT-CLÉ (KEYWORD)
            found_keyword = None
            if found_trigger: # On ne cherche le concept que si l'action est identifiée
                for keyw in defs["Keywords"]:
                    if keyw in qi_lower:
                        found_keyword = keyw
                        break
            
            # C. VALIDATION DU COUPLE (CONGRUENCE)
            if found_trigger and found_keyword:
                # C'est un MATCH ! On calcule la preuve mathématique.
                metrics = compute_congruity(qi, found_trigger, found_keyword)
                
                # On ne garde que les scores pertinents (> 0.5)
                if metrics["SCORE_FINAL"] > 0.5:
                    results.append({
                        "Chapitre": defs["Chapitre"],
                        "QC_Invariant": defs["QC_Invariant"],
                        "Déclencheur (T)": found_trigger.upper(),
                        "Mot-Clé (K)": found_keyword.upper(),
                        "Qi_Source": qi,
                        **metrics # Injection des variables n_q, N_tot, Tau...
                    })
                    matched_qc = True
                    break # Priorité au premier match fort
                    
    return pd.DataFrame(results)

# --- INTERFACE ---
st.title("🛡️ SMAXIA PROD - Audit de Congruence (V5)")
st.markdown("### Preuve d'Alignement : [Trigger] + [Mot-Clé] ➔ [QC] (Validé par l'Équation)")

uploaded_files = st.file_uploader("Injecter PDF Sujets", type=['pdf'], accept_multiple_files=True)

if uploaded_files:
    df = run_smaxia_engine(uploaded_files)
    
    if not df.empty:
        # Tri pour présentation
        chapters = sorted(df['Chapitre'].unique())
        
        for chap in chapters:
            st.markdown("---")
            st.header(f"📘 {chap}")
            
            df_chap = df[df['Chapitre'] == chap]
            unique_qcs = df_chap['QC_Invariant'].unique()
            
            for qc in unique_qcs:
                df_qc = df_chap[df_chap['QC_Invariant'] == qc]
                
                # Header QC
                st.info(f"🗝️ **QC CIBLE :** {qc}")
                
                # TABLEAU DE PREUVE (Variables Visibles)
                st.dataframe(
                    df_qc[[
                        "Déclencheur (T)", "Mot-Clé (K)", # La preuve sémantique
                        "Qi_Source", 
                        "SCORE_FINAL",
                        "n_q", "N_tot", "Tau_rec", "Psi", "Sigma" # La preuve mathématique
                    ]].sort_values(by="SCORE_FINAL", ascending=False),
                    column_config={
                        "Qi_Source": st.column_config.TextColumn("Source (Contexte Élève)", width="large"),
                        "SCORE_FINAL": st.column_config.ProgressColumn("Score", format="%.3f", min_value=0, max_value=3),
                        "Déclencheur (T)": st.column_config.TextColumn("Trigger", width="small"),
                        "Mot-Clé (K)": st.column_config.TextColumn("Concept", width="small"),
                        "N_tot": st.column_config.NumberColumn("N_tot", format="%d"),
                        "Tau_rec": st.column_config.NumberColumn("Tau", format="%.1f"),
                    },
                    use_container_width=True,
                    hide_index=True
                )
    else:
        st.warning("Aucune congruence détectée. Vérifiez que les PDF contiennent bien des couples [Verbe Action] + [Concept Mathématique] définis dans la Matrice.")
