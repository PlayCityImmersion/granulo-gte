import streamlit as st
import pandas as pd
import pdfplumber
import re
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Moteur Invariant P6")
st.markdown("""
<style>
    .qc-header { font-size: 24px; font-weight: bold; color: #1E3A8A; }
    .stDataFrame { border: 1px solid #ccc; }
</style>
""", unsafe_allow_html=True)

# --- 1. BIBLIOTHÈQUE D'ABSTRACTION (QC INVARIANTES) ---
# C'est ici qu'on transforme le spécifique (Qi) en canonique (QC).
# On supprime les variables (A, f(x), lambda) pour ne garder que la compétence.

QC_LIBRARY = {
    # ANALYSE
    "LIMIT_INF": {
        "pattern": r"(limite.*(infini|\+∞|-\∞)|tend vers.*(infini|\+∞|-\∞))",
        "QC_Invariant": "COMMENT Calculer une limite en l'infini",
        "Chapitre": "ANALYSE - LIMITES"
    },
    "PRIMITIVE": {
        "pattern": r"(primitive|intégrale définie)",
        "QC_Invariant": "COMMENT Déterminer une primitive d'une fonction",
        "Chapitre": "ANALYSE - INTÉGRATION"
    },
    "DERIVATION": {
        "pattern": r"(dérivée|variations|croissante|décroissante)",
        "QC_Invariant": "COMMENT Étudier les variations d'une fonction",
        "Chapitre": "ANALYSE - DÉRIVATION"
    },
    "RECURRENCE": {
        "pattern": r"(récurrence|initialisation|hérédité)",
        "QC_Invariant": "COMMENT Démontrer une propriété par récurrence",
        "Chapitre": "ANALYSE - SUITES"
    },
    # GÉOMÉTRIE
    "PLAN_ESPACE": {
        "pattern": r"(plan|vecteur normal|orthogonal|coplanaires)",
        "QC_Invariant": "COMMENT Caractériser la position relative de droites et plans",
        "Chapitre": "GÉOMÉTRIE DANS L'ESPACE"
    },
    # PROBABILITÉS
    "LOI_NORMALE": {
        "pattern": r"(loi normale|espérance|écart-type)",
        "QC_Invariant": "COMMENT Calculer des probabilités avec une loi continue",
        "Chapitre": "PROBABILITÉS"
    }
}

# --- 2. MOTEUR D'EXTRACTION & NORMALISATION ---
def extract_qi_segments(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            extract = page.extract_text()
            if extract: text += extract + "\n"
    
    # Nettoyage des sauts de ligne intempestifs
    text = text.replace('\n', ' ')
    
    # Découpage par instructions (Phrase terminant par . ? ou :)
    raw_segments = re.split(r'[.;?!]', text)
    return [s.strip() for s in raw_segments if len(s) > 20]

# --- 3. CALCULATEUR SCORE SMAXIA (FORMULE EXACTE) ---
def compute_smaxia_score(qi_text, qc_context_keywords):
    # Variables de l'équation
    words = re.findall(r'\w+', qi_text.lower())
    clean_words = [w for w in words if len(w) > 2]
    
    # 1. n_q (Nombre de termes significatifs dans le Qi)
    n_q = len(clean_words)
    
    # 2. N_total (Normalisation locale - fixée pour comparatif)
    N_total = 30.0 
    
    # 3. Alpha (Delta) : Pertinence contextuelle par rapport au Chapitre
    # On regarde si les mots du Qi matchent le contexte
    matches = sum(1 for w in clean_words if w in qc_context_keywords)
    Alpha = matches * 1.0
    
    # 4. Tau_rec (Constante de réglage)
    Tau_rec = 5.0
    
    # 5. Psi_q (Densité sémantique : Mots uniques / Mots totaux)
    unique = set(clean_words)
    Psi_q = len(unique) / n_q if n_q > 0 else 0
    
    # 6. Sigma (Bruit/Pénalité)
    # Mots interdits dans un Qi propre (bruit administratif)
    noise_list = ['candidat', 'copie', 'sujet', 'page', 'points', 'annexe']
    noise_count = sum(1 for w in clean_words if w in noise_list)
    Sigma = noise_count * 0.2
    if Sigma > 0.9: Sigma = 0.9

    # --- L'ÉQUATION SMAXIA ---
    # Score = (n_q / N_total) * [1 + (Alpha / Tau)] * Psi * product(1-Sigma)
    
    term_vol = (n_q / N_total)
    term_ctx = (1 + (Alpha / Tau_rec))
    term_penal = (1 - Sigma)
    
    Score = term_vol * term_ctx * Psi_q * term_penal * 10 # *10 pour lisibilité
    
    return {
        "n_q": n_q,
        "N_tot": N_total,
        "Alpha": Alpha,
        "Tau": Tau_rec,
        "Psi": round(Psi_q, 3),
        "Sigma": round(Sigma, 2),
        "SCORE_FINAL": round(Score, 4)
    }

# --- 4. CLASSIFICATION & ABSTRACTION ---
def process_p6_pipeline(files):
    results = []
    
    # 1. Lecture de tous les fichiers
    all_qi = []
    for f in files:
        all_qi.extend(extract_qi_segments(f))
        
    # 2. Matching QC (Abstraction)
    for qi in all_qi:
        qi_lower = qi.lower()
        matched = False
        
        for key, config in QC_LIBRARY.items():
            if re.search(config["pattern"], qi_lower):
                # QC DÉTECTÉE !
                # On calcule le score pour voir si ce Qi est un bon représentant
                # On génère des mots clés contextuels basés sur le pattern
                ctx_keywords = config["pattern"].replace('|', ' ').replace('(', '').replace(')', '').split()
                
                metrics = compute_smaxia_score(qi, ctx_keywords)
                
                results.append({
                    "Matière": "MATHÉMATIQUES", # Auto-détection à améliorer plus tard
                    "Chapitre": config["Chapitre"],
                    "QC_Invariant": config["QC_Invariant"], # LE VRAI QC SANS VARIABLES
                    "Qi_Source": qi,
                    **metrics # Injection des variables de l'équation
                })
                matched = True
                break # Une Qi appartient à une seule QC prioritaire
        
        if not matched:
            # Rejet (Angle mort ou bruit)
            pass
            
    return pd.DataFrame(results)

# --- INTERFACE ---
st.title("🛡️ SMAXIA PROD - Matrice QC Invariante")
st.markdown("### Mapping : [Matière] > [Chapitre] > [QC Invariante] > [Sources Qi]")

uploaded_files = st.file_uploader("Injecter PDF Sujets", type=['pdf'], accept_multiple_files=True)

if uploaded_files:
    df = process_p6_pipeline(uploaded_files)
    
    if not df.empty:
        # Filtrer les scores trop faibles (Bruit)
        df_valid = df[df['SCORE_FINAL'] > 0.5]
        
        # --- AFFICHAGE HIÉRARCHIQUE ---
        
        # 1. Grouper par CHAPITRE
        chapters = df_valid['Chapitre'].unique()
        
        for chap in sorted(chapters):
            st.divider()
            st.markdown(f"## 📘 CHAPITRE : {chap}")
            
            # 2. Grouper par QC INVARIANTE dans le chapitre
            df_chap = df_valid[df_valid['Chapitre'] == chap]
            qcs = df_chap['QC_Invariant'].unique()
            
            for qc in qcs:
                df_qc = df_chap[df_chap['QC_Invariant'] == qc]
                
                # En-tête de la QC
                st.markdown(f"""
                <div style="background-color:#f0f2f6; padding:10px; border-radius:5px; margin-top:10px;">
                    <span style="font-size:18px; font-weight:bold;">🗝️ {qc}</span>
                    <span style="float:right; color:grey;">{len(df_qc)} Qi liées</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Tableau des variables (Preuve Mathématique)
                st.dataframe(
                    df_qc[[
                        "SCORE_FINAL", 
                        "n_q", "Psi", "Alpha", "Tau", "Sigma", # Les variables de l'équation
                        "Qi_Source"
                    ]].sort_values(by="SCORE_FINAL", ascending=False),
                    column_config={
                        "Qi_Source": st.column_config.TextColumn("Source (Exercice Spécifique)", width="large"),
                        "SCORE_FINAL": st.column_config.ProgressColumn("Pertinence", format="%.2f", min_value=0, max_value=5),
                        "Sigma": st.column_config.NumberColumn("Sigma (Bruit)", format="%.2f"),
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
    else:
        st.warning("Aucune QC identifiée. Les fichiers ne contiennent pas de mots-clés mathématiques reconnus par la bibliothèque SMAXIA actuelle.")
