import streamlit as st
import pandas as pd
import pdfplumber
import re
import numpy as np

# --- CONFIGURATION SMAXIA ---
st.set_page_config(layout="wide", page_title="SMAXIA - Moteur Intégral V4.5")
st.markdown("""
<style>
    .stDataFrame { border: 1px solid #444; }
    .big-score { color: #1E3A8A; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 1. BIBLIOTHÈQUE D'ABSTRACTION (Le Moteur V4 qui fonctionne) ---
QC_LIBRARY = {
    # ANALYSE
    "ANA_LIM": {
        "pattern": r"(limite.*(infini|\+∞|-\∞)|tend vers.*(infini|\+∞|-\∞))",
        "QC_Invariant": "COMMENT Calculer une limite en l'infini",
        "Chapitre": "ANALYSE - LIMITES"
    },
    "ANA_PRIM": {
        "pattern": r"(primitive|intégrale)",
        "QC_Invariant": "COMMENT Déterminer une primitive d'une fonction",
        "Chapitre": "ANALYSE - INTÉGRATION"
    },
    "ANA_VAR": {
        "pattern": r"(variations|dérivée|croissante|décroissante)",
        "QC_Invariant": "COMMENT Étudier les variations d'une fonction",
        "Chapitre": "ANALYSE - DÉRIVATION"
    },
    "ANA_REC": {
        "pattern": r"(récurrence|initialisation|hérédité)",
        "QC_Invariant": "COMMENT Démontrer une propriété par récurrence",
        "Chapitre": "ANALYSE - SUITES"
    },
    # GÉOMÉTRIE (Retour de la détection large)
    "GEO_ESPACE": {
        "pattern": r"(plan|vecteur normal|orthogonal|coplanaires|sécants|représentation paramétrique)",
        "QC_Invariant": "COMMENT Caractériser la position relative de droites et plans",
        "Chapitre": "GÉOMÉTRIE DANS L'ESPACE"
    },
    # PROBABILITÉS
    "PROBA_LOI": {
        "pattern": r"(loi normale|espérance|écart-type|probabilité)",
        "QC_Invariant": "COMMENT Calculer des probabilités avec une loi continue",
        "Chapitre": "PROBABILITÉS"
    }
}

# --- 2. EXTRACTION ---
def extract_qi_segments(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            extract = page.extract_text()
            if extract: text += extract + "\n"
    text = text.replace('\n', ' ')
    raw_segments = re.split(r'[.;?!]', text)
    return [s.strip() for s in raw_segments if len(s) > 20]

# --- 3. CALCULATEUR SCORE COMPLET (Equation SMAXIA) ---
def compute_full_equation(qi_text, context_keywords):
    # Nettoyage et Tokenization
    all_words = re.findall(r'\w+', qi_text.lower())
    
    # N_total : Nombre TOTAL de mots dans la phrase (Dynamique)
    N_total = len(all_words)
    if N_total == 0: return None
    
    # n_q : Nombre de mots "utiles" (longueur > 2 et pas des stopwords basiques)
    stopwords = ['le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'ou', 'est', 'sont', 'par', 'pour']
    meaningful_words = [w for w in all_words if len(w) > 2 and w not in stopwords]
    n_q = len(meaningful_words)
    
    # Alpha (Pertinence) : Match avec le contexte du chapitre
    matches = sum(1 for w in meaningful_words if w in context_keywords)
    Alpha = matches * 1.0 # Poids simple
    
    # Tau_rec (Constante de Récurrence) - Fixée
    Tau_rec = 5.0
    
    # Psi (Densité Sémantique) : Mots uniques / Mots utiles
    unique_words = set(meaningful_words)
    Psi = len(unique_words) / n_q if n_q > 0 else 0
    
    # Sigma (Pénalité Bruit)
    noise_list = ['candidat', 'copie', 'sujet', 'page', 'points', 'annexe', 'rendu']
    noise_count = sum(1 for w in meaningful_words if w in noise_list)
    Sigma = noise_count * 0.2
    if Sigma > 0.9: Sigma = 0.9

    # --- ÉQUATION SMAXIA ---
    # Score = (n_q / N_total) * [1 + (Alpha / Tau)] * Psi * product(1-Sigma)
    
    term_densite = (n_q / N_total) # Vraie densité sémantique
    term_contexte = (1 + (Alpha / Tau_rec))
    term_penalite = (1 - Sigma)
    
    Score = term_densite * term_contexte * Psi * term_penalite * 10 
    
    return {
        "n_q": n_q,
        "N_tot": N_total,
        "Alpha": Alpha,
        "Tau": Tau_rec,
        "Psi": round(Psi, 3),
        "Sigma": round(Sigma, 2),
        "SCORE_FINAL": round(Score, 4)
    }

# --- 4. PIPELINE PRINCIPAL ---
def process_pipeline(files):
    results = []
    all_qi = []
    for f in files: all_qi.extend(extract_qi_segments(f))
    
    for qi in all_qi:
        qi_lower = qi.lower()
        matched = False
        
        # On scanne la bibliothèque (Méthode V4)
        for key, config in QC_LIBRARY.items():
            if re.search(config["pattern"], qi_lower):
                # Détection réussie !
                
                # On génère les keywords pour Alpha depuis le pattern
                keywords_ctx = config["pattern"].replace('|', ' ').replace('(', '').replace(')', '').split()
                
                # Calcul Complet
                metrics = compute_full_equation(qi, keywords_ctx)
                
                if metrics and metrics["SCORE_FINAL"] > 0.4: # Filtre qualité minimale
                    results.append({
                        "Chapitre": config["Chapitre"],
                        "QC_Invariant": config["QC_Invariant"],
                        "Qi_Source": qi,
                        **metrics # Injection de toutes les variables
                    })
                    matched = True
                    break # Une Qi = Une QC
        
    return pd.DataFrame(results)

# --- INTERFACE ---
st.title("🛡️ SMAXIA PROD - Audit Mathématique Complet")
st.markdown("### Équation : $Score(q) = (n_q / N_{tot}) \\times [1 + \\alpha/\\tau] \\times \\Psi \\times (1 - \\sigma)$")

uploaded_files = st.file_uploader("Injecter PDF Sujets", type=['pdf'], accept_multiple_files=True)

if uploaded_files:
    df = process_pipeline(uploaded_files)
    
    if not df.empty:
        # Tri par Score global
        df = df.sort_values(by="SCORE_FINAL", ascending=False)
        
        chapters = sorted(df['Chapitre'].unique())
        
        for chap in chapters:
            st.markdown("---")
            st.header(f"📘 {chap}")
            
            df_chap = df[df['Chapitre'] == chap]
            unique_qcs = df_chap['QC_Invariant'].unique()
            
            for qc in unique_qcs:
                df_qc = df_chap[df_chap['QC_Invariant'] == qc]
                
                # En-tête QC + Compteur
                st.info(f"🗝️ **{qc}** ({len(df_qc)} Qi liées)")
                
                # TABLEAU COMPLET AVEC TOUTES LES VARIABLES
                st.dataframe(
                    df_qc[[
                        "SCORE_FINAL",
                        "Qi_Source", 
                        "n_q", "N_tot", "Alpha", "Tau", "Psi", "Sigma"
                    ]],
                    column_config={
                        "Qi_Source": st.column_config.TextColumn("Source (Qi)", width="large"),
                        "SCORE_FINAL": st.column_config.ProgressColumn("Score (q)", format="%.3f", min_value=0, max_value=4),
                        "N_tot": st.column_config.NumberColumn("N_tot (Dyn)", format="%d"),
                        "n_q": st.column_config.NumberColumn("n_q", format="%d"),
                        "Alpha": st.column_config.NumberColumn("α", format="%.1f"),
                        "Tau": st.column_config.NumberColumn("τ", format="%.1f"),
                        "Psi": st.column_config.NumberColumn("Ψ", format="%.3f"),
                        "Sigma": st.column_config.NumberColumn("σ", format="%.2f"),
                    },
                    use_container_width=True,
                    hide_index=True
                )
    else:
        st.warning("Aucune donnée détectée. Vérifiez les fichiers.")
