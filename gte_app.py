import streamlit as st
import pandas as pd
import pdfplumber
import re
import numpy as np
from collections import Counter

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Moteur Audit V6 (Maths Réelles)")
st.markdown("""
<style>
    .stDataFrame { border: 1px solid #444; }
    .metric-box { border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 1. DÉFINITION DES QC ET PATTERNS ---
QC_LIBRARY = {
    "ANA_LIM": {
        "QC_Invariant": "COMMENT Calculer une limite en l'infini",
        "pattern": r"(limite.*(infini|\+∞|-\∞)|tend vers.*(infini|\+∞|-\∞))",
        "keywords": ["limite", "tend", "infini", "asymptote"]
    },
    "ANA_PRIM": {
        "QC_Invariant": "COMMENT Déterminer une primitive",
        "pattern": r"(primitive|intégrale)",
        "keywords": ["primitive", "intégrale", "fonction", "dérivée"]
    },
    "ANA_VAR": {
        "QC_Invariant": "COMMENT Étudier les variations",
        "pattern": r"(variations|dérivée|croissante|décroissante|tableau)",
        "keywords": ["variations", "signe", "dérivée", "tableau"]
    },
    "GEO_ESPACE": {
        "QC_Invariant": "COMMENT Caractériser la position relative (Espace)",
        "pattern": r"(plan|vecteur normal|orthogonal|coplanaires|sécants|représentation paramétrique)",
        "keywords": ["plan", "droite", "vecteur", "orthogonal", "normal", "repère"]
    }
}

# --- 2. FONCTIONS MATHÉMATIQUES SMAXIA (STRICTES) ---

def get_word_counts(text):
    # Tokenization stricte pour compter N_total et n_q
    words = re.findall(r'\w+', text.lower())
    
    # N_total = TOUS les mots (y compris "le", "de", "et")
    N_total = len(words)
    
    stopwords = ['le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'ou', 'est', 'sont', 'par', 'pour', 'que', 'qui', 'dans', 'sur']
    # n_q = Mots porteurs de sens (> 2 lettres, pas stopword)
    useful_words = [w for w in words if len(w) > 2 and w not in stopwords]
    n_q = len(useful_words)
    
    return N_total, n_q, useful_words

def compute_sigma(words):
    # Pénalité de bruit (Bruit = Mots administratifs)
    noise_list = ['candidat', 'copie', 'sujet', 'page', 'points', 'annexe', 'rendu', 'exercice']
    noise_count = sum(1 for w in words if w in noise_list)
    # Formule Sigma : 0.15 par mot de bruit, max 0.9
    sigma = min(noise_count * 0.15, 0.9)
    return sigma

def compute_equation_smaxia(text, keywords_ctx, tau_global):
    # 1. Variables de base
    N_total, n_q, useful_words = get_word_counts(text)
    
    if N_total == 0: return None
    
    # 2. Alpha (Pertinence contextuelle)
    # Combien de mots utiles sont des mots-clés du chapitre ?
    matches = sum(1 for w in useful_words if w in keywords_ctx)
    alpha = matches # Valeur brute
    
    # 3. Psi (Densité sémantique)
    # Ratio : Mots uniques / Mots utiles
    unique_words = set(useful_words)
    psi = len(unique_words) / n_q if n_q > 0 else 0
    
    # 4. Sigma (Bruit)
    sigma = compute_sigma(useful_words)
    
    # 5. ÉQUATION SMAXIA
    # Score = (n_q / N_total) * [1 + (Alpha / Tau)] * Psi * (1 - Sigma)
    
    # Sécurité div par 0 pour Tau
    tau_safe = tau_global if tau_global > 0 else 1.0
    
    term_densite = n_q / N_total
    term_contexte = 1 + (alpha / tau_safe)
    term_bruit = 1 - sigma
    
    raw_score = term_densite * term_contexte * psi * term_bruit
    
    # Mise à l'échelle pour affichage (x10)
    final_score = raw_score * 10
    
    return {
        "N_tot": N_total,
        "n_q": n_q,
        "Alpha": alpha,
        "Tau": tau_safe,
        "Psi": round(psi, 3),
        "Sigma": round(sigma, 2),
        "SCORE": round(final_score, 4)
    }

# --- 3. PIPELINE D'ANALYSE ---

def run_analysis(files):
    # ÉTAPE 1 : EXTRACTION ET CALCUL DE TAU (GLOBAL)
    all_segments = []
    global_recurrence = {k: 0 for k in QC_LIBRARY.keys()}
    
    # Lecture complète pour statistique globale
    for f in files:
        text = ""
        with pdfplumber.open(f) as pdf:
            for page in pdf.pages:
                extract = page.extract_text()
                if extract: text += extract + "\n"
        text = text.replace('\n', ' ')
        raw_segs = [s.strip() for s in re.split(r'[.;?!]', text) if len(s) > 20]
        
        for seg in raw_segs:
            # On cherche à quelle QC appartient ce segment pour incrémenter Tau
            for code, lib in QC_LIBRARY.items():
                if re.search(lib['pattern'], seg, re.IGNORECASE):
                    global_recurrence[code] += 1
                    all_segments.append({
                        "Code": code,
                        "Qi": seg,
                        "Source_File": f.name
                    })
                    break # Un segment = Une QC unique
    
    # ÉTAPE 2 : CALCUL DES SCORES AVEC LE VRAI TAU
    results = []
    for item in all_segments:
        code = item["Code"]
        lib = QC_LIBRARY[code]
        
        # Le Tau est maintenant la vraie fréquence dans le corpus injecté
        real_tau = global_recurrence[code]
        
        metrics = compute_equation_smaxia(item["Qi"], lib["keywords"], real_tau)
        
        if metrics:
            results.append({
                "QC_Invariant": lib["QC_Invariant"],
                "Qi_Source": item["Qi"],
                **metrics # Injection des résultats
            })
            
    return pd.DataFrame(results)

# --- INTERFACE ---
st.title("🛡️ SMAXIA V6 - La Vérité Mathématique")

# --- BARRE LATÉRALE : CONTRE-EXPERTISE ---
with st.sidebar:
    st.header("🧮 Outil de Vérification")
    st.info("Collez une Qi ici pour vérifier manuellement les variables.")
    test_txt = st.text_area("Phrase à tester", height=100)
    test_tau = st.number_input("Tau supposé (Récurrence)", value=5, min_value=1)
    
    if test_txt:
        # On utilise le moteur 'ANA_LIM' par défaut pour tester les maths
        debug_res = compute_equation_smaxia(test_txt, ["limite", "infini"], test_tau)
        st.write("---")
        st.markdown(f"**N_total (Mots totaux):** {debug_res['N_tot']}")
        st.markdown(f"**n_q (Mots utiles):** {debug_res['n_q']}")
        st.markdown(f"**Psi (Densité):** {debug_res['Psi']}")
        st.markdown(f"**Sigma (Bruit):** {debug_res['Sigma']}")
        st.markdown(f"### SCORE: {debug_res['SCORE']}")

# --- ZONE PRINCIPALE ---
st.write("Injectez vos sujets. Tau ($\tau$) sera calculé dynamiquement selon la fréquence d'apparition dans VOS fichiers.")

uploaded_files = st.file_uploader("PDF Sujets", type=['pdf'], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Calcul de la récurrence globale et scoring..."):
        df = run_analysis(uploaded_files)
    
    if not df.empty:
        # Affichage groupé par QC
        unique_qcs = df['QC_Invariant'].unique()
        
        for qc in unique_qcs:
            st.markdown("---")
            df_qc = df[df['QC_Invariant'] == qc].sort_values(by="SCORE", ascending=False)
            
            # On récupère le Tau réel utilisé pour ce groupe
            tau_real = df_qc.iloc[0]['Tau']
            
            st.markdown(f"#### 🗝️ {qc}")
            st.caption(f"Récurrence détectée dans le lot : **Tau = {tau_real}** (Ce concept apparait {tau_real} fois)")
            
            # TABLEAU DE PREUVE
            st.dataframe(
                df_qc[["SCORE", "Qi_Source", "N_tot", "n_q", "Psi", "Alpha", "Sigma"]],
                column_config={
                    "Qi_Source": st.column_config.TextColumn("Qi (Phrase Élève)", width="large"),
                    "SCORE": st.column_config.ProgressColumn("Score", format="%.4f", min_value=0, max_value=5),
                    "N_tot": st.column_config.NumberColumn("N_tot (Longueur)", format="%d"),
                    "Psi": st.column_config.NumberColumn("Ψ (Richesse)", format="%.3f"),
                    "Sigma": st.column_config.NumberColumn("σ (Pénalité)", format="%.2f"),
                },
                use_container_width=True,
                hide_index=True
            )
    else:
        st.warning("Aucune donnée SMAXIA détectée dans les fichiers.")
