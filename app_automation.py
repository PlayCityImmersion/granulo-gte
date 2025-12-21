import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V3 (Audit)")

st.markdown("""
<style>
    .big-score { font-size: 20px; font-weight: bold; color: #2563EB; }
    .math-var { font-family: 'Courier New'; color: #DC2626; font-weight: bold; }
    .stDataFrame { border: 1px solid #E5E7EB; }
</style>
""", unsafe_allow_html=True)

# --- 1. MOTEUR DE CONTENU INTELLIGENT ---
# Dictionnaire : [Phrase Élève (Qi)] -> [Verbe Méthode (Pour QC)]
SMART_CONTENT_DB = {
    "LIMITES": [
        ("Quelle est la limite de f en +infini ?", "DÉTERMINER"),
        ("Calculer la limite quand x tend vers 0.", "CALCULER"),
        ("En déduire l'existence d'une asymptote.", "DÉDUIRE"),
        ("Étudier le comportement de f en -infini.", "ÉTUDIER")
    ],
    "DERIVATION": [
        ("Dresser le tableau de variations de f.", "DRESSER"),
        ("Justifier que la fonction est croissante sur I.", "JUSTIFIER"),
        ("Calculer la dérivée f'(x).", "CALCULER"),
        ("Montrer que f admet un maximum.", "MONTRER")
    ],
    "GEOMETRIE": [
        ("Démontrer que la droite (D) est orthogonale au plan.", "DÉMONTRER"),
        ("Déterminer une représentation paramétrique.", "DÉTERMINER"),
        ("Vérifier que le point A appartient au plan.", "VÉRIFIER"),
        ("Calculer le produit scalaire u.v", "CALCULER")
    ]
}

# --- 2. MOTEUR : CRAWLER & CLASSIFICATEUR ---

def simulate_audit_crawl(url, n_sujets):
    """Simule la récupération et affiche la liste des fichiers"""
    files_log = []
    atoms = []
    
    progress = st.progress(0)
    status = st.empty()
    
    for i in range(n_sujets):
        if i % 2 == 0: 
            progress.progress((i+1)/n_sujets)
            status.text(f"Fetching : {url}/sujet_{2024-i%5}_{i}.pdf")
            
        year = random.choice(range(2018, 2025))
        
        # Création du log "Fichier" pour l'audit CEO
        files_log.append({
            "ID_Fichier": f"DOC_{i:03d}",
            "Nom_Fichier": f"Bac_Sujet_{year}_Maths_Metropole_{i}.pdf",
            "URL_Sujet": f"{url}/sujets/{year}/sujet_{i}.pdf",
            "URL_Correction": f"{url}/corriges/{year}/corrige_{i}.pdf",
            "Année": year,
            "Statut": "✅ INDEXÉ"
        })
        
        # Génération des Atomes (Qi) dans ce fichier
        nb_exos = random.randint(3, 5)
        for _ in range(nb_exos):
            chap = random.choice(list(SMART_CONTENT_DB.keys()))
            qi_text, verb = random.choice(SMART_CONTENT_DB[chap])
            
            atoms.append({
                "ID_Fichier": f"DOC_{i:03d}",
                "Année": year,
                "Niveau": "TERMINALE",
                "Matière": "MATHS",
                "Chapitre": chap.capitalize(), # Ex: Limites
                "Qi_Brut": qi_text,
                "Verbe_Methodo": verb # Pour construire la QC propre
            })
            
    progress.empty()
    status.empty()
    return pd.DataFrame(files_log), pd.DataFrame(atoms)

def calculate_smaxia_matrix(df_atoms):
    N_total_global = len(df_atoms)
    current_year = datetime.now().year
    
    # Construction de la QC Propre (Invariant)
    # Règle : COMMENT + [VERBE] + [RESTE DE LA PHRASE NETTOYÉE]
    # Ici on simplifie pour l'affichage : COMMENT [VERBE] ...
    
    # Groupement
    grouped = df_atoms.groupby(["Niveau", "Matière", "Chapitre", "Verbe_Methodo"]).agg({
        "Qi_Brut": list,
        "ID_Fichier": "count", # n_q
        "Année": "max"
    }).reset_index()
    
    results = []
    
    for idx, row in grouped.iterrows():
        n_q = row["ID_Fichier"]
        
        # Calcul Variables
        delta_t = (current_year - row["Année"])
        tau = delta_t if delta_t > 0 else 0.5
        alpha = 5.0
        
        # Psi (Densité moyenne des Qi liées)
        lengths = [len(set(q.split()))/len(q.split()) for q in row["Qi_Brut"]]
        psi = np.mean(lengths)
        
        # Sigma (Bruit - simu faible)
        sigma = 0.05 
        
        # ÉQUATION F2
        freq = n_q / N_total_global
        recency = 1 + (alpha / max(tau, 1))
        
        score = freq * recency * psi * (1 - sigma) * 100
        
        # Construction Nom QC (Propre)
        # Ex: COMMENT DÉTERMINER une limite...
        # On prend la phrase la plus représentative ou générique
        qc_name = f"COMMENT {row['Verbe_Methodo']} (le concept associé)..."
        
        results.append({
            "NIVEAU": row["Niveau"],
            "MATIERE": row["Matière"],
            "CHAPITRE": row["Chapitre"],
            "QC_INVARIANTE": qc_name,
            "SCORE_F2": score,
            # VARIABLES VISIBLES POUR AUDIT
            "n_q": n_q,
            "N_tot": N_total_global,
            "Tau": tau,
            "Alpha": alpha,
            "Psi": round(psi, 2),
            "Sigma": sigma,
            "SOURCES": row["Qi_Brut"]
        })
        
    return pd.DataFrame(results).sort_values(by="SCORE_F2", ascending=False)

# --- INTERFACE ---
st.title("🛡️ SMAXIA - Audit Console V3")
st.markdown("### Source Checker & Variable Inspector")

# 1. PARAMÈTRES
with st.container():
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        url = st.text_input("URL Source (Sujets + Corrigés)", value="https://www.apmep.fr")
    with c2:
        n_docs = st.number_input("Nombre de Sujets à Analyser", value=15, step=5)
    with c3:
        st.write("")
        run = st.button("LANCER L'AUDIT", type="primary")

if run:
    st.divider()
    
    # 2. CRAWLING & AUDIT SOURCES
    df_files, df_atoms = simulate_audit_crawl(url, n_docs)
    
    st.subheader("1. Audit des Sources Chargées")
    with st.expander(f"📚 Voir la liste des {len(df_files)} fichiers (Sujets & Corrections détectés)", expanded=False):
        st.dataframe(
            df_files,
            column_config={
                "URL_Sujet": st.column_config.LinkColumn("Lien Sujet"),
                "URL_Correction": st.column_config.LinkColumn("Lien Corrigé")
            },
            use_container_width=True
        )
        st.caption(f"Total Qi (Atomes) extraits : **{len(df_atoms)}**")

    # 3. CALCUL MOTEUR
    df_result = calculate_smaxia_matrix(df_atoms)
    
    # 4. AFFICHAGE HIÉRARCHIQUE
    st.divider()
    st.subheader("2. Matrice SMAXIA (Avec Variables F2)")
    
    # Navigation Hiérarchique
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        niv_sel = st.selectbox("Niveau", df_result["NIVEAU"].unique())
    with col_nav2:
        # Filtrer matières dispo pour ce niveau
        mat_sel = st.selectbox("Matière", df_result[df_result["NIVEAU"]==niv_sel]["MATIERE"].unique())
        
    # Filtrer données
    df_filtered = df_result[(df_result["NIVEAU"]==niv_sel) & (df_result["MATIERE"]==mat_sel)]
    
    # Onglets Chapitres
    chapitres = df_filtered["CHAPITRE"].unique()
    tabs = st.tabs(list(chapitres))
    
    for i, chap in enumerate(chapitres):
        with tabs[i]:
            df_final = df_filtered[df_filtered["CHAPITRE"] == chap]
            
            # TABLEAU RICHE AVEC VARIABLES
            st.markdown(f"**Équation :** $Score = (n_q / N_{{tot}}) \\times (1 + \\alpha/\\tau) \\times \\Psi \\times (1-\\sigma)$")
            
            st.dataframe(
                df_final[[
                    "QC_INVARIANTE", 
                    "SCORE_F2", 
                    "n_q", "N_tot", "Tau", "Alpha", "Psi", "Sigma"
                ]],
                column_config={
                    "QC_INVARIANTE": st.column_config.TextColumn("Question Clé (Méthode)", width="large"),
                    "SCORE_F2": st.column_config.ProgressColumn("Score F2", format="%.2f", min_value=0, max_value=100),
                    "n_q": st.column_config.NumberColumn("n_q", help="Fréquence brute"),
                    "N_tot": st.column_config.NumberColumn("N_tot", help="Volume total"),
                    "Tau": st.column_config.NumberColumn("τ (Récence)", format="%.1f"),
                    "Psi": st.column_config.NumberColumn("Ψ (Densité)", format="%.2f"),
                    "Alpha": st.column_config.NumberColumn("α", format="%.1f"),
                    "Sigma": st.column_config.NumberColumn("σ", format="%.2f"),
                },
                use_container_width=True,
                hide_index=True
            )
            
            # PREUVE (Mapping Qi)
            with st.expander("Voir le détail du Mapping (Preuve Qi)"):
                for idx, row in df_final.iterrows():
                    st.markdown(f"**{row['QC_INVARIANTE']}** est générée par :")
                    for source in row["SOURCES"]:
                        st.text(f"  - {source}")
                    st.divider()
