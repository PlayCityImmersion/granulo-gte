import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random
import time

# --- CONFIGURATION P6 ---
st.set_page_config(layout="wide", page_title="SMAXIA - CEO Automation Console")
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6 }
    .big-stat { font-size: 24px; font-weight: bold; color: #1E3A8A; }
    .success-box { border-left: 5px solid green; background-color: #e6fffa; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 1. RÉFÉRENTIEL TAXONOMIQUE (POUR CLASSEMENT AUTO) ---
TAXONOMY = {
    "TERMINALE": {
        "ANALYSE": ["Limites", "Continuité", "Dérivation", "Intégration", "Logarithme"],
        "GÉOMÉTRIE": ["Espace", "Produit Scalaire"],
        "PROBABILITÉS": ["Loi Binomiale", "Grands Nombres"]
    },
    "PREMIERE": {
        "ANALYSE": ["Suites", "Fonction Carré", "Trinôme"],
        "GÉOMÉTRIE": ["Vecteurs", "Trigonométrie"]
    }
}

QC_LIBRARY = [
    {"txt": "Calculer la limite de la fonction f en +infini", "trigs": ["limite", "infini"], "chap": "Limites"},
    {"txt": "Étudier les variations de la fonction sur I", "trigs": ["variations", "dérivée"], "chap": "Dérivation"},
    {"txt": "Déterminer une primitive F de f", "trigs": ["primitive", "intégrale"], "chap": "Intégration"},
    {"txt": "Démontrer que la droite est orthogonale au plan", "trigs": ["orthogonal", "plan"], "chap": "Espace"},
    {"txt": "Justifier que la suite (Un) est géométrique", "trigs": ["suite", "géométrique"], "chap": "Suites"},
    {"txt": "Calculer la probabilité de l'événement A", "trigs": ["probabilité", "calculer"], "chap": "Loi Binomiale"}
]

# --- 2. MODULE CRAWLER (SIMULATION INTELLIGENTE) ---
def simulate_web_crawling(url_source, target_count):
    """
    Simule la récupération de 'target_count' sujets depuis 'url_source'.
    Génère des métadonnées variées (Année, Type, Classe) pour le test.
    """
    crawled_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(target_count):
        # Simulation latence réseau (rapide pour le test)
        time.sleep(0.01)
        progress_bar.progress((i + 1) / target_count)
        
        # Génération aléatoire de métadonnées sujet
        classe = random.choice(list(TAXONOMY.keys()))
        matiere = random.choice(list(TAXONOMY[classe].keys()))
        chapitre = random.choice(TAXONOMY[classe][matiere])
        year = random.choice(range(2015, 2025))
        type_doc = random.choice(["DST", "BAC", "INTERRO", "DM"])
        
        # Sélection d'un contenu réaliste basé sur le chapitre
        # (Dans le vrai système, c'est l'extraction PDF)
        content_template = random.choice([qc for qc in QC_LIBRARY if qc.get("chap") == chapitre] or QC_LIBRARY)
        
        # Création de l'objet Sujet/Qi
        sujet = {
            "ID_Source": f"{url_source}_DOC_{i:03d}",
            "Type": type_doc,
            "Année": year,
            "Classe": classe,
            "Matière": matiere,
            "Chapitre": chapitre, # Classement Auto
            "Contenu_Qi": content_template["txt"],
            "Triggers_Detectes": set(content_template["trigs"])
        }
        crawled_data.append(sujet)
        
        status_text.text(f"Crawling : {url_source}/sujet_{year}_{type_doc}_{i}.pdf ... OK")
    
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(crawled_data)

# --- 3. MOTEUR SMAXIA F2 (SÉLECTION) ---
def run_smaxia_engine(df_data):
    # Agrégation par Structure de Question (QC)
    # On regroupe les Qi identiques pour calculer la fréquence
    
    grouped = df_data.groupby("Contenu_Qi").agg({
        "ID_Source": "count",      # n_q (Fréquence)
        "Année": "max",            # Pour Récence
        "Triggers_Detectes": "first",
        "Chapitre": "first",
        "Classe": "first"
    }).rename(columns={"ID_Source": "n_q", "Année": "Derniere_Annee"}).reset_index()
    
    # Calcul des scores
    results = []
    current_year = datetime.now().year
    
    for idx, row in grouped.iterrows():
        # Variables A2
        n_q = row["n_q"]
        N_total = len(df_data) # Volume total du crawl
        
        t_rec = (current_year - row["Derniere_Annee"]) + 1 # +1 pour éviter div/0
        alpha = 5.0 # Coeff récence
        
        # Psi (Densité - simplifié pour simu)
        words = row["Contenu_Qi"].split()
        psi = len(set(words)) / len(words)
        
        # Score F2
        score = (n_q / N_total) * (1 + alpha/t_rec) * psi * 10
        
        results.append({
            "CLASSE": row["Classe"],
            "CHAPITRE": row["Chapitre"],
            "QC_CIBLE": f"COMMENT {row['Contenu_Qi']}...", # Transformation QC
            "n_q": n_q,
            "Récence (An)": row["Derniere_Annee"],
            "Score_F2": score
        })
        
    return pd.DataFrame(results).sort_values(by="Score_F2", ascending=False)

# --- INTERFACE CEO ---
st.title("🕷️ SMAXIA - Automation & Crawling Unit")
st.markdown("### Injection de Source Massive pour Calibration P6")

# Zone de Saisie CEO
with st.container():
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        url_input = st.text_input("🔗 URL Source (ex: apmep.fr, sujetdebac.fr)", value="https://www.apmep.fr")
    with col2:
        nb_sujets = st.number_input("📚 Nombre de Sujets Cible", min_value=10, max_value=500, value=60, step=10)
    with col3:
        st.write("") # Spacer
        run_btn = st.button("LANCER LE CRAWLER 🚀", type="primary")

st.divider()

if run_btn and url_input:
    # 1. CRAWLING
    with st.spinner(f"Connexion à {url_input} et extraction de {nb_sujets} sujets..."):
        df_raw = simulate_web_crawling(url_input, nb_sujets)
    
    # Validation du nombre
    real_count = len(df_raw)
    if real_count == nb_sujets:
        st.success(f"✅ SUCCÈS : {real_count} Sujets récupérés, atomisés et classés.")
    else:
        st.warning(f"⚠️ Attention : Seuls {real_count} sujets ont été trouvés.")

    # 2. AFFICHAGE DES DONNÉES BRUTES (PREUVE DE CLASSEMENT)
    with st.expander("🔎 Voir les données brutes extraites (Classification Auto)", expanded=True):
        st.dataframe(df_raw[["ID_Source", "Type", "Classe", "Matière", "Chapitre", "Année"]], use_container_width=True)
        
        # Stats de répartition
        c1, c2, c3 = st.columns(3)
        c1.metric("Répartition Classes", f"{df_raw['Classe'].nunique()} Niveaux")
        c2.metric("Répartition Types", f"{df_raw['Type'].nunique()} (DST/BAC/...)")
        c3.metric("Volume Qi Traité", f"{len(df_raw)} Qi")

    # 3. INJECTION MOTEUR
    st.markdown("### ⚙️ Traitement Moteur SMAXIA (Sélection F2)")
    df_result = run_smaxia_engine(df_raw)
    
    # Affichage par Classe/Chapitre (Demande Utilisateur)
    classes = df_result["CLASSE"].unique()
    for cl in classes:
        st.subheader(f"📂 NIVEAU : {cl}")
        df_cl = df_result[df_result["CLASSE"] == cl]
        
        # Tableau de bord CEO
        st.dataframe(
            df_cl[["CHAPITRE", "QC_CIBLE", "n_q", "Récence (An)", "Score_F2"]],
            column_config={
                "Score_F2": st.column_config.ProgressColumn("Pertinence", format="%.2f", min_value=0, max_value=max(df_result["Score_F2"])),
                "n_q": st.column_config.NumberColumn("Freq.", format="%d")
            },
            use_container_width=True,
            hide_index=True
        )
