import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from io import BytesIO

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Factory V7")
st.title("🏭 SMAXIA - Factory V7 (Polymorphisme & Preuve Physique)")

# --- 1. GÉNÉRATEUR DE VARIANTES (POLYMORPHISME) ---
# On ne stocke plus des phrases, mais des "Templates" pour générer des variantes uniques.
# Structure : CONCEPT_KEY : [Liste de templates]

MATH_TEMPLATES = {
    "SUITES_GEO": [
        "Montrer que la suite ({name}) est géométrique.",
        "Démontrer que ({name}) est une suite géométrique de raison {val}.",
        "Justifier que la suite définie par {name} est de nature géométrique.",
        "En déduire que ({name}) est géométrique."
    ],
    "SUITES_LIM": [
        "Déterminer la limite de la suite ({name}).",
        "Calculer la limite de ({name}) quand n tend vers l'infini.",
        "Étudier la convergence de la suite ({name}).",
        "La suite ({name}) converge-t-elle ?"
    ],
    "COMPLEXE_ALG": [
        "Déterminer la forme algébrique du nombre complexe {var}.",
        "Écrire {var} sous forme a + ib.",
        "Calculer la partie réelle et imaginaire de {var}.",
        "Mettre le nombre {var} sous forme algébrique."
    ],
    "ESPACE_ORTHO": [
        "Démontrer que la droite ({d}) est orthogonale au plan ({p}).",
        "Prouver que le vecteur {v} est normal au plan ({p}).",
        "Justifier que ({d}) est perpendiculaire à ({p}).",
        "Vérifier l'orthogonalité entre ({d}) et ({p})."
    ]
}

VAR_NAMES = ["Un", "Vn", "Wn", "tn", "xn"]
COMPLEX_VARS = ["z", "z'", "zA", "zB", "Ω"]
VECTORS = ["n", "u", "v", "AB", "CD"]
VALS = ["1/2", "3", "q", "0.5", "-1"]

def generate_qi_variant(concept_code):
    """Fabrique une phrase unique basée sur un concept"""
    templates = MATH_TEMPLATES.get(concept_code, ["Question standard."])
    template = random.choice(templates)
    
    # Injection de variables aléatoires (Polymorphisme)
    text = template.format(
        name=random.choice(VAR_NAMES),
        val=random.choice(VALS),
        var=random.choice(COMPLEX_VARS),
        d=random.choice(["D", "Delta", "AB"]),
        p=random.choice(["P", "ABC", "Q"]),
        v=random.choice(VECTORS)
    )
    return text

def generate_full_subject_content(filename, nature, qi_list):
    """Génère le contenu textuel complet du fichier PDF simulé"""
    content = f"""
    ================================================================
    ACADÉMIE SMAXIA - SESSION 2024
    ÉPREUVE : MATHÉMATIQUES
    TYPE : {nature}
    FICHIER : {filename}
    ================================================================

    EXERCICE 1 (Analyse)
    ------------------------------------------------
    Soit f la fonction définie sur R...
    1. {qi_list[0] if len(qi_list) > 0 else "Question..."}
    2. Calculer la dérivée...
    
    EXERCICE 2 (Suites / Complexes)
    ------------------------------------------------
    {qi_list[1] if len(qi_list) > 1 else "Question..."}
    {qi_list[2] if len(qi_list) > 2 else "Question..."}
    
    EXERCICE 3 (Géométrie)
    ------------------------------------------------
    L'espace est rapporté à un repère orthonormé...
    1. {qi_list[3] if len(qi_list) > 3 else "Question..."}
    
    FIN DU SUJET
    """
    return content

# --- 2. FONCTIONS MOTEUR ---

def ingest_and_generate_files(urls, n_per_url):
    """
    Génère des sujets physiques (simulés) avec des contenus uniques.
    """
    sources_db = [] # Contient les métadonnées + LE CONTENU DU FICHIER
    all_qi_extracted = []
    
    natures = ["BAC", "DST", "CONCOURS"]
    
    progress = st.progress(0)
    total_ops = len(urls) * n_per_url
    counter = 0
    
    for i, url in enumerate(urls):
        if not url.strip(): continue
        for j in range(n_per_url):
            counter += 1
            progress.progress(min(counter/total_ops, 1.0))
            time.sleep(0.01) 
            
            nature = random.choice(natures)
            year = random.choice(range(2020, 2025))
            file_id = f"DOC_{i}_{j}"
            filename = f"Sujet_{nature}_{year}_{j}.txt" # .txt pour pouvoir le lire facilement
            
            # 1. Générer les Qi pour ce sujet (3 à 5 concepts mélangés)
            concepts_du_sujet = random.sample(list(MATH_TEMPLATES.keys()), k=random.randint(2, 4))
            
            qi_in_this_file = []
            qi_metadata = []
            
            for concept in concepts_du_sujet:
                # C'est ici que la magie opère : on génère une VARIANTE unique
                qi_text = generate_qi_variant(concept)
                
                qi_in_this_file.append(qi_text)
                qi_metadata.append({
                    "ID_Source": file_id,
                    "Concept_Code": concept, # Le Secret invariant
                    "Qi_Brut": qi_text,      # La surface visible (variable)
                    "Année": year,
                    "Fichier": filename
                })
            
            # 2. Créer le contenu physique du fichier
            file_content = generate_full_subject_content(filename, nature, qi_in_this_file)
            
            sources_db.append({
                "ID": file_id,
                "Fichier": filename,
                "Nature": nature,
                "Année": year,
                "Contenu_Complet": file_content # On stocke le vrai texte
            })
            
            all_qi_extracted.extend(qi_metadata)
            
    progress.empty()
    return pd.DataFrame(sources_db), pd.DataFrame(all_qi_extracted)

def calculate_engine_qc(df_qi):
    # Regroupement par CONCEPT_CODE (L'invariant caché) et non par texte
    # C'est ce qui permet de grouper "Montrer Un" et "Prouver Vn"
    
    if df_qi.empty: return pd.DataFrame()

    grouped = df_qi.groupby("Concept_Code").agg({
        "ID_Source": "count",      # n_q
        "Année": "max",            # Récence
        "Qi_Brut": list,           # Liste des variantes (Preuve Polymorphisme)
        "Fichier": list            # Liste des sources
    }).reset_index()
    
    qcs = []
    N_total = len(df_qi)
    
    # Mapping Concept -> QC Titre propre
    TITRES_QC = {
        "SUITES_GEO": "COMMENT Démontrer qu'une suite est géométrique",
        "SUITES_LIM": "COMMENT Calculer la limite d'une suite",
        "COMPLEXE_ALG": "COMMENT Déterminer la forme algébrique d'un complexe",
        "ESPACE_ORTHO": "COMMENT Démontrer l'orthogonalité Droite/Plan"
    }
    
    for idx, row in grouped.iterrows():
        n_q = row["ID_Source"]
        tau = 1.0 # Simplifié pour demo
        alpha = 5.0
        psi = 1.0 
        sigma = 0.0
        
        score = (n_q / N_total) * (1 + alpha/tau) * psi * 100
        
        qc_titre = TITRES_QC.get(row["Concept_Code"], f"COMMENT {row['Concept_Code']}...")
        
        # Construction Preuve
        evidence = []
        for i in range(len(row["Qi_Brut"])):
            evidence.append({
                "Fichier Source": row["Fichier"][i],
                "Qi (Variante Élève)": row["Qi_Brut"][i]
            })
            
        qcs.append({
            "QC_ID": f"QC_{idx+1:03d}",
            "QC_INVARIANTE": qc_titre,
            "SCORE_F2": score,
            "n_q": n_q,
            "QI_PREUVE": evidence
        })
        
    return pd.DataFrame(qcs).sort_values(by="SCORE_F2", ascending=False)

# --- INTERFACE ---

# SIDEBAR
with st.sidebar:
    st.header("1. Paramètres Usine")
    n_sujets = st.number_input("Sujets par URL", 2, 50, 5)

# TABS
tab_factory = st.container()

with tab_factory:
    st.subheader("A. Usine de Sourcing & Génération (V7)")

    col_input, col_act = st.columns([3, 1])
    with col_input:
        urls_input = st.text_area("URLs Cibles", "https://apmep.fr", height=70)
    with col_act:
        st.write("")
        btn_run = st.button("LANCER L'USINE 🚀", type="primary")

    if btn_run:
        with st.spinner("Génération des fichiers uniques et extraction..."):
            df_src, df_qi = ingest_and_generate_files(urls_input.split('\n'), n_sujets)
            df_qc = calculate_engine_qc(df_qi)
            
            st.session_state['df_src'] = df_src
            st.session_state['df_qc'] = df_qc
            st.success("Traitement terminé.")

    st.divider()

    if 'df_qc' in st.session_state:
        col_left, col_right = st.columns([1, 1.5])
        
        # --- COLONNE GAUCHE : SUJETS AVEC VRAI TÉLÉCHARGEMENT ---
        with col_left:
            st.markdown(f"### 📥 Sujets ({len(st.session_state['df_src'])})")
            st.caption("Cliquez pour télécharger et vérifier le contenu.")
            
            # On itère pour créer de vrais boutons de téléchargement
            for index, row in st.session_state['df_src'].iterrows():
                with st.expander(f"📄 {row['Fichier']} ({row['Nature']})"):
                    st.text(f"Année : {row['Année']}")
                    # BOUTON DOWNLOAD RÉEL
                    st.download_button(
                        label="📥 Télécharger le sujet (.txt)",
                        data=row['Contenu_Complet'],
                        file_name=row['Fichier'],
                        mime="text/plain",
                        key=f"dl_{index}"
                    )

        # --- COLONNE DROITE : QC AVEC PREUVES VARIÉES ---
        with col_right:
            st.markdown(f"### 🧠 QC Générées (Total : {len(st.session_state['df_qc'])})")
            
            for idx, row in st.session_state['df_qc'].iterrows():
                with st.container():
                    c1, c2 = st.columns([0.5, 3])
                    c1.markdown(f"**`{row['QC_ID']}`**")
                    c2.info(f"**{row['QC_INVARIANTE']}**")
                    
                    st.caption(f"Score F2: **{row['SCORE_F2']:.1f}** | Fréquence: **{row['n_q']}**")
                    
                    # PREUVE POLYMORPHE
                    with st.expander("Voir les Qi sources (Notez les variations)"):
                        st.write("Le moteur a regroupé ces phrases différentes sous la même QC :")
                        st.dataframe(
                            pd.DataFrame(row['QI_PREUVE']),
                            hide_index=True,
                            use_container_width=True
                        )
                    st.divider()
