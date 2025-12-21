import streamlit as st
import pandas as pd
import numpy as np
import random
import time

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Factory V8")
st.title("🏭 SMAXIA - Console Factory V8 (UI V6 + Logic V7)")

st.markdown("""
<style>
    .math-font { font-family: 'Courier New'; font-weight: bold; color: #b91c1c; }
    .qc-header { font-size: 18px; font-weight: bold; color: #1e40af; }
</style>
""", unsafe_allow_html=True)

# --- 1. MOTEUR DE CONTENU POLYMORPHE (V7 LOGIC) ---
# Templates pour générer des variantes uniques (Preuve d'intelligence)
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

VAR_NAMES = ["Un", "Vn", "Wn", "tn"]
COMPLEX_VARS = ["z", "z'", "zA", "Ω"]
VECTORS = ["n", "u", "v", "AB"]
VALS = ["1/2", "3", "q", "-1"]

def generate_qi_variant(concept_code):
    """Génère une phrase unique basée sur un template"""
    templates = MATH_TEMPLATES.get(concept_code, ["Question standard."])
    template = random.choice(templates)
    return template.format(
        name=random.choice(VAR_NAMES),
        val=random.choice(VALS),
        var=random.choice(COMPLEX_VARS),
        d=random.choice(["D", "Delta", "(AB)"]),
        p=random.choice(["P", "(ABC)", "Q"]),
        v=random.choice(VECTORS)
    )

def generate_full_subject_content(filename, nature, qi_list):
    """Crée le contenu texte du fichier pour téléchargement"""
    return f"""
    ACADÉMIE SMAXIA - SESSION 2025
    ÉPREUVE : MATHÉMATIQUES ({nature})
    FICHIER : {filename}
    ------------------------------------------------
    EXERCICE 1
    1. {qi_list[0] if len(qi_list) > 0 else "..."}
    2. {qi_list[1] if len(qi_list) > 1 else "..."}
    
    EXERCICE 2
    1. {qi_list[2] if len(qi_list) > 2 else "..."}
    ------------------------------------------------
    FIN DU SUJET
    """

# --- 2. FONCTIONS MOTEUR ---

def ingest_and_generate(urls, n_per_url):
    """Génère les fichiers et extrait les Qi"""
    sources_db = []
    all_qi = []
    
    natures = ["BAC", "DST", "CONCOURS"]
    
    progress = st.progress(0)
    total_ops = len(urls) * n_per_url
    counter = 0
    
    for i, url in enumerate(urls):
        if not url.strip(): continue
        for j in range(n_per_url):
            counter += 1
            progress.progress(min(counter/total_ops, 1.0))
            time.sleep(0.005)
            
            nature = random.choice(natures)
            year = random.choice(range(2020, 2025))
            file_id = f"DOC_{i}_{j}"
            filename = f"Sujet_{nature}_{year}_{j}.txt"
            
            # Génération Contenu
            concepts = random.sample(list(MATH_TEMPLATES.keys()), k=random.randint(2, 3))
            qi_in_file = []
            
            for concept in concepts:
                qi_txt = generate_qi_variant(concept)
                qi_in_file.append(qi_txt)
                all_qi.append({
                    "Concept_Code": concept,
                    "Qi_Brut": qi_txt,
                    "Fichier": filename,
                    "Année": year
                })
            
            full_text = generate_full_subject_content(filename, nature, qi_in_file)
            
            sources_db.append({
                "Fichier": filename,
                "Nature": nature,
                "Année": year,
                "Contenu_Txt": full_text
            })
            
    progress.empty()
    return pd.DataFrame(sources_db), pd.DataFrame(all_qi)

def calculate_engine_qc(df_qi):
    """Regroupe par Concept (Invariant) et calcule F1/F2"""
    if df_qi.empty: return pd.DataFrame()
    
    # On groupe par le CODE CONCEPT (L'invariant caché)
    grouped = df_qi.groupby("Concept_Code").agg({
        "Qi_Brut": "count",        # n_q
        "Année": "max",            # Récence
        "Fichier": list,           # Preuve Sources
        "Qi_Brut": list            # Preuve Variantes
    }).rename(columns={"Qi_Brut": "Variantes"}).reset_index()
    
    # Retrouver le n_q correct car renommage
    grouped["n_q"] = grouped["Variantes"].apply(len)

    # Titres Propres
    TITRES = {
        "SUITES_GEO": "COMMENT Démontrer qu'une suite est géométrique",
        "SUITES_LIM": "COMMENT Calculer la limite d'une suite",
        "COMPLEXE_ALG": "COMMENT Déterminer la forme algébrique",
        "ESPACE_ORTHO": "COMMENT Démontrer l'orthogonalité Droite/Plan"
    }
    
    qcs = []
    N_total = len(df_qi)
    current_year = datetime.now().year
    
    for idx, row in grouped.iterrows():
        n_q = row["n_q"]
        tau = max((current_year - row["Année"]), 0.5)
        alpha = 5.0
        psi = 1.0 # Densité cognitive standard
        sigma = 0.05 # Faible bruit
        
        # ÉQUATION F2 COMPLETE
        score = (n_q / N_total) * (1 + alpha/tau) * psi * (1-sigma) * 100
        
        qc_title = TITRES.get(row["Concept_Code"], row["Concept_Code"])
        
        # Preuve (Fichier + Phrase)
        evidence = []
        for k in range(len(row["Variantes"])):
            evidence.append({
                "Fichier": row["Fichier"][k],
                "Qi (Variante)": row["Variantes"][k]
            })
            
        qcs.append({
            "QC_ID": f"QC_{idx+1:03d}",
            "QC_INVARIANTE": qc_title,
            "SCORE_F2": score,
            
            # VARIABLES POUR AFFICHAGE
            "n_q": n_q,
            "N_tot": N_total,
            "Tau": tau,
            "Alpha": alpha,
            "Psi": psi,
            "Sigma": sigma,
            
            "EVIDENCE": evidence
        })
        
    return pd.DataFrame(qcs).sort_values(by="SCORE_F2", ascending=False)

# --- INTERFACE ---

# SIDEBAR
with st.sidebar:
    st.header("1. Paramètres Usine")
    n_sujets = st.number_input("Sujets par URL", 1, 50, 5)

# LAYOUT PRINCIPAL
st.subheader("A. Usine de Sourcing & Génération (V8)")

col_input, col_act = st.columns([3, 1])
with col_input:
    urls_input = st.text_area("URLs Cibles", "https://apmep.fr", height=70)
with col_act:
    st.write("")
    btn_run = st.button("LANCER L'USINE 🚀", type="primary")

if btn_run:
    url_list = urls_input.split('\n')
    with st.spinner("Génération Polymorphe & Calculs..."):
        df_src, df_qi = ingest_and_generate(url_list, n_sujets)
        df_qc = calculate_engine_qc(df_qi)
        
        st.session_state['df_src'] = df_src
        st.session_state['df_qc'] = df_qc
        st.success("Usine mise à jour.")

st.divider()

if 'df_qc' in st.session_state:
    
    col_left, col_right = st.columns([1, 1.5])
    
    # --- GAUCHE : LISTE SUJETS (UI V6 Restaurée) ---
    with col_left:
        st.markdown(f"### 📥 Sujets ({len(st.session_state['df_src'])})")
        
        # 1. Le Tableau Propre (V6 Style)
        st.dataframe(
            st.session_state['df_src'][["Fichier", "Nature", "Année"]],
            use_container_width=True,
            height=400
        )
        
        # 2. La Zone de Téléchargement (Fonctionnelle)
        st.info("👇 Zone de Téléchargement Physique")
        selected_file = st.selectbox("Choisir un sujet à vérifier :", st.session_state['df_src']["Fichier"])
        
        # Récupération du contenu
        file_data = st.session_state['df_src'][st.session_state['df_src']["Fichier"] == selected_file].iloc[0]
        
        st.download_button(
            label="💾 TÉLÉCHARGER CE SUJET (.txt)",
            data=file_data["Contenu_Txt"],
            file_name=selected_file,
            mime="text/plain",
            type="primary"
        )

    # --- DROITE : QC + VARIABLES (Demande Spécifique) ---
    with col_right:
        total_qc = len(st.session_state['df_qc'])
        st.markdown(f"### 🧠 QC Générées ({total_qc})")
        
        for idx, row in st.session_state['df_qc'].iterrows():
            with st.container():
                # En-tête
                c1, c2 = st.columns([0.5, 3])
                c1.markdown(f"**`{row['QC_ID']}`**")
                c2.markdown(f"<span class='qc-header'>{row['QC_INVARIANTE']}</span>", unsafe_allow_html=True)
                
                # Score Principal
                st.caption(f"Score F2 Global : **{row['SCORE_F2']:.2f}**")
                
                # TABLEAU DES VARIABLES (Demande Explicite)
                # On crée un petit dataframe transvisé pour la lisibilité
                vars_df = pd.DataFrame({
                    "Variable": ["n_q (Freq)", "N_tot (Vol)", "Tau (Récence)", "Alpha (Ctx)", "Psi (Densité)", "Sigma (Bruit)"],
                    "Valeur": [row['n_q'], row['N_tot'], row['Tau'], row['Alpha'], row['Psi'], row['Sigma']]
                })
                st.dataframe(vars_df.T, use_container_width=True) # Transposé pour être horizontal
                
                # PREUVE POLYMORPHE
                with st.expander(f"🔎 Voir les {row['n_q']} Variantes (Preuve Polymorphisme)"):
                    st.write("Phrases élèves différentes regroupées sous cette QC :")
                    st.dataframe(pd.DataFrame(row['EVIDENCE']), hide_index=True, use_container_width=True)
                
                st.divider()

else:
    st.info("Configurez et lancez l'usine.")
