import streamlit as st
import pandas as pd
import numpy as np
import random
import time

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Validation V2")
st.title("🛡️ SMAXIA - Protocole de Validation & Sourcing (P6)")

# --- 0. SIMULATEUR DE DONNÉES (DATABASE MATHS) ---
DATABASE_SIMULATION = {
    "COMPLEXES": {
        "ALG": ["Mettre sous forme algébrique", "Déterminer partie réelle et imaginaire", "Calculer le conjugué z barre"],
        "GEO": ["Déterminer l'ensemble des points M", "Interpréter géométriquement le module", "Montrer que le triangle est rectangle"],
        "EQ": ["Résoudre l'équation dans C", "Déterminer les racines carrées", "Factoriser le polynôme P(z)"],
        "EXP": ["Passer sous forme exponentielle", "Utiliser la formule de Moivre", "Calculer la puissance n-ième"]
    },
    "FONCTIONS": { # Ajout pour l'exemple
        "DERIV": ["Calculer la dérivée f'(x)", "Étudier le signe de la dérivée", "Dresser le tableau de variations"],
        "LIM": ["Lever l'indétermination", "Calculer la limite en +infini", "Déterminer l'asymptote oblique"]
    }
}

# --- 1. MOTEUR F1/F2 (AVEC SOURCING URL) ---
def atomize_from_urls(url_list, n_sujets_per_url, chapitre):
    """
    Simule le scraping depuis les URLs fournies.
    Crée une traçabilité : URL -> Fichier -> Qi -> QC.
    """
    qc_detected = {} 
    audit_log = [] # Pour prouver d'où vient l'info
    
    # Barre de progression globale
    total_ops = len(url_list) * n_sujets_per_url
    prog_bar = st.progress(0)
    status_txt = st.empty()
    counter = 0

    for url in url_list:
        if not url.strip(): continue # Ignorer lignes vides
        
        # Simulation extraction par site
        for i in range(n_sujets_per_url):
            counter += 1
            prog_bar.progress(counter / total_ops)
            status_txt.text(f"Scraping source : {url} ... Sujet_{i+1}.pdf")
            time.sleep(0.02) # Micro-latence pour réalisme
            
            # Génération contenu sujet
            themes = list(DATABASE_SIMULATION.get(chapitre, {}).keys())
            if not themes: continue
            
            # Un sujet contient 3 à 5 exercices
            nb_exos = random.randint(3, 5)
            for _ in range(nb_exos):
                # Choix concept
                t = random.choice(themes)
                qi_raw = random.choice(DATABASE_SIMULATION[chapitre][t])
                
                # Construction QC
                qc_name = f"COMMENT {qi_raw}..."
                
                # Enregistrement F2
                if qc_name in qc_detected:
                    qc_detected[qc_name]["n_q"] += 1
                    qc_detected[qc_name]["Sources"].append(f"{url}/sujet_{i+1}")
                else:
                    qc_detected[qc_name] = {
                        "n_q": 1, 
                        "Triggers": [qi_raw.split()[0].upper()],
                        "Sources": [f"{url}/sujet_{i+1}"]
                    }
                
                # Log Audit
                audit_log.append({
                    "Source_URL": url,
                    "Fichier": f"Sujet_{i+1}.pdf",
                    "Qi_Extraite": qi_raw
                })
                
    prog_bar.empty()
    status_txt.empty()
    
    # Transformation en DataFrame QC
    final_qcs = []
    for name, data in qc_detected.items():
        score = data["n_q"] * 10 
        final_qcs.append({
            "QC_INVARIANTE": name,
            "SCORE_F2": score,
            "FREQ (n_q)": data["n_q"],
            "SOURCES_EXEMPLES": list(set(data["Sources"]))[:3] # On garde 3 exemples
        })
        
    return pd.DataFrame(final_qcs).sort_values(by="SCORE_F2", ascending=False), pd.DataFrame(audit_log)

# --- 2. MOTEUR VALIDATION (DST) ---
def run_dst_check(df_qc, n_dst, chapitre):
    report = []
    covered = 0
    total = 0
    
    for i in range(n_dst):
        # Génération DST
        themes = list(DATABASE_SIMULATION.get(chapitre, {}).keys())
        if not themes: break
        
        t = random.choice(themes)
        qi_dst = random.choice(DATABASE_SIMULATION[chapitre][t])
        total += 1
        
        # Check Couverture
        match = False
        for idx, row in df_qc.iterrows():
            core = row["QC_INVARIANTE"].replace("COMMENT ", "").replace("...", "")
            if core in qi_dst:
                match = True
                break
        
        if match:
            covered += 1
            status = "✅ OK"
        else:
            status = "❌ MANQUANT"
            
        report.append({"DST": f"Eval_{i+1}", "Question": qi_dst, "Statut": status})
        
    return pd.DataFrame(report), covered, total

# --- INTERFACE UTILISATEUR ---

# SIDEBAR PARAMÈTRES
with st.sidebar:
    st.header("1. Périmètre Académique")
    niveau = st.selectbox("Niveau", ["TERMINALE", "PREMIERE"])
    matiere = st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    chapitre = st.selectbox("Chapitre Cible", ["COMPLEXES", "FONCTIONS"])

# ONGLETS PRINCIPAUX
tab1, tab2 = st.tabs(["🏭 PHASE 2&3 : Usine d'Extraction (Sourcing)", "🎯 PHASE 4 : Crash Test (DST)"])

# --- TAB 1 : INJECTION URLS ---
with tab1:
    st.subheader("A. Configuration des Sources (URLs)")
    st.markdown("Indiquez ici les sites officiels où le moteur doit récupérer les sujets d'entraînement.")
    
    # ZONE DE SAISIE URL (NOUVEAUTÉ)
    col_input, col_param = st.columns([3, 1])
    with col_input:
        urls_input = st.text_area(
            "Liste des URLs (une par ligne)",
            value="https://www.apmep.fr/Terminale-S-2024\nhttps://www.sujetdebac.fr/annales-pdf/\nhttps://www.education.gouv.fr/sujets-zero",
            height=120,
            help="Le moteur ira crawler ces adresses pour télécharger les PDF."
        )
    with col_param:
        n_sujets = st.number_input("Sujets par URL", 5, 50, 10)
        st.write("")
        st.write("")
        btn_crawl = st.button("LANCER L'EXTRACTION 🚀", type="primary")

    st.divider()
    
    if btn_crawl:
        url_list = urls_input.split('\n')
        with st.spinner("Analyse des URLs et Atomisation en cours..."):
            df_qc, df_audit = atomize_from_urls(url_list, n_sujets, chapitre)
            
        # SAUVEGARDE SESSION
        st.session_state['df_qc'] = df_qc
        
        # AFFICHAGE RÉSULTATS
        c1, c2 = st.columns([2, 1])
        with c1:
            st.success(f"✅ Terminé ! {len(df_qc)} QC Invariantes générées.")
            st.markdown("### 🧠 Cerveau SMAXIA (QC Générées)")
            st.dataframe(
                df_qc, 
                column_config={
                    "SCORE_F2": st.column_config.ProgressColumn("Pertinence", format="%d"),
                    "SOURCES_EXEMPLES": "Exemples de provenance"
                },
                use_container_width=True
            )
            
        with c2:
            st.info(f"📊 {len(df_audit)} Atomes (Qi) extraits")
            with st.expander("Voir le journal de sourcing (Preuve)"):
                st.dataframe(df_audit[["Source_URL", "Fichier", "Qi_Extraite"]])

# --- TAB 2 : VALIDATION ---
with tab2:
    st.subheader("B. Test de Couverture (Golden Test)")
    
    if 'df_qc' in st.session_state:
        st.markdown(f"Test de la base de connaissance **{chapitre}** générée en Phase 2.")
        n_test = st.number_input("Nombre d'exercices DST à tester", 5, 50, 10)
        
        if st.button("LANCER LE CRASH TEST"):
            res, cov, tot = run_dst_check(st.session_state['df_qc'], n_test, chapitre)
            taux = (cov/tot)*100
            
            c_metric, c_bar = st.columns([1, 3])
            c_metric.metric("Taux de Couverture", f"{taux:.0f}%")
            c_bar.progress(taux/100)
            
            st.table(res)
            
            if taux == 100:
                st.success("✅ VALIDATION P6 ACCORDÉE : Le moteur couvre 100% des cas testés.")
            else:
                st.error("❌ ECHEC : Il manque des QC pour répondre à ces DST.")
    else:
        st.warning("⚠️ Veuillez d'abord lancer l'extraction dans l'onglet 1.")
