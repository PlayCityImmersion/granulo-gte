import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V19.1")
st.title("🛡️ SMAXIA - Console V19.1 (Audit & Download Fix)")

# ==============================================================================
# 🎨 STYLES CSS
# ==============================================================================
st.markdown("""
<style>
    /* EN-TÊTE QC */
    .qc-header-row {
        background-color: #f8f9fa; border-left: 5px solid #2563eb;
        padding: 12px 15px; margin-bottom: 8px; border-radius: 4px;
        font-family: 'Source Sans Pro', sans-serif;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        display: flex; justify-content: space-between; align-items: center;
    }
    .qc-title-group { display: flex; align-items: center; flex-grow: 1; }
    .qc-id { color: #d97706; font-weight: 800; font-size: 1.1em; margin-right: 15px; min-width: 80px; }
    .qc-text { color: #111827; font-weight: 600; font-size: 1.1em; }
    .qc-stats { 
        font-family: 'Courier New', monospace; font-size: 0.9em; font-weight: 700; color: #4b5563;
        background-color: #e5e7eb; padding: 5px 10px; border-radius: 4px; white-space: nowrap; margin-left: 10px;
    }

    /* CONTENEURS DÉTAILS */
    .trigger-container { background-color: #fff1f2; padding: 10px; border-radius: 6px; border: 1px solid #fecdd3; }
    .trigger-item { background-color: #ffffff; color: #be123c; padding: 4px 8px; border-radius: 12px; font-size: 0.85em; font-weight: 700; border: 1px solid #fda4af; display: inline-block; margin: 3px; }
    
    .ari-box { background-color: #f3f4f6; padding: 10px; border-radius: 6px; font-family: monospace; font-size: 0.9em; color: #374151; border: 1px dashed #9ca3af; }
    
    .frt-box { background-color: #ecfdf5; padding: 15px; border-radius: 6px; font-family: sans-serif; line-height: 1.5; color: #065f46; border: 1px solid #6ee7b7; white-space: pre-wrap; }
    
    /* TABLEAUX HTML */
    .qi-table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
    .qi-table th { background: #f9fafb; text-align: left; padding: 8px; border-bottom: 2px solid #e5e7eb; color: #6b7280; }
    .qi-table td { padding: 8px; border-bottom: 1px solid #f3f4f6; vertical-align: top; color: #1f2937; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. DATA KERNEL
# ==============================================================================
LISTE_CHAPITRES = {
    "MATHS": [
        "SUITES NUMÉRIQUES", "FONCTIONS & DÉRIVATION", "LIMITES DE FONCTIONS", 
        "CONTINUITÉ & CONVEXITÉ", "FONCTION LOGARITHME", "PRIMITIVES & ÉQUATIONS DIFF", 
        "CALCUL INTÉGRAL", "LOI BINOMIALE", "GÉOMÉTRIE DANS L'ESPACE"
    ],
    "PHYSIQUE": [
        "MÉCANIQUE DE NEWTON", "MOUVEMENT DANS UN CHAMP", "ONDES MÉCANIQUES", "TRANSFORMATIONS CHIMIQUES"
    ]
}

UNIVERS_SMAXIA = {
    "FRT_M_SUITE_01": {
        "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", "Proba": 0.9,
        "QC": "comment démontrer qu'une suite est géométrique ?",
        "Triggers": ["montrer que la suite est géométrique", "quelle est la nature de la suite", "déterminer la raison q"],
        "ARI": ["Calcul u(n+1)", "Ratio u(n+1)/u(n)", "Simplification", "Identification Constante"],
        "FRT": """🔔 **Quand utiliser ?** Lorsque l'énoncé demande la nature de la suite.\n\n✅ **Méthode Standard :**\n1. Exprimer $u_{n+1}$.\n2. Calculer le rapport $\\frac{u_{n+1}}{u_n}$.\n3. Simplifier pour trouver $q$."""
    },
    "FRT_M_SUITE_02": {
        "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", "Proba": 0.8,
        "QC": "comment lever une indétermination (limite) ?",
        "Triggers": ["déterminer la limite", "calculer la limite quand n tend vers l'infini"],
        "ARI": ["Identifier FI", "Factoriser terme dominant", "Limites usuelles", "Opérations"],
        "FRT": """🔔 **Quand utiliser ?** Présence d'une forme indéterminée.\n\n✅ **Méthode Standard :**\n1. Identifier le terme dominant.\n2. Factoriser.\n3. Conclure."""
    },
    "FRT_M_FCT_01": {
        "Matiere": "MATHS", "Chap": "FONCTIONS & DÉRIVATION", "Proba": 0.9,
        "QC": "comment étudier les variations d'une fonction ?",
        "Triggers": ["étudier le sens de variation", "dresser le tableau de variations"],
        "ARI": ["Dérivée f'", "Signe f'", "Tableau"],
        "FRT": """🔔 **Quand utiliser ?** Pour connaitre la croissance.\n\n✅ **Méthode Standard :**\n1. Calculer $f'(x)$.\n2. Étudier le signe.\n3. Conclure sur les variations."""
    },
    "FRT_P_MECA_01": {
        "Matiere": "PHYSIQUE", "Chap": "MÉCANIQUE DE NEWTON", "Proba": 0.9,
        "QC": "comment déterminer le vecteur accélération ?",
        "Triggers": ["déterminer les coordonnées du vecteur accélération", "appliquer la deuxième loi de newton"],
        "ARI": ["Référentiel", "Bilan Forces", "2e Loi Newton", "Projection"],
        "FRT": """🔔 **Quand utiliser ?** Pour trouver l'accélération.\n\n✅ **Méthode Standard :**\n1. Bilan des forces.\n2. Appliquer $\\sum \\vec{F} = m\\vec{a}$.\n3. Projeter."""
    }
}

QI_PATTERNS = {
    "FRT_M_SUITE_01": ["Montrer que (Un) est géométrique.", "Quelle est la nature de la suite (Vn) ?", "Justifier que la suite est géométrique de raison 3."],
    "FRT_M_SUITE_02": ["Déterminer la limite de la suite.", "Calculer la limite quand n tend vers l'infini.", "Étudier la convergence."],
    "FRT_M_FCT_01": ["Étudier les variations de f.", "Dresser le tableau de variations complet.", "Quel est le sens de variation de la fonction ?"],
    "FRT_P_MECA_01": ["En déduire les coordonnées du vecteur accélération.", "Appliquer la 2e loi de Newton pour trouver a(t)."]
}

# ==============================================================================
# 2. MOTEUR
# ==============================================================================

def ingest_factory(urls, volume, matiere, chapitres):
    target_frts = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere and v["Chap"] in chapitres]
    if not target_frts and volume > 0: return pd.DataFrame(), pd.DataFrame()
    
    sources, atoms = [], []
    progress = st.progress(0)
    
    for i in range(volume):
        progress.progress((i+1)/volume)
        nature = random.choice(["BAC", "DST", "INTERRO"])
        annee = random.choice(range(2020, 2025))
        filename = f"Sujet_{matiere}_{nature}_{annee}_{i}.pdf"
        
        # Pour l'audit, on veut BEAUCOUP de Qi dans le sujet
        nb_qi = random.randint(5, 12) 
        frts = random.choices(target_frts, k=nb_qi)
        
        qi_data_list = []
        for frt_id in frts:
            qi_txt = random.choice(QI_PATTERNS[frt_id]) + f" [Ex:{random.randint(1,20)}]"
            atoms.append({"FRT_ID": frt_id, "Qi": qi_txt, "File": filename, "Year": annee, "Chap": UNIVERS_SMAXIA[frt_id]["Chap"]})
            qi_data_list.append({"Qi": qi_txt, "FRT_ID": frt_id})
            
        # Lien simulé pour data_editor
        dl_link = f"https://fake-smaxia-cloud.com/dl/{filename}"
        
        sources.append({
            "Fichier": filename, "Nature": nature, "Année": annee,
            "Télécharger": dl_link, # Le lien qui sera cliquable
            "Blob": f"Contenu simulé de {filename}", "Qi_Data": qi_data_list
        })
        
    return pd.DataFrame(sources), pd.DataFrame(atoms)

def compute_qc(df_atoms):
    if df_atoms.empty: return pd.DataFrame()
    grouped = df_atoms.groupby("FRT_ID").agg({"Qi": list, "File": list, "Year": "max", "Chap": "first"}).reset_index()
    qcs = []
    N_tot = len(df_atoms)
    
    for idx, row in grouped.iterrows():
        meta = UNIVERS_SMAXIA[row["FRT_ID"]]
        n_q = len(row["Qi"])
        t_rec = max(datetime.now().year - row["Year"], 0.5)
        psi = 0.85
        score = (n_q / N_tot) * (1 + 5.0/t_rec) * psi * 100
        qcs.append({
            "Chapitre": row["Chap"], "QC_ID": f"QC-{idx+1:02d}", "FRT_ID": row["FRT_ID"],
            "Titre": meta["QC"], "Score": score, "n_q": n_q, "Psi": psi, "N_tot": N_tot, "t_rec": t_rec,
            "Triggers": meta["Triggers"], "ARI": meta["ARI"], "FRT": meta["FRT"],
            "Evidence": [{"Fichier": f, "Qi": q} for f, q in zip(row["File"], row["Qi"])]
        })
    return pd.DataFrame(qcs).sort_values(by="Score", ascending=False)

def analyze_external(file_obj, matiere, chapitres):
    # Simulation d'extraction sur un gros fichier
    target_frts = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere and v["Chap"] in chapitres]
    if not target_frts: return []
    
    # On simule un sujet long (15 questions)
    nb_qi = 15
    frts = random.choices(target_frts, k=nb_qi)
    result = []
    for frt_id in frts:
        qi_txt = random.choice(QI_PATTERNS[frt_id]) + " (Extrait PDF)"
        result.append({"Qi": qi_txt, "FRT_ID": frt_id})
    return result

# ==============================================================================
# 3. INTERFACE
# ==============================================================================

with st.sidebar:
    st.header("Paramètres Académiques")
    st.selectbox("Classe", ["Terminale"], disabled=True)
    sel_matiere = st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    sel_chapitres = st.multiselect("Chapitres", LISTE_CHAPITRES[sel_matiere], default=[LISTE_CHAPITRES[sel_matiere][0]])

tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine", "✅ Onglet 2 : Audit"])

# --- USINE ---
with tab_usine:
    st.subheader("1. Configuration Sourcing")
    c1, c2 = st.columns([3, 1])
    with c1: urls = st.text_area("URLs Sources", "https://apmep.fr", height=68)
    with c2: 
        vol = st.number_input("Volume", 5, 500, 20, step=5)
        run = st.button("LANCER L'USINE 🚀", type="primary")

    if run:
        df_src, df_atoms = ingest_factory(urls.split('\n'), vol, sel_matiere, sel_chapitres)
        df_qc = compute_qc(df_atoms)
        st.session_state['df_src'] = df_src
        st.session_state['df_qc'] = df_qc
        st.success(f"Ingestion terminée : {len(df_src)} sujets traités.")

    st.divider()

    if 'df_src' in st.session_state and not st.session_state['df_src'].empty:
        # TABLEAU SUJETS (Téléchargement via LinkColumn)
        st.markdown(f"### 📥 Sujets Traités ({len(st.session_state['df_src'])})")
        
        st.data_editor(
            st.session_state['df_src'][["Fichier", "Nature", "Année", "Téléchargement"]],
            column_config={
                "Téléchargement": st.column_config.LinkColumn(
                    "Téléchargement",
                    help="Cliquer pour télécharger",
                    validate="^https://.*",
                    display_text="📥 Télécharger PDF"
                )
            },
            hide_index=True,
            use_container_width=True,
            disabled=True # Lecture seule
        )

        st.divider()

        # LISTE QC
        st.markdown("### 🧠 Base de Connaissance (QC)")
        if not st.session_state['df_qc'].empty:
            chapters = st.session_state['df_qc']["Chapitre"].unique()
            for chap in chapters:
                subset = st.session_state['df_qc'][st.session_state['df_qc']["Chapitre"] == chap]
                st.markdown(f"#### 📘 Chapitre {chap} : {len(subset)} QC")
                
                for idx, row in subset.iterrows():
                    st.markdown(f"""
                    <div class="qc-header-row">
                        <div class="qc-title-group">
                            <span class="qc-id">{row['QC_ID']}</span>
                            <span class="qc-text">{row['Titre']}</span>
                        </div>
                        <span class="qc-stats">Score(q)={row['Score']:.0f} | n_q={row['n_q']} | Ψ={row['Psi']} | N_tot={row['N_tot']} | t_rec={row['t_rec']:.1f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        with st.expander("🔥 Déclencheurs"):
                            html_trig = "<div class='trigger-container'>"
                            for t in row['Triggers']: html_trig += f"<span class='trigger-item'>{t}</span>"
                            html_trig += "</div>"
                            st.markdown(html_trig, unsafe_allow_html=True)
                    with c2:
                        with st.expander("⚙️ ARI (Moteur)"):
                            st.markdown(f"<div class='ari-box'>{' > '.join(row['ARI'])}</div>", unsafe_allow_html=True)
                    with c3:
                        with st.expander("🧾 FRT (Élève)"):
                            st.markdown(f"<div class='frt-box'>{row['FRT']}</div>", unsafe_allow_html=True)
                    with c4:
                        with st.expander(f"📄 Qi ({row['n_q']})"):
                            html = "<table class='qi-table'>"
                            for item in row['Evidence']:
                                html += f"<tr><td>{item['Fichier']}</td><td>{item['Qi']}</td></tr>"
                            html += "</table>"
                            st.markdown(html, unsafe_allow_html=True)
                    st.write("")
        else:
            st.warning("Aucune QC générée.")

# --- AUDIT ---
with tab_audit:
    st.subheader("Validation Booléenne")
    
    if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
        
        # TEST 1
        st.markdown("#### ✅ 1. Test Interne (Sujet Traité)")
        t1_file = st.selectbox("Choisir un sujet traité", st.session_state['df_src']["Fichier"])
        
        if st.button("LANCER TEST INTERNE"):
            # Extraction COMPLETE des Qi du sujet
            data = st.session_state['df_src'][st.session_state['df_src']["Fichier"]==t1_file].iloc[0]["Qi_Data"]
            known_ids = st.session_state['df_qc']["FRT_ID"].unique()
            
            ok_count = 0
            rows = []
            for item in data:
                is_ok = item["FRT_ID"] in known_ids
                if is_ok: ok_count += 1
                status = "✅ MATCH" if is_ok else "❌ ERREUR"
                
                qc_nom = "---"
                if is_ok:
                    qc_info = st.session_state['df_qc'][st.session_state['df_qc']["FRT_ID"]==item["FRT_ID"]].iloc[0]
                    qc_nom = f"{qc_info['QC_ID']} {qc_info['Titre']}"
                
                rows.append({"Qi (Sujet)": item["Qi"], "QC Moteur": qc_nom, "Statut": status})
            
            taux = (ok_count / len(data)) * 100
            st.markdown(f"### Taux de Couverture : {taux:.0f}% ({ok_count}/{len(data)} Qi)")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.divider()

        # TEST 2
        st.markdown("#### 🌍 2. Test Externe (Mapping Nouveau Sujet)")
        up_file = st.file_uploader("Charger un PDF externe", type="pdf")
        
        if up_file:
            extracted_qi = analyze_external(up_file, sel_matiere, sel_chapitres)
            
            if not extracted_qi:
                st.error("Aucune Qi reconnue ou hors périmètre.")
            else:
                rows_ext = []
                ok_ext = 0
                known_ids = st.session_state['df_qc']["FRT_ID"].unique()
                
                for item in extracted_qi:
                    frt = item["FRT_ID"]
                    is_known = frt in known_ids
                    if is_known: ok_ext += 1
                    status = "✅ MATCH" if is_known else "❌ GAP"
                    
                    qc_n = "---"
                    frt_n = frt
                    if is_known:
                        info = st.session_state['df_qc'][st.session_state['df_qc']["FRT_ID"]==frt].iloc[0]
                        qc_n = f"{info['QC_ID']} {info['Titre']}"
                    
                    rows_ext.append({"Qi (Enoncé)": item["Qi"], "QC Correspondante": qc_n, "FRT Associé": frt_n, "Statut": status})
                
                taux_ext = (ok_ext / len(extracted_qi)) * 100
                st.markdown(f"### Taux de Couverture : {taux_ext:.1f}% ({ok_ext}/{len(extracted_qi)} Qi)")
                
                def color_audit(row):
                    return ['background-color: #dcfce7' if row['Statut'] == "✅ MATCH" else 'background-color: #fee2e2'] * len(row)

                st.dataframe(pd.DataFrame(rows_ext).style.apply(color_audit, axis=1), use_container_width=True)
                
    else:
        st.info("Veuillez lancer l'usine d'abord.")
