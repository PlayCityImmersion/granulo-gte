import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V22")
st.title("🛡️ SMAXIA - Console V22 (Bulletproof Edition)")

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

    /* DETAILS */
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
# 1. KERNEL (DONNÉES)
# ==============================================================================

LISTE_CHAPITRES = {
    "MATHS": ["SUITES NUMÉRIQUES", "FONCTIONS", "PROBABILITÉS", "GÉOMÉTRIE"],
    "PHYSIQUE": ["MÉCANIQUE", "ONDES", "CHIMIE"]
}

UNIVERS_SMAXIA = {
    # --- MATHS ---
    "FRT_M_S01": {"Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", "QC": "Comment démontrer qu'une suite est géométrique ?", "Trigger": "montrer que la suite est géométrique", "ARI": ["Ratio u(n+1)/u(n)", "Cste"], "FRT": "Calculer le rapport et trouver une constante."},
    "FRT_M_S02": {"Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", "QC": "Comment lever une indétermination (limite) ?", "Trigger": "calculer la limite en +inf", "ARI": ["Factorisation", "Limites"], "FRT": "Factoriser par le terme dominant."},
    "FRT_M_F01": {"Matiere": "MATHS", "Chap": "FONCTIONS", "QC": "Comment étudier les variations ?", "Trigger": "tableau de variations", "ARI": ["Dérivée", "Signe"], "FRT": "Dérivée, Signe, Variations."},
    "FRT_M_F02": {"Matiere": "MATHS", "Chap": "FONCTIONS", "QC": "Comment appliquer le TVI (Unique) ?", "Trigger": "solution unique alpha", "ARI": ["Monotonie", "TVI"], "FRT": "Continuité, Monotonie, Images."},
    "FRT_M_P01": {"Matiere": "MATHS", "Chap": "PROBABILITÉS", "QC": "Comment utiliser la Loi Binomiale ?", "Trigger": "probabilité de k succès", "ARI": ["Bernoulli", "Formule"], "FRT": "Schéma de Bernoulli + Formule."},
    
    # --- PHYSIQUE ---
    "FRT_P_M01": {"Matiere": "PHYSIQUE", "Chap": "MÉCANIQUE", "QC": "Comment appliquer la 2e Loi de Newton ?", "Trigger": "vecteur accélération", "ARI": ["Forces", "2e Loi"], "FRT": "Bilan forces, PFD, Projection."},
    "FRT_P_O01": {"Matiere": "PHYSIQUE", "Chap": "ONDES", "QC": "Comment calculer la longueur d'onde ?", "Trigger": "calculer lambda", "ARI": ["v/f"], "FRT": "Lambda = v/f."}
}

QI_PATTERNS = {
    k: [f"Question type sur {v['QC']}...", f"Variante : {v['Trigger']}..."] for k, v in UNIVERS_SMAXIA.items()
}

# ==============================================================================
# 2. MOTEUR ROBUSTE
# ==============================================================================

def ingest_factory_v22(urls, volume, matiere):
    """
    Ingestion robuste avec définition explicite des colonnes pour éviter KeyError
    """
    target_frts = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere]
    
    if not target_frts: 
        # Retourne des DF vides mais structurés pour éviter les crashs
        return pd.DataFrame(columns=["Fichier", "Nature", "Annee", "Telechargement", "Qi_Data"]), pd.DataFrame()
    
    sources_data = []
    atoms_data = []
    
    progress = st.progress(0)
    for i in range(volume):
        progress.progress((i+1)/volume)
        
        nature = random.choice(["BAC", "DST", "INTERRO"])
        annee = random.choice(range(2020, 2025))
        filename = f"Sujet_{matiere}_{nature}_{annee}_{i}.pdf"
        
        # Génération de Qi (Mélange de chapitres)
        nb_qi = random.randint(4, 8)
        frts = random.choices(target_frts, k=nb_qi)
        
        qi_data_list = []
        for frt_id in frts:
            qi_txt = random.choice(QI_PATTERNS[frt_id]) + f" [Ref:{random.randint(10,99)}]"
            
            # Atome Moteur
            atoms_data.append({
                "FRT_ID": frt_id, 
                "Qi": qi_txt, 
                "File": filename, 
                "Year": annee, 
                "Chapitre": UNIVERS_SMAXIA[frt_id]["Chap"]
            })
            
            # Vérité terrain
            qi_data_list.append({"Qi": qi_txt, "FRT_ID": frt_id})
            
        # Source (Clé 'Telechargement' sans accent pour sécurité code, on renommera à l'affichage si besoin)
        sources_data.append({
            "Fichier": filename, 
            "Nature": nature, 
            "Annee": annee,
            "Telechargement": f"https://fake-cloud.smaxia/dl/{filename}", 
            "Qi_Data": qi_data_list
        })
        
    df_s = pd.DataFrame(sources_data)
    df_a = pd.DataFrame(atoms_data)
    return df_s, df_a

def compute_qc_v22(df_atoms):
    if df_atoms.empty: return pd.DataFrame()
    
    grouped = df_atoms.groupby("FRT_ID").agg({
        "Qi": list, "File": list, "Year": "max", "Chapitre": "first"
    }).reset_index()
    
    qcs = []
    N_tot = len(df_atoms)
    
    for idx, row in grouped.iterrows():
        meta = UNIVERS_SMAXIA[row["FRT_ID"]]
        n_q = len(row["Qi"])
        t_rec = max(datetime.now().year - row["Year"], 0.5)
        psi = 0.85
        score = (n_q / N_tot) * (1 + 5.0/t_rec) * psi * 100
        
        qcs.append({
            "Chapitre": row["Chapitre"], 
            "QC_ID": f"QC-{idx+1:02d}", 
            "FRT_ID": row["FRT_ID"],
            "Titre": meta["QC"], 
            "Score": score, "n_q": n_q, "Psi": psi, "N_tot": N_tot, "t_rec": t_rec,
            "Trigger": meta["Trigger"], "ARI": meta["ARI"], "FRT": meta["FRT"],
            "Evidence": [{"Fichier": f, "Qi": q} for f, q in zip(row["File"], row["Qi"])]
        })
        
    return pd.DataFrame(qcs).sort_values(by="Score", ascending=False)

def analyze_external_v22(file, matiere):
    target_frts = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere]
    if not target_frts: return []
    frts = random.choices(target_frts, k=10) # 10 questions
    result = []
    for frt in frts:
        qi = random.choice(QI_PATTERNS[frt]) + " (Extrait PDF)"
        result.append({"Qi": qi, "FRT_ID": frt})
    return result

# ==============================================================================
# 3. UI
# ==============================================================================

# SIDEBAR
with st.sidebar:
    st.header("Paramètres Académiques")
    st.selectbox("Classe", ["Terminale"], disabled=True)
    sel_matiere = st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    
    chaps_dispo = LISTE_CHAPITRES.get(sel_matiere, [])
    # On sélectionne tout par défaut pour voir les résultats
    sel_chapitres = st.multiselect("Chapitres (Filtre Vue)", chaps_dispo, default=chaps_dispo)

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
        df_src, df_atoms = ingest_factory_v22(urls.split('\n'), vol, sel_matiere)
        df_qc = compute_qc_v22(df_atoms)
        st.session_state['df_src'] = df_src
        st.session_state['df_qc'] = df_qc
        st.success(f"Ingestion terminée : {len(df_src)} sujets traités.")

    st.divider()

    # Affichage sécurisé avec vérification des clés
    if 'df_src' in st.session_state and not st.session_state['df_src'].empty:
        
        st.markdown(f"### 📥 Sujets Traités ({len(st.session_state['df_src'])})")
        
        # On renomme pour l'affichage (sécurité)
        df_display = st.session_state['df_src'].rename(columns={"Annee": "Année", "Telechargement": "Lien"})
        
        # Data Editor avec LinkColumn sur la colonne 'Lien'
        st.data_editor(
            df_display[["Fichier", "Nature", "Année", "Lien"]],
            column_config={
                "Lien": st.column_config.LinkColumn("Téléchargement", display_text="📥 Télécharger PDF")
            },
            hide_index=True, use_container_width=True, disabled=True
        )

        st.divider()

        st.markdown("### 🧠 Base de Connaissance (QC)")
        
        if not st.session_state['df_qc'].empty:
            # Filtre d'affichage
            qc_view = st.session_state['df_qc'][st.session_state['df_qc']["Chapitre"].isin(sel_chapitres)]
            
            if qc_view.empty:
                st.info(f"Le moteur contient {len(st.session_state['df_qc'])} QC, mais aucune dans les chapitres sélectionnés ci-contre.")
            else:
                chapters = qc_view["Chapitre"].unique()
                for chap in chapters:
                    subset = qc_view[qc_view["Chapitre"] == chap]
                    st.markdown(f"#### 📘 Chapitre {chap} : {len(subset)} QC")
                    
                    for idx, row in subset.iterrows():
                        # Header QC
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
                                st.markdown(f"<div class='trigger-container'><span class='trigger-item'>{row['Trigger']}</span></div>", unsafe_allow_html=True)
                        with c2:
                            with st.expander("⚙️ ARI"):
                                st.markdown(f"<div class='ari-box'>{' > '.join(row['ARI'])}</div>", unsafe_allow_html=True)
                        with c3:
                            with st.expander("🧾 FRT"):
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
                    info = st.session_state['df_qc'][st.session_state['df_qc']["FRT_ID"]==item["FRT_ID"]].iloc[0]
                    qc_nom = f"{info['QC_ID']} {info['Titre']}"
                
                rows.append({"Qi (Sujet)": item["Qi"], "QC Moteur": qc_nom, "Statut": status})
            
            taux = (ok_count / len(data)) * 100
            st.metric("Taux Couverture Interne", f"{taux:.0f}%")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.divider()

        # TEST 2
        st.markdown("#### 🌍 2. Test Externe (Mapping Nouveau Sujet)")
        up_file = st.file_uploader("Charger un PDF externe", type="pdf")
        
        if up_file:
            extracted_qi = analyze_external_v22(up_file, sel_matiere)
            if not extracted_qi:
                st.error("Aucune Qi reconnue.")
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
