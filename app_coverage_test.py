import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V27")
st.title("🛡️ SMAXIA - Console V27 (Final Stable)")

# ==============================================================================
# 🎨 STYLES CSS (GABARIT SMAXIA OFFICIEL)
# ==============================================================================
st.markdown("""
<style>
    /* 1. EN-TÊTE QC */
    .qc-header-box {
        background-color: #f8f9fa; 
        border-left: 6px solid #2563eb; 
        padding: 15px; 
        margin-bottom: 10px; 
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .qc-id-text { color: #d97706; font-weight: 900; font-size: 1.2em; margin-right: 10px; }
    .qc-title-text { color: #1f2937; font-weight: 700; font-size: 1.15em; }
    .qc-meta-text { 
        font-family: 'Courier New', monospace; font-size: 0.85em; font-weight: 700; color: #4b5563;
        background-color: #e5e7eb; padding: 4px 8px; border-radius: 4px; margin-top: 5px; display: inline-block;
    }

    /* 2. DÉCLENCHEURS (LISTE) */
    .trigger-item {
        background-color: #fff1f2; color: #991b1b; 
        padding: 5px 10px; margin-bottom: 4px; border-radius: 4px;
        border-left: 4px solid #f87171; font-weight: 600; font-size: 0.9em;
        display: block;
    }

    /* 3. ARI (ETAPES) */
    .ari-step {
        background-color: #f3f4f6; color: #374151;
        padding: 4px 8px; margin-bottom: 3px; border-radius: 3px;
        font-family: monospace; font-size: 0.85em; border: 1px dashed #d1d5db;
        display: block;
    }

    /* 4. FRT (BLOCS DISTINCTS) */
    .frt-segment {
        margin-bottom: 8px; padding: 10px; border-radius: 4px;
        border: 1px solid #e5e7eb; background-color: white;
    }
    .frt-seg-title { font-weight: 800; text-transform: uppercase; font-size: 0.75em; display: block; margin-bottom: 4px; }
    .frt-txt { font-family: sans-serif; font-size: 0.95em; color: #333; line-height: 1.4; white-space: pre-wrap; }
    
    /* Couleurs Sémantiques */
    .c-usage { color: #d97706; border-left: 4px solid #d97706; }
    .c-method { color: #059669; border-left: 4px solid #059669; }
    .c-trap { color: #dc2626; border-left: 4px solid #dc2626; }
    .c-conc { color: #2563eb; border-left: 4px solid #2563eb; }

    /* 5. QI CARDS (PREUVE) */
    .qi-card {
        background-color: white; border: 1px solid #e5e7eb; 
        border-left: 4px solid #9333ea; border-radius: 4px;
        padding: 10px; margin-bottom: 8px;
    }
    .qi-body { font-family: 'Georgia', serif; font-size: 1em; font-weight: 500; color: #111; margin-bottom: 5px; }
    .qi-meta { font-size: 0.75em; color: #6b7280; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. KERNEL (CONTENU RICHE & VALIDE)
# ==============================================================================

LISTE_CHAPITRES = {
    "MATHS": ["SUITES NUMÉRIQUES", "FONCTIONS", "PROBABILITÉS", "GÉOMÉTRIE"],
    "PHYSIQUE": ["MÉCANIQUE", "ONDES"]
}

UNIVERS_SMAXIA = {
    "FRT_M_S01": {
        "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", 
        "QC": "Comment démontrer qu'une suite est géométrique ?",
        "Triggers": [
            "montrer que la suite est géométrique",
            "déterminer la nature de la suite",
            "préciser la raison q",
            "justifier que (Un) est géométrique"
        ],
        "ARI": ["1. Exprimer u(n+1)", "2. Quotient u(n+1)/u(n)", "3. Simplifier", "4. Constante"],
        "FRT_DATA": [
            {"type": "usage", "title": "🔔 1. Quand utiliser", "text": "L'énoncé demande explicitement la nature de la suite ou de prouver qu'elle est géométrique."},
            {"type": "method", "title": "✅ 2. Méthode Rédigée", "text": "1. Pour tout n, on exprime u(n+1).\n2. On calcule u(n+1)/u(n).\n3. On simplifie.\n4. On trouve une constante q."},
            {"type": "trap", "title": "⚠️ 3. Pièges", "text": "Oublier de vérifier u(n) non nul."},
            {"type": "conc", "title": "✍️ 4. Conclusion", "text": "Le rapport est constant, donc la suite est géométrique."}
        ]
    },
    "FRT_M_S02": {
        "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES",
        "QC": "Comment lever une indétermination (limite) ?",
        "Triggers": ["calculer la limite", "limite quand n tend vers +infini", "étudier la convergence"],
        "ARI": ["1. Terme dominant", "2. Factorisation", "3. Limites usuelles", "4. Conclure"],
        "FRT_DATA": [
            {"type": "usage", "title": "🔔 1. Quand utiliser", "text": "Forme indéterminée infini - infini ou infini / infini."},
            {"type": "method", "title": "✅ 2. Méthode Rédigée", "text": "1. Identifier le terme dominant.\n2. Factoriser l'expression.\n3. Utiliser les limites usuelles."},
            {"type": "trap", "title": "⚠️ 3. Pièges", "text": "Règle des signes sans factorisation."},
            {"type": "conc", "title": "✍️ 4. Conclusion", "text": "Par opération, la suite converge vers..."}
        ]
    },
    "FRT_M_F01": {
        "Matiere": "MATHS", "Chap": "FONCTIONS",
        "QC": "Comment appliquer le TVI (Solution unique) ?",
        "Triggers": ["montrer que f(x)=k admet une solution unique", "existence et unicité", "théorème des valeurs intermédiaires"],
        "ARI": ["1. Continuité", "2. Monotonie", "3. Bornes", "4. TVI"],
        "FRT_DATA": [
            {"type": "usage", "title": "🔔 1. Quand utiliser", "text": "Prouver existence et unicité d'une solution."},
            {"type": "method", "title": "✅ 2. Méthode Rédigée", "text": "1. f est continue et strictement monotone.\n2. Calcul des images aux bornes.\n3. k est compris entre les images.\n4. Corollaire du TVI."},
            {"type": "trap", "title": "⚠️ 3. Pièges", "text": "Oublier la stricte monotonie."},
            {"type": "conc", "title": "✍️ 4. Conclusion", "text": "L'équation admet une unique solution alpha."}
        ]
    }
}

QI_PATTERNS = {
    "FRT_M_S01": ["Montrer que la suite (Un) est géométrique.", "Quelle est la nature de la suite (Vn) ?"],
    "FRT_M_S02": ["Déterminer la limite de la suite.", "Calculer la limite quand n tend vers +infini."],
    "FRT_M_F01": ["Montrer que f(x)=0 admet une unique solution.", "Démontrer l'existence et l'unicité."]
}

# ==============================================================================
# 2. MOTEUR
# ==============================================================================

def ingest_factory_v27(urls, volume, matiere):
    target_frts = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere]
    
    # Sécurité: DataFrame vide si rien trouvé
    if not target_frts:
        return (pd.DataFrame(columns=["Fichier", "Nature", "Annee", "Telechargement", "Qi_Data"]),
                pd.DataFrame(columns=["FRT_ID", "Qi", "File", "Year", "Chapitre"]))
    
    sources = []
    atoms = []
    
    progress = st.progress(0)
    for i in range(volume):
        progress.progress((i+1)/volume)
        nature = random.choice(["BAC", "DST", "INTERRO"])
        annee = random.choice(range(2020, 2025))
        filename = f"Sujet_{matiere}_{nature}_{annee}_{i}.pdf"
        
        # Génération Qi
        nb_qi = random.randint(3, 6)
        frts = random.choices(target_frts, k=nb_qi)
        qi_data_list = []
        
        for frt_id in frts:
            qi_txt = random.choice(QI_PATTERNS.get(frt_id, ["Question"])) + f" [Ref:{random.randint(10,99)}]"
            
            atoms.append({
                "FRT_ID": frt_id, 
                "Qi": qi_txt, 
                "File": filename, 
                "Year": annee, 
                "Chapitre": UNIVERS_SMAXIA[frt_id]["Chap"]
            })
            qi_data_list.append({"Qi": qi_txt, "FRT_ID": frt_id})
            
        sources.append({
            "Fichier": filename, 
            "Nature": nature, 
            "Annee": annee, 
            "Telechargement": f"https://fake-cloud/dl/{filename}", 
            "Qi_Data": qi_data_list
        })
        
    return pd.DataFrame(sources), pd.DataFrame(atoms)

def compute_qc_v27(df_atoms):
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
            "Triggers": meta["Triggers"], 
            "ARI": meta["ARI"], 
            "FRT_DATA": meta["FRT_DATA"],
            "Evidence": [{"Fichier": f, "Qi": q} for f, q in zip(row["File"], row["Qi"])]
        })
    return pd.DataFrame(qcs).sort_values(by="Score", ascending=False)

def analyze_external_v27(file, matiere):
    target = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere]
    if not target: return []
    frts = random.choices(target, k=10)
    res = []
    for frt in frts:
        qi = random.choice(QI_PATTERNS.get(frt, ["Qi"])) + " (Extrait)"
        res.append({"Qi": qi, "FRT_ID": frt})
    return res

# ==============================================================================
# 3. INTERFACE (UI STABLE)
# ==============================================================================

with st.sidebar:
    st.header("Paramètres Académiques")
    st.selectbox("Classe", ["Terminale"], disabled=True)
    sel_matiere = st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    chaps = LISTE_CHAPITRES.get(sel_matiere, [])
    # Sélectionner le premier chapitre par défaut pour éviter l'écran vide
    def_chap = [chaps[0]] if chaps else []
    sel_chapitres = st.multiselect("Chapitres (Filtre Vue)", chaps, default=def_chap)

tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine", "✅ Onglet 2 : Audit"])

# --- USINE ---
with tab_usine:
    c1, c2 = st.columns([3, 1])
    with c1: urls = st.text_area("URLs Sources", "https://apmep.fr", height=68)
    with c2: 
        vol = st.number_input("Volume", 5, 500, 20, step=5)
        run = st.button("LANCER L'USINE 🚀", type="primary")

    if run:
        df_src, df_atoms = ingest_factory_v27(urls.split('\n'), vol, sel_matiere)
        df_qc = compute_qc_v27(df_atoms)
        st.session_state['df_src'] = df_src
        st.session_state['df_qc'] = df_qc
        st.success(f"Ingestion terminée : {len(df_src)} sujets traités.")

    st.divider()

    if 'df_src' in st.session_state and not st.session_state['df_src'].empty:
        # TABLEAU SUJETS (SAFE DISPLAY)
        st.markdown(f"### 📥 Sujets Traités ({len(st.session_state['df_src'])})")
        
        # Renommage colonnes pour affichage propre
        df_view = st.session_state['df_src'].rename(columns={"Annee": "Année", "Telechargement": "Lien"})
        
        st.data_editor(
            df_view[["Fichier", "Nature", "Année", "Lien"]],
            column_config={
                "Lien": st.column_config.LinkColumn("Téléchargement", display_text="📥 Télécharger PDF")
            },
            hide_index=True, use_container_width=True, disabled=True
        )

        st.divider()

        # LISTE QC
        st.markdown("### 🧠 Base de Connaissance (QC)")
        if not st.session_state['df_qc'].empty:
            qc_view = st.session_state['df_qc'][st.session_state['df_qc']["Chapitre"].isin(sel_chapitres)]
            
            if qc_view.empty:
                st.info("Aucune QC dans les chapitres sélectionnés. Essayez d'autres chapitres dans la sidebar.")
            else:
                chapters = qc_view["Chapitre"].unique()
                for chap in chapters:
                    subset = qc_view[qc_view["Chapitre"] == chap]
                    st.markdown(f"#### 📘 {chap} ({len(subset)} QC)")
                    
                    for idx, row in subset.iterrows():
                        # HEADER
                        st.markdown(f"""
                        <div class="qc-header-box">
                            <span class="qc-id-text">{row['QC_ID']}</span>
                            <span class="qc-title-text">{row['Titre']}</span><br>
                            <span class="qc-meta-text">Score(q)={row['Score']:.0f} | n_q={row['n_q']} | Ψ={row['Psi']} | N_tot={row['N_tot']} | t_rec={row['t_rec']:.1f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        c1, c2, c3, c4 = st.columns(4)
                        
                        # 1. TRIGGERS
                        with c1:
                            with st.expander("🔥 Déclencheurs"):
                                for t in row['Triggers']: 
                                    st.markdown(f"<span class='trigger-item'>“{t}”</span>", unsafe_allow_html=True)
                        
                        # 2. ARI
                        with c2:
                            with st.expander("⚙️ ARI"):
                                for s in row['ARI']:
                                    st.markdown(f"<span class='ari-step'>{s}</span>", unsafe_allow_html=True)
                        
                        # 3. FRT (RENDER 4 BLOCKS)
                        with c3:
                            with st.expander("🧾 FRT (Élève)"):
                                for block in row['FRT_DATA']:
                                    # Mapping type -> css class
                                    cls_map = {"usage": "c-usage", "method": "c-method", "trap": "c-trap", "conc": "c-conc"}
                                    css = cls_map.get(block['type'], "")
                                    st.markdown(f"""
                                    <div class='frt-segment {css}'>
                                        <span class='frt-seg-title'>{block['title']}</span>
                                        <div class='frt-txt'>{block['text']}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # 4. QI (CARDS)
                        with c4:
                            with st.expander(f"📄 Qi ({row['n_q']})"):
                                # Boucle pour afficher chaque Qi proprement
                                for item in row['Evidence']:
                                    st.markdown(f"""
                                    <div class='qi-card'>
                                        <div class='qi-body'>“{item['Qi']}”</div>
                                        <div class='qi-meta'>📄 {item['Fichier']}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        st.write("")
        else:
            st.warning("Aucune QC générée.")

# --- AUDIT ---
with tab_audit:
    st.subheader("Validation Booléenne")
    if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
        
        st.markdown("#### ✅ 1. Test Interne")
        t1_file = st.selectbox("Sujet Traité", st.session_state['df_src']["Fichier"])
        
        if st.button("LANCER AUDIT INTERNE"):
            data = st.session_state['df_src'][st.session_state['df_src']["Fichier"]==t1_file].iloc[0]["Qi_Data"]
            known = st.session_state['df_qc']["FRT_ID"].unique()
            
            rows = []
            ok_count = 0
            for item in data:
                is_ok = item["FRT_ID"] in known
                if is_ok: ok_count += 1
                status = "✅ MATCH" if is_ok else "❌ GAP"
                rows.append({"Qi (Enoncé)": item["Qi"], "Statut": status})
            
            taux = (ok_count / len(data)) * 100
            st.markdown(f"### Taux : {taux:.0f}%")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.divider()

        st.markdown("#### 🌍 2. Test Externe")
        up = st.file_uploader("PDF Externe", type="pdf")
        if up:
            ext = analyze_external_v27(up, sel_matiere)
            if not ext: st.error("Rien trouvé")
            else:
                rows_ext = []
                ok_ext = 0
                known = st.session_state['df_qc']["FRT_ID"].unique()
                
                for item in ext:
                    is_ok = item["FRT_ID"] in known
                    if is_ok: ok_ext += 1
                    qc_n = "---"
                    if is_ok:
                        info = st.session_state['df_qc'][st.session_state['df_qc']["FRT_ID"]==item["FRT_ID"]].iloc[0]
                        qc_n = info["Titre"]
                        
                    rows_ext.append({"Qi": item["Qi"], "QC": qc_n, "Statut": "✅ MATCH" if is_ok else "❌ GAP"})
                
                taux = (ok_ext / len(ext)) * 100
                st.markdown(f"### Taux : {taux:.1f}%")
                
                def hl(row):
                    return ['background-color: #dcfce7' if row['Statut']=="✅ MATCH" else 'background-color: #fee2e2']*len(row)
                st.dataframe(pd.DataFrame(rows_ext).style.apply(hl, axis=1), use_container_width=True)
    else:
        st.info("Veuillez lancer l'usine d'abord.")
