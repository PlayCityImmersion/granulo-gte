import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V28")
st.title("🛡️ SMAXIA - Console V28 (Grouped Evidence)")

# ==============================================================================
# 🎨 STYLES CSS (GABARIT SMAXIA - PRESERVED)
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

    /* 2. DÉCLENCHEURS */
    .trigger-item {
        background-color: #fff1f2; color: #991b1b; 
        padding: 5px 10px; margin-bottom: 4px; border-radius: 4px;
        border-left: 4px solid #f87171; font-weight: 600; font-size: 0.9em;
        display: block;
    }

    /* 3. ARI */
    .ari-step {
        background-color: #f3f4f6; color: #374151;
        padding: 4px 8px; margin-bottom: 3px; border-radius: 3px;
        font-family: monospace; font-size: 0.85em; border: 1px dashed #d1d5db;
        display: block;
    }

    /* 4. FRT */
    .frt-segment {
        margin-bottom: 8px; padding: 10px; border-radius: 4px;
        border: 1px solid #e5e7eb; background-color: white;
    }
    .frt-seg-title { font-weight: 800; text-transform: uppercase; font-size: 0.75em; display: block; margin-bottom: 4px; }
    .frt-txt { font-family: sans-serif; font-size: 0.95em; color: #333; line-height: 1.4; white-space: pre-wrap; }
    .c-usage { color: #d97706; border-left: 4px solid #d97706; }
    .c-method { color: #059669; border-left: 4px solid #059669; }
    .c-trap { color: #dc2626; border-left: 4px solid #dc2626; }
    .c-conc { color: #2563eb; border-left: 4px solid #2563eb; }

    /* 5. QI CARDS (GROUPED) */
    .file-block {
        margin-bottom: 12px; border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden;
    }
    .file-header {
        background-color: #f3f4f6; padding: 8px 12px; font-weight: 700; font-size: 0.85em;
        color: #4b5563; border-bottom: 1px solid #e5e7eb; display: flex; align-items: center;
    }
    .qi-item {
        background-color: white; padding: 10px 12px; border-bottom: 1px solid #f9fafb;
        font-family: 'Georgia', serif; font-size: 0.95em; color: #111;
        border-left: 3px solid #9333ea; margin: 5px; border-radius: 0 4px 4px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. KERNEL
# ==============================================================================

LISTE_CHAPITRES = {
    "MATHS": ["SUITES NUMÉRIQUES", "FONCTIONS", "PROBABILITÉS", "GÉOMÉTRIE"],
    "PHYSIQUE": ["MÉCANIQUE", "ONDES"]
}

UNIVERS_SMAXIA = {
    "FRT_M_S01": {
        "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", 
        "QC": "Comment démontrer qu'une suite est géométrique ?",
        "Triggers": ["montrer que la suite est géométrique", "déterminer la nature de la suite", "préciser la raison q", "justifier que (Un) est géométrique"],
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

def ingest_factory_v28(urls, volume, matiere):
    target_frts = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere]
    if not target_frts:
        return (pd.DataFrame(columns=["Fichier", "Nature", "Annee", "Telechargement", "Qi_Data"]),
                pd.DataFrame(columns=["FRT_ID", "Qi", "File", "Year", "Chapitre"]))
    
    sources, atoms = [], []
    progress = st.progress(0)
    for i in range(volume):
        progress.progress((i+1)/volume)
        nature = random.choice(["BAC", "DST", "INTERRO"])
        annee = random.choice(range(2020, 2025))
        filename = f"Sujet_{matiere}_{nature}_{annee}_{i}.pdf"
        
        nb_qi = random.randint(3, 6)
        frts = random.choices(target_frts, k=nb_qi)
        qi_data_list = []
        for frt_id in frts:
            qi_txt = random.choice(QI_PATTERNS.get(frt_id, ["Question"])) + f" [Ref:{random.randint(10,99)}]"
            atoms.append({"FRT_ID": frt_id, "Qi": qi_txt, "File": filename, "Year": annee, "Chapitre": UNIVERS_SMAXIA[frt_id]["Chap"]})
            qi_data_list.append({"Qi": qi_txt, "FRT_ID": frt_id})
            
        sources.append({
            "Fichier": filename, "Nature": nature, "Annee": annee, 
            "Telechargement": f"https://fake-cloud/dl/{filename}", "Qi_Data": qi_data_list
        })
        
    return pd.DataFrame(sources), pd.DataFrame(atoms)

def compute_qc_v28(df_atoms):
    if df_atoms.empty: return pd.DataFrame()
    grouped = df_atoms.groupby("FRT_ID").agg({"Qi": list, "File": list, "Year": "max", "Chapitre": "first"}).reset_index()
    qcs = []
    N_tot = len(df_atoms)
    
    for idx, row in grouped.iterrows():
        meta = UNIVERS_SMAXIA[row["FRT_ID"]]
        n_q = len(row["Qi"])
        t_rec = max(datetime.now().year - row["Year"], 0.5)
        psi = 0.85
        score = (n_q / N_tot) * (1 + 5.0/t_rec) * psi * 100
        
        qcs.append({
            "Chapitre": row["Chapitre"], "QC_ID": f"QC-{idx+1:02d}", "FRT_ID": row["FRT_ID"],
            "Titre": meta["QC"], "Score": score, "n_q": n_q, "Psi": psi, "N_tot": N_tot, "t_rec": t_rec,
            "Triggers": meta["Triggers"], "ARI": meta["ARI"], "FRT_DATA": meta["FRT_DATA"],
            "Evidence": [{"Fichier": f, "Qi": q} for f, q in zip(row["File"], row["Qi"])]
        })
    return pd.DataFrame(qcs).sort_values(by="Score", ascending=False)

def analyze_external_v28(file, matiere):
    target = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere]
    if not target: return []
    frts = random.choices(target, k=10)
    res = []
    for frt in frts:
        qi = random.choice(QI_PATTERNS.get(frt, ["Qi"])) + " (Extrait)"
        res.append({"Qi": qi, "FRT_ID": frt})
    return res

# ==============================================================================
# 3. INTERFACE
# ==============================================================================

with st.sidebar:
    st.header("Paramètres Académiques")
    st.selectbox("Classe", ["Terminale"], disabled=True)
    sel_matiere = st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    chaps = LISTE_CHAPITRES.get(sel_matiere, [])
    sel_chapitres = st.multiselect("Chapitres (Filtre Vue)", chaps, default=chaps[:1] if chaps else [])

tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine", "✅ Onglet 2 : Audit"])

# --- USINE ---
with tab_usine:
    c1, c2 = st.columns([3, 1])
    with c1: urls = st.text_area("URLs Sources", "https://apmep.fr", height=68)
    with c2: 
        vol = st.number_input("Volume", 5, 500, 20, step=5)
        run = st.button("LANCER L'USINE 🚀", type="primary")

    if run:
        df_src, df_atoms = ingest_factory_v28(urls.split('\n'), vol, sel_matiere)
        df_qc = compute_qc_v28(df_atoms)
        st.session_state['df_src'] = df_src
        st.session_state['df_qc'] = df_qc
        st.success(f"Ingestion terminée : {len(df_src)} sujets traités.")

    st.divider()

    if 'df_src' in st.session_state and not st.session_state['df_src'].empty:
        st.markdown(f"### 📥 Sujets Traités ({len(st.session_state['df_src'])})")
        df_view = st.session_state['df_src'].rename(columns={"Annee": "Année", "Telechargement": "Lien"})
        st.data_editor(
            df_view[["Fichier", "Nature", "Année", "Lien"]],
            column_config={"Lien": st.column_config.LinkColumn("Téléchargement", display_text="📥 Télécharger PDF")},
            hide_index=True, use_container_width=True, disabled=True
        )

        st.divider()

        st.markdown("### 🧠 Base de Connaissance (QC)")
        if not st.session_state['df_qc'].empty:
            qc_view = st.session_state['df_qc'][st.session_state['df_qc']["Chapitre"].isin(sel_chapitres)]
            
            if qc_view.empty:
                st.info("Aucune QC dans les chapitres sélectionnés.")
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
                        
                        # 3. FRT
                        with c3:
                            with st.expander("🧾 FRT (Élève)"):
                                for block in row['FRT_DATA']:
                                    cls_map = {"usage": "c-usage", "method": "c-method", "trap": "c-trap", "conc": "c-conc"}
                                    css = cls_map.get(block['type'], "")
                                    st.markdown(f"""
                                    <div class='frt-segment {css}'>
                                        <span class='frt-seg-title'>{block['title']}</span>
                                        <div class='frt-txt'>{block['text']}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # 4. QI (GROUPÉ PAR FICHIER)
                        with c4:
                            with st.expander(f"📄 Qi ({row['n_q']})"):
                                # LOGIQUE DE REGROUPEMENT
                                qi_by_file = defaultdict(list)
                                for item in row['Evidence']:
                                    qi_by_file[item['Fichier']].append(item['Qi'])
                                
                                # GENERATION HTML
                                html_qi = ""
                                for fname, qilist in qi_by_file.items():
                                    html_qi += f"<div class='file-block'><div class='file-header'>📁 {fname}</div>"
                                    for q in qilist:
                                        html_qi += f"<div class='qi-item'>“{q}”</div>"
                                    html_qi += "</div>"
                                st.markdown(html_qi, unsafe_allow_html=True)
                        st.write("")
        else:
            st.warning("Aucune QC générée.")

# --- AUDIT ---
with tab_audit:
    st.subheader("Validation Booléenne")
    if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
        st.markdown("#### ✅ 1. Test Interne")
        t1_file = st.selectbox("Sujet Traité", st.session_state['df_src']["Fichier"])
        if st.button("LANCER AUDIT"):
            data = st.session_state['df_src'][st.session_state['df_src']["Fichier"]==t1_file].iloc[0]["Qi_Data"]
            known = st.session_state['df_qc']["FRT_ID"].unique()
            ok = sum(1 for x in data if x["FRT_ID"] in known)
            st.metric("Couverture", f"{(ok/len(data))*100:.0f}%")
            
        st.divider()
        st.markdown("#### 🌍 2. Test Externe")
        up = st.file_uploader("PDF Externe", type="pdf")
        if up:
            ext = analyze_external_v28(up, sel_matiere)
            if not ext: st.error("Rien trouvé")
            else:
                ok = sum(1 for x in ext if x["FRT_ID"] in st.session_state['df_qc']["FRT_ID"].unique())
                st.markdown(f"### Taux : {(ok/len(ext))*100:.1f}%")
                rows = []
                for item in ext:
                    status = "✅ MATCH" if item["FRT_ID"] in st.session_state['df_qc']["FRT_ID"].unique() else "❌ GAP"
                    rows.append({"Qi": item["Qi"], "Statut": status})
                def color(row):
                    return ['background-color: #dcfce7' if row['Statut']=="✅ MATCH" else 'background-color: #fee2e2']*len(row)
                st.dataframe(pd.DataFrame(rows).style.apply(color, axis=1), use_container_width=True)
    else:
        st.info("Lancez l'usine.")
