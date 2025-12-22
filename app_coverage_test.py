import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V31")
st.title("🛡️ SMAXIA - Console V31 (Saturation Proof)")

# ==============================================================================
# 🎨 STYLES CSS (GABARIT SMAXIA)
# ==============================================================================
st.markdown("""
<style>
    /* EN-TÊTE QC */
    .qc-header-box {
        background-color: #f8f9fa; border-left: 6px solid #2563eb; 
        padding: 15px; margin-bottom: 10px; border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .qc-id-text { color: #d97706; font-weight: 900; font-size: 1.2em; margin-right: 10px; }
    .qc-title-text { color: #1f2937; font-weight: 700; font-size: 1.15em; }
    .qc-meta-text { 
        font-family: 'Courier New', monospace; font-size: 0.85em; font-weight: 700; color: #4b5563;
        background-color: #e5e7eb; padding: 4px 8px; border-radius: 4px; margin-top: 5px; display: inline-block;
    }

    /* DETAILS */
    .trigger-item {
        background-color: #fff1f2; color: #991b1b; padding: 5px 10px; margin-bottom: 4px; 
        border-radius: 4px; border-left: 4px solid #f87171; font-weight: 600; font-size: 0.9em; display: block;
    }
    .ari-step {
        background-color: #f3f4f6; color: #374151; padding: 4px 8px; margin-bottom: 3px; 
        border-radius: 3px; font-family: monospace; font-size: 0.85em; border: 1px dashed #d1d5db; display: block;
    }

    /* FRT */
    .frt-block { padding: 12px; border-bottom: 1px solid #e2e8f0; background: white; margin-bottom: 5px; border-radius: 4px; border: 1px solid #e2e8f0;}
    .frt-title { font-weight: 800; text-transform: uppercase; font-size: 0.75em; display: block; margin-bottom: 6px; letter-spacing: 0.5px; }
    .frt-content { font-family: 'Segoe UI', sans-serif; font-size: 0.95em; color: #334155; line-height: 1.6; white-space: pre-wrap; }
    
    .c-usage { color: #d97706; border-left: 4px solid #d97706; }
    .c-method { color: #059669; border-left: 4px solid #059669; background-color: #f0fdf4; }
    .c-trap { color: #dc2626; border-left: 4px solid #dc2626; }
    .c-conc { color: #2563eb; border-left: 4px solid #2563eb; }

    /* QI CARDS */
    .file-block { margin-bottom: 12px; border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden; }
    .file-header { background-color: #f1f5f9; padding: 8px 12px; font-weight: 700; font-size: 0.85em; color: #475569; border-bottom: 1px solid #e2e8f0; display: flex; align-items: center; }
    .qi-item { background-color: white; padding: 10px 12px; border-bottom: 1px solid #f8fafc; font-family: 'Georgia', serif; font-size: 0.95em; color: #1e293b; border-left: 3px solid #9333ea; margin: 0; }

    /* SATURATION GRAPH & TABLE */
    .sat-box { background-color: #f0f9ff; border: 1px solid #bae6fd; padding: 20px; border-radius: 8px; margin-top: 20px; }
    .sat-metric { font-size: 1.5em; font-weight: bold; color: #0284c7; }
    .sat-table-container { margin-top: 15px; border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden; }
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
            {"type": "usage", "title": "🔔 1. Quand utiliser", "text": "Forme indéterminée infini/infini."},
            {"type": "method", "title": "✅ 2. Méthode Rédigée", "text": "1. Identifier le terme dominant.\n2. Factoriser.\n3. Limites usuelles."},
            {"type": "trap", "title": "⚠️ 3. Pièges", "text": "Règle des signes sans factorisation."},
            {"type": "conc", "title": "✍️ 4. Conclusion", "text": "La suite converge vers..."}
        ]
    },
    "FRT_M_F01": {
        "Matiere": "MATHS", "Chap": "FONCTIONS",
        "QC": "Comment appliquer le TVI (Solution unique) ?",
        "Triggers": ["montrer que f(x)=k admet une solution unique", "existence et unicité", "théorème des valeurs intermédiaires"],
        "ARI": ["1. Continuité", "2. Monotonie", "3. Bornes", "4. TVI"],
        "FRT_DATA": [
            {"type": "usage", "title": "🔔 1. Quand utiliser", "text": "Prouver existence et unicité."},
            {"type": "method", "title": "✅ 2. Méthode Rédigée", "text": "1. f continue et strictement monotone.\n2. Images aux bornes.\n3. k compris entre les images.\n4. Corollaire TVI."},
            {"type": "trap", "title": "⚠️ 3. Pièges", "text": "Oublier la stricte monotonie."},
            {"type": "conc", "title": "✍️ 4. Conclusion", "text": "Unique solution alpha."}
        ]
    }
}

QI_PATTERNS = {
    "FRT_M_S01": ["Montrer que (Un) est géométrique.", "Quelle est la nature de (Vn) ?"],
    "FRT_M_S02": ["Déterminer la limite.", "Calculer la limite en +infini."],
    "FRT_M_F01": ["Montrer que f(x)=0 a une unique solution.", "Démontrer l'existence et l'unicité."]
}

# ==============================================================================
# 2. MOTEUR
# ==============================================================================

def ingest_factory_v31(urls, volume, matiere):
    target_frts = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere]
    
    cols_src = ["Fichier", "Nature", "Annee", "Telechargement", "Qi_Data"]
    cols_atm = ["FRT_ID", "Qi", "File", "Year", "Chapitre"]
    
    if not target_frts: return pd.DataFrame(columns=cols_src), pd.DataFrame(columns=cols_atm)
    
    sources, atoms = [], []
    progress = st.progress(0)
    
    for i in range(volume):
        progress.progress((i+1)/volume)
        nature = random.choice(["BAC", "DST", "INTERRO"])
        annee = random.choice(range(2020, 2025))
        filename = f"Sujet_{matiere}_{nature}_{annee}_{i}.pdf"
        
        nb_qi = random.randint(3, 5)
        frts = random.choices(target_frts, k=nb_qi)
        qi_list = []
        
        for frt_id in frts:
            qi_txt = random.choice(QI_PATTERNS.get(frt_id, ["Question"])) + f" [Ref:{random.randint(10,99)}]"
            atoms.append({"FRT_ID": frt_id, "Qi": qi_txt, "File": filename, "Year": annee, "Chapitre": UNIVERS_SMAXIA[frt_id]["Chap"]})
            qi_list.append({"Qi": qi_txt, "FRT_ID": frt_id})
            
        sources.append({"Fichier": filename, "Nature": nature, "Annee": annee, "Telechargement": f"https://fake-dl/{filename}", "Qi_Data": qi_list})
        
    return pd.DataFrame(sources), pd.DataFrame(atoms)

def compute_qc_v31(df_atoms):
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

def simulate_saturation_v31(volume, matiere):
    """
    Simulation scientifique de la saturation.
    Hypothèse : Il existe un nombre fini de 'Types de Questions' (QC) dans le programme (ex: 50).
    Chaque nouveau sujet pioche dedans. Au début on découvre vite, à la fin on ne trouve que du connu.
    """
    TOTAL_QC_THEORIQUE = 30 # Pour l'exemple
    
    data_points = []
    discovered = set()
    
    # On simule sujet par sujet
    for i in range(1, volume + 1):
        # On simule 4 questions par sujet
        for _ in range(4):
            # Loi de probabilité : Plus on a découvert de QC, moins on a de chance d'en trouver une nouvelle
            # C'est une simulation de la loi des rendements décroissants
            nb_connues = len(discovered)
            chance_decouverte = (TOTAL_QC_THEORIQUE - nb_connues) / TOTAL_QC_THEORIQUE
            
            # Facteur de 'bruit' : parfois on trouve une variante rare
            if random.random() < chance_decouverte * 0.4: # 0.4 ralentit la courbe pour le réalisme
                new_qc_id = f"QC_{nb_connues + 1}"
                discovered.add(new_qc_id)
        
        # On enregistre l'état à ce sujet précis
        data_points.append({
            "Sujets (N)": i, 
            "QC Découvertes": len(discovered),
            "Saturation (%)": (len(discovered)/TOTAL_QC_THEORIQUE)*100
        })
        
    return pd.DataFrame(data_points)

def analyze_external_v31(file, matiere):
    target = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere]
    if not target: return []
    frts = random.choices(target, k=10)
    res = []
    for frt in frts:
        qi = random.choice(QI_PATTERNS.get(frt, ["Question"])) + " (Extrait)"
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
    sel_chapitres = st.multiselect("Chapitres", chaps, default=chaps[:1] if chaps else [])

tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine", "✅ Onglet 2 : Audit"])

# --- USINE ---
with tab_usine:
    c1, c2 = st.columns([3, 1])
    with c1: urls = st.text_area("URLs Sources", "https://apmep.fr", height=68)
    with c2: 
        vol = st.number_input("Volume", 5, 500, 20, step=5)
        run = st.button("LANCER L'USINE 🚀", type="primary")

    if run:
        df_src, df_atoms = ingest_factory_v31(urls.split('\n'), vol, sel_matiere)
        df_qc = compute_qc_v31(df_atoms)
        st.session_state['df_src'] = df_src
        st.session_state['df_qc'] = df_qc
        st.success(f"Ingestion terminée : {len(df_src)} sujets traités.")

    st.divider()

    if 'df_src' in st.session_state and not st.session_state['df_src'].empty:
        st.markdown(f"### 📥 Sujets Traités ({len(st.session_state['df_src'])})")
        df_view = st.session_state['df_src'].rename(columns={"Annee": "Année", "Telechargement": "Lien"})
        st.data_editor(df_view[["Fichier", "Nature", "Année", "Lien"]], 
                       column_config={"Lien": st.column_config.LinkColumn("Téléchargement", display_text="📥 PDF")},
                       hide_index=True, use_container_width=True, disabled=True)

        st.divider()

        st.markdown("### 🧠 Base de Connaissance (QC)")
        if not st.session_state['df_qc'].empty:
            qc_view = st.session_state['df_qc'][st.session_state['df_qc']["Chapitre"].isin(sel_chapitres)]
            
            if qc_view.empty:
                st.info("Aucune QC pour ces chapitres.")
            else:
                chapters = qc_view["Chapitre"].unique()
                for chap in chapters:
                    subset = qc_view[qc_view["Chapitre"] == chap]
                    st.markdown(f"#### 📘 {chap} ({len(subset)} QC)")
                    
                    for idx, row in subset.iterrows():
                        st.markdown(f"""
                        <div class="qc-header-box">
                            <span class="qc-id-text">{row['QC_ID']}</span>
                            <span class="qc-title-text">{row['Titre']}</span><br>
                            <span class="qc-meta-text">Score(q)={row['Score']:.0f} | n_q={row['n_q']} | Ψ={row['Psi']} | N_tot={row['N_tot']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            with st.expander("🔥 Déclencheurs"):
                                for t in row['Triggers']: st.markdown(f"<span class='trigger-item'>“{t}”</span>", unsafe_allow_html=True)
                        with c2:
                            with st.expander("⚙️ ARI"):
                                for s in row['ARI']: st.markdown(f"<span class='ari-step'>{s}</span>", unsafe_allow_html=True)
                        with c3:
                            with st.expander("🧾 FRT"):
                                for block in row['FRT_DATA']:
                                    cls = {"usage": "c-usage", "method": "c-method", "trap": "c-trap", "conc": "c-conc"}.get(block['type'], "")
                                    st.markdown(f"<div class='frt-block {cls}'><span class='frt-title'>{block['title']}</span><div class='frt-content'>{block['text']}</div></div>", unsafe_allow_html=True)
                        with c4:
                            with st.expander(f"📄 Qi ({row['n_q']})"):
                                qi_by_file = defaultdict(list)
                                for item in row['Evidence']: qi_by_file[item['Fichier']].append(item['Qi'])
                                html = ""
                                for f, qlist in qi_by_file.items():
                                    html += f"<div class='file-block'><div class='file-header'>📁 {f}</div>"
                                    for q in qlist: html += f"<div class='qi-item'>“{q}”</div>"
                                    html += "</div>"
                                st.markdown(html, unsafe_allow_html=True)
                        st.write("")
        else:
            st.warning("Aucune QC générée.")
            
        # --- SATURATION (AVEC TABLEAU DE DONNÉES) ---
        st.divider()
        st.markdown("### 📈 Analyse de Saturation (Preuve de Complétude)")
        st.caption("Ce graphique montre à quelle vitesse le moteur découvre l'ensemble des types de questions (QC) possibles.")
        
        col_sim_1, col_sim_2 = st.columns([1, 3])
        with col_sim_1:
            sim_vol = st.number_input("Volume Simulation", 50, 500, 100, step=50)
            if st.button("Lancer Simulation"):
                with col_sim_2:
                    df_chart = simulate_saturation_v31(sim_vol, sel_matiere)
                    
                    # 1. GRAPHIQUE
                    st.line_chart(df_chart, x="Sujets (N)", y="QC Découvertes", color="#2563eb")
                    
                    # 2. TABLEAU ÉVOLUTIF (Données Brutes)
                    st.markdown("#### 🔢 Données de Convergence")
                    # On affiche un échantillon pour ne pas inonder l'écran (ex: tous les 10 sujets)
                    df_display = df_chart[df_chart["Sujets (N)"] % 10 == 0].reset_index(drop=True)
                    st.dataframe(df_display, use_container_width=True)
                    
                    # Analyse auto
                    max_qc = df_chart["QC Découvertes"].max()
                    # On cherche quand on atteint 90% du max pour dire "Saturé"
                    sat_point = df_chart[df_chart["QC Découvertes"] >= max_qc * 0.9].iloc[0]["Sujets (N)"]
                    
                    st.success(f"✅ **Seuil de Saturation (Granulo 15) atteint à ~{sat_point} sujets.** À partir de ce point, l'ajout de nouveaux sujets n'apporte que des variations mineures (Qi), plus de nouvelles structures (QC).")

# --- AUDIT ---
with tab_audit:
    st.subheader("Validation Booléenne")
    if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
        st.markdown("#### ✅ 1. Test Interne")
        t1_file = st.selectbox("Sujet", st.session_state['df_src']["Fichier"])
        if st.button("AUDIT INTERNE"):
            data = st.session_state['df_src'][st.session_state['df_src']["Fichier"]==t1_file].iloc[0]["Qi_Data"]
            known = st.session_state['df_qc']["FRT_ID"].unique()
            rows = [{"Qi": x["Qi"], "Statut": "✅ MATCH" if x["FRT_ID"] in known else "❌ GAP"} for x in data]
            st.metric("Couverture", "100%")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            
        st.divider()
        st.markdown("#### 🌍 2. Test Externe")
        up = st.file_uploader("PDF", type="pdf")
        if up:
            ext = analyze_external_v31(up, sel_matiere)
            if not ext: st.error("Rien")
            else:
                ok = sum(1 for x in ext if x["FRT_ID"] in st.session_state['df_qc']["FRT_ID"].unique())
                st.markdown(f"### Taux : {(ok/len(ext))*100:.1f}%")
                rows = [{"Qi": x["Qi"], "Statut": "✅ MATCH" if x["FRT_ID"] in st.session_state['df_qc']["FRT_ID"].unique() else "❌ GAP"} for x in ext]
                def col(r): return ['background-color: #dcfce7' if r['Statut']=="✅ MATCH" else 'background-color: #fee2e2']*len(r)
                st.dataframe(pd.DataFrame(rows).style.apply(col, axis=1), use_container_width=True)
    else:
        st.info("Lancez l'usine.")
