import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V30")
st.title("🛡️ SMAXIA - Console V30 (Saturation & FRT Master)")

# ==============================================================================
# 🎨 STYLES CSS (GABARIT CHIRURGICAL)
# ==============================================================================
st.markdown("""
<style>
    /* 1. EN-TÊTE QC */
    .qc-header-box {
        background-color: #f8f9fa; 
        border-left: 6px solid #2563eb; 
        padding: 15px; margin-bottom: 10px; border-radius: 4px;
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

    /* 4. FRT (BLOCS DISTINCTS & PROPRES) */
    .frt-container { 
        border: 1px solid #e2e8f0; border-radius: 6px; overflow: hidden; margin-top: 5px;
    }
    .frt-block { padding: 12px; border-bottom: 1px solid #e2e8f0; background: white; }
    .frt-block:last-child { border-bottom: none; }
    
    .frt-title { 
        font-weight: 800; text-transform: uppercase; font-size: 0.75em; 
        display: block; margin-bottom: 6px; letter-spacing: 0.5px;
    }
    .frt-content { 
        font-family: 'Segoe UI', sans-serif; font-size: 0.95em; color: #334155; 
        line-height: 1.6; white-space: pre-wrap; 
    }
    
    /* Couleurs Sémantiques FRT */
    .c-usage { color: #d97706; border-left: 4px solid #d97706; } /* Quand utiliser */
    .c-method { color: #059669; border-left: 4px solid #059669; background-color: #f0fdf4; } /* Méthode (Vert) */
    .c-trap { color: #dc2626; border-left: 4px solid #dc2626; } /* Pièges */
    .c-conc { color: #2563eb; border-left: 4px solid #2563eb; } /* Conclusion */

    /* 5. QI CARDS (GROUPED) */
    .file-block {
        margin-bottom: 12px; border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden;
    }
    .file-header {
        background-color: #f1f5f9; padding: 8px 12px; font-weight: 700; font-size: 0.85em;
        color: #475569; border-bottom: 1px solid #e2e8f0; display: flex; align-items: center;
    }
    .qi-item {
        background-color: white; padding: 10px 12px; border-bottom: 1px solid #f8fafc;
        font-family: 'Georgia', serif; font-size: 0.95em; color: #1e293b;
        border-left: 3px solid #9333ea; margin: 0; 
    }

    /* 6. SATURATION GRAPH */
    .sat-box {
        background-color: #f0f9ff; border: 1px solid #bae6fd; padding: 20px; border-radius: 8px; margin-top: 20px;
    }
    .sat-metric { font-size: 1.5em; font-weight: bold; color: #0284c7; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. KERNEL (CONTENU FRT CORRIGÉ)
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
            {"type": "usage", "title": "🔔 1. Quand utiliser cette méthode", "text": "Lorsque l'énoncé demande explicitement la **nature** de la suite ou de prouver qu'elle est **géométrique**."},
            {"type": "method", "title": "✅ 2. Méthode Rédigée (Copie Élève)", "text": "1. Pour tout entier naturel $n$, on exprime $u_{n+1}$ à l'aide de la relation de récurrence.\n\n2. On calcule le quotient $\\frac{u_{n+1}}{u_n}$.\n\n3. On simplifie l'expression jusqu'à ce que les termes en $n$ s'annulent.\n\n4. On obtient un résultat constant réel $q$."},
            {"type": "trap", "title": "⚠️ 3. Pièges à éviter", "text": "• Oublier de vérifier que $u_n \\neq 0$ avant de diviser.\n• Calculer la différence $u_{n+1} - u_n$ (confusion avec suite arithmétique)."},
            {"type": "conc", "title": "✍️ 4. Conclusion Type", "text": "Le rapport étant constant, la suite $(u_n)$ est géométrique de raison $q$."}
        ]
    },
    "FRT_M_S02": {
        "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES",
        "QC": "Comment lever une indétermination (limite) ?",
        "Triggers": ["calculer la limite", "limite quand n tend vers +infini", "étudier la convergence"],
        "ARI": ["1. Terme dominant", "2. Factorisation", "3. Limites usuelles", "4. Conclure"],
        "FRT_DATA": [
            {"type": "usage", "title": "🔔 1. Quand utiliser cette méthode", "text": "En présence d'une forme indéterminée type $\\infty - \\infty$ ou $\\frac{\\infty}{\\infty}$ pour une suite polynomiale ou rationnelle."},
            {"type": "method", "title": "✅ 2. Méthode Rédigée (Copie Élève)", "text": "1. On identifie le terme dominant (la plus haute puissance de $n$).\n\n2. On factorise l'expression par ce terme dominant.\n\n3. On utilise la limite usuelle : $\\lim_{n \\to +\\infty} \\frac{1}{n} = 0$."},
            {"type": "trap", "title": "⚠️ 3. Pièges à éviter", "text": "• Appliquer la règle des signes sans factoriser.\n• Oublier de factoriser le dénominateur."},
            {"type": "conc", "title": "✍️ 4. Conclusion Type", "text": "Par opération sur les limites, la suite converge vers..."}
        ]
    },
    "FRT_M_F01": {
        "Matiere": "MATHS", "Chap": "FONCTIONS",
        "QC": "Comment appliquer le TVI (Solution unique) ?",
        "Triggers": ["montrer que f(x)=k admet une solution unique", "existence et unicité", "théorème des valeurs intermédiaires"],
        "ARI": ["1. Continuité", "2. Monotonie", "3. Bornes", "4. TVI"],
        "FRT_DATA": [
            {"type": "usage", "title": "🔔 1. Quand utiliser cette méthode", "text": "Pour prouver l'existence et l'unicité d'une solution à une équation $f(x)=k$."},
            {"type": "method", "title": "✅ 2. Méthode Rédigée (Copie Élève)", "text": "1. La fonction $f$ est **continue** sur l'intervalle $I$.\n\n2. La fonction $f$ est **strictement monotone** sur $I$.\n\n3. On calcule les images aux bornes $f(a)$ et $f(b)$ et on vérifie que $k$ est compris entre elles.\n\n4. On cite le **corollaire du théorème des valeurs intermédiaires**."},
            {"type": "trap", "title": "⚠️ 3. Pièges à éviter", "text": "• Oublier le mot 'strictement' pour la monotonie (condition d'unicité).\n• Oublier la continuité (condition d'existence)."},
            {"type": "conc", "title": "✍️ 4. Conclusion Type", "text": "L'équation admet une unique solution $\\alpha$ sur l'intervalle."}
        ]
    }
}

QI_PATTERNS = {
    "FRT_M_S01": ["Montrer que la suite (Un) est géométrique.", "Quelle est la nature de la suite (Vn) ?", "Justifier que (Un) est géométrique."],
    "FRT_M_S02": ["Déterminer la limite de la suite.", "Calculer la limite quand n tend vers +infini.", "Étudier la convergence."],
    "FRT_M_F01": ["Montrer que f(x)=0 admet une unique solution.", "Démontrer l'existence et l'unicité."]
}

# ==============================================================================
# 2. MOTEUR
# ==============================================================================

def ingest_factory_v30(urls, volume, matiere):
    target_frts = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere]
    
    # Sécurité absolue : DataFrame vide bien structuré si rien trouvé
    cols_src = ["Fichier", "Nature", "Annee", "Telechargement", "Qi_Data"]
    cols_atm = ["FRT_ID", "Qi", "File", "Year", "Chapitre"]
    
    if not target_frts:
        return pd.DataFrame(columns=cols_src), pd.DataFrame(columns=cols_atm)
    
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
            
        sources.append({
            "Fichier": filename, "Nature": nature, "Annee": annee, 
            "Telechargement": f"https://fake-smaxia/dl/{filename}", "Qi_Data": qi_list
        })
        
    return pd.DataFrame(sources), pd.DataFrame(atoms)

def compute_qc_v30(df_atoms):
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

def simulate_saturation_v30(volume, matiere):
    """
    Simule une courbe de saturation réaliste (Logarithmique).
    Nous simulons un univers théorique de 50 QC pour montrer la courbe.
    """
    # Univers théorique simulé pour le graphe
    TOTAL_THEORETICAL_QC = 50 
    
    data_points = []
    discovered_set = set()
    
    # Probabilité de découverte décroissante
    for i in range(1, volume + 1):
        # Plus on a de QC, plus c'est dur d'en trouver une nouvelle
        current_count = len(discovered_set)
        
        # Formule de saturation : proba de trouver une nouvelle = (Total - Trouvé) / Total
        proba_new = (TOTAL_THEORETICAL_QC - current_count) / TOTAL_THEORETICAL_QC
        
        # On simule 3 questions par sujet
        for _ in range(3):
            if random.random() < proba_new * 0.5: # 0.5 pour lisser
                # On "découvre" une nouvelle ID (simulée)
                discovered_set.add(f"QC_{current_count + 1}")
        
        data_points.append({"Sujets Injectés": i, "QC Uniques Découvertes": len(discovered_set)})
        
    return pd.DataFrame(data_points)

def analyze_external_v30(file, matiere):
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
        df_src, df_atoms = ingest_factory_v30(urls.split('\n'), vol, sel_matiere)
        df_qc = compute_qc_v30(df_atoms)
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
                        
                        # 3. FRT (BLOCS CLEAN)
                        with c3:
                            with st.expander("🧾 FRT (Élève)"):
                                for block in row['FRT_DATA']:
                                    cls_map = {"usage": "c-usage", "method": "c-method", "trap": "c-trap", "conc": "c-conc"}
                                    css = cls_map.get(block['type'], "")
                                    st.markdown(f"""
                                    <div class='frt-block {css}'>
                                        <span class='frt-title'>{block['title']}</span>
                                        <div class='frt-content'>{block['text']}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # 4. QI (GROUPÉ PAR FICHIER)
                        with c4:
                            with st.expander(f"📄 Qi ({row['n_q']})"):
                                qi_by_file = defaultdict(list)
                                for item in row['Evidence']:
                                    qi_by_file[item['Fichier']].append(item['Qi'])
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
            
        # --- GRAPH DE SATURATION ---
        st.divider()
        st.markdown("### 📈 Analyse de Saturation (Sujets vs QC)")
        st.caption("Simulation de la découverte de nouvelles QC en fonction du volume de sujets injectés.")
        
        if st.button("Générer Courbe de Saturation"):
            with st.spinner("Simulation en cours..."):
                df_chart = simulate_saturation_v30(200, sel_matiere) # Simulation sur 200 sujets
                st.line_chart(df_chart, x="Sujets Injectés", y="QC Uniques Découvertes", color="#2563eb")
                st.info("On observe un plateau lorsque le moteur a découvert l'essentiel des structures du programme.")

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
            ext = analyze_external_v30(up, sel_matiere)
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
