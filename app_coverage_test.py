import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V25 (Official Standard)")
st.title("🛡️ SMAXIA - Console V25 (Official UI Standard)")

# ==============================================================================
# 🎨 STYLES CSS (GABARIT OFFICIEL v1.0)
# ==============================================================================
st.markdown("""
<style>
    /* --- 1. EN-TÊTE QC (IDENTITÉ FORTE) --- */
    .qc-header-container {
        background-color: #f8f9fa;
        border-left: 6px solid #2563eb; /* Bleu SMAXIA */
        padding: 15px 20px;
        margin-bottom: 15px;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        display: flex; justify-content: space-between; align-items: center;
        flex-wrap: wrap;
    }
    .qc-identity { display: flex; align-items: center; gap: 15px; }
    .qc-id { 
        color: #d97706; font-weight: 900; font-size: 1.3em; 
        font-family: 'Segoe UI', sans-serif; letter-spacing: 0.5px;
    }
    .qc-title { 
        color: #1f2937; font-weight: 700; font-size: 1.2em; 
        font-family: 'Segoe UI', sans-serif;
    }
    .qc-badge { 
        font-family: 'Consolas', 'Monaco', monospace; 
        font-size: 0.85em; font-weight: 700; color: #374151;
        background-color: #e5e7eb; padding: 6px 12px; 
        border-radius: 4px; border: 1px solid #d1d5db;
    }

    /* --- 2. DÉCLENCHEURS (OBSERVABLES) --- */
    .trigger-card {
        background-color: #fff1f2; border: 1px solid #fecaca; 
        border-radius: 6px; padding: 12px; height: 100%;
    }
    .trigger-item {
        display: block; margin-bottom: 6px; color: #991b1b; 
        font-weight: 600; font-size: 0.9em; font-family: sans-serif;
        padding-left: 8px; border-left: 3px solid #ef4444;
    }

    /* --- 3. ARI (LOGIQUE MOTEUR) --- */
    .ari-card {
        background-color: #f3f4f6; border: 1px dashed #9ca3af;
        border-radius: 6px; padding: 12px; height: 100%;
    }
    .ari-step {
        font-family: 'Courier New', monospace; font-size: 0.9em; 
        color: #1f2937; margin-bottom: 4px; display: block;
    }

    /* --- 4. FRT (COPIE ÉLÈVE) --- */
    .frt-card {
        background-color: #ffffff; border: 1px solid #10b981; 
        border-left: 5px solid #10b981; border-radius: 6px; 
        overflow: hidden; height: 100%;
    }
    .frt-section {
        padding: 10px 15px; border-bottom: 1px solid #f0fdf4;
    }
    .frt-label {
        font-weight: 800; font-size: 0.75em; text-transform: uppercase; 
        letter-spacing: 1px; display: block; margin-bottom: 6px;
    }
    .frt-text {
        font-family: 'Segoe UI', sans-serif; font-size: 0.95em; 
        color: #334155; line-height: 1.5; white-space: pre-wrap;
    }
    /* Couleurs Sémantiques FRT */
    .lbl-when { color: #d97706; }
    .lbl-how { color: #059669; }
    .lbl-trap { color: #dc2626; }
    .lbl-conc { color: #2563eb; }

    /* --- 5. Qi (PREUVE TERRAIN - CARD) --- */
    .qi-scroll-zone {
        max-height: 500px; overflow-y: auto; padding-right: 5px;
    }
    .qi-card {
        background-color: #ffffff; border: 1px solid #e5e7eb;
        border-radius: 6px; padding: 12px; margin-bottom: 10px;
        border-left: 4px solid #9333ea; /* Violet Audit */
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        transition: transform 0.1s;
    }
    .qi-card:hover { transform: translateX(2px); }
    .qi-header { 
        font-size: 0.75em; font-weight: 800; color: #9333ea; 
        text-transform: uppercase; margin-bottom: 4px;
    }
    .qi-content {
        font-family: 'Georgia', serif; font-size: 1.05em; 
        color: #111827; font-weight: 600; line-height: 1.4;
        margin-bottom: 8px;
    }
    .qi-source {
        font-family: sans-serif; font-size: 0.8em; color: #6b7280;
        display: flex; align-items: center; gap: 5px;
    }

    /* UTILS */
    .section-title { 
        font-size: 1.1em; font-weight: 700; color: #4b5563; 
        margin-bottom: 10px; text-transform: uppercase; border-bottom: 2px solid #e5e7eb; padding-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. KERNEL SMAXIA (DONNÉES CONFORMES GABARIT)
# ==============================================================================

LISTE_CHAPITRES = {
    "MATHS": ["SUITES NUMÉRIQUES", "FONCTIONS", "PROBABILITÉS", "GÉOMÉTRIE"],
    "PHYSIQUE": ["MÉCANIQUE", "ONDES"]
}

UNIVERS_SMAXIA = {
    "FRT_M_S01": {
        "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", 
        "QC_ID": "QC-01",
        "QC_Title": "Comment démontrer qu'une suite est géométrique ?",
        
        # 🔥 DÉCLENCHEURS (Exacts & Observables)
        "Triggers": [
            "montrer que la suite est géométrique",
            "déterminer la nature de la suite",
            "préciser la raison q",
            "justifier que (Un) est une suite géométrique"
        ],
        
        # ⚙️ ARI (Algorithme Moteur)
        "ARI": [
            "1. Exprimer u(n+1)",
            "2. Former quotient u(n+1)/u(n)",
            "3. Simplifier algèbre",
            "4. Identifier constante q"
        ],
        
        # 🧾 FRT (Copie Élève - 4 Blocs)
        "FRT_HTML": """
<div class='frt-section'>
    <span class='frt-label lbl-when'>🔔 1. Quand utiliser</span>
    <div class='frt-text'>Lorsque l'énoncé demande explicitement la nature de la suite ou de prouver qu'elle est géométrique (suite définie par récurrence).</div>
</div>
<div class='frt-section'>
    <span class='frt-label lbl-how'>✅ 2. Méthode Rédigée</span>
    <div class='frt-text'>
1. Pour tout entier $n$, on exprime $u_{n+1}$ à l'aide de la relation donnée.<br>
2. On calcule le quotient $\\frac{u_{n+1}}{u_n}$.<br>
3. On simplifie l'expression jusqu'à éliminer $n$.<br>
4. On obtient un réel constant $q$.
    </div>
</div>
<div class='frt-section'>
    <span class='frt-label lbl-trap'>⚠️ 3. Pièges</span>
    <div class='frt-text'>• Oublier de vérifier $u_n \\neq 0$.<br>• Calculer la différence (suite arithmétique).</div>
</div>
<div class='frt-section'>
    <span class='frt-label lbl-conc'>✍️ 4. Conclusion</span>
    <div class='frt-text'>"Le rapport est constant égal à $q$, la suite est donc géométrique."</div>
</div>
"""
    },
    
    "FRT_M_F02": {
        "Matiere": "MATHS", "Chap": "FONCTIONS",
        "QC_ID": "QC-02",
        "QC_Title": "Comment appliquer le TVI (Solution unique) ?",
        
        "Triggers": [
            "montrer que l'équation f(x)=k admet une unique solution",
            "démontrer l'existence et l'unicité",
            "théorème des valeurs intermédiaires",
            "justifier qu'il existe un unique alpha"
        ],
        
        "ARI": [
            "1. Check Continuité",
            "2. Check Monotonie Stricte",
            "3. Calc Images Bornes",
            "4. Invoke Corollaire TVI"
        ],
        
        "FRT_HTML": """
<div class='frt-section'>
    <span class='frt-label lbl-when'>🔔 1. Quand utiliser</span>
    <div class='frt-text'>Pour prouver l'existence et l'unicité d'une solution à $f(x)=k$ sans calcul explicite.</div>
</div>
<div class='frt-section'>
    <span class='frt-label lbl-how'>✅ 2. Méthode Rédigée</span>
    <div class='frt-text'>
1. La fonction $f$ est <b>continue</b> et <b>strictement monotone</b> sur $I$.<br>
2. On calcule les images des bornes $f(a)$ et $f(b)$.<br>
3. On constate que $k$ est compris entre $f(a)$ et $f(b)$.<br>
4. D'après le corollaire du TVI...
    </div>
</div>
<div class='frt-section'>
    <span class='frt-label lbl-trap'>⚠️ 3. Pièges</span>
    <div class='frt-text'>• Oublier "strictement" (perte unicité).<br>• Oublier continuité (perte existence).</div>
</div>
<div class='frt-section'>
    <span class='frt-label lbl-conc'>✍️ 4. Conclusion</span>
    <div class='frt-text'>"L'équation admet une unique solution $\\alpha$ sur l'intervalle."</div>
</div>
"""
    }
}

# Générateur de Qi (Doit contenir les déclencheurs)
QI_PATTERNS = {
    "FRT_M_S01": [
        "Montrer que la suite (Un) est géométrique.",
        "Déterminer la nature de la suite (Vn).",
        "Justifier que (Wn) est une suite géométrique."
    ],
    "FRT_M_F02": [
        "Montrer que l'équation f(x)=0 admet une unique solution alpha.",
        "Démontrer l'existence et l'unicité de la solution sur [0;1]."
    ]
}

# ==============================================================================
# 2. MOTEUR (INGESTION & CALCUL)
# ==============================================================================

def ingest_factory(urls, volume, matiere):
    target_frts = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere]
    if not target_frts: return pd.DataFrame(), pd.DataFrame()
    
    sources, atoms = [], []
    progress = st.progress(0)
    
    for i in range(volume):
        progress.progress((i+1)/volume)
        nature = random.choice(["BAC", "DST", "INTERRO"])
        annee = random.choice(range(2020, 2025))
        filename = f"Sujet_{matiere}_{nature}_{annee}_{i}.pdf"
        
        nb_qi = random.randint(3, 6)
        frts = random.choices(target_frts, k=nb_qi)
        
        qi_list = []
        for frt_id in frts:
            # On force la présence du déclencheur dans la Qi
            qi_txt = random.choice(QI_PATTERNS.get(frt_id, ["Question type"])) 
            atoms.append({
                "FRT_ID": frt_id, "Qi": qi_txt, "File": filename, 
                "Year": annee, "Chapitre": UNIVERS_SMAXIA[frt_id]["Chap"]
            })
            qi_list.append({"Qi": qi_txt, "FRT_ID": frt_id})
            
        sources.append({
            "Fichier": filename, "Nature": nature, "Année": annee,
            "Telechargement": f"http://fake/{filename}", "Qi_Data": qi_list
        })
        
    return pd.DataFrame(sources), pd.DataFrame(atoms)

def compute_qc(df_atoms):
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
            "Chapitre": row["Chapitre"], "QC_ID": meta["QC_ID"], "FRT_ID": row["FRT_ID"],
            "Titre": meta["QC_Title"], "Score": score, "n_q": n_q, "Psi": psi, "N_tot": N_tot, "t_rec": t_rec,
            "Triggers": meta["Triggers"], "ARI": meta["ARI"], "FRT_HTML": meta["FRT_HTML"],
            "Evidence": [{"Fichier": f, "Qi": q} for f, q in zip(row["File"], row["Qi"])]
        })
    return pd.DataFrame(qcs).sort_values(by="Score", ascending=False)

def analyze_external(file, matiere):
    target = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere]
    if not target: return []
    frts = random.choices(target, k=8)
    return [{"Qi": random.choice(QI_PATTERNS.get(f, ["Qi"]))+" (Extrait)", "FRT_ID": f} for f in frts]

# ==============================================================================
# 3. INTERFACE (CONFORME GABARIT)
# ==============================================================================

with st.sidebar:
    st.header("Paramètres Académiques")
    st.selectbox("Classe", ["Terminale"], disabled=True)
    sel_matiere = st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    sel_chapitres = st.multiselect("Chapitres", LISTE_CHAPITRES.get(sel_matiere, []), default=LISTE_CHAPITRES.get(sel_matiere, [])[:2])

tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine", "✅ Onglet 2 : Audit"])

# --- USINE ---
with tab_usine:
    c1, c2 = st.columns([3, 1])
    with c1: urls = st.text_area("URLs Sources", "https://apmep.fr", height=68)
    with c2: 
        vol = st.number_input("Volume", 5, 500, 20, step=5)
        run = st.button("LANCER L'USINE 🚀", type="primary")

    if run:
        df_src, df_atoms = ingest_factory(urls.split('\n'), vol, sel_matiere)
        df_qc = compute_qc(df_atoms)
        st.session_state['df_src'] = df_src
        st.session_state['df_qc'] = df_qc
        st.success(f"Ingestion terminée : {len(df_src)} sujets traités.")

    st.divider()

    if 'df_src' in st.session_state and not st.session_state['df_src'].empty:
        st.markdown(f"### 📥 Sujets Traités ({len(st.session_state['df_src'])})")
        df_disp = st.session_state['df_src'].rename(columns={"Année": "Annee", "Telechargement": "Lien"})
        st.data_editor(df_disp[["Fichier", "Nature", "Annee", "Lien"]], 
                       column_config={"Lien": st.column_config.LinkColumn("Téléchargement", display_text="📥 PDF")},
                       hide_index=True, use_container_width=True, disabled=True)

        st.divider()

        st.markdown("### 🧠 Base de Connaissance (QC)")
        if not st.session_state['df_qc'].empty:
            qc_view = st.session_state['df_qc'][st.session_state['df_qc']["Chapitre"].isin(sel_chapitres)]
            
            if qc_view.empty:
                st.info("Pas de QC dans ces chapitres (voir autres chapitres).")
            else:
                chapters = qc_view["Chapitre"].unique()
                for chap in chapters:
                    subset = qc_view[qc_view["Chapitre"] == chap]
                    st.markdown(f"#### 📘 {chap} ({len(subset)} QC)")
                    
                    for idx, row in subset.iterrows():
                        # --- 1. EN-TÊTE QC (IDENTITÉ FORTE) ---
                        st.markdown(f"""
                        <div class="qc-header-container">
                            <div class="qc-identity">
                                <span class="qc-id">{row['QC_ID']}</span>
                                <span class="qc-title">{row['Titre']}</span>
                            </div>
                            <div class="qc-badge">
                                Score(q)={row['Score']:.0f} | n_q={row['n_q']} | Ψ={row['Psi']} | N_tot={row['N_tot']} | t_rec={row['t_rec']:.1f}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # --- BLOCS DÉTAILS ---
                        c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1.2])
                        
                        # 🔥 TRIGGERS
                        with c1:
                            st.markdown("<div class='section-title'>🔥 Déclencheurs</div>", unsafe_allow_html=True)
                            with st.container(height=350, border=True): # Scrollable si long
                                html = "<div class='trigger-card'>"
                                for t in row['Triggers']: 
                                    html += f"<span class='trigger-item'>“{t}”</span>"
                                html += "</div>"
                                st.markdown(html, unsafe_allow_html=True)
                        
                        # ⚙️ ARI
                        with c2:
                            st.markdown("<div class='section-title'>⚙️ ARI</div>", unsafe_allow_html=True)
                            with st.container(height=350, border=True):
                                html = "<div class='ari-card'>"
                                for s in row['ARI']: 
                                    html += f"<span class='ari-step'>{s}</span>"
                                html += "</div>"
                                st.markdown(html, unsafe_allow_html=True)
                        
                        # 🧾 FRT (HTML INJECTÉ)
                        with c3:
                            st.markdown("<div class='section-title'>🧾 FRT (Copie Élève)</div>", unsafe_allow_html=True)
                            with st.container(height=350, border=True):
                                st.markdown(f"<div class='frt-card'>{row['FRT_HTML']}</div>", unsafe_allow_html=True)
                        
                        # 📄 Qi (PREUVE TERRAIN - CARDS)
                        with c4:
                            st.markdown(f"<div class='section-title'>📄 Qi Associées ({row['n_q']})</div>", unsafe_allow_html=True)
                            with st.container(height=350, border=True):
                                qi_html = "<div class='qi-scroll-zone'>"
                                for i, item in enumerate(row['Evidence']):
                                    qi_html += f"""
                                    <div class="qi-card">
                                        <div class="qi-header">🧠 Qi #{i+1}</div>
                                        <div class="qi-content">“ {item['Qi']} ”</div>
                                        <div class="qi-source">📎 Source : {item['Fichier']}</div>
                                    </div>
                                    """
                                qi_html += "</div>"
                                st.markdown(qi_html, unsafe_allow_html=True)
                        
                        st.write("") # Spacer
                        st.divider()

# --- AUDIT ---
with tab_audit:
    st.subheader("Validation Booléenne")
    if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
        st.markdown("#### ✅ 1. Test Interne (Sujet Traité)")
        t1_file = st.selectbox("Sujet Traité", st.session_state['df_src']["Fichier"])
        if st.button("LANCER AUDIT"):
            data = st.session_state['df_src'][st.session_state['df_src']["Fichier"]==t1_file].iloc[0]["Qi_Data"]
            known = st.session_state['df_qc']["FRT_ID"].unique()
            
            rows = []
            ok = 0
            for item in data:
                is_ok = item["FRT_ID"] in known
                if is_ok: ok += 1
                status = "✅ MATCH" if is_ok else "❌ GAP"
                rows.append({"Qi": item["Qi"], "Statut": status})
                
            st.metric("Couverture", f"{(ok/len(data))*100:.0f}%")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            
        st.divider()
        st.markdown("#### 🌍 2. Test Externe")
        up = st.file_uploader("PDF Externe", type="pdf")
        if up:
            ext = analyze_external(up, sel_matiere)
            if not ext: st.error("Rien trouvé")
            else:
                rows_ext = []
                ok = 0
                known = st.session_state['df_qc']["FRT_ID"].unique()
                for item in ext:
                    is_ok = item["FRT_ID"] in known
                    if is_ok: ok += 1
                    status = "✅ MATCH" if is_ok else "❌ GAP"
                    # Recherche de la QC correspondante
                    qc_name = "---"
                    if is_ok:
                        qc_info = st.session_state['df_qc'][st.session_state['df_qc']["FRT_ID"]==item["FRT_ID"]].iloc[0]
                        qc_name = f"{qc_info['QC_ID']} {qc_info['Titre']}"
                        
                    rows_ext.append({"Qi": item["Qi"], "QC Correspondante": qc_name, "Statut": status})
                    
                st.markdown(f"### Taux : {(ok/len(ext))*100:.1f}%")
                
                def color(row):
                    return ['background-color: #dcfce7' if row['Statut']=="✅ MATCH" else 'background-color: #fee2e2']*len(row)
                st.dataframe(pd.DataFrame(rows_ext).style.apply(color, axis=1), use_container_width=True)
    else:
        st.info("Lancez l'usine.")
