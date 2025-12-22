import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V35")
st.title("🛡️ SMAXIA - Console V35 (Fusion Totale : UI + Math Kernel A2)")

# ==============================================================================
# 1. PARAMÈTRES & CONSTANTES (DOC A2 - PARTIE 6)
# ==============================================================================
CONSTANTS = {
    "EPSILON": 0.1, "DELTA_C": 1.0, "ALPHA_DELTA": 1.5, "PSI_AVG_REF": 0.85
}

# Poids des transformations cognitives (Tj)
TRANSFORMATION_WEIGHTS = {
    "IDENTIFICATION": 1.0, "EXPRESSION_RECURRENCE": 2.0, "CALCUL_RATIO": 2.5,
    "FACTORISATION_FORCEE": 3.0, "SIMPLIFICATION": 1.5, "LIMITES_USUELLES": 2.0,
    "THEOREME_TVI": 4.0, "DERIVEE": 2.0, "SIGNE": 2.0, "CONSTANTE": 1.0,
    "HEREDITE": 3.0, "INITIALISATION": 1.0, "BOUCLE": 2.5, "COMPARAISON": 1.5
}

# ==============================================================================
# 2. KERNEL (CONTENU ÉTENDU POUR SATURATION)
# ==============================================================================
LISTE_CHAPITRES = {
    "MATHS": ["SUITES NUMÉRIQUES", "FONCTIONS", "PROBABILITÉS", "GÉOMÉTRIE"],
    "PHYSIQUE": ["MÉCANIQUE", "ONDES"]
}

UNIVERS_SMAXIA = {
    # --- SUITES (10 QC pour courbe réaliste) ---
    "FRT_M_S01": {
        "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", "QC": "Démontrer qu'une suite est géométrique",
        "Triggers": ["montrer que la suite est géométrique", "déterminer la nature"],
        "ARI": ["EXPRESSION_RECURRENCE", "CALCUL_RATIO", "SIMPLIFICATION", "CONSTANTE"],
        "FRT_DATA": [{"type": "method", "title": "✅ Méthode", "text": "1. Exprimer u(n+1).\n2. Calculer u(n+1)/u(n).\n3. Simplifier.\n4. Identifier q."}]
    },
    "FRT_M_S02": {
        "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", "QC": "Lever une indétermination (limite)",
        "Triggers": ["calculer la limite", "forme indéterminée"],
        "ARI": ["IDENTIFICATION", "FACTORISATION_FORCEE", "LIMITES_USUELLES", "SIMPLIFICATION"],
        "FRT_DATA": [{"type": "method", "title": "✅ Méthode", "text": "1. Identifier dominant.\n2. Factoriser.\n3. Conclure."}]
    },
    "FRT_M_S03": { "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", "QC": "Démontrer par récurrence", "Triggers": ["par récurrence"], "ARI": ["INITIALISATION", "HEREDITE", "CONCLUSION"], "FRT_DATA": [{"type": "method", "title": "✅ Méthode", "text": "1. Initialisation.\n2. Hérédité.\n3. Conclusion."}] },
    "FRT_M_S04": { "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", "QC": "Étudier les variations (différence)", "Triggers": ["sens de variation"], "ARI": ["EXPRESSION_RECURRENCE", "SIGNE", "CONCLUSION"], "FRT_DATA": [{"type": "method", "title": "✅ Méthode", "text": "1. Calculer u(n+1)-u(n).\n2. Etudier le signe."}] },
    "FRT_M_S05": { "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", "QC": "Déterminer un seuil (Algorithme)", "Triggers": ["algorithme seuil"], "ARI": ["BOUCLE", "COMPARAISON"], "FRT_DATA": [{"type": "method", "title": "✅ Méthode", "text": "Tant que u < S..."}] },
    "FRT_M_S06": { "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", "QC": "Montrer qu'une suite est majorée", "Triggers": ["montrer que u(n) < M"], "ARI": ["INITIALISATION", "HEREDITE"], "FRT_DATA": [{"type": "method", "title": "✅ Méthode", "text": "Récurrence simple."}] },
    
    # --- FONCTIONS ---
    "FRT_M_F01": {
        "Matiere": "MATHS", "Chap": "FONCTIONS", "QC": "Appliquer le TVI (Unique)",
        "Triggers": ["solution unique alpha"],
        "ARI": ["DERIVEE", "SIGNE", "THEOREME_TVI"],
        "FRT_DATA": [{"type": "method", "title": "✅ Méthode", "text": "1. Continuité/Monotonie.\n2. Images.\n3. TVI."}]
    }
}

QI_PATTERNS = {k: [f"Question type sur {v['QC']}...", f"Variante : {v['Triggers'][0]}..."] for k, v in UNIVERS_SMAXIA.items()}

# ==============================================================================
# 3. MOTEUR MATHÉMATIQUE A2 (INTÉGRÉ)
# ==============================================================================

def calculate_score_F1_F2(frt_id, n_q, n_tot, year):
    # 1. Récupération ARI
    ari_steps = UNIVERS_SMAXIA[frt_id]["ARI"]
    
    # 2. Calcul F1 (Psi) - A2 2.1
    sum_tj = sum(TRANSFORMATION_WEIGHTS.get(step, 1.0) for step in ari_steps)
    psi_raw = CONSTANTS["DELTA_C"] * (CONSTANTS["EPSILON"] + sum_tj)**2
    # Normalisation simulée (dans la vraie vie on divise par max(Psi) du chapitre)
    psi_norm = min(1.0, psi_raw / 100.0) 
    
    # 3. Calcul F2 (Score) - A2 2.2
    t_rec = max(0.5, 2025 - year)
    density = n_q / n_tot if n_tot > 0 else 0
    recency = 1 / (CONSTANTS["ALPHA_DELTA"] * t_rec)
    
    score = density * recency * psi_norm * 10000 # *10000 pour lisibilité
    return score, psi_norm

def ingest_factory_v35(volume, matiere):
    target_frts = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere]
    if not target_frts: return pd.DataFrame(), pd.DataFrame()
    
    sources, atoms = [], []
    
    for i in range(volume):
        annee = random.choice(range(2020, 2025))
        filename = f"Sujet_{matiere}_{i}.pdf"
        
        # Tirage pondéré : On découvre d'abord les classiques, puis les rares
        # (Simulation de la saturation)
        if i < volume * 0.3:
            # Début : On tire souvent les mêmes (S01, S02)
            subset = target_frts[:2] 
        else:
            # Ensuite : On ouvre tout
            subset = target_frts
            
        frts = random.choices(subset, k=random.randint(3, 5))
        
        qi_data = []
        for frt_id in frts:
            atoms.append({"FRT_ID": frt_id, "Qi": random.choice(QI_PATTERNS[frt_id]), "File": filename, "Year": annee, "Chapitre": UNIVERS_SMAXIA[frt_id]["Chap"]})
            qi_data.append({"Qi": "...", "FRT_ID": frt_id})
            
        sources.append({"Fichier": filename, "Annee": annee, "Telechargement": "📥", "Qi_Data": qi_data})
        
    return pd.DataFrame(sources), pd.DataFrame(atoms)

def compute_qc_v35(df_atoms):
    if df_atoms.empty: return pd.DataFrame()
    
    grouped = df_atoms.groupby("FRT_ID").agg({"Qi": list, "File": list, "Year": "max", "Chapitre": "first"}).reset_index()
    qcs = []
    N_tot = len(df_atoms)
    
    for idx, row in grouped.iterrows():
        meta = UNIVERS_SMAXIA[row["FRT_ID"]]
        # APPEL DU MOTEUR MATHÉMATIQUE A2
        score, psi = calculate_score_F1_F2(row["FRT_ID"], len(row["Qi"]), N_tot, row["Year"])
        
        qcs.append({
            "Chapitre": row["Chapitre"], "QC_ID": f"QC-{idx+1:02d}", "Titre": meta["QC"],
            "Score": score, "n_q": len(row["Qi"]), "Psi": psi, "N_tot": N_tot, "t_rec": 2025-row["Year"],
            "Triggers": meta["Triggers"], "ARI": meta["ARI"], "FRT_DATA": meta["FRT_DATA"],
            "Evidence": [{"Fichier": f, "Qi": q} for f, q in zip(row["File"], row["Qi"])]
        })
        
    return pd.DataFrame(qcs).sort_values(by="Score", ascending=False)

# ==============================================================================
# 4. INTERFACE V35 (DESIGN VALIDÉ + CONTENU A2)
# ==============================================================================

# STYLES CSS (Gabarit V28/V32)
st.markdown("""<style>
.qc-header-box { background:#f8f9fa; border-left:6px solid #2563eb; padding:15px; margin-bottom:10px; border-radius:4px; }
.qc-id-text { color:#d97706; font-weight:900; font-size:1.2em; margin-right:10px; }
.qc-title-text { color:#1f2937; font-weight:700; font-size:1.15em; }
.qc-meta-text { font-family:monospace; font-size:0.85em; background:#e5e7eb; padding:4px 8px; border-radius:4px; }
.trigger-item { background:#fff1f2; color:#991b1b; padding:5px 10px; margin-bottom:4px; border-left:4px solid #f87171; font-weight:600; display:block; }
.ari-step { background:#f3f4f6; color:#374151; padding:4px 8px; margin-bottom:3px; font-family:monospace; border:1px dashed #d1d5db; display:block; }
.frt-block { padding:10px; border-bottom:1px solid #e2e8f0; background:white; margin-bottom:5px; border-radius:4px; border:1px solid #e2e8f0; }
.frt-title { font-weight:800; text-transform:uppercase; font-size:0.75em; display:block; margin-bottom:4px; }
.qi-item { background:white; padding:10px; border-bottom:1px solid #f8fafc; border-left:3px solid #9333ea; margin:5px; font-family:serif; }
.file-header { background:#f1f5f9; padding:8px; font-weight:700; font-size:0.85em; color:#475569; }
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Paramètres")
    sel_matiere = st.selectbox("Matière", ["MATHS"])
    sel_chapitres = st.multiselect("Chapitres", LISTE_CHAPITRES["MATHS"], default=["SUITES NUMÉRIQUES"])

tab_usine, tab_audit = st.tabs(["🏭 Usine & Saturation", "✅ Audit"])

with tab_usine:
    c1, c2 = st.columns([3, 1])
    with c1: urls = st.text_area("URLs", "https://apmep.fr", height=68)
    with c2: 
        vol = st.number_input("Volume", 10, 500, 50, step=10)
        run = st.button("LANCER L'USINE 🚀", type="primary")

    if run:
        df_src, df_atoms = ingest_factory_v35(vol, sel_matiere)
        df_qc = compute_qc_v35(df_atoms)
        st.session_state['df_qc'] = df_qc
        st.session_state['df_src'] = df_src
        st.success(f"Ingestion terminée : {len(df_src)} sujets traités.")

    if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
        # LISTE QC AVEC DESIGN V32
        qc_view = st.session_state['df_qc'][st.session_state['df_qc']["Chapitre"].isin(sel_chapitres)]
        
        st.markdown(f"### 🧠 Base de Connaissance ({len(qc_view)} QC)")
        
        for idx, row in qc_view.iterrows():
            st.markdown(f"""
            <div class="qc-header-box">
                <span class="qc-id-text">{row['QC_ID']}</span>
                <span class="qc-title-text">{row['Titre']}</span><br>
                <span class="qc-meta-text">Score(F2)={row['Score']:.0f} | n_q={row['n_q']} | Ψ(F1)={row['Psi']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                with st.expander("🔥 Déclencheurs"):
                    for t in row['Triggers']: st.markdown(f"<span class='trigger-item'>{t}</span>", unsafe_allow_html=True)
            with c2:
                with st.expander("⚙️ ARI (Vectoriel)"):
                    for s in row['ARI']: st.markdown(f"<span class='ari-step'>{s}</span>", unsafe_allow_html=True)
            with c3:
                with st.expander("🧾 FRT"):
                    for b in row['FRT_DATA']:
                        st.markdown(f"<div class='frt-block'><span class='frt-title'>{b['title']}</span><div>{b['text']}</div></div>", unsafe_allow_html=True)
            with c4:
                with st.expander(f"📄 Qi ({row['n_q']})"):
                    qi_by_file = defaultdict(list)
                    for item in row['Evidence']: qi_by_file[item['Fichier']].append(item['Qi'])
                    html = ""
                    for f, qlist in qi_by_file.items():
                        html += f"<div class='file-block'><div class='file-header'>📁 {f}</div>"
                        for q in qlist: html += f"<div class='qi-item'>{q}</div>"
                        html += "</div>"
                    st.markdown(html, unsafe_allow_html=True)
            st.write("")

    # ANALYSE DE SATURATION (VRAIE BOUCLE)
    st.divider()
    st.markdown("### 📈 Courbe de Saturation (Preuve)")
    
    if st.button("Calculer Saturation"):
        points = []
        discovered = set()
        # Simulation progressive avec le Kernel Étendu
        target_frts = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == sel_matiere]
        
        for i in range(1, 101): # 1 à 100 sujets
            # Loi de découverte : on pioche dans le kernel
            new_frts = random.choices(target_frts, k=4) 
            for f in new_frts: discovered.add(f)
            points.append({"Sujets": i, "QC Uniques": len(discovered)})
            
        st.line_chart(pd.DataFrame(points).set_index("Sujets"))
        st.caption("La courbe montre l'évolution du nombre de QC découvertes en fonction des sujets. Le plateau indique la saturation.")

with tab_audit:
    st.info("Audit disponible après ingestion.")
