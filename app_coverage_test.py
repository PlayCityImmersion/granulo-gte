import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V17")
st.title("🛡️ SMAXIA - Console V17 (Vertical Layout & External Upload)")

# ==============================================================================
# 🎨 STYLES CSS
# ==============================================================================
st.markdown("""
<style>
    /* LIGNE PRINCIPALE QC */
    .qc-header-row {
        display: flex; align-items: center; justify-content: space-between;
        background-color: #f8f9fa; border-left: 5px solid #2563eb;
        padding: 12px 15px; margin-bottom: 8px; border-radius: 4px;
        font-family: 'Source Sans Pro', sans-serif;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .qc-title-group { display: flex; align-items: center; flex-grow: 1; }
    .qc-id { color: #d97706; font-weight: 800; font-size: 1.1em; margin-right: 15px; min-width: 80px; }
    .qc-text { color: #111827; font-weight: 600; font-size: 1.1em; }
    .qc-vars { 
        font-family: 'Courier New', monospace; font-size: 0.9em; font-weight: 700; color: #4b5563;
        background-color: #e5e7eb; padding: 5px 10px; border-radius: 4px; white-space: nowrap;
    }

    /* DETAILS */
    .trigger-box { background-color: #fef2f2; border: 1px solid #fecaca; color: #991b1b; padding: 10px; border-radius: 6px; font-weight: 600; }
    .ari-box { background-color: #f3f4f6; border: 1px dashed #d1d5db; color: #374151; padding: 10px; border-radius: 6px; font-family: monospace; font-size: 0.9em; }
    .frt-box { background-color: #ffffff; border: 1px solid #10b981; border-left: 5px solid #10b981; padding: 15px; border-radius: 6px; font-family: sans-serif; line-height: 1.6; color: #064e3b; }
    
    /* TABLEAUX */
    .qi-table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
    .qi-table th { background: #f9fafb; text-align: left; padding: 8px; border-bottom: 2px solid #e5e7eb; color: #6b7280; }
    .qi-table td { padding: 8px; border-bottom: 1px solid #f3f4f6; vertical-align: top; color: #1f2937; }
    
    /* AUDIT MATCHING */
    .match-success { background-color: #dcfce7; color: #166534; font-weight: bold; padding: 4px 8px; border-radius: 4px; }
    .match-fail { background-color: #fee2e2; color: #991b1b; font-weight: bold; padding: 4px 8px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🧠 KERNEL SMAXIA
# ==============================================================================

UNIVERS_SMAXIA = {
    # --- MATHS ---
    "FRT_M_SUITE_01": {
        "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", "Proba": 0.9,
        "QC": "comment démontrer qu'une suite est géométrique ?",
        "Trigger": "L'énoncé demande la nature de la suite définie par une relation $u_{n+1} = f(u_n)$ multiplicative.",
        "ARI": ["Calcul u(n+1)", "Ratio u(n+1)/u(n)", "Simplification", "Identification Constante"],
        "FRT": """1. Pour tout entier $n$, j'exprime $u_{n+1}$ en fonction de $n$.\n2. Je forme le rapport $\\frac{u_{n+1}}{u_n}$.\n3. Je simplifie jusqu'à trouver une constante réelle $q$.\n4. **Conclusion :** La suite est géométrique de raison $q$."""
    },
    "FRT_M_SUITE_02": {
        "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", "Proba": 0.8,
        "QC": "comment lever une indétermination (limite) ?",
        "Trigger": "Présence de termes de même ordre en $n$ (polynômes/fractions) créant un conflit $\\infty - \\infty$ ou $\\infty / \\infty$.",
        "ARI": ["Identifier Dominant", "Factorisation Forcée", "Limites Usuelles", "Opérations"],
        "FRT": """1. J'identifie le terme de plus haut degré (le terme dominant).\n2. Je factorise toute l'expression par ce terme.\n3. J'utilise $\\lim \\frac{1}{n} = 0$.\n4. Je conclus par produit ou somme de limites."""
    },
    "FRT_M_FCT_01": {
        "Matiere": "MATHS", "Chap": "FONCTIONS", "Proba": 0.9,
        "QC": "comment étudier les variations d'une fonction ?",
        "Trigger": "L'énoncé demande explicitement le 'sens de variation' ou de 'dresser le tableau'.",
        "ARI": ["Dérivabilité", "Calcul f'", "Signe f'", "Tableau"],
        "FRT": """1. Je justifie que $f$ est dérivable sur $I$.\n2. Je calcule la dérivée $f'(x)$.\n3. J'étudie le signe de $f'(x)$ (racines, signe).\n4. Si $f' > 0$, $f$ est croissante. Je dresse le tableau complet."""
    },
    "FRT_M_FCT_02": {
        "Matiere": "MATHS", "Chap": "FONCTIONS", "Proba": 0.7,
        "QC": "comment appliquer le TVI (solution unique) ?",
        "Trigger": "Montrer que l'équation $f(x)=k$ admet une **unique** solution $\\alpha$.",
        "ARI": ["Continuité", "Monotonie Stricte", "Images Bornes", "Corollaire TVI"],
        "FRT": """1. Je cite que $f$ est **continue** et **strictement monotone** sur l'intervalle.\n2. Je calcule les images des bornes.\n3. Je vérifie que $k$ est compris entre ces images.\n4. J'invoque le corollaire du Théorème des Valeurs Intermédiaires."""
    },
    # --- PHYSIQUE ---
    "FRT_P_MECA_01": {
        "Matiere": "PHYSIQUE", "Chap": "MÉCANIQUE", "Proba": 0.9,
        "QC": "comment déterminer le vecteur accélération ?",
        "Trigger": "Demande des coordonnées de l'accélération ou application de la 2e Loi de Newton.",
        "ARI": ["Référentiel", "Bilan Forces", "Sigma F = ma", "Projection"],
        "FRT": """1. Je définis le système et le référentiel galiléen.\n2. Je fais le bilan des forces extérieures.\n3. J'applique la 2e Loi de Newton : $\\sum \\vec{F}_{ext} = m\\vec{a}$.\n4. Je projette la relation sur les axes $(Ox, Oy)$."""
    }
}

QI_PATTERNS = {
    "FRT_M_SUITE_01": ["Montrer que (Un) est géométrique.", "Quelle est la nature de la suite (Vn) ?", "Justifier que la suite est géométrique."],
    "FRT_M_SUITE_02": ["Déterminer la limite de la suite.", "Lever l'indétermination pour calculer la limite.", "Étudier la convergence en +l'infini."],
    "FRT_M_FCT_01": ["Étudier les variations de f.", "Donner le sens de variation de la fonction.", "Dresser le tableau de variations."],
    "FRT_M_FCT_02": ["Montrer que l'équation f(x)=0 a une unique solution.", "Démontrer qu'il existe un unique alpha tel que f(alpha)=3."],
    "FRT_P_MECA_01": ["En déduire les coordonnées du vecteur accélération.", "Déterminer l'expression de a(t)."]
}

# ==============================================================================
# ⚙️ MOTEUR
# ==============================================================================

def ingest_factory(urls, volume, matiere, chapitres):
    universe = [k for k, v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere and v["Chap"] in chapitres]
    if not universe: return pd.DataFrame(), pd.DataFrame()
    
    sources, atoms = [], []
    progress = st.progress(0)
    
    for i in range(volume):
        progress.progress((i+1)/volume)
        nature = random.choice(["BAC", "DST", "INTERRO", "CONCOURS"])
        annee = random.choice(range(2020, 2025))
        filename = f"Sujet_{matiere}_{nature}_{annee}_{i}.pdf"
        
        weights = [UNIVERS_SMAXIA[k]["Proba"] for k in universe]
        drawn_frts = random.choices(universe, weights=weights, k=random.randint(2, 4))
        
        qi_data_list = []
        for frt_id in drawn_frts:
            qi_txt = random.choice(QI_PATTERNS[frt_id]) + f" ({random.randint(100,999)})"
            atoms.append({"FRT_ID": frt_id, "Qi": qi_txt, "File": filename, "Year": annee, "Chap": UNIVERS_SMAXIA[frt_id]["Chap"]})
            qi_data_list.append({"Qi": qi_txt, "FRT_ID": frt_id})
            
        sources.append({"Fichier": filename, "Nature": nature, "Année": annee, "Télécharger": "📥 PDF", "Qi_Data": qi_data_list})
        
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
            "Trigger": meta["Trigger"], "ARI": meta["ARI"], "FRT": meta["FRT"],
            "Evidence": [{"Fichier": f, "Qi": q} for f, q in zip(row["File"], row["Qi"])]
        })
    return pd.DataFrame(qcs).sort_values(by="Score", ascending=False)

def analyze_external_file(file, matiere, chapitres):
    """Simule l'extraction d'un fichier uploadé"""
    # Dans la simulation, on génère des Qi aléatoires basées sur l'univers
    # pour montrer le comportement du mapping.
    universe = [k for k, v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere and v["Chap"] in chapitres]
    if not universe: return []
    
    nb_qi = random.randint(3, 6)
    drawn_frts = random.sample(universe, k=min(len(universe), nb_qi))
    
    qi_extracted = []
    for frt_id in drawn_frts:
        qi_txt = random.choice(QI_PATTERNS[frt_id]) + " (Source Externe)"
        qi_extracted.append({"Qi": qi_txt, "FRT_ID": frt_id})
        
    return qi_extracted

# ==============================================================================
# 🖥️ UI
# ==============================================================================

with st.sidebar:
    st.header("Paramètres Académiques")
    st.selectbox("Classe", ["Terminale"], disabled=True)
    sel_matiere = st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    chaps_dispo = sorted(list(set([v["Chap"] for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == sel_matiere])))
    sel_chapitres = st.multiselect("Chapitre", chaps_dispo, default=chaps_dispo)

tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine", "✅ Onglet 2 : Audit"])

# --- ONGLET 1 : USINE (LAYOUT VERTICAL) ---
with tab_usine:
    # 1. ZONE URL (TOP)
    st.subheader("1. Configuration Sourcing")
    c1, c2 = st.columns([3, 1])
    with c1: urls = st.text_area("URLs Sources", "https://apmep.fr", height=70)
    with c2: 
        vol = st.number_input("Volume", 5, 500, 15, step=5)
        run = st.button("LANCER L'USINE 🚀", type="primary")

    if run:
        df_src, df_atoms = ingest_factory(urls.split('\n'), vol, sel_matiere, sel_chapitres)
        df_qc = compute_qc(df_atoms)
        st.session_state['df_src'] = df_src
        st.session_state['df_qc'] = df_qc
        st.success(f"Ingestion terminée : {len(df_src)} sujets traités.")

    st.divider()

    if 'df_qc' in st.session_state:
        # 2. TABLEAU SUJETS (MILIEU - FULL WIDTH)
        st.markdown(f"### 📥 Sujets Traités ({len(st.session_state['df_src'])})")
        st.dataframe(
            st.session_state['df_src'][["Fichier", "Nature", "Année", "Téléchargement"]],
            use_container_width=True, height=300, hide_index=True
        )
        
        st.divider()

        # 3. LISTE QC (BAS - FULL WIDTH)
        st.markdown(f"### 🧠 Base de Connaissance (QC)")
        
        if not st.session_state['df_qc'].empty:
            chapters = st.session_state['df_qc']["Chapitre"].unique()
            for chap in chapters:
                subset = st.session_state['df_qc'][st.session_state['df_qc']["Chapitre"] == chap]
                st.markdown(f"#### 📘 Chapitre {chap} : {len(subset)} QC")
                
                for idx, row in subset.iterrows():
                    # HEADER
                    st.markdown(f"""
                    <div class="qc-header-row">
                        <div class="qc-title-group">
                            <span class="qc-id">{row['QC_ID']}</span>
                            <span class="qc-text">{row['Titre']}</span>
                        </div>
                        <span class="qc-vars">Score(q)={row['Score']:.0f} | n_q={row['n_q']} | Ψ={row['Psi']} | N_tot={row['N_tot']} | t_rec={row['t_rec']:.1f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # DETAILS
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        with st.expander("🔥 Déclencheurs"):
                            st.markdown(f"<div class='trigger-box'>{row['Trigger']}</div>", unsafe_allow_html=True)
                    with c2:
                        with st.expander("⚙️ ARI"):
                            st.markdown("<div class='ari-box'>", unsafe_allow_html=True)
                            for s in row['ARI']: st.markdown(f"- {s}")
                            st.markdown("</div>", unsafe_allow_html=True)
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

# --- ONGLET 2 : AUDIT ---
with tab_audit:
    st.subheader("Validation Booléenne")
    
    if 'df_qc' in st.session_state:
        # TEST 1 : INTERNE
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
                
                qc_disp = "---"
                if is_ok:
                    qc_row = st.session_state['df_qc'][st.session_state['df_qc']["FRT_ID"]==item["FRT_ID"]].iloc[0]
                    qc_disp = f"{qc_row['QC_ID']} {qc_row['Titre']}"
                
                status = "✅ MATCH" if is_ok else "❌ ERREUR"
                rows.append({"Qi (Sujet)": item["Qi"], "QC Moteur": qc_disp, "Statut": status})
            
            taux = (ok_count / len(data)) * 100
            st.metric("Taux Couverture Interne", f"{taux:.0f}%")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.divider()

        # TEST 2 : EXTERNE (UPLOAD)
        st.markdown("#### 🌍 2. Test Externe (Sujet Inconnu)")
        uploaded_file = st.file_uploader("Télécharger un Sujet Externe (PDF)", type="pdf")
        
        if uploaded_file is not None:
            # Simulation Extraction
            qi_extracted = analyze_external_file(uploaded_file, sel_matiere, sel_chapitres)
            
            if not qi_extracted:
                st.warning("Aucune question détectée ou hors périmètre.")
            else:
                known_ids = st.session_state['df_qc']["FRT_ID"].unique()
                match_count = 0
                ext_rows = []
                
                for item in qi_extracted:
                    is_ok = item["FRT_ID"] in known_ids
                    if is_ok: match_count += 1
                    
                    qc_disp = "Pas de QC"
                    frt_disp = item["FRT_ID"]
                    if is_ok:
                        qc_row = st.session_state['df_qc'][st.session_state['df_qc']["FRT_ID"]==item["FRT_ID"]].iloc[0]
                        qc_disp = f"{qc_row['QC_ID']} {qc_row['Titre']}"
                    
                    status = "✅ MATCH" if is_ok else "❌ GAP"
                    ext_rows.append({
                        "Qi (Enoncé)": item["Qi"], 
                        "QC Correspondante": qc_disp, 
                        "FRT Associé": frt_disp,
                        "Statut": status # Pour coloration
                    })
                
                taux_ext = (match_count / len(qi_extracted)) * 100
                st.markdown(f"### Taux de Couverture = {taux_ext:.1f}%")
                
                def color_audit(row):
                    return ['background-color: #dcfce7' if row['Statut'] == "✅ MATCH" else 'background-color: #fee2e2'] * len(row)

                st.dataframe(pd.DataFrame(ext_rows).style.apply(color_audit, axis=1), use_container_width=True)

    else:
        st.info("Veuillez lancer l'usine d'abord.")
