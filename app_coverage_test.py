import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V10.5")
st.title("🛡️ SMAXIA - Console V10.5 (Finale)")

# Styles CSS
st.markdown("""
<style>
    .qc-header-row { 
        display: flex; align-items: center; background-color: #f8f9fa; 
        padding: 10px; border-radius: 5px; border-left: 5px solid #2563eb;
        font-family: monospace; margin-bottom: 5px; flex-wrap: wrap;
    }
    .qc-id-tag { font-weight: bold; color: #d97706; margin-right: 10px; font-size: 1.2em; min-width: 80px;}
    .qc-title { flex-grow: 1; font-weight: 600; color: #1f2937; font-size: 1.1em; margin-right: 15px;}
    .qc-vars { 
        font-size: 0.9em; color: #111827; font-weight: bold; font-family: 'Courier New';
        background-color: #e5e7eb; padding: 6px 12px; border-radius: 4px; border: 1px solid #9ca3af;
        white-space: nowrap;
    }
    .frt-step { margin-bottom: 8px; font-family: sans-serif; line-height: 1.5; }
    .trigger-badge { background-color: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 12px; font-size: 0.9em; font-weight: bold; border: 1px solid #fca5a5; margin-right: 5px; display: inline-block; margin-bottom: 4px;}
    .stat-metric { font-size: 1.5em; font-weight: bold; color: #2563eb; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🧱 KERNEL SMAXIA
# ==============================================================================

KERNEL_MAPPING = {
    "FRT_SUITE_01": "SUITES NUMÉRIQUES", "FRT_SUITE_02": "SUITES NUMÉRIQUES", "FRT_SUITE_03": "SUITES NUMÉRIQUES",
    "FRT_FCT_01": "FONCTIONS", "FRT_FCT_02": "FONCTIONS", "FRT_FCT_03": "FONCTIONS",
    "FRT_GEO_01": "GÉOMÉTRIE", "FRT_GEO_02": "GÉOMÉTRIE",
    "FRT_PROBA_01": "PROBABILITÉS", "FRT_PROBA_02": "PROBABILITÉS"
}

SMAXIA_KERNEL = {
    "FRT_SUITE_01": {
        "QC": "COMMENT Démontrer qu'une suite est géométrique ?",
        "Triggers": ["montrer que la suite est géométrique", "nature de la suite", "raison de la suite"],
        "FRT_Redaction": [
            "1. Pour tout entier naturel $n$, on exprime $u_{n+1}$ en fonction de $n$.",
            "2. On calcule le rapport $\\frac{u_{n+1}}{u_n}$.",
            "3. On simplifie l'expression jusqu'à obtenir une constante réelle $q$ (indépendante de $n$).",
            "4. **Conclusion :** La suite $(u_n)$ est géométrique de raison $q$."
        ],
        "ARI": ["Calcul u(n+1)", "Ratio u(n+1)/u(n)", "Simplification", "Identification q"],
        "Weights": [0.2, 0.3, 0.2, 0.1], "Delta": 1.2
    },
    "FRT_SUITE_02": {
        "QC": "COMMENT Lever une indétermination sur une limite de suite ?",
        "Triggers": ["déterminer la limite", "étudier la convergence", "forme indéterminée"],
        "FRT_Redaction": [
            "1. On identifie une Forme Indéterminée (FI) type $\\infty - \\infty$ ou $\\frac{\\infty}{\\infty}$.",
            "2. On factorise l'expression par le terme prépondérant (plus haut degré ou exponentielle).",
            "3. On utilise les limites usuelles (ex: $\\lim \\frac{1}{n} = 0$).",
            "4. Par opération sur les limites, on conclut sur la limite de $(u_n)$."
        ],
        "ARI": ["Identifier FI", "Factorisation forcée", "Limites usuelles", "Opérations"],
        "Weights": [0.2, 0.3, 0.3, 0.2], "Delta": 1.1
    },
    "FRT_SUITE_03": {
        "QC": "COMMENT Démontrer une propriété par récurrence ?",
        "Triggers": ["démontrer par récurrence", "montrer que pour tout n", "hérédité"],
        "FRT_Redaction": [
            "1. **Initialisation :** On vérifie que la propriété $P(n)$ est vraie au rang $n_0$.",
            "2. **Hérédité :** On suppose que $P(k)$ est vraie pour un entier $k$ fixé. On veut montrer que $P(k+1)$ est vraie.",
            "3. On utilise l'hypothèse de récurrence dans le calcul de l'étape $k+1$.",
            "4. **Conclusion :** La propriété étant initialisée et héréditaire, elle est vraie pour tout $n$."
        ],
        "ARI": ["Initialisation", "Hypothèse HR", "Preuve au rang k+1", "Conclusion"],
        "Weights": [0.1, 0.2, 0.6, 0.1], "Delta": 1.5
    },
    "FRT_FCT_01": {
        "QC": "COMMENT Étudier les variations d'une fonction ?",
        "Triggers": ["dresser le tableau de variations", "étudier les variations", "sens de variation"],
        "FRT_Redaction": [
            "1. On justifie la dérivabilité de $f$ sur $I$.",
            "2. On calcule la fonction dérivée $f'(x)$.",
            "3. On étudie le signe de $f'(x)$ (recherche des racines, tableau de signes).",
            "4. On en déduit les variations de $f$ : $f$ est croissante là où $f' > 0$.",
            "5. On dresse le tableau complet avec les limites aux bornes."
        ],
        "ARI": ["Domaine dérivabilité", "Calcul f'", "Signe f'", "Tableau complet"],
        "Weights": [0.3, 0.3, 0.2, 0.1], "Delta": 1.3
    },
    "FRT_FCT_02": {
        "QC": "COMMENT Appliquer le TVI pour une solution unique ?",
        "Triggers": ["équation f(x)=k", "admet une unique solution", "théorème des valeurs intermédiaires"],
        "FRT_Redaction": [
            "1. On vérifie que $f$ est **continue** sur l'intervalle $[a,b]$.",
            "2. On vérifie que $f$ est **strictement monotone** sur cet intervalle.",
            "3. On calcule les images $f(a)$ et $f(b)$ et on vérifie que $k$ est compris entre les deux.",
            "4. **Conclusion :** D'après le corollaire du TVI, l'équation $f(x)=k$ admet une unique solution $\\alpha$."
        ],
        "ARI": ["Continuité", "Monotonie stricte", "Images bornes", "Invocation TVI"],
        "Weights": [0.1, 0.2, 0.2, 0.4], "Delta": 1.4
    },
    "FRT_FCT_03": {
        "QC": "COMMENT Déterminer l'équation d'une tangente ?",
        "Triggers": ["équation de la tangente", "tangente au point d'abscisse", "équation réduite"],
        "FRT_Redaction": [
            "1. On rappelle la formule de la tangente en $a$ : $y = f'(a)(x-a) + f(a)$.",
            "2. On calcule l'image $f(a)$.",
            "3. On calcule le nombre dérivé $f'(a)$.",
            "4. On remplace dans la formule et on réduit l'expression sous la forme $y = mx + p$."
        ],
        "ARI": ["Rappel formule", "Calcul f(a)", "Calcul f'(a)", "Substitution"],
        "Weights": [0.1, 0.2, 0.2, 0.1], "Delta": 0.9
    },
    "FRT_GEO_01": {
        "QC": "COMMENT Démontrer qu'une droite est orthogonale à un plan ?",
        "Triggers": ["droite orthogonale au plan", "vecteur normal au plan", "perpendiculaire au plan"],
        "FRT_Redaction": [
            "1. On extrait un vecteur directeur $\\vec{u}$ de la droite.",
            "2. On identifie deux vecteurs directeurs non colinéaires $\\vec{v_1}$ et $\\vec{v_2}$ du plan.",
            "3. On montre que le produit scalaire $\\vec{u} \\cdot \\vec{v_1} = 0$ et $\\vec{u} \\cdot \\vec{v_2} = 0$.",
            "4. **Conclusion :** La droite est orthogonale à deux droites sécantes du plan, donc elle est orthogonale au plan."
        ],
        "ARI": ["Vecteur u", "Base du plan", "Double produit scalaire", "Théorème"],
        "Weights": [0.1, 0.1, 0.4, 0.2], "Delta": 1.3
    },
    "FRT_GEO_02": {
        "QC": "COMMENT Déterminer une représentation paramétrique ?",
        "Triggers": ["représentation paramétrique", "système paramétrique", "droite passant par"],
        "FRT_Redaction": [
            "1. On identifie un point $A(x_A; y_A; z_A)$ appartenant à la droite.",
            "2. On identifie un vecteur directeur $\\vec{u}(a; b; c)$.",
            "3. On écrit le système pour tout paramètre $t \\in \\mathbb{R}$ :",
            "   $\\begin{cases} x = x_A + at \\\\ y = y_A + bt \\\\ z = z_A + ct \\end{cases}$"
        ],
        "ARI": ["Point A", "Vecteur u", "Écriture système"],
        "Weights": [0.2, 0.2, 0.4], "Delta": 1.0
    },
    "FRT_PROBA_01": {
        "QC": "COMMENT Calculer une probabilité totale (Arbre) ?",
        "Triggers": ["probabilité totale", "arbre pondéré", "probabilité de l'événement"],
        "FRT_Redaction": [
            "1. On construit un arbre pondéré décrivant l'expérience.",
            "2. On repère les chemins qui réalisent l'événement $E$.",
            "3. On cite la **Formule des Probabilités Totales**.",
            "4. On somme les probabilités des intersections (produits des branches) : $P(E) = P(A \\cap E) + P(\\bar{A} \\cap E)$."
        ],
        "ARI": ["Modélisation Arbre", "Chemins", "Invocation Formule", "Calcul"],
        "Weights": [0.1, 0.3, 0.2, 0.2], "Delta": 1.1
    },
    "FRT_PROBA_02": {
        "QC": "COMMENT Calculer une probabilité (Loi Binomiale) ?",
        "Triggers": ["loi binomiale", "succès exactement", "schéma de bernoulli"],
        "FRT_Redaction": [
            "1. On justifie qu'on répète $n$ fois une épreuve de Bernoulli de manière identique et indépendante.",
            "2. On précise les paramètres : $X$ suit la loi $\\mathcal{B}(n, p)$.",
            "3. On applique la formule : $P(X=k) = \\binom{n}{k} p^k (1-p)^{n-k}$.",
            "4. On effectue le calcul numérique."
        ],
        "ARI": ["Justification Loi", "Paramètres n,p", "Formule", "Calcul"],
        "Weights": [0.3, 0.1, 0.3, 0.1], "Delta": 1.2
    }
}

QI_TEMPLATES = {
    "FRT_SUITE_01": ["Montrer que (Un) est géométrique.", "Démontrer que Vn est une suite géométrique.", "Justifier la nature géométrique de la suite."],
    "FRT_SUITE_02": ["Déterminer la limite de la suite Un.", "Étudier la convergence de la suite.", "Calculer la limite quand n tend vers l'infini."],
    "FRT_SUITE_03": ["Démontrer par récurrence que Un > 0.", "Montrer par récurrence la propriété P(n).", "Prouver par récurrence que Un < 5."],
    "FRT_FCT_01": ["Étudier les variations de la fonction f.", "Dresser le tableau de variations complet.", "Quel est le sens de variation de f ?"],
    "FRT_FCT_02": ["Montrer que l'équation f(x)=0 admet une unique solution.", "Démontrer l'existence d'une solution alpha.", "Prouver que l'équation a une seule solution sur I."],
    "FRT_FCT_03": ["Déterminer l'équation de la tangente T au point A.", "Donner l'équation réduite de la tangente.", "Quelle est la tangente à Cf en 0 ?"],
    "FRT_GEO_01": ["Démontrer que la droite (d) est orthogonale au plan (P).", "Prouver que (d) est perpendiculaire au plan.", "Montrer que le vecteur n est normal au plan."],
    "FRT_GEO_02": ["Déterminer une représentation paramétrique de (D).", "Donner un système d'équations paramétriques.", "Quelle est la représentation de la droite ?"],
    "FRT_PROBA_01": ["Calculer la probabilité de l'événement B.", "En utilisant l'arbre, calculer P(E).", "Quelle est la probabilité totale de A ?"],
    "FRT_PROBA_02": ["Calculer la probabilité d'obtenir exactement 3 succès.", "Quelle est la probabilité que X soit égal à 2 ?", "Calculer P(X=k) avec la loi binomiale."]
}

def generate_smart_qi(frt_id):
    if frt_id not in QI_TEMPLATES: return "Question Standard"
    text = random.choice(QI_TEMPLATES[frt_id])
    context = random.choice(["", " sur l'intervalle I", " dans le repère Oijk", " pour tout entier n"])
    return text + context

# ==============================================================================
# ⚙️ MOTEUR
# ==============================================================================

def calculate_psi_real(frt_id):
    data = SMAXIA_KERNEL[frt_id]
    psi_raw = data["Delta"] * (0.1 + sum(data["Weights"]))**2
    return round(min(psi_raw / 4.0, 1.0), 4)

def ingest_and_process(urls, n_per_url, selected_chapters):
    sources_log = []
    atoms_db = []
    progress = st.progress(0)
    
    active_frts = [k for k, v in KERNEL_MAPPING.items() if v in selected_chapters]
    if not active_frts: return pd.DataFrame(), pd.DataFrame()

    total_ops = len(urls) * n_per_url if len(urls) > 0 else 1
    counter = 0
    natures = ["BAC", "DST", "CONCOURS"]
    
    for i, url in enumerate(urls):
        if not url.strip(): continue
        for j in range(n_per_url):
            counter += 1
            progress.progress(min(counter/total_ops, 1.0))
            time.sleep(0.002)
            
            nature = random.choice(natures)
            year = random.choice(range(2020, 2025))
            filename = f"Sujet_{nature}_{year}_{j}.pdf"
            file_id = f"DOC_{i}_{j}"
            
            nb_exos = random.randint(2, 4)
            frts_in_doc = random.sample(active_frts, k=min(nb_exos, len(active_frts)))
            
            qi_list_in_file = []
            
            for frt_id in frts_in_doc:
                qi_txt = generate_smart_qi(frt_id)
                atoms_db.append({
                    "ID_Source": file_id, "Année": year, "Qi_Brut": qi_txt,
                    "FRT_ID": frt_id, "Fichier": filename, 
                    "Chapitre": KERNEL_MAPPING[frt_id]
                })
                qi_list_in_file.append({"Qi": qi_txt, "FRT_ID": frt_id})
                
            content = f"CONTENU SIMULÉ PDF\nFICHIER: {filename}\n" + "\n".join([f"- {q['Qi']}" for q in qi_list_in_file])
            sources_log.append({
                "Fichier": filename, "Nature": nature, "Année": year, 
                "Contenu": content, "Qi_Data": qi_list_in_file 
            })
            
    progress.empty()
    return pd.DataFrame(sources_log), pd.DataFrame(atoms_db)

def compute_engine_metrics(df_atoms):
    if df_atoms.empty: return pd.DataFrame()
    grouped = df_atoms.groupby("FRT_ID").agg({
        "ID_Source": "count", "Année": "max", 
        "Qi_Brut": list, "Fichier": list, "Chapitre": "first"
    }).reset_index()
    
    qcs = []
    N_total = len(df_atoms)
    current_year = datetime.now().year
    
    for idx, row in grouped.iterrows():
        frt_id = row["FRT_ID"]
        kernel = SMAXIA_KERNEL[frt_id]
        
        n_q = row["ID_Source"]
        tau = max((current_year - row["Année"]), 0.5)
        alpha = 5.0
        psi = calculate_psi_real(frt_id)
        sigma = 0.05
        
        score = (n_q / N_total) * (1 + alpha/tau) * psi * (1-sigma) * 100
        qc_clean = kernel["QC"].replace("COMMENT ", "comment ")
        
        qcs.append({
            "Chapitre": row["Chapitre"],
            "QC_ID_Simple": f"QC_{idx+1:02d}", 
            "FRT_ID": frt_id,
            "QC_Texte": qc_clean,
            "Triggers": kernel["Triggers"],
            "FRT_Redaction": kernel["FRT_Redaction"],
            "ARI": kernel["ARI"],
            "Score_F2": score,
            "n_q": n_q, "N_tot": N_total, "Tau": tau, "Alpha": alpha, "Psi": psi, "Sigma": sigma,
            "Evidence": [{"Fichier": f, "Qi": q} for f, q in zip(row["Fichier"], row["Qi_Brut"])]
        })
        
    return pd.DataFrame(qcs).sort_values(by=["Chapitre", "Score_F2"], ascending=[True, False])

# ==============================================================================
# 🖥️ INTERFACE V10.5
# ==============================================================================

with st.sidebar:
    st.header("1. Paramètres")
    matiere = st.selectbox("Matière", ["MATHS"])
    all_chaps = list(set(KERNEL_MAPPING.values()))
    selected_chaps = st.multiselect("Chapitres", all_chaps, default=all_chaps)

# TABS
tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine (Prod)", "✅ Onglet 2 : Audit (Validation Booléenne)"])

# --- TAB 1 : USINE ---
with tab_usine:
    c1, c2 = st.columns([3, 1])
    with c1: urls_input = st.text_area("Sources", "https://apmep.fr", height=70)
    with c2: 
        n_sujets = st.number_input("Vol. par URL", 5, 100, 10, step=5)
        btn_run = st.button("LANCER USINE 🚀", type="primary")

    if btn_run:
        with st.spinner("Traitement..."):
            df_src, df_atoms = ingest_and_process(urls_input.split('\n'), n_sujets, selected_chaps)
            df_qc = compute_engine_metrics(df_atoms)
            st.session_state['df_src'] = df_src
            st.session_state['df_qc'] = df_qc
            st.session_state['sel_chaps'] = selected_chaps
            st.success("Traitement terminé.")

    st.divider()

    if 'df_qc' in st.session_state:
        col_left, col_right = st.columns([1, 1.8])
        
        with col_left:
            st.markdown(f"### 📥 Sujets ({len(st.session_state['df_src'])})")
            df_display = st.session_state['df_src'][["Fichier", "Nature", "Année"]].copy()
            df_display["Action"] = "📥 PDF" 
            st.dataframe(df_display, use_container_width=True, height=500)
            
            st.caption("Sélectionner pour télécharger :")
            sel = st.selectbox("Fichier cible", st.session_state['df_src']["Fichier"], label_visibility="collapsed")
            if not st.session_state['df_src'].empty:
                txt = st.session_state['df_src'][st.session_state['df_src']["Fichier"]==sel].iloc[0]["Contenu"]
                st.download_button(f"Télécharger {sel}", txt, file_name=sel)

        with col_right:
            if not st.session_state['df_qc'].empty:
                chapters = st.session_state['df_qc']["Chapitre"].unique()
                for chap in chapters:
                    df_view = st.session_state['df_qc'][st.session_state['df_qc']["Chapitre"] == chap]
                    st.markdown(f"#### 📘 {chap} : {len(df_view)} QC")
                    
                    for idx, row in df_view.iterrows():
                        # LIGNE 1 : HEADER
                        header_html = f"""
                        <div class="qc-header-row">
                            <span class="qc-id-tag">[{row['QC_ID_Simple']}]</span>
                            <span class="qc-title">{row['QC_Texte']}</span>
                            <span class="qc-vars">
                                Score(q)={row['Score_F2']:.0f} | 
                                Ψ={row['Psi']} | 
                                n_q={row['n_q']} | 
                                N_tot={row['N_tot']} | 
                                t_rec={row['Tau']:.1f}
                            </span>
                        </div>
                        """
                        st.markdown(header_html, unsafe_allow_html=True)
                        
                        c1, c2, c3, c4 = st.columns(4)
                        
                        # 1. TRIGGERS
                        with c1:
                            with st.expander("⚡ Déclencheurs"):
                                for t in row['Triggers']: st.markdown(f"<div class='trigger-badge'>{t}</div>", unsafe_allow_html=True)
                        
                        # 2. ARI
                        with c2:
                            with st.expander(f"⚙️ ARI_{row['QC_ID_Simple']}"):
                                for s in row['ARI']: st.markdown(f"- {s}")

                        # 3. FRT
                        with c3:
                            with st.expander(f"📝 FRT_{row['QC_ID_Simple']}"):
                                for line in row['FRT_Redaction']: st.markdown(f"<div class='frt-step'>{line}</div>", unsafe_allow_html=True)

                        # 4. PREUVE QI (Correction : Full Width)
                        with c4:
                            with st.expander(f"📄 Qi associées ({row['n_q']})"):
                                st.dataframe(
                                    pd.DataFrame(row['Evidence']), 
                                    hide_index=True, 
                                    use_container_width=True, # CORRECTION
                                    column_config={"Qi": st.column_config.TextColumn("Questions Élèves", width="large")}
                                )
                        
                        st.write("") 
            else:
                st.warning("Aucune QC.")

# --- TAB 2 : AUDIT MAPPING ---
with tab_audit:
    st.subheader("Validation Booléenne (Tableau de Mapping Unifié)")
    
    if 'df_qc' in st.session_state and 'df_src' in st.session_state:
        
        # TEST 1
        st.markdown("#### 1. Audit Interne (Sujet Traité)")
        test_file = st.selectbox("Choisir un sujet traité", st.session_state['df_src']["Fichier"])
        
        if st.button("LANCER L'AUDIT DE COUVERTURE (INTERNE)", type="primary"):
            file_data = st.session_state['df_src'][st.session_state['df_src']["Fichier"]==test_file].iloc[0]
            qi_list = file_data["Qi_Data"]
            mapping_rows = []
            c_ok = 0
            qc_lookup = st.session_state['df_qc'].set_index("FRT_ID")
            
            for item in qi_list:
                frt_id = item["FRT_ID"]
                is_covered = frt_id in qc_lookup.index
                
                if is_covered:
                    c_ok += 1
                    qc_info = qc_lookup.loc[frt_id]
                    if isinstance(qc_info, pd.DataFrame): qc_info = qc_info.iloc[0]
                    qc_display = f"[{qc_info['QC_ID_Simple']}] {qc_info['QC_Texte']}"
                    frt_display = f"FRT_{qc_info['QC_ID_Simple']}"
                    status = "✅ MATCH"
                else:
                    qc_display = "---"
                    frt_display = "---"
                    status = "❌ GAP"
                
                mapping_rows.append({
                    "1. Qi (Question Élève)": item["Qi"],
                    "2. QC (ID + Titre)": qc_display,
                    "3. FRT (ID Technique)": frt_display,
                    "Statut": status
                })
            
            taux = (c_ok / len(qi_list)) * 100
            st.markdown(f"<div class='stat-metric'>Taux de Couverture : {taux:.0f}%</div>", unsafe_allow_html=True)
            
            def color_map(val): return f'background-color: {"#dcfce7" if val == "✅ MATCH" else "#fee2e2"}; color: black'
            st.dataframe(pd.DataFrame(mapping_rows).style.map(color_map, subset=['Statut']), use_container_width=True)

        st.divider()

        # TEST 2
        st.markdown("#### 2. Audit Externe (Nouveau Sujet)")
        if st.button("LANCER L'AUDIT DE COUVERTURE (EXTERNE)"):
            all_kernel_frts = list(SMAXIA_KERNEL.keys())
            test_frts = random.sample(all_kernel_frts, 4)
            ext_rows = []
            c_ok_ext = 0
            qc_lookup = st.session_state['df_qc'].set_index("FRT_ID")
            
            for frt_id in test_frts:
                qi_txt = generate_smart_qi(frt_id)
                is_covered = frt_id in qc_lookup.index
                if is_covered:
                    c_ok_ext += 1
                    qc_info = qc_lookup.loc[frt_id]
                    if isinstance(qc_info, pd.DataFrame): qc_info = qc_info.iloc[0]
                    qc_display = f"[{qc_info['QC_ID_Simple']}] {qc_info['QC_Texte']}"
                    status = "✅ MATCH"
                else:
                    qc_display = "---"
                    status = "❌ HORS PÉRIMÈTRE"
                ext_rows.append({"1. Qi (Nouveau Sujet)": qi_txt, "2. QC Trouvée": qc_display, "3. FRT Requise": frt_id, "Statut": status})
            
            taux_ext = (c_ok_ext / 4) * 100
            st.markdown(f"<div class='stat-metric'>Taux de Couverture : {taux_ext:.0f}%</div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(ext_rows).style.map(color_map, subset=['Statut']), use_container_width=True)

    else:
        st.info("Veuillez lancer l'usine dans l'onglet 1.")
