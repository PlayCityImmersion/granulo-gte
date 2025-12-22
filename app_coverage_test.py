import streamlit as st
import pandas as pd
import numpy as np
import random
import io
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V19")
st.title("🛡️ SMAXIA - Console V19 (Audit Compliant)")

# ==============================================================================
# 🎨 STYLES CSS (ALIGNÉS SUR DEMANDE AUDIT)
# ==============================================================================
st.markdown("""
<style>
    /* EN-TÊTE QC STRICT */
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
# 1. LISTE CHAPITRES (COMPLÈTE)
# ==============================================================================
LISTE_CHAPITRES = {
    "MATHS": [
        "SUITES NUMÉRIQUES", "FONCTIONS & DÉRIVATION", "LIMITES DE FONCTIONS", 
        "CONTINUITÉ & CONVEXITÉ", "FONCTION LOGARITHME", "PRIMITIVES & ÉQUATIONS DIFF", 
        "CALCUL INTÉGRAL", "COMBINATOIRE & DÉNOMBREMENT", "PROBABILITÉS DISCRÈTES", 
        "LOI BINOMIALE", "GÉOMÉTRIE DANS L'ESPACE", "ORTHOGONALITÉ & DISTANCES"
    ],
    "PHYSIQUE": [
        "MOUVEMENT & INTERACTIONS", "MÉCANIQUE DE NEWTON", "MOUVEMENT DANS UN CHAMP", 
        "THERMODYNAMIQUE", "ONDES MÉCANIQUES", "LUMIÈRE & ONDES", "TRANSFORMATIONS CHIMIQUES"
    ]
}

# ==============================================================================
# 2. KERNEL SMAXIA (CONTENU VALIDÉ)
# ==============================================================================

UNIVERS_SMAXIA = {
    # --- MATHS ---
    "FRT_M_SUITE_01": {
        "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", "Proba": 0.9,
        "QC": "comment démontrer qu'une suite est géométrique ?",
        # Déclencheurs Multiples & Observables
        "Triggers": ["montrer que la suite est géométrique", "quelle est la nature de la suite", "déterminer la raison q", "justifier que (Un) est géométrique"],
        "ARI": ["Calcul u(n+1)", "Ratio u(n+1)/u(n)", "Simplification", "Identification Constante"],
        # FRT Complète (Méthode + Pièges + Conclusion)
        "FRT": """🔔 **Quand utiliser ?** Lorsque l'énoncé demande la nature de la suite ou d'identifier une suite géométrique.\n\n✅ **Méthode Standard :**\n1. Exprimer $u_{n+1}$ en fonction de $n$.\n2. Calculer le rapport $\\frac{u_{n+1}}{u_n}$.\n3. Simplifier jusqu'à obtenir une constante réelle $q$.\n\n⚠️ **Pièges :** Ne pas vérifier que $u_n \\neq 0$. Confondre avec suite arithmétique.\n\n✍️ **Conclusion Type :** "Le rapport est constant égal à $q$, donc la suite est géométrique de raison $q$." """
    },
    "FRT_M_SUITE_02": {
        "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", "Proba": 0.8,
        "QC": "comment lever une indétermination (limite) ?",
        "Triggers": ["déterminer la limite", "calculer la limite quand n tend vers l'infini", "étudier la convergence", "limite de la suite"],
        "ARI": ["Identifier FI", "Factoriser terme dominant", "Limites usuelles", "Opérations"],
        "FRT": """🔔 **Quand utiliser ?** Présence d'une forme indéterminée ($\\infty - \\infty$ ou $\\infty / \\infty$).\n\n✅ **Méthode Standard :**\n1. Identifier le terme de plus haut degré (dominant).\n2. Factoriser toute l'expression par ce terme.\n3. Utiliser $\\lim 1/n = 0$.\n\n⚠️ **Pièges :** Appliquer la règle des signes sans factoriser.\n\n✍️ **Conclusion Type :** "Par produit/somme de limites, $\\lim u_n = \\dots$." """
    },
    "FRT_M_FCT_01": {
        "Matiere": "MATHS", "Chap": "FONCTIONS & DÉRIVATION", "Proba": 0.9,
        "QC": "comment étudier les variations d'une fonction ?",
        "Triggers": ["étudier le sens de variation", "dresser le tableau de variations", "variations de f", "f est-elle croissante"],
        "ARI": ["Dérivée f'", "Signe f'", "Tableau"],
        "FRT": """🔔 **Quand utiliser ?** Pour connaitre la croissance/décroissance.\n\n✅ **Méthode Standard :**\n1. Calculer la dérivée $f'(x)$.\n2. Étudier le signe de $f'(x)$.\n3. Conclure : $f' > 0 \\Rightarrow f$ croissante.\n\n⚠️ **Pièges :** Confondre signe de f et variations de f.\n\n✍️ **Conclusion Type :** "La dérivée étant positive sur I, la fonction est strictement croissante." """
    },
    
    # --- PHYSIQUE ---
    "FRT_P_MECA_01": {
        "Matiere": "PHYSIQUE", "Chap": "MÉCANIQUE DE NEWTON", "Proba": 0.9,
        "QC": "comment déterminer le vecteur accélération ?",
        "Triggers": ["déterminer les coordonnées du vecteur accélération", "appliquer la deuxième loi de newton", "trouver a(t)", "bilan des forces"],
        "ARI": ["Référentiel", "Bilan Forces", "2e Loi Newton", "Projection"],
        "FRT": """🔔 **Quand utiliser ?** Pour trouver l'accélération à partir des forces.\n\n✅ **Méthode Standard :**\n1. Définir système et référentiel.\n2. Bilan des forces.\n3. Appliquer $\\sum \\vec{F} = m\\vec{a}$.\n4. Projeter sur les axes.\n\n⚠️ **Pièges :** Oublier de préciser le référentiel galiléen.\n\n✍️ **Conclusion Type :** "Par projection, on obtient $a_x = \\dots$ et $a_y = \\dots$." """
    }
}

QI_PATTERNS = {
    "FRT_M_SUITE_01": ["Montrer que (Un) est géométrique.", "Quelle est la nature de la suite (Vn) ?", "Justifier que la suite est géométrique de raison 3."],
    "FRT_M_SUITE_02": ["Déterminer la limite de la suite.", "Calculer la limite quand n tend vers l'infini.", "Étudier la convergence."],
    "FRT_M_FCT_01": ["Étudier les variations de f.", "Dresser le tableau de variations complet.", "Quel est le sens de variation de la fonction ?"],
    "FRT_P_MECA_01": ["En déduire les coordonnées du vecteur accélération.", "Appliquer la 2e loi de Newton pour trouver a(t)."]
}

# ==============================================================================
# 3. MOTEUR
# ==============================================================================

def ingest_factory(urls, volume, matiere, chapitres):
    """Sourcing et Extraction"""
    # Univers filtré
    target_frts = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere and v["Chap"] in chapitres]
    
    # Si le chapitre sélectionné n'est pas dans le Kernel simulé, on ne plante pas
    if not target_frts and volume > 0:
        return pd.DataFrame(), pd.DataFrame()
    
    sources = []
    atoms = []
    progress = st.progress(0)
    
    for i in range(volume):
        progress.progress((i+1)/volume)
        nature = random.choice(["BAC", "DST", "INTERRO"])
        annee = random.choice(range(2020, 2025))
        filename = f"Sujet_{matiere}_{nature}_{annee}_{i}.pdf"
        
        # Extraction Qi
        nb_qi = random.randint(2, 4)
        weights = [UNIVERS_SMAXIA[k]["Proba"] for k in target_frts]
        frts = random.choices(target_frts, weights=weights, k=nb_qi)
        
        qi_data_list = [] # Pour la vérité terrain (Audit)
        
        for frt_id in frts:
            qi_txt = random.choice(QI_PATTERNS[frt_id]) + f" [Réf:{random.randint(10,99)}]"
            atoms.append({
                "FRT_ID": frt_id, "Qi": qi_txt, "File": filename, 
                "Year": annee, "Chap": UNIVERS_SMAXIA[frt_id]["Chap"]
            })
            qi_data_list.append({"Qi": qi_txt, "FRT_ID": frt_id})
            
        sources.append({
            "Fichier": filename, "Nature": nature, "Année": annee,
            "Télécharger": "📥 PDF", # Visuel
            "Blob": f"Contenu simulé de {filename}", # Data pour DL
            "Qi_Data": qi_data_list # Data pour Audit
        })
        
    return pd.DataFrame(sources), pd.DataFrame(atoms)

def compute_qc(df_atoms):
    """Calcul F2 et Clustering"""
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

def extract_external(file):
    """Simulation extraction fichier externe"""
    # Pour la démo, on génère 5 questions aléatoires de l'univers
    # Dans la réalité, on parserait le PDF
    return [{"Qi": f"Question simulée {i}", "FRT_ID": random.choice(list(UNIVERS_SMAXIA.keys()))} for i in range(5)]

# ==============================================================================
# 🖥️ UI
# ==============================================================================

# SIDEBAR
with st.sidebar:
    st.header("Paramètres Académiques")
    st.selectbox("Classe", ["Terminale"], disabled=True)
    sel_matiere = st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    # Liste complète
    sel_chapitres = st.multiselect("Chapitres", LISTE_CHAPITRES[sel_matiere], default=[LISTE_CHAPITRES[sel_matiere][0]])

# TABS
tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine", "✅ Onglet 2 : Audit"])

# --- USINE ---
with tab_usine:
    # 1. ZONE URL
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
        # 2. TABLEAU SUJETS
        st.markdown(f"### 📥 Sujets Traités ({len(st.session_state['df_src'])})")
        
        # DataFrame avec colonnes demandées et Config pour le téléchargement
        # Note : On ne peut pas mettre un bouton cliquable DANS le dataframe natif facilement sans composant tiers
        # On affiche donc une colonne "Action" textuelle et un sélecteur dessous pour le téléchargement réel
        
        df_view = st.session_state['df_src'][["Fichier", "Nature", "Année"]].copy()
        df_view["Téléchargement"] = "📄 Disponible"
        
        st.dataframe(df_view, use_container_width=True, height=300, hide_index=True)
        
        # Zone de téléchargement réel (Contournement limitation technique Streamlit)
        col_dl, _ = st.columns([1, 2])
        with col_dl:
            file_to_dl = st.selectbox("📥 Télécharger un sujet :", st.session_state['df_src']["Fichier"])
            if file_to_dl:
                blob = st.session_state['df_src'][st.session_state['df_src']["Fichier"]==file_to_dl].iloc[0]["Blob"]
                st.download_button("Télécharger le fichier", blob, file_name=file_to_dl)

        st.divider()

        # 3. TABLEAU QC
        st.markdown("### 🧠 Base de Connaissance (QC)")
        if not st.session_state['df_qc'].empty:
            chapters = st.session_state['df_qc']["Chapitre"].unique()
            for chap in chapters:
                subset = st.session_state['df_qc'][st.session_state['df_qc']["Chapitre"] == chap]
                st.markdown(f"#### 📘 Chapitre {chap} : {len(subset)} QC")
                
                for idx, row in subset.iterrows():
                    # HEADER QC STRICT
                    st.markdown(f"""
                    <div class="qc-header-row">
                        <div class="qc-title-group">
                            <span class="qc-id">{row['QC_ID']}</span>
                            <span class="qc-text">{row['Titre']}</span>
                        </div>
                        <span class="qc-stats">Score(q)={row['Score']:.0f} | n_q={row['n_q']} | Ψ={row['Psi']} | N_tot={row['N_tot']} | t_rec={row['t_rec']:.1f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # DETAILS
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
                            # Table HTML
                            html = "<table class='qi-table'>"
                            for item in row['Evidence']:
                                html += f"<tr><td>{item['Fichier']}</td><td>{item['Qi']}</td></tr>"
                            html += "</table>"
                            st.markdown(html, unsafe_allow_html=True)
                    st.write("")
        else:
            st.warning("Aucune QC générée (Vérifiez le périmètre sélectionné).")

# --- AUDIT ---
with tab_audit:
    st.subheader("Validation Booléenne")
    
    if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
        
        # TEST 1
        st.markdown("#### ✅ 1. Test Interne (Sujet Traité)")
        t1_file = st.selectbox("Sujet Traité", st.session_state['df_src']["Fichier"])
        
        if st.button("LANCER TEST INTERNE"):
            data = st.session_state['df_src'][st.session_state['df_src']["Fichier"]==t1_file].iloc[0]["Qi_Data"]
            known_ids = st.session_state['df_qc']["FRT_ID"].unique()
            
            ok_count = 0
            rows = []
            for item in data:
                is_ok = item["FRT_ID"] in known_ids
                if is_ok: ok_count += 1
                status = "✅ MATCH" if is_ok else "❌ ERREUR"
                
                qc_n = "---"
                if is_ok:
                    info = st.session_state['df_qc'][st.session_state['df_qc']["FRT_ID"]==item["FRT_ID"]].iloc[0]
                    qc_n = f"{info['QC_ID']} {info['Titre']}"
                
                rows.append({"Qi": item["Qi"], "QC": qc_n, "Statut": status})
            
            taux = (ok_count / len(data)) * 100
            st.markdown(f"### Taux de Couverture : {taux:.0f}%")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.divider()

        # TEST 2
        st.markdown("#### 🌍 2. Test Externe (Nouveau Sujet)")
        up_file = st.file_uploader("Charger un PDF externe", type="pdf")
        
        if up_file:
            # Extraction Simulée
            # On prend des FRT au hasard dans l'univers connu pour simuler le fichier
            possible_frts = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == sel_matiere]
            
            if not possible_frts:
                st.error("Impossible de simuler : Univers vide pour cette matière.")
            else:
                extracted_frts = random.sample(possible_frts, k=min(5, len(possible_frts)))
                
                rows_ext = []
                ok_ext = 0
                known_ids = st.session_state['df_qc']["FRT_ID"].unique()
                
                for frt in extracted_frts:
                    qi_txt = random.choice(QI_PATTERNS.get(frt, ["Question..."])) + " (Externe)"
                    is_known = frt in known_ids
                    
                    if is_known: ok_ext += 1
                    status = "✅ MATCH" if is_known else "❌ GAP"
                    
                    qc_n = "---"
                    frt_n = frt
                    if is_known:
                        info = st.session_state['df_qc'][st.session_state['df_qc']["FRT_ID"]==frt].iloc[0]
                        qc_n = f"{info['QC_ID']} {info['Titre']}"
                    
                    rows_ext.append({"Qi (Enoncé)": qi_txt, "QC Correspondante": qc_n, "FRT": frt_n, "Statut": status})
                
                taux_ext = (ok_ext / len(extracted_frts)) * 100
                st.markdown(f"### Taux de Couverture : {taux_ext:.1f}%")
                
                def color_audit(row):
                    return ['background-color: #dcfce7' if row['Statut'] == "✅ MATCH" else 'background-color: #fee2e2'] * len(row)

                st.dataframe(pd.DataFrame(rows_ext).style.apply(color_audit, axis=1), use_container_width=True)
                
    else:
        st.info("Veuillez lancer l'usine d'abord.")
