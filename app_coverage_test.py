import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Golden Master V14")
st.title("🛡️ SMAXIA - Environnement de Test V14 (Golden Master)")

# ==============================================================================
# 🎨 STYLES CSS (FIGÉS)
# ==============================================================================
st.markdown("""
<style>
    /* Carte QC Principale */
    .qc-card {
        border-left: 5px solid #2563eb; background-color: #f8f9fa;
        padding: 10px; margin-bottom: 5px; border-radius: 4px;
        display: flex; flex-wrap: wrap; align-items: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .qc-id { font-weight: bold; color: #d97706; font-size: 1.1em; margin-right: 10px; min-width: 80px;}
    .qc-title { flex-grow: 1; font-weight: 600; color: #1f2937; font-size: 1.1em; font-family: sans-serif; }
    .qc-stats { 
        font-family: 'Courier New'; font-size: 0.85em; font-weight: bold; color: #374151;
        background: #e5e7eb; padding: 4px 8px; border-radius: 4px; margin-left: 10px; white-space: nowrap;
    }
    
    /* Contenu Détails */
    .frt-box { background: white; border: 1px solid #d1d5db; padding: 15px; border-radius: 6px; font-family: sans-serif; white-space: pre-wrap; line-height: 1.5; }
    .trigger-badge { background: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: 700; border: 1px solid #fca5a5; display: inline-block; margin: 2px; }
    .ari-step { margin-left: 15px; color: #4b5563; font-family: monospace; }
    
    /* Tableaux */
    .full-table { width: 100%; border-collapse: collapse; font-size: 0.9em; font-family: sans-serif; }
    .full-table th { background: #f3f4f6; padding: 8px; text-align: left; border-bottom: 2px solid #e5e7eb; color: #374151; }
    .full-table td { padding: 8px; border-bottom: 1px solid #e5e7eb; vertical-align: top; color: #1f2937; }
    .full-table tr:hover { background: #f9fafb; }
    
    /* Métriques Audit */
    .audit-metric { font-size: 2em; font-weight: 800; color: #059669; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🧠 KERNEL UNIVERSEL (MATHS + PHYSIQUE)
# ==============================================================================

# Structure : ID -> {QC, Chapitre, Proba, ARI, FRT, Triggers}
UNIVERS_SMAXIA = {
    # --- MATHS ---
    "FRT_M_SUITE_01": {
        "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", "Proba": 0.9,
        "QC": "Comment démontrer qu'une suite est géométrique ?",
        "Triggers": ["montrer que la suite est géométrique", "nature de la suite", "raison q"],
        "ARI": ["Calcul u(n+1)", "Ratio u(n+1)/u(n)", "Simplification", "Identification constante"],
        "FRT": "**Méthode :**\n1. Exprimer $u_{n+1}$ en fonction de $n$.\n2. Calculer le rapport $\\frac{u_{n+1}}{u_n}$.\n3. Simplifier pour trouver une constante $q$."
    },
    "FRT_M_SUITE_02": {
        "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", "Proba": 0.8,
        "QC": "Comment lever une indétermination (limite) ?",
        "Triggers": ["déterminer la limite", "forme indéterminée", "convergence"],
        "ARI": ["Identifier FI", "Factoriser terme dominant", "Limites usuelles"],
        "FRT": "**Méthode :**\n1. Identifier la F.I.\n2. Factoriser par le terme de plus haut degré.\n3. Conclure par produit."
    },
    "FRT_M_FCT_01": {
        "Matiere": "MATHS", "Chap": "FONCTIONS", "Proba": 0.9,
        "QC": "Comment étudier les variations d'une fonction ?",
        "Triggers": ["dresser le tableau de variations", "sens de variation"],
        "ARI": ["Dérivée f'", "Signe f'", "Tableau"],
        "FRT": "**Méthode :**\n1. Calculer $f'(x)$.\n2. Étudier le signe de la dérivée.\n3. En déduire les variations (Croissante si +)."
    },
    "FRT_M_FCT_02": {
        "Matiere": "MATHS", "Chap": "FONCTIONS", "Proba": 0.7,
        "QC": "Comment appliquer le TVI (Solution unique) ?",
        "Triggers": ["équation f(x)=k", "solution unique alpha"],
        "ARI": ["Continuité", "Stricte Monotonie", "Bornes", "Corollaire"],
        "FRT": "**Méthode :**\n1. Vérifier continuité et stricte monotonie.\n2. Calculer les images aux bornes.\n3. Invoquer le corollaire du TVI."
    },

    # --- PHYSIQUE ---
    "FRT_P_MECA_01": {
        "Matiere": "PHYSIQUE", "Chap": "MÉCANIQUE DE NEWTON", "Proba": 0.9,
        "QC": "Comment déterminer l'accélération (2e Loi) ?",
        "Triggers": ["déterminer le vecteur accélération", "appliquer la deuxième loi de Newton", "bilan des forces"],
        "ARI": ["Système/Référentiel", "Bilan Forces", "2e Loi Newton", "Projection"],
        "FRT": "**Méthode :**\n1. Définir le système et le référentiel galiléen.\n2. Faire le bilan des forces.\n3. Appliquer $\\sum \\vec{F} = m\\vec{a}$.\n4. Projeter sur les axes."
    },
    "FRT_P_ONDE_01": {
        "Matiere": "PHYSIQUE", "Chap": "ONDES MÉCANIQUES", "Proba": 0.8,
        "QC": "Comment calculer une longueur d'onde ?",
        "Triggers": ["calculer la longueur d'onde", "célérité et fréquence", "période spatiale"],
        "ARI": ["Identifier v et f", "Formule lambda", "Calcul"],
        "FRT": "**Méthode :**\n1. Repérer la célérité $v$ et la fréquence $f$ (ou période $T$).\n2. Appliquer $\\lambda = v/f$ ou $\\lambda = v \\times T$."
    }
}

# Générateur de phrases élèves (Polymorphisme)
QI_PATTERNS = {
    "FRT_M_SUITE_01": ["Montrer que (Un) est géométrique.", "Prouver que la suite est géométrique de raison 2."],
    "FRT_M_SUITE_02": ["Calculer la limite de Un.", "La suite converge-t-elle ?", "Lever l'indétermination."],
    "FRT_M_FCT_01": ["Étudier les variations de f.", "Dresser le tableau de variations complet."],
    "FRT_M_FCT_02": ["Montrer que f(x)=0 a une solution unique.", "Justifier l'existence d'un unique alpha."],
    "FRT_P_MECA_01": ["En déduire les coordonnées du vecteur accélération.", "Appliquer la 2e loi de Newton pour trouver a."],
    "FRT_P_ONDE_01": ["Déterminer la longueur d'onde lambda.", "Quelle est la période spatiale de l'onde ?"]
}

# ==============================================================================
# ⚙️ MOTEUR & FONCTIONS
# ==============================================================================

def generate_qi(frt_id):
    """Génère une phrase élève réaliste"""
    base = random.choice(QI_PATTERNS.get(frt_id, ["Question type..."]))
    ctx = random.choice(["", " sur I", " dans le repère", " pour t > 0"])
    return base + ctx

def ingest_factory(urls, n_volume, subject_filter, chapter_filter):
    """Simule l'usine de traitement"""
    # 1. Filtrer l'univers des possibles
    universe = [k for k, v in UNIVERS_SMAXIA.items() 
                if v["Matiere"] == subject_filter and v["Chap"] in chapter_filter]
    
    if not universe: return pd.DataFrame(), pd.DataFrame()
    
    sources = []
    atoms = []
    
    # 2. Simulation Sourcing
    progress = st.progress(0)
    total = len(urls) * n_volume
    counter = 0
    
    for i, url in enumerate(urls):
        if not url.strip(): continue
        for j in range(n_volume):
            counter += 1
            progress.progress(min(counter/total, 1.0))
            
            # Création Sujet
            f_nature = random.choice(["BAC", "DST", "INTERRO"])
            f_year = random.choice(range(2020, 2025))
            f_name = f"Sujet_{subject_filter}_{f_nature}_{f_year}_{j}.pdf"
            
            # Extraction Qi (Tirage pondéré selon Proba)
            nb_qi = random.randint(2, 4)
            weights = [UNIVERS_SMAXIA[k]["Proba"] for k in universe]
            drawn_frts = random.choices(universe, weights=weights, k=nb_qi)
            
            qi_data = []
            for frt_id in drawn_frts:
                qi_txt = generate_qi(frt_id)
                # Base de données atomique (Moteur)
                atoms.append({
                    "FRT_ID": frt_id, "Qi": qi_txt, "File": f_name, 
                    "Year": f_year, "Chap": UNIVERS_SMAXIA[frt_id]["Chap"]
                })
                # Données pour Audit Interne (Vérité Terrain)
                qi_data.append({"Qi": qi_txt, "FRT_ID": frt_id})
            
            sources.append({
                "Fichier": f_name, 
                "Nature": f_nature, 
                "Année": f_year, 
                "Télécharger": "📥 PDF", # Visuel
                "Blob_Content": f"Contenu simulé de {f_name}...", # Pour DL réel
                "Qi_Data": qi_data # Pour l'audit
            })
            
    progress.empty()
    return pd.DataFrame(sources), pd.DataFrame(atoms)

def compute_qc(df_atoms):
    """Clustering dynamique et Calcul F2"""
    if df_atoms.empty: return pd.DataFrame()
    
    # Regroupement par FRT (Structure)
    grouped = df_atoms.groupby("FRT_ID").agg({
        "Qi": "count", "Year": "max", "File": list, "Qi": list, "Chap": "first"
    }).rename(columns={"Qi": "Qi_List", "File": "File_List"}).reset_index()
    
    qcs = []
    N_tot = len(df_atoms)
    
    for idx, row in grouped.iterrows():
        frt_id = row["FRT_ID"]
        meta = UNIVERS_SMAXIA[frt_id]
        n_q = len(row["Qi_List"]) # Correction count
        
        # Calcul Variables
        tau = max((datetime.now().year - row["Year"]), 0.5)
        psi = 0.85 # Simulé (densité cognitive)
        score = (n_q / N_tot) * (1 + 5.0/tau) * psi * 100
        
        qcs.append({
            "Chapitre": row["Chap"],
            "QC_ID": f"QC-{idx+1:02d}", # Format demandé
            "FRT_ID": frt_id,
            "Titre": meta["QC"],
            "Score": score, "n_q": n_q, "Psi": psi, "N_tot": N_tot, "t_rec": tau,
            "Triggers": meta["Triggers"], "ARI": meta["ARI"], "FRT": meta["FRT"],
            "Evidence": [{"Fichier": f, "Qi": q} for f, q in zip(row["File_List"], row["Qi_List"])]
        })
        
    return pd.DataFrame(qcs).sort_values(by="Score", ascending=False)

# ==============================================================================
# 🖥️ SIDEBAR (NAVIGATION)
# ==============================================================================

with st.sidebar:
    st.header("1. Paramètres")
    st.selectbox("Classe", ["Terminale"], disabled=True)
    
    # Sélecteur Matière
    matiere_sel = st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    
    # Calcul dynamique des chapitres dispos dans le Kernel pour cette matière
    available_chaps = sorted(list(set(
        v["Chap"] for v in UNIVERS_SMAXIA.values() if v["Matiere"] == matiere_sel
    )))
    
    chapitres_sel = st.multiselect("Chapitres", available_chaps, default=available_chaps)

# ==============================================================================
# 📑 ONGLETS PRINCIPAUX
# ==============================================================================

tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine (Sourcing & QC)", "✅ Onglet 2 : Audit (Validation)"])

# ------------------------------------------------------------------------------
# ONGLET 1 : USINE
# ------------------------------------------------------------------------------
with tab_usine:
    # ZONE URL
    st.subheader("Zone URL & Volume")
    c1, c2 = st.columns([3, 1])
    with c1: 
        urls = st.text_area("Zone d'input URLs", "https://apmep.fr\nhttps://labolycee.org", height=70)
    with c2: 
        vol = st.number_input("Volume (Sujets)", 5, 200, 15, step=5)
        run = st.button("LANCER LE SOURCING 🚀", type="primary")

    if run:
        with st.spinner("Traitement Usine en cours..."):
            df_s, df_a = ingest_factory(urls.split('\n'), vol, matiere_sel, chapitres_sel)
            df_qc = compute_qc(df_a)
            st.session_state['df_s'] = df_s
            st.session_state['df_q'] = df_qc
            st.success(f"Terminé : {len(df_s)} sujets traités.")

    st.divider()

    # ZONES RESULTATS
    if 'df_q' in st.session_state:
        col_L, col_R = st.columns([1, 1.8])
        
        # --- TABLEAU DES SUJETS (GAUCHE) ---
        with col_L:
            st.markdown(f"#### 📄 Sujets Sourcés ({len(st.session_state['df_s'])})")
            
            # Tableau 3 Colonnes (Fichier, Nature, Année/DL)
            # On prépare un DF pour l'affichage propre
            df_view_s = st.session_state['df_s'].copy()
            df_view_s["Infos / DL"] = df_view_s["Année"].astype(str) + " - " + df_view_s["Télécharger"]
            
            st.dataframe(
                df_view_s[["Fichier", "Nature", "Infos / DL"]], 
                use_container_width=True, 
                height=600,
                hide_index=True
            )
            
            # Bouton de DL Réel (sous le tableau pour stabilité)
            sel_file = st.selectbox("Sélectionner pour télécharger :", st.session_state['df_s']["Fichier"])
            file_blob = st.session_state['df_s'][st.session_state['df_s']["Fichier"]==sel_file].iloc[0]["Blob_Content"]
            st.download_button(f"📥 Télécharger {sel_file}", file_blob, file_name=sel_file)

        # --- TABLEAU DES QC (DROITE) ---
        with col_R:
            if not st.session_state['df_q'].empty:
                # Groupement par Chapitre
                all_chaps = st.session_state['df_q']["Chapitre"].unique()
                
                for chap in all_chaps:
                    qc_subset = st.session_state['df_q'][st.session_state['df_q']["Chapitre"] == chap]
                    st.markdown(f"#### 📘 Chapitre {chap} : {len(qc_subset)} QC")
                    
                    for _, row in qc_subset.iterrows():
                        # CARTE QC
                        with st.container():
                            # LIGNE 1 : ID, Titre, Stats
                            # Format : QC_ID « Nom » Score...
                            header_html = f"""
                            <div class="qc-card">
                                <span class="qc-id">{row['QC_ID']}</span>
                                <span class="qc-title">« {row['Titre']} »</span>
                                <span class="qc-stats">
                                    Score(q)={row['Score']:.0f} | n_q={row['n_q']} | Ψ={row['Psi']} | N_tot={row['N_tot']} | t_rec={row['t_rec']:.1f}
                                </span>
                            </div>
                            """
                            st.markdown(header_html, unsafe_allow_html=True)
                            
                            # LIGNE 2+ : DÉTAILS (Déclencheurs, ARI, FRT, Qi)
                            c1, c2, c3, c4 = st.columns(4)
                            
                            with c1:
                                with st.expander("⚡ Déclencheurs"):
                                    for t in row['Triggers']: 
                                        st.markdown(f"<span class='trigger-badge'>{t}</span>", unsafe_allow_html=True)
                            
                            with c2:
                                with st.expander(f"⚙️ ARI"):
                                    for s in row['ARI']: st.markdown(f"<div class='ari-step'>- {s}</div>", unsafe_allow_html=True)
                                    
                            with c3:
                                with st.expander(f"📝 FRT"):
                                    st.markdown(f"<div class='frt-box'>{row['FRT']}</div>", unsafe_allow_html=True)
                                    
                            with c4:
                                with st.expander(f"📄 Qi ({row['n_q']})"):
                                    # Table HTML pour Qi
                                    html_qi = "<table class='full-table'><thead><tr><th>Fichier</th><th>Qi</th></tr></thead><tbody>"
                                    for item in row['Evidence']:
                                        html_qi += f"<tr><td>{item['Fichier']}</td><td>{item['Qi']}</td></tr>"
                                    html_qi += "</tbody></table>"
                                    st.markdown(html_qi, unsafe_allow_html=True)
                            
                            st.write("") # Spacer
            else:
                st.warning("Aucune QC trouvée. Augmentez le volume.")

# ------------------------------------------------------------------------------
# ONGLET 2 : AUDIT
# ------------------------------------------------------------------------------
with tab_audit:
    st.subheader("Validation Booléenne")
    
    if 'df_q' in st.session_state:
        
        # --- TEST 1 : INTERNE ---
        st.markdown("#### 1. Test 1 : Couverture Interne (Sujet Traité)")
        t1_file = st.selectbox("Choisir un sujet traité", st.session_state['df_s']["Fichier"])
        
        if st.button("LANCER TEST 1"):
            # Données vérité
            data = st.session_state['df_s'][st.session_state['df_s']["Fichier"]==t1_file].iloc[0]["Qi_Data"]
            
            # Vérif existence dans le moteur actuel
            known_ids = st.session_state['df_q']["FRT_ID"].unique()
            
            match_count = 0
            res_rows = []
            
            for item in data:
                is_ok = item["FRT_ID"] in known_ids
                if is_ok: match_count += 1
                status = "✅ MATCH" if is_ok else "❌ ERREUR"
                res_rows.append({"Qi": item["Qi"], "Statut": status})
                
            taux = (match_count / len(data)) * 100
            st.markdown(f"<div class='audit-metric'>Couverture : {taux:.0f}%</div>", unsafe_allow_html=True)
            st.table(pd.DataFrame(res_rows))

        st.divider()

        # --- TEST 2 : EXTERNE ---
        st.markdown("#### 2. Test 2 : Couverture Externe (Sujet Inconnu)")
        
        if st.button("LANCER TEST 2 (Génération Sujet)"):
            # Génération d'un sujet "Hors Usine" (On pioche dans l'univers complet du périmètre)
            universe = [k for k, v in UNIVERS_SMAXIA.items() 
                        if v["Matiere"] == matiere_sel and v["Chap"] in chapitres_sel]
            
            if not universe:
                st.error("Périmètre vide.")
            else:
                # On pioche 5 questions au hasard
                test_frts = random.sample(universe, k=min(5, len(universe)))
                
                known_ids = st.session_state['df_q']["FRT_ID"].unique()
                match_count = 0
                ext_rows = []
                
                for frt in test_frts:
                    qi_txt = generate_qi(frt) # Nouvelle phrase
                    is_ok = frt in known_ids
                    if is_ok: match_count += 1
                    
                    # Récup info QC si trouvée
                    if is_ok:
                        qc_info = st.session_state['df_q'][st.session_state['df_q']["FRT_ID"]==frt].iloc[0]
                        qc_disp = f"{qc_info['QC_ID']} {qc_info['Titre']}"
                        status = "✅ MATCH"
                    else:
                        qc_disp = "---"
                        status = "❌ MANQUANT"
                        
                    ext_rows.append({"Qi (Externe)": qi_txt, "QC Moteur": qc_disp, "Statut": status})
                
                taux_ext = (match_count / len(test_frts)) * 100
                st.markdown(f"<div class='audit-metric'>Couverture : {taux_ext:.0f}%</div>", unsafe_allow_html=True)
                
                # Table colorée pour le mapping
                def color_row(row):
                    return ['background-color: #dcfce7' if row['Statut'] == "✅ MATCH" else 'background-color: #fee2e2'] * len(row)
                
                st.dataframe(pd.DataFrame(ext_rows).style.apply(color_row, axis=1), use_container_width=True)

    else:
        st.info("Veuillez lancer l'usine en Onglet 1 d'abord.")
