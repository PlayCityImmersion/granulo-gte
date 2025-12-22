import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V18")
st.title("🛡️ SMAXIA - Console V18 (Stable & Complète)")

# ==============================================================================
# 🎨 STYLES CSS (FIGÉS)
# ==============================================================================
st.markdown("""
<style>
    /* EN-TÊTE QC */
    .qc-header {
        background-color: #f8f9fa; border-left: 5px solid #2563eb;
        padding: 12px; margin-bottom: 5px; border-radius: 4px;
        font-family: 'Source Sans Pro', sans-serif; display: flex; justify-content: space-between; align-items: center;
    }
    .qc-info { display: flex; align-items: center; }
    .qc-id { color: #d97706; font-weight: 800; font-size: 1.1em; margin-right: 15px; min-width: 80px; }
    .qc-text { color: #1f2937; font-weight: 600; font-size: 1.1em; }
    .qc-stats { 
        font-family: 'Courier New', monospace; font-size: 0.9em; font-weight: 700; color: #4b5563;
        background-color: #e5e7eb; padding: 4px 10px; border-radius: 4px; white-space: nowrap;
    }

    /* DETAILS */
    .trigger-box { background-color: #fff1f2; border: 1px solid #fecdd3; color: #be123c; padding: 8px; border-radius: 6px; font-weight: 600; font-size: 0.9em; }
    .ari-box { background-color: #f3f4f6; border: 1px dashed #9ca3af; color: #374151; padding: 8px; border-radius: 6px; font-family: monospace; font-size: 0.85em; }
    .frt-box { background-color: #ecfdf5; border: 1px solid #6ee7b7; padding: 10px; border-radius: 6px; font-family: sans-serif; line-height: 1.5; color: #065f46; white-space: pre-wrap; }
    
    /* TABLEAUX HTML (Pas de Scroll) */
    .qi-table { width: 100%; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; }
    .qi-table th { background: #f9fafb; text-align: left; padding: 6px; border-bottom: 2px solid #e5e7eb; color: #6b7280; }
    .qi-table td { padding: 6px; border-bottom: 1px solid #f3f4f6; vertical-align: top; color: #1f2937; }
    
    /* METRIQUES AUDIT */
    .audit-box { padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 15px; }
    .audit-success { background-color: #dcfce7; border: 1px solid #86efac; color: #166534; }
    .audit-warning { background-color: #fef9c3; border: 1px solid #fde047; color: #854d0e; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. LISTE DES CHAPITRES (COMPLÈTE)
# ==============================================================================

LISTE_CHAPITRES = {
    "MATHS": [
        "SUITES NUMÉRIQUES", "FONCTIONS & DÉRIVATION", "LIMITES DE FONCTIONS", 
        "CONTINUITÉ & CONVEXITÉ", "FONCTION LOGARITHME", "PRIMITIVES & ÉQUATIONS DIFF", 
        "CALCUL INTÉGRAL", "COMBINATOIRE & DÉNOMBREMENT", "PROBABILITÉS DISCRÈTES", 
        "LOI BINOMIALE", "SOMMES DE VARIABLES ALÉATOIRES", "CONCENTRATION & LGN", 
        "GÉOMÉTRIE DANS L'ESPACE", "VECTEURS, DROITES & PLANS", "ORTHOGONALITÉ & DISTANCES"
    ],
    "PHYSIQUE": [
        "MOUVEMENT & INTERACTIONS", "MÉCANIQUE DE NEWTON", "MOUVEMENT DANS UN CHAMP", 
        "THERMODYNAMIQUE", "ONDES MÉCANIQUES", "LUMIÈRE & ONDES", "LUNETTE ASTRONOMIQUE", 
        "CIRCUITS ÉLECTRIQUES (RC)", "TRANSFORMATIONS CHIMIQUES", "ACIDE-BASE", "DOSAGES"
    ]
}

# ==============================================================================
# 2. KERNEL SIMULÉ (Vérité Terrain pour le Moteur)
# ==============================================================================

UNIVERS_SMAXIA = {
    # MATHS - SUITES
    "FRT_M_SUITE_01": {"Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", "QC": "Comment démontrer qu'une suite est géométrique ?", "Trigger": "Montrer que la suite est géométrique / Nature de la suite", "ARI": ["u(n+1)", "Ratio", "Constante"], "FRT": "1. Exprimer u(n+1).\n2. Calculer le rapport u(n+1)/u(n).\n3. Identifier la raison q."},
    "FRT_M_SUITE_02": {"Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", "QC": "Comment calculer la limite (Forme Indéterminée) ?", "Trigger": "Déterminer la limite / Lever l'indétermination", "ARI": ["FI", "Factorisation", "Limites usuelles"], "FRT": "1. Identifier la FI.\n2. Factoriser par le terme dominant.\n3. Conclure."},
    
    # MATHS - FONCTIONS
    "FRT_M_FCT_01": {"Matiere": "MATHS", "Chap": "FONCTIONS & DÉRIVATION", "QC": "Comment étudier les variations ?", "Trigger": "Sens de variation / Tableau de variations", "ARI": ["Dérivée", "Signe", "Tableau"], "FRT": "1. Calculer f'(x).\n2. Étudier le signe.\n3. Conclure sur les variations."},
    "FRT_M_FCT_02": {"Matiere": "MATHS", "Chap": "FONCTIONS & DÉRIVATION", "QC": "Comment appliquer le TVI (Unique solution) ?", "Trigger": "Montrer que f(x)=k a une solution unique", "ARI": ["Continuité", "Monotonie", "Bornes", "Corollaire"], "FRT": "1. Vérifier continuité et stricte monotonie.\n2. Calculer les bornes.\n3. Appliquer le corollaire du TVI."},
    
    # PHYSIQUE - MECA
    "FRT_P_MECA_01": {"Matiere": "PHYSIQUE", "Chap": "MÉCANIQUE DE NEWTON", "QC": "Comment trouver le vecteur accélération ?", "Trigger": "Déterminer les coordonnées de a(t) / 2e Loi Newton", "ARI": ["Bilan", "2e Loi", "Projection"], "FRT": "1. Bilan des forces.\n2. Somme F = ma.\n3. Projection sur les axes."},
}

QI_PATTERNS = {
    "FRT_M_SUITE_01": ["Montrer que (Un) est géométrique.", "Justifier que la suite est géométrique."],
    "FRT_M_SUITE_02": ["Déterminer la limite de Un.", "Lever l'indétermination de la limite."],
    "FRT_M_FCT_01": ["Dresser le tableau de variations.", "Étudier le sens de variation de f."],
    "FRT_M_FCT_02": ["Montrer que f(x)=0 a une unique solution alpha.", "Prouver l'existence d'une solution unique."],
    "FRT_P_MECA_01": ["En déduire l'accélération a(t).", "Appliquer la 2e loi de Newton."]
}

# ==============================================================================
# 3. LOGIQUE MOTEUR
# ==============================================================================

def ingest_factory(urls, volume, matiere, chapitres):
    sources = []
    atoms = []
    
    # Filtrer l'univers connu (Simulation)
    target_frts = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere and v["Chap"] in chapitres]
    
    # Si le chapitre sélectionné n'est pas dans le Kernel simulé, on ne plante pas, on renvoie vide
    if not target_frts and volume > 0:
        # Pour ne pas crasher, on renvoie des DF vides
        pass 
    
    progress = st.progress(0)
    
    for i in range(volume):
        progress.progress((i+1)/volume)
        
        # Si pas de FRT dispo pour ce chapitre, on crée un sujet "vide" ou hors scope
        has_content = len(target_frts) > 0
        
        filename = f"Sujet_{matiere}_{i}.pdf"
        nature = random.choice(["BAC", "DST", "CONCOURS"])
        annee = random.choice(range(2020, 2025))
        
        qi_data_list = []
        
        if has_content:
            nb_qi = random.randint(2, 4)
            frts = random.choices(target_frts, k=nb_qi)
            for frt_id in frts:
                qi_txt = random.choice(QI_PATTERNS[frt_id]) + f" [Ref:{random.randint(100,999)}]"
                atoms.append({
                    "FRT_ID": frt_id, "Qi": qi_txt, "File": filename, "Year": annee,
                    "Chap": UNIVERS_SMAXIA[frt_id]["Chap"]
                })
                qi_data_list.append({"Qi": qi_txt, "FRT_ID": frt_id})
        
        # Construction Ligne Sujet (IMPORTANT : Clés exactes pour le DataFrame)
        sources.append({
            "Fichier": filename,
            "Nature": nature,
            "Année": annee,
            "Télécharger": "📥 PDF", # Placeholder pour le bouton
            "Qi_Data": qi_data_list
        })
        
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

# ==============================================================================
# 🖥️ INTERFACE
# ==============================================================================

# --- SIDEBAR ---
with st.sidebar:
    st.header("Paramètres Académiques")
    st.selectbox("Classe", ["Terminale"], disabled=True)
    sel_matiere = st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    # Liste complète des chapitres
    sel_chapitres = st.multiselect("Chapitres", LISTE_CHAPITRES[sel_matiere], default=[LISTE_CHAPITRES[sel_matiere][0]])

# --- TABS ---
tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine", "✅ Onglet 2 : Audit"])

# --- ONGLET 1 : USINE ---
with tab_usine:
    # 1. SOURCING
    st.subheader("1. Configuration Sourcing")
    c1, c2 = st.columns([3, 1])
    with c1: urls = st.text_area("URLs Sources", "https://apmep.fr", height=68)
    with c2: 
        vol = st.number_input("Volume", 5, 500, 20, step=5)
        btn_run = st.button("LANCER L'USINE 🚀", type="primary")

    if btn_run:
        df_src, df_atoms = ingest_factory(urls.split('\n'), vol, sel_matiere, sel_chapitres)
        df_qc = compute_qc(df_atoms)
        st.session_state['df_src'] = df_src
        st.session_state['df_qc'] = df_qc
        st.success(f"Ingestion terminée : {len(df_src)} sujets traités.")

    st.divider()

    # 2. SUJETS & QC (VERTICAL)
    if 'df_src' in st.session_state and not st.session_state['df_src'].empty:
        # TABLEAU SUJETS
        st.markdown(f"### 📥 Sujets Traités ({len(st.session_state['df_src'])})")
        # Affichage propre avec configuration de colonnes
        st.dataframe(
            st.session_state['df_src'][["Fichier", "Nature", "Année", "Téléchargement"]],
            use_container_width=True,
            height=300,
            hide_index=True
        )
        
        st.divider()
        
        # LISTE QC
        st.markdown("### 🧠 Base de Connaissance (QC)")
        if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
            chapters = st.session_state['df_qc']["Chapitre"].unique()
            for chap in chapters:
                subset = st.session_state['df_qc'][st.session_state['df_qc']["Chapitre"] == chap]
                st.markdown(f"#### 📘 {chap} : {len(subset)} QC")
                
                for idx, row in subset.iterrows():
                    # Header QC
                    st.markdown(f"""
                    <div class="qc-header">
                        <div class="qc-info">
                            <span class="qc-id">{row['QC_ID']}</span>
                            <span class="qc-text">{row['Titre']}</span>
                        </div>
                        <span class="qc-stats">Score(q)={row['Score']:.0f} | n_q={row['n_q']} | Ψ={row['Psi']} | t_rec={row['t_rec']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Détails
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        with st.expander("🔥 Déclencheurs"):
                            st.markdown(f"<div class='trigger-box'>{row['Trigger']}</div>", unsafe_allow_html=True)
                    with c2:
                        with st.expander("⚙️ ARI"):
                            st.markdown(f"<div class='ari-box'>{str(row['ARI'])}</div>", unsafe_allow_html=True)
                    with c3:
                        with st.expander("🧾 FRT"):
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
            st.warning("Aucune QC générée pour ce périmètre (Vérifiez que le simulateur contient des FRT pour ces chapitres).")
    elif 'df_src' in st.session_state and st.session_state['df_src'].empty:
        st.warning("Aucun sujet généré (Périmètre vide ou erreur).")

# --- ONGLET 2 : AUDIT ---
with tab_audit:
    st.subheader("Validation Booléenne")
    
    if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
        
        # TEST 2 : UPLOAD EXTERNE
        st.markdown("#### 🌍 2. Test Externe (Mapping Nouveau Sujet)")
        uploaded_file = st.file_uploader("Télécharger un sujet (PDF) hors scope", type="pdf")
        
        if uploaded_file is not None:
            # SIMULATION DE L'EXTRACTION SUR FICHIER EXTERNE
            # On prend des FRT au hasard dans l'univers SMAXIA qui correspondent à la matière
            universe = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == sel_matiere]
            
            if universe:
                frts_detected = random.sample(universe, k=min(5, len(universe)))
                
                rows = []
                ok_count = 0
                known_ids = st.session_state['df_qc']["FRT_ID"].unique()
                
                for frt in frts_detected:
                    # On génère une Qi simulée
                    qi_text = random.choice(QI_PATTERNS[frt]) + " (Extrait du PDF)"
                    
                    is_covered = frt in known_ids
                    if is_covered: ok_count += 1
                    
                    qc_nom = "Pas de QC"
                    frt_nom = frt
                    status = "❌ GAP"
                    
                    if is_covered:
                        qc_data = st.session_state['df_qc'][st.session_state['df_qc']["FRT_ID"]==frt].iloc[0]
                        qc_nom = f"{qc_data['QC_ID']} {qc_data['Titre']}"
                        status = "✅ MATCH"
                    
                    rows.append({
                        "Qi (Enoncé)": qi_text,
                        "QC Correspondante": qc_nom,
                        "FRT Associée": frt_nom,
                        "Statut": status
                    })
                
                taux = (ok_count / len(frts_detected)) * 100
                
                # Affichage Titre Taux
                color_class = "audit-success" if taux == 100 else "audit-warning"
                st.markdown(f"<div class='audit-box {color_class}'><h3>Taux de Couverture = {taux:.1f}%</h3></div>", unsafe_allow_html=True)
                
                def color_row(row):
                    return ['background-color: #dcfce7' if row['Statut'] == "✅ MATCH" else 'background-color: #fee2e2'] * len(row)

                st.dataframe(pd.DataFrame(rows).style.apply(color_row, axis=1), use_container_width=True)
            else:
                st.error("Impossible d'analyser : Univers vide pour cette matière.")
                
    else:
        st.info("Veuillez lancer l'usine d'abord.")
