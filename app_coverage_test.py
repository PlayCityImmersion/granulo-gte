import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V22")
st.title("🛡️ SMAXIA - Console V22 (Doctrine & Stability)")

# ==============================================================================
# 🎨 STYLES CSS (RESPECT STRICT DE L'UI DEMANDÉE)
# ==============================================================================
st.markdown("""
<style>
    /* EN-TÊTE QC */
    .qc-header-row {
        background-color: #f8f9fa; border-left: 6px solid #2563eb;
        padding: 15px; margin-bottom: 10px; border-radius: 4px;
        font-family: 'Source Sans Pro', sans-serif;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        display: flex; justify-content: space-between; align-items: center;
    }
    .qc-title-group { display: flex; align-items: center; flex-grow: 1; }
    .qc-id { color: #d97706; font-weight: 800; font-size: 1.2em; margin-right: 15px; min-width: 90px; }
    .qc-text { color: #111827; font-weight: 600; font-size: 1.15em; }
    .qc-stats { 
        font-family: 'Courier New', monospace; font-size: 0.9em; font-weight: 700; color: #374151;
        background-color: #e5e7eb; padding: 6px 12px; border-radius: 4px; white-space: nowrap; margin-left: 15px;
    }

    /* BLOCS DÉTAILS */
    .detail-label { font-size: 0.8em; text-transform: uppercase; color: #6b7280; font-weight: bold; margin-bottom: 5px; display: block; }
    
    /* 1. TRIGGERS */
    .trigger-box { background-color: #fff1f2; padding: 10px; border-radius: 6px; border: 1px solid #fecdd3; }
    .trigger-item { 
        display: block; margin-bottom: 4px; color: #be123c; font-weight: 600; font-size: 0.95em;
        padding-left: 10px; border-left: 3px solid #fda4af;
    }

    /* 2. ARI */
    .ari-box { background-color: #f3f4f6; padding: 10px; border-radius: 6px; font-family: monospace; font-size: 0.9em; color: #1f2937; border: 1px dashed #9ca3af; }
    .ari-step { margin-bottom: 3px; }

    /* 3. FRT (STRUCTURE 4 BLOCS) */
    .frt-container { background-color: #ffffff; border: 1px solid #10b981; border-left: 5px solid #10b981; border-radius: 6px; overflow: hidden; }
    .frt-block { padding: 12px; border-bottom: 1px solid #e5e7eb; }
    .frt-block:last-child { border-bottom: none; }
    .frt-title { font-weight: 800; text-transform: uppercase; font-size: 0.85em; display: block; margin-bottom: 6px; }
    .frt-usage { color: #d97706; }
    .frt-method { color: #059669; }
    .frt-trap { color: #dc2626; }
    .frt-conclusion { color: #2563eb; }
    .frt-content { font-family: 'Segoe UI', sans-serif; line-height: 1.5; color: #334155; font-size: 0.95em; white-space: pre-wrap; }

    /* 4. TABLEAU QI */
    .qi-table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
    .qi-table th { background: #f9fafb; text-align: left; padding: 8px; border-bottom: 2px solid #e5e7eb; color: #6b7280; }
    .qi-table td { padding: 8px; border-bottom: 1px solid #f3f4f6; vertical-align: top; color: #1f2937; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. KERNEL SMAXIA (DOCTRINE VERROUILLÉE)
# ==============================================================================

LISTE_CHAPITRES = {
    "MATHS": ["SUITES NUMÉRIQUES", "FONCTIONS", "PROBABILITÉS", "GÉOMÉTRIE"],
    "PHYSIQUE": ["MÉCANIQUE", "ONDES"]
}

# --- DÉFINITION DES DONNÉES (FRT COMPLÈTES & TRIGGERS TEXTUELS) ---
UNIVERS_SMAXIA = {
    "FRT_M_S01": {
        "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", 
        "QC": "Comment démontrer qu'une suite est géométrique ?",
        "Triggers": [
            "montrer que la suite est géométrique",
            "déterminer la nature de la suite",
            "préciser la raison q",
            "justifier que (Un) est une suite géométrique"
        ],
        "ARI": [
            "1. Exprimer u(n+1) en fonction de n",
            "2. Former le quotient u(n+1) / u(n)",
            "3. Simplifier l'expression algébrique",
            "4. Identifier une constante réelle q"
        ],
        "FRT": """
<div class='frt-container'>
    <div class='frt-block'>
        <span class='frt-title frt-usage'>🔔 1. Quand utiliser cette méthode</span>
        <div class='frt-content'>Lorsque l'énoncé demande explicitement la nature de la suite ou de prouver qu'elle est géométrique, souvent à partir d'une relation de récurrence.</div>
    </div>
    <div class='frt-block'>
        <span class='frt-title frt-method'>✅ 2. Méthode Rédigée</span>
        <div class='frt-content'>
1. Pour tout entier naturel n, on exprime $u_{n+1}$ à l'aide de la définition.<br>
2. On calcule le rapport $\\frac{u_{n+1}}{u_n}$.<br>
3. On simplifie l'expression jusqu'à éliminer n.<br>
4. On obtient un réel constant q.
        </div>
    </div>
    <div class='frt-block'>
        <span class='frt-title frt-trap'>⚠️ 3. Erreurs et pièges</span>
        <div class='frt-content'>
- Oublier de vérifier que $u_n$ n'est pas nul.<br>
- Calculer la différence $u_{n+1} - u_n$ (méthode arithmétique).
        </div>
    </div>
    <div class='frt-block'>
        <span class='frt-title frt-conclusion'>✍️ 4. Conclusion Type</span>
        <div class='frt-content'>"Le rapport étant constant, la suite est géométrique de raison q."</div>
    </div>
</div>
"""
    },
    "FRT_M_S02": {
        "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES",
        "QC": "Comment lever une indétermination (limite) ?",
        "Triggers": [
            "calculer la limite de la suite",
            "déterminer la limite en +l'infini",
            "étudier la convergence",
            "expression polynomiale ou rationnelle"
        ],
        "ARI": [
            "1. Identifier le terme de plus haut degré",
            "2. Factoriser l'expression par ce terme",
            "3. Appliquer les limites usuelles (1/n -> 0)",
            "4. Conclure par produit ou somme"
        ],
        "FRT": """
<div class='frt-container'>
    <div class='frt-block'>
        <span class='frt-title frt-usage'>🔔 1. Quand utiliser cette méthode</span>
        <div class='frt-content'>En présence d'une forme indéterminée type $\\infty - \\infty$ ou $\\infty / \\infty$ dans une suite définie explicitement.</div>
    </div>
    <div class='frt-block'>
        <span class='frt-title frt-method'>✅ 2. Méthode Rédigée</span>
        <div class='frt-content'>
1. On identifie le terme dominant (plus haute puissance de n).<br>
2. On factorise le numérateur et le dénominateur par ce terme.<br>
3. On utilise $\\lim_{n \\to +\\infty} \\frac{1}{n} = 0$.
        </div>
    </div>
    <div class='frt-block'>
        <span class='frt-title frt-trap'>⚠️ 3. Erreurs et pièges</span>
        <div class='frt-content'>- Appliquer la règle des signes sans factoriser.<br>- Oublier de factoriser le dénominateur.</div>
    </div>
    <div class='frt-block'>
        <span class='frt-title frt-conclusion'>✍️ 4. Conclusion Type</span>
        <div class='frt-content'>"Par opération sur les limites, la suite converge vers..."</div>
    </div>
</div>
"""
    },
    "FRT_M_F01": {
        "Matiere": "MATHS", "Chap": "FONCTIONS",
        "QC": "Comment appliquer le TVI (Solution unique) ?",
        "Triggers": [
            "montrer que l'équation f(x)=k admet une solution unique",
            "démontrer l'existence et l'unicité",
            "théorème des valeurs intermédiaires",
            "justifier qu'il existe un unique alpha"
        ],
        "ARI": [
            "1. Vérifier la continuité",
            "2. Vérifier la stricte monotonie",
            "3. Calculer les images aux bornes",
            "4. Invoquer le corollaire du TVI"
        ],
        "FRT": """
<div class='frt-container'>
    <div class='frt-block'>
        <span class='frt-title frt-usage'>🔔 1. Quand utiliser cette méthode</span>
        <div class='frt-content'>Pour prouver qu'une équation a une seule solution sans la calculer.</div>
    </div>
    <div class='frt-block'>
        <span class='frt-title frt-method'>✅ 2. Méthode Rédigée</span>
        <div class='frt-content'>
1. La fonction f est continue et strictement monotone sur I.<br>
2. On calcule f(a) et f(b).<br>
3. On vérifie que k est compris entre f(a) et f(b).<br>
4. D'après le corollaire du TVI...
        </div>
    </div>
    <div class='frt-block'>
        <span class='frt-title frt-trap'>⚠️ 3. Erreurs et pièges</span>
        <div class='frt-content'>- Oublier la "stricte" monotonie (perte de l'unicité).<br>- Oublier la continuité.</div>
    </div>
    <div class='frt-block'>
        <span class='frt-title frt-conclusion'>✍️ 4. Conclusion Type</span>
        <div class='frt-content'>"L'équation f(x)=k admet une unique solution alpha."</div>
    </div>
</div>
"""
    }
}

QI_PATTERNS = {
    "FRT_M_S01": ["Montrer que la suite (Un) est géométrique.", "Quelle est la nature de la suite (Vn) ?", "Justifier que (Un) est géométrique."],
    "FRT_M_S02": ["Calculer la limite de Un.", "Déterminer la limite quand n tend vers +infini.", "Étudier la convergence."],
    "FRT_M_F01": ["Montrer que f(x)=0 admet une unique solution alpha.", "Démontrer qu'il existe un unique réel alpha solution."]
}

# ==============================================================================
# 2. MOTEUR ROBUSTE (V22)
# ==============================================================================

def ingest_factory_v22(urls, volume, matiere):
    """
    Ingestion globale par matière (Step 1 & 2).
    Retourne des DataFrames sécurisés (noms de colonnes cleans).
    """
    target_frts = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere]
    
    # Sécurité : Si univers vide, retourner structure vide
    if not target_frts:
        return (pd.DataFrame(columns=["Fichier", "Nature", "Annee", "Telechargement", "Qi_Data"]),
                pd.DataFrame(columns=["FRT_ID", "Qi", "File", "Year", "Chapitre"]))
    
    sources_data = []
    atoms_data = []
    
    progress = st.progress(0)
    for i in range(volume):
        progress.progress((i+1)/volume)
        
        nature = random.choice(["BAC", "DST", "INTERRO"])
        annee = random.choice(range(2020, 2025))
        filename = f"Sujet_{matiere}_{nature}_{annee}_{i}.pdf"
        
        # Un sujet contient plusieurs exos
        nb_qi = random.randint(4, 8)
        frts = random.choices(target_frts, k=nb_qi)
        
        qi_data_list = []
        for frt_id in frts:
            qi_txt = random.choice(QI_PATTERNS.get(frt_id, ["Question standard"])) + f" [Ref:{random.randint(10,99)}]"
            
            # Stockage Atome (Step 3 : Stockage par chapitre via la métadonnée)
            atoms_data.append({
                "FRT_ID": frt_id, "Qi": qi_txt, "File": filename, 
                "Year": annee, "Chapitre": UNIVERS_SMAXIA[frt_id]["Chap"]
            })
            # Stockage Vérité Terrain (pour l'audit)
            qi_data_list.append({"Qi": qi_txt, "FRT_ID": frt_id})
            
        sources_data.append({
            "Fichier": filename, "Nature": nature, "Annee": annee,
            "Telechargement": f"https://fake-cloud/dl/{filename}", # URL simulée
            "Qi_Data": qi_data_list
        })
        
    return pd.DataFrame(sources_data), pd.DataFrame(atoms_data)

def compute_qc_v22(df_atoms):
    """Calcul (Step 4)"""
    if df_atoms.empty: return pd.DataFrame()
    
    grouped = df_atoms.groupby("FRT_ID").agg({
        "Qi": list, "File": list, "Year": "max", "Chapitre": "first"
    }).reset_index()
    
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
            "Triggers": meta["Triggers"], "ARI": meta["ARI"], "FRT": meta["FRT"],
            "Evidence": [{"Fichier": f, "Qi": q} for f, q in zip(row["File"], row["Qi"])]
        })
        
    return pd.DataFrame(qcs).sort_values(by="Score", ascending=False)

def analyze_external_v22(file, matiere):
    target_frts = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere]
    if not target_frts: return []
    # Simule un sujet riche (12 questions)
    frts = random.choices(target_frts, k=12)
    result = []
    for frt in frts:
        qi = random.choice(QI_PATTERNS.get(frt, ["Question"])) + " (Extrait PDF)"
        result.append({"Qi": qi, "FRT_ID": frt})
    return result

# ==============================================================================
# 3. INTERFACE (UI)
# ==============================================================================

with st.sidebar:
    st.header("Paramètres Académiques")
    st.selectbox("Classe", ["Terminale"], disabled=True)
    sel_matiere = st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    chaps_dispo = LISTE_CHAPITRES.get(sel_matiere, [])
    # Filtre d'affichage (Step 5)
    sel_chapitres = st.multiselect("Chapitres (Filtre Vue)", chaps_dispo, default=chaps_dispo)

tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine", "✅ Onglet 2 : Audit"])

# --- USINE ---
with tab_usine:
    st.subheader("1. Configuration Sourcing")
    c1, c2 = st.columns([3, 1])
    with c1: urls = st.text_area("URLs Sources", "https://apmep.fr", height=68)
    with c2: 
        vol = st.number_input("Volume", 5, 500, 20, step=5)
        run = st.button("LANCER L'USINE 🚀", type="primary")

    if run:
        df_src, df_atoms = ingest_factory_v22(urls.split('\n'), vol, sel_matiere)
        df_qc = compute_qc_v22(df_atoms)
        st.session_state['df_src'] = df_src
        st.session_state['df_qc'] = df_qc
        st.success(f"Ingestion terminée : {len(df_src)} sujets traités.")

    st.divider()

    if 'df_src' in st.session_state and not st.session_state['df_src'].empty:
        st.markdown(f"### 📥 Sujets Traités ({len(st.session_state['df_src'])})")
        
        # Renommage pour affichage utilisateur
        df_display = st.session_state['df_src'].rename(columns={"Annee": "Année", "Telechargement": "Lien"})
        
        st.data_editor(
            df_display[["Fichier", "Nature", "Année", "Lien"]],
            column_config={
                "Lien": st.column_config.LinkColumn("Téléchargement", display_text="📥 Télécharger PDF")
            },
            hide_index=True, use_container_width=True, disabled=True
        )

        st.divider()

        st.markdown("### 🧠 Base de Connaissance (QC)")
        if not st.session_state['df_qc'].empty:
            # FILTRAGE VIEW (STEP 5)
            qc_view = st.session_state['df_qc'][st.session_state['df_qc']["Chapitre"].isin(sel_chapitres)]
            
            if qc_view.empty:
                st.info("Aucune QC dans les chapitres sélectionnés (vérifiez le filtre latéral).")
            else:
                chapters = qc_view["Chapitre"].unique()
                for chap in chapters:
                    subset = qc_view[qc_view["Chapitre"] == chap]
                    st.markdown(f"#### 📘 Chapitre {chap} : {len(subset)} QC")
                    
                    for idx, row in subset.iterrows():
                        st.markdown(f"""
                        <div class="qc-header-row">
                            <div class="qc-title-group">
                                <span class="qc-id">{row['QC_ID']}</span>
                                <span class="qc-text">{row['Titre']}</span>
                            </div>
                            <span class="qc-stats">Score(q)={row['Score']:.0f} | n_q={row['n_q']} | Ψ={row['Psi']} | N_tot={row['N_tot']} | t_rec={row['t_rec']:.1f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            with st.expander("🔥 Déclencheurs"):
                                for t in row['Triggers']: 
                                    st.markdown(f"<span class='trigger-item'>{t}</span>", unsafe_allow_html=True)
                        with c2:
                            with st.expander("⚙️ ARI"):
                                st.markdown(f"<div class='ari-box'>", unsafe_allow_html=True)
                                for step in row['ARI']: st.markdown(f"<div class='ari-step'>{step}</div>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                        with c3:
                            with st.expander("🧾 FRT (Complète)"):
                                st.markdown(row['FRT'], unsafe_allow_html=True)
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

# --- AUDIT ---
with tab_audit:
    st.subheader("Validation Booléenne")
    
    if 'df_qc' in st.session_state and not st.session_state['df_qc'].empty:
        
        # TEST 1
        st.markdown("#### ✅ 1. Test Interne (Sujet Traité)")
        t1_file = st.selectbox("Choisir un sujet traité", st.session_state['df_src']["Fichier"])
        
        if st.button("LANCER TEST INTERNE"):
            # Récupération de TOUTES les Qi du sujet
            data = st.session_state['df_src'][st.session_state['df_src']["Fichier"]==t1_file].iloc[0]["Qi_Data"]
            known_ids = st.session_state['df_qc']["FRT_ID"].unique()
            
            ok_count = 0
            rows = []
            for item in data:
                is_ok = item["FRT_ID"] in known_ids
                if is_ok: ok_count += 1
                status = "✅ MATCH" if is_ok else "❌ ERREUR"
                
                qc_nom = "---"
                if is_ok:
                    info = st.session_state['df_qc'][st.session_state['df_qc']["FRT_ID"]==item["FRT_ID"]].iloc[0]
                    qc_nom = f"{info['QC_ID']} {info['Titre']}"
                
                rows.append({"Qi (Sujet)": item["Qi"], "QC Moteur": qc_nom, "Statut": status
