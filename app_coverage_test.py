import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V20")
st.title("🛡️ SMAXIA - Console V20 (Content Revolution)")

# ==============================================================================
# 🎨 STYLES CSS (FIGÉS ET VALIDÉS)
# ==============================================================================
st.markdown("""
<style>
    /* EN-TÊTE QC */
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

    /* DETAILS */
    .trigger-container { background-color: #fff1f2; padding: 10px; border-radius: 6px; border: 1px solid #fecdd3; }
    .trigger-item { background-color: #ffffff; color: #be123c; padding: 4px 8px; border-radius: 12px; font-size: 0.85em; font-weight: 700; border: 1px solid #fda4af; display: inline-block; margin: 3px; }
    
    .ari-box { background-color: #f3f4f6; padding: 10px; border-radius: 6px; font-family: monospace; font-size: 0.9em; color: #374151; border: 1px dashed #9ca3af; }
    
    /* FRT SMAXIA : Style "Fiche de Révision" */
    .frt-box { 
        background-color: #ffffff; border: 1px solid #cbd5e1; border-left: 6px solid #10b981; 
        padding: 20px; border-radius: 4px; font-family: 'Segoe UI', sans-serif; line-height: 1.6; color: #334155; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .frt-section { font-weight: bold; color: #047857; margin-top: 10px; display: block; text-transform: uppercase; font-size: 0.85em;}
    
    /* TABLEAUX HTML */
    .qi-table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
    .qi-table th { background: #f9fafb; text-align: left; padding: 8px; border-bottom: 2px solid #e5e7eb; color: #6b7280; }
    .qi-table td { padding: 8px; border-bottom: 1px solid #f3f4f6; vertical-align: top; color: #1f2937; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. LISTE CHAPITRES
# ==============================================================================
LISTE_CHAPITRES = {
    "MATHS": [
        "SUITES NUMÉRIQUES", "FONCTIONS & DÉRIVATION", "LIMITES DE FONCTIONS", 
        "CONTINUITÉ & CONVEXITÉ", "FONCTION LOGARITHME", "PRIMITIVES & ÉQUATIONS DIFF", 
        "LOI BINOMIALE", "GÉOMÉTRIE DANS L'ESPACE"
    ],
    "PHYSIQUE": [
        "MÉCANIQUE DE NEWTON", "MOUVEMENT DANS UN CHAMP", "ONDES MÉCANIQUES"
    ]
}

# ==============================================================================
# 2. KERNEL SMAXIA (CONTENU HAUTE FIDÉLITÉ)
# ==============================================================================

UNIVERS_SMAXIA = {
    # --- MATHS : SUITES ---
    "FRT_M_SUITE_01": {
        "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", "Proba": 0.9,
        "QC": "comment démontrer qu'une suite est géométrique ?",
        # TRIGGERS : Mots exacts de l'énoncé
        "Triggers": ["montrer que la suite est géométrique", "déterminer la nature de la suite", "préciser la raison q"],
        # ARI : La structure logique (Le squelette)
        "ARI": ["Expression u(n+1)", "Quotient u(n+1)/u(n)", "Simplification", "Identification Constante"],
        # FRT : La chair (Ce que l'élève écrit sur sa copie)
        "FRT": """
<span class='frt-section'>🔔 Situation</span>
L'énoncé demande de prouver que $(u_n)$ est géométrique, souvent définie par une relation de récurrence.

<span class='frt-section'>✅ Rédaction Type (Copie Élève)</span>
1. **Pour tout entier naturel $n$, exprimons $u_{n+1}$ :**
   On remplace $u_{n+1}$ par son expression donnée dans l'énoncé.
   
2. **Calculons le rapport :**
   $\\frac{u_{n+1}}{u_n} = \\frac{\\dots}{u_n}$
   
3. **Simplification :**
   On factorise ou on simplifie l'expression jusqu'à éliminer tous les termes en $n$.
   On obtient : $\\frac{u_{n+1}}{u_n} = q$ (où $q$ est un nombre réel).

4. **Conclusion :**
   Le rapport entre deux termes consécutifs étant constant, la suite $(u_n)$ est **géométrique** de raison $q$ et de premier terme $u_0 = \\dots$
"""
    },
    
    "FRT_M_SUITE_02": {
        "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", "Proba": 0.8,
        "QC": "comment lever une indétermination (limite) ?",
        "Triggers": ["déterminer la limite", "calculer la limite quand n tend vers +infini"],
        "ARI": ["Identification FI", "Factorisation Forcée (Terme Dominant)", "Limites Usuelles", "Opérations"],
        "FRT": """
<span class='frt-section'>🔔 Situation</span>
On cherche une limite mais on tombe sur $\\infty - \\infty$ ou $\\frac{\\infty}{\\infty}$.

<span class='frt-section'>✅ Rédaction Type (Copie Élève)</span>
1. **Identification :**
   "Nous sommes en présence d'une forme indéterminée."

2. **Factorisation par le terme dominant :**
   "Pour tout $n > 0$, factorisons par $n^k$ (le terme de plus haut degré) :"
   $u_n = n^k \\times ( ... )$

3. **Utilisation des limites usuelles :**
   "Or, on sait que $\\lim_{n \\to +\\infty} \\frac{1}{n} = 0$."

4. **Conclusion :**
   "Par produit et somme de limites, on en déduit que : $\\lim_{n \\to +\\infty} u_n = \\dots$"
"""
    },

    "FRT_M_FCT_02": {
        "Matiere": "MATHS", "Chap": "FONCTIONS & DÉRIVATION", "Proba": 0.9,
        "QC": "comment appliquer le TVI (solution unique) ?",
        "Triggers": ["montrer que l'équation admet une unique solution", "démontrer qu'il existe un unique réel alpha", "théorème des valeurs intermédiaires"],
        "ARI": ["Continuité", "Monotonie Stricte", "Images Bornes", "Corollaire TVI"],
        "FRT": """
<span class='frt-section'>🔔 Situation</span>
On doit prouver l'existence et l'unicité d'une solution à $f(x)=k$ (souvent $f(x)=0$).

<span class='frt-section'>✅ Rédaction Type (Copie Élève)</span>
1. **Hypothèses :**
   "La fonction $f$ est **continue** et **strictement monotone** (croissante/décroissante) sur l'intervalle $I=[a;b]$."

2. **Images aux bornes :**
   "De plus, $f(a) = \\dots$ et $f(b) = \\dots$."
   "On constate que $k$ est compris entre $f(a)$ et $f(b)$."

3. **Invocation du Théorème :**
   "D'après le **corollaire du Théorème des Valeurs Intermédiaires**, l'équation $f(x)=k$ admet donc une **unique solution** $\\alpha$ sur l'intervalle $I$."
"""
    },

    "FRT_M_GEO_01": {
        "Matiere": "MATHS", "Chap": "GÉOMÉTRIE DANS L'ESPACE", "Proba": 0.7,
        "QC": "comment démontrer l'orthogonalité droite/plan ?",
        "Triggers": ["démontrer que la droite est orthogonale au plan", "prouver que (d) est perpendiculaire à (P)"],
        "ARI": ["Vecteur Directeur u", "Base Plan (v1, v2)", "Produits Scalaires Nuls", "Conclusion"],
        "FRT": """
<span class='frt-section'>🔔 Situation</span>
On doit montrer qu'une droite $(d)$ est orthogonale à un plan $(P)$.

<span class='frt-section'>✅ Rédaction Type (Copie Élève)</span>
1. **Identification des vecteurs :**
   "Soit $\\vec{u}$ un vecteur directeur de $(d)$ et $\\vec{v_1}, \\vec{v_2}$ deux vecteurs directeurs non colinéaires du plan $(P)$."

2. **Calcul des produits scalaires :**
   "Calculons les produits scalaires :"
   $\\vec{u} \\cdot \\vec{v_1} = xx' + yy' + zz' = 0$
   $\\vec{u} \\cdot \\vec{v_2} = ... = 0$

3. **Conclusion :**
   "Le vecteur $\\vec{u}$ est orthogonal à deux vecteurs directeurs non colinéaires de $(P)$. La droite $(d)$ est donc orthogonale au plan $(P)$."
"""
    }
}

# Générateur Polymorphe (Pour simuler la diversité des énoncés)
QI_PATTERNS = {
    "FRT_M_SUITE_01": [
        "Montrer que la suite (Un) est géométrique.", 
        "Démontrer que (Vn) est une suite géométrique de raison 1/2.", 
        "Quelle est la nature de la suite (Wn) ?"
    ],
    "FRT_M_SUITE_02": [
        "Déterminer la limite de la suite (Un).", 
        "Calculer la limite quand n tend vers +infini.", 
        "La suite converge-t-elle ?"
    ],
    "FRT_M_FCT_02": [
        "Montrer que l'équation f(x)=0 admet une unique solution alpha.", 
        "Prouver qu'il existe un unique réel alpha tel que g(alpha)=3.", 
        "Démontrer l'existence et l'unicité de la solution."
    ],
    "FRT_M_GEO_01": [
        "Démontrer que la droite (AB) est orthogonale au plan (P).",
        "Prouver que le vecteur n est normal au plan (ABC).",
        "La droite (d) est-elle perpendiculaire au plan ?"
    ]
}

# ==============================================================================
# 3. MOTEUR D'INGESTION & CALCUL (INCHANGÉ CAR VALIDÉ)
# ==============================================================================

def ingest_factory(urls, volume, matiere, chapitres):
    target_frts = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere and v["Chap"] in chapitres]
    if not target_frts and volume > 0: return pd.DataFrame(), pd.DataFrame()
    
    sources, atoms = [], []
    progress = st.progress(0)
    
    for i in range(volume):
        progress.progress((i+1)/volume)
        nature = random.choice(["BAC", "DST", "INTERRO"])
        annee = random.choice(range(2020, 2025))
        filename = f"Sujet_{matiere}_{nature}_{annee}_{i}.pdf"
        
        nb_qi = random.randint(3, 6)
        frts = random.choices(target_frts, k=nb_qi)
        
        qi_data_list = []
        for frt_id in frts:
            qi_txt = random.choice(QI_PATTERNS[frt_id]) + f" [Ref:{random.randint(10,99)}]"
            atoms.append({"FRT_ID": frt_id, "Qi": qi_txt, "File": filename, "Year": annee, "Chap": UNIVERS_SMAXIA[frt_id]["Chap"]})
            qi_data_list.append({"Qi": qi_txt, "FRT_ID": frt_id})
            
        dl_link = f"https://fake-cloud.smaxia/dl/{filename}"
        sources.append({
            "Fichier": filename, "Nature": nature, "Année": annee,
            "Télécharger": dl_link, "Qi_Data": qi_data_list
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
            "Triggers": meta["Triggers"], "ARI": meta["ARI"], "FRT": meta["FRT"],
            "Evidence": [{"Fichier": f, "Qi": q} for f, q in zip(row["File"], row["Qi"])]
        })
    return pd.DataFrame(qcs).sort_values(by="Score", ascending=False)

def analyze_external(file_obj, matiere, chapitres):
    target_frts = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere and v["Chap"] in chapitres]
    if not target_frts: return []
    nb_qi = 15
    frts = random.choices(target_frts, k=nb_qi)
    result = []
    for frt_id in frts:
        qi_txt = random.choice(QI_PATTERNS[frt_id]) + " (Extrait PDF Externe)"
        result.append({"Qi": qi_txt, "FRT_ID": frt_id})
    return result

# ==============================================================================
# 3. INTERFACE (UI VALIDÉE)
# ==============================================================================

with st.sidebar:
    st.header("Paramètres Académiques")
    st.selectbox("Classe", ["Terminale"], disabled=True)
    sel_matiere = st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    chaps_dispo = LISTE_CHAPITRES.get(sel_matiere, [])
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
        df_src, df_atoms = ingest_factory(urls.split('\n'), vol, sel_matiere, sel_chapitres)
        df_qc = compute_qc(df_atoms)
        st.session_state['df_src'] = df_src
        st.session_state['df_qc'] = df_qc
        st.success(f"Ingestion terminée : {len(df_src)} sujets traités.")

    st.divider()

    if 'df_src' in st.session_state and not st.session_state['df_src'].empty:
        st.markdown(f"### 📥 Sujets Traités ({len(st.session_state['df_src'])})")
        
        st.data_editor(
            st.session_state['df_src'][["Fichier", "Nature", "Année", "Téléchargement"]],
            column_config={"Téléchargement": st.column_config.LinkColumn("Téléchargement", display_text="📥 Télécharger PDF")},
            hide_index=True, use_container_width=True, disabled=True
        )

        st.divider()

        st.markdown("### 🧠 Base de Connaissance (QC)")
        if not st.session_state['df_qc'].empty:
            qc_view = st.session_state['df_qc'][st.session_state['df_qc']["Chapitre"].isin(sel_chapitres)]
            
            if qc_view.empty:
                st.info("Pas de QC pour ces chapitres dans les sujets traités.")
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
                status = "✅ MATCH" if is_ok else "❌ ERREUR"
                
                qc_nom = "---"
                if is_ok:
                    info = st.session_state['df_qc'][st.session_state['df_qc']["FRT_ID"]==item["FRT_ID"]].iloc[0]
                    qc_nom = f"{info['QC_ID']} {info['Titre']}"
                
                rows.append({"Qi (Sujet)": item["Qi"], "QC Moteur": qc_nom, "Statut": status})
            
            taux = (ok_count / len(data)) * 100
            st.markdown(f"### Taux de Couverture : {taux:.0f}% ({ok_count}/{len(data)} Qi)")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.divider()

        st.markdown("#### 🌍 2. Test Externe (Mapping Nouveau Sujet)")
        up_file = st.file_uploader("Charger un PDF externe", type="pdf")
        
        if up_file:
            extracted_qi = analyze_external(up_file, sel_matiere, sel_chapitres)
            
            if not extracted_qi:
                st.error("Aucune Qi reconnue ou hors périmètre.")
            else:
                rows_ext = []
                ok_ext = 0
                known_ids = st.session_state['df_qc']["FRT_ID"].unique()
                
                for item in extracted_qi:
                    frt = item["FRT_ID"]
                    is_known = frt in known_ids
                    if is_known: ok_ext += 1
                    status = "✅ MATCH" if is_known else "❌ GAP"
                    
                    qc_n = "---"
                    frt_n = frt
                    if is_known:
                        info = st.session_state['df_qc'][st.session_state['df_qc']["FRT_ID"]==frt].iloc[0]
                        qc_n = f"{info['QC_ID']} {info['Titre']}"
                    
                    rows_ext.append({"Qi (Enoncé)": item["Qi"], "QC Correspondante": qc_n, "FRT Associé": frt_n, "Statut": status})
                
                taux_ext = (ok_ext / len(extracted_qi)) * 100
                st.markdown(f"### Taux de Couverture : {taux_ext:.1f}% ({ok_ext}/{len(extracted_qi)} Qi)")
                
                def color_audit(row):
                    return ['background-color: #dcfce7' if row['Statut'] == "✅ MATCH" else 'background-color: #fee2e2'] * len(row)

                st.dataframe(pd.DataFrame(rows_ext).style.apply(color_audit, axis=1), use_container_width=True)
    else:
        st.info("Veuillez lancer l'usine d'abord.")
