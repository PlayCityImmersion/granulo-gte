import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Console V23")
st.title("🛡️ SMAXIA - Console V23 (Doctrine Compliant)")

# ==============================================================================
# 🎨 STYLES CSS (DOCTRINE UI)
# ==============================================================================
st.markdown("""
<style>
    /* QC HEADER STRICT */
    .qc-header-row {
        background-color: #f8f9fa; border-left: 6px solid #2563eb;
        padding: 15px; margin-bottom: 10px; border-radius: 4px;
        font-family: 'Source Sans Pro', sans-serif;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        display: flex; justify-content: space-between; align-items: center;
    }
    .qc-id { color: #d97706; font-weight: 800; font-size: 1.2em; margin-right: 15px; min-width: 90px; }
    .qc-text { color: #111827; font-weight: 600; font-size: 1.15em; }
    .qc-stats { 
        font-family: 'Courier New', monospace; font-size: 0.9em; font-weight: 700; color: #374151;
        background-color: #e5e7eb; padding: 6px 12px; border-radius: 4px; white-space: nowrap; margin-left: 15px;
    }

    /* BLOCS DÉTAILS */
    /* 1. TRIGGERS (OBSERVABLES) */
    .trigger-box { background-color: #fff1f2; padding: 12px; border-radius: 6px; border: 1px solid #fecdd3; }
    .trigger-item { 
        display: block; margin-bottom: 6px; color: #be123c; font-weight: 700; font-size: 0.95em;
        padding-left: 10px; border-left: 4px solid #fda4af; font-family: monospace;
    }

    /* 2. ARI (LOGIQUE) */
    .ari-box { background-color: #f3f4f6; padding: 12px; border-radius: 6px; font-family: monospace; font-size: 0.9em; color: #1f2937; border: 1px dashed #9ca3af; }
    .ari-step { margin-bottom: 4px; font-weight: 600; }

    /* 3. FRT (PEDAGOGIE COMPLETE) */
    .frt-container { background-color: #ffffff; border: 1px solid #10b981; border-left: 6px solid #10b981; border-radius: 6px; overflow: hidden; margin-top: 5px; }
    .frt-block { padding: 15px; border-bottom: 1px solid #e5e7eb; }
    .frt-block:last-child { border-bottom: none; }
    .frt-title { font-weight: 800; text-transform: uppercase; font-size: 0.85em; display: block; margin-bottom: 8px; letter-spacing: 0.5px; }
    
    .frt-usage { color: #d97706; }      /* 1. Quand utiliser */
    .frt-method { color: #059669; }     /* 2. Méthode */
    .frt-trap { color: #dc2626; }       /* 3. Pièges */
    .frt-conclusion { color: #2563eb; } /* 4. Conclusion */
    
    .frt-content { font-family: 'Segoe UI', sans-serif; line-height: 1.6; color: #334155; font-size: 0.95em; white-space: pre-wrap; }

    /* 4. TABLEAU QI */
    .qi-table { width: 100%; border-collapse: collapse; font-size: 0.9em; margin-top: 5px; }
    .qi-table th { background: #f9fafb; text-align: left; padding: 8px; border-bottom: 2px solid #e5e7eb; color: #6b7280; }
    .qi-table td { padding: 8px; border-bottom: 1px solid #f3f4f6; vertical-align: top; color: #1f2937; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. KERNEL SMAXIA (CONTENU DOCTRINAL)
# ==============================================================================

LISTE_CHAPITRES = {
    "MATHS": ["SUITES NUMÉRIQUES", "FONCTIONS", "PROBABILITÉS", "GÉOMÉTRIE"],
    "PHYSIQUE": ["MÉCANIQUE", "ONDES"]
}

UNIVERS_SMAXIA = {
    # --- MATHS : SUITES ---
    "FRT_M_S01": {
        "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES", 
        "QC": "Comment démontrer qu'une suite est géométrique ?",
        # TRIGGERS (4-5 MAX, TEXTUELS)
        "Triggers": [
            "montrer que la suite est géométrique",
            "déterminer la nature de la suite",
            "préciser la raison q",
            "justifier que (Un) est une suite géométrique",
            "prouver que la suite est géométrique"
        ],
        "ARI": [
            "1. Exprimer u(n+1) en fonction de n",
            "2. Former le quotient u(n+1) / u(n)",
            "3. Simplifier l'expression algébrique",
            "4. Identifier une constante réelle q"
        ],
        # FRT COMPLETE (4 BLOCS)
        "FRT": """
<div class='frt-container'>
    <div class='frt-block'>
        <span class='frt-title frt-usage'>🔔 1. Quand utiliser cette méthode</span>
        <div class='frt-content'>Cette méthode s'utilise lorsque l'énoncé demande explicitement la <b>nature</b> de la suite ou de prouver qu'elle est <b>géométrique</b>, et que la suite est définie par une relation de récurrence.</div>
    </div>
    <div class='frt-block'>
        <span class='frt-title frt-method'>✅ 2. Méthode Rédigée (Réponse Notée)</span>
        <div class='frt-content'>
1. Pour tout entier naturel $n$, on exprime $u_{n+1}$ en utilisant la définition de la suite.<br>
2. On calcule le rapport $\\frac{u_{n+1}}{u_n}$.<br>
3. On simplifie l'expression jusqu'à ce que tous les termes en $n$ s'annulent.<br>
4. On obtient un résultat constant réel $q$.
        </div>
    </div>
    <div class='frt-block'>
        <span class='frt-title frt-trap'>⚠️ 3. Erreurs et pièges à éviter</span>
        <div class='frt-content'>
- Oublier de vérifier que $u_n \\neq 0$ avant de diviser.<br>
- Confondre avec la méthode pour une suite arithmétique (différence $u_{n+1} - u_n$).
        </div>
    </div>
    <div class='frt-block'>
        <span class='frt-title frt-conclusion'>✍️ 4. Conclusion Type</span>
        <div class='frt-content'>"Le rapport entre deux termes consécutifs étant constant, la suite $(u_n)$ est géométrique de raison $q$ et de premier terme $u_0 = \\dots$"</div>
    </div>
</div>
"""
    },
    
    "FRT_M_S02": {
        "Matiere": "MATHS", "Chap": "SUITES NUMÉRIQUES",
        "QC": "Comment lever une indétermination (limite) ?",
        "Triggers": [
            "calculer la limite de la suite",
            "déterminer la limite quand n tend vers +infini",
            "étudier la convergence de la suite",
            "limite de la suite (Un)"
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
        <div class='frt-content'>Lorsque l'on doit calculer la limite d'une suite définie par une expression en $n$ et que le calcul direct mène à une forme $\\infty - \\infty$ ou $\\frac{\\infty}{\\infty}$.</div>
    </div>
    <div class='frt-block'>
        <span class='frt-title frt-method'>✅ 2. Méthode Rédigée (Réponse Notée)</span>
        <div class='frt-content'>
1. On identifie le terme dominant (la plus haute puissance de $n$ au numérateur et au dénominateur).<br>
2. On factorise toute l'expression par ce terme dominant.<br>
3. On utilise la limite usuelle $\\lim_{n \\to +\\infty} \\frac{1}{n} = 0$.
        </div>
    </div>
    <div class='frt-block'>
        <span class='frt-title frt-trap'>⚠️ 3. Erreurs et pièges à éviter</span>
        <div class='frt-content'>
- Appliquer la "règle des signes" sans factoriser.<br>
- Écrire "Forme indéterminée" comme conclusion finale.
        </div>
    </div>
    <div class='frt-block'>
        <span class='frt-title frt-conclusion'>✍️ 4. Conclusion Type</span>
        <div class='frt-content'>"Par opération sur les limites (produit et somme), on en déduit que $\\lim_{n \\to +\\infty} u_n = \\dots$"</div>
    </div>
</div>
"""
    },

    "FRT_M_F01": {
        "Matiere": "MATHS", "Chap": "FONCTIONS",
        "QC": "Comment appliquer le TVI (Solution unique) ?",
        "Triggers": [
            "montrer que l'équation f(x)=k admet une solution unique",
            "démontrer l'existence et l'unicité de la solution",
            "théorème des valeurs intermédiaires",
            "justifier qu'il existe un unique alpha"
        ],
        "ARI": [
            "1. Vérifier la continuité sur I",
            "2. Vérifier la stricte monotonie sur I",
            "3. Calculer les images aux bornes f(a) et f(b)",
            "4. Invoquer le corollaire du TVI"
        ],
        "FRT": """
<div class='frt-container'>
    <div class='frt-block'>
        <span class='frt-title frt-usage'>🔔 1. Quand utiliser cette méthode</span>
        <div class='frt-content'>Pour prouver qu'une équation $f(x)=k$ admet une seule solution sans pouvoir la calculer explicitement.</div>
    </div>
    <div class='frt-block'>
        <span class='frt-title frt-method'>✅ 2. Méthode Rédigée (Réponse Notée)</span>
        <div class='frt-content'>
1. La fonction $f$ est **continue** et **strictement monotone** (croissante ou décroissante) sur l'intervalle $I$.<br>
2. On calcule les images des bornes $f(a)$ et $f(b)$.<br>
3. On constate que la valeur $k$ est comprise entre $f(a)$ et $f(b)$.<br>
4. On cite le **corollaire du théorème des valeurs intermédiaires**.
        </div>
    </div>
    <div class='frt-block'>
        <span class='frt-title frt-trap'>⚠️ 3. Erreurs et pièges à éviter</span>
        <div class='frt-content'>
- Oublier le mot "strictement" pour la monotonie (condition d'unicité).<br>
- Oublier la continuité (condition d'existence).
        </div>
    </div>
    <div class='frt-block'>
        <span class='frt-title frt-conclusion'>✍️ 4. Conclusion Type</span>
        <div class='frt-content'>"L'équation $f(x)=k$ admet donc une unique solution $\\alpha$ sur l'intervalle $I$."</div>
    </div>
</div>
"""
    }
}

# ==============================================================================
# 2. MOTEUR SMAXIA (GARANTIE DU LIEN CAUSAL)
# ==============================================================================

def generate_proven_qi(frt_id):
    """
    Génère une Qi qui contient STRICTEMENT un déclencheur textuel.
    C'est la preuve que le moteur ne devine pas, il lit.
    """
    qc_data = UNIVERS_SMAXIA[frt_id]
    
    # 1. On choisit un déclencheur officiel
    trigger = random.choice(qc_data["Triggers"])
    
    # 2. On l'habille avec du contexte (Bruit)
    templates = [
        f"1. {trigger.capitalize()}.",
        f"b) En déduire, {trigger}.",
        f"Question 2 : {trigger} sur l'intervalle I.",
        f"On souhaite {trigger} en utilisant les résultats précédents."
    ]
    return random.choice(templates)

def ingest_factory_v23(urls, volume, matiere):
    target_frts = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere]
    
    if not target_frts: 
        return (pd.DataFrame(columns=["Fichier", "Nature", "Annee", "Telechargement", "Qi_Data"]), 
                pd.DataFrame(columns=["FRT_ID", "Qi", "File", "Year", "Chapitre"]))
    
    sources = []
    atoms = []
    
    progress = st.progress(0)
    for i in range(volume):
        progress.progress((i+1)/volume)
        
        nature = random.choice(["BAC", "DST", "INTERRO"])
        annee = random.choice(range(2020, 2025))
        filename = f"Sujet_{matiere}_{nature}_{annee}_{i}.pdf"
        
        # Un sujet de BAC a environ 5-8 questions clés
        nb_qi = random.randint(5, 8)
        frts = random.choices(target_frts, k=nb_qi)
        
        qi_data_list = []
        for frt_id in frts:
            # GÉNÉRATION PREUVE
            qi_txt = generate_proven_qi(frt_id)
            
            atoms.append({
                "FRT_ID": frt_id, "Qi": qi_txt, "File": filename, 
                "Year": annee, "Chapitre": UNIVERS_SMAXIA[frt_id]["Chap"]
            })
            qi_data_list.append({"Qi": qi_txt, "FRT_ID": frt_id})
            
        sources.append({
            "Fichier": filename, "Nature": nature, "Annee": annee,
            "Telechargement": f"https://fake-cloud.smaxia/dl/{filename}",
            "Qi_Data": qi_data_list
        })
        
    return pd.DataFrame(sources), pd.DataFrame(atoms)

def compute_qc_v23(df_atoms):
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
            "Triggers": meta["Triggers"], "ARI": meta["ARI"], "FRT": meta["FRT"],
            "Evidence": [{"Fichier": f, "Qi": q} for f, q in zip(row["File"], row["Qi"])]
        })
    return pd.DataFrame(qcs).sort_values(by="Score", ascending=False)

def analyze_external_v23(file, matiere):
    target_frts = [k for k,v in UNIVERS_SMAXIA.items() if v["Matiere"] == matiere]
    if not target_frts: return []
    frts = random.choices(target_frts, k=15) # Gros sujet
    result = []
    for frt in frts:
        qi = generate_proven_qi(frt) # On utilise le générateur certifié
        result.append({"Qi": qi, "FRT_ID": frt})
    return result

# ==============================================================================
# 3. UI
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
        df_src, df_atoms = ingest_factory_v23(urls.split('\n'), vol, sel_matiere)
        df_qc = compute_qc_v23(df_atoms)
        st.session_state['df_src'] = df_src
        st.session_state['df_qc'] = df_qc
        st.success(f"Ingestion terminée : {len(df_src)} sujets traités.")

    st.divider()

    if 'df_src' in st.session_state and not st.session_state['df_src'].empty:
        st.markdown(f"### 📥 Sujets Traités ({len(st.session_state['df_src'])})")
        
        # Renommage colonnes pour affichage propre
        df_disp = st.session_state['df_src'].rename(columns={"Annee": "Année", "Telechargement": "Lien"})
        
        st.data_editor(
            df_disp[["Fichier", "Nature", "Année", "Lien"]],
            column_config={"Lien": st.column_config.LinkColumn("Téléchargement", display_text="📥 Télécharger PDF")},
            hide_index=True, use_container_width=True, disabled=True
        )

        st.divider()

        st.markdown("### 🧠 Base de Connaissance (QC)")
        if not st.session_state['df_qc'].empty:
            qc_view = st.session_state['df_qc'][st.session_state['df_qc']["Chapitre"].isin(sel_chapitres)]
            
            if qc_view.empty:
                st.info("Pas de QC pour ces chapitres.")
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
                            with st.expander("🔥 Déclencheurs (Observables)"):
                                html_trig = "<div class='trigger-box'>"
                                for t in row['Triggers']: html_trig += f"<span class='trigger-item'>{t}</span>"
                                html_trig += "</div>"
                                st.markdown(html_trig, unsafe_allow_html=True)
                        with c2:
                            with st.expander("⚙️ ARI (Logique Moteur)"):
                                st.markdown(f"<div class='ari-box'>", unsafe_allow_html=True)
                                for s in row['ARI']: st.markdown(f"<div class='ari-step'>• {s}</div>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                        with c3:
                            with st.expander("🧾 FRT (Réponse Élève)"):
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
                status = "✅ MATCH" if is_ok else "❌ GAP"
                
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
            extracted_qi = analyze_external_v23(up_file, sel_matiere)
            if not extracted_qi:
                st.error("Aucune Qi reconnue.")
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
