import streamlit as st
import pandas as pd
from collections import defaultdict

# ==============================================================================
# CONFIG
# ==============================================================================
st.set_page_config(
    page_title="SMAXIA - Console V31 (Saturation Proof)",
    layout="wide"
)

st.title("🛡️ SMAXIA - Console V31 (Saturation Proof)")

# ==============================================================================
# STYLES CSS — UI PREMIUM
# ==============================================================================
st.markdown("""
<style>

/* QC HEADER */
.qc-header {
    background: #f8fafc;
    border-left: 6px solid #2563eb;
    padding: 16px;
    border-radius: 6px;
    margin-bottom: 12px;
}
.qc-title {
    font-size: 1.15em;
    font-weight: 800;
    color: #111827;
}
.qc-meta {
    margin-top: 6px;
    font-family: monospace;
    font-size: 0.85em;
    background: #e5e7eb;
    padding: 4px 8px;
    border-radius: 4px;
    display: inline-block;
}

/* TRIGGERS */
.trigger {
    background: #fff1f2;
    border-left: 4px solid #ef4444;
    padding: 6px 10px;
    border-radius: 4px;
    margin-bottom: 6px;
    font-weight: 600;
}

/* ARI */
.ari-step {
    background: #f3f4f6;
    border: 1px dashed #cbd5f5;
    padding: 6px;
    border-radius: 4px;
    margin-bottom: 5px;
    font-family: monospace;
}

/* FRT BLOCKS */
.frt {
    padding: 12px;
    border-radius: 6px;
    margin-bottom: 8px;
    border-left: 6px solid;
}
.frt-usage { background:#fff7ed; border-color:#f59e0b; }
.frt-method { background:#f0fdf4; border-color:#22c55e; }
.frt-trap { background:#fef2f2; border-color:#ef4444; }
.frt-conc { background:#eff6ff; border-color:#3b82f6; }

.frt-title {
    font-weight: 800;
    font-size: 0.8em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

/* QI */
.file-box {
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    margin-bottom: 10px;
}
.file-header {
    background: #f1f5f9;
    padding: 8px 12px;
    font-weight: 700;
}
.qi {
    padding: 8px 12px;
    border-left: 4px solid #8b5cf6;
    font-family: Georgia, serif;
}

</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.header("Paramètres Académiques")
    st.selectbox("Classe", ["Terminale"], disabled=True)
    matiere = st.selectbox("Matière", ["MATHS", "PHYSIQUE"])
    chapitres = {
        "MATHS": ["SUITES NUMÉRIQUES", "FONCTIONS", "PROBABILITÉS", "GÉOMÉTRIE"],
        "PHYSIQUE": ["MÉCANIQUE", "ONDES"]
    }
    sel_chaps = st.multiselect("Chapitres", chapitres[matiere])

# ==============================================================================
# TABS
# ==============================================================================
tab_usine, tab_audit = st.tabs(["🏭 Onglet 1 : Usine", "✅ Onglet 2 : Audit"])

# ==============================================================================
# ONGLET 1 — USINE
# ==============================================================================
with tab_usine:

    st.subheader("🧪 Injection des sujets")

    c1, c2 = st.columns([3,1])
    with c1:
        urls = st.text_area("URLs Sources (références)", "https://apmep.fr")
    with c2:
        volume = st.number_input("Volume de sujets", 1, 200, 15)
        st.button("🚀 LANCER L'USINE")

    st.divider()

    # --------------------------------------------------------------------------
    # SUJETS TRAITÉS (FAKE UI)
    # --------------------------------------------------------------------------
    st.subheader("📥 Sujets traités")

    df_sources = pd.DataFrame([
        {"Fichier":"Sujet_MATHS_INTERRO_2021.pdf","Nature":"INTERRO","Année":2021,"Source":"APMEP"},
        {"Fichier":"Sujet_MATHS_BAC_2024.pdf","Nature":"BAC","Année":2024,"Source":"Education Nationale"}
    ])

    st.dataframe(df_sources, use_container_width=True)

    st.divider()

    # --------------------------------------------------------------------------
    # BASE QC
    # --------------------------------------------------------------------------
    st.subheader("🧠 Base de connaissance (QC)")

    st.markdown("""
    <div class="qc-header">
        <div class="qc-title">
            Chapitre : SUITES NUMÉRIQUES — QC-03 : Comment lever une indétermination (limite) ?
        </div>
        <div class="qc-meta">
            Score(q)=212 | n_q=25 | Ψ=0.85 | N_tot=60 | t_réc=2.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("### 🔥 Déclencheurs")
        for t in ["calculer la limite","limite quand n tend vers +∞","étudier la convergence"]:
            st.markdown(f"<div class='trigger'>{t}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("### ⚙️ ARI")
        for s in [
            "1. Identifier le terme dominant",
            "2. Factoriser",
            "3. Utiliser les limites usuelles",
            "4. Conclure"
        ]:
            st.markdown(f"<div class='ari-step'>{s}</div>", unsafe_allow_html=True)

    with c3:
        st.markdown("### 📘 FRT")
        st.markdown("<div class='frt frt-usage'><div class='frt-title'>Quand utiliser</div>Forme indéterminée ∞/∞.</div>", unsafe_allow_html=True)
        st.markdown("<div class='frt frt-method'><div class='frt-title'>Méthode rédigée</div>Identifier le terme dominant. Factoriser. Appliquer les limites usuelles.</div>", unsafe_allow_html=True)
        st.markdown("<div class='frt frt-trap'><div class='frt-title'>Pièges</div>Règle des signes sans factorisation.</div>", unsafe_allow_html=True)
        st.markdown("<div class='frt frt-conc'><div class='frt-title'>Conclusion</div>La suite converge vers une limite finie.</div>", unsafe_allow_html=True)

    with c4:
        st.markdown("### 📄 Qi associées")
        qi_map = {
            "Sujet_MATHS_INTERRO_2021.pdf":[
                "Déterminer la limite. [Ref:94]",
                "Calculer la limite en +∞. [Ref:77]"
            ],
            "Sujet_MATHS_BAC_2024.pdf":[
                "Déterminer la limite. [Ref:71]",
                "Calculer la limite en +∞. [Ref:63]"
            ]
        }
        for f, qs in qi_map.items():
            html = f"<div class='file-box'><div class='file-header'>{f}</div>"
            for q in qs:
                html += f"<div class='qi'>{q}</div>"
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

# ==============================================================================
# ONGLET 2 — AUDIT (UI ONLY)
# ==============================================================================
with tab_audit:

    st.subheader("🔍 Audit du moteur Granulo")

    st.success("Audit interne — objectif : 100 % de couverture Qi → QC")
    st.info("Audit externe — objectif ≥ 95 % de couverture")

    st.caption("Aucune logique métier implémentée ici.")

