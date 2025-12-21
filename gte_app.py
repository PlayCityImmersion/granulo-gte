# gte_app.py
# GRANULO TEST ENGINE — GTE-T1
# Rôle : Banc d'audit impitoyable

import streamlit as st
import pandas as pd
from granulo_engine import GranuloEngine
from PyPDF2 import PdfReader

st.set_page_config(page_title="GTE — Granulo Test Engine", layout="wide")

st.title("🔴 GRANULO TEST ENGINE")
st.subheader("Auditeur mathématique — Verdict binaire")

# ======================
# SIDEBAR
# ======================
st.sidebar.header("Configuration")

uploaded_files = st.sidebar.file_uploader(
    "Charger sujets (PDF ou TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

similarity = st.sidebar.slider(
    "Seuil de similarité sémantique",
    min_value=0.7,
    max_value=0.95,
    value=0.85,
    step=0.01
)

engine = GranuloEngine(similarity_threshold=similarity)

# ======================
# LOAD FILES
# ======================
def read_files(files):
    texts = []
    for f in files:
        if f.name.endswith(".pdf"):
            reader = PdfReader(f)
            content = "\n".join(page.extract_text() or "" for page in reader.pages)
            texts.append(content)
        else:
            texts.append(f.read().decode("utf-8"))
    return texts

# ======================
# RUN TEST
# ======================
if st.button("🚨 LANCER GRANULO 15"):

    if not uploaded_files:
        st.error("Aucun fichier chargé.")
    else:
        texts = read_files(uploaded_files)
        result = engine.process(texts)

        qcs = result["qcs"]
        coverage = result["coverage"]
        orphans = result["orphans"]

        # ======================
        # SECTION 1 — QC
        # ======================
        st.header("1️⃣ QC Invariantes détectées")

        df_qc = pd.DataFrame([{
            "QC_ID": q.qc_id,
            "V": q.signature.verb,
            "O": q.signature.obj,
            "C": q.signature.context,
            "Psi": q.psi_score,
            "Sigma": q.sigma_class,
            "Qi couvertes": q.qi_covered,
            "Black Swan": q.is_black_swan
        } for q in qcs])

        st.dataframe(df_qc, use_container_width=True)

        # ======================
        # SECTION 2 — COUVERTURE
        # ======================
        st.header("2️⃣ Test de couverture (booléen)")

        if coverage >= 0.95:
            st.success(f"🟢 COUVERTURE OK — {coverage*100:.1f}%")
        else:
            st.error(f"🔴 ÉCHEC — Couverture {coverage*100:.1f}%")

        # ======================
        # SECTION 3 — ORPHELINES
        # ======================
        st.header("3️⃣ Questions orphelines (FAILURES)")

        if not orphans:
            st.success("Aucune Qi orpheline détectée.")
        else:
            st.warning(f"{len(orphans)} Qi sans QC parente")
            st.write(orphans)
