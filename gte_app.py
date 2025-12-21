import streamlit as st
import pdfplumber
import re
import uuid
import pandas as pd

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title="SMAXIA – Granulo Test Engine",
    layout="wide"
)

st.title("SMAXIA – Granulo Test Engine (MODE TEST STRICT)")
st.caption("QC • Déclencheurs • Mapping • FRT • F1/F2/F3 • BOOLÉEN")

# ======================================================
# EXTRACTION QI
# ======================================================
def extract_text(file):
    if file.type == "application/pdf":
        text = ""
        with pdfplumber.open(file) as pdf:
            for p in pdf.pages:
                if p.extract_text():
                    text += p.extract_text() + "\n"
        return text
    return file.read().decode("utf-8")


def split_qi(text):
    chunks = re.split(r"(?:Exercice|Question|\n\d+\.)", text)
    qi = []
    for c in chunks:
        t = c.strip()
        if len(t) >= 80:
            qi.append({
                "qi_id": str(uuid.uuid4())[:8],
                "text": t
            })
    return qi

# ======================================================
# GRANULO – LOGIQUE TEST (NON PROD)
# ======================================================
def detect_operator(qi_text):
    t = qi_text.lower()
    if "calcul" in t or "déterminer" in t:
        return "CALCULER"
    if "démontrer" in t or "montrer" in t:
        return "DEMONTRER"
    if "résoudre" in t or "solution" in t:
        return "RESOUDRE"
    if "étudier" in t or "variation" in t:
        return "ETUDIER"
    return None


def build_qc(qi_list):
    qc_map = {}
    rejected = []

    for qi in qi_list:
        op = detect_operator(qi["text"])
        if not op:
            rejected.append(qi)
            continue

        if op not in qc_map:
            qc_map[op] = {
                "qc_id": f"QC_{op}",
                "operator": op,
                "triggers": ["OPERATOR=" + op],
                "qi": []
            }
        qc_map[op]["qi"].append(qi)

    return list(qc_map.values()), rejected


def build_frt(qc):
    return {
        "INPUT": "Données mathématiques structurées",
        "STEPS": [
            f"Appliquer l’opérateur {qc['operator']}",
            "Effectuer les transformations algébriques nécessaires",
            "Vérifier la cohérence du résultat"
        ],
        "OUTPUT": "Résultat mathématique valide"
    }


def compute_F1(qc):
    # Densité opératoire
    return len(qc["qi"])


def compute_F2(qc, total_qi):
    # Couverture
    return round(len(qc["qi"]) / total_qi, 3)


def compute_F3(qc):
    # Criticité pédagogique (bloquant si peu de QC)
    return 1 if len(qc["qi"]) >= 2 else 0

# ======================================================
# UI
# ======================================================
uploaded_files = st.file_uploader(
    "Injecter sujets (PDF/TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    all_qi = []
    for f in uploaded_files:
        all_qi.extend(split_qi(extract_text(f)))

    st.metric("Qi extraites", len(all_qi))

    qc_list, rejected_qi = build_qc(all_qi)

    st.divider()
    st.header("📦 LIVRABLES GRANULO (TEST)")

    total_qi = len(all_qi)
    mapped_qi = 0

    rows = []

    for qc in qc_list:
        mapped_qi += len(qc["qi"])
        frt = build_frt(qc)

        F1 = compute_F1(qc)
        F2 = compute_F2(qc, total_qi)
        F3 = compute_F3(qc)

        rows.append({
            "QC_ID": qc["qc_id"],
            "OPERATEUR": qc["operator"],
            "DECLENCHEURS": qc["triggers"],
            "NB_QI": len(qc["qi"]),
            "F1": F1,
            "F2": F2,
            "F3": F3
        })

        with st.expander(f"{qc['qc_id']} – {qc['operator']}"):
            st.markdown("### Déclencheurs")
            st.write(qc["triggers"])

            st.markdown("### Mapping QC ⇄ Qi")
            for qi in qc["qi"]:
                st.write(f"- {qi['qi_id']} : {qi['text'][:120]}…")

            st.markdown("### FRT (référence)")
            st.json(frt)

            st.markdown("### Métriques")
            st.write({"F1": F1, "F2": F2, "F3": F3})

    st.divider()
    st.subheader("Tableau de synthèse")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ======================================================
    # BOOLÉEN FINAL
    # ======================================================
    B_QC = len(qc_list) > 0
    B_MAPPING = mapped_qi == total_qi
    B_REJECT = len(rejected_qi) == 0

    B_TEST = B_QC and B_MAPPING and B_REJECT

    st.divider()
    st.header("🧮 VERDICT BOOLÉEN")

    st.write({
        "B_QC_CANONIQUES": B_QC,
        "B_MAPPING_TOTAL": B_MAPPING,
        "B_QI_REJETÉES": B_REJECT,
        "B_TEST": B_TEST
    })

    if B_TEST:
        st.success("✅ TEST VALIDÉ — GRANULO SAIN — P6 AUTORISÉ")
    else:
        st.error("❌ TEST ÉCHOUÉ — ANGLE MORT DÉTECTÉ — P6 INTERDIT")
