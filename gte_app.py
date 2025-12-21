import streamlit as st
import pandas as pd
import pdfplumber
import re
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMAXIA - Moteur Méthodologique V3")

# --- 1. MOTEUR D'EXTRACTION (PDF) ---
def extract_granules_from_pdf(uploaded_file):
    text_content = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text_content += extracted + "\n"
    
    # Nettoyage basique
    text_content = text_content.replace('\n', ' ')
    
    # DÉCOUPAGE INTELLIGENT PAR INSTRUCTION
    # On cherche les phrases qui contiennent des verbes d'action clés
    # Cela permet d'isoler les "Atomes de savoir"
    sentences = re.split(r'[.;?!]', text_content)
    
    granules = []
    for sent in sentences:
        sent = sent.strip()
        # On ne garde que les phrases qui ont du sens (longueur > 20 chars)
        if len(sent) > 20:
            granules.append(sent)
                
    return granules

# --- 2. CALCULATEUR DÉTERMINISTE (PSI & SCORE) ---
def calculate_metrics(text):
    stopwords = set(['le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'ou', 'est', 'sont', 'a', 'par', 'pour', 'dans', 'sur'])
    words = [w.lower() for w in re.findall(r'\w+', text)]
    meaningful_words = [w for w in words if w not in stopwords and len(w) > 2]
    
    if len(words) == 0: return 0, 0
    
    # F1 : PSI (Densité d'information)
    psi = len(set(meaningful_words)) / len(words)
    
    # F2 : Score de Pertinence SMAXIA
    # Bonus si la phrase contient des mots clés académiques
    keywords = ['intégrale', 'fonction', 'dérivée', 'suite', 'probabilité', 'guerre', 'traité', 'atome', 'molécule']
    bonus = sum(1 for w in meaningful_words if w in keywords)
    score = (psi * 10) + (bonus * 0.5)
    
    return round(psi, 3), round(score, 2)

# --- 3. TRANSFORMATION EN QC SMAXIA ("COMMENT...") ---
def transform_to_smaxia_qc(raw_text):
    # Dictionnaire de transformation : Verbe Impératif -> Formule Méthodologique
    transformations = {
        r"calculer": "COMMENT CALCULER",
        r"déterminer": "COMMENT DÉTERMINER",
        r"démontrer": "COMMENT DÉMONTRER",
        r"montrer": "COMMENT MONTRER",
        r"justifier": "COMMENT JUSTIFIER",
        r"analyser": "COMMENT ANALYSER",
        r"résoudre": "COMMENT RÉSOUDRE",
        r"tracer": "COMMENT TRACER",
        r"étudier": "COMMENT ÉTUDIER"
    }
    
    detected_qc = None
    trigger_verb = None
    
    text_lower = raw_text.lower()
    
    for pattern, prefix in transformations.items():
        if pattern in text_lower:
            trigger_verb = pattern
            # On nettoie le texte pour l'affichage propre
            # Ex: "Calculer l'intégrale..." -> "l'intégrale..."
            # On cherche la position du verbe et on prend la suite
            match = re.search(pattern, text_lower)
            if match:
                start_idx = match.start()
                end_idx = match.end()
                # Le corps du concept est ce qui suit le verbe
                concept_body = raw_text[end_idx:].strip()
                # Nettoyage final (supprimer points finaux etc)
                concept_body = concept_body.rstrip('.;?!')
                
                detected_qc = f"{prefix} {concept_body}"
                break
    
    return detected_qc, trigger_verb

# --- 4. EXÉCUTION DU MOTEUR ---
def process_granulo_engine(granules):
    results = []
    
    for segment in granules:
        psi, score = calculate_metrics(segment)
        
        # TRANSFORMATION EN QC
        smaxia_qc, trigger = transform_to_smaxia_qc(segment)
        
        # LOGIQUE DE VALIDATION (P6)
        # 1. Il faut avoir détecté un verbe d'action SMAXIA
        # 2. Le score F2 doit être suffisant (> 2.0)
        # 3. Le texte ne doit pas être trop court (bruit)
        
        verdict = "FAIL"
        motif = "Hors Périmètre"
        
        if smaxia_qc:
            if score > 1.5: # Seuil de qualité SMAXIA
                verdict = "PASS"
                motif = ""
            else:
                motif = "Contenu Pauvre (F2 faible)"
        else:
            motif = "Pas de Verbe Action (Non Méthodologique)"

        # On n'ajoute au tableau que ce qui est potentiellement pertinent (ou les gros rejets pour audit)
        if verdict == "PASS" or (verdict == "FAIL" and score > 1.0):
            results.append({
                "ID_Qi": hash(segment) % 10000, # ID court pour lecture
                "QC_SMAXIA (Format Cible)": smaxia_qc if smaxia_qc else "---",
                "Source (Qi Brut)": segment[:80] + "...",
                "Déclencheur": trigger.upper() if trigger else "AUCUN",
                "F1 (Densité)": psi,
                "F2 (Pertinence)": score,
                "VERDICT": verdict
            })
        
    return pd.DataFrame(results)

# --- INTERFACE ---
st.title("🛡️ SMAXIA - Moteur Granulo V3 (Strict 'COMMENT')")
st.markdown("""
**Règle d'Or :** Une QC SMAXIA ne demande pas de *faire*, elle explique *comment faire*.
toute QC doit commencer par **COMMENT**.
""")

uploaded_files = st.file_uploader("Injecter PDF (Sujets Examens)", type=['pdf'], accept_multiple_files=True)

if uploaded_files:
    all_raw_data = []
    
    with st.status("Analyse Sémantique et Structuration...", expanded=True):
        for file in uploaded_files:
            st.write(f"Lecture de {file.name}...")
            raw_segments = extract_granules_from_pdf(file)
            all_raw_data.extend(raw_segments)
            
    if all_raw_data:
        df_result = process_granulo_engine(all_raw_data)
        
        if not df_result.empty:
            st.divider()
            
            # FILTRE D'AFFICHAGE : On montre d'abord les PASS (les QC Valides)
            st.subheader("✅ QC SMAXIA GÉNÉRÉES (Prêtes pour P6)")
            df_pass = df_result[df_result['VERDICT'] == "PASS"]
            
            if not df_pass.empty:
                st.dataframe(
                    df_pass[['QC_SMAXIA (Format Cible)', 'F2 (Pertinence)', 'Déclencheur', 'Source (Qi Brut)']],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("Aucune QC valide trouvée. Vérifiez que le PDF contient des questions type 'Calculer', 'Démontrer'...")

            # ZONE DE REJET (POUR AUDIT)
            with st.expander("Voir les éléments rejetés (Hors structure 'COMMENT')"):
                st.dataframe(df_result[df_result['VERDICT'] == "FAIL"])
                
        else:
            st.error("Le PDF semble contenir du texte, mais aucune structure méthodologique n'a été détectée.")
    else:
        st.warning("PDF vide ou illisible.")
