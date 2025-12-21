import hashlib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sympy import sympify

# =============================================================================
# CONSTANTES SECRETES (PARTIE 6 - DOC A2)
# =============================================================================
EPSILON = 0.1          # Constante de stabilité
SIGMA_SEUIL = 0.85     # Seuil anti-redondance
ALPHA_DYNAMIQUE = 30   # Paramètre de récence
NB_QC_TARGET = 15      # Invariant Axiomatique

# =============================================================================
# STRUCTURES DE DONNÉES INVARIANTES (DOC A3)
# =============================================================================

@dataclass
class TriggerSignature:
    """
    AXIOME DELTA : Signature (V, O, C)
    """
    verb: str       # V: Action (Calculer, Dériver...)
    obj: str        # O: Objet Math (Fonction, Suite...)
    context: str    # C: Contexte (Physique, Eco...)

    def get_hash(self) -> str:
        """Hachage Atomique Unique"""
        raw = f"{self.verb}|{self.obj}|{self.context}".lower().strip()
        return hashlib.sha256(raw.encode()).hexdigest()

@dataclass
class QuestionCle:
    """
    L'Invariant QC. 
    Contient le Noyau Psi et la Variabilité Sigma.
    """
    id: str
    signature: TriggerSignature
    psi_score: float        # F1: Densité Cognitive
    sigma_class: int        # Difficulté (1-4)
    canonical_text: str
    is_black_swan: bool = False # QC #15
    covered_qi_count: int = 0
    
    def to_dict(self):
        return {
            "ID": self.id,
            "Signature (V|O|C)": f"{self.signature.verb} | {self.signature.obj} | {self.signature.context}",
            "Psi (F1)": round(self.psi_score, 2),
            "Type": "BLACK SWAN" if self.is_black_swan else "STANDARD",
            "Couverture": self.covered_qi_count
        }

# =============================================================================
# MOTEUR GRANULO 15 (LOI DE RÉDUCTION)
# =============================================================================

class GranuloEngine:
    def __init__(self):
        self.qcs: Dict[str, QuestionCle] = {}
        self.raw_atoms = []

    def compute_psi_f1(self, steps_ari: int, delta_c: float = 1.0) -> float:
        """
        FORMULE F1 : Poids Prédictif Purifié
        Psi_q = delta_c * (epsilon + Sum(Tj))
        """
        sum_tj = steps_ari * 0.5 # Simulation pondération étapes
        psi = delta_c * (EPSILON + sum_tj)
        return psi

    def ingest_qi(self, text: str, source_type: str):
        """
        Simule l'extraction (V,O,C) depuis le texte brut.
        """
        # Simulation d'extraction pour le test
        v, o, c = "Calculer", "Inconnu", "Standard"
        
        if "dériv" in text.lower(): v, o = "Dériver", "Fonction"
        if "limit" in text.lower(): v, o = "Calculer", "Limite"
        if "intégr" in text.lower(): v, o = "Calculer", "Intégrale"
        if "suit" in text.lower(): o = "Suite"
        if "probab" in text.lower(): o = "Probabilité"
        
        sig = TriggerSignature(v, o, c)
        
        # Calcul F1 à la volée
        psi = self.compute_psi_f1(steps_ari=3) # Moyenne
        
        atom = {
            "text": text,
            "signature": sig,
            "hash": sig.get_hash(),
            "psi": psi,
            "source": source_type
        }
        self.raw_atoms.append(atom)

    def run_reduction_process(self) -> List[QuestionCle]:
        """
        ALGORITHME DE COMPRESSION (CLUSTERING)
        Réduit N atomes en ~15 QC.
        """
        if not self.raw_atoms:
            return []

        # 1. Regroupement par Hash Atomique (Axiome Delta)
        clusters = {}
        for atom in self.raw_atoms:
            h = atom['hash']
            if h not in clusters:
                clusters[h] = []
            clusters[h].append(atom)

        # 2. Création des QC Invariantes
        qc_list = []
        counter = 1
        
        for h, atoms in clusters.items():
            # Sélection du centroïde (celui avec le meilleur Psi moyen)
            rep_atom = atoms[0] 
            avg_psi = np.mean([a['psi'] for a in atoms])
            
            qc = QuestionCle(
                id=f"QC-{counter:02d}",
                signature=rep_atom['signature'],
                psi_score=avg_psi,
                sigma_class=2, # Default Moyen
                canonical_text=f"QC Canonique pour {rep_atom['signature'].verb} {rep_atom['signature'].obj}",
                covered_qi_count=len(atoms)
            )
            qc_list.append(qc)
            counter += 1

        # 3. Injection QC #15 (Transposition)
        # Bouclier Anti-Black Swan
        qc15 = QuestionCle(
            id="QC-15-META",
            signature=TriggerSignature("Transposer", "Inédit", "Hors-Piste"),
            psi_score=9.99, # Max priorité
            sigma_class=4,  # Expert
            canonical_text="Méta-Protocole : Identifier l'atome Psi caché",
            is_black_swan=True,
            covered_qi_count=0
        )
        qc_list.append(qc15)
        
        # Tri par Psi décroissant (Importance Stratégique)
        qc_list.sort(key=lambda x: x.psi_score, reverse=True)
        
        # Limitation axiomatique (autour de 15)
        return qc_list[:NB_QC_TARGET + 1] # +1 pour la QC#15

    def check_coverage(self, qc_list: List[QuestionCle]) -> dict:
        """
        AUDIT BOOLEEN
        Vérifie si 100% des Qi ont une QC.
        """
        total_qi = len(self.raw_atoms)
        covered_qi = sum(qc.covered_qi_count for qc in qc_list if not qc.is_black_swan)
        
        coverage_rate = (covered_qi / total_qi) * 100 if total_qi > 0 else 0
        
        return {
            "total_qi": total_qi,
            "covered": covered_qi,
            "rate": coverage_rate,
            "is_valid": coverage_rate >= 95.0 # Seuil doc A3
        }
