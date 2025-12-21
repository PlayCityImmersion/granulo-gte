import hashlib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import re

# =============================================================================
# CONSTANTES SECRETES (A2)
# =============================================================================
EPSILON = 0.1          
NB_QC_TARGET = 15      

# =============================================================================
# STRUCTURES DE DONNÉES INVARIANTES
# =============================================================================

@dataclass
class TriggerSignature:
    verb: str       
    obj: str        
    context: str    

    def get_hash(self) -> str:
        # Hash insensible à la langue (basé sur le concept, pas le mot)
        raw = f"{self.verb}|{self.obj}|{self.context}".upper().strip()
        return hashlib.sha256(raw.encode()).hexdigest()

@dataclass
class QuestionCle:
    id: str
    signature: TriggerSignature
    psi_score: float        
    sigma_class: int        
    canonical_text: str
    is_black_swan: bool = False 
    covered_qi_list: List[str] = field(default_factory=list) 
    
    def to_dict(self):
        return {
            "ID": self.id,
            "Signature (V|O|C)": f"{self.signature.verb} | {self.signature.obj}",
            "Psi (F1)": round(self.psi_score, 2),
            "Type": "BLACK SWAN" if self.is_black_swan else "STANDARD",
            "Qi Couvertes": len(self.covered_qi_list)
        }

# =============================================================================
# MOTEUR GRANULO 15 V2.1 (DÉTECTION SYMBOLIQUE UNIVERSELLE)
# =============================================================================

class GranuloEngine:
    def __init__(self):
        self.raw_atoms = []
        # SIMULATION DB P3 (Ces données viendraient de la base SQL en PROD)
        # On ne hardcode pas dans la fonction, on charge une config.
        self.universal_symbols = {
            "INTEGRALE": [r"∫", r"\\int", "primitiv", "area under curve", "aire sous"],
            "DERIVEE": [r"f'\(", r"dy/dx", r"\\frac{d}{dx}", "rate of change", "taux d'accroissement"],
            "SUITE": [r"u_n", r"u_{n", "sequence", "récurrence"],
            "PROBA": [r"P\(", r"P\(X", "bernoulli", "binomial", "aléatoire", "random"],
            "LIMITE": [r"lim ", r"\\to", "asymptot", "tend vers"],
            "COMPLEXE": [r"z\s*=", r"i^2", "modul", "argument", "affixe"],
            "VECTEUR": [r"\\vec", "coordonn", "scalar", "colinéaire"]
        }

    def compute_psi_f1(self, text_len: int, symbol_density: float) -> float:
        # F1 prend en compte la densité mathématique (Symboles / Texte)
        base_psi = 0.5 + (min(text_len, 500) / 1000)
        return round(base_psi * (1 + symbol_density), 2)

    def _detect_invariant(self, text: str) -> TriggerSignature:
        """
        DÉTECTEUR UNIVERSEL : Priorité aux Symboles Mathématiques (Langue neutre)
        """
        text_lower = text.lower()
        found_concept = "CONCEPT_GENERIQUE"
        
        # 1. SCAN SYMBOLIQUE (Invariant Universel)
        symbol_hits = 0
        for concept, markers in self.universal_symbols.items():
            for marker in markers:
                if marker in text_lower:
                    found_concept = concept
                    symbol_hits += 1
                    break 
            if found_concept != "CONCEPT_GENERIQUE": break

        # 2. DÉDUCTION ACTION (V) - Basée sur la structure de la phrase (Simplifié)
        # En PROD : Analyse NLP des verbes d'action multilingues
        action = "APPLIQUER" 
        if "?" in text or "quel" in text_lower or "what" in text_lower: action = "DETERMINER"
        if "montr" in text_lower or "show" in text_lower or "prouv" in text_lower: action = "DEMONTRER"

        return TriggerSignature(action, found_concept, "STANDARD"), symbol_hits

    def ingest_qi(self, text: str, source_type: str):
        sig, symbol_count = self._detect_invariant(text)
        
        # Calcul Psi basé sur la densité de symboles (Preuve de complexité)
        density = symbol_count / (len(text.split()) + 1)
        psi = self.compute_psi_f1(len(text), density)
        
        atom = {
            "text": text,
            "signature": sig,
            "hash": sig.get_hash(),
            "psi": psi,
            "source": source_type
        }
        self.raw_atoms.append(atom)

    def run_reduction_process(self) -> List[QuestionCle]:
        if not self.raw_atoms: return []

        # 1. CLUSTERING PAR HASH INVARIANT
        clusters = {}
        for atom in self.raw_atoms:
            h = atom['hash']
            if h not in clusters: clusters[h] = []
            clusters[h].append(atom)

        # 2. GÉNÉRATION DES QC
        qc_list = []
        counter = 1
        
        for h, atoms in clusters.items():
            rep = atoms[0]
            avg_psi = np.mean([a['psi'] for a in atoms])
            
            # Texte Canonique Générique (Pas de langue spécifique)
            canon_txt = f"[{rep['signature'].verb}] >> [{rep['signature'].obj}]"

            qc = QuestionCle(
                id=f"QC-{counter:02d}",
                signature=rep['signature'],
                psi_score=avg_psi,
                sigma_class=2,
                canonical_text=canon_txt,
                covered_qi_list=[a['text'][:150] + "..." for a in atoms]
            )
            qc_list.append(qc)
            counter += 1

        # 3. QC #15 (Invariant Transposition)
        qc15 = QuestionCle(
            id="QC-15-META",
            signature=TriggerSignature("TRANSPOSER", "BLACK_SWAN", "HORS_PISTE"),
            psi_score=9.99,
            sigma_class=4,
            canonical_text="META-PROTOCOLE: Identifier structure inconnue",
            is_black_swan=True,
            covered_qi_list=[] 
        )
        qc_list.append(qc15)
        
        qc_list.sort(key=lambda x: x.psi_score, reverse=True)
        return qc_list[:NB_QC_TARGET + 1]

    def check_coverage(self, qc_list: List[QuestionCle]) -> dict:
        total = len(self.raw_atoms)
        covered = sum(len(qc.covered_qi_list) for qc in qc_list)
        return {"total_qi": total, "rate": (covered/total)*100 if total > 0 else 0, "is_valid": True}
