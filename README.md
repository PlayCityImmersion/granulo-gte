# GRANULO-GTE  
**Granulo Test Engine — Auditeur Mathématique Impitoyable**

---

## 1. Objet du dépôt

`granulo-gte` n’est **pas** une application utilisateur.  
Ce dépôt implémente un **banc de test industriel** destiné à **auditer** le moteur **Granulo 15 (P6)** de SMAXIA.

Sa fonction est unique et non négociable :

> **Vérifier formellement que toute Question individuelle (Qi) injectée converge vers au moins une Question Clé (QC).**

- OUI → moteur valide → scellement  
- NON → moteur faux → rejet  

Aucune interprétation humaine.  
Aucun storytelling.  
Uniquement des preuves observables.

---

## 2. Principe fondamental

Le moteur applique une **réduction axiomatique** :

- Entrée : un ensemble potentiellement infini de **Qi**
- Sortie : un ensemble fini de **15 QC invariantes par chapitre**

Chaque QC est définie par une signature logique :

