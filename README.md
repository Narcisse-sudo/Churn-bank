#  Churn Bank – Analyse de l’attrition client

Ce projet a pour but d’**analyser les comportements clients** dans le secteur bancaire afin de prédire si un client est susceptible de quitter la banque (`Exited = 1`). Les données sont fictives, générées à des fins pédagogiques, et partagées sur Kaggle par un utilisateur nommé Mathchi.

---

##  Description du dataset

Le dataset contient **14 colonnes** décrivant les caractéristiques démographiques et financières de 143.579 clients.

| Colonne           | Description |
|-------------------|-------------|
| `ID`              | Identifiant interne (souvent redondant) |
| `CustomerId`      | Identifiant unique du client |
| `Surname`         | Nom de famille du client |
| `CreditScore`     | Score de crédit |
| `Geography`       | Pays du client (France, Allemagne, Espagne) |
| `Gender`          | Sexe (Male/Female) |
| `Age`             | Âge du client |
| `Tenure`          | Nombre d’années en tant que client |
| `Balance`         | Solde du compte bancaire |
| `NumOfProducts`   | Nombre de produits souscrits (épargne, carte, prêt, etc.) |
| `HasCrCard`       | Possède une carte de crédit (1 = Oui, 0 = Non) |
| `IsActiveMember`  | Est considéré comme client actif (1 = Oui, 0 = Non) |
| `EstimatedSalary` | Salaire estimé |
| `Exited`          | Cible : a quitté la banque (1) ou non (0) |

---

##  Objectifs

- Effectuer une **analyse exploratoire des données (EDA)**
- Prétraiter les données : gestion des identifiants, encodage, mise à l’échelle
- Construire plusieurs modèles de classification :
  - Régression logistique
  - Arbres de décision / Forêts aléatoires
  - XGBoost
  - (optionnel) Réseau de neurones
- Comparer les performances : accuracy, F1, ROC-AUC
- Identifier les **variables les plus influentes sur le départ d’un client**

---

##  Outils utilisés

- `Python (>= 3.13)`
- `Jupyter Notebook`
- `pandas`, `numpy`, `scikit-learn`
- `matplotlib`, `seaborn`
- `xgboost`
- (optionnel) `tensorflow` ou `pytorch`

---

## Installation des dépendances
Clonez ce dépôt puis installez les dépendances à l’aide du fichier `requirements.txt` :

1. Cloner le dépôt

```bash
git clone https://github.com/Narcisse-sudo/Churn-bank.git
cd Churn-bank
```

2. Installer les dépendances
_Installer Poetry (si pas déjà installé)_
_Voir la documentation officielle : https://python-poetry.org/docs/#installation_

```bash
poetry install
```
3. Activer l’environnement virtuel Poetry 
```bash
poetry shell
```


