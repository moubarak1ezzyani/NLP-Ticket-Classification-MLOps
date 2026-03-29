# NLP-Ticket-Classification-MLOps
End-to-end MLOps pipeline for automated IT support ticket classification. Features NLP embeddings (Hugging Face), Vector Search (ChromaDB), Drift Monitoring (Evidently AI), and Cloud-Native deployment (Kubernetes, Prometheus, Grafana).


C'est un excellent projet MLOps, très complet (NLP + Vector DB + K8s + Monitoring). Pour ne pas te perdre entre le code Python, les configurations Docker/K8s et les rapports Evidently, une structure rigoureuse est **indispensable**.


### 📂 Arborescence proposée

```text
nlp-ticket-classification/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # Pipeline GitHub Actions (Build & Lint)
│
├── config/                    # Fichiers de configuration
│   └── config.yaml            # Paramètres (paths, hyperparamètres, noms des modèles)
│
├── data/                      # Données (À AJOUTER AU .GITIGNORE !)
│   ├── raw/                   # Dataset initial (csv, json)
│   ├── processed/             # Données nettoyées
│   └── chromadb/              # Persistance locale de ChromaDB
│
├── k8s/                       # Manifests Kubernetes (Étape 6)
│   ├── deployment.yaml        # Déploiement du pipeline
│   ├── cronjob.yaml           # Pour l'exécution périodique
│   └── configmap.yaml         # Variables d'env pour K8s
│
├── monitoring/                # Monitoring Infrastructure (Étape 7)
│   ├── docker-compose.yml     # Stack Prometheus + Grafana + cAdvisor
│   └── prometheus.yml         # Config de Prometheus
│
├── notebooks/                 # Pour l'exploration (Étape 1)
│   ├── 01_eda.ipynb           # Analyse exploratoire
│   └── 02_prototyping.ipynb   # Tests des modèles Hugging Face
│
├── reports/                   # Rapports générés (Étape 5)
│   └── evidently_drift.html   # Rapport de drift (HTML)
│
├── src/                       # Code source Python (Le cœur du projet)
│   ├── __init__.py
│   ├── preprocessing.py       # Nettoyage NLP (SpaCy/NLTK)
│   ├── embeddings.py          # Génération vecteurs + ChromaDB
│   ├── model.py               # Entraînement Scikit-learn
│   ├── monitoring.py          # Script Evidently AI
│   └── main.py                # Pipeline principal qui orchestre tout
│
├── tests/                     # Tests unitaires
│   ├── test_preprocessing.py
│   └── test_model.py
│
├── .gitignore                 # TRÈS IMPORTANT
├── Dockerfile                 # Image Docker du pipeline
├── README.md                  # Documentation du projet
└── requirements.txt           # Dépendances Python

```

---

### 💡 Détails du contenu des dossiers clés

Voici comment remplir ces dossiers en suivant les étapes de ton énoncé :

#### 1. `src/` (Le moteur)

C'est ici que tu vas "industrialiser" le code. Ne mets pas tout dans un seul fichier.

* **`preprocessing.py` (Étape 1) :** Fonctions pour lowercase, stopwords, tokenization.
* **`embeddings.py` (Étape 2) :** Chargement du modèle Hugging Face et insertion dans ChromaDB.
* **`model.py` (Étape 3) :** Split train/test, entraînement Scikit-learn, sauvegarde du modèle (pickle/joblib).
* **`monitoring.py` (Étape 5) :** Code pour générer le rapport Evidently et détecter le drift.
* **`main.py` :** Le script qui appelle les autres modules dans l'ordre. C'est ce script que Docker lancera.

#### 2. `monitoring/` (L'infrastructure)

C'est pour l'**Étape 7**. Tu y mets le `docker-compose.yml` et le `prometheus.yml` fournis dans l'énoncé. Cela te permet de lancer la stack de surveillance (Grafana/Prometheus) indépendamment du pipeline de calcul.

#### 3. `k8s/` (L'orchestration)

C'est pour l'**Étape 6**. Au lieu de lancer ton conteneur Docker à la main, tu vas écrire des fichiers YAML ici pour dire à Kubernetes (Minikube) comment lancer ton job.

#### 4. `.gitignore` (Indispensable)

Tu ne dois **jamais** commiter les données brutes ou les dossiers lourds sur GitHub. Crée un fichier `.gitignore` à la racine contenant :

```text
# Python
__pycache__/
*.pyc

# Data & Models
data/
*.pkl
*.joblib
chromadb/

# Environnement virtuel
.venv/
env/

```

### 🚀 Par quoi commencer ?

Je te conseille de suivre cet ordre pour remplir tes dossiers :

1. **Exploration :** Fais ton code "sale" dans `notebooks/01_eda.ipynb` pour valider que tu arrives à lire les données et nettoyer le texte.
2. **Refactoring :** Copie-colle les fonctions qui marchent du notebook vers `src/preprocessing.py`.
3. **Pipeline :** Crée `src/main.py` pour tester l'enchaînement complet en local.
4. **Docker :** Une fois que `python src/main.py` fonctionne, écris le `Dockerfile`.

Veux-tu que je t'aide à rédiger le fichier `requirements.txt` avec les bonnes bibliothèques pour ce projet ?

***
Branche,Étape du projet,Objectif
`feature/eda-preprocessing`,Étape 1,"Nettoyage des données, analyse et script `preprocessing.py`."
***
`feature/embeddings-vector-db`,Étape 2,Intégration de Hugging Face et configuration de ChromaDB.
***
`feature/classification-model`,Étape 3,Entraînement du modèle Scikit-learn et sauvegarde du .pkl.
***
`feature/ml-monitoring`,Étape 5,Implémentation d'Evidently AI pour le Data Drift.
***
`infra/docker-k8s`,Étape 6,Création du Dockerfile et des manifests Kubernetes.
***
`ops/infra-monitoring`,Étape 7,Configuration Prometheus et Grafana.
