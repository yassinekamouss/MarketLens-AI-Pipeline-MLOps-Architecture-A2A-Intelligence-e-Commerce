# 🚀 Smart eCommerce Intelligence

Plateforme **end-to-end d’intelligence e-commerce** qui automatise toute la chaîne de valeur: **scraping multi-plateformes (A2A)** → **enrichissement sémantique LLM sécurisé (MCP)** → **orchestration MLOps sur Kubernetes/Kubeflow** → **restitution BI interactive**.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-326CE5?logo=kubernetes&logoColor=white)
![Kubeflow](https://img.shields.io/badge/Kubeflow-Pipelines-005CFF)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-1C3C3C)
![DeepSeek](https://img.shields.io/badge/DeepSeek-LLM-6A5ACD)
![Flask](https://img.shields.io/badge/Flask-API-000000?logo=flask&logoColor=white)
![MinIO](https://img.shields.io/badge/MinIO-Object%20Storage-C72E49?logo=minio&logoColor=white)

---

## 🎯 Overview — Valeur ajoutée business & technique

Smart eCommerce Intelligence permet de:
- **Industrialiser l’extraction** de catalogues Shopify/WooCommerce via des agents autonomes.
- **Identifier les Top-K produits** à fort potentiel via un scoring hybride supervisé/non-supervisé.
- **Analyser les comportements produits** (segmentation KMeans, projection PCA, règles d’association).
- **Transformer les données en décisions** via un dashboard BI avec KPIs, dataviz et assistant conversationnel.

Le système est conçu pour un usage **MLOps production-ready**: pipeline reproductible, artefacts versionnables, composants découplés et intégration cloud-native.

---

## 🏗️ Architecture Système

![Architecture Globale](docs/architecture.png)

### 1) Couche Scraping A2A
- Agents spécialisés Shopify et WooCommerce.
- Stratégie **API-first** pour la robustesse et la performance.
- **Fallback Playwright** pour les cas dynamiques ou API indisponible.
- Normalisation des produits vers un schéma commun.

### 2) Couche Enrichissement LLM
- Enrichissement sémantique avec **DeepSeek** (catégorisation, résumé court, tags).
- Gestion des erreurs/réessais et cache local d’enrichissement.
- Exposition d’outils de lecture sécurisés via **MCP (Model Context Protocol)**.

### 3) Couche Orchestration MLOps
- Orchestration pipeline avec **Kubeflow Pipelines (KFP)** sur Kubernetes.
- Étapes composables: preprocessing, entraînement, scoring.
- Gestion des entrées/sorties par artefacts, avec stockage objet **MinIO**.

### 4) Couche Restitution BI
- Backend **Flask** pour APIs dashboard et endpoints de chat.
- Dataviz interactive côté frontend (incluant visualisations Plotly/PCA/KMeans).
- Restitution orientée pilotage business (classement, filtres, KPIs).

---

## 🔄 Pipeline MLOps (DAG)

![Pipeline DAG](docs/dag.png)

Le DAG Kubeflow implémente:
1. **Prétraitement**: validation/normalisation des données enrichies et génération du dataset ML-ready.
2. **Entraînement parallèle**:
   - **XGBoost supervisé** pour la probabilité de succès produit.
   - **KMeans non-supervisé** (avec scaler + PCA) pour la segmentation.
3. **Moteur de scoring**: fusion des signaux (proba supervisée, qualité cluster, rating, reviews, stock) pour produire le **Top-K produits** final.

---

## 📊 Interface & Dashboard BI

![Dashboard BI](docs/dashboard.png)

Fonctionnalités clés:
- **KPIs temps réel** (volume produits, score moyen, prix moyen, etc.).
- **Dataviz analytique** (projection PCA, segmentation KMeans, tableaux filtrables).
- **Assistant virtuel intégré** via **LangChain + DeepSeek** pour l’exploration conversationnelle des données.

---

## ⚙️ Prérequis & Installation (DevOps Focus)

### Prérequis
- Python 3.10+
- Docker
- Kubernetes local via Minikube 1.33+ (ou équivalent compatible)
- kubectl
- accès réseau aux APIs cibles et au provider LLM

### Variables d’environnement essentielles

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"
export MCP_TRANSPORT="stdio"            # stdio ou sse
export TOP_PRODUCTS_PATH="data/processed/top_k_products.json"  # optionnel
```

### Initialisation via Makefile

```bash
# 1) Démarrer le cluster Kubernetes local
make k8s-start

# 2) Installer Kubeflow Pipelines + patch namespace
make kfp-install

# 3) Vérifier l’état des ressources
make k8s-status

# 4) Exposer l’UI Kubeflow en local
make kfp-ui

# 5) Nettoyage cluster
make k8s-clean
```

---

## 🗂️ Structure du projet

```text
.
├── data/                  # Jeux de données bruts, enrichis, et outputs de scoring
├── docs/                  # Documentation projet et visuels d’architecture/DAG/dashboard
├── frontend/              # Service Flask (API + rendu templates/static) du dashboard BI
├── k8s-manifests/         # Manifests Kubernetes/Kubeflow (overlays/patches)
├── llm_agents/            # Agents LLM (enrichissement DeepSeek, serveur MCP)
├── ml_models/             # Prétraitement, apprentissage, scoring, règles d’association
├── pipelines/             # Définition DAG Kubeflow + composants KFP
├── scraping/              # Agents A2A Shopify/WooCommerce et schémas communs
├── scripts/               # Scripts d’exécution et utilitaires du projet
├── Dockerfile
├── Makefile
└── requirements.txt
```

---

## 👥 Auteurs & Contributeurs

Architecture et développement initial:
- **Yassine Kamouss**
- **Mohammed Salhi**
