# 📋 CAHIER DES CHARGES — Smart eCommerce Intelligence
## ML & Data Mining Pipelines, A2A Agents, and LLMs

---

## 🏫 Contexte Académique

| Champ         | Détail                          |
|---------------|---------------------------------|
| Établissement | FST Tanger                      |
| Filière       | LSI 2                           |
| Module        | Data Mining & Systèmes d'Information Décisionnels (DM & SID) |
| Année         | 2025/2026                       |

---

## 🎯 Objectif Général du Projet

Développer un **système intelligent et automatisé** capable de :

1. **Scraper** des données produits sur des sites Shopify et WooCommerce
2. **Analyser** les produits et identifier les meilleurs (Top-K)
3. **Orchestrer** un pipeline ML avec Kubeflow
4. **Visualiser** les résultats dans un dashboard Business Intelligence (BI)
5. **Enrichir** l'analyse via des LLMs (résumés, recommandations, synthèses)
6. **Concevoir** une architecture d'agents responsables basée sur le Model Context Protocol (MCP) d'Anthropic

---

## 🧩 Modules du Projet

### Module 1 — Web Scraping Distribué avec Agents A2A
- Extraire des données de plusieurs plateformes e-commerce (Shopify, WooCommerce)
- Utiliser des agents A2A (Agent-to-Agent) autonomes

### Module 2 — Analyse ML + BI
- Identifier les produits à fort potentiel
- Construire un modèle prédictif
- Visualiser les résultats dans un tableau de bord interactif

### Module 3 — Kubeflow Pipelines
- Orchestrer les étapes ML avec Docker, GitHub et CI/CD

### Module 4 — Intelligence Augmentée par LLMs
- Enrichissement des données
- Synthèse automatique
- Génération de recommandations stratégiques

### Module 5 — Architecture Responsable (MCP)
- Concevoir des agents responsables selon le Model Context Protocol d'Anthropic

---

## 🗂️ Structure des Données à Collecter

### 1. Données Descriptives du Produit
| Variable      | Description              |
|---------------|--------------------------|
| product_id    | Identifiant unique        |
| nom           | Nom du produit            |
| description   | Description textuelle     |
| categorie     | Catégorie principale      |
| sous_categorie| Sous-catégorie            |
| marque        | Marque ou vendeur         |
| images        | URL des images            |
| tags          | Mots-clés associés        |

**Utilité DM :** classification, analyse textuelle, segmentation marché

---

### 2. Données de Prix
| Variable      | Description              |
|---------------|--------------------------|
| prix_actuel   | Prix courant              |
| prix_promo    | Prix promotionnel         |
| ancien_prix   | Prix avant promotion      |
| remise_pct    | Pourcentage de remise     |
| devise        | Monnaie (EUR, USD, etc.)  |

**Utilité DM :** analyse stratégie de prix, prédiction ventes, clustering premium vs low-cost

---

### 3. Données de Popularité
| Variable      | Description              |
|---------------|--------------------------|
| rating        | Note moyenne (ex: 4.7)   |
| nb_reviews    | Nombre d'avis             |
| nb_etoiles    | Notation par étoiles      |
| nb_commentaires | Nombre de commentaires  |
| classement    | Rang dans la catégorie    |

**Utilité DM :** variable cible pour classification, scoring produit

---

### 4. Données de Stock et Disponibilité
| Variable         | Description              |
|------------------|--------------------------|
| en_stock         | Boolean disponibilité     |
| quantite_dispo   | Quantité en stock         |
| delai_livraison  | Délai estimé              |
| localisation     | Entrepôt / région         |

**Utilité DM :** gestion stock, corrélation disponibilité/ventes

---

### 5. Données sur les Variantes
| Variable  | Description   |
|-----------|---------------|
| couleur   | Couleur dispo |
| taille    | Taille dispo  |
| modele    | Modèle dispo  |
| version   | Version dispo |

**Utilité DM :** règles d'association, analyse préférences clients

---

### 6. Données sur le Vendeur / Boutique
| Variable         | Description              |
|------------------|--------------------------|
| nom_shop         | Nom de la boutique        |
| pays             | Pays du vendeur           |
| nb_produits      | Nombre de produits vendus |
| anciennete_shop  | Ancienneté de la boutique |

**Utilité DM :** analyse concurrentielle, ciblage géographique

---

### 7. Données Marketing
| Variable               | Description                    |
|------------------------|--------------------------------|
| produits_similaires    | Produits recommandés similaires|
| produits_recommandes   | Recommandations de la plateforme|
| achetes_ensemble       | Fréquemment achetés ensemble   |
| produits_tendance      | Produits en tendance           |

**Utilité DM :** règles d'association (ex: `{smartphone} → {coque}`)

---

### 8. Données Temporelles (scraping régulier)
| Variable          | Description                 |
|-------------------|-----------------------------|
| date_mise_en_ligne| Date de publication         |
| date_promotion    | Début/fin de promo          |
| evolution_prix    | Historique des prix         |
| evolution_rating  | Évolution de la note        |

**Utilité DM :** analyse des tendances produits

---

### 9. Données Textuelles (NLP)
| Variable             | Description             |
|----------------------|-------------------------|
| description_produit  | Texte de description    |
| commentaires_clients | Avis textuels           |
| avis_detailles       | Revues approfondies     |

**Utilité DM :** analyse de sentiment, clustering sémantique

---

### 10. Dataset Final Recommandé
id | produit          | categorie   | prix | rating | reviews | stock | shop       | pays
1  | Wireless Earbuds | Electronics | 59   | 4.6    | 1200    | 35    | TechStore  | USA
2  | Fitness Tracker  | Sport       | 49   | 4.2    | 340     | 10    | FitShop    | UK
3  | LED Desk Lamp    | Home        | 29   | 4.8    | 980     | 80    | BrightHome | CA

---

**Volume recommandé :** 2000 à 5000 produits, 10 à 20 variables

---

## ⚙️ Étapes Techniques Détaillées

---

### Étape 1 — Scraping de Données (Agents A2A)

**Objectif :** Extraire automatiquement les données produits depuis Shopify et WooCommerce.

**Concepts clés :**
- **Agent A2A (Agent-to-Agent)** : composant logiciel autonome qui se connecte à un site, lit ses pages et extrait des données
- **Scraping** : automatisation de lecture de contenu HTML
- **Crawling** : navigation systématique sur plusieurs pages

**Outils recommandés :**
- `requests` + `BeautifulSoup` → scraping statique
- `Selenium` ou `Playwright` → gestion JavaScript / actions dynamiques
- `Scrapy` → projets de scraping structurés
- **Shopify** → Storefront API
- **WooCommerce** → REST API WooCommerce

**Données à extraire :**
- Titre, prix, disponibilité, note moyenne, description, vendeur, catégorie, géographie, trafic

---

### Étape 2 — Analyse et Sélection des Top-K Produits

**Objectif :** Identifier les produits les plus attractifs selon des critères définis.

**Concepts clés :**
- **Top-K Selection** : sélectionner les K meilleurs éléments selon un score composite
- **Scoring** : attribuer un score synthétique à chaque produit
- **Normalisation / Pondération** : combiner plusieurs métriques (note, ventes, prix)
- **Classement des shops** avec leurs produits phare et géographie

**Outils et algorithmes :**
- `pandas`, `numpy` → préparation des données
- `scikit-learn` → clustering, régression
- `xgboost`, `lightgbm` → prédiction du succès potentiel
- **Algorithmes :** Random Forest, KMeans, DBSCAN, règles d'association, PCA

**Variables pour Random Forest / XGBoost :**
- Entrées : prix, rating, nb_reviews, catégorie, vendeur, stock
- Variable cible : `produit_succes` (top produit = 1, sinon = 0)

**Variables pour KMeans / Clustering hiérarchique :**
- Segmentation : produits premium, produits discount, produits populaires

**DBSCAN :**
- Détection d'anomalies : produits atypiques, anomalies de prix

**PCA :**
- Visualisation des produits dans un espace 2D

**Règles d'association :**
- Exemple : `{coque iphone} → {chargeur}`

---

### Étape 3 — Kubeflow Pipelines pour l'Orchestration ML

**Objectif :** Créer un pipeline reproductible pour l'analyse, le scoring et la sélection Top-K.

**Concepts clés :**
- **Pipeline ML** : chaîne d'étapes (prétraitement → entraînement → évaluation → prédiction)
- **Kubeflow Pipelines** : framework pour déployer des pipelines ML sur Kubernetes
- **MLOps** : pratiques DevOps appliquées au cycle de vie des modèles ML

**Outils :**
- `Kubeflow` → orchestration principale
- `kfp SDK` → écriture des pipelines en Python
- `Docker` → conteneurisation des composants
- `Minikube` ou `Kind` → tests locaux avec Kubernetes

---

### Étape 4 — Dashboard Business Intelligence

**Objectif :** Permettre aux décideurs de visualiser les produits sélectionnés et les résultats d'analyse.

**Concepts clés :**
- **KPI** : produits populaires, stock faible, comparaison de prix
- **Dataviz** : visualisation synthétique et interactive
- **Storytelling data** : narration visuelle des tendances

**Outils :**
- `Streamlit` → dashboard interactif en Python (recommandé)
- `Power BI` ou `Metabase` → outil BI professionnel
- `Plotly`, `Seaborn`, `Altair` → librairies de visualisation

---

### Étape 5 — LLM pour Enrichissement et Synthèse

**Objectif :** Enrichir l'analyse en générant des synthèses intelligentes, résumés et recommandations.

**Concepts clés :**
- **LLM** : modèle de langage entraîné sur de vastes corpus
- **Prompt Engineering** : conception de requêtes textuelles pour interagir avec un LLM
- **Chain of Thought** : raisonnement explicite pour justifier les réponses

**Sous-tâches LLM :**

#### 5.1 — Agents Intelligents pilotés par LLM
- Générer automatiquement des prompts de scraping spécifiques selon la plateforme
- Reformuler / nettoyer les données extraites (uniformiser les titres)
- Résumer les descriptions longues en quelques phrases clés
- Créer un "profil client" basé sur les produits les plus consultés

#### 5.2 — Analyse Concurrentielle Augmentée
- Comparer automatiquement les caractéristiques de produits concurrents
- Générer des rapports automatiques : *"Quels sont les 5 produits émergents cette semaine ?"*
- Proposer des stratégies marketing basées sur l'analyse LLM du marché

#### 5.3 — Chatbot Conversationnel (optionnel)
Intégré dans le dashboard BI, capable de répondre à :
- *"Montre-moi les produits les mieux notés sur Shopify cette semaine."*
- *"Quelles sont les promotions concurrentes détectées ?"*

**Outils :**
- `OpenAI API`, `Claude (Anthropic)`, `LLaMA`, `HuggingFace Transformers`
- `LangChain` → orchestration d'appels complexes
- `Gradio` ou `Streamlit Chat` → interface conversationnelle

---

### Étape 6 — Architecture Responsable avec Model Context Protocol (MCP)

**Objectif :** Encadrer les interactions des agents avec les outils et les données de manière responsable et sécurisée.

**Concepts clés du MCP :**
- **MCP (Model Context Protocol)** : protocole standardisé d'Anthropic pour permettre aux LLMs d'interagir avec des outils tout en respectant éthique, contrôle et contextualisation
- **Responsabilité** : un agent déclare ses intentions, ses sources, et respecte les règles d'usage
- **Isolation** : les serveurs MCP n'exposent que le strict nécessaire

**Composants MCP :**
| Composant    | Rôle                                              |
|--------------|---------------------------------------------------|
| MCP Host     | Environnement principal (ex: app Streamlit)        |
| MCP Client   | Composant qui interagit avec les serveurs MCP      |
| MCP Server   | Expose des outils/données spécifiques              |
| Logs + Permissions | Journalisation des requêtes, validation des accès |

**Références :**
- [Anthropic MCP Overview](https://modelcontextprotocol.io)
- [Spécification technique MCP 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26)
- [Dépôt GitHub officiel MCP](https://github.com/modelcontextprotocol)

---

### Étape Transversale — CI/CD

**Objectif :** Automatiser le cycle de développement et déploiement.

**Outils :**
- `GitHub Actions` → automatisation des workflows
- `Docker` → conteneurisation
- `Kubeflow Pipelines` → déploiement ML
- Tests automatisés

---

## 📊 Évaluation et Validation des Modèles

### Approches Supervisées (Random Forest, XGBoost)
- Séparation train/test ou validation croisée (k-fold)
- Métriques : `accuracy`, `précision`, `rappel`, `F1-score`, `matrice de confusion`

### Méthodes Non Supervisées (KMeans, Clustering hiérarchique, DBSCAN)
- `Silhouette Score` → qualité des clusters
- Interprétation visuelle des clusters

### Règles d'Association
- Métriques : `support`, `confidence`, `lift`

### Interprétation
- Résultats discutés du point de vue **business et décisionnel**

---

## 📦 Livrables Attendus

| # | Livrable                                                                 | Obligatoire |
|---|--------------------------------------------------------------------------|-------------|
| 1 | Code des agents A2A de scraping + documentation                          | ✅ Oui      |
| 2 | Pipeline Kubeflow (fichiers YAML ou code Python)                         | ✅ Oui      |
| 3 | Tableau Top-K produits + dashboard BI                                    | ✅ Oui      |
| 4 | Module LLM pour enrichissement et synthèse automatique                   | ✅ Oui      |
| 5 | Rapport d'analyse incluant réflexion sur le Model Context Protocol       | ✅ Oui      |
| 6 | Vidéo de démonstration                                                   | ⬜ Optionnel|

---

## 🛠️ Stack Technologique Complète

| Domaine          | Outils / Technologies                                      |
|------------------|------------------------------------------------------------|
| Scraping         | Selenium, Scrapy, Playwright, requests, BeautifulSoup      |
| APIs             | Shopify Storefront API, WooCommerce REST API               |
| ML / DM          | Scikit-learn, XGBoost, LightGBM, pandas, numpy             |
| Pipelines        | Kubeflow, kfp SDK, Kubernetes, Docker, Minikube / Kind     |
| CI/CD            | GitHub Actions, Docker, Kubeflow Pipelines                 |
| BI / Dataviz     | Streamlit, Power BI, Metabase, Plotly, Seaborn, Altair     |
| LLMs             | OpenAI GPT, Claude (Anthropic), LLaMA2, HuggingFace        |
| Agents LLM       | LangChain, OpenAgents                                      |
| Interfaces       | Gradio, Streamlit Chat                                     |
| Protocole Agents | Model Context Protocol (MCP) — Anthropic                   |

---

## 📐 Architecture Générale du Système

        [Shopify API / WooCommerce API]
            ↓
        [Agents A2A Scraping] ← Selenium / Scrapy / Playwright
            ↓
        [Dataset brut : 2000–5000 produits, 10–20 variables]
            ↓
        [Pipeline Kubeflow]
        ├── Prétraitement (pandas, numpy)
        ├── Scoring & Top-K Selection
        ├── ML Models (RandomForest, XGBoost, KMeans, DBSCAN, PCA)
        └── Règles d'Association
            ↓
        [Module LLM] ← LangChain + Claude/GPT/LLaMA
        ├── Résumés & Synthèses
        ├── Analyse Concurrentielle
        └── Recommandations Stratégiques
            ↓
        [Dashboard BI] ← Streamlit / Power BI
        ├── Tableau Top-K produits
        ├── KPIs & Visualisations
        └── Chatbot conversationnel (optionnel)
            ↓
        [Architecture MCP] ← Anthropic Model Context Protocol
        ├── MCP Host (Streamlit App)
        ├── MCP Client
        ├── MCP Server (outils exposés)
        └── Logs + Permissions

---

---

## 🔗 Ressources et Références

- [Shopify](https://www.shopify.com/)
- [WooCommerce](https://woocommerce.com/)
- [Kubeflow](https://www.kubeflow.org/)
- [LangChain](https://www.langchain.com/)
- [Anthropic Claude](https://www.anthropic.com/)
- [HuggingFace](https://huggingface.co/)
- [Model Context Protocol](https://modelcontextprotocol.io)
- [MCP Spécification 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26)
- [Streamlit](https://streamlit.io/)
- [Scikit-learn](https://scikit-learn.org/)

---

*Cahier des charges généré à partir des documents de projet FST Tanger — LSI2 — DM & SID 2025/2026*