# Smart eCommerce Intelligence Platform

> **An End-to-End Autonomous eCommerce Intelligence & MLOps Platform**
> From automated multi-platform web scraping to ML-driven behavioral analysis, LLM semantic enrichment, and real-time BI visualization — all orchestrated on Kubernetes.

[![MLOps Pipeline](https://img.shields.io/badge/MLOps-Kubeflow-blue.svg?style=flat-square&logo=kubernetes)](https://kubeflow.org)
[![LLM Backend](https://img.shields.io/badge/LLM-DeepSeek--Chat-orange.svg?style=flat-square)](https://deepseek.com)
[![Architecture](https://img.shields.io/badge/Architecture-Distributed_A2A-green.svg?style=flat-square)]()
[![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Academic](https://img.shields.io/badge/FST_Tanger-LSI_2-red.svg?style=flat-square)](https://www.fstt.ac.ma)

---

## Table of Contents

- [Overview & Business Value](#-overview--business-value)
- [System Architecture](#-system-architecture)
- [MLOps Pipeline (DAG)](#-mlops-pipeline-dag)
- [BI Dashboard](#-bi-dashboard)
- [Prerequisites & Installation](#-prerequisites--installation)
- [Project Structure](#-project-structure)
- [Authors & Contributors](#-authors--contributors)

---

## 🎯 Overview & Business Value

**Smart eCommerce Intelligence** is a production-grade platform that automates the full data lifecycle for e-commerce competitive intelligence: **ingestion → enrichment → modeling → visualization**.

The system addresses a critical business need — continuously monitoring competitor catalogs, pricing strategies, and product performance across multiple platforms (Shopify, WooCommerce) without manual intervention. It delivers:

| Capability | Description |
|---|---|
| **Automated Ingestion** | Zero-touch A2A scraping agents extract structured product data from heterogeneous e-commerce APIs with Playwright fallback for JS-rendered pages. |
| **Semantic Enrichment** | DeepSeek LLM, exposed through a secure MCP server, transforms raw descriptions into ML-ready features (sentiment, category mapping, attribute extraction). |
| **Predictive Scoring** | Parallel XGBoost (supervised) and K-Means/PCA (unsupervised) pipelines rank products, detect behavioral segments, and surface Top-K opportunities. |
| **Actionable BI** | Real-time dashboard with KPI tracking, cluster visualization, and a LangChain-powered conversational assistant for natural-language data queries. |

---

## 🏗️ System Architecture

The platform is organized into four decoupled layers, each independently deployable and observable:

![Architecture Globale](docs/architecture.png)

### 1. A2A Scraping Layer (`/scraping`)

Autonomous Agent-to-Agent crawlers target Shopify and WooCommerce storefronts:

- **`shopify_agent.py`** — Connects via Storefront API / HTML scraping for product catalogs.
- **`woocommerce_agent.py`** — Leverages WooCommerce REST API with structured pagination.
- **Fallback Engine** — Playwright handles JavaScript-rendered pages when static parsing fails.
- **Schema Validation** — Pydantic models enforce strict data contracts (`schemas.py`).

### 2. LLM Enrichment Layer (`/llm_agents`)

DeepSeek-powered semantic processing behind a secure MCP boundary:

- **`mcp_server.py`** — Model Context Protocol server exposing controlled tools (enrichment, summarization, classification) to the LLM.
- **`enricher.py`** — LangChain-driven pipeline that transforms raw product text into structured features.
- **`schemas.py`** — Typed request/response contracts for all MCP tool interactions.
- **Isolation Principle** — Enforces the Responsible Agent pattern by minimizing the exposed surface area.

### 3. MLOps Orchestration Layer (`/pipelines` + `/ml_models`)

Kubernetes-native ML lifecycle management:

- **Kubeflow Pipelines** — Declarative DAG execution with robust artifact tracking.
- **Kustomize Overlays** — Infrastructure-as-code for cluster-scoped and platform-agnostic manifests.
- **MinIO** — Object storage for model artifacts, datasets, and pipeline outputs.
- **Docker** — Containerized components ensuring highly reproducible environments.

### 4. BI & Presentation Layer (`/frontend`)

Flask backend serving an interactive analytics interface built with custom CSS and native JS:

- **Plotly Visualizations** — Interactive PCA projections, K-Means cluster maps, and pricing distributions.
- **Real-Time KPIs** — Scraping throughput, model accuracy (F1-score, Silhouette), and Top-K product metrics.
- **LangChain Chat Assistant** — Natural-language querying over the enriched dataset via DeepSeek AI.

---

## ⚙️ MLOps Pipeline (DAG)

The core ML workflow executes as a Directed Acyclic Graph on Kubeflow Pipelines:

![Pipeline DAG](docs/dag.png)

| Stage | Component | Description |
|---|---|---|
| **Preprocessing** | `preprocessing.py` | Data cleaning, missing-value imputation (seeded randomness), and semantic feature engineering. |
| **Supervised Training** | `supervised.py` | XGBoost classifier predicting Top-K probability from pricing, rating, review, and stock features. |
| **Unsupervised Training** | `unsupervised.py` | K-Means clustering for product segmentation + PCA for dimensionality reduction and visualization. |
| **Association Rules** | `association_rules.py` | Apriori/FP-Growth algorithm mining frequent itemsets with support, confidence, and lift metrics. |
| **Scoring Engine** | `scoring.py` | Composite scoring unifying model outputs to produce the final Top-K recommendation table. |

---

## 📊 BI Dashboard

The interactive analytics interface provides operational visibility across the entire pipeline:

![Dashboard BI](docs/dashboard.png)

- **KPI Cards** — Real-time metrics overview.
- **PCA Projection** — 2D scatter plot of products in reduced feature space.
- **K-Means Clusters** — Distribution of product segments (Premium, Discount, Atypical).
- **Association Rules Explorer** — Interactive table of mined purchasing rules.
- **Top-K Rankings** — Sorted product table with composite scores.
- **AI Virtual Assistant** — Conversational interface powered by DeepSeek.

---

## 🚀 Prerequisites & Installation (DevOps Bootstrap)

### System Requirements

| Dependency | Minimum Version |
|---|---|
| Python | 3.10+ |
| kubectl | 1.28+ |
| Minikube | 1.32+ |
| Docker | 24.0+ |
| kfp SDK | 2.5.0 |

### Environment Variables

Create a `.env` file at the project root with the following required keys:

```bash
# DeepSeek API Key (Required for semantic enrichment)
DEEPSEEK_API_KEY="your_deepseek_api_key_here"

# MinIO Object Storage Credentials (Optional, if using external storage)
MINIO_ACCESS_KEY="minioadmin"
MINIO_SECRET_KEY="minioadmin"
MINIO_ENDPOINT="localhost:9000"
```

### Quick Start (Makefile)

The ̀`Makefile` provides a unified interface for the entire DevOps lifecycle:

```bash
# 1. Provision a local Kubernetes cluster (Minikube)
make k8s-start

# 2. Deploy Kubeflow Pipelines with Kustomize overlays
make kfp-install

# 3. Access the Kubeflow Pipelines UI (localhost:8080)
make kfp-ui

# 4. Check cluster health and pod status
make k8s-status

# 5. Tear down the cluster
make k8s-clean

```

### Manual Pipeline Execution

```bash
# Activate virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -U langchain-deepseek

# 1. Execute A2A Batch Scraping
PYTHONPATH=. python scraping/main.py

# 2. Execute High-Speed Async LLM Enrichment
PYTHONPATH=. python llm_agents/main.py

# 3. Submit Kubeflow Pipeline
PYTHONPATH=. python pipelines/submit_pipeline.py

```

#### Launch BI Dashboard

```bash
# Start the Flask web application locally
PYTHONPATH=. python frontend/app.py
```

Open your browser to `http://127.0.0.1:5000`.

---

# 📁 Project Structure
```Plaintext
.
├── data/
│   ├── raw/             # Raw JSON data (sample_products.json)
│   └── processed/       # Enriched data & ML-ready CSVs
├── docs/                # Architecture diagrams & screenshots
├── frontend/            # Native Flask BI Application
│   ├── static/          # Custom CSS and JavaScript
│   ├── templates/       # HTML Jinja2 views
│   └── app.py           # Flask server & REST API
├── k8s-manifests/       # Kubernetes & Kustomize manifests
├── llm_agents/          # AI Layer (DeepSeek & LangChain)
│   ├── enricher.py      # Structured data enrichment
│   ├── main.py          # High-speed batch execution
│   └── mcp_server.py    # FastMCP server implementation
├── ml_models/           # Data Science Pipeline
│   ├── preprocessing.py # Feature engineering & data imputation
│   ├── supervised.py    # Classification models
│   ├── unsupervised.py  # Clustering & PCA models
│   ├── association_rules.py # Apriori rule mining
│   └── scoring.py       # Final scoring logic
├── pipelines/           # MLOps Orchestration
│   ├── kfp_components.py# Kubeflow component definitions
│   └── main_pipeline.py # DAG assembly
├── scraping/            # Data Ingestion Layer
│   ├── shopify_agent.py
│   └── woocommerce_agent.py
├── Dockerfile           # Container definition
├── Makefile             # Automation & deployment scripts
└── requirements.txt     # Python dependencies
```

---
# 👨‍💻 Authors & Contributors

This platform was architected and engineered by:

- Yassine Kamouss — Cloud Architecture, MLOps Engineering, Kubernetes & Kubeflow

- Yahya Ahmane — AI Integration, LLM & Agent Systems

---
Smart eCommerce Intelligence — FST Tanger, LSI 2, Modules: Data Mining & SID, 2025/2026