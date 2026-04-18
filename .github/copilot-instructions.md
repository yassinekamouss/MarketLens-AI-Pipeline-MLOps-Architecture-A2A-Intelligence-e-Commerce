# Project Context: Smart eCommerce Intelligence (Data Product)

## 🎯 Global Objective
We are building an industrial-grade, automated Data Product for competitive eCommerce intelligence. This is NOT an academic script; it is a production-ready microservices architecture. The system scrapes Shopify/WooCommerce products , runs ML pipelines to identify Top-K products , orchestrates the workflow via Kubeflow on Kubernetes , and uses LLMs with the Model Context Protocol (MCP) for augmented intelligence and safe tool use.

## 🏗️ Architecture & Domains
The project is strictly modular. Respect the separation of concerns:
1. **`scraping/` (A2A Agents)**: Distributed web scraping using Playwright/Scrapy/BeautifulSoup. Must handle rate limits, retries, and dynamic content.
2. **`ml_models/` (Data Mining)**: Supervised (XGBoost/RandomForest) and unsupervised (KMeans/DBSCAN) models. Code must be modular, allowing independent training and evaluation.
3. **`pipelines/` (MLOps)**: Kubeflow Pipelines (kfp) orchestrating Dockerized ML components.
4. **`frontend/` (BI & UI)**: Streamlit dashboard for KPI visualization (Plotly) and chatbot integration. Clean, responsive front-end practices apply.
5. **`llm_agents/` (AI & MCP)**: LangChain agents orchestrating OpenAI/Anthropic/LLaMA models. Must implement an Anthropic MCP Server for responsible and isolated data access.

## 💻 Coding Standards & Rules for the AI Agent
- **Production First**: Always write robust code. Include error handling (`try/except`), structured logging (no basic `print` statements for production logic), and edge-case management.
- **Typing & Docstrings**: Enforce strict Python type hints (`->`, `Optional`, `Dict`, etc.) and write PEP-257 compliant docstrings for every function and class.
- **Kubernetes/Cloud Native Mindset**: Assume all code will run in isolated Docker containers inside a Kubernetes cluster (Minikube initially). File paths must be dynamic or rely on environment variables.
- **Security**: Never hardcode API keys. Always use `.env` files and `os.environ`. Ensure MCP implementations follow strict isolation principles.
- **Language**: Comments and commit messages should be in English.

## 🚀 Current Phase
Always check the latest commits to understand the current phase. We follow a strict CI/CD and MLOps lifecycle.