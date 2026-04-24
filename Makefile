SHELL := /bin/bash

# Configuration des ressources
MINIKUBE_CPUS ?= 4
MINIKUBE_MEMORY ?= 8192
KFP_VERSION ?= 2.5.0
KFP_NS ?= kubeflow

# Manifests officiels
KFP_CLUSTER_SCOPED_MANIFEST ?= github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$(KFP_VERSION)
KFP_PLATFORM_AGNOSTIC_MANIFEST ?= github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$(KFP_VERSION)

.PHONY: k8s-start kfp-install kfp-ui k8s-clean k8s-status

# 1. Démarrage du cluster avec provisionnement de stockage
k8s-start:
	minikube start --cpus $(MINIKUBE_CPUS) --memory $(MINIKUBE_MEMORY)
	minikube addons enable default-storageclass
	minikube addons enable storage-provisioner

# 2. Installation déclarative via Kustomize overlay
kfp-install:
	kubectl apply -k "$(KFP_CLUSTER_SCOPED_MANIFEST)"
	kubectl apply -k k8s-manifests/kubeflow-fix/
	kubectl wait --for=condition=Available --timeout=900s deployment --all -n $(KFP_NS)

# 3. Accès à l'interface
kfp-ui:
	kubectl -n $(KFP_NS) port-forward svc/ml-pipeline-ui 8080:80

# 4. Nettoyage
k8s-clean:
	minikube delete

# 5. Diagnostic rapide
k8s-status:
	kubectl get pods -n $(KFP_NS)
	kubectl get pvc -n $(KFP_NS)