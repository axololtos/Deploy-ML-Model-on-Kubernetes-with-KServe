🚀 Real-World ML Model Deployment with KServe on Kubernetes
This repository contains the manifests, configuration logic, and documentation for transitioning a trained machine learning model from a local environment to a production-grade online inference endpoint using KServe on Kubernetes.

Instead of a theoretical overview, this project focuses on the practical architecture: wiring cloud storage, configuring the KServe control plane, exposing scalable endpoints, and handling live inference requests.

🏗️ Architecture & Stack
The deployment follows a standard MLOps workflow to ensure the model is scalable, resilient, and easily accessible via HTTP:

Orchestration: Kubernetes (K8s)

Model Serving: KServe (InferenceService)

Inference Service: Predictor (Triton/SKLearn/Custom)

Storage: S3-compatible storage / PVC / Google Cloud Storage

Networking: Istio / Knative for ingress and autoscaling

🛠️ Key Features & Implementation
This project walks through the "heavy lifting" required to productionize ML models:

Seamless ML Integration: How KServe fits into the modern ML stack to simplify model lifecycle management.

The Deployment Pipeline: A high-level flow covering model registration in storage, defining the InferenceService manifest, and generating a clean HTTP endpoint.

Infrastructure as Code: Minimalist YAML manifests designed to get you from "local notebook" to "K8s-hosted" without the overhead.

Production Lessons: Insights into common pitfalls when moving to Kubernetes, including storage authentication and resource limits.

📂 Repository Structure
Bash
├── manifests/          # KServe InferenceService YAML files
├── scripts/            # Python scripts for testing inference endpoints
├── docs/               # Detailed write-up and architectural diagrams
└── requirements.txt    # Local dependencies for testing
🚀 Getting Started
Configure Storage: Ensure your trained model artifacts are uploaded to your chosen provider (S3/GCS/PVC).

Apply Manifests: ```bash
kubectl apply -f manifests/inference-service.yaml

Test Endpoint: Use the provided Python client to send a sample payload and receive predictions.

💡 Lessons Learned
Moving from a local environment to a Kubernetes cluster introduces unique challenges in Solution Architecture. Key takeaways included in this repo cover:

Configuring service accounts for secure storage access.

Managing cold-start latency with Knative.

Debugging container logs within the KServe pods.
