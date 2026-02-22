# 🚀 Real-World ML Model Deployment with KServe on Kubernetes

This repository provides a practical, hands-on guide to transitioning from a "trained model sitting in a notebook" to a **production-grade online inference endpoint** running on a Kubernetes cluster. 

Instead of focusing on theory, this project documents the actual "heavy lifting" involved in MLOps: wiring storage, configuring the KServe control plane, exposing scalable endpoints, and handling live inference requests.

---

## 🏗️ The Architecture

The implementation focuses on the practical path of model serving, moving beyond local testing to a resilient K8s-hosted environment.



### Key Components:
* **Model Serving:** KServe (InferenceService) for high-level abstraction.
* **Orchestration:** Kubernetes (K8s) for container management.
* **Inference Pipeline:** Integration with S3/GCS/PVC for model artifact storage.
* **Networking:** Clean HTTP endpoints ready for live predictions.

---

## 🛠️ What’s Inside this Repository

### 🔹 KServe in the ML Stack
A breakdown of why KServe is the preferred choice for model serving on Kubernetes, focusing on its ability to handle autoscaling (including scale-to-zero) and health checking out of the box.

### 🔹 The High-Level Flow
Documentation on the end-to-end lifecycle:
1. **Storage:** Staging the model in a Cloud Storage bucket or Persistent Volume Claim (PVC).
2. **Manifests:** Configuring the `InferenceService` without getting lost in "YAML hell."
3. **Exposure:** Mapping the service to a reachable HTTP endpoint.

### 🔹 Practical Lessons & Pitfalls
Real-world insights gained from moving from a "local notebook" to "K8s-hosted." This includes:
* Handling authentication between KServe and storage providers.
* Resource limits and request/limit tuning for ML workloads.
* Debugging common `Predictor` and `Transformer` container errors.

---

## 📂 Project Structure

```bash
├── manifests/          # KServe InferenceService & ConfigMaps
├── src/                # Sample model artifacts and preprocessing scripts
├── tests/              # Python scripts for sending live inference requests
└── README.md           # Documentation
```

## 🚀 Getting Started

* **Prerequisites:** A running K8s cluster with KServe, Knative, and Istio installed.
* **Model Setup:** Ensure your model is saved in a supported format (e.g., `.pkl`, `.onnx`, or `SavedModel`).
* **Deploy:**
    ```bash
    kubectl apply -f manifests/inference-service.yaml
    ```
* **Predict:** Use the scripts in `/tests` to send a JSON payload to your new endpoint.

---

## 💡 Why This Project?

As someone who has managed infrastructure handling over **3 million web requests** and **91k Compute Engine units**, I know that the hardest part of AI isn't training the model—it's keeping it alive and scalable in production. This repo is designed for MLOps Engineers, DevOps Professionals, and Architects looking to productionize their AI workflows.

---

## 👨‍💻 About the Author

**Google Cloud Innovator | Solution Architect**

I have spent over 2 years leading high-impact technical communities, specifically as a **Cluster Coordinator for the Microsoft Club**, and architecting cloud solutions for startups. My experience bridges the gap between raw code and scalable, cloud-native infrastructure, with a focus on solving the "scale-up" problems that students and early-stage startups often face.

**Interested in MLOps or Cloud Architecture? Let's connect!**
