# 🧠 MLOps Sentiment Analysis Project

[![CI Pipeline](https://github.com/DeepuML/Ml-OPS-Project-2/actions/workflows/ci.yaml/badge.svg)](https://github.com/DeepuML/Ml-OPS-Project-2/actions/workflows/ci.yaml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![DVC](https://img.shields.io/badge/DVC-data%20versioning-945DD6?logo=dvc)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-experiment%20tracking-0194E2?logo=mlflow)](https://mlflow.org/)
[![DagsHub](https://img.shields.io/badge/DagsHub-model%20registry-FF6B35)](https://dagshub.com/DeepuML/Ml-OPS-Project-2)
[![Docker](https://img.shields.io/badge/Docker-containerized-2496ED?logo=docker)](https://hub.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An end-to-end MLOps pipeline for tweet sentiment analysis (happiness vs. sadness), featuring automated data versioning, experiment tracking, model registry, CI/CD, and a production-ready Flask web application.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [ML Pipeline](#ml-pipeline)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
- [Running the Pipeline](#running-the-pipeline)
- [Flask Web Application](#flask-web-application)
  - [Local Development](#local-development)
  - [API Reference](#api-reference)
- [Docker Deployment](#docker-deployment)
- [Experiment Tracking with MLflow & DagsHub](#experiment-tracking-with-mlflow--dagshub)
- [CI/CD Pipeline](#cicd-pipeline)
- [Model Promotion Criteria](#model-promotion-criteria)
- [Testing](#testing)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements a complete **MLOps lifecycle** for a Natural Language Processing (NLP) task — binary sentiment classification of tweets into **happiness** (positive) or **sadness** (negative).

The project goes beyond a standard machine learning project by incorporating industry-grade MLOps practices:

- **Reproducible pipelines** with [DVC](https://dvc.org/) for data and model versioning
- **Experiment tracking** with [MLflow](https://mlflow.org/) hosted on [DagsHub](https://dagshub.com/)
- **Automated model promotion** from staging to production based on quality thresholds
- **Containerized deployment** with Docker and [Gunicorn](https://gunicorn.org/)
- **Automated CI/CD** with GitHub Actions — including pipeline execution, model testing, and Docker build/push

---

## Features

| Feature | Description |
|---|---|
| 🔄 **Reproducible Pipeline** | DVC-managed stages from raw data to deployed model |
| 📊 **Experiment Tracking** | MLflow integrated with DagsHub for metrics, params, and artifact logging |
| 🏷️ **Model Registry** | Versioned models stored in MLflow Model Registry with staging/production tags |
| 🚀 **Automated Promotion** | Model automatically promoted to production when accuracy ≥ 75%, AUC ≥ 75% |
| 🌐 **REST API** | Flask app exposes a `/api/predict` JSON endpoint for programmatic access |
| 🖥️ **Web Interface** | User-friendly HTML frontend for real-time sentiment prediction |
| 🐳 **Dockerized** | Multi-arch Docker image (amd64 + arm64) with health check and Gunicorn |
| ✅ **CI/CD** | GitHub Actions automates testing, DVC pipeline, Docker build, and deployment |
| 🧪 **Automated Tests** | Unit and integration tests for model logic and Flask endpoints |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MLOps Architecture                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Raw Data (CSV)                                                             │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐    │
│  │  Data Ingestion  │───▶│ Data Preprocessing│───▶│ Feature Engineering│   │
│  │ (train/test split│    │(NLTK, lemmatize, │    │  (Bag of Words /   │   │
│  │  25% test size)  │    │ stop-words, URLs) │    │  CountVectorizer)  │   │
│  └─────────────────┘    └──────────────────┘    └────────────────────┘   │
│                                                           │                 │
│                                                           ▼                 │
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐    │
│  │  Model Promotion │◀───│ Model Evaluation │◀───│   Model Building   │   │
│  │ (auto-promote if │    │(accuracy, prec., │    │ (Logistic Regression│  │
│  │  criteria met)   │    │ recall, AUC logged│   │  C=1, liblinear)   │   │
│  └─────────────────┘    │  to MLflow)       │    └────────────────────┘   │
│           │              └──────────────────┘                              │
│           ▼                                                                 │
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐    │
│  │  Model Registry  │───▶│   Flask App      │───▶│  Docker Container  │   │
│  │  (DagsHub /      │    │ (Web UI + REST   │    │  (Gunicorn, prod)  │   │
│  │   MLflow)        │    │  API)            │    │                    │   │
│  └─────────────────┘    └──────────────────┘    └────────────────────┘   │
│                                                                             │
│  All stages versioned with DVC · Experiments tracked with MLflow           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Ml-OPS-Project-2/
├── .github/
│   └── workflows/
│       └── ci.yaml                   # GitHub Actions CI/CD pipeline
│
├── .dvc/                             # DVC configuration and cache metadata
├── dvc.yaml                          # DVC pipeline stage definitions
├── dvc.lock                          # DVC lock file (reproducibility snapshot)
├── params.yaml                       # Hyperparameters and pipeline config
│
├── src/                              # Core source code
│   ├── __init__.py
│   ├── data/
│   │   ├── data_ingestion.py         # Downloads & splits raw tweet data
│   │   └── data_preprocessing.py     # NLTK-based text normalization
│   ├── features/
│   │   └── feature_engineering.py    # Bag-of-Words vectorization (CountVectorizer)
│   └── model/
│       ├── model_building.py         # Trains Logistic Regression classifier
│       ├── model_evaluation.py       # Evaluates & logs metrics to MLflow
│       ├── register_model.py         # Registers model to MLflow Model Registry
│       └── promote_model.py          # Promotes model to production if criteria met
│
├── flask_app/                        # Production web application
│   ├── app.py                        # Flask routes, model loading, prediction logic
│   ├── preprocessing_utility.py      # Text preprocessing utilities for Flask
│   ├── templates/
│   │   └── index.html                # Web UI template
│   ├── static/                       # CSS, JS, and static assets
│   ├── requirements.txt              # Flask-specific dependencies
│   └── .env.example                  # Example environment configuration
│
├── models/                           # Serialized model artifacts
│   ├── model.pkl                     # Trained Logistic Regression model
│   └── vectorizer.pkl                # Fitted CountVectorizer
│
├── data/                             # Data directory (DVC-tracked)
│   ├── raw/                          # Original train/test CSVs after ingestion
│   ├── interim/                      # Text-normalized intermediate data
│   └── processed/                    # BoW-vectorized features (train_bow, test_bow)
│
├── reports/                          # Generated evaluation outputs
│   ├── metrics.json                  # Model evaluation metrics
│   └── experiment_info.json          # MLflow run ID and model path
│
├── tests/                            # Automated test suite
│   ├── __init__.py
│   ├── test_flask_app.py             # Flask integration tests
│   └── test_model.py                 # Model unit tests
│
├── notebooks/                        # Jupyter notebooks for exploration
├── docs/                             # Project documentation
├── references/                       # Data dictionaries and reference materials
│
├── Dockerfile                        # Multi-stage Docker build for Flask app
├── Makefile                          # Development automation commands
├── requirements.txt                  # Full project dependencies
├── requirements_essential.txt        # Minimal CI/CD dependencies
├── requirements_clean.txt            # Clean dependency list
├── setup.py                          # Package installation configuration
├── tox.ini                           # Tox testing configuration
└── README.md                         # Project documentation (this file)
```

---

## Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.10 |
| **ML Framework** | scikit-learn (Logistic Regression, CountVectorizer) |
| **NLP** | NLTK (lemmatization, stop-words, tokenization) |
| **Experiment Tracking** | MLflow + DagsHub |
| **Data Versioning** | DVC (Data Version Control) |
| **Web Framework** | Flask + Gunicorn |
| **Containerization** | Docker (multi-arch: amd64, arm64) |
| **CI/CD** | GitHub Actions |
| **Testing** | pytest + unittest |
| **Configuration** | YAML (params.yaml) + python-dotenv |

---

## ML Pipeline

The pipeline is orchestrated by **DVC** and consists of six sequential stages:

```
data_ingestion → data_preprocessing → feature_engineering
      → model_building → model_evaluation → model_registration → model_promotion
```

### Stage Details

| Stage | Script | Input | Output |
|---|---|---|---|
| `data_ingestion` | `src/data/data_ingestion.py` | Remote CSV URL | `data/raw/train.csv`, `data/raw/test.csv` |
| `data_preprocessing` | `src/data/data_preprocessing.py` | `data/raw/` | `data/interim/train_processed.csv`, `data/interim/test_processed.csv` |
| `feature_engineering` | `src/features/feature_engineering.py` | `data/interim/` | `data/processed/train_bow.csv`, `data/processed/test_bow.csv`, `models/vectorizer.pkl` |
| `model_building` | `src/model/model_building.py` | `data/processed/` | `models/model.pkl` |
| `model_evaluation` | `src/model/model_evaluation.py` | `models/`, `data/processed/` | `reports/metrics.json`, `reports/experiment_info.json` |
| `model_registration` | `src/model/register_model.py` | `reports/experiment_info.json` | MLflow Model Registry entry |
| `model_promotion` | `src/model/promote_model.py` | MLflow Registry | Production model tag |

### Text Preprocessing Steps

Raw tweet text is normalized through the following sequential transformations:

1. **Lowercase conversion** — standardize casing
2. **Stop-word removal** — remove common English words (NLTK corpus)
3. **Number removal** — strip all numeric characters
4. **Punctuation removal** — remove punctuation and extra whitespace
5. **URL removal** — strip HTTP/HTTPS and `www.` URLs
6. **Lemmatization** — reduce words to their base form (NLTK WordNetLemmatizer)

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip or conda
- Docker (for containerized deployment)
- A [DagsHub](https://dagshub.com/) account with a Personal Access Token (PAT)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/DeepuML/Ml-OPS-Project-2.git
   cd Ml-OPS-Project-2
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   # or
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   # or for a minimal install:
   pip install -r requirements_essential.txt
   ```

4. **Install the project as a package** (enables `src` imports)

   ```bash
   pip install -e .
   ```

### Environment Variables

Copy the example environment file and populate it with your credentials:

```bash
cp flask_app/.env.example .env
```

Edit `.env`:

```env
# DagsHub Personal Access Token (used for MLflow authentication)
DAGSHUB_PAT=your_dagshub_personal_access_token_here
```

> **Security Note:** Never commit your `.env` file or PAT to version control.
> The `.gitignore` already excludes `.env` files.

---

## Running the Pipeline

### Run the Full DVC Pipeline

```bash
dvc repro
```

This executes all pipeline stages in order. DVC caches intermediate results and only re-runs stages whose dependencies have changed.

### Run Individual Stages

```bash
# Data ingestion only
python src/data/data_ingestion.py

# Data preprocessing only
python src/data/data_preprocessing.py

# Feature engineering only
python src/features/feature_engineering.py

# Model training only
python src/model/model_building.py

# Model evaluation + MLflow logging
python src/model/model_evaluation.py

# Register model in MLflow Registry
python src/model/register_model.py

# Promote model to production (if criteria met)
python src/model/promote_model.py
```

### Check Pipeline Status

```bash
dvc status          # Check which stages are outdated
dvc dag             # Visualize the pipeline DAG
```

---

## Flask Web Application

The Flask app loads the production model from the **MLflow Model Registry** (with local pickle fallback) and serves two interfaces:

### Local Development

```bash
cd flask_app
export FLASK_ENV=development
python app.py
```

The app will be available at `http://localhost:5000`.

### API Reference

#### `GET /`
Serves the web UI for interactive sentiment analysis.

#### `POST /predict`
Web form submission endpoint (HTML response).

| Field | Type | Description |
|---|---|---|
| `text` | string (form) | Input tweet text |

#### `POST /api/predict`
JSON API endpoint for programmatic sentiment prediction.

**Request:**
```json
{
  "text": "I am so happy today!"
}
```

**Response:**
```json
{
  "text": "I am so happy today!",
  "sentiment": "positive",
  "prediction": 1,
  "confidence": "high"
}
```

**Status Codes:**
- `200 OK` — Prediction successful
- `400 Bad Request` — Missing or empty text
- `500 Internal Server Error` — Model not loaded or prediction failure

#### `GET /health`
Health check endpoint for load balancers and container orchestrators.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "vectorizer_loaded": true
}
```

---

## Docker Deployment

### Build the Image

```bash
docker build -t mlops-sentiment-app:latest .
```

### Run the Container

```bash
docker run -d \
  --name sentiment-app \
  -p 5000:5000 \
  -e DAGSHUB_PAT=your_dagshub_pat_here \
  -e FLASK_ENV=production \
  mlops-sentiment-app:latest
```

### Verify the Container

```bash
# Health check
curl http://localhost:5000/health

# Prediction
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"text": "I love this amazing project!"}' \
     http://localhost:5000/api/predict
```

### Docker Image on Docker Hub

The CI/CD pipeline automatically builds and pushes a multi-architecture image (linux/amd64, linux/arm64) to Docker Hub on every push to `main`:

```bash
docker pull <DOCKER_USERNAME>/mlops-sentiment-app:latest
```

---

## Experiment Tracking with MLflow & DagsHub

All experiments are tracked at: [https://dagshub.com/DeepuML/Ml-OPS-Project-2.mlflow](https://dagshub.com/DeepuML/Ml-OPS-Project-2.mlflow)

### Logged Artifacts per Run

| Type | Details |
|---|---|
| **Metrics** | `accuracy`, `precision`, `recall`, `auc` |
| **Parameters** | All `LogisticRegression` hyperparameters (C, solver, penalty) |
| **Model** | Serialized sklearn model with input signature and example |
| **Artifacts** | `metrics.json`, `experiment_info.json`, error logs |

### View Experiments Locally

```bash
mlflow ui --backend-store-uri https://dagshub.com/DeepuML/Ml-OPS-Project-2.mlflow
```

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yaml`) runs on every push and pull request to `main`, consisting of two jobs:

### Job 1: `project-testing`

| Step | Description |
|---|---|
| Checkout code | Fetch repository at the triggering commit |
| Set up Python 3.10 | Configure the Python environment |
| Cache pip dependencies | Speed up installs using GitHub Actions cache |
| Install dependencies | Install from `requirements_essential.txt` |
| Configure Git for DVC | Set git identity for DVC operations |
| Verify DAGSHUB_PAT | Validate secret is present before pipeline runs |
| Create `.env` file | Write DAGSHUB_PAT to `.env` for downstream scripts |
| Test DagsHub Connection | Validate MLflow connectivity to DagsHub |
| Setup DVC | Configure DVC remote with authentication |
| Run DVC Pipeline | Execute `dvc repro` to reproduce the full ML pipeline |
| Run Model Tests | Execute `pytest tests/` with verbose output |
| Promote Model | Run `promote_model.py` to tag production model |

### Job 2: `docker-build-deploy`

Runs only on successful `project-testing` and on `main` branch.

| Step | Description |
|---|---|
| Set up Docker Buildx | Enable multi-platform builds |
| Log in to Docker Hub | Authenticate with Docker Hub secrets |
| Extract metadata | Generate image tags (branch, PR, SHA, `latest`) |
| Build and push image | Multi-arch build (amd64, arm64) with layer caching |
| Test Docker container | Start container, wait, and verify `/health` and `/api/predict` |
| Deploy to Production | Placeholder for Kubernetes/ECS/Cloud Run deployment |
| Notify Deployment Status | Log final deployment outcome |

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `DAGSHUB_PAT` | DagsHub Personal Access Token (MLflow auth) |
| `DOCKER_USERNAME` | Docker Hub username |
| `DOCKER_PASSWORD` | Docker Hub password or access token |

---

## Model Promotion Criteria

The `promote_model.py` script automatically promotes the latest registered model version to production if **all** of the following thresholds are met:

| Metric | Minimum Threshold |
|---|---|
| Accuracy | ≥ 0.75 (75%) |
| Precision | ≥ 0.75 (75%) |
| Recall | ≥ 0.70 (70%) |
| AUC | ≥ 0.75 (75%) |

If promoted, the model version receives the tag `environment=production` in the MLflow Model Registry. The Flask app preferentially loads this production-tagged version.

---

## Testing

### Run All Tests

```bash
pytest tests/ -v --tb=short
```

### Test Coverage

| Test File | What It Tests |
|---|---|
| `tests/test_flask_app.py` | Flask routes: home, predict, API predict, health check |
| `tests/test_model.py` | Model loading, vectorizer, and prediction logic |

### Testing in CI

In CI environments (`FLASK_ENV=testing` or `CI=true`), the app automatically creates **mock model and vectorizer objects** if real artifacts are unavailable, ensuring tests always run without requiring a live MLflow connection.

---

## Configuration

### `params.yaml`

Central configuration for pipeline hyperparameters:

```yaml
data_ingestion:
  test_size: 0.25             # 25% of data held out for testing

feature_engineering:
  max_features: 5000          # Vocabulary size for Bag-of-Words vectorizer
```

### `dvc.yaml`

Defines the reproducible ML pipeline stages. Edit this file to add, remove, or modify pipeline stages.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes with descriptive commits
4. Run tests to verify: `pytest tests/ -v`
5. Push to your fork and open a Pull Request against `main`

Please ensure your code:
- Passes all existing tests
- Includes appropriate logging using the module-level logger
- Follows the existing code style and structure

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ using <a href="https://dvc.org/">DVC</a>, <a href="https://mlflow.org/">MLflow</a>, <a href="https://dagshub.com/">DagsHub</a>, and <a href="https://flask.palletsprojects.com/">Flask</a>
</p>
