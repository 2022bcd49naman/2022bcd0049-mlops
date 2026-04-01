# 2022bcd0049-mlops

**Student:** Naman Omar
**Roll No:** 2022bcd0049
**Course:** CSS426 — MLOps

End-to-end MLOps pipeline implementing data versioning, experiment tracking, automated CI/CD, and containerized model deployment for Wine classification.

---

## Pipeline Overview

```
GitHub Push
    │
    ▼
GitHub Actions CI/CD
    │
    ├── DVC Pull (wine_v1.csv, wine_v2.csv from S3)
    ├── python train.py  ──► MLflow logs 5 runs
    ├── docker build
    ├── docker push ──► Docker Hub
    └── Inference validation (health + predict)
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Data Versioning | DVC + AWS S3 |
| Experiment Tracking | MLflow |
| CI/CD | GitHub Actions |
| Model Serving | FastAPI |
| Containerization | Docker + Docker Hub |
| ML Models | scikit-learn (RandomForest, LogisticRegression) |
| Dataset | Wine Recognition Dataset (UCI / sklearn) |

---

## Dataset

- **Wine Recognition Dataset** — 178 samples, 13 features, 3 classes
- **v1:** 100 rows (partial) → `data/wine_v1.csv`
- **v2:** 178 rows (full) → `data/wine_v2.csv`
- Both versions tracked with DVC and stored in S3

---

## MLflow Experiment Results

**Experiment:** `2022bcd0049_experiment`

| Run | Dataset | Model | Key Config | Accuracy | F1 |
|-----|---------|-------|-----------|----------|----|
| Run 1 | v1 | RandomForest | n_estimators=100, all features | 0.9000 | 0.8968 |
| Run 2 | v1 | RandomForest | n_estimators=200, max_depth=5 | **0.9500** | **0.9506** |
| Run 3 | v2 | RandomForest | n_estimators=100, all features | 0.9444 | 0.9435 |
| Run 4 | v2 | RandomForest | n_estimators=100, reduced features | 0.8611 | 0.8536 |
| Run 5 | v2 | LogisticRegression | C=0.1, reduced features | 0.8333 | 0.8327 |

**Best model:** Run 2 — accuracy **0.9500**

---

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/<your-username>/2022bcd0049-mlops.git
cd 2022bcd0049-mlops
pip install -r requirements.txt
```

### 2. Pull Data
```bash
dvc pull data/wine_v1.csv data/wine_v2.csv
```

### 3. Train
```bash
python train.py
```

### 4. Run API locally
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 5. Run via Docker
```bash
docker pull 2022bcd0049namanomar/2022bcd0049-mlops:latest
docker run -d -p 8000:8000 2022bcd0049namanomar/2022bcd0049-mlops:latest
```

---

## API Endpoints

### `GET /health`
```json
{
  "status": "healthy",
  "name": "Naman Omar",
  "roll_no": "2022bcd0049",
  "model_loaded": true,
  "model_accuracy": 0.95
}
```

### `POST /predict`

**Request:**
```json
{
  "alcohol": 13.2,
  "malic_acid": 1.78,
  "ash": 2.14,
  "alcalinity_of_ash": 11.2,
  "magnesium": 100.0,
  "total_phenols": 2.65,
  "flavanoids": 2.76,
  "nonflavonoid_phenols": 0.26,
  "proanthocyanins": 1.28,
  "color_intensity": 4.38,
  "hue": 1.05,
  "od280_od315": 3.4,
  "proline": 1050.0
}
```

**Response:**
```json
{
  "prediction": "class_0",
  "prediction_class": 0,
  "dataset": "wine",
  "features_used": ["alcohol", "malic_acid", "..."],
  "name": "Naman Omar",
  "roll_no": "2022bcd0049"
}
```

---

## Project Structure

```
2022bcd0049-mlops/
├── train.py                  # Training + MLflow logging (5 runs)
├── app.py                    # FastAPI server
├── Dockerfile                # Container definition
├── requirements.txt          # Pinned dependencies
├── dvc.yaml                  # DVC pipeline
├── params.yaml               # Training parameters
├── metrics.json              # Auto-generated run metrics
├── .github/
│   └── workflows/
│       └── mlops.yml         # GitHub Actions pipeline
└── data/
    ├── wine_v1.csv           # 100-row partial dataset
    └── wine_v2.csv           # 178-row full dataset
```

---

## Links

- **GitHub:** `https://github.com/<your-username>/2022bcd0049-mlops`
- **Docker Hub:** `https://hub.docker.com/r/2022bcd0049namanomar/2022bcd0049-mlops`

---

*Naman Omar — 2022bcd0049*
