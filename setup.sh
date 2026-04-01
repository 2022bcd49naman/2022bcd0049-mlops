#!/bin/bash
# ============================================================
# MLOps Pipeline Setup Script
# Student: Naman Omar | Roll No: 2022bcd0049
# ============================================================

set -e

# Load env vars
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

echo "==> Installing dependencies"
pip install -r requirements.txt

echo "==> Initializing git (if not already)"
git init 2>/dev/null || true

echo "==> Initializing DVC"
dvc init --no-scm 2>/dev/null || dvc init

echo "==> Configuring DVC S3 remote"
dvc remote add -d myremote s3://${S3_BUCKET_NAME:-2022bcd0049-mlops-bucket}/data
dvc remote modify myremote access_key_id ${AWS_ACCESS_KEY_ID}
dvc remote modify myremote secret_access_key ${AWS_SECRET_ACCESS_KEY}
dvc remote modify myremote session_token ${AWS_SESSION_TOKEN}
dvc remote modify myremote region ${AWS_DEFAULT_REGION:-us-east-1}

echo "==> Generating datasets (v1 + v2)"
python -c "
import os, pandas as pd
from sklearn.datasets import load_iris
os.makedirs('data', exist_ok=True)
iris = load_iris()
import pandas as pd
df = pd.DataFrame(iris.data, columns=['sepal_length','sepal_width','petal_length','petal_width'])
df['species'] = iris.target
df.iloc[:100].to_csv('data/iris_v1.csv', index=False)
df.to_csv('data/iris_v2.csv', index=False)
print('iris_v1.csv (100 rows) and iris_v2.csv (150 rows) created.')
"

echo "==> Tracking data v1 with DVC (version 1)"
dvc add data/iris_v1.csv
git add data/.gitignore data/iris_v1.csv.dvc
git commit -m "DVC: Add dataset version 1 (100 rows)" 2>/dev/null || true
dvc push

echo "==> Tracking data v2 with DVC (version 2)"
dvc add data/iris_v2.csv
git add data/iris_v2.csv.dvc
git commit -m "DVC: Add dataset version 2 (150 rows)" 2>/dev/null || true
dvc push

echo "==> Running training (5 MLflow runs)"
python train.py

echo "==> Building Docker image"
docker build -t ${DOCKERHUB_USERNAME}/2022bcd0049-mlops:latest .

echo "==> Pushing to Docker Hub"
docker login -u ${DOCKERHUB_USERNAME} -p ${DOCKERHUB_TOKEN}
docker push ${DOCKERHUB_USERNAME}/2022bcd0049-mlops:latest

echo ""
echo "============================================================"
echo "Setup complete!"
echo "  MLflow UI:  mlflow ui  (then open http://localhost:5000)"
echo "  API:        uvicorn app:app --reload"
echo "  Docker:     docker run -p 8000:8000 ${DOCKERHUB_USERNAME}/2022bcd0049-mlops"
echo "============================================================"
