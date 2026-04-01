import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_iris
from dotenv import load_dotenv

load_dotenv()

STUDENT_NAME = "Naman Omar"
ROLL_NO = "2022bcd0049"
EXPERIMENT_NAME = f"{ROLL_NO}_experiment"

ALL_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
REDUCED_FEATURES = ["petal_length", "petal_width"]


def prepare_data():
    os.makedirs("data", exist_ok=True)
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=ALL_FEATURES)
    df["species"] = iris.target

    # Version 1: partial dataset (first 100 rows)
    df_v1 = df.iloc[:100].copy()
    df_v1.to_csv("data/iris_v1.csv", index=False)

    # Version 2: full dataset (all 150 rows)
    df.to_csv("data/iris_v2.csv", index=False)

    print("Datasets prepared:")
    print(f"  iris_v1.csv -> {len(df_v1)} rows (partial)")
    print(f"  iris_v2.csv -> {len(df)} rows (full)")


def run_experiment(run_name, dataset_version, model, features, extra_params):
    df = pd.read_csv(f"data/iris_{dataset_version}.csv")
    X = df[features]
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("student_name", STUDENT_NAME)
        mlflow.log_param("roll_no", ROLL_NO)
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("features_used", ",".join(features))
        mlflow.log_param("num_features", len(features))
        for k, v in extra_params.items():
            mlflow.log_param(k, v)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("accuracy", round(acc, 4))
        mlflow.log_metric("f1_score", round(f1, 4))
        mlflow.sklearn.log_model(model, "model")

        print(f"  [{run_name}] accuracy={acc:.4f}, f1={f1:.4f}")
        return acc, f1, model, features


def main():
    prepare_data()
    os.makedirs("models", exist_ok=True)

    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\nMLflow experiment: {EXPERIMENT_NAME}")
    print("=" * 60)

    results = []
    best_acc = -1
    best_model = None
    best_features = None

    # Run 1: v1, RandomForest, all features, base config
    acc, f1, mdl, feats = run_experiment(
        run_name="Run1_v1_RF_base",
        dataset_version="v1",
        model=RandomForestClassifier(n_estimators=100, random_state=42),
        features=ALL_FEATURES,
        extra_params={"model_type": "RandomForest", "n_estimators": 100},
    )
    results.append({"Run": "Run 1", "Dataset": "v1", "Model": "RandomForest",
                    "Key Parameters": "n_estimators=100, all features",
                    "Metric 1 (Accuracy)": round(acc, 4), "Metric 2 (F1)": round(f1, 4)})
    if acc > best_acc:
        best_acc, best_model, best_features = acc, mdl, feats

    # Run 2: v1, RandomForest, all features, hyperparameter change
    acc, f1, mdl, feats = run_experiment(
        run_name="Run2_v1_RF_n200",
        dataset_version="v1",
        model=RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
        features=ALL_FEATURES,
        extra_params={"model_type": "RandomForest", "n_estimators": 200, "max_depth": 5},
    )
    results.append({"Run": "Run 2", "Dataset": "v1", "Model": "RandomForest",
                    "Key Parameters": "n_estimators=200, max_depth=5",
                    "Metric 1 (Accuracy)": round(acc, 4), "Metric 2 (F1)": round(f1, 4)})
    if acc > best_acc:
        best_acc, best_model, best_features = acc, mdl, feats

    # Run 3: v2, RandomForest, all features, base config
    acc, f1, mdl, feats = run_experiment(
        run_name="Run3_v2_RF_base",
        dataset_version="v2",
        model=RandomForestClassifier(n_estimators=100, random_state=42),
        features=ALL_FEATURES,
        extra_params={"model_type": "RandomForest", "n_estimators": 100},
    )
    results.append({"Run": "Run 3", "Dataset": "v2", "Model": "RandomForest",
                    "Key Parameters": "n_estimators=100, all features",
                    "Metric 1 (Accuracy)": round(acc, 4), "Metric 2 (F1)": round(f1, 4)})
    if acc > best_acc:
        best_acc, best_model, best_features = acc, mdl, feats

    # Run 4: v2, RandomForest, reduced features (feature selection)
    acc, f1, mdl, feats = run_experiment(
        run_name="Run4_v2_RF_feature_selection",
        dataset_version="v2",
        model=RandomForestClassifier(n_estimators=100, random_state=42),
        features=REDUCED_FEATURES,
        extra_params={"model_type": "RandomForest", "n_estimators": 100,
                      "feature_selection": "petal_length,petal_width"},
    )
    results.append({"Run": "Run 4", "Dataset": "v2", "Model": "RandomForest",
                    "Key Parameters": "n_estimators=100, petal features only",
                    "Metric 1 (Accuracy)": round(acc, 4), "Metric 2 (F1)": round(f1, 4)})
    if acc > best_acc:
        best_acc, best_model, best_features = acc, mdl, feats

    # Run 5: v2, LogisticRegression, reduced features (different model + feature selection)
    acc, f1, mdl, feats = run_experiment(
        run_name="Run5_v2_LR_feature_selection",
        dataset_version="v2",
        model=LogisticRegression(C=1.0, max_iter=300, random_state=42),
        features=REDUCED_FEATURES,
        extra_params={"model_type": "LogisticRegression", "C": 1.0, "max_iter": 300,
                      "feature_selection": "petal_length,petal_width"},
    )
    results.append({"Run": "Run 5", "Dataset": "v2", "Model": "LogisticRegression",
                    "Key Parameters": "C=1.0, petal features only",
                    "Metric 1 (Accuracy)": round(acc, 4), "Metric 2 (F1)": round(f1, 4)})
    if acc > best_acc:
        best_acc, best_model, best_features = acc, mdl, feats

    # Save best model with metadata
    model_data = {
        "model": best_model,
        "features": best_features,
        "accuracy": round(best_acc, 4),
        "student_name": STUDENT_NAME,
        "roll_no": ROLL_NO,
    }
    joblib.dump(model_data, "models/model.pkl")
    print(f"\nBest model saved -> accuracy={best_acc:.4f}, features={best_features}")

    # Save metrics.json
    metrics_output = {
        "student_name": STUDENT_NAME,
        "roll_no": ROLL_NO,
        "experiment_name": EXPERIMENT_NAME,
        "best_accuracy": round(best_acc, 4),
        "runs": results,
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=2)
    print("metrics.json saved.")


if __name__ == "__main__":
    main()
