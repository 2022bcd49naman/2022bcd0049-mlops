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
from sklearn.datasets import load_wine
from dotenv import load_dotenv
from warnings import filterwarnings

filterwarnings("ignore")

load_dotenv()

STUDENT_NAME = "Naman Omar"
ROLL_NO = "2022bcd0049"
EXPERIMENT_NAME = f"{ROLL_NO}_experiment"

ALL_FEATURES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavonoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315", "proline"
]
# Weaker subset -> naturally lower accuracy without any tricks
REDUCED_FEATURES = ["ash", "magnesium", "hue", "alcalinity_of_ash"]


def prepare_data():
    os.makedirs("data", exist_ok=True)
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=ALL_FEATURES)
    df["target"] = wine.target

    # Shuffle so all 3 classes appear proportionally in both splits
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Version 1: partial dataset (first 100 rows)
    df_v1 = df.iloc[:100].copy()
    df_v1.to_csv("data/wine_v1.csv", index=False)

    # Version 2: full dataset (all 178 rows)
    df.to_csv("data/wine_v2.csv", index=False)

    print("Datasets prepared:")
    print(f"  wine_v1.csv -> {len(df_v1)} rows (partial)")
    print(f"  wine_v2.csv -> {len(df)} rows (full)")


def run_experiment(run_name, dataset_version, model, features, extra_params):
    df = pd.read_csv(f"data/wine_{dataset_version}.csv")
    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
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

    # Run 1: v1 (100 rows), RandomForest, all 13 features, base config
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

    # Run 2: v1 (100 rows), RandomForest, all features, constrained depth
    acc, f1, mdl, feats = run_experiment(
        run_name="Run2_v1_RF_depth5",
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

    # Run 3: v2 (178 rows), RandomForest, all features, base config
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

    # Run 4: v2 (178 rows), RandomForest, reduced features only
    acc, f1, mdl, feats = run_experiment(
        run_name="Run4_v2_RF_feature_selection",
        dataset_version="v2",
        model=RandomForestClassifier(n_estimators=100, random_state=42),
        features=REDUCED_FEATURES,
        extra_params={"model_type": "RandomForest", "n_estimators": 100,
                      "feature_selection": ",".join(REDUCED_FEATURES)},
    )
    results.append({"Run": "Run 4", "Dataset": "v2", "Model": "RandomForest",
                    "Key Parameters": "n_estimators=100, reduced features",
                    "Metric 1 (Accuracy)": round(acc, 4), "Metric 2 (F1)": round(f1, 4)})
    if acc > best_acc:
        best_acc, best_model, best_features = acc, mdl, feats

    # Run 5: v2 (178 rows), LogisticRegression, reduced features
    acc, f1, mdl, feats = run_experiment(
        run_name="Run5_v2_LR_feature_selection",
        dataset_version="v2",
        model=LogisticRegression(C=0.1, max_iter=500, random_state=42),
        features=REDUCED_FEATURES,
        extra_params={"model_type": "LogisticRegression", "C": 0.1, "max_iter": 500,
                      "feature_selection": ",".join(REDUCED_FEATURES)},
    )
    results.append({"Run": "Run 5", "Dataset": "v2", "Model": "LogisticRegression",
                    "Key Parameters": "C=0.1, reduced features",
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
