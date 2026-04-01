import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

STUDENT_NAME = "Naman Omar"
ROLL_NO = "2022bcd0049"

app = FastAPI(
    title="MLOps Pipeline API",
    description=f"Iris Classification API — {STUDENT_NAME} ({ROLL_NO})",
    version="1.0.0",
)

model_data = None
SPECIES_MAP = {0: "setosa", 1: "versicolor", 2: "virginica"}


@app.on_event("startup")
async def load_model():
    global model_data
    model_path = "models/model.pkl"
    if os.path.exists(model_path):
        model_data = joblib.load(model_path)
        print(f"Model loaded. Features: {model_data['features']}")
    else:
        print("WARNING: models/model.pkl not found. Run train.py first.")


@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "name": STUDENT_NAME,
        "roll_no": ROLL_NO,
        "model_loaded": model_data is not None,
        "model_accuracy": model_data["accuracy"] if model_data else None,
    }


class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                }
            ]
        }
    }


@app.post("/predict")
def predict(request: PredictRequest):
    if model_data is None:
        return {
            "error": "Model not loaded. Run train.py first.",
            "name": STUDENT_NAME,
            "roll_no": ROLL_NO,
        }

    model = model_data["model"]
    features = model_data["features"]

    input_map = {
        "sepal_length": request.sepal_length,
        "sepal_width": request.sepal_width,
        "petal_length": request.petal_length,
        "petal_width": request.petal_width,
    }
    feature_values = [[input_map[f] for f in features]]

    prediction_class = int(model.predict(feature_values)[0])
    prediction_label = SPECIES_MAP.get(prediction_class, str(prediction_class))

    return {
        "prediction": prediction_label,
        "prediction_class": prediction_class,
        "features_used": features,
        "name": STUDENT_NAME,
        "roll_no": ROLL_NO,
    }
