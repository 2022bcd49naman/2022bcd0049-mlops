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
    description=f"Wine Classification API — {STUDENT_NAME} ({ROLL_NO})",
    version="1.0.0",
)

model_data = None
CLASS_MAP = {0: "class_0", 1: "class_1", 2: "class_2"}


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
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavonoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315: float
    proline: float

    model_config = {
        "json_schema_extra": {
            "examples": [
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
                    "proline": 1050.0,
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
        "alcohol": request.alcohol,
        "malic_acid": request.malic_acid,
        "ash": request.ash,
        "alcalinity_of_ash": request.alcalinity_of_ash,
        "magnesium": request.magnesium,
        "total_phenols": request.total_phenols,
        "flavanoids": request.flavanoids,
        "nonflavonoid_phenols": request.nonflavonoid_phenols,
        "proanthocyanins": request.proanthocyanins,
        "color_intensity": request.color_intensity,
        "hue": request.hue,
        "od280_od315": request.od280_od315,
        "proline": request.proline,
    }
    feature_values = [[input_map[f] for f in features]]

    prediction_class = int(model.predict(feature_values)[0])
    prediction_label = CLASS_MAP.get(prediction_class, str(prediction_class))

    return {
        "prediction": prediction_label,
        "prediction_class": prediction_class,
        "dataset": "wine",
        "features_used": features,
        "name": STUDENT_NAME,
        "roll_no": ROLL_NO,
    }
