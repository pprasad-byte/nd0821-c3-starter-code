# Put the code for your API here.
# Put the code for your API here.
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from starter.ml.data import process_data
from starter.ml.model import inference, load_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Anchor model directory relative to this file (starter/main.py -> starter/)
MODEL_DIR = Path(__file__).resolve().parent / "model"

app = FastAPI(
    title="Census Income Classifier",
    description="Predict whether income exceeds $50K/yr from Census data.",
    version="1.0.0",
)

# Load artifacts once at startup — avoids reloading on every request.
model, encoder, lb = load_model(MODEL_DIR)
logger.info("Model artifacts loaded at startup.")

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class CensusItem(BaseModel):
    """Pydantic model for a single Census inference request.

    Field aliases handle the hyphenated column names from the original CSV
    (e.g. 'marital-status') which are not valid Python identifiers.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        },
    )

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


@app.get("/")
async def root() -> dict:
    """Welcome message on the root path."""
    return {"message": "Welcome to the Census Income Classifier API!"}


@app.post("/predict")
async def predict(item: CensusItem) -> dict:
    """Run model inference on a single Census record.

    Accepts a JSON body matching the CensusItem schema and returns the
    predicted income class: '<=50K' or '>50K'.
    """
    # Reconstruct DataFrame using the original hyphenated column names
    # to match the column names the encoder was trained on.
    input_data = {
        "age": item.age,
        "workclass": item.workclass,
        "fnlgt": item.fnlgt,
        "education": item.education,
        "education-num": item.education_num,
        "marital-status": item.marital_status,
        "occupation": item.occupation,
        "relationship": item.relationship,
        "race": item.race,
        "sex": item.sex,
        "capital-gain": item.capital_gain,
        "capital-loss": item.capital_loss,
        "hours-per-week": item.hours_per_week,
        "native-country": item.native_country,
    }
    df = pd.DataFrame([input_data])

    X, _, _, _ = process_data(
        df,
        categorical_features=CAT_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = inference(model, X)
    prediction = lb.inverse_transform(preds)[0]
    logger.info(f"Prediction: {prediction}")
    return {"prediction": prediction}