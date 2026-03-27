"""
API tests for main.py using FastAPI TestClient.

Covers the rubric requirements:
  - test_get_root: GET / returns 200 AND checks response content
  - test_post_predict_low_income:  POST /predict returns <=50K
  - test_post_predict_high_income: POST /predict returns  >50K
"""
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------

# First row of UCI Adult dataset — known to be <=50K
LOW_INCOME_PAYLOAD = {
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

# High-income profile: senior exec, high capital gain, long hours
HIGH_INCOME_PAYLOAD = {
    "age": 52,
    "workclass": "Self-emp-not-inc",
    "fnlgt": 209642,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 45,
    "native-country": "United-States",
}

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_root_status_and_content():
    """GET / must return 200 and a welcome message in the response body."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "message" in body
    assert "welcome" in body["message"].lower()


def test_post_predict_low_income():
    """POST /predict with a <=50K profile must return 200 and '<=50K'."""
    response = client.post("/predict", json=LOW_INCOME_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert body["prediction"] == "<=50K"


def test_post_predict_high_income():
    """POST /predict with a >50K profile must return 200 and '>50K'."""
    response = client.post("/predict", json=HIGH_INCOME_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert body["prediction"] == ">50K"