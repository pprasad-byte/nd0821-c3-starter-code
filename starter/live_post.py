"""
Script to POST a sample record to the live deployed API and print
the prediction result and HTTP status code.
"""
import requests

# Update this URL after deploying to Render/Heroku
URL = "https://pprasad-census.onrender.com/predict"

# High-income profile — expected prediction: >50K
payload = {
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

response = requests.post(URL, json=payload)

print(f"Status code: {response.status_code}")
print(f"Prediction:  {response.json()['prediction']}")