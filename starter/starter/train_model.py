# Script to train machine learning model.
import logging
from pathlib import Path
 
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    save_model,
    train_model,
)
 
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add code to load in the data.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "census_clean.csv"
MODEL_DIR = BASE_DIR / "model"
SLICE_OUTPUT = BASE_DIR / "slice_output.txt"

logger.info(f"Loading data from {DATA_PATH}")
data = pd.read_csv(DATA_PATH)
logger.info(f"Data shape: {data.shape}")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)
 
# Train and save a model.
logger.info("Training model...")
model = train_model(X_train, y_train)

# Evaluate overall test metrics.
preds = inference(model, X_test)
precision, recall, f1 = compute_model_metrics(y_test, preds)
logger.info(
    f"Overall — Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}"
)
 
# Save artifacts.
save_model(model, encoder, lb, MODEL_DIR)
logger.info(f"Artifacts saved to {MODEL_DIR}")
 
# Compute and save slice metrics.
logger.info("Computing slice metrics...")

def compute_slice_metrics(test_df, feature):
    """
    Compute model metrics for every unique value of a given categorical feature.
 
    For each unique value the feature is held fixed, the slice is processed in
    inference mode, and precision / recall / F1 are recorded.
 
    Inputs
    ------
    test_df : pd.DataFrame
        Test split DataFrame including the label column 'salary'.
    feature : str
        Categorical column to slice on.
    """
    loaded_model, loaded_encoder, loaded_lb = load_model(MODEL_DIR)
    results = []
 
    for value in sorted(test_df[feature].unique()):
        slice_df = test_df[test_df[feature] == value]
        if slice_df.empty:
            continue
 
        X_slice, y_slice, _, _ = process_data(
            slice_df,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=loaded_encoder,
            lb=loaded_lb,
        )
        slice_preds = inference(loaded_model, X_slice)
        precision, recall, f1 = compute_model_metrics(y_slice, slice_preds)
        results.append({
            "feature": feature,
            "value": value,
            "n": len(slice_df),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return results
 

all_results = []
for feature in cat_features:
    results = compute_slice_metrics(test, feature)
    all_results.extend(results)
    logger.info(f"  {feature}: {len(results)} slices")
 
# Write slice output to file.
with open(SLICE_OUTPUT, "w") as f:
    for r in all_results:
        f.write(
            f"Feature: {r['feature']:<20} Value: {r['value']:<30} "
            f"n={r['n']:<6} "
            f"Precision: {r['precision']:.4f}  "
            f"Recall: {r['recall']:.4f}  "
            f"F1: {r['f1']:.4f}\n"
        )
 
logger.info(f"Slice output written to {SLICE_OUTPUT}")
logger.info("Done.")
 