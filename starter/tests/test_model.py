"""
Unit tests for ml/model.py and ml/data.py.

Covers:
  1. train_model returns a RandomForestClassifier instance
  2. inference returns np.ndarray with correct length
  3. compute_model_metrics returns correct values for perfect predictions
  4. save_model / load_model roundtrip produces identical predictions
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    save_model,
    train_model,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

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


@pytest.fixture(scope="module")
def small_dataset():
    """
    Minimal synthetic Census-shaped DataFrame sufficient to exercise the full
    process_data → train_model → inference pipeline without loading real data.
    """
    n = 100
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "age": rng.integers(20, 65, size=n),
        "workclass": rng.choice(["Private", "Self-emp", "Gov"], size=n),
        "fnlgt": rng.integers(100000, 500000, size=n),
        "education": rng.choice(["Bachelors", "HS-grad", "Masters"], size=n),
        "education-num": rng.integers(8, 16, size=n),
        "marital-status": rng.choice(["Married", "Never-married"], size=n),
        "occupation": rng.choice(["Tech-support", "Sales", "Other"], size=n),
        "relationship": rng.choice(["Husband", "Not-in-family"], size=n),
        "race": rng.choice(["White", "Black", "Asian"], size=n),
        "sex": rng.choice(["Male", "Female"], size=n),
        "capital-gain": rng.integers(0, 5000, size=n),
        "capital-loss": rng.integers(0, 500, size=n),
        "hours-per-week": rng.integers(30, 60, size=n),
        "native-country": rng.choice(["United-States", "Mexico"], size=n),
        "salary": rng.choice(["<=50K", ">50K"], size=n),
    })
    return df


@pytest.fixture(scope="module")
def trained_artifacts(small_dataset):
    """Returns (model, encoder, lb, X_test, y_test) from a quick training run."""
    X, y, encoder, lb = process_data(
        small_dataset,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )
    model = train_model(X, y)
    return model, encoder, lb, X, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_train_model_returns_classifier(trained_artifacts):
    """train_model must return a fitted RandomForestClassifier."""
    model, *_ = trained_artifacts
    assert isinstance(model, RandomForestClassifier), (
        f"Expected RandomForestClassifier, got {type(model)}"
    )


def test_inference_returns_array_correct_length(trained_artifacts):
    """inference must return an np.ndarray with len equal to number of input rows."""
    model, _, _, X, _ = trained_artifacts
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray), (
        f"Expected np.ndarray, got {type(preds)}"
    )
    assert len(preds) == len(X), (
        f"Prediction length {len(preds)} != input length {len(X)}"
    )


def test_compute_model_metrics_perfect_predictions():
    """compute_model_metrics must return 1.0 for all metrics when preds == y."""
    y = np.array([0, 1, 0, 1, 1, 0])
    preds = np.array([0, 1, 0, 1, 1, 0])   # perfect predictions
    precision, recall, f1 = compute_model_metrics(y, preds)
    assert precision == pytest.approx(1.0), f"Precision: {precision}"
    assert recall == pytest.approx(1.0), f"Recall: {recall}"
    assert f1 == pytest.approx(1.0), f"F1: {f1}"


def test_save_load_model_roundtrip(trained_artifacts):
    """Loaded model must produce identical predictions to the original model."""
    model, encoder, lb, X, _ = trained_artifacts
    original_preds = inference(model, X)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_model(model, encoder, lb, Path(tmpdir))
        loaded_model, loaded_encoder, loaded_lb = load_model(Path(tmpdir))

    loaded_preds = inference(loaded_model, X)
    np.testing.assert_array_equal(
        original_preds,
        loaded_preds,
        err_msg="Loaded model predictions differ from original",
    )
