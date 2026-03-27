from sklearn.metrics import fbeta_score, precision_score, recall_score
import logging
from pathlib import Path
 
import joblib
from sklearn.ensemble import RandomForestClassifier
logger = logging.getLogger(__name__)



def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info(f"Model trained on {X_train.shape[0]} samples, {X_train.shape[1]} features.")
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(
    model: RandomForestClassifier,
    encoder,
    lb,
    model_dir: Path,
) -> None:
    """
    Saves model, encoder, and label binarizer to disk via joblib.
 
    Inputs
    ------
    model : RandomForestClassifier
    encoder : sklearn.preprocessing.OneHotEncoder
    lb : sklearn.preprocessing.LabelBinarizer
    model_dir : Path
        Directory to save artifacts into.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
 
    joblib.dump(model, model_dir / "model.pkl")
    joblib.dump(encoder, model_dir / "encoder.pkl")
    joblib.dump(lb, model_dir / "lb.pkl")
    logger.info(f"Artifacts saved to {model_dir}")
 
 
def load_model(model_dir: Path) -> tuple:
    """
    Loads model, encoder, and label binarizer from disk.
 
    Inputs
    ------
    model_dir : Path
        Directory containing model.pkl, encoder.pkl, lb.pkl.
    Returns
    -------
    model : RandomForestClassifier
    encoder : OneHotEncoder
    lb : LabelBinarizer
    """
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "model.pkl")
    encoder = joblib.load(model_dir / "encoder.pkl")
    lb = joblib.load(model_dir / "lb.pkl")
    logger.info(f"Artifacts loaded from {model_dir}")
    return model, encoder, lb
