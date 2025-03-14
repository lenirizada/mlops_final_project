from dagster import op, job, repository

import mlflow
from mlflow.models import infer_signature

from evidently.metric_preset import ClassificationPreset
from evidently.report import Report
from matplotlib import pyplot as plt

try:
    from data_preprocessing import load_data, preprocess_data
except ImportError:
    from assets.data_preprocessing import load_data, preprocess_data

try:
    from experiments import *
except ImportError:
    from assets.experiments import *

from loguru import logger

# For model evluation
from sklearn.metrics import roc_auc_score
import shap

MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@op
def predict(model, X):
    return model.predict(X)

@op
def evaluate_model(model, split_data) -> dict:
    """
    Evaluate the model on the testing data.
    
    Args:
        model: sklearn model
        split_data: Dict(
            X_test: pandas DataFrame
            y_test: pandas Series
        )
    
    Returns:
        evaluation metrics: Dict(
            accuracy: float
            auc_roc: float
        )
    """
    accuracy = model.score(split_data['X_test'], split_data['y_test'])
    auc_roc = roc_auc_score(
        split_data['y_test'],
        model.predict_proba(split_data['X_test'])[:, 1]
    )
    print(f"Accuracy: {accuracy}")
    print(f"AUC-ROC: {auc_roc}")

    return {
        "accuracy": accuracy,
        "auc_roc": auc_roc
    }

@op
def log_model(model, metrics: dict, split_data: dict) -> None:
    if type(model).__name__ == 'GridSearchCV':
        model_name = type(model.estimator).__name__
    else:
        model_name = type(model).__name__

    try:
        # Log metrics
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("auc_roc", metrics['auc_roc'])
        
        # Log model with signature and input example
        signature = infer_signature(split_data["X_train"],
                                    model.predict(split_data["X_train"]))
        input_example = split_data["X_test"][:5]
        
        mlflow.sklearn.log_model(model, "model", signature=signature,
                                 input_example=input_example)
        
        # Register the model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, model_name)
    except Exception as e:
        print(f"Failed to log and register model {model_name}: {e}")
        
@job
def ml_pipeline():
    features, targets = load_data()
    split_data = preprocess_data(features, targets)

    with mlflow.start_run(run_name="GBM", nested=True):
        # Train model
        model = train_gb_model_grid_search(split_data)
        
        # Evaluate model
        metrics = evaluate_model(model, split_data)
        
        # Log and register model
        log_model(model, metrics, split_data)


@repository
def mlflow_repo():
    return [ml_pipeline]
