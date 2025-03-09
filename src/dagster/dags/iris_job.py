import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from dagster import asset, op, job, repository

from evidently.metric_preset import ClassificationPreset
from evidently.report import Report
from matplotlib import pyplot as plt

try:
    from data_preprocessing import load_data
except ImportError:
    from src.data_preprocessing import load_data

try:
    from experiments import *
except ImportError:
    from src.experiments import *

from loguru import logger

# For model evluation
from sklearn.metrics import roc_auc_score
import shap


@asset
def split_data():
    """
    Split the data into training and testing sets.
    
    Returns:
        X_train: pandas DataFrame
        X_test: pandas DataFrame
        y_train: pandas Series
        y_test: pandas Series
    """
    X_train, X_test, y_train, y_test = load_data()
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    

@op
def predict(model, X):
    return model.predict(X)

@op
def evaluate_model(model, X_test, y_test, metric='accuracy'):
    """
    Evaluate the model on the testing data.
    
    Args:
        model: sklearn model
        X_test: pandas DataFrame
        y_test: pandas Series
    """
    if metric == 'accuracy':
        return model.score(X_test, y_test)
    
    if metric == 'aucroc':
        return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

@op
def log_model(model_train_func, model_name):
    try:
        with mlflow.start_run(run_name=model_name):
            X_train, X_test, y_train, y_test = load_data()
            
            model = model_train_func(X_train, y_train)
            accuracy = evaluate_model(model, X_test, y_test)
            auc_roc = evaluate_model(model, X_test, y_test, metric='aucroc')
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("auc_roc", auc_roc)
            
            # Log model with signature and input example
            input_example = X_test[:5]
            signature = infer_signature(X_train, predict(model, X_train))
            mlflow.sklearn.log_model(model, "model", signature=signature,
                                     input_example=input_example)
            
            # Register the model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, model_name)
            
            print(f"{model_name} - Accuracy: {accuracy}")
            print(f"{model_name} - AUC ROC: {auc_roc}")
    except Exception as e:
        print(f"Failed to log and register model {model_name}: {e}")
        
@job
def ml_pipeline():
    """Train and evaluate multiple models."""
    for model_name in model_list:
        log_model(model_list[model_name], model_name)
        
@repository
def mlflow_repo():
    return [ml_pipeline]
