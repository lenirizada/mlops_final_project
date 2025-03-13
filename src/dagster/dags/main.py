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
    from dags.data_preprocessing import load_data

try:
    from experiments import *
except ImportError:
    from dags.experiments import *

from loguru import logger

# For model evluation
from sklearn.metrics import roc_auc_score
import shap

MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

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
    

@asset
def predict(model, X):
    return model.predict(X)

@asset
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

@asset
def log_model(model_name, model, data_split):
    try:
        with mlflow.start_run(run_name=model_name):
            X_test = data_split["X_test"]
            y_test = data_split["y_test"]
            accuracy = evaluate_model(model, X_test, y_test)
            auc_roc = evaluate_model(model, X_test, y_test, metric='aucroc')
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("auc_roc", auc_roc)
            
            # Log model with signature and input example
            input_example = X_test[:5]
            signature = infer_signature(data_split["X_train"], predict(model, data_split["X_train"]))
            mlflow.sklearn.log_model(model, "model", signature=signature,
                                     input_example=input_example)
            
            # Register the model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, model_name)
            
            print(f"{model_name} - Accuracy: {accuracy}")
            print(f"{model_name} - AUC ROC: {auc_roc}")
    except Exception as e:
        print(f"Failed to log and register model {model_name}: {e}")
        
@asset
def ml_pipeline():
    split_data = load_data()
    
    # Train, evaluate, and log all models
    for model_name in model_list:
        model_train_func = model_list[model_name]
        
        # Train model
        model = model_train_func(split_data)
        
        # Log model (incl. evaluation)
        log_model(model_name, model, split_data)

@repository
def mlflow_repo():
    return [ml_pipeline]
