from ucimlrepo import fetch_ucirepo
from dagster import asset, repository
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@asset
def load_data():
    """
    Load and preprocess the Online Shoppers Purchasing Intention dataset from UCI repo.
    
    This applies OneHotEncoding to categorical columns and then SMOTE to the training data.
    
    Returns:
        X_train_resampled: pandas DataFrame
        y_train_resampled: pandas Series
        X_test: pandas DataFrame
        y_test: pandas Series
    """
    # fetch dataset 
    online_shoppers_purchasing_intention_dataset = fetch_ucirepo(id=468) 
  
    # data (as pandas dataframes) 
    X = online_shoppers_purchasing_intention_dataset.data.features 
    y = online_shoppers_purchasing_intention_dataset.data.targets 

    # Identify categorical columns
    categorical_cols = ['Administrative', 'Informational', 'ProductRelated',
                        'SpecialDay', 'Month', 'VisitorType', 'Weekend',
                        'OperatingSystem', 'Browser', 'Region', 'TrafficType']

    # Apply OneHotEncoder to categorical columns
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_categorical_data = encoder.fit_transform(X[categorical_cols])

    # Convert encoded data to DataFrame
    encoded_categorical_df = pd.DataFrame(
        encoded_categorical_data,
        columns=encoder.get_feature_names_out(categorical_cols))

    # Drop original categorical columns and concatenate encoded columns
    X = X.drop(columns=categorical_cols).reset_index(drop=True)
    X = pd.concat([X, encoded_categorical_df], axis=1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2)
    
    # Apply SMOTE to training data
    smote = SMOTE(random_state=2)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Log data loading to MLflow
    with mlflow.start_run(run_name="load_data"):
        mlflow.log_param("data_split", "train_test_split")
        mlflow.log_param("smote", "applied")
        mlflow.log_artifact("data_loading_completed")

    return {
        "X_train": X_train_resampled, 
        "X_test": X_test, 
        "y_train": y_train_resampled, 
        "y_test": y_test
    }

@asset
def train_gb_model_grid_search(data_split):
    """
    Train a GradientBoostingClassifier model on the training data with GridSearchCV.

    Args:
        data_split: dict containing X_train and y_train

    Returns:
        GridSearchCV
    """
    gb = GradientBoostingClassifier(random_state=2)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 3, 4]
    }
    gb_grid = GridSearchCV(gb, param_grid, cv=3)
    gb_grid.fit(data_split["X_train"], data_split["y_train"])

    # Log model training to MLflow
    with mlflow.start_run(run_name="train_gb_model_grid_search"):
        mlflow.log_params(param_grid)
        mlflow.log_artifact("gb_model_training_completed")

    return gb_grid

@asset
def train_rf_model_grid_search(data_split):
    """
    Train a RandomForestClassifier model on the training data with GridSearchCV.

    Args:
        data_split: dict containing X_train and y_train

    Returns:
        GridSearchCV
    """
    rf = RandomForestClassifier(random_state=2)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 3, 4]
    }
    rf_grid = GridSearchCV(rf, param_grid, cv=3)
    rf_grid.fit(data_split["X_train"], data_split["y_train"])

    # Log model training to MLflow
    with mlflow.start_run(run_name="train_rf_model_grid_search"):
        mlflow.log_params(param_grid)
        mlflow.log_artifact("rf_model_training_completed")

    return rf_grid

@asset
def train_svc_model_grid_search(data_split):
    """
    Train a SVC model on the training data with GridSearchCV.

    Args:
        data_split: dict containing X_train and y_train

    Returns:
        GridSearchCV
    """
    svc = SVC(random_state=2)
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly']
    }
    svc_grid = GridSearchCV(svc, param_grid, cv=3)
    svc_grid.fit(data_split["X_train"], data_split["y_train"])

    # Log model training to MLflow
    with mlflow.start_run(run_name="train_svc_model_grid_search"):
        mlflow.log_params(param_grid)
        mlflow.log_artifact("svc_model_training_completed")

    return svc_grid

@asset
def train_lr_model_grid_search(data_split):
    """
    Train a LogisticRegression model on the training data with GridSearchCV.

    Args:
        data_split: dict containing X_train and y_train

    Returns:
        GridSearchCV
    """
    lr = LogisticRegression(random_state=2)
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2']
    }
    lr_grid = GridSearchCV(lr, param_grid, cv=3)
    lr_grid.fit(data_split["X_train"], data_split["y_train"])

    # Log model training to MLflow
    with mlflow.start_run(run_name="train_lr_model_grid_search"):
        mlflow.log_params(param_grid)
        mlflow.log_artifact("lr_model_training_completed")

    return lr_grid

model_list = {
    "GradientBoostingClassifier": train_gb_model_grid_search,
    "RandomForestClassifier": train_rf_model_grid_search,
    "SVC": train_svc_model_grid_search,
    "LogisticRegression": train_lr_model_grid_search
}

@asset
def evaluate_model(model, X_test, y_test, metric='accuracy'):
    """
    Evaluate the model on the testing data.
    
    Args:
        model: sklearn model
        X_test: pandas DataFrame
        y_test: pandas Series
        metric: str, metric to evaluate the model

    Returns:
        float, evaluation score
    """
    if metric == 'accuracy':
        score = model.score(X_test, y_test)
    elif metric == 'aucroc':
        score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Log model evaluation to MLflow
    with mlflow.start_run(run_name="evaluate_model"):
        mlflow.log_metric(metric, score)
        mlflow.log_artifact("model_evaluation_completed")

    return score

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
            signature = infer_signature(data_split["X_train"], model.predict(data_split["X_train"]))
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
    data_split = load_data()
    
    # Train, evaluate, and log all models
    for model_name in model_list:
        model_train_func = model_list[model_name]
        
        # Train model
        model = model_train_func(data_split)
        
        # Log model (incl. evaluation)
        log_model(model_name, model, data_split)

@repository
def mlflow_repo():
    return [ml_pipeline]