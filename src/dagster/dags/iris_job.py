import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from evidently.metric_preset import ClassificationPreset
from evidently.report import Report
from loguru import logger
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from dagster import Definitions, OpExecutionContext, asset, job, op

MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@asset
def iris_dataset() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df = iris.frame
    df["target"] = iris.target
    return df


@asset
def split_data(iris_dataset: pd.DataFrame):
    X = iris_dataset.drop(columns=["target"])
    y = iris_dataset["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


@op
def train_model(split_data):
    X_train = split_data["X_train"]
    y_train = split_data["y_train"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model  # No mlflow logging here, handled in `log_to_mlflow`


@op
def predict(model, split_data):
    X_train = split_data["X_train"]
    y_train_pred = model.predict(X_train)
    return {"y_train_pred": y_train_pred}


@op
def log_to_mlflow(context: OpExecutionContext, train_model, split_data, predict):
    logger.add("mlflow_training.log", rotation="1 MB", level="INFO")

    X_train = split_data["X_train"]
    y_train = split_data["y_train"]
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]
    y_train_pred = predict["y_train_pred"]  # Get predictions

    model = train_model

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Run started: {run_id}")

        mlflow.sklearn.log_model(model, "iris_rf_model")
        mlflow.log_params({"n_estimators": 100, "random_state": 42})

        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)  # Log test accuracy

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_train)

        for i in range(3):  # Loop over classes
            plt.figure(figsize=(6, 4))
            shap.plots.waterfall(shap_values[0, :, i], show=False)
            plt.title(f"Waterfall Plot - Class {i}")
            plt.savefig(f"shap_summary_class_{i}.png", bbox_inches="tight")
            mlflow.log_artifact(f"shap_summary_class_{i}.png")

        # Include predictions in the report data
        report_data = pd.DataFrame({"target": y_train, "prediction": y_train_pred})

        report = Report(metrics=[ClassificationPreset()])
        report.run(reference_data=report_data, current_data=report_data)
        report.save_html("evidently_report.html")
        mlflow.log_artifact("evidently_report.html")

        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/iris_rf_model",  # Use run_id from the active run
            name="iris_classifier",
        )

        logger.info(f"Model registered: {result.name}, version: {result.version}")
        logger.info(f"Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")


@op
def log_to_mlflow_skewed(context: OpExecutionContext, train_model, split_data, predict):
    logger.add("mlflow_training.log", rotation="1 MB", level="INFO")

    X_train = split_data["X_train"]
    y_train = split_data["y_train"]
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    # Ensure y_train_pred is a pandas Series with the same index as y_train
    y_train_pred = pd.Series(predict["y_train_pred"], index=y_train.index)

    model = train_model

    # Introduce skew to data
    skewed_data = skew_data(split_data)

    # Ensure predictions match new y_train index after skewing
    y_train_pred_skewed = y_train_pred.loc[skewed_data["y_train"].index]

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Run started: {run_id}")

        # Log original model
        mlflow.sklearn.log_model(model, "iris_rf_model")
        mlflow.log_params({"n_estimators": 100, "random_state": 42})

        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_train)

        for i in range(3):  # Loop over classes
            plt.figure(figsize=(6, 4))
            shap.plots.waterfall(shap_values[0, :, i], show=False)
            plt.title(f"Waterfall Plot - Class {i}")
            plt.savefig(f"shap_summary_class_{i}.png", bbox_inches="tight")
            mlflow.log_artifact(f"shap_summary_class_{i}.png")

        # Evidently Report (Drift Detection)
        report_data = pd.DataFrame({"target": y_train, "prediction": y_train_pred})
        skewed_report_data = pd.DataFrame(
            {"target": skewed_data["y_train"], "prediction": y_train_pred_skewed}
        )

        report = Report(metrics=[ClassificationPreset()])
        report.run(reference_data=report_data, current_data=skewed_report_data)
        report.save_html("evidently_report.html")
        mlflow.log_artifact("evidently_report.html")

        # Register Model Under Different Name
        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/iris_rf_model",
            name="iris_rf_model_skewed",
        )

        logger.info(f"Model registered: {result.name}, version: {result.version}")
        logger.info(f"Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")


@op
def skew_data(split_data):
    """Applies artificial drift to data for Evidently AI to detect."""
    X_train = split_data["X_train"].copy()
    y_train = split_data["y_train"].copy()

    # Feature shift (increase first feature by 20%)
    X_train.iloc[:, 0] *= 1.2

    # Add noise
    noise = np.random.normal(0, 0.1, X_train.shape)
    X_train += noise

    # Reduce one class by 50%
    if len(y_train.unique()) > 1:
        class_to_reduce = y_train.value_counts().idxmin()
        indices_to_drop = (
            y_train[y_train == class_to_reduce].sample(frac=0.5, random_state=42).index
        )
        X_train = X_train.drop(index=indices_to_drop)
        y_train = y_train.drop(index=indices_to_drop)

    return {"X_train": X_train, "y_train": y_train}


@job
def iris_training_job():
    dataset = iris_dataset()
    split = split_data(dataset)
    model = train_model(split)
    predictions = predict(model, split)
    log_to_mlflow(model, split, predictions)
    log_to_mlflow_skewed(model, split, predictions)


defs = Definitions(
    assets=[iris_dataset, split_data],
    jobs=[iris_training_job],
)
