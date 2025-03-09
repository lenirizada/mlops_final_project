import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
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
    y_train_pred = predict["y_train_pred"]  # Get predictions

    model = train_model

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Run started: {run_id}")

        mlflow.sklearn.log_model(model, "iris_rf_model")
        mlflow.log_params({"n_estimators": 100, "random_state": 42})
        mlflow.log_metric("train_accuracy", model.score(X_train, y_train))

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        shap.summary_plot(shap_values, X_train, show=False)
        plt.savefig("shap_summary.png")
        mlflow.log_artifact("shap_summary.png")

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


@job
def iris_training_job():
    dataset = iris_dataset()
    split = split_data(dataset)
    model = train_model(split)
    predictions = predict(model, split)
    log_to_mlflow(model, split, predictions)


defs = Definitions(
    assets=[iris_dataset, split_data],
    jobs=[iris_training_job],
)
